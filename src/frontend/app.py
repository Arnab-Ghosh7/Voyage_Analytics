"""Streamlit dashboard for Voyage Analytics (project objective #9).

An interactive front end over the three validated models, plus the EDA findings
and the measured model scorecard.

Two serving modes, chosen in the sidebar:

``direct``
    Loads the joblib pipelines in-process. Always works; no API required.
``api``
    Calls the Flask service at ``/predict/*``. Exercises the real deployment
    path, so the dashboard doubles as a smoke test for the API.

Keeping both is deliberate: the dashboard stays usable when the API is down, and
switching modes proves the two paths agree.

Run::

    streamlit run src/frontend/app.py
    python -m src.frontend                     # same thing
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ``streamlit run`` executes this file as a script, so the repo root is not on
# sys.path the way it is for ``python -m``.
_ROOT = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
             if (p / "requirements.txt").exists())
import sys                                                          # noqa: E402
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import PATHS                                         # noqa: E402

PALETTE = ["#0B6E4F", "#08A045", "#6BBF59", "#F2C14E", "#F78154", "#B4436C", "#4C6EF5"]
API_DEFAULT = "http://127.0.0.1:5000"

st.set_page_config(page_title="Voyage Analytics", page_icon="✈️",
                   layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# Cached loaders                                                               #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models…")
def get_model(name: str):
    from src.serving.app import load_model
    return load_model(name)


@st.cache_resource(show_spinner=False)
def get_reference() -> dict:
    from src.serving.app import reference
    return reference()


@st.cache_data(show_spinner=False)
def get_report(stem: str) -> dict:
    path = PATHS.reports / f"{stem}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner="Loading data…")
def get_table(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a processed table. `columns` keeps big tables light in memory."""
    from src.data_ingestion import load_processed
    df = load_processed(name)
    return df[columns] if columns else df


# --------------------------------------------------------------------------- #
# Prediction helpers — direct or via the API                                   #
# --------------------------------------------------------------------------- #
def _api_post(base: str, path: str, body: dict) -> dict:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"{base}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": json.loads(e.read()).get("error", str(e))}
    except Exception as e:                                    # noqa: BLE001
        return {"error": f"API unreachable at {base} — {e}"}


def api_is_up(base: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=2) as r:
            return r.status == 200
    except Exception:                                         # noqa: BLE001
        return False


def predict_flight_price(mode: str, base: str, payload: dict) -> dict:
    if mode == "api":
        return _api_post(base, "/predict/flight-price", payload)

    ref = get_reference()
    key = f"{payload['from']}|{payload['to']}"
    if key not in ref["distances"]:
        return {"error": f"No route from {payload['from']} to {payload['to']}."}
    when = pd.Timestamp(payload.get("date") or date.today())
    X = pd.DataFrame([{
        "from": payload["from"], "to": payload["to"],
        "flightType": payload["flightType"], "agency": payload["agency"],
        "day_of_week": when.day_name(), "distance": ref["distances"][key],
        "year": int(when.year), "month": int(when.month),
    }])
    price = float(get_model("flight_price").predict(X)[0])
    return {"predicted_price": round(price, 2), "currency": "BRL",
            "derived": {"distance_km": ref["distances"][key]}}


def predict_gender(mode: str, base: str, name: str) -> dict:
    if mode == "api":
        return _api_post(base, "/predict/gender", {"name": name})

    first = name.strip().split()[0].lower() if name.strip() else ""
    if not first:
        return {"error": "Enter a name."}
    model = get_model("gender")
    X = pd.DataFrame([{"first_name": first}])
    label = "male" if int(model.predict(X)[0]) == 1 else "female"
    try:
        conf = float(model.predict_proba(X).max())
    except Exception:                                         # noqa: BLE001
        conf = None
    return {"predicted_gender": label, "confidence": conf,
            "inputs": {"first_name_used": first}}


def recommend_hotels(mode: str, base: str, destination: str | None, k: int) -> dict:
    if mode == "api":
        body = {"top_k": k}
        if destination:
            body["destination"] = destination
        return _api_post(base, "/recommend/hotels", body)

    from src.models.recommender import recommend
    artifact = get_model("hotel_recommender")
    hotels = recommend(artifact, destination=destination, k=k)
    catalog = artifact.get("catalog", {})
    return {"recommendations": [
        {"rank": i + 1, "hotel": h,
         "city": catalog.get(h, {}).get("place"),
         "nightly_rate": catalog.get(h, {}).get("nightly_rate")}
        for i, h in enumerate(hotels)],
        "strategy": "destination-aware" if destination else "popularity"}


# --------------------------------------------------------------------------- #
# Pages                                                                        #
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    st.title("✈️ Voyage Analytics")
    st.caption("Integrating MLOps in Travel — productionised machine learning on "
               "Brazilian corporate travel data, 2019–2023")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flight legs", "271,888")
    c2.metric("Hotel stays", "40,552")
    c3.metric("Travellers", "1,340")
    c4.metric("Cities · routes", "9 · 70")

    st.subheader("Model scorecard")
    st.caption("All figures cross-validated on held-out splits — see the "
               "caveats below each one.")

    val = get_report("model_validation")
    rows = []
    if val:
        fp = val.get("flight_price", {})
        if fp:
            best = max(fp, key=lambda k: fp[k]["r2"]["mean"])
            rows.append({
                "Model": f"Flight price ({best})", "Type": "Regression",
                "Metric": "R²", "Score": fp[best]["r2"]["mean"],
                "95% CI": f"[{fp[best]['r2']['ci_low']:.3f}, {fp[best]['r2']['ci_high']:.3f}]",
                "vs chance": "—", "Status": "✅ Valid"})
        g = {k: v for k, v in val.get("gender", {}).items() if k != "permutation_test"}
        if g:
            best = max(g, key=lambda k: g[k]["accuracy"]["mean"])
            perm = val["gender"].get("permutation_test", {})
            rows.append({
                "Model": f"Gender ({best})", "Type": "Classification",
                "Metric": "Accuracy", "Score": g[best]["accuracy"]["mean"],
                "95% CI": f"[{g[best]['accuracy']['ci_low']:.3f}, {g[best]['accuracy']['ci_high']:.3f}]",
                "vs chance": f"0.500 · p={perm.get('p_value')}",
                "Status": "✅ Valid" if perm.get("significant_at_0.05") else "❌ At chance"})
        rk = val.get("hotel_ranking", {})
        if rk:
            best = max(rk, key=lambda k: rk[k]["hit_rate"])
            rows.append({
                "Model": f"Hotel ranking ({best})", "Type": "Collaborative filtering",
                "Metric": "hit@3", "Score": rk[best]["hit_rate"],
                "95% CI": f"[{rk[best]['ci_low']:.3f}, {rk[best]['ci_high']:.3f}]",
                "vs chance": f"{rk[best]['random_baseline']:.3f}", "Status": "✅ Valid"})
        at = val.get("hotel_attach", {})
        if at:
            rows.append({
                "Model": "Hotel attach", "Type": "Classification",
                "Metric": "ROC AUC", "Score": at["roc_auc"]["mean"],
                "95% CI": f"[{at['roc_auc']['ci_low']:.3f}, {at['roc_auc']['ci_high']:.3f}]",
                "vs chance": "0.500", "Status": "❌ At chance — not served"})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={"Score": st.column_config.NumberColumn(format="%.4f")})
    else:
        st.info("Run `python main.py` to generate the model reports.")

    st.subheader("What the data turned out to be")
    a, b = st.columns(2)
    with a:
        st.markdown("""
**Flight price is a rate card.** `(route, class, agency)` fixes the fare exactly.
A random train/test split therefore scores R² ≈ 1.0 by *memorising* it — so models
are selected on **held-out routes** instead. Predicting **price per km** rather than
raw price cut error by **49%**, because the rate transfers to routes never seen.
""")
    with b:
        st.markdown("""
**Two models sit at chance, honestly reported.** Gender is unpredictable from travel
behaviour (|r| ≤ 0.06) — the working model reads the *first name* instead. Hotel
attach is uniform ~30% across every segment (AUC 0.498), so it is logged but
**neither registered nor served**.
""")


def page_flight_price(mode: str, base: str) -> None:
    st.title("Flight price prediction")
    st.caption("Objective #1 · gradient boosting on rate-per-km · R² 0.936 on unseen routes")

    ref = get_reference()
    cities = ref["cities"]

    with st.form("flight"):
        c1, c2 = st.columns(2)
        origin = c1.selectbox("From", cities, index=cities.index("Sao Paulo (SP)")
                              if "Sao Paulo (SP)" in cities else 0)
        dest = c2.selectbox("To", cities, index=cities.index("Rio de Janeiro (RJ)")
                            if "Rio de Janeiro (RJ)" in cities else 1)
        c3, c4, c5 = st.columns(3)
        ftype = c3.selectbox("Class", ref["flight_types"],
                             index=ref["flight_types"].index("firstClass")
                             if "firstClass" in ref["flight_types"] else 0)
        agency = c4.selectbox("Agency", ref["agencies"])
        when = c5.date_input("Travel date", value=date.today())
        submitted = st.form_submit_button("Predict fare", type="primary",
                                          use_container_width=True)

    if submitted:
        if origin == dest:
            st.error("Origin and destination must differ.")
            return
        res = predict_flight_price(mode, base, {
            "from": origin, "to": dest, "flightType": ftype,
            "agency": agency, "date": str(when)})
        if "error" in res:
            st.error(res["error"])
            return

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted fare", f"R$ {res['predicted_price']:,.2f}")
        m2.metric("Distance", f"{res['derived']['distance_km']:,.0f} km")
        m3.metric("Price per km",
                  f"R$ {res['predicted_price']/res['derived']['distance_km']:.2f}")
        st.caption("Expect roughly ±R$89 (RMSE) on a route the model has not seen "
                   "in training. On routes already in the network the error is far smaller.")

        # Compare the three classes on this route
        rows = []
        for ft in ref["flight_types"]:
            r = predict_flight_price(mode, base, {
                "from": origin, "to": dest, "flightType": ft,
                "agency": agency, "date": str(when)})
            if "predicted_price" in r:
                rows.append({"Class": ft, "Fare": r["predicted_price"]})
        if rows:
            cmp = pd.DataFrame(rows)
            fig = px.bar(cmp, x="Class", y="Fare", text_auto=".0f",
                         color="Class", color_discrete_sequence=PALETTE,
                         title=f"Fare ladder — {origin} → {dest} ({agency})")
            fig.update_layout(showlegend=False, yaxis_title="fare (R$)")
            st.plotly_chart(fig, use_container_width=True)


def page_gender(mode: str, base: str) -> None:
    st.title("Gender classification")
    st.caption("Objective #8 · character n-grams over the first name · accuracy 0.884")

    st.warning(
        "**What this model actually does.** Travel behaviour carries no gender signal "
        "in this dataset (every feature correlates at |r| ≤ 0.06, and behaviour-only "
        "models measure at chance). The working model infers gender from the **given "
        "name**, which is a different claim — and it reflects the naming conventions of "
        "its training data, so it should be re-validated before use on a different "
        "population. On names never seen in training, accuracy is 0.743.",
        icon="⚠️")

    name = st.text_input("Full name", value="Charlotte Johnson",
                         placeholder="e.g. Joseph Holsten")
    if st.button("Classify", type="primary"):
        res = predict_gender(mode, base, name)
        if "error" in res:
            st.error(res["error"])
            return
        c1, c2 = st.columns(2)
        c1.metric("Predicted gender", res["predicted_gender"].title())
        if res.get("confidence") is not None:
            c2.metric("Confidence", f"{res['confidence']:.1%}")
            if res["confidence"] < 0.6:
                st.info("Low confidence — this name carries little signal either way.")


def page_hotels(mode: str, base: str) -> None:
    st.title("Hotel recommendations")
    st.caption("Objective #9 · item-item collaborative filtering · hit@3 0.858 vs 0.333 random")

    ref = get_reference()
    c1, c2 = st.columns([3, 1])
    use_dest = c1.checkbox("I know the destination", value=True)
    k = c2.slider("How many", 1, 5, 3)
    destination = c1.selectbox("Destination", ref["cities"]) if use_dest else None

    if st.button("Recommend", type="primary"):
        res = recommend_hotels(mode, base, destination, k)
        if "error" in res:
            st.error(res["error"])
            return
        recs = pd.DataFrame(res["recommendations"])
        st.dataframe(recs, use_container_width=True, hide_index=True)
        if destination:
            st.info("Each city has exactly one hotel, so a known destination gives an "
                    "exact match — no model needed. Ranking only matters when the "
                    "destination is unknown.", icon="ℹ️")


def page_insights() -> None:
    st.title("Data insights")
    st.caption("Findings from the exploratory analysis")

    catalog = get_table("hotel_catalog")
    users = get_table("users_features",
                      ["code", "company", "gender", "age", "age_band", "home_city",
                       "n_trips", "total_spend", "recency_days"])

    t1, t2, t3 = st.tabs(["Hotels", "Travellers", "Trips"])

    with t1:
        c1, c2 = st.columns(2)
        fig = px.bar(catalog.sort_values("revenue", ascending=False),
                     x="place", y="revenue", color="nightly_rate",
                     color_continuous_scale="Greens",
                     title="Hotel revenue by city (colour = nightly rate)")
        fig.update_layout(xaxis_title="", yaxis_title="revenue (R$)")
        c1.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(catalog, x="nightly_rate", y="bookings", text="hotel",
                          size="revenue", color="place",
                          color_discrete_sequence=PALETTE,
                          title="Rate vs demand — one hotel per city")
        fig2.update_traces(textposition="top center")
        c2.plotly_chart(fig2, use_container_width=True)
        st.caption("Revenue is volume × a fixed nightly rate. Because each city has "
                   "exactly one hotel, price competition does not exist in this data — "
                   "revenue grows through attach rate and length of stay, not pricing.")

    with t2:
        c1, c2 = st.columns(2)
        fig = px.histogram(users, x="age", nbins=22, color_discrete_sequence=[PALETTE[0]],
                           title="Age distribution")
        c1.plotly_chart(fig, use_container_width=True)
        gender_counts = users["gender"].value_counts().reset_index()
        gender_counts.columns = ["gender", "count"]
        fig2 = px.pie(gender_counts, names="gender", values="count", hole=0.45,
                      color_discrete_sequence=PALETTE,
                      title="Gender — 'none' is undisclosed, not a category")
        c2.plotly_chart(fig2, use_container_width=True)

        spend = (users.dropna(subset=["total_spend"])
                 .groupby("company", observed=True)["total_spend"].mean()
                 .reset_index().sort_values("total_spend"))
        fig3 = px.bar(spend, x="total_spend", y="company", orientation="h",
                      color_discrete_sequence=[PALETTE[1]],
                      title="Mean lifetime spend by company")
        fig3.update_layout(xaxis_title="R$", yaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Spend is essentially flat across companies and age bands "
                   "(correlations ≈ 0). Customer value here is behavioural, not "
                   "demographic — which is why segmentation uses RFM rather than age.")

    with t3:
        trips = get_table("trips", ["dest", "origin", "flightType", "agency",
                                    "trip_nights", "has_hotel", "trip_spend"])
        c1, c2 = st.columns(2)
        dest_counts = trips["dest"].value_counts().reset_index()
        dest_counts.columns = ["destination", "trips"]
        fig = px.bar(dest_counts, x="trips", y="destination", orientation="h",
                     color_discrete_sequence=[PALETTE[0]], title="Trips by destination")
        fig.update_layout(yaxis_title="")
        c1.plotly_chart(fig, use_container_width=True)

        attach = (trips.groupby("dest", observed=True)["has_hotel"]
                  .mean().reset_index().sort_values("has_hotel"))
        fig2 = px.bar(attach, x="has_hotel", y="dest", orientation="h",
                      color_discrete_sequence=[PALETTE[5]],
                      title="Hotel attach rate by destination")
        fig2.update_layout(xaxis_title="share of trips with a hotel", yaxis_title="",
                           xaxis_tickformat=".0%")
        c2.plotly_chart(fig2, use_container_width=True)
        st.caption("The attach rate sits near 30% everywhere — flat across destinations, "
                   "classes, agencies and years. That uniformity is precisely why the "
                   "attach model measures at chance: there is no segment to target.")


def page_performance() -> None:
    st.title("Model performance")
    st.caption("Measured results, including the ones that did not work")

    t1, t2, t3 = st.tabs(["Flight price", "Validation", "Tuning"])

    with t1:
        fp = get_report("flight_price_metrics")
        if not fp:
            st.info("Run `python main.py` first.")
        else:
            res = pd.DataFrame(fp["results"])
            st.markdown(f"**Selected model:** `{fp['best_model']}` — chosen on "
                        f"*{fp['selected_on']}*")
            piv = res.pivot(index="model", columns="split", values="r2").reset_index()
            fig = px.bar(piv.melt(id_vars="model", var_name="split", value_name="r2"),
                         x="model", y="r2", color="split", barmode="group",
                         color_discrete_sequence=[PALETTE[0], PALETTE[5]],
                         title="R² — random split vs held-out routes")
            st.plotly_chart(fig, use_container_width=True)
            st.error(
                "**The reason the split matters.** On a random split gradient boosting "
                "reaches R² ≈ 1.000 and looks perfect — it has memorised the rate card. "
                "On routes it has never seen, that same model is the *worst* of the "
                "candidates. Reporting the random-split number would have shipped a "
                "model that degrades the moment a new route opens.", icon="🚨")
            st.dataframe(res[res.split == "grouped"][
                ["model", "mse", "rmse", "mae", "r2", "mape_pct"]]
                .sort_values("mse"), use_container_width=True, hide_index=True)

    with t2:
        val = get_report("model_validation")
        if not val:
            st.info("Run `python main.py --stages validate_models`.")
        else:
            for k, v in val.get("verdicts", {}).items():
                st.markdown(f"- **{k}** — {v}")
            perm = val.get("gender", {}).get("permutation_test", {})
            if perm:
                st.markdown(f"""
**Permutation test on the gender model.** Labels were shuffled 100 times and the model
retrained each time. Real score **{perm['score']}** against a shuffled mean of
**{perm['permutation_mean']}**, giving **p = {perm['p_value']}** — significant, so the
result is not an artefact of a lucky split.
""")

    with t3:
        tune = get_report("tuning_results")
        if not tune:
            st.info("Run `python main.py --stages tune`.")
        else:
            comp = pd.DataFrame(tune["comparison"])
            st.dataframe(comp[["model", "metric", "before_tuning", "after_tuning",
                               "gain", "meaningful"]],
                         use_container_width=True, hide_index=True)
            st.info(
                "**Tuning was not where the wins came from.** Reformulating the "
                "problem delivered an order of magnitude more: predicting rate-per-km "
                "instead of raw price cut MSE by 49%, and switching to first-name "
                "features lifted gender accuracy by 0.38. Every hyperparameter search "
                "after that returned changes within fold-to-fold noise.", icon="💡")


# --------------------------------------------------------------------------- #
# Shell                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    with st.sidebar:
        st.title("✈️ Voyage Analytics")
        page = st.radio("Navigate", [
            "Overview", "Flight price", "Gender", "Hotels",
            "Data insights", "Model performance"], label_visibility="collapsed")

        st.divider()
        st.caption("Prediction source")
        mode = st.radio("mode", ["direct", "api"],
                        format_func=lambda m: {"direct": "Load models in-process",
                                               "api": "Call the Flask API"}[m],
                        label_visibility="collapsed")
        base = API_DEFAULT
        if mode == "api":
            base = st.text_input("API base URL", value=API_DEFAULT)
            if api_is_up(base):
                st.success("API reachable", icon="✅")
            else:
                st.error("API not reachable — start it with "
                         "`python -m src.serving`", icon="🚫")

        st.divider()
        st.caption("Brazilian corporate travel · 2019–2023\n\n"
                   "9 cities · 70 routes · 271,888 flight legs")

    if page == "Overview":
        page_overview()
    elif page == "Flight price":
        page_flight_price(mode, base)
    elif page == "Gender":
        page_gender(mode, base)
    elif page == "Hotels":
        page_hotels(mode, base)
    elif page == "Data insights":
        page_insights()
    else:
        page_performance()


# Streamlit runs this file as a script with __name__ == "__main__", so the guard
# still launches the UI — while allowing the prediction helpers to be imported
# and tested without rendering anything.
if __name__ == "__main__":
    main()
