"""Streamlit dashboard for Voyage Analytics (project objective #9).

An interactive front end over the three validated models, plus the EDA findings
and the measured model scorecard. Styling lives in :mod:`src.frontend.theme`.

Two serving modes, chosen in the header:

``direct``
    Loads the joblib pipelines in-process. Always works; no API required.
``api``
    Calls the Flask service at ``/predict/*``. Exercises the real deployment
    path, so the dashboard doubles as a smoke test for the API.

Run::

    streamlit run src/frontend/app.py
    python -m src.frontend                     # same thing
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ``streamlit run`` executes this file as a script, so the repo root is not on
# sys.path the way it is for ``python -m``.
_ROOT = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
             if (p / "requirements.txt").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import PATHS                                          # noqa: E402
from src.frontend import theme as T                                  # noqa: E402

API_DEFAULT = "http://127.0.0.1:5000"

st.set_page_config(page_title="Voyage Analytics", page_icon="✈️",
                   layout="wide", initial_sidebar_state="collapsed")


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
def page_home() -> None:
    T.hero()
    T.action_cards()
    T.feature_cards()

    T.section("Platform at a glance", "The dataset behind every prediction", "📊")
    c = st.columns(4)
    for col, (lbl, val) in zip(c, [("Flight legs", "271,888"), ("Hotel stays", "40,552"),
                                   ("Travellers", "1,340"), ("Cities · routes", "9 · 70")]):
        col.metric(lbl, val)

    T.section("What the data turned out to be", "", "🔍")
    a, b = st.columns(2)
    with a:
        st.markdown(T.panel_open("Flight price is a rate card", "objective #1") + """
<p style="font-size:.9rem;color:#5A6B85;line-height:1.6">
<code>(route, class, agency)</code> fixes the fare exactly, so a random train/test split
scores R² ≈ 1.0 by <b>memorising</b> it. Models are selected on <b>held-out routes</b>
instead. Predicting <b>price per km</b> rather than raw price cut error by <b>49%</b>,
because the rate transfers to routes never seen.</p></div>""",
                    unsafe_allow_html=True)
    with b:
        st.markdown(T.panel_open("Two models sit at chance", "reported honestly") + """
<p style="font-size:.9rem;color:#5A6B85;line-height:1.6">
Gender is unpredictable from travel behaviour (|r| ≤ 0.06) — the working model reads the
<b>first name</b> instead. Hotel attach is a uniform ~30% across every segment
(AUC 0.498), so it is logged but <b>neither registered nor served</b>.</p></div>""",
                    unsafe_allow_html=True)


def page_dashboard() -> None:
    T.section("Travel Analytics Dashboard",
              "Monitor model performance, travel insights and ML pipeline status", "📈")

    val = get_report("model_validation")
    fp = val.get("flight_price", {})
    gm = {k: v for k, v in val.get("gender", {}).items() if k != "permutation_test"}
    rk = val.get("hotel_ranking", {})
    at = val.get("hotel_attach", {})

    fb = max(fp, key=lambda k: fp[k]["r2"]["mean"]) if fp else None
    gb = max(gm, key=lambda k: gm[k]["accuracy"]["mean"]) if gm else None
    rb = max(rk, key=lambda k: rk[k]["hit_rate"]) if rk else None

    if not fb:
        st.info("Run `python main.py` to generate the model reports.")
        return

    r2 = fp[fb]["r2"]["mean"]
    acc = gm[gb]["accuracy"]["mean"]
    hit = rk[rb]["hit_rate"]

    st.markdown('<div class="mgrid">'
                + T.metric_card("✈️", "Flight Price Model", f"{r2*100:.1f}%", "R² Score",
                                "High prediction accuracy on unseen routes",
                                T.GRADIENTS["blue"], r2)
                + T.metric_card("📊", "Gender Prediction", f"{acc*100:.1f}%",
                                "Model Accuracy",
                                "From the first name — not travel behaviour",
                                T.GRADIENTS["green"], acc)
                + T.metric_card("🏨", "Hotel Recommender", f"{hit*100:.1f}%", "Hit@3",
                                "Recommendation success rate vs 33.3% random",
                                T.GRADIENTS["orange"], hit)
                + '</div>', unsafe_allow_html=True)

    # The model that failed gets a card too — omitting it would misrepresent the work.
    if at:
        auc = at["roc_auc"]["mean"]
        st.markdown('<div class="mgrid" style="grid-template-columns:1fr 2fr">'
                    + T.metric_card("🚫", "Hotel Attach Model", f"{auc*100:.1f}%",
                                    "ROC AUC — chance is 50%",
                                    "Not registered, not served",
                                    "linear-gradient(135deg,#64748B 0%,#475569 100%)",
                                    auc, status="At chance", warn=True)
                    + '</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            T.panel_open("ML Pipeline", "Current machine learning workflow", "Running")
            + T.prow("🗄️", "Data Validation", "22 checks passed on raw data")
            + T.prow("⚙️", "Data Ingestion", "6 tables built from 271,888 rows")
            + T.prow("🧬", "Feature Engineering", "3 model-ready feature sets")
            + T.prow("🔎", "Output Audit", "25 checks passed on generated tables")
            + T.prow("🤖", "Model Training", "3 models trained and validated")
            + T.prow("📦", "MLflow Registry", "3 models registered, 1 withheld")
            + "</div>", unsafe_allow_html=True)
    with right:
        st.markdown(
            T.panel_open("Quick Facts", "Measured, not estimated")
            + T.prow("⚡", "API latency", "21 ms p95 vs a 500 ms budget", bg="#FFF1DC")
            + T.prow("🎯", "Best MSE", "7,919 · RMSE R$88.99 · MAE R$69.08", bg="#E0EDFF")
            + T.prow("🧪", "Gender p-value", "0.0099 — significant", bg="#DFF5E6")
            + T.prow("📉", "Tuning gain", "≤ 0.015 — reformulation won instead", bg="#F3E8FF")
            + "</div>", unsafe_allow_html=True)


def page_flight_price(mode: str, base: str) -> None:
    T.section("Flight Price Prediction",
              "Objective #1 · gradient boosting on rate-per-km · R² 0.936 on unseen routes",
              "🛫")

    ref = get_reference()
    cities = ref["cities"]

    with st.form("flight"):
        c1, c2 = st.columns(2)
        origin = c1.selectbox("From", cities,
                              index=cities.index("Sao Paulo (SP)")
                              if "Sao Paulo (SP)" in cities else 0)
        dest = c2.selectbox("To", cities,
                            index=cities.index("Rio de Janeiro (RJ)")
                            if "Rio de Janeiro (RJ)" in cities else 1)
        c3, c4, c5 = st.columns(3)
        ftype = c3.selectbox("Class", ref["flight_types"],
                             index=ref["flight_types"].index("firstClass")
                             if "firstClass" in ref["flight_types"] else 0)
        agency = c4.selectbox("Agency", ref["agencies"])
        when = c5.date_input("Travel date", value=date.today())
        submitted = st.form_submit_button("✈️  Predict fare")

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

        km = res["derived"]["distance_km"]
        st.markdown('<div class="mgrid">'
                    + T.metric_card("💰", "Predicted Fare",
                                    f"R$ {res['predicted_price']:,.0f}",
                                    f"{origin.split(' (')[0]} → {dest.split(' (')[0]}",
                                    "Expect roughly ±R$89 on an unseen route",
                                    T.GRADIENTS["blue"], 1.0, status="Predicted")
                    + T.metric_card("📏", "Route Distance", f"{km:,.0f}", "kilometres",
                                    "Derived from the route, not supplied",
                                    T.GRADIENTS["green"], min(km / 1000, 1.0),
                                    status="Derived")
                    + T.metric_card("📊", "Price per km",
                                    f"R$ {res['predicted_price']/km:.2f}", "unit rate",
                                    f"{ftype} on {agency}",
                                    T.GRADIENTS["orange"],
                                    min(res['predicted_price']/km / 4, 1.0),
                                    status=ftype)
                    + '</div>', unsafe_allow_html=True)

        rows = []
        for ft in ref["flight_types"]:
            r = predict_flight_price(mode, base, {
                "from": origin, "to": dest, "flightType": ft,
                "agency": agency, "date": str(when)})
            if "predicted_price" in r:
                rows.append({"Class": ft, "Fare": r["predicted_price"]})
        if rows:
            cmp = pd.DataFrame(rows)
            fig = px.bar(cmp, x="Class", y="Fare", text_auto=".0f", color="Class",
                         color_discrete_sequence=T.CHART,
                         title=f"Fare ladder — {origin} → {dest} ({agency})")
            fig.update_layout(showlegend=False, yaxis_title="fare (R$)",
                              plot_bgcolor="white", height=380)
            st.plotly_chart(fig, use_container_width=True)


def page_gender(mode: str, base: str) -> None:
    T.section("Gender Classification",
              "Objective #8 · character n-grams over the first name · accuracy 0.884", "👥")

    st.warning(
        "**What this model actually does.** Travel behaviour carries no gender signal in "
        "this dataset (every feature correlates at |r| ≤ 0.06, and behaviour-only models "
        "measure at chance). The working model infers gender from the **given name**, "
        "which is a different claim — and it reflects the **Brazilian naming conventions** "
        "of its training data, so it is unreliable on names from other cultures. On names "
        "never seen in training, accuracy is 0.743.", icon="⚠️")

    c1, c2 = st.columns([2, 1])
    name = c1.text_input("Full name", value="Charlotte Johnson",
                         placeholder="e.g. Joseph Holsten")
    c2.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
    go = c2.button("🔎  Classify")

    if go:
        res = predict_gender(mode, base, name)
        if "error" in res:
            st.error(res["error"])
            return
        conf = res.get("confidence") or 0
        grad = T.GRADIENTS["green"] if res["predicted_gender"] == "female" else T.GRADIENTS["blue"]
        st.markdown('<div class="mgrid" style="grid-template-columns:1fr 1fr">'
                    + T.metric_card("🧑", "Predicted Gender",
                                    res["predicted_gender"].title(),
                                    f"from first name '{res['inputs']['first_name_used']}'",
                                    "Inferred from the name, not behaviour",
                                    grad, 1.0, status="Predicted")
                    + T.metric_card("📉", "Confidence", f"{conf:.1%}", "model certainty",
                                    "Below 60% means the name carries little signal",
                                    T.GRADIENTS["orange"] if conf < 0.6 else T.GRADIENTS["green"],
                                    conf, status="Low" if conf < 0.6 else "Good",
                                    warn=conf < 0.6)
                    + '</div>', unsafe_allow_html=True)


def page_hotels(mode: str, base: str) -> None:
    T.section("Hotel Recommendations",
              "Objective #9 · item-item collaborative filtering · hit@3 0.858 vs 0.333 random",
              "🏨")

    ref = get_reference()
    c1, c2, c3 = st.columns([1, 2, 1])
    use_dest = c1.checkbox("Know the destination", value=True)
    destination = c2.selectbox("Destination", ref["cities"]) if use_dest else None
    k = c3.slider("How many", 1, 5, 3)

    if st.button("🔍  Recommend hotels"):
        res = recommend_hotels(mode, base, destination, k)
        if "error" in res:
            st.error(res["error"])
            return
        recs = res["recommendations"]
        grads = [T.GRADIENTS["blue"], T.GRADIENTS["green"], T.GRADIENTS["orange"]]
        cards = "".join(
            T.metric_card("🏩", f"#{r['rank']}  {r['hotel']}",
                          f"R$ {r['nightly_rate']:,.0f}" if r.get("nightly_rate") else "—",
                          "per night", r.get("city") or "",
                          grads[i % 3], 1 - i * 0.18, status=f"Rank {r['rank']}")
            for i, r in enumerate(recs[:3]))
        st.markdown(f'<div class="mgrid">{cards}</div>', unsafe_allow_html=True)
        if len(recs) > 3:
            st.dataframe(pd.DataFrame(recs[3:]), use_container_width=True, hide_index=True)
        if destination:
            st.info("Each city has exactly one hotel, so a known destination gives an "
                    "exact match — no model needed. Ranking only matters when the "
                    "destination is unknown.", icon="ℹ️")


def page_insights() -> None:
    T.section("Data Insights", "Findings from the exploratory analysis", "🔍")

    catalog = get_table("hotel_catalog")
    users = get_table("users_features",
                      ["code", "company", "gender", "age", "age_band", "home_city",
                       "n_trips", "total_spend", "recency_days"])

    t1, t2, t3 = st.tabs(["🏨  Hotels", "👥  Travellers", "🧳  Trips"])

    with t1:
        c1, c2 = st.columns(2)
        fig = px.bar(catalog.sort_values("revenue", ascending=False),
                     x="place", y="revenue", color="nightly_rate",
                     color_continuous_scale="Teal",
                     title="Hotel revenue by city (colour = nightly rate)")
        fig.update_layout(xaxis_title="", yaxis_title="revenue (R$)",
                          plot_bgcolor="white", height=400)
        c1.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(catalog, x="nightly_rate", y="bookings", text="hotel",
                          size="revenue", color="place",
                          color_discrete_sequence=T.CHART,
                          title="Rate vs demand — one hotel per city")
        fig2.update_traces(textposition="top center")
        fig2.update_layout(plot_bgcolor="white", height=400)
        c2.plotly_chart(fig2, use_container_width=True)
        st.caption("Revenue is volume × a fixed nightly rate. Because each city has "
                   "exactly one hotel, price competition does not exist in this data — "
                   "revenue grows through attach rate and length of stay, not pricing.")

    with t2:
        c1, c2 = st.columns(2)
        fig = px.histogram(users, x="age", nbins=22,
                           color_discrete_sequence=[T.CHART[0]],
                           title="Age distribution")
        fig.update_layout(plot_bgcolor="white", height=380)
        c1.plotly_chart(fig, use_container_width=True)
        gc = users["gender"].value_counts().reset_index()
        gc.columns = ["gender", "count"]
        fig2 = px.pie(gc, names="gender", values="count", hole=0.5,
                      color_discrete_sequence=T.CHART,
                      title="Gender — 'none' is undisclosed, not a category")
        fig2.update_layout(height=380)
        c2.plotly_chart(fig2, use_container_width=True)

        spend = (users.dropna(subset=["total_spend"])
                 .groupby("company", observed=True)["total_spend"].mean()
                 .reset_index().sort_values("total_spend"))
        fig3 = px.bar(spend, x="total_spend", y="company", orientation="h",
                      color_discrete_sequence=[T.CHART[1]],
                      title="Mean lifetime spend by company")
        fig3.update_layout(xaxis_title="R$", yaxis_title="",
                           plot_bgcolor="white", height=340)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Spend is essentially flat across companies and age bands "
                   "(correlations ≈ 0). Customer value here is behavioural, not "
                   "demographic — which is why segmentation uses RFM rather than age.")

    with t3:
        trips = get_table("trips", ["dest", "origin", "flightType", "agency",
                                    "trip_nights", "has_hotel", "trip_spend"])
        c1, c2 = st.columns(2)
        dc = trips["dest"].value_counts().reset_index()
        dc.columns = ["destination", "trips"]
        fig = px.bar(dc, x="trips", y="destination", orientation="h",
                     color_discrete_sequence=[T.CHART[0]], title="Trips by destination")
        fig.update_layout(yaxis_title="", plot_bgcolor="white", height=400)
        c1.plotly_chart(fig, use_container_width=True)

        attach = (trips.groupby("dest", observed=True)["has_hotel"]
                  .mean().reset_index().sort_values("has_hotel"))
        fig2 = px.bar(attach, x="has_hotel", y="dest", orientation="h",
                      color_discrete_sequence=[T.CHART[4]],
                      title="Hotel attach rate by destination")
        fig2.update_layout(xaxis_title="share of trips with a hotel", yaxis_title="",
                           xaxis_tickformat=".0%", plot_bgcolor="white", height=400)
        c2.plotly_chart(fig2, use_container_width=True)
        st.caption("The attach rate sits near 30% everywhere — flat across destinations, "
                   "classes, agencies and years. That uniformity is precisely why the "
                   "attach model measures at chance: there is no segment to target.")


def page_performance() -> None:
    T.section("Model Performance", "Measured results, including the ones that did not work",
              "🎯")

    t1, t2, t3 = st.tabs(["✈️  Flight price", "🧪  Validation", "⚙️  Tuning"])

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
                         color_discrete_sequence=[T.CHART[0], T.CHART[4]],
                         title="R² — random split vs held-out routes")
            fig.update_layout(plot_bgcolor="white", height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.error(
                "**The reason the split matters.** On a random split gradient boosting "
                "reaches R² ≈ 1.000 and looks perfect — it has memorised the rate card. "
                "On routes it has never seen, that same model is the *worst* of the "
                "candidates. Reporting the random-split number would have shipped a "
                "model that degrades the moment a new route opens.", icon="🚨")
            st.dataframe(res[res.split == "grouped"][
                ["model", "mse", "rmse", "mae", "r2", "mape_pct"]].sort_values("mse"),
                use_container_width=True, hide_index=True)

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
                "**Tuning was not where the wins came from.** Reformulating the problem "
                "delivered an order of magnitude more: predicting rate-per-km instead of "
                "raw price cut MSE by 49%, and switching to first-name features lifted "
                "gender accuracy by 0.38. Every hyperparameter search after that returned "
                "changes within fold-to-fold noise.", icon="💡")


# --------------------------------------------------------------------------- #
# Shell                                                                        #
# --------------------------------------------------------------------------- #
PAGES = {
    "🏠  Home": "home",
    "🛫  Flight Price": "flight",
    "👥  Gender": "gender",
    "🏨  Hotels": "hotels",
    "🔍  Data Insights": "insights",
    "📈  Dashboard": "dashboard",
    "🎯  Performance": "performance",
}


def main() -> None:
    T.inject_css()
    T.navbar()

    nav, src = st.columns([4, 1])
    with nav:
        label = st.radio("nav", list(PAGES), horizontal=True, label_visibility="collapsed")
    with src:
        mode = st.selectbox("Prediction source", ["direct", "api"],
                            format_func=lambda m: {"direct": "⚡ In-process",
                                                   "api": "🌐 Flask API"}[m],
                            label_visibility="collapsed")

    base = API_DEFAULT
    if mode == "api":
        base = st.text_input("API base URL", value=API_DEFAULT)
        if api_is_up(base):
            st.success("API reachable", icon="✅")
        else:
            st.error("API not reachable — start it with `python -m src.serving`", icon="🚫")

    page = PAGES[label]
    if page == "home":
        page_home()
    elif page == "flight":
        page_flight_price(mode, base)
    elif page == "gender":
        page_gender(mode, base)
    elif page == "hotels":
        page_hotels(mode, base)
    elif page == "insights":
        page_insights()
    elif page == "dashboard":
        page_dashboard()
    else:
        page_performance()


# Streamlit runs this file as a script with __name__ == "__main__", so the guard
# still launches the UI — while allowing the prediction helpers to be imported
# and tested without rendering anything.
if __name__ == "__main__":
    main()
