"""Design system for the Voyage Analytics dashboard.

Streamlit's default chrome is deliberately plain, so the look here is built from
injected CSS plus a handful of HTML component helpers. Keeping all of it in one
module means the page files stay readable — they call ``metric_card(...)`` rather
than carrying blocks of markup inline.
"""
from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------- #
# Tokens                                                                       #
# --------------------------------------------------------------------------- #
NAVY = "#0B3D6B"
TEAL = "#0E7490"
GOLD = "#F5B921"
BLUE = "#2563EB"
GREEN = "#15803D"
ORANGE = "#F59E0B"
INK = "#12263F"
MUTED = "#5A6B85"
PAGE_BG = "#F6F8FC"

# Chart palette, kept consistent with the report figures
CHART = ["#0B6E4F", "#2563EB", "#F5B921", "#F78154", "#B4436C", "#0E7490", "#6BBF59"]

GRADIENTS = {
    "blue": f"linear-gradient(135deg, {BLUE} 0%, #1D4ED8 100%)",
    "green": f"linear-gradient(135deg, #16A34A 0%, {GREEN} 100%)",
    "orange": f"linear-gradient(135deg, #FBBF24 0%, {ORANGE} 100%)",
    "navy": f"linear-gradient(90deg, {NAVY} 0%, {TEAL} 100%)",
    "hero": ("linear-gradient(135deg, #0B2A4A 0%, #12456F 32%, "
             "#6D4B8A 62%, #E4823C 100%)"),
}


def inject_css() -> None:
    """Inject the stylesheet once per session."""
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }}
.stApp {{ background: {PAGE_BG}; }}

/* hide default streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ display: none; }}
.block-container {{ padding: 0 2.2rem 3rem 2.2rem !important; max-width: 1400px; }}

/* ---------------- navbar ---------------- */
.nav {{
  background: {GRADIENTS['navy']};
  margin: 0 -2.2rem 0 -2.2rem; padding: .85rem 2.2rem;
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: 0 4px 18px rgba(11,61,107,.28);
}}
.nav-brand {{ display: flex; align-items: center; gap: .7rem; }}
.nav-logo {{
  width: 38px; height: 38px; border-radius: 50%; background: {GOLD};
  display: grid; place-items: center; font-size: 1.15rem;
  box-shadow: 0 3px 10px rgba(0,0,0,.22);
}}
.nav-title {{ color: #fff; font-weight: 800; font-size: 1.22rem; letter-spacing: .2px; }}
.nav-title span {{ color: {GOLD}; }}
.nav-badge {{
  background: rgba(255,255,255,.16); color: #fff; border: 1px solid rgba(255,255,255,.28);
  padding: .34rem .85rem; border-radius: 999px; font-size: .76rem; font-weight: 600;
}}

/* ---------------- top nav pills (styled radio) ---------------- */
div[role="radiogroup"] {{ gap: .45rem !important; flex-wrap: wrap; }}
div[role="radiogroup"] > label {{
  background: #fff; border: 1px solid #E3E9F2; border-radius: 11px;
  padding: .55rem 1.05rem !important; margin: 0 !important;
  font-weight: 600; font-size: .93rem; color: {MUTED}; cursor: pointer;
  transition: all .16s ease; box-shadow: 0 1px 3px rgba(18,38,63,.05);
}}
div[role="radiogroup"] > label:hover {{ border-color: {GOLD}; color: {INK}; }}
div[role="radiogroup"] > label:has(input:checked) {{
  background: {GOLD}; border-color: {GOLD}; color: {INK};
  box-shadow: 0 4px 12px rgba(245,185,33,.42);
}}
div[role="radiogroup"] > label > div:first-child {{ display: none; }}

/* ---------------- hero ---------------- */
.hero {{
  background: {GRADIENTS['hero']};
  margin: 0 -2.2rem; padding: 3.4rem 2.2rem 3.1rem;
  text-align: center; position: relative; overflow: hidden;
}}
.hero:after {{
  content: ""; position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 120%, rgba(255,255,255,.20), transparent 62%);
}}
.hero-plane {{ font-size: 2rem; opacity: .95; }}
.hero h1 {{
  color: #fff; font-size: 3.5rem; font-weight: 800; margin: .3rem 0 .1rem;
  letter-spacing: -1.2px; text-shadow: 0 3px 22px rgba(0,0,0,.34);
}}
.hero h1 span {{ color: {GOLD}; }}
.hero p {{ color: rgba(255,255,255,.94); font-size: 1.18rem; margin: 0; font-weight: 500; }}
.hero-rule {{
  width: 260px; height: 1px; margin: 1.1rem auto 0;
  background: linear-gradient(90deg, transparent, {GOLD}, transparent); position: relative;
}}
.hero-rule:after {{
  content: "✈"; position: absolute; left: 50%; top: -11px; transform: translateX(-50%);
  color: {GOLD}; font-size: 1rem;
}}

/* ---------------- cards ---------------- */
.cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.1rem; margin: 1.5rem 0; }}
.acard {{
  border-radius: 15px; padding: 1.35rem 1.5rem; color: #fff;
  display: flex; align-items: center; gap: 1rem;
  box-shadow: 0 8px 24px rgba(18,38,63,.16); transition: transform .16s ease;
}}
.acard:hover {{ transform: translateY(-3px); }}
.acard .ico {{ font-size: 2.05rem; line-height: 1; }}
.acard h4 {{ margin: 0 0 .18rem; font-size: 1.11rem; font-weight: 700; }}
.acard p {{ margin: 0; font-size: .845rem; opacity: .93; line-height: 1.35; }}

.feats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.1rem; margin-top: 1.2rem; }}
.feat {{
  background: #fff; border: 1px solid #E7EDF6; border-radius: 15px;
  padding: 1.5rem 1.15rem; text-align: center; box-shadow: 0 2px 10px rgba(18,38,63,.05);
}}
.feat .circle {{
  width: 52px; height: 52px; border-radius: 50%; margin: 0 auto .8rem;
  display: grid; place-items: center; font-size: 1.4rem;
}}
.feat h5 {{ margin: 0 0 .38rem; font-size: 1rem; font-weight: 700; color: {INK}; }}
.feat p {{ margin: 0; font-size: .82rem; color: {MUTED}; line-height: 1.45; }}

/* ---------------- metric cards ---------------- */
.mgrid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; margin: 1.4rem 0; }}
.mcard {{
  border-radius: 17px; padding: 1.45rem 1.6rem; color: #fff; position: relative;
  overflow: hidden; box-shadow: 0 10px 28px rgba(18,38,63,.18); min-height: 196px;
}}
.mcard:after {{
  content: ""; position: absolute; right: -34px; bottom: -34px;
  width: 132px; height: 132px; border-radius: 50%; background: rgba(255,255,255,.10);
}}
.mcard .top {{ display: flex; justify-content: space-between; align-items: flex-start; }}
.mcard .ico {{
  width: 46px; height: 46px; border-radius: 12px; background: rgba(255,255,255,.20);
  display: grid; place-items: center; font-size: 1.35rem;
}}
.pill {{
  background: rgba(255,255,255,.22); border-radius: 999px; padding: .26rem .72rem;
  font-size: .73rem; font-weight: 700; display: inline-flex; align-items: center; gap: .3rem;
}}
.pill.warn {{ background: rgba(0,0,0,.24); }}
.mcard h3 {{ margin: 1rem 0 .45rem; font-size: 1.09rem; font-weight: 700; }}
.mcard .big {{ font-size: 2.75rem; font-weight: 800; line-height: 1; letter-spacing: -1.4px; }}
.mcard .lbl {{ font-size: .84rem; opacity: .92; margin-top: .28rem; }}
.bar {{ height: 6px; background: rgba(255,255,255,.26); border-radius: 99px; margin: .95rem 0 .55rem; }}
.bar > div {{ height: 100%; background: #fff; border-radius: 99px; }}
.mcard .foot {{ font-size: .77rem; opacity: .9; }}

/* ---------------- panels ---------------- */
.panel {{
  background: #fff; border: 1px solid #E7EDF6; border-radius: 16px;
  padding: 1.45rem 1.6rem; box-shadow: 0 2px 12px rgba(18,38,63,.05); height: 100%;
}}
.panel h4 {{ margin: 0 0 .2rem; font-size: 1.13rem; font-weight: 700; color: {INK}; }}
.panel .sub {{ font-size: .84rem; color: {MUTED}; margin-bottom: 1rem; }}
.prow {{
  display: flex; align-items: center; gap: .85rem; padding: .78rem 0;
  border-bottom: 1px solid #F1F5FA;
}}
.prow:last-child {{ border-bottom: none; }}
.prow .sq {{
  width: 38px; height: 38px; border-radius: 10px; background: #EEF4FF;
  display: grid; place-items: center; font-size: 1.05rem; flex-shrink: 0;
}}
.prow .txt {{ flex: 1; }}
.prow .txt b {{ display: block; font-size: .93rem; color: {INK}; font-weight: 700; }}
.prow .txt small {{ color: {MUTED}; font-size: .79rem; }}
.tick {{ color: #16A34A; font-size: 1.15rem; font-weight: 700; }}

.section-title {{
  font-size: 1.5rem; font-weight: 800; color: {INK};
  margin: 1.9rem 0 .25rem; display: flex; align-items: center; gap: .55rem;
}}
.section-sub {{ color: {MUTED}; font-size: .93rem; margin-bottom: .9rem; }}

/* ---------------- widgets ---------------- */
.stButton > button {{
  background: {GRADIENTS['navy']}; color: #fff; border: none; border-radius: 11px;
  padding: .68rem 1.6rem; font-weight: 700; font-size: .95rem; width: 100%;
  box-shadow: 0 4px 14px rgba(11,61,107,.28); transition: all .16s ease;
}}
.stButton > button:hover {{ transform: translateY(-1px); box-shadow: 0 7px 20px rgba(11,61,107,.36); }}
[data-testid="stForm"] {{
  background: #fff; border: 1px solid #E7EDF6; border-radius: 16px;
  padding: 1.5rem; box-shadow: 0 2px 12px rgba(18,38,63,.05);
}}
[data-testid="stMetric"] {{
  background: #fff; border: 1px solid #E7EDF6; border-radius: 14px;
  padding: 1.05rem 1.2rem; box-shadow: 0 2px 10px rgba(18,38,63,.05);
}}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 800; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
.stTabs [data-baseweb="tab"] {{
  background: #fff; border: 1px solid #E7EDF6; border-radius: 10px 10px 0 0;
  padding: .55rem 1.15rem; font-weight: 600;
}}
.stTabs [aria-selected="true"] {{ background: {GOLD}; border-color: {GOLD}; color: {INK}; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Components                                                                   #
# --------------------------------------------------------------------------- #
def navbar(active: str = "") -> None:
    st.markdown(f"""
<div class="nav">
  <div class="nav-brand">
    <div class="nav-logo">✈️</div>
    <div class="nav-title">Voyage <span>Analytics</span></div>
  </div>
  <div class="nav-badge">🟢 All Systems Operational</div>
</div>""", unsafe_allow_html=True)


def hero() -> None:
    st.markdown("""
<div class="hero">
  <div class="hero-plane">✈️</div>
  <h1>Voyage <span>Analytics</span></h1>
  <p>AI Powered Travel Intelligence Platform</p>
  <div class="hero-rule"></div>
</div>""", unsafe_allow_html=True)


def action_cards() -> None:
    st.markdown(f"""
<div class="cards">
  <div class="acard" style="background:{GRADIENTS['blue']}">
    <div class="ico">🛫</div>
    <div><h4>Predict Flight Price</h4><p>Fare estimates on 70 routes, R² 0.936</p></div>
  </div>
  <div class="acard" style="background:{GRADIENTS['green']}">
    <div class="ico">👥</div>
    <div><h4>Predict Gender</h4><p>Name-based classification, 87.8% accurate</p></div>
  </div>
  <div class="acard" style="background:{GRADIENTS['orange']}">
    <div class="ico">🏨</div>
    <div><h4>Recommend Hotels</h4><p>Top-3 suggestions, 85.8% hit rate</p></div>
  </div>
</div>""", unsafe_allow_html=True)


def feature_cards() -> None:
    st.markdown(f"""
<div class="feats">
  <div class="feat">
    <div class="circle" style="background:#E0EDFF">🧠</div>
    <h5>AI Powered</h5><p>Gradient boosting and collaborative filtering on 271,888 flight legs.</p>
  </div>
  <div class="feat">
    <div class="circle" style="background:#DFF5E6">📈</div>
    <h5>Honestly Validated</h5><p>Held-out routes, confidence intervals and permutation tests.</p>
  </div>
  <div class="feat">
    <div class="circle" style="background:#F3E8FF">🛡️</div>
    <h5>Reliable &amp; Fast</h5><p>21 ms p95 latency against a 500 ms budget.</p>
  </div>
  <div class="feat">
    <div class="circle" style="background:#FFF1DC">🌍</div>
    <h5>Travel Better</h5><p>Price, gender and hotel insight in one platform.</p>
  </div>
</div>""", unsafe_allow_html=True)


def metric_card(icon: str, title: str, value: str, label: str, footer: str,
                gradient: str, pct: float, status: str = "Active",
                warn: bool = False) -> str:
    """One coloured KPI card. `pct` drives the progress bar (0–1)."""
    cls = "pill warn" if warn else "pill"
    dot = "⚠" if warn else "✅"
    return f"""
<div class="mcard" style="background:{gradient}">
  <div class="top">
    <div class="ico">{icon}</div>
    <div class="{cls}">{dot} {status}</div>
  </div>
  <h3>{title}</h3>
  <div class="big">{value}</div>
  <div class="lbl">{label}</div>
  <div class="bar"><div style="width:{max(0, min(1, pct))*100:.0f}%"></div></div>
  <div class="foot">{footer}</div>
</div>"""


def panel_open(title: str, subtitle: str, badge: str = "") -> str:
    b = (f'<span class="pill" style="background:#DFF5E6;color:#166534;float:right">'
         f'{badge}</span>') if badge else ""
    return f'<div class="panel">{b}<h4>{title}</h4><div class="sub">{subtitle}</div>'


def prow(icon: str, name: str, detail: str, ok: bool = True, bg: str = "#EEF4FF") -> str:
    mark = '<span class="tick">✓</span>' if ok else '<span style="color:#F59E0B">●</span>'
    return (f'<div class="prow"><div class="sq" style="background:{bg}">{icon}</div>'
            f'<div class="txt"><b>{name}</b><small>{detail}</small></div>{mark}</div>')


def section(title: str, subtitle: str = "", icon: str = "") -> None:
    st.markdown(f'<div class="section-title">{icon} {title}</div>'
                + (f'<div class="section-sub">{subtitle}</div>' if subtitle else ""),
                unsafe_allow_html=True)
