"""
SPOTIFY TOP 50 - Music Analytics & Popularity Intelligence
Single-file Streamlit app.  Run with:   streamlit run app.py
Keep this file in the same folder as  top_50_songs_2026.csv
(or upload the CSV from the sidebar when the app opens).
"""
import html
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PREDICTING MOST PLAYED SONGS ON SPOTIFY USING MACHINE LEARNING", page_icon="🎧", layout="wide")

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
GREEN = "#1DB954"
RED = "#E22134"
CSV_NAME = "top_50_songs_2026.csv"

try:
    _VER = tuple(int(x) for x in st.__version__.split(".")[:2])
except ValueError:
    _VER = (1, 50)
WIDE = {"width": "stretch"} if _VER >= (1, 50) else {"use_container_width": True}
_keys = itertools.count()


def esc(x):
    return html.escape(str(x))


def label(col):
    return str(col).replace("_", " ").strip().title()


def style(fig, h=380):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6E6E6"), margin=dict(l=10, r=10, t=55, b=10), height=h,
        title_font=dict(size=16), legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2a2a2a", zeroline=False)
    fig.update_yaxes(gridcolor="#2a2a2a", zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, key=f"chart_{next(_keys)}", **WIDE)


def show_table(data, **kw):
    return st.dataframe(data, **WIDE, **kw)


def cards(items):
    """items = list of (value, caption). Responsive grid of stat cards."""
    body = "".join(
        f'<div class="stat"><div class="stat-v">{esc(v)}</div><div class="stat-l">{esc(c)}</div></div>'
        for v, c in items
    )
    st.markdown(f'<div class="stat-grid">{body}</div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec-title">{esc(title)}</div>'
                + (f'<div class="sec-sub">{esc(sub)}</div>' if sub else ""), unsafe_allow_html=True)


def callout(text, kind="info"):
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Theme (Spotify-inspired dark UI, responsive)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.stApp{background:radial-gradient(1200px 500px at 10% -10%,#0f2a1a 0%,#0b0b0b 55%) fixed;color:#EDEDED}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.4rem;max-width:1250px}
section[data-testid="stSidebar"]{background:#0f0f0f}
h1,h2,h3,h4{color:#fff;letter-spacing:-0.01em}
.hero{padding:28px 26px;border-radius:22px;border:1px solid #1f3d2b;
  background:linear-gradient(135deg,#123a22 0%,#121212 60%);margin-bottom:16px}
.hero h1{font-size:clamp(1.9rem,6vw,3.2rem);font-weight:900;margin:0;line-height:1.05}
.hero h1 span{color:#1DB954}
.hero .tag{color:#1DB954;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  font-size:clamp(.75rem,2.4vw,.95rem);margin-top:6px}
.hero p{color:#b8b8b8;margin:12px 0 0 0;max-width:820px;font-size:clamp(.9rem,2.6vw,1rem)}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:8px 0 18px 0}
.stat{background:linear-gradient(160deg,#1b1b1b,#141414);border:1px solid #262626;border-radius:16px;
  padding:16px 16px;transition:.2s}
.stat:hover{border-color:#1DB954;transform:translateY(-2px)}
.stat-v{font-size:clamp(1.3rem,4.5vw,1.9rem);font-weight:800;color:#1DB954;line-height:1.1;word-break:break-word}
.stat-l{color:#a7a7a7;font-size:.8rem;margin-top:4px;text-transform:uppercase;letter-spacing:.06em}
.sec-title{font-size:1.5rem;font-weight:800;margin:10px 0 2px 0;color:#fff}
.sec-sub{color:#a7a7a7;margin-bottom:12px;font-size:.95rem}
.callout{border-radius:14px;padding:14px 16px;margin:10px 0;font-size:.93rem;line-height:1.5;border:1px solid #2b2b2b;background:#161616}
.callout.warn{border-color:#8a6d1d;background:#241e0c;color:#f3e2a6}
.callout.good{border-color:#1b6b38;background:#0e2216;color:#bdecc9}
.callout.info{border-color:#2d4a7a;background:#101a2b;color:#c5d6f2}
.song-card{background:linear-gradient(135deg,#173d26,#151515);border:1px solid #1f3d2b;border-radius:18px;padding:18px;margin:8px 0 14px 0}
.song-card .nm{font-size:clamp(1.3rem,4.5vw,1.8rem);font-weight:900;color:#fff;line-height:1.15}
.song-card .ar{color:#b8f5cc;font-size:1rem;margin-top:2px}
.pill{display:inline-block;background:#1DB954;color:#000;font-weight:800;border-radius:999px;padding:3px 12px;font-size:.8rem;margin:10px 6px 0 0}
.pill.dark{background:#242424;color:#ddd;font-weight:600}
.feat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin:10px 0}
.feat{background:#161616;border:1px solid #262626;border-radius:12px;padding:10px 12px}
.feat .k{color:#9a9a9a;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}
.feat .v{font-size:1.25rem;font-weight:800;color:#fff}
.feat .a{color:#777;font-size:.72rem}
.rank-row{display:flex;align-items:center;gap:12px;background:#151515;border:1px solid #232323;border-radius:12px;
  padding:9px 12px;margin-bottom:8px;transition:.15s}
.rank-row:hover{border-color:#1DB954;background:#1a1a1a}
.rank-n{font-weight:900;color:#1DB954;width:2.2rem;font-size:1.1rem}
.rank-t{flex:1;min-width:0}
.rank-t b{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#fff}
.rank-t small{color:#9a9a9a}
.rank-s{font-weight:800;color:#fff;background:#222;border-radius:8px;padding:3px 10px}
.big-pred{font-size:clamp(2.4rem,9vw,3.6rem);font-weight:900;color:#1DB954;line-height:1}
.stTabs [data-baseweb="tab-list"]{gap:4px;overflow-x:auto;border-bottom:1px solid #222}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:10px 10px 0 0;padding:8px 14px;color:#b3b3b3;white-space:nowrap}
.stTabs [aria-selected="true"]{color:#1DB954 !important}
.stTabs [data-baseweb="tab-highlight"]{background-color:#1DB954}
.stButton>button,.stFormSubmitButton>button{background:#1DB954;color:#000;border:0;border-radius:999px;font-weight:800;padding:.55rem 1.6rem}
.stButton>button:hover,.stFormSubmitButton>button:hover{background:#1ed760;color:#000}
div[data-testid="stDataFrame"]{border:1px solid #232323;border-radius:14px;overflow:hidden}
@media (max-width:640px){.block-container{padding-left:.7rem;padding-right:.7rem}.hero{padding:20px 16px}.stTabs [data-baseweb="tab"]{padding:8px 10px}}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load the CSV (next to this file, or uploaded from the sidebar)
# ----------------------------------------------------------------------------
def read_csv_any(source):
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise last


with st.sidebar:
    st.markdown("### 🎧 Spotify Top 50")
    uploaded = st.file_uploader("Use a different CSV (optional)", type=["csv"])
    st.caption("By default the app reads " + CSV_NAME + " from the same folder as app.py.")

raw = None
try:
    if uploaded is not None:
        raw = read_csv_any(uploaded)
    else:
        for p in (Path(__file__).resolve().parent / CSV_NAME, Path.cwd() / CSV_NAME):
            if p.exists():
                raw = read_csv_any(p)
                break
except Exception as e:  # unreadable file
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

if raw is None:
    st.markdown('<div class="hero"><h1>SPOTIFY <span>TOP 50</span></h1>'
                '<div class="tag">Music Analytics & Popularity Intelligence</div>'
                f'<p>I could not find <b>{CSV_NAME}</b> next to app.py. Upload it below to continue.</p></div>',
                unsafe_allow_html=True)
    up2 = st.file_uploader("Upload your Spotify CSV", type=["csv"], key="main_upload")
    if up2 is None:
        st.stop()
    raw = read_csv_any(up2)


def find_col(cols, candidates):
    lower = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


df = raw.copy()
df.columns = [str(c).strip() for c in df.columns]
df = df.dropna(how="all").reset_index(drop=True)

NAME = find_col(df.columns, ["track_name", "song", "song_name", "name", "title", "track"])
ARTIST = find_col(df.columns, ["artist", "artists", "artist_name", "artist(s)"])
POP = find_col(df.columns, ["track_popularity", "popularity", "popularity_score"])

obj_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
if NAME is None and obj_cols:
    NAME = obj_cols[0]
if NAME is None:
    st.error("I could not find a song-name column in this CSV.")
    st.stop()

for c in obj_cols:
    df[c] = df[c].astype(str).str.strip()
    if c not in (NAME, ARTIST):
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().mean() > 0.8:
            df[c] = conv

IGNORE = {"rank", "position", "index", "id", "unnamed: 0"}
FEATS = [c for c in df.select_dtypes(include=[np.number]).columns
         if c != POP and str(c).lower() not in IGNORE]
if not FEATS:
    st.error("No numeric audio-feature columns were found in this CSV.")
    st.stop()

HAS_POP = POP is not None and pd.api.types.is_numeric_dtype(df[POP])
METRIC = POP if HAS_POP else (find_col(df.columns, ["energy"]) or FEATS[0])
df["Rank"] = df[METRIC].rank(method="first", ascending=False).astype(int)
df = df.sort_values("Rank").reset_index(drop=True)
CONST = [c for c in FEATS if df[c].nunique() <= 1]          # columns with no variation
VAR_FEATS = [c for c in FEATS if c not in CONST]            # usable for charts / model
N = len(df)
N_ART = df[ARTIST].nunique() if ARTIST else None
dup_names = df[df.duplicated(NAME, keep=False)][NAME].nunique()


def name_artist(r):
    return f"{r[NAME]} — {r[ARTIST]}" if ARTIST else str(r[NAME])


df["_label"] = df.apply(name_artist, axis=1)
df["_rlabel"] = df.apply(lambda r: f"#{r['Rank']} {r[NAME]}", axis=1)
unit_range = {c: bool(df[c].min() >= 0 and df[c].max() <= 1) for c in FEATS}


# ----------------------------------------------------------------------------
# Hero + headline numbers
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <h1>SPOTIFY <span>TOP 50</span></h1>
  <div class="tag">Music Analytics &amp; Popularity Intelligence</div>
  <p>An interactive look at {N} top Spotify songs from <b>{esc(CSV_NAME if uploaded is None else uploaded.name)}</b>:
  every song with its audio features{" and popularity score" if HAS_POP else ""}.
  Explore the library, see the rankings, study the charts, and test a machine-learning model that tries
  to predict popularity.</p>
</div>""", unsafe_allow_html=True)

head = [(N, "Total songs")]
if N_ART is not None:
    head.append((N_ART, "Unique artists"))
if HAS_POP:
    head.append((f"{df[POP].mean():.1f}", "Average popularity"))
head.append((len(FEATS), "Audio features"))
cards(head)

tabs = st.tabs(["🏠 Song Library", "🏆 Top Songs", "📊 Analytics", "🔍 Song Explorer",
                "🤖 AI Predictor", "📈 Model Performance", "ℹ️ About"])


# ----------------------------------------------------------------------------
# Reusable song detail panel (card + features + radar)
# ----------------------------------------------------------------------------
def radar_fig(row, title):
    axes, vals, avgs = [], [], []
    for c in VAR_FEATS:
        lo, hi = df[c].min(), df[c].max()
        if unit_range[c]:
            v, a, nm = row[c], df[c].mean(), label(c)
        else:
            v, a, nm = (row[c] - lo) / (hi - lo), (df[c].mean() - lo) / (hi - lo), label(c) + " (scaled)"
        axes.append(nm); vals.append(float(v)); avgs.append(float(a))
    if len(axes) < 3:
        return None
    axes_c, vals_c, avgs_c = axes + axes[:1], vals + vals[:1], avgs + avgs[:1]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=avgs_c, theta=axes_c, name="Dataset average",
                                  line=dict(color="#8a8a8a", dash="dot"), fill="none"))
    fig.add_trace(go.Scatterpolar(r=vals_c, theta=axes_c, name=str(row[NAME])[:28],
                                  line=dict(color=GREEN, width=3), fill="toself",
                                  fillcolor="rgba(29,185,84,0.28)"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 1], gridcolor="#333", tickfont=dict(size=9)),
                                 angularaxis=dict(gridcolor="#333")),
                      title=title, showlegend=True, legend=dict(orientation="h", y=-0.1))
    return style(fig, 430)


def song_panel(row, radar=True):
    pills = f'<span class="pill">Rank #{int(row["Rank"])} of {N}</span>'
    if HAS_POP:
        pills += f'<span class="pill dark">{label(POP)}: {row[POP]:g}</span>'
    st.markdown(f'<div class="song-card"><div class="nm">{esc(row[NAME])}</div>'
                + (f'<div class="ar">{esc(row[ARTIST])}</div>' if ARTIST else "") + pills + "</div>",
                unsafe_allow_html=True)
    boxes = ""
    for c in FEATS:
        v = row[c]
        boxes += (f'<div class="feat"><div class="k">{esc(label(c))}</div><div class="v">{v:g}</div>'
                  f'<div class="a">avg {df[c].mean():.2f}</div></div>')
    st.markdown(f'<div class="feat-grid">{boxes}</div>', unsafe_allow_html=True)
    if radar:
        fig = radar_fig(row, "Audio profile vs. dataset average")
        if fig is not None:
            show(fig)


# ----------------------------------------------------------------------------
# TAB 1 - Song Library
# ----------------------------------------------------------------------------
with tabs[0]:
    section("Complete Song Library", f"All {N} songs from your CSV. Search, filter, sort, "
            "then click a row (or use the picker below) to open its details.")
    c1, c2 = st.columns([2, 2])
    q = c1.text_input("🔎 Search song or artist", placeholder="e.g. Blinding Lights, Taylor Swift")
    if ARTIST:
        arts = c2.multiselect("Filter by artist", sorted(df[ARTIST].unique()))
    else:
        arts = []
    c3, c4, c5 = st.columns([2, 2, 1.4])
    sort_options = {"Rank": "Rank"}
    sort_options.update({label(c): c for c in ([POP] if HAS_POP else []) + FEATS})
    sort_label = c3.selectbox("Sort by", list(sort_options.keys()))
    order = c4.radio("Order", ["Highest first", "Lowest first"], horizontal=True)
    if HAS_POP:
        lo_p, hi_p = int(df[POP].min()), int(df[POP].max())
        pr = st.slider("Popularity range", lo_p, hi_p, (lo_p, hi_p)) if lo_p < hi_p else (lo_p, hi_p)
    f = df.copy()
    if q:
        m = f[NAME].str.contains(q, case=False, regex=False)
        if ARTIST:
            m = m | f[ARTIST].str.contains(q, case=False, regex=False)
        f = f[m]
    if arts:
        f = f[f[ARTIST].isin(arts)]
    if HAS_POP:
        f = f[(f[POP] >= pr[0]) & (f[POP] <= pr[1])]
    sc = sort_options[sort_label]
    asc = (order == "Lowest first") if sc != "Rank" else (order == "Highest first")
    f = f.sort_values(sc, ascending=asc, kind="stable")

    st.caption(f"Showing {len(f)} of {N} songs")
    if f.empty:
        st.warning("No songs match your search and filters. Try clearing them.")
    else:
        view = pd.DataFrame({"Rank": f["Rank"], "Song": f[NAME]})
        if ARTIST:
            view["Artist"] = f[ARTIST]
        cfg = {"Rank": st.column_config.NumberColumn("#", width="small", format="%d")}
        for c in ([POP] if HAS_POP else []) + FEATS:
            lab = label(c) if c != POP else "Popularity"
            view[lab] = f[c]
            if c == POP:
                cfg[lab] = st.column_config.ProgressColumn(lab, min_value=0, max_value=100, format="%d")
            else:
                fmt = "%.0f" if "tempo" in str(c).lower() else ("%.1f" if "loud" in str(c).lower() else "%.2f")
                cfg[lab] = st.column_config.NumberColumn(lab, format=fmt)
        sel_idx = None
        try:
            ev = st.dataframe(view, hide_index=True, column_config=cfg, height=520,
                              on_select="rerun", selection_mode="single-row", key="lib_table", **WIDE)
            rows = ev.selection.rows if ev is not None else []
            if rows:
                sel_idx = view.index[rows[0]]
        except TypeError:  # very old Streamlit: no row selection
            show_table(view, hide_index=True, column_config=cfg, height=520)

        choices = list(f.index)
        default = choices.index(sel_idx) if sel_idx in choices else 0
        pick = st.selectbox("Open song details", choices, index=default,
                            format_func=lambda i: f"#{df.loc[i, 'Rank']}  {df.loc[i, '_label']}",
                            key=f"pick_{sel_idx}")
        song_panel(df.loc[pick])


# ----------------------------------------------------------------------------
# TAB 2 - Top / Bottom songs
# ----------------------------------------------------------------------------
def ranking_block(sub, title, color):
    left, right = st.columns([1, 1.25])
    with left:
        rows = ""
        for _, r in sub.iterrows():
            rows += (f'<div class="rank-row"><div class="rank-n">#{int(r["Rank"])}</div>'
                     f'<div class="rank-t"><b>{esc(r[NAME])}</b><small>{esc(r[ARTIST]) if ARTIST else ""}</small></div>'
                     f'<div class="rank-s">{r[METRIC]:g}</div></div>')
        st.markdown(rows, unsafe_allow_html=True)
    with right:
        s = sub.sort_values("Rank")
        fig = px.bar(s, x=METRIC, y="_rlabel", orientation="h", text=METRIC,
                     hover_data={"_rlabel": False, METRIC: True}, title=title)
        fig.update_traces(marker_color=color, textposition="outside")
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_xaxes(title=label(METRIC) if METRIC != POP else "Popularity score",
                         range=[0, max(100 if HAS_POP else 1, float(s[METRIC].max()) * 1.12)])
        show(style(fig, 440))


with tabs[1]:
    if HAS_POP:
        section("Top 10 Most Popular Songs", "Ranked by the popularity score in your CSV (higher = more popular).")
        ranking_block(df.head(10), "Top 10 by popularity", GREEN)
        section("Bottom 10 (Least Popular in this Top 50)",
                "Still hits, but the lowest popularity scores in this dataset.")
        ranking_block(df.tail(10).sort_values("Rank"), "Bottom 10 by popularity", RED)
    else:
        callout(f"Your CSV has no popularity column, so songs are ranked by <b>{esc(label(METRIC))}</b> instead.", "warn")
        ranking_block(df.head(10), f"Top 10 by {label(METRIC)}", GREEN)
        ranking_block(df.tail(10).sort_values("Rank"), f"Bottom 10 by {label(METRIC)}", RED)


# ----------------------------------------------------------------------------
# TAB 3 - Analytics
# ----------------------------------------------------------------------------
with tabs[2]:
    section("Exploratory Data Analysis", "Every chart is built only from columns that exist in your CSV.")
    sub = st.tabs(["Popularity", "Audio features", "Features vs popularity", "Correlation", "Artists"])

    with sub[0]:
        if HAS_POP:
            fig = px.histogram(df, x=POP, nbins=12, title="Popularity distribution", marginal="box")
            fig.update_traces(marker_color=GREEN)
            fig.update_xaxes(title="Popularity score"); fig.update_yaxes(title="Number of songs")
            show(style(fig, 420))
            cards([(f"{df[POP].mean():.1f}", "Mean"), (f"{df[POP].median():.1f}", "Median"),
                   (f"{df[POP].min():g}", "Lowest"), (f"{df[POP].max():g}", "Highest"),
                   (f"{df[POP].std():.1f}", "Std. deviation")])
            st.caption("All songs here are already chart hits, so scores sit in a fairly narrow high range. "
                       "That makes popularity harder to predict.")
        else:
            st.info("No popularity column in this CSV, so popularity charts are not available.")

    with sub[1]:
        feat = st.selectbox("Choose an audio feature", VAR_FEATS, format_func=label)
        fig = px.histogram(df, x=feat, nbins=15, marginal="box", hover_data=[NAME],
                           title=f"{label(feat)} distribution")
        fig.update_traces(marker_color=GREEN)
        fig.update_xaxes(title=label(feat)); fig.update_yaxes(title="Number of songs")
        show(style(fig, 420))
        stats = df[VAR_FEATS].describe().T[["mean", "std", "min", "50%", "max"]].round(3)
        stats.index = [label(i) for i in stats.index]
        stats.columns = ["Mean", "Std", "Min", "Median", "Max"]
        show_table(stats)
        if CONST:
            st.caption("Not charted (every song has the same value): " + ", ".join(label(c) for c in CONST))

    with sub[2]:
        if HAS_POP:
            xf = st.selectbox("Compare popularity with", VAR_FEATS, format_func=label)
            d = df[[xf, POP, NAME] + ([ARTIST] if ARTIST else [])]
            fig = px.scatter(d, x=xf, y=POP, hover_name=NAME, hover_data=[ARTIST] if ARTIST else None,
                             title=f"{label(xf)} vs Popularity")
            fig.update_traces(marker=dict(color=GREEN, size=10, line=dict(width=1, color="#000")))
            slope, icpt = np.polyfit(d[xf], d[POP], 1)
            xs = np.linspace(d[xf].min(), d[xf].max(), 20)
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines", name="Trend line",
                                     line=dict(color="#fff", dash="dash")))
            fig.update_xaxes(title=label(xf)); fig.update_yaxes(title="Popularity score")
            show(style(fig, 440))
            cor = df[VAR_FEATS + [POP]].corr()[POP].drop(POP).sort_values()
            fig2 = px.bar(x=cor.values, y=[label(i) for i in cor.index], orientation="h",
                          title="Correlation of each feature with popularity",
                          labels={"x": "Correlation (-1 to +1)", "y": ""}, text=cor.round(2).values)
            fig2.update_traces(marker_color=[GREEN if v >= 0 else RED for v in cor.values], textposition="outside")
            show(style(fig2, 400))
            top = cor.abs().sort_values(ascending=False).index[0]
            callout(f"Strongest link with popularity: <b>{esc(label(top))}</b> (correlation {cor[top]:+.2f}). "
                    "Values close to 0 mean almost no straight-line relationship.", "info")
        else:
            st.info("This section needs a popularity column.")

    with sub[3]:
        cols = VAR_FEATS + ([POP] if HAS_POP else [])
        if len(cols) >= 2:
            corr = df[cols].corr().round(2)
            corr.index = corr.columns = [label(c) if c != POP else "Popularity" for c in cols]
            fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                            color_continuous_scale=[[0, RED], [0.5, "#161616"], [1, GREEN]],
                            title="Correlation heatmap (numeric columns)")
            show(style(fig, 520))
            st.caption("Green = move together, red = move in opposite directions, dark = no clear relationship.")
        else:
            st.info("Not enough numeric columns for a heatmap.")

    with sub[4]:
        if ARTIST:
            cnt = df[ARTIST].value_counts().head(10).sort_values()
            fig = px.bar(x=cnt.values, y=cnt.index, orientation="h", title="Artists with the most songs",
                         labels={"x": "Songs in dataset", "y": ""}, text=cnt.values)
            fig.update_traces(marker_color=GREEN)
            show(style(fig, 420))
            if HAS_POP:
                g = df.groupby(ARTIST)[POP].agg(["mean", "count"]).reset_index()
                g = g[g["count"] >= 2].sort_values("mean")
                if len(g):
                    fig = px.bar(g, x="mean", y=ARTIST, orientation="h", text=g["mean"].round(1),
                                 hover_data={"count": True}, title="Average popularity (artists with 2+ songs)",
                                 labels={"mean": "Average popularity", ARTIST: ""})
                    fig.update_traces(marker_color=GREEN, textposition="outside")
                    show(style(fig, 80 + 38 * len(g)))
                else:
                    st.info("No artist has 2 or more songs, so there is nothing to compare.")
            st.caption("Artists are counted exactly as written in the CSV, so a collaboration "
                       "(e.g. 'A & B') counts as its own artist.")
        else:
            st.info("No artist column in this CSV.")


# ----------------------------------------------------------------------------
# TAB 4 - Song Explorer
# ----------------------------------------------------------------------------
with tabs[3]:
    section("Song Explorer", "Pick any song and see its full audio fingerprint compared with the average song.")
    e1, e2 = st.columns(2)
    ex_art = e1.selectbox("Artist", ["All artists"] + (sorted(df[ARTIST].unique()) if ARTIST else []))
    pool = df if ex_art == "All artists" else df[df[ARTIST] == ex_art]
    ex_q = e2.text_input("Search song", key="ex_q", placeholder="type part of a title")
    if ex_q:
        pool = pool[pool[NAME].str.contains(ex_q, case=False, regex=False)]
    if pool.empty:
        st.warning("No songs match. Change the artist or search text.")
    else:
        idx = st.selectbox("Select a song", list(pool.index), key="ex_pick",
                           format_func=lambda i: f"#{df.loc[i, 'Rank']}  {df.loc[i, '_label']}")
        song_panel(df.loc[idx])
        if len(df) > 1 and len(VAR_FEATS) >= 1:
            sims = df[VAR_FEATS].copy()
            z = (sims - sims.mean()) / sims.std().replace(0, 1)
            dist = ((z - z.loc[idx]) ** 2).sum(axis=1) ** 0.5
            near = dist.drop(idx).nsmallest(3).index
            st.markdown("**Most similar songs by audio features**")
            show_table(df.loc[near, ["Rank", NAME] + ([ARTIST] if ARTIST else []) + ([POP] if HAS_POP else [])]
                       .rename(columns={NAME: "Song", POP: "Popularity"}), hide_index=True)


# ----------------------------------------------------------------------------
# Machine learning (regression on popularity, grouped CV to avoid leakage)
# ----------------------------------------------------------------------------
def build_models():
    return {
        "Average baseline (no learning)": DummyRegressor(strategy="mean"),
        "Ridge Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=3,
                                                             random_state=42)),
    }


def score(y, p):
    return {"R2": float(r2_score(y, p)), "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p)))}


@st.cache_resource(show_spinner="Training model on your songs...")
def run_ml(data, target, group_col, features):
    d = data.dropna(subset=[target]).reset_index(drop=True)
    X, y = d[list(features)].astype(float), d[target].astype(float)
    groups = d[group_col].str.lower().str.strip().values if group_col else np.arange(len(d))
    n_groups = len(set(groups))
    n_splits = max(2, min(5, n_groups))
    cv = GroupKFold(n_splits=n_splits)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(split.split(X, y, groups))
    rows, oof = [], {}
    for nm, mdl in build_models().items():
        p = cross_val_predict(clone(mdl), X, y, cv=cv, groups=groups)
        oof[nm] = p
        cvm = score(y, p)
        fitted = clone(mdl).fit(X.iloc[tr], y.iloc[tr])
        hm = score(y.iloc[te], fitted.predict(X.iloc[te]))
        rows.append({"Model": nm, "CV R²": cvm["R2"], "CV MAE": cvm["MAE"], "CV RMSE": cvm["RMSE"],
                     "Test R²": hm["R2"], "Test MAE": hm["MAE"], "Test RMSE": hm["RMSE"]})
    table = pd.DataFrame(rows)
    real = table[~table["Model"].str.startswith("Average")]
    best = real.sort_values("CV R²", ascending=False).iloc[0]["Model"]
    final = clone(build_models()[best]).fit(X, y)
    last = final[-1]
    if hasattr(last, "feature_importances_"):
        imp = pd.Series(last.feature_importances_, index=features)
    else:
        c = np.abs(last.coef_)
        imp = pd.Series(c / c.sum() if c.sum() else c, index=features)
    return {"table": table, "best": best, "model": final, "oof": oof[best], "y": y.values,
            "names": d[NAME].values, "importance": imp.sort_values(), "n": len(d), "n_groups": n_groups,
            "n_train": len(tr), "n_test": len(te), "n_splits": n_splits, "features": list(features)}


ML = None
ML_ERR = None
if not HAS_POP:
    ML_ERR = "Your CSV has no popularity column, so there is no target to predict."
elif len(VAR_FEATS) < 2:
    ML_ERR = "Not enough varying audio-feature columns to train a model."
elif N < 20:
    ML_ERR = "Too few songs (fewer than 20) to train and test a model reliably."
else:
    try:
        ML = run_ml(df[[NAME, POP] + VAR_FEATS], POP, NAME, tuple(VAR_FEATS))
    except Exception as e:
        ML_ERR = f"The model could not be trained: {e}"

BEATS = ML is not None and float(ML["table"].set_index("Model").loc[ML["best"], "CV R²"]) > 0


# ----------------------------------------------------------------------------
# TAB 5 - AI Song Popularity Predictor
# ----------------------------------------------------------------------------
with tabs[4]:
    section("AI SONG POPULARITY PREDICTOR",
            "Enter a song's audio features and a model trained on your 50 songs estimates its popularity score.")
    if ML is None:
        st.error(ML_ERR)
    else:
        tb = ML["table"].set_index("Model")
        best_r2 = float(tb.loc[ML["best"], "CV R²"])
        base_mae = float(tb.loc[[m for m in tb.index if m.startswith("Average")][0], "CV MAE"])
        best_mae = float(tb.loc[ML["best"], "CV MAE"])
        if BEATS:
            callout(f"<b>Model:</b> {esc(ML['best'])}, a regression model. Cross-validated R² = {best_r2:.2f} "
                    f"(typical error ≈ {best_mae:.1f} points). It does better than just guessing the average, "
                    "but with only 50 songs treat the result as a rough estimate.", "good")
        else:
            callout(f"<b>Honest result:</b> the best model ({esc(ML['best'])}) scored a cross-validated R² of "
                    f"<b>{best_r2:.2f}</b>, which is no better than always guessing the average popularity "
                    f"(average-guess error ≈ {base_mae:.1f} points, model error ≈ {best_mae:.1f}). "
                    "In plain words: with these features and only 50 songs, popularity is <b>not reliably predictable</b>. "
                    "The predictor below is a working demonstration of the method, not a trustworthy forecast.", "warn")

        start = st.selectbox("Start from a song in the dataset (optional)", ["— none, use average values —"]
                             + list(df.index), format_func=lambda i: i if isinstance(i, str)
                             else f"{df.loc[i, NAME]}" + (f" — {df.loc[i, ARTIST]}" if ARTIST else ""))
        with st.form("predict_form"):
            vals = {}
            cols = st.columns(2)
            for i, c in enumerate(ML["features"]):
                lo, hi = float(df[c].min()), float(df[c].max())
                init = float(df[c].mean()) if isinstance(start, str) else float(df.loc[start, c])
                step = round((hi - lo) / 100, 4) or 0.01
                vals[c] = cols[i % 2].slider(label(c), lo, hi, min(max(init, lo), hi), step=step,
                                             key=f"in_{c}_{start}",
                                             help=f"Range seen in your dataset: {lo:g} to {hi:g}")
            go_btn = st.form_submit_button("🎯 Predict popularity")
        if go_btn:
            bad = [label(c) for c, v in vals.items() if v is None or not np.isfinite(v)]
            if bad:
                st.error("Please give a valid number for: " + ", ".join(bad))
            else:
                x = pd.DataFrame([vals])[ML["features"]]
                pred = float(np.clip(ML["model"].predict(x)[0], 0, 100))
                mean_p, q1, q3 = float(df[POP].mean()), float(df[POP].quantile(.25)), float(df[POP].quantile(.75))
                rmse = float(tb.loc[ML["best"], "CV RMSE"])
                level = "High" if pred >= q3 else ("Low" if pred <= q1 else "Medium")
                r1, r2c = st.columns([1, 1])
                with r1:
                    st.markdown(f'<div class="song-card"><div class="nm">Predicted popularity</div>'
                                f'<div class="big-pred">{pred:.1f}<span style="font-size:1.1rem;color:#aaa"> / 100</span></div>'
                                f'<span class="pill">{level} within this Top 50</span>'
                                f'<span class="pill dark">± {rmse:.1f} typical error</span></div>', unsafe_allow_html=True)
                    diff = pred - mean_p
                    st.write(f"This is **{abs(diff):.1f} points {'above' if diff >= 0 else 'below'}** the dataset average "
                             f"({mean_p:.1f}). Songs in your data range from {df[POP].min():g} to {df[POP].max():g}, "
                             f"so the typical error of about {rmse:.1f} points is large compared with that spread.")
                    if not BEATS:
                        st.caption("Reminder: this model did not beat the average-guess baseline, so don't over-read this number.")
                with r2c:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=pred, number={"valueformat": ".1f"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": GREEN},
                               "bgcolor": "#1a1a1a",
                               "steps": [{"range": [float(df[POP].min()), float(df[POP].max())], "color": "#26352c"}],
                               "threshold": {"line": {"color": "#fff", "width": 3}, "value": mean_p}},
                        title={"text": "Gauge (white line = dataset average)"}))
                    show(style(fig, 300))
                z = (df[VAR_FEATS] - df[VAR_FEATS].mean()) / df[VAR_FEATS].std()
                zi = (pd.Series(vals)[VAR_FEATS] - df[VAR_FEATS].mean()) / df[VAR_FEATS].std()
                near = (((z - zi) ** 2).sum(axis=1) ** 0.5).nsmallest(3).index
                st.markdown("**Real songs with the most similar audio features**")
                show_table(df.loc[near, ["Rank", NAME] + ([ARTIST] if ARTIST else []) + [POP]]
                           .rename(columns={NAME: "Song", POP: "Popularity"}), hide_index=True)


# ----------------------------------------------------------------------------
# TAB 6 - Model Performance
# ----------------------------------------------------------------------------
with tabs[5]:
    section("Model Performance", "How well does the model really predict popularity?")
    if ML is None:
        st.error(ML_ERR)
    else:
        tb = ML["table"].set_index("Model")
        b = tb.loc[ML["best"]]
        st.markdown(f"**Task:** regression (predict the number `{POP}`) &nbsp;|&nbsp; **Chosen model:** {ML['best']} "
                    f"&nbsp;|&nbsp; **Inputs:** {', '.join(label(c) for c in ML['features'])} "
                    f"&nbsp;|&nbsp; The target column is never used as an input.", unsafe_allow_html=True)
        cards([(f"{b['CV R²']:.2f}", "R² (cross-validated)"), (f"{b['CV MAE']:.1f}", "MAE (cross-validated)"),
               (f"{b['CV RMSE']:.1f}", "RMSE (cross-validated)"), (f"{b['Test R²']:.2f}", "R² (hold-out test)")])
        if not BEATS:
            callout("R² at or below 0 means the model predicts <b>no better than the average popularity</b>. "
                    "That is a genuine finding for this dataset, not a bug.", "warn")
        show_table(tb.round(3), column_config={c: st.column_config.NumberColumn(c, format="%.2f") for c in tb.columns})
        st.caption(f"CV = {ML['n_splits']}-fold cross-validation where a song's duplicates always stay on the same side, "
                   f"so the model never sees a copy of a test song. Hold-out test = {ML['n_train']} songs train / "
                   f"{ML['n_test']} songs test (only {ML['n_test']} test songs, so that column is very noisy). "
                   "The chosen model is the better of Ridge and Random Forest by CV R².")
        g1, g2 = st.columns(2)
        with g1:
            od = pd.DataFrame({"Actual": ML["y"], "Predicted": ML["oof"], "Song": ML["names"]})
            fig = px.scatter(od, x="Actual", y="Predicted", hover_name="Song",
                             title="Actual vs predicted popularity (out-of-fold)")
            fig.update_traces(marker=dict(color=GREEN, size=9))
            lo_, hi_ = float(min(od.Actual.min(), od.Predicted.min())), float(max(od.Actual.max(), od.Predicted.max()))
            fig.add_trace(go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode="lines", name="Perfect prediction",
                                     line=dict(color="#fff", dash="dash")))
            show(style(fig, 400))
        with g2:
            imp = ML["importance"]
            fig = px.bar(x=imp.values, y=[label(i) for i in imp.index], orientation="h",
                         title="Feature importance", labels={"x": "Relative importance", "y": ""})
            fig.update_traces(marker_color=GREEN)
            show(style(fig, 400))
        section("What do these metrics mean?")
        st.markdown("""
- **R²** – how much of the ups and downs in popularity the model explains. 1.0 is perfect, **0 means no better than guessing the average**, below 0 means worse than guessing the average.
- **MAE (Mean Absolute Error)** – the average number of popularity points the prediction is off by. Easy to understand: lower is better.
- **RMSE** – like MAE but punishes big mistakes more. Lower is better.
- **Cross-validation** – the data is split several times; each time the model is tested on songs it has not seen. This is more trustworthy than one split when you only have 50 songs.
- Classification metrics (accuracy, precision, recall, F1, confusion matrix) are **not shown** because this is a regression task, not a classification one.
""")


# ----------------------------------------------------------------------------
# TAB 7 - About
# ----------------------------------------------------------------------------
with tabs[6]:
    section("About this project")
    feat_list = ", ".join(label(c) for c in FEATS)
    st.markdown(f"""
**What it does.** This dashboard turns a CSV of {N} Spotify songs into an interactive music-analytics website:
a searchable song library, rankings, charts, a song explorer, and a machine-learning popularity predictor.

**What the dataset contains.** {N} rows and {len(raw.columns)} columns: song name{", artist" if ARTIST else ""}{", popularity score" if HAS_POP else ""}
and these audio features: {feat_list}. Nothing is invented or hard-coded: the app reads your CSV every time it starts.

**Why audio features are useful.** Spotify describes each track with numbers: how danceable it is, how energetic, how loud,
how happy it sounds (valence), how acoustic, and its tempo in beats per minute. They let us compare songs objectively.

**What EDA means here.** Exploratory Data Analysis is looking at the data before modelling: distributions, relationships
between features, correlations, and which artists appear most. It tells us what patterns exist (and whether any exist at all).

**How machine learning is used.** Because popularity is a number, this is a *regression* problem. The app compares a Ridge
Regression and a Random Forest against a simple "always guess the average" baseline using cross-validation, then keeps the better
of the two. The popularity column is the target and is never used as an input.

**Limitations (important).**
- Only {N} songs: far too few for a reliable model.
- Every song is already a hit, so popularity only ranges from {df[POP].min() if HAS_POP else '?'} to {df[POP].max() if HAS_POP else '?'}, which leaves little to learn from.
- {f"{dup_names} song titles appear twice with identical audio features but different popularity scores, so the same sound is linked to two different answers. The model evaluation keeps such duplicates together to avoid cheating." if dup_names else "No duplicate titles were found."}
- {("These columns never change, so they carry no information and are left out of charts and the model: " + ", ".join(label(c) for c in CONST) + ".") if CONST else "All audio features vary."}
- Popularity depends on marketing, release date, playlists, and fans, none of which are in the audio features. There is no genre, year, or duration column in this file.
- Correlation is not causation.
""")
    st.caption("Built with Python, Streamlit, Pandas, NumPy, Scikit-learn and Plotly.")
