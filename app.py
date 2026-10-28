import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ---------------------------------------------------------
# Page Configuration & Dark Sci-Fi Aesthetic
# ---------------------------------------------------------
st.set_page_config(
    page_title="The Search for Earth 2.0",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Dark Cosmic Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 20%, #0d1322 0%, #080c14 60%, #030508 100%);
        color: #e2e8f0;
    }

    /* Hero Header */
    .hero-container {
        padding: 1.8rem 2rem 1.4rem 2rem;
        background: linear-gradient(135deg, rgba(20, 29, 47, 0.75) 0%, rgba(10, 15, 26, 0.85) 100%);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 18px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(14px);
        margin-bottom: 1.8rem;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        background: linear-gradient(90deg, #ffffff 0%, #38bdf8 50%, #00f0ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }

    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.6rem;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.2rem;
        margin-bottom: 1.8rem;
    }

    .kpi-card {
        background: linear-gradient(135deg, rgba(19, 27, 46, 0.8) 0%, rgba(12, 17, 30, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0, 240, 255, 0.15);
    }

    .kpi-card.kpi-blue {
        border-left: 4px solid #38bdf8;
    }
    .kpi-card.kpi-green {
        border-left: 4px solid #10b981;
    }
    .kpi-card.kpi-purple {
        border-left: 4px solid #a855f7;
    }

    .kpi-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94a3b8;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }

    .kpi-sub {
        font-size: 0.8rem;
        color: #64748b;
    }

    /* Plot containers */
    .chart-card {
        background: rgba(15, 22, 38, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }

    .chart-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .chart-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.8rem;
    }

    /* Sidebar Tweaks */
    section[data-testid="stSidebar"] {
        background-color: #0b101b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Ingestion & Preprocessing
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    # Identify dataset path
    file_path = "exoplanets.csv"
    if not os.path.exists(file_path):
        candidates = [f for f in os.listdir(".") if f.startswith("PSCompPars") and f.endswith(".csv")]
        if candidates:
            file_path = candidates[0]
        else:
            raise FileNotFoundError("Exoplanet dataset (exoplanets.csv or PSCompPars_*.csv) was not found in the workspace.")

    # Load dataset with comment handling for NASA Exoplanet Archive headers
    df = pd.read_csv(file_path, comment="#")

    # Safe numeric conversion for required columns
    cols_to_numeric = ["pl_rade", "st_teff", "pl_orbsmax", "sy_dist", "disc_year", "pl_bmasse"]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step requirement: drops rows missing pl_rade, st_teff, or pl_orbsmax
    df = df.dropna(subset=["pl_rade", "st_teff", "pl_orbsmax"]).copy()

    # Step requirement: converts sy_dist to a new distance_light_years column (multiply by 3.26)
    df["distance_light_years"] = df["sy_dist"] * 3.26

    # Step requirement: categorizes planet_type by radius (standard NASA planetary classification)
    def categorize_planet_type(radius):
        if pd.isna(radius):
            return "Unknown"
        elif radius < 1.25:
            return "Terrestrial"
        elif radius <= 2.0:
            return "Super-Earth"
        elif radius <= 6.0:
            return "Neptune-like"
        else:
            return "Gas Giant"

    df["planet_type"] = df["pl_rade"].apply(categorize_planet_type)

    # Calculate Circumstellar Habitable Zone Boundaries (Kopparapu et al. model)
    # L/L_sun ~ (st_teff / 5778)**4 (main-sequence Stefan-Boltzmann scaling)
    t_ratio = df["st_teff"] / 5778.0
    l_star = t_ratio ** 4

    # Conservative & Optimistic Habitable Zone boundaries in AU
    df["hz_inner_cons"] = np.sqrt(l_star / 1.1)
    df["hz_outer_cons"] = np.sqrt(l_star / 0.53)
    df["hz_inner_opt"] = np.sqrt(l_star / 1.77)
    df["hz_outer_opt"] = np.sqrt(l_star / 0.32)

    # General habitable zone flag (optimistic liquid-water boundary)
    df["is_habitable_zone"] = (df["pl_orbsmax"] >= df["hz_inner_opt"]) & (df["pl_orbsmax"] <= df["hz_outer_opt"])

    # Classification status for plotting
    def classify_hz_status(row):
        if row["pl_orbsmax"] < row["hz_inner_opt"]:
            return "Too Hot (Interior)"
        elif row["pl_orbsmax"] > row["hz_outer_opt"]:
            return "Too Cold (Exterior)"
        else:
            return "Habitable Zone"

    df["hz_status"] = df.apply(classify_hz_status, axis=1)

    return df

try:
    df_raw = load_and_preprocess_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# ---------------------------------------------------------
# Sidebar Controls & Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🪐 Mission Controls")
    st.markdown("<span style='color:#94a3b8; font-size:0.85rem;'>Configure observatory filters to survey the cosmos.</span>", unsafe_allow_html=True)
    st.markdown("---")

    # Filter: Discovery Year
    min_year = int(df_raw["disc_year"].min()) if not df_raw["disc_year"].isna().all() else 1995
    max_year = int(df_raw["disc_year"].max()) if not df_raw["disc_year"].isna().all() else 2026
    
    selected_year_range = st.slider(
        "📅 Discovery Year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
        help="Filter exoplanets discovered between the specified years."
    )

    # Filter: Planet Type
    all_planet_types = ["Terrestrial", "Super-Earth", "Neptune-like", "Gas Giant"]
    available_types = [t for t in all_planet_types if t in df_raw["planet_type"].unique()]
    
    selected_planet_types = st.multiselect(
        "🌍 Planet Type",
        options=available_types,
        default=available_types,
        help="Filter by planetary classification based on radius (Earth radii)."
    )

    st.markdown("---")
    st.markdown("#### 🔬 Habitability Parameters")
    
    hz_only = st.checkbox(
        "Show Only Habitable Zone Candidates",
        value=False,
        help="Restrict catalog to exoplanets orbiting within the circumstellar habitable zone."
    )

    hz_model = st.radio(
        "Habitable Zone Boundary Model",
        options=["Optimistic (0.75 - 1.77 AU equivalent)", "Conservative (0.95 - 1.37 AU equivalent)"],
        index=0,
        help="Conservative model uses Runaway/Maximum Greenhouse limits. Optimistic model includes Recent Venus/Early Mars limits."
    )

    st.markdown("---")
    # Quick filter summary
    st.markdown("<span style='font-size:0.8rem; color:#64748b;'>Classification Radii:</span>", unsafe_allow_html=True)
    st.markdown("""
    <ul style='font-size:0.75rem; color:#94a3b8; padding-left: 1.2rem; margin-top: 0.2rem;'>
        <li><b>Terrestrial:</b> &lt; 1.25 R⊕</li>
        <li><b>Super-Earth:</b> 1.25 – 2.0 R⊕</li>
        <li><b>Neptune-like:</b> 2.0 – 6.0 R⊕</li>
        <li><b>Gas Giant:</b> &gt; 6.0 R⊕</li>
    </ul>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Filter Application
# ---------------------------------------------------------
filtered_df = df_raw[
    (df_raw["disc_year"] >= selected_year_range[0]) &
    (df_raw["disc_year"] <= selected_year_range[1]) &
    (df_raw["planet_type"].isin(selected_planet_types))
].copy()

# Adjust habitability check if conservative chosen
if "Conservative" in hz_model:
    filtered_df["is_habitable_zone"] = (filtered_df["pl_orbsmax"] >= filtered_df["hz_inner_cons"]) & (filtered_df["pl_orbsmax"] <= filtered_df["hz_outer_cons"])
else:
    filtered_df["is_habitable_zone"] = (filtered_df["pl_orbsmax"] >= filtered_df["hz_inner_opt"]) & (filtered_df["pl_orbsmax"] <= filtered_df["hz_outer_opt"])

if hz_only:
    filtered_df = filtered_df[filtered_df["is_habitable_zone"]].copy()


# ---------------------------------------------------------
# Top-Level Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">
        <span>The Search for Earth 2.0</span>
    </h1>
    <div class="hero-subtitle">
        Interactive NASA Exoplanet Archive observatory mapping circumstellar habitable zones, discovery trajectories, and 3D stellar landscapes.
    </div>
    <div class="badge-bar">
        <span class="badge-pill">🌌 NASA Exoplanet Archive</span>
        <span class="badge-pill">🔬 Circumstellar Habitable Zone (Liquid Water)</span>
        <span class="badge-pill">🪐 5,614 Filtered Systems</span>
        <span class="badge-pill">✨ Photometric & Spectroscopic Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Top-Level KPI Metrics
# ---------------------------------------------------------
total_planets = len(filtered_df)
earth_sized_worlds = len(filtered_df[filtered_df["planet_type"] == "Terrestrial"])
hz_candidates = int(filtered_df["is_habitable_zone"].sum())

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card kpi-blue">
        <div class="kpi-label">
            <span>Total Planets</span>
            <span>🪐</span>
        </div>
        <div class="kpi-value">{total_planets:,}</div>
        <div class="kpi-sub">Surveyed across {selected_year_range[0]} – {selected_year_range[1]}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    pct_earth = (earth_sized_worlds / total_planets * 100) if total_planets > 0 else 0
    st.markdown(f"""
    <div class="kpi-card kpi-green">
        <div class="kpi-label">
            <span>Earth-Sized Worlds</span>
            <span>🌍</span>
        </div>
        <div class="kpi-value">{earth_sized_worlds:,}</div>
        <div class="kpi-sub">{pct_earth:.1f}% of catalog (Radius &lt; 1.25 R⊕)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    pct_hz = (hz_candidates / total_planets * 100) if total_planets > 0 else 0
    st.markdown(f"""
    <div class="kpi-card kpi-purple">
        <div class="kpi-label">
            <span>Habitable Zone Candidates</span>
            <span>💧</span>
        </div>
        <div class="kpi-value">{hz_candidates:,}</div>
        <div class="kpi-sub">{pct_hz:.1f}% in liquid-water equilibrium zone</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if total_planets == 0:
    st.warning("No planets match the current filter criteria. Please broaden your sidebar selection.")
    st.stop()


# ---------------------------------------------------------
# Palette & Color Constants
# ---------------------------------------------------------
TYPE_COLOR_MAP = {
    "Terrestrial": "#00f0ff",     # Radiant Cyan (Earth analog)
    "Super-Earth": "#818cf8",     # Lavender / Electric Indigo
    "Neptune-like": "#06b6d4",    # Ice Blue
    "Gas Giant": "#f59e0b"        # Golden Amber
}


# ---------------------------------------------------------
# Tabs for Clean Analytical Exploration
# ---------------------------------------------------------
tab_hz, tab_sunburst, tab_3d, tab_data = st.tabs([
    "💧 Habitable Zone Analysis",
    "☀️ Discovery Method Sunburst",
    "🌌 3D Stellar & Radius Atlas",
    "📋 Planetary Catalog Explorer"
])


# ---------------------------------------------------------
# TAB 1: Habitable Zone 2D Scatter Plot
# ---------------------------------------------------------
with tab_hz:
    st.markdown("""
    <div class="chart-desc">
        Plotting <b>Orbital Distance (Semi-Major Axis in AU)</b> against <b>Stellar Effective Temperature (Kelvin)</b>. 
        The shaded emerald corridor marks the circumstellar habitable zone where surface liquid water can exist. 
        Notice the <b>Earth reference landmark (1.0 AU, 5778 K)</b>.
    </div>
    """, unsafe_allow_html=True)

    fig_hz = go.Figure()

    # Generate synthetic temperature curve for continuous Habitable Zone band
    teff_curve = np.linspace(2200, 9500, 200)
    l_curve = (teff_curve / 5778.0) ** 4
    
    if "Conservative" in hz_model:
        hz_in = np.sqrt(l_curve / 1.1)
        hz_out = np.sqrt(l_curve / 0.53)
        band_name = "Conservative Habitable Zone"
    else:
        hz_in = np.sqrt(l_curve / 1.77)
        hz_out = np.sqrt(l_curve / 0.32)
        band_name = "Optimistic Habitable Zone"

    # Add shaded Habitable Zone polygon
    fig_hz.add_trace(go.Scatter(
        x=np.concatenate([hz_in, hz_out[::-1]]),
        y=np.concatenate([teff_curve, teff_curve[::-1]]),
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.18)',
        line=dict(color='rgba(16, 185, 129, 0.45)', width=1, dash='dash'),
        name=band_name,
        hoverinfo='skip'
    ))

    # Add planet scatter points grouped by planet_type
    for p_type in ["Terrestrial", "Super-Earth", "Neptune-like", "Gas Giant"]:
        sub_df = filtered_df[filtered_df["planet_type"] == p_type]
        if not sub_df.empty:
            fig_hz.add_trace(go.Scatter(
                x=sub_df["pl_orbsmax"],
                y=sub_df["st_teff"],
                mode="markers",
                name=p_type,
                marker=dict(
                    color=TYPE_COLOR_MAP.get(p_type, "#94a3b8"),
                    size=7 if p_type in ["Terrestrial", "Super-Earth"] else 6,
                    opacity=0.75,
                    line=dict(width=0.5, color="#ffffff")
                ),
                customdata=np.stack((
                    sub_df["pl_name"],
                    sub_df["hostname"],
                    sub_df["pl_rade"],
                    sub_df["distance_light_years"].round(1),
                    sub_df["discoverymethod"],
                    sub_df["hz_status"]
                ), axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>" +
                    "Host Star: %{customdata[1]}<br>" +
                    "Orbit Semi-Major Axis: %{x:.4f} AU<br>" +
                    "Stellar Temp: %{y:.0f} K<br>" +
                    "Radius: %{customdata[2]:.2f} R⊕<br>" +
                    "Distance: %{customdata[3]} ly<br>" +
                    "Method: %{customdata[4]}<br>" +
                    "Status: <b>%{customdata[5]}</b>" +
                    "<extra></extra>"
                )
            ))

    # Add Earth Benchmark Landmark
    fig_hz.add_trace(go.Scatter(
        x=[1.0],
        y=[5778],
        mode="markers+text",
        name="Earth (Sol)",
        marker=dict(
            symbol="diamond",
            size=15,
            color="#00f0ff",
            line=dict(width=2, color="#ffffff")
        ),
        text=["🌍 Earth (1.0 AU, 5778 K)"],
        textposition="top right",
        textfont=dict(color="#00f0ff", size=13, family="Space Grotesk"),
        hoverinfo="text",
        hovertext="<b>Earth (Home Planet)</b><br>Distance: 1.0 AU<br>Host Star Temp: 5778 K<br>Radius: 1.00 R⊕"
    ))

    fig_hz.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 16, 28, 0.7)",
        xaxis=dict(
            title="Orbit Semi-Major Axis [AU] (Logarithmic Scale)",
            type="log",
            gridcolor="rgba(255, 255, 255, 0.07)",
            zerolinecolor="rgba(255, 255, 255, 0.15)",
            range=[np.log10(0.008), np.log10(200)],
        ),
        yaxis=dict(
            title="Stellar Effective Temperature [K]",
            gridcolor="rgba(255, 255, 255, 0.07)",
            zerolinecolor="rgba(255, 255, 255, 0.15)",
            range=[2200, 9500],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1
        ),
        height=620,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    st.plotly_chart(fig_hz, use_container_width=True)


# ---------------------------------------------------------
# TAB 2: Discovery Method Sunburst Chart
# ---------------------------------------------------------
with tab_sunburst:
    st.markdown("""
    <div class="chart-desc">
        Hierarchical breakdown depicting how different astronomical observation methods contribute to the detection 
        of distinct planetary archetypes (<b>Discovery Method → Planet Type</b>).
    </div>
    """, unsafe_allow_html=True)

    col_sb, col_sb_metrics = st.columns([2.5, 1])

    with col_sb:
        # Prepare aggregated data for sunburst
        sunburst_df = filtered_df.groupby(["discoverymethod", "planet_type"]).size().reset_index(name="count")

        fig_sunburst = px.sunburst(
            sunburst_df,
            path=["discoverymethod", "planet_type"],
            values="count",
            color="planet_type",
            color_discrete_map={
                "Terrestrial": "#00f0ff",
                "Super-Earth": "#818cf8",
                "Neptune-like": "#06b6d4",
                "Gas Giant": "#f59e0b",
                "(?)": "#475569"
            },
            template="plotly_dark"
        )

        fig_sunburst.update_traces(
            textinfo="label+percent entry",
            insidetextorientation="radial",
            marker=dict(line=dict(color="rgba(10, 16, 28, 0.9)", width=1.5)),
            hovertemplate="<b>%{label}</b><br>Worlds: %{value:,}<br>Share: %{percentParent:.1%}<extra></extra>"
        )

        fig_sunburst.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=600,
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(fig_sunburst, use_container_width=True)

    with col_sb_metrics:
        st.markdown("#### 🔭 Discovery Breakdown")
        method_counts = filtered_df["discoverymethod"].value_counts().head(6)
        for method, cnt in method_counts.items():
            pct = (cnt / len(filtered_df)) * 100
            st.markdown(f"""
            <div style="background: rgba(19, 27, 46, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #f1f5f9;">{method}</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #38bdf8;">{cnt:,} <span style="font-size:0.75rem; color:#94a3b8; font-weight:400;">({pct:.1f}%)</span></div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 3: 3D Scatter Plot
# ---------------------------------------------------------
with tab_3d:
    st.markdown("""
    <div class="chart-desc">
        Explore a 3-dimensional mapping of <b>Distance from Earth (Light Years)</b>, <b>Host Star Effective Temperature (Kelvin)</b>, 
        and <b>Planet Radius (Earth Radii)</b>. Rotate, zoom, and pan across cosmic coordinates.
    </div>
    """, unsafe_allow_html=True)

    # Filter out missing distances for the 3D projection
    df_3d = filtered_df.dropna(subset=["distance_light_years"]).copy()

    # Log distance toggle for better spatial distribution
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        dist_scale = st.radio("Distance Axis Scale", ["Linear", "Logarithmic"], horizontal=True)

    if dist_scale == "Logarithmic":
        df_3d["plot_dist"] = np.log10(df_3d["distance_light_years"].clip(lower=1))
        dist_title = "log10(Distance [Light Years])"
    else:
        df_3d["plot_dist"] = df_3d["distance_light_years"]
        dist_title = "Distance [Light Years]"

    fig_3d = px.scatter_3d(
        df_3d,
        x="plot_dist",
        y="st_teff",
        z="pl_rade",
        color="planet_type",
        color_discrete_map=TYPE_COLOR_MAP,
        hover_name="pl_name",
        hover_data={
            "plot_dist": False,
            "distance_light_years": ":.1f",
            "st_teff": ":.0f",
            "pl_rade": ":.2f",
            "pl_orbsmax": ":.4f",
            "discoverymethod": True,
            "planet_type": True
        },
        template="plotly_dark",
        opacity=0.75
    )

    fig_3d.update_traces(
        marker=dict(
            size=4,
            line=dict(width=0.5, color="rgba(255, 255, 255, 0.4)")
        )
    )

    fig_3d.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(
                title=dist_title,
                backgroundcolor="rgba(10, 16, 28, 0.6)",
                gridcolor="rgba(255, 255, 255, 0.08)",
                zerolinecolor="rgba(255, 255, 255, 0.2)"
            ),
            yaxis=dict(
                title="Stellar Temp [K]",
                backgroundcolor="rgba(10, 16, 28, 0.6)",
                gridcolor="rgba(255, 255, 255, 0.08)",
                zerolinecolor="rgba(255, 255, 255, 0.2)"
            ),
            zaxis=dict(
                title="Planet Radius [R⊕]",
                backgroundcolor="rgba(10, 16, 28, 0.6)",
                gridcolor="rgba(255, 255, 255, 0.08)",
                zerolinecolor="rgba(255, 255, 255, 0.2)"
            ),
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2)
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1
        ),
        height=680,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig_3d, use_container_width=True)


# ---------------------------------------------------------
# TAB 4: Planetary Catalog & Earth 2.0 Candidates
# ---------------------------------------------------------
with tab_data:
    st.markdown("""
    <div class="chart-desc">
        Survey the filtered dataset, sort by proximity, or inspect potential <b>Earth 2.0 Candidates</b> 
        (terrestrial or super-Earth worlds in the habitable zone).
    </div>
    """, unsafe_allow_html=True)

    # Search bar & filter
    col_search, col_filter_hz = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("🔍 Search by Planet or Host Star Name", placeholder="e.g., TRAPPIST, Kepler, Proxima...")
    with col_filter_hz:
        earth_2_only = st.checkbox("🎯 Earth 2.0 Analogs Only (Rocky + Habitable Zone)", value=False)

    display_df = filtered_df.copy()

    if search_query:
        mask = (
            display_df["pl_name"].astype(str).str.contains(search_query, case=False, na=False) |
            display_df["hostname"].astype(str).str.contains(search_query, case=False, na=False)
        )
        display_df = display_df[mask]

    if earth_2_only:
        display_df = display_df[
            display_df["is_habitable_zone"] & 
            (display_df["planet_type"].isin(["Terrestrial", "Super-Earth"]))
        ]

    # Present formatted columns
    show_cols = [
        "pl_name", "hostname", "discoverymethod", "disc_year", 
        "planet_type", "pl_rade", "pl_orbsmax", "st_teff", 
        "distance_light_years", "hz_status"
    ]
    rename_cols = {
        "pl_name": "Planet Name",
        "hostname": "Host Star",
        "discoverymethod": "Discovery Method",
        "disc_year": "Year",
        "planet_type": "Classification",
        "pl_rade": "Radius [R⊕]",
        "pl_orbsmax": "Orbit Axis [AU]",
        "st_teff": "Star Temp [K]",
        "distance_light_years": "Distance [ly]",
        "hz_status": "Habitability Zone"
    }

    table_data = display_df[show_cols].rename(columns=rename_cols)
    
    st.dataframe(
        table_data.style.format({
            "Radius [R⊕]": "{:.2f}",
            "Orbit Axis [AU]": "{:.4f}",
            "Star Temp [K]": "{:.0f}",
            "Distance [ly]": "{:.1f}",
            "Year": "{:.0f}"
        }),
        use_container_width=True,
        height=450
    )

    # CSV Download action
    csv_bytes = table_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Exoplanet Data (CSV)",
        data=csv_bytes,
        file_name="earth_2_candidates_filtered.csv",
        mime="text/csv"
    )

# Footer Note
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem; padding-bottom: 2rem;'>"
    "Data Source: NASA Exoplanet Archive Composite Parameters &bull; Built with Streamlit, Plotly &amp; Pandas"
    "</div>",
    unsafe_allow_html=True
)
