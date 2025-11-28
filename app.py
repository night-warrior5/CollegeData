import warnings
import os
import logging
import json
import sqlite3
from sqlalchemy import text  # Imported 'text'

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from CollegeBase import load_data


#  Global Configuration and User Experience Polishing


warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)

# Optional imports for profile matching
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.sidebar.warning(
        "Install scikit-learn (`pip install scikit-learn`) to enable the 'Find Similar Profiles' tab."
    )


# Streamlit page config
st.set_page_config(
    layout="wide",
    page_title="Admissions Insights Dashboard",
    page_icon="🎓",
)

# Simple custom styling
st.markdown(
    """
    <style>
        .main > div {
            padding-top: 1rem;
        }
        .metric-label {
            font-weight: 600 !important;
        }
        .filter-chip {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 999px;
            background-color: #f0f2f6;
            margin: 0 4px 4px 0;
            font-size: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Admissions Insights Dashboard")
st.caption(
    "Explore real applicant data, filter by profile, and analyze acceptance patterns at top schools."
)



#  ACT to SAT Concordance


ACT_TO_SAT_CONCORDANCE = {
    36: 1600, 35: 1570, 34: 1520, 33: 1490, 32: 1450, 31: 1420, 30: 1390,
    29: 1360, 28: 1330, 27: 1300, 26: 1260, 25: 1230, 24: 1200, 23: 1160,
    22: 1130, 21: 1100, 20: 1060, 19: 1020, 18: 980,  17: 940,  16: 900,
    15: 860,  14: 810,  13: 770,  12: 720,  11: 670,  10: 620,  9: 560
}
SAT_TO_ACT_EXACT = {v: k for k, v in ACT_TO_SAT_CONCORDANCE.items()}
SAT_CONCORDANCE_POINTS = sorted(SAT_TO_ACT_EXACT.keys())


def get_act_equivalent(sat_score: int):
    #Find the nearest ACT equivalent for a given SAT score.
    if sat_score in SAT_TO_ACT_EXACT:
        return SAT_TO_ACT_EXACT[sat_score]
    closest_sat = min(SAT_CONCORDANCE_POINTS, key=lambda x: abs(x - sat_score))
    return SAT_TO_ACT_EXACT[closest_sat]



#  Rating System (Persistent Database)


# Initialize the connection to a persistent SQLite database
conn = st.connection("ratings_db", type="sql", url="sqlite:///profile_ratings.db")

def init_db():
    #Create the ratings table if it doesn't exist.
    with conn.session as s:
        # Wrapped query in text()
        s.execute(text("""
            CREATE TABLE IF NOT EXISTS ratings (
                profile_id TEXT,
                rating INTEGER
            );
        """))
        s.commit()

# Run the DB initialization once
init_db()


def save_rating(profile_id, rating):
    #Save a rating for a profile to the database.
    with conn.session as s:
        # Wrapped query in text()
        s.execute(
            text("INSERT INTO ratings (profile_id, rating) VALUES (:id, :rating)"),
            params={"id": profile_id, "rating": rating},
        )
        s.commit()


def get_avg_rating(profile_id):
    #Get average rating for a profile from the database.
    with conn.session as s:
        # Wrapped query in text()
        result = s.execute(
            text("SELECT AVG(rating) FROM ratings WHERE profile_id = :id"),
            params={"id": profile_id},
        ).fetchone()

    avg = result[0] if result else None
    return float(avg) if avg is not None else None



#  Data Loading and Caching

@st.cache_data
def get_data():
    #Load and cache the profile data.
    try:
        df_ = load_data("profiles.jsonl")
        return df_
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


df = get_data()

if df.empty:
    st.error("No data found in profiles.jsonl. Please add profiles first.")
    st.stop()

# Precompute option lists from raw data
all_majors = sorted(list(df["majors"].explode().dropna().astype(str).unique()))
all_races = sorted(list(df["race"].explode().dropna().astype(str).unique()))
acceptances_set = set(df["acceptances"].explode().dropna().astype(str).unique())
rejections_set = set(df["rejections"].explode().dropna().astype(str).unique())
all_schools = sorted(list(acceptances_set | rejections_set))

min_sat_eq = 400
max_sat_eq = 1600



#  Session State Initialization

if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = False
    st.session_state.active_gpa_range = (0.0, 4.0)
    st.session_state.active_sat_eq_range = (min_sat_eq, max_sat_eq)
    st.session_state.active_is_stem = False
    st.session_state.active_test_optional = False
    st.session_state.active_major_filter = []
    st.session_state.active_race_filter = []
    st.session_state.active_tier_filter = []

if "selected_profile_idx" not in st.session_state:
    st.session_state.selected_profile_idx = None


#  Sidebar Filters

st.sidebar.header("Filters")
st.sidebar.caption(
    "Adjust the filters below and click **Apply Filters** to update the dashboard."
)

with st.sidebar.expander("📊 Academic Filters", expanded=True):
    gpa_range = st.slider(
        "Unweighted GPA Range",
        min_value=0.0,
        max_value=4.0,
        value=st.session_state.active_gpa_range,
        step=0.01,
        key="widget_gpa_range",
    )

    sat_eq_range = st.slider(
        "SAT Equivalent Score Range",
        min_value=min_sat_eq,
        max_value=max_sat_eq,
        value=st.session_state.active_sat_eq_range,
        step=10,
        key="widget_sat_eq_range",
    )

    # Display ACT equivalents
    min_act_eq = get_act_equivalent(sat_eq_range[0])
    max_act_eq = get_act_equivalent(sat_eq_range[1])
    st.caption(
        f"Selected Range: **SAT {sat_eq_range[0]}–{sat_eq_range[1]}** "
        f"(≈ **ACT {min_act_eq}–{max_act_eq}**)"
    )

with st.sidebar.expander("🎯 Profile Attributes", expanded=True):
    is_test_optional = st.checkbox(
        "Show Test Optional Only",
        value=st.session_state.active_test_optional,
        key="widget_test_optional",
        help="Show only profiles that submitted neither an SAT nor an ACT score.",
    )

    is_stem = st.checkbox(
        "STEM Major Only",
        value=st.session_state.active_is_stem,
        key="widget_is_stem",
    )

    major_filter = st.multiselect(
        "Filter by Major:",
        options=all_majors,
        default=st.session_state.active_major_filter,
        key="widget_major_filter",
    )

    race_filter = st.multiselect(
        "Filter by Race:",
        options=all_races,
        default=st.session_state.active_race_filter,
        key="widget_race_filter",
    )

with st.sidebar.expander("🏫 Acceptance Tiers"):
    tier_filter = st.multiselect(
        "Accepted to:",
        options=["T5", "T10", "T20", "T50"],
        default=st.session_state.active_tier_filter,
        key="widget_tier_filter",
    )

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("Apply Filters", width='stretch'): # <-- FIX
        st.session_state.filters_applied = True
        st.session_state.active_gpa_range = st.session_state.widget_gpa_range
        st.session_state.active_sat_eq_range = st.session_state.widget_sat_eq_range
        st.session_state.active_is_stem = st.session_state.widget_is_stem
        st.session_state.active_test_optional = st.session_state.widget_test_optional
        st.session_state.active_major_filter = st.session_state.widget_major_filter
        st.session_state.active_race_filter = st.session_state.widget_race_filter
        st.session_state.active_tier_filter = st.session_state.widget_tier_filter
        st.session_state.selected_profile_idx = None
        st.rerun()

with col_btn2:
    if st.button("Reset", width='stretch'): # <-- FIX
        for key in [
            "filters_applied",
            "active_gpa_range",
            "active_sat_eq_range",
            "active_is_stem",
            "active_test_optional",
            "active_major_filter",
            "active_race_filter",
            "active_tier_filter",
        ]:
            st.session_state.pop(key, None)
        st.session_state.selected_profile_idx = None
        st.rerun()

# Sidebar metrics
st.sidebar.metric("Total Profiles in DB", len(df))



#  Filter Logic

def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    #Apply all active filters from session_state to the dataframe.
    filtered_df_ = df_in.copy()

    gpa_range_ = st.session_state.active_gpa_range
    sat_eq_range_ = st.session_state.active_sat_eq_range
    is_stem_ = st.session_state.active_is_stem
    is_test_optional_ = st.session_state.active_test_optional
    major_filter_ = st.session_state.active_major_filter
    race_filter_ = st.session_state.active_race_filter
    tier_filter_ = st.session_state.active_tier_filter

    # Test Optional
    if is_test_optional_:
        filtered_df_ = filtered_df_[filtered_df_["test_optional"] == True]  # noqa
    else:
        # Apply SAT filter only if not in "Test Optional Only" mode
        if "sat_equivalent" in filtered_df_.columns:
            filtered_df_ = filtered_df_[
                filtered_df_["sat_equivalent"]
                .fillna(-1) # Use -1 to drop NaNs unless range starts at min
                .between(sat_eq_range_[0], sat_eq_range_[1])
            ]


    # GPA Filter
    if "gpa_unweighted" in filtered_df_.columns:
        filtered_df_ = filtered_df_[
            filtered_df_["gpa_unweighted"]
            .fillna(0)
            .between(gpa_range_[0], gpa_range_[1])
        ]

    # STEM Filter
    if is_stem_ and "stem_major" in filtered_df_.columns:
        filtered_df_ = filtered_df_[filtered_df_["stem_major"] == True]  # noqa

    # Major Filter
    if major_filter_:
        def has_major(majors):
            if not isinstance(majors, list):
                return False
            return any(m in majors for m in major_filter_)

        filtered_df_ = filtered_df_[filtered_df_["majors"].apply(has_major)]

    # Race Filter
    if race_filter_:
        def has_race(races):
            if not isinstance(races, list):
                return False
            return any(r in races for r in race_filter_)

        filtered_df_ = filtered_df_[filtered_df_["race"].apply(has_race)]

    # Tier Filter
    if tier_filter_:
        tier_df = filtered_df_.copy()
        mask = pd.Series(False, index=tier_df.index)

        if "T5" in tier_filter_ and "t5_accepted" in tier_df.columns:
            mask |= tier_df["t5_accepted"]
        if "T10" in tier_filter_ and "t10_accepted" in tier_df.columns:
            mask |= tier_df["t10_accepted"]
        if "T20" in tier_filter_ and "t20_accepted" in tier_df.columns:
            mask |= tier_df["t20_accepted"]
        if "T50" in tier_filter_ and "t50_accepted" in tier_df.columns:
            mask |= tier_df["t50_accepted"]

        filtered_df_ = filtered_df_[mask]

    return filtered_df_


if st.session_state.get("filters_applied", False):
    filtered_df = apply_filters(df)
else:
    filtered_df = df.copy()

st.sidebar.metric("Filtered Profiles", len(filtered_df))


#  Active Filter Summary

filter_chips = []

if st.session_state.get("filters_applied", False):
    g1, g2 = st.session_state.active_gpa_range
    if (g1, g2) != (0.0, 4.0):
        filter_chips.append(f"GPA {g1:.2f}–{g2:.2f}")

    s1, s2 = st.session_state.active_sat_eq_range
    if (s1, s2) != (min_sat_eq, max_sat_eq):
        filter_chips.append(f"SAT-Eq {s1}–{s2}")

    if st.session_state.active_is_stem:
        filter_chips.append("STEM Only")

    if st.session_state.active_test_optional:
        filter_chips.append("Test Optional Only")

    if st.session_state.active_major_filter:
        filter_chips.append(f"Majors: {', '.join(st.session_state.active_major_filter)}")

    if st.session_state.active_race_filter:
        filter_chips.append(f"Race: {', '.join(st.session_state.active_race_filter)}")

    if st.session_state.active_tier_filter:
        filter_chips.append(f"Tiers: {', '.join(st.session_state.active_tier_filter)}")

if filter_chips:
    st.markdown("**Active Filters:**", help="These filters are applied across most views.")
    chips_html = "".join(
        [f'<span class="filter-chip">{chip}</span>' for chip in filter_chips]
    )
    st.markdown(chips_html, unsafe_allow_html=True)
else:
    st.info("Apply filters in the sidebar to begin exploring the dataset.")


#  Profile Matcher Model (Cached)

@st.cache_resource
def get_nn_model(_filtered_df):
    #Creates and caches the NN model and scaler based on the filtered data.
    #The _filtered_df argument ensures this function re-runs when filters change.
    
    if not SKLEARN_AVAILABLE:
        return None, None, None, None

    numeric_cols = [
        "gpa_unweighted",
        "sat_equivalent",
        "ap_classes",
        "num_ecs",
        "num_awards",
    ]
    
    # Use only profiles that have all the numeric data needed for matching
    train_df = _filtered_df.copy().dropna(subset=numeric_cols)
    
    # Need at least N+1 samples for N neighbors
    if len(train_df) <= 5: 
        return None, None, None, None

    X_train = train_df[numeric_cols].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Emphasize GPA & SAT a bit more
    feature_weights = np.array([2.0, 2.0, 1.0, 1.0, 1.0])
    X_train_scaled *= feature_weights

    n_neighbors = min(5, len(train_df))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_train_scaled)
    
    # Return the fitted model, scaler, the data used to train it, and weights
    return nn, scaler, train_df, feature_weights


#  Profile Viewer (Dialog)

def display_profile_modal(profile_idx, context: str = ""):
    #Display a full profile in a st.dialog (modal) window.
    
    if profile_idx not in filtered_df.index:
        st.warning("Selected profile is no longer in the filtered view. Please re-apply filters.")
        st.session_state.selected_profile_idx = None
        return

    profile = filtered_df.loc[profile_idx]
    profile_id = str(profile.get("profile_id", "N/A"))
    unique_suffix = f"{profile_idx}_{context}" if context else str(profile_idx)

    @st.dialog(f"Full Profile Details (ID: {profile_id[:10]}...)")
    def show_profile_dialog():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Academic Stats")
            st.write(f"**Unweighted GPA:** {profile.get('gpa_unweighted', 'N/A')}")
            st.write(f"**Weighted GPA:** {profile.get('gpa_weighted', 'N/A')}")
            st.write(f"**SAT:** {profile.get('sat', 'N/A')}")
            st.write(f"**ACT:** {profile.get('act', 'N/A')}")
            st.write(f"**SAT Equivalent:** {profile.get('sat_equivalent', 'N/A')}")
            st.write(f"**Test Optional:** {'Yes' if profile.get('test_optional', False) else 'No'}")
            st.write(f"**AP Classes:** {profile.get('ap_classes', 0)}")

        with col2:
            st.subheader("Demographics")
            majors = profile.get("majors", [])
            st.write(f"**Majors:** {', '.join(majors) if isinstance(majors, list) else majors}")
            st.write(f"**Gender:** {profile.get('gender', 'N/A')}")
            races = profile.get("race", [])
            st.write(f"**Race:** {', '.join(races) if isinstance(races, list) else races}")
            st.write(f"**STEM Major:** {'Yes' if profile.get('stem_major', False) else 'No'}")

        st.subheader("Extracurriculars")
        ecs = profile.get("extracurriculars", [])
        if isinstance(ecs, list) and ecs:
            for i, ec in enumerate(ecs, 1):
                st.write(f"{i}. {ec}")
        else:
            st.write("None listed")

        st.subheader("Awards")
        awards = profile.get("awards", [])
        if isinstance(awards, list) and awards:
            for i, award in enumerate(awards, 1):
                st.write(f"{i}. {award}")
        else:
            st.write("None listed")

        st.subheader("Acceptances")
        accepts = profile.get("acceptances", [])
        if isinstance(accepts, list) and accepts:
            for school in accepts:
                st.write(f"✅ {school}")
        else:
            st.write("None listed")

        st.subheader("Rejections")
        rejects = profile.get("rejections", [])
        if isinstance(rejects, list) and rejects:
            for school in rejects:
                st.write(f"❌ {school}")
        else:
            st.write("None listed")

        # Rating Section
        st.subheader("Rate This Profile")
        avg_rating = get_avg_rating(profile_id)
        if avg_rating is not None:
            st.write(f"**Current Average Rating:** {avg_rating:.1f}/10")

        rating_col1, rating_col2 = st.columns([3, 1])
        with rating_col1:
            rating = st.slider(
                "Impact Rating (1–10):",
                1,
                10,
                5,
                key=f"rating_{unique_suffix}",
            )
        with rating_col2:
            st.write(" ") # Align button
            if st.button("Submit Rating", key=f"submit_{unique_suffix}"):
                save_rating(profile_id, rating)
                st.success(f"Rating of {rating}/10 submitted!")
                # Button click closes dialog, rerun will show new avg
                st.rerun()

    # Call the function to open the dialog
    show_profile_dialog()

    # After the dialog closes, clear the selected_profile_idx
    # so it doesn't re-open on the next rerun
    st.session_state.selected_profile_idx = None


#  School Selection for Charts

if all_schools:
    school_options = ["All Schools (from filters)"] + all_schools
    selected_school = st.selectbox(
        "📌 Optional: Focus analytics on a specific school (overrides filters **for charts only**)",
        options=school_options,
        index=0,
        key="graph_school_filter",
        help="Choose a school to analyze all applicants to that school in the database, or use 'All Schools' to use filter-based data.",
    )

    if selected_school == "All Schools (from filters)":
        graph_df = filtered_df.copy()
        graph_title_suffix = " (Filtered)"
    else:
        # build mask once rather than iterrows
        accept_mask = df["acceptances"].apply(
            lambda xs: isinstance(xs, list) and selected_school in xs
        )
        reject_mask = df["rejections"].apply(
            lambda xs: isinstance(xs, list) and selected_school in xs
        )
        mask = accept_mask | reject_mask
        graph_df = df.loc[mask].copy()
        graph_title_suffix = f" – All Applicants to {selected_school}"
else:
    selected_school = "All Schools (from filters)"
    graph_df = filtered_df.copy()
    graph_title_suffix = " (Filtered)"

# Intro text per selection
if selected_school != "All Schools (from filters)":
    st.info(
        f"Showing data for **{len(graph_df)}** applicants who applied to **{selected_school}**."
    )
elif not st.session_state.get("filters_applied", False):
    st.info("Charts will update once you apply filters in the sidebar.")
else:
    st.info(f"Showing **{len(graph_df)}** filtered profiles in charts.")


#  Main Tabs

tab_names = [
    "📊 Overview Charts",
    "📋 Applicant Browser",
    "📈 Advanced Analytics",
    "🎯 Acceptance Patterns",
    "📊 Summary Statistics",
]
if SKLEARN_AVAILABLE:
    tab_names.insert(4, "🔍 Find Similar Profiles")

tabs = st.tabs(tab_names)

# Assign tabs dynamically
tab1 = tabs[0]
tab2 = tabs[1]
tab3 = tabs[2]
tab4 = tabs[3]
if SKLEARN_AVAILABLE:
    tab5 = tabs[4]
    tab6 = tabs[5]
else:
    tab6 = tabs[4] # tab5 (matching) is skipped


# TAB 1 – Overview Charts

with tab1:
    col1, col2 = st.columns(2)

    # GPA vs SAT Scatter
    with col1:
        st.subheader(f"GPA vs SAT Equivalent{graph_title_suffix}")

        if len(graph_df) > 0:
            if "gpa_unweighted" in graph_df.columns and "sat_equivalent" in graph_df.columns:
                plot_df = graph_df.dropna(subset=["gpa_unweighted", "sat_equivalent"]).copy()

                if selected_school != "All Schools (from filters)":
                    def get_school_status(row):
                        accepts = row.get("acceptances", [])
                        if isinstance(accepts, list) and selected_school in accepts:
                            return "Accepted"
                        return "Rejected/Waitlisted"

                    plot_df["school_status"] = plot_df.apply(get_school_status, axis=1)
                    color_col = "school_status"
                    color_map = {
                        "Accepted": "#2E8B57",
                        "Rejected/Waitlisted": "#DC143C",
                    }
                else:
                    plot_df["T20 Accepted"] = plot_df["t20_accepted"].map({True: "Yes", False: "No"})
                    color_col = "T20 Accepted"
                    color_map = {"Yes": "#2E8B57", "No": "#DC143C"}

                required_cols = [
                    "gpa_unweighted",
                    "sat_equivalent",
                    color_col,
                    "majors",
                    "num_ecs",
                    "num_awards",
                ]
                available_cols = [c for c in required_cols if c in plot_df.columns]
                plot_df = plot_df[available_cols]

                if len(plot_df) > 0:
                    fig = px.scatter(
                        plot_df,
                        x="gpa_unweighted",
                        y="sat_equivalent",
                        color=color_col,
                        hover_data=["majors", "num_ecs", "num_awards"],
                        custom_data=[plot_df.index],
                        labels={
                            "gpa_unweighted": "Unweighted GPA",
                            "sat_equivalent": "SAT Equivalent Score",
                            color_col: "Status"
                            if selected_school != "All Schools (from filters)"
                            else "T20 Accepted",
                        },
                        title=f"GPA vs SAT Equivalent{graph_title_suffix}",
                        color_discrete_map=color_map,
                    )
                    fig.update_traces(
                        marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey"))
                    )
                    fig.update_layout(clickmode="event+select")
                    
                    # 1. Capture the chart's event data
                    chart_event = st.plotly_chart(
                        fig, 
                        width='stretch', # <-- FIX
                        key="scatter_plot",
                        on_select="rerun" # Use rerun on select
                    )

                    st.caption("💡 Tip: Click a profile on the chart to view full details.")

                    # 2. Check if a point was clicked in selection
                    if chart_event.selection:
                        clicked_point = chart_event.selection.get('points')
                        if clicked_point:
                            # 3. Get the index from the clicked point's customdata
                            selected_idx_from_plot = clicked_point[0]['customdata'][0]
                            
                            # 4. Set the session state
                            if st.session_state.selected_profile_idx != selected_idx_from_plot:
                                st.session_state.selected_profile_idx = selected_idx_from_plot
                                st.rerun() # Rerun to show the modal

                else:
                    st.info(
                        "No data points available for GPA vs SAT plot (e.g., all profiles are test optional or missing GPA)."
                    )
            else:
                st.info("GPA or SAT data not available for plotting.")
        else:
            st.info("No profiles match the current filters or school selection.")

    # EC & Award Histograms
    with col2:
        st.subheader(f"Extracurricular & Award Counts{graph_title_suffix}")
        if len(graph_df) > 0:
            if "num_ecs" in graph_df.columns:
                fig_ecs = px.histogram(
                    graph_df,
                    x="num_ecs",
                    labels={
                        "num_ecs": "Number of Extracurriculars",
                        "count": "Frequency",
                    },
                    title=f"Distribution of EC Counts{graph_title_suffix}",
                )
                st.plotly_chart(fig_ecs, width='stretch') # <-- FIX

            else:
                st.info("Extracurricular data not available.")

            if "num_awards" in graph_df.columns:
                fig_awards = px.histogram(
                    graph_df,
                    x="num_awards",
                    labels={
                        "num_awards": "Number of Awards",
                        "count": "Frequency",
                    },
                    title=f"Distribution of Award Counts{graph_title_suffix}",
                )
                st.plotly_chart(fig_awards, width='stretch') # <-- FIX
            else:
                st.info("Award data not available.")
        else:
            st.info("No profiles to display.")

    # This call will now be triggered by the chart click
    if st.session_state.selected_profile_idx is not None:
        display_profile_modal(st.session_state.selected_profile_idx, context="tab1")


# TAB 2 – Applicant Browser

with tab2:
    st.subheader("Searchable Applicant Database")
    st.caption(f"Showing data for **{len(graph_df)}** profiles based on your selection.")

    if len(graph_df) > 0:
        display_cols = [
            "gpa_unweighted",
            "sat_equivalent",
            "test_optional",
            "num_ecs",
            "num_awards",
            "ap_classes",
            "stem_major",
        ]
        available_display_cols = [c for c in display_cols if c in graph_df.columns]
        
        # Start with index for later selection
        display_df = graph_df[available_display_cols].copy()
        display_df['Profile ID'] = graph_df.index # Use the original index

        if "majors" in graph_df.columns:
            display_df["Majors"] = graph_df["majors"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x)
            )
        if "race" in graph_df.columns:
            display_df["Race"] = graph_df["race"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x)
            )
        if "acceptances" in graph_df.columns:
            display_df["Acceptances"] = graph_df["acceptances"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

        search_term = st.text_input(
            "🔎 Search (GPA, SAT, Major, Race, etc.):", key="applicant_search"
        )

        if search_term:
            search_lower = search_term.lower()
            mask = pd.Series(False, index=display_df.index)

            searchable_cols = [c for c in display_df.columns if c != "Profile ID"]
            for col in searchable_cols:
                mask |= display_df[col].astype(str).str.lower().str.contains(
                    search_lower, na=False
                )
            display_df = display_df[mask]

        items_per_page = 25
        total_pages = max(1, (len(display_df) + items_per_page - 1) // items_per_page)
        
        # Ensure page number is valid
        if "current_page_browser" not in st.session_state:
            st.session_state.current_page_browser = 1
        
        page_col1, page_col2 = st.columns([3, 1])
        with page_col1:
            st.session_state.current_page_browser = st.number_input(
                "Page:", 1, total_pages, st.session_state.current_page_browser,
                key="page_number_input"
            )
        with page_col2:
            st.write(f"**Total Pages: {total_pages}**")


        current_page = st.session_state.current_page_browser
        start_idx = (current_page - 1) * items_per_page
        end_idx = start_idx + items_per_page

        st.caption(
            f"Showing {start_idx + 1}-{min(end_idx, len(display_df))} "
            f"of {len(display_df)} applicants"
        )

        page_df = display_df.iloc[start_idx:end_idx].copy()
        display_cols_final = [c for c in page_df.columns if c != "Profile ID"]
        st.dataframe(
            page_df[display_cols_final],
            width='stretch', # <-- FIX
            hide_index=True,
        )

        st.subheader("View Full Profile")
        profile_options = {}
        for idx, row in page_df.iterrows():
            profile_id = row["Profile ID"]
            gpa = row.get("gpa_unweighted", "N/A")
            sat_eq_val = row.get("sat_equivalent", "N/A")
            is_to = row.get("test_optional", False)

            gpa_str = (
                f"{gpa:.2f}"
                if isinstance(gpa, (int, float)) and not pd.isna(gpa)
                else "N/A"
            )
            sat_str = (
                f"{int(sat_eq_val)}"
                if isinstance(sat_eq_val, (int, float)) and not pd.isna(sat_eq_val)
                else "N/A"
            )
            if is_to:
                sat_str = "Test Optional"

            profile_options[profile_id] = f"GPA: {gpa_str}, Score: {sat_str} (ID: ...{str(profile_id)[-6:]})"

        if profile_options:
            
            # Find the index of the currently selected profile, if it's in the list
            current_selection = st.session_state.get("selected_profile_idx")
            options_keys = list(profile_options.keys())
            try:
                default_index = options_keys.index(current_selection) + 1
            except ValueError:
                default_index = 0 # Default to "--- Select a Profile ---"

            selected_profile_id = st.selectbox(
                "Select a profile to view full details:",
                options=[None] + options_keys,
                format_func=lambda x: "--- Select a Profile ---"
                if x is None
                else profile_options[x],
                key="table_profile_selector",
                index=default_index
            )
            
            # If selection changes, update state and rerun to open modal
            if selected_profile_id != current_selection:
                st.session_state.selected_profile_idx = selected_profile_id
                st.rerun()
    else:
        st.info("No applicants to display with current filters.")
    
    # Modal display logic for this tab
    if st.session_state.selected_profile_idx is not None:
        display_profile_modal(st.session_state.selected_profile_idx, context="tab2")


# TAB 3 – Advanced Analytics

with tab3:
    st.subheader(f"Advanced Analytics{graph_title_suffix}")

    if len(graph_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            if "gpa_unweighted" in graph_df.columns:
                st.subheader("GPA Distribution")
                fig_gpa_box = px.box(
                    graph_df,
                    y="gpa_unweighted",
                    title=f"GPA Distribution{graph_title_suffix}",
                    labels={"gpa_unweighted": "Unweighted GPA"},
                )
                st.plotly_chart(fig_gpa_box, width='stretch') # <-- FIX

            if "sat_equivalent" in graph_df.columns:
                st.subheader("SAT Equivalent Distribution")
                sat_data = graph_df[graph_df["sat_equivalent"].notna()]
                if len(sat_data) > 0:
                    fig_sat_box = px.box(
                        sat_data,
                        y="sat_equivalent",
                        title=f"SAT Equivalent Distribution{graph_title_suffix}",
                        labels={"sat_equivalent": "SAT Equivalent Score"},
                    )
                    st.plotly_chart(fig_sat_box, width='stretch') # <-- FIX
                else:
                    st.info("No profiles with SAT scores to display.")

        with col2:
            if (
                "stem_major" in graph_df.columns
                and selected_school != "All Schools (from filters)"
            ):
                st.subheader("Acceptance Rate by Major Type")

                def accepted_to_selected(row_):
                    acc = row_.get("acceptances", [])
                    return isinstance(acc, list) and selected_school in acc
                
                graph_df['accepted_target'] = graph_df.apply(accepted_to_selected, axis=1)

                type_stats = []
                for is_stem_val, group in graph_df.groupby("stem_major"):
                    if len(group) == 0:
                        continue
                    accepted_count = group['accepted_target'].sum()
                    rate = accepted_count / len(group) * 100
                    type_stats.append(
                        {
                            "Major Type": "STEM" if is_stem_val else "Non-STEM",
                            "Applicants": len(group),
                            "Acceptance Rate (%)": rate,
                        }
                    )

                if type_stats:
                    grouped = pd.DataFrame(type_stats)
                    fig_major = px.bar(
                        grouped,
                        x="Major Type",
                        y="Acceptance Rate (%)",
                        title=f"Acceptance Rate by Major Type – {selected_school}",
                        labels={
                            "Major Type": "Major Type",
                            "Acceptance Rate (%)": "Acceptance Rate (%)",
                        },
                        color="Major Type",
                        color_discrete_map={"STEM": "#4CAF50", "Non-STEM": "#2196F3"},
                    )
                    st.plotly_chart(fig_major, width='stretch') # <-- FIX
            elif selected_school != "All Schools (from filters)":
                st.info("STEM major data not available for this view.")
            else:
                st.info("Select a specific school to see acceptance rate by major type.")

        st.subheader("Correlation Matrix")
        numeric_cols = [
            "gpa_unweighted",
            "sat_equivalent",
            "ap_classes",
            "num_ecs",
            "num_awards",
        ]
        available_numeric = [c for c in numeric_cols if c in graph_df.columns]

        if len(available_numeric) > 1:
            corr_df = graph_df[available_numeric].corr()
            fig_corr = px.imshow(
                corr_df,
                labels=dict(color="Correlation"),
                title=f"Correlation Matrix{graph_title_suffix}",
                color_continuous_scale="RdBu",
                text_auto=True,
                aspect="auto",
            )
            st.plotly_chart(fig_corr, width='stretch') # <-- FIX
        else:
            st.info("Not enough numeric columns for correlation analysis.")
    else:
        st.info("No data available for advanced analytics.")


# TAB 4 – Acceptance Patterns

with tab4:
    st.subheader(f"🎯 Acceptance Patterns{graph_title_suffix}")

    if len(graph_df) > 0:
        acceptance_col = None
        analysis_target = ""

        if selected_school != "All Schools (from filters)":
            # --- Logic for specific school ---
            def accepted_to_selected(row_):
                acc = row_.get("acceptances", [])
                return isinstance(acc, list) and selected_school in acc
            
            graph_df['accepted_target'] = graph_df.apply(accepted_to_selected, axis=1)
            analysis_target = selected_school
            acceptance_col = 'accepted_target'
        
        else:
            # --- NEW LOGIC FOR FILTERED DATA ---
            st.info("Showing filtered acceptance patterns. Defaulting to T20. Use filters to analyze other tiers.")
            if 't20_accepted' in graph_df.columns:
                 acceptance_col = 't20_accepted'
                 analysis_target = "T20 Schools"
            # You could add a selectbox here to choose T5/T10/T20/T50
            
        if not acceptance_col or acceptance_col not in graph_df.columns:
            st.info("Select a specific school or apply a Tier filter (e.g., T20) to view patterns.")
        else:
            total_applicants = len(graph_df)
            total_accepted = graph_df[acceptance_col].sum()
            overall_rate = (
                (total_accepted / total_applicants) * 100 if total_applicants > 0 else 0
            )

            st.metric(
                f"Overall Acceptance Rate for {analysis_target}",
                f"{overall_rate:.1f}%",
                f"{total_accepted} / {total_applicants} accepted",
            )
            st.divider()

            col1, col2 = st.columns(2)

            # Acceptance rate by GPA range
            with col1:
                st.subheader("Acceptance Rate by GPA Range")
                gpa_ranges = [(0.0, 3.5), (3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
                gpa_summary = []

                for low, high in gpa_ranges:
                    mask = (graph_df["gpa_unweighted"] >= low) & (
                        graph_df["gpa_unweighted"] < high
                    )
                    range_df = graph_df[mask]
                    label = f"{low:.1f}–{high:.1f}"

                    if len(range_df) > 0:
                        accepted = range_df[acceptance_col].sum()
                        rate = (accepted / len(range_df)) * 100
                        gpa_summary.append(
                            {
                                "GPA Range": label,
                                "Applicants": len(range_df),
                                "Acceptance Rate (%)": rate,
                            }
                        )

                if gpa_summary:
                    gpa_summary_df = pd.DataFrame(gpa_summary)
                    fig_gpa_range = px.bar(
                        gpa_summary_df,
                        x="GPA Range",
                        y="Acceptance Rate (%)",
                        title=f"Acceptance Rate by GPA Range – {analysis_target}",
                        text="Applicants",
                        color="Acceptance Rate (%)",
                        color_continuous_scale="Greens",
                    )
                    st.plotly_chart(fig_gpa_range, width='stretch') # <-- FIX
                    st.dataframe(gpa_summary_df, hide_index=True, width='stretch') # <-- FIX
                else:
                    st.info("Not enough GPA data to compute ranges.")

            # Acceptance rate by SAT range
            with col2:
                st.subheader("Acceptance Rate by SAT Equivalent Range")
                sat_ranges = [(400, 1400), (1400, 1500), (1500, 1550), (1550, 1601)]
                sat_summary = []

                for low, high in sat_ranges:
                    mask = (
                        graph_df["sat_equivalent"].notna()
                        & (graph_df["sat_equivalent"] >= low)
                        & (graph_df["sat_equivalent"] < high)
                    )
                    range_df = graph_df[mask]
                    label = f"{low}–{high - 1}"

                    if len(range_df) > 0:
                        accepted = range_df[acceptance_col].sum()
                        rate = (accepted / len(range_df)) * 100
                        sat_summary.append(
                            {
                                "SAT-Eq Range": label,
                                "Applicants": len(range_df),
                                "Acceptance Rate (%)": rate,
                            }
                        )

                # Test optional bucket
                to_df = graph_df[graph_df["test_optional"] == True]  # noqa
                if len(to_df) > 0:
                    accepted = to_df[acceptance_col].sum()
                    rate = (accepted / len(to_df)) * 100
                    sat_summary.append(
                        {
                            "SAT-Eq Range": "Test Optional",
                            "Applicants": len(to_df),
                            "Acceptance Rate (%)": rate,
                        }
                    )

                if sat_summary:
                    sat_summary_df = pd.DataFrame(sat_summary)
                    fig_sat_range = px.bar(
                        sat_summary_df,
                        x="SAT-Eq Range",
                        y="Acceptance Rate (%)",
                        title=f"Acceptance Rate by SAT-Eq Range – {analysis_target}",
                        text="Applicants",
                        color="Acceptance Rate (%)",
                        color_continuous_scale="Blues",
                    )
                    st.plotly_chart(fig_sat_range, width='stretch') # <-- FIX
                    st.dataframe(sat_summary_df, hide_index=True, width='stretch') # <-- FIX
                else:
                    st.info("Not enough SAT data to compute ranges.")
    else:
        st.info("No profiles to display.")


# TAB 5 – Profile Matching

if SKLEARN_AVAILABLE:
    with tab5:
        st.subheader("🔍 Find Similar Profiles")
        st.caption(
            "This tool uses the **currently filtered data**. Broader filters will yield more diverse matches."
        )

        with st.expander("Enter your profile to find similar applicants", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                user_gpa = st.number_input(
                    "Your Unweighted GPA:", 0.0, 4.0, 3.5, 0.01, key="match_gpa"
                )
                user_sat_eq = st.number_input(
                    "Your SAT Equivalent Score:",
                    min_sat_eq,
                    max_sat_eq,
                    1400,
                    10,
                    key="match_sat",
                )
                user_ap = st.number_input(
                    "Number of AP Classes:", 0, 30, 0, key="match_ap"
                )

            with col2:
                user_ecs = st.number_input(
                    "Number of Extracurriculars:", 0, 50, 0, key="match_ecs"
                )
                user_awards = st.number_input(
                    "Number of Awards:", 0, 50, 0, key="match_awards"
                )
                user_stem = st.checkbox("STEM Major?", value=False, key="match_stem")
                # Note: user_stem is not currently used in the model, but could be
                # by adding a 'stem_major_numeric' column.

            if st.button("Find Similar Profiles"):
                # 1. Get the cached model
                nn, scaler, train_df, feature_weights = get_nn_model(filtered_df)

                if nn is None:
                    st.warning(
                        "Not enough data for matching (need > 5 profiles with full data). Try relaxing your filters."
                    )
                else:
                    # 2. Get user features
                    user_features = np.array(
                        [[user_gpa, user_sat_eq, user_ap, user_ecs, user_awards]]
                    )
                    
                    # 3. Transform user features
                    user_features_scaled = scaler.transform(user_features)
                    user_features_scaled *= feature_weights # Apply same weights
                    
                    # 4. Find neighbors (this is now super fast)
                    distances, indices = nn.kneighbors(user_features_scaled)

                    st.subheader("Most Similar Profiles (from filtered data):")

                    for i, (dist, idx) in enumerate(
                        zip(distances[0], indices[0]), 1
                    ):
                        # Use .iloc[idx] on the train_df returned from the model
                        similar_profile = train_df.iloc[idx]
                        profile_id = similar_profile.name # .name is the index

                        with st.container(border=True):
                            st.write(f"**Similar Profile #{i}** (Distance: {dist:.2f})")
                            st.write(
                                f"**GPA:** {similar_profile.get('gpa_unweighted', 'N/A'):.2f} | "
                                f"**SAT-Eq:** {similar_profile.get('sat_equivalent', 'N/A')} | "
                                f"**APs:** {similar_profile.get('ap_classes', 0)} | "
                                f"**ECs:** {similar_profile.get('num_ecs', 0)} | "
                                f"**Awards:** {similar_profile.get('num_awards', 0)}"
                            )
                            majors = similar_profile.get("majors", [])
                            st.write(
                                f"**Majors:** {', '.join(majors) if isinstance(majors, list) else majors}"
                            )
                            accepts = similar_profile.get("acceptances", [])
                            if isinstance(accepts, list) and accepts:
                                st.write(
                                    f"**Acceptances:** {', '.join(accepts[:5])}{'...' if len(accepts) > 5 else ''}"
                                )

                            if st.button(
                                "View Full Profile",
                                key=f"match_view_{profile_id}",
                            ):
                                st.session_state.selected_profile_idx = profile_id
                                st.rerun()
        
        # Modal display logic for this tab
        if st.session_state.selected_profile_idx is not None:
            display_profile_modal(st.session_state.selected_profile_idx, context="tab5")


# TAB 6 – Summary Statistics

with tab6:
    st.subheader("Summary Statistics (Based on Filters)")

    if st.session_state.get("filters_applied", False) and len(filtered_df) > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Profiles in DB", len(df))
            if "test_optional" in filtered_df.columns:
                to_count = int(filtered_df["test_optional"].sum())
                st.metric(
                    "Test Optional",
                    f"{to_count} ({to_count/len(filtered_df)*100:.1f}%)",
                )

        with col2:
            if "gpa_unweighted" in filtered_df.columns:
                avg_gpa = filtered_df["gpa_unweighted"].mean()
                st.metric(
                    "Average GPA",
                    f"{avg_gpa:.2f}" if not pd.isna(avg_gpa) else "N/A",
                )

            if "sat_equivalent" in filtered_df.columns:
                avg_sat = filtered_df[filtered_df["sat_equivalent"] > 0][
                    "sat_equivalent"
                ].mean()
                st.metric(
                    "Average SAT-Eq (Testers)",
                    f"{int(avg_sat)}" if not pd.isna(avg_sat) else "N/A",
                )

        with col3:
            if "num_ecs" in filtered_df.columns:
                avg_ecs = filtered_df["num_ecs"].mean()
                st.metric(
                    "Average ECs",
                    f"{avg_ecs:.1f}" if not pd.isna(avg_ecs) else "N/A",
                )

            if "num_awards" in filtered_df.columns:
                avg_awards = filtered_df["num_awards"].mean()
                st.metric(
                    "Average Awards",
                    f"{avg_awards:.1f}" if not pd.isna(avg_awards) else "N/A",
                )

        with col4:
            if "stem_major" in filtered_df.columns:
                stem_count = int(filtered_df["stem_major"].sum())
                st.metric(
                    "STEM Majors",
                    f"{stem_count} ({stem_count/len(filtered_df)*100:.1f}%)",
                )

            if "t20_accepted" in filtered_df.columns:
                t20_count = int(filtered_df["t20_accepted"].sum())
                st.metric(
                    "T20 Accepted",
                    f"{t20_count} ({t20_count/len(filtered_df)*100:.1f}%)",
                )
        
        st.divider()
        st.subheader(f"Profile Distributions (Filtered)")
        col1, col2 = st.columns(2)

        with col1:
            # Use graph_df here as it respects the school filter
            if 'majors' in graph_df.columns and len(graph_df) > 0:
                st.write("#### Top 15 Majors")
                major_counts = graph_df['majors'].explode().value_counts().nlargest(15)
                if not major_counts.empty:
                    fig_majors = px.bar(
                        major_counts, 
                        x=major_counts.index, 
                        y=major_counts.values,
                        labels={'x': 'Major', 'y': 'Count'}
                    )
                    fig_majors.update_layout(xaxis_title="Major", yaxis_title="Count")
                    st.plotly_chart(fig_majors, width='stretch') # <-- FIX
                else:
                    st.info("No major data to display for this filter.")

        with col2:
            if 'race' in graph_df.columns and len(graph_df) > 0:
                st.write("#### Race Distribution")
                race_counts = graph_df['race'].explode().value_counts()
                if not race_counts.empty:
                    fig_race = px.pie(
                        race_counts, 
                        names=race_counts.index, 
                        values=race_counts.values,
                        hole=0.3,
                        title="Race Distribution"
                    )
                    st.plotly_chart(fig_race, width='stretch') # <-- FIX
                else:
                    st.info("No race data to display for this filter.")

    elif st.session_state.get("filters_applied", False):
        st.info("No profiles match the current filter selection.")
    else:
        st.info("Apply filters to see summary statistics based on the filtered dataset.")