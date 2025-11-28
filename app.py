import warnings
import os
import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
from CollegeBase import load_data

# Suppress warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)

# Optional import for profile matching feature
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn not available. Profile matching feature will be disabled.")


# Set up

st.set_page_config(layout="wide", page_title="CollegeBase Analytics")
st.title("CollegeBase Admissions Analytics ")


#SAT/ACT Concordance Logic(ik this is super innefficient but idk please dont make fun of me)
ACT_TO_SAT_CONCORDANCE = {
    36: 1600, 35: 1570, 34: 1520, 33: 1490, 32: 1450, 31: 1420, 30: 1390,
    29: 1360, 28: 1330, 27: 1300, 26: 1260, 25: 1230, 24: 1200, 23: 1160,
    22: 1130, 21: 1100, 20: 1060, 19: 1020, 18: 980,  17: 940,  16: 900,
    15: 860,  14: 810,  13: 770,  12: 720,  11: 670,  10: 620,  9: 560
}
# Create a reverse mapping (SAT score to ACT score)
SAT_TO_ACT_EXACT = {v: k for k, v in ACT_TO_SAT_CONCORDANCE.items()}
# Create a sorted list of the official SAT scores for finding the closest match
SAT_CONCORDANCE_POINTS = sorted(SAT_TO_ACT_EXACT.keys())

def get_act_equivalent(sat_score):
    "Finds the nearest ACT equivalent for a given SAT score."
    if sat_score in SAT_TO_ACT_EXACT:
        return SAT_TO_ACT_EXACT[sat_score]
    
    # Find the closest SAT score in the official table
    closest_sat = min(SAT_CONCORDANCE_POINTS, key=lambda x: abs(x - sat_score))
    return SAT_TO_ACT_EXACT[closest_sat]
#End of Sat to Act logic 


# RATING SYSTEM (Session State & File Storage)

RATINGS_FILE = "profile_ratings.json"

def load_ratings():
    "Load ratings from file."
    if os.path.exists(RATINGS_FILE):
        try:
            with open(RATINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_rating(profile_id, rating):
    "Save a rating for a profile."
    ratings = load_ratings()
    if profile_id not in ratings:
        ratings[profile_id] = []
    ratings[profile_id].append(rating)
    with open(RATINGS_FILE, 'w') as f:
        json.dump(ratings, f)

def get_avg_rating(profile_id):
    "Get average rating for a profile."
    ratings = load_ratings()
    if profile_id in ratings and ratings[profile_id]:
        return np.mean(ratings[profile_id])
    return None

#  DATA LOADING (CACHED)
@st.cache_data
def get_data():
    "Load and cache the profile data."
    try:
        df = load_data("profiles.jsonl")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = get_data()

if df.empty:
    st.error("No data found in profiles.jsonl. Please add profiles first.")
    st.stop()

#Extract options for filters from the raw data 
# .dropna().astype(str) to prevent sorting mixed types (str and float/NaN)
all_majors = sorted(list(df['majors'].explode().dropna().astype(str).unique()))
all_races = sorted(list(df['race'].explode().dropna().astype(str).unique()))
acceptances_set = set(df['acceptances'].explode().dropna().astype(str).unique())
rejections_set = set(df['rejections'].explode().dropna().astype(str).unique())
all_schools = sorted(list(acceptances_set | rejections_set))


min_sat_eq = 400
max_sat_eq = 1600


# For Apply Filters button

if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False
    
    # Initialize all active filters to their default values
    st.session_state.active_gpa_range = (0.0, 4.0)
    st.session_state.active_sat_eq_range = (min_sat_eq, max_sat_eq)
    st.session_state.active_is_stem = False
    st.session_state.active_test_optional = False
    st.session_state.active_major_filter = []
    st.session_state.active_race_filter = []
    st.session_state.active_tier_filter = []

if 'selected_profile_idx' not in st.session_state:
    st.session_state.selected_profile_idx = None

# Sidebar Filters

st.sidebar.header("Applicant Filters")
st.sidebar.caption("Set your filters below and click 'Apply Filters' to update the dashboard.")

# Filter Widgets (key system to store current value)

gpa_range = st.sidebar.slider(
    "Unweighted GPA Range",
    min_value=0.0, max_value=4.0,
    value=st.session_state.active_gpa_range,
    step=0.01,
    key="widget_gpa_range"
)

sat_eq_range = st.sidebar.slider(
    "SAT Equivalent Score Range",
    min_value=min_sat_eq, max_value=max_sat_eq,
    value=st.session_state.active_sat_eq_range,
    step=10,
    key="widget_sat_eq_range"
)

#  Display ACT Equivalents 
min_act_eq = get_act_equivalent(sat_eq_range[0])
max_act_eq = get_act_equivalent(sat_eq_range[1])
st.sidebar.caption(f"Selected Range: SAT {sat_eq_range[0]}-{sat_eq_range[1]} (ACT ~{min_act_eq}-{max_act_eq})")


#  Test Optional Filter 
is_test_optional = st.sidebar.checkbox(
    "Show Test Optional Only", 
    value=st.session_state.active_test_optional,
    key="widget_test_optional",
    help="Show only profiles that submitted neither an SAT nor an ACT score."
)

is_stem = st.sidebar.checkbox(
    "STEM Major Only", 
    value=st.session_state.active_is_stem,
    key="widget_is_stem"
)

major_filter = st.sidebar.multiselect(
    "Filter by Major:",
    options=all_majors,
    default=st.session_state.active_major_filter,
    key="widget_major_filter"
)

race_filter = st.sidebar.multiselect(
    "Filter by Race:",
    options=all_races,
    default=st.session_state.active_race_filter,
    key="widget_race_filter"
)

tier_filter = st.sidebar.multiselect(
    "Accepted to:",
    options=["T5", "T10", "T20", "T50"],
    default=st.session_state.active_tier_filter,
    key="widget_tier_filter"
)

#  Apply Filters Button 
if st.sidebar.button("Apply Filters"):
    # Update the active session state filters from the widget keys
    st.session_state.filters_applied = True
    st.session_state.active_gpa_range = st.session_state.widget_gpa_range
    st.session_state.active_sat_eq_range = st.session_state.widget_sat_eq_range
    st.session_state.active_is_stem = st.session_state.widget_is_stem
    st.session_state.active_test_optional = st.session_state.widget_test_optional
    st.session_state.active_major_filter = st.session_state.widget_major_filter
    st.session_state.active_race_filter = st.session_state.widget_race_filter
    st.session_state.active_tier_filter = st.session_state.widget_tier_filter
    
    # Clear selected profile when filters change
    st.session_state.selected_profile_idx = None
    st.rerun()

# Filter Logic

def apply_filters(df):
    "Apply all active filters from st.session_state to the dataframe."
    filtered_df = df.copy()
    
    # Read filters from session state
    gpa_range = st.session_state.active_gpa_range
    sat_eq_range = st.session_state.active_sat_eq_range
    is_stem = st.session_state.active_is_stem
    is_test_optional = st.session_state.active_test_optional
    major_filter = st.session_state.active_major_filter
    race_filter = st.session_state.active_race_filter
    tier_filter = st.session_state.active_tier_filter

    # Test Optional Filter 
    if is_test_optional:
        filtered_df = filtered_df[filtered_df['test_optional'] == True]
    else:
        # Only apply SAT filter if NOT filtering for test optional
        if 'sat_equivalent' in filtered_df.columns:
            # fillna with 0 to handle test-optional folks, but the slider range
            # will typically exclude them unless the user is specifically looking for them.
            # A user looking for 1400-1600 will naturally exclude TO (0).
            filtered_df = filtered_df[
                filtered_df['sat_equivalent'].fillna(0).between(sat_eq_range[0], sat_eq_range[1])
            ]

    # GPA Filter
    if 'gpa_unweighted' in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df['gpa_unweighted'].fillna(0).between(gpa_range[0], gpa_range[1])
        ]
    
    # STEM Filter
    if is_stem and 'stem_major' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['stem_major'] == True]
    
    # Major Filter
    if major_filter:
        def has_major(majors):
            if not isinstance(majors, list): return False
            return any(major in majors for major in major_filter)
        filtered_df = filtered_df[filtered_df['majors'].apply(has_major)]
    
    # Race Filter
    if race_filter:
        def has_race(races):
            if not isinstance(races, list): return False
            return any(race in races for race in race_filter)
        filtered_df = filtered_df[filtered_df['race'].apply(has_race)]
    
    # Tier Filter
    if tier_filter:
        # copy to avoid SettingWithCopyWarning
        tier_df = filtered_df.copy()
        mask = pd.Series(False, index=tier_df.index)
        if "T5" in tier_filter and 't5_accepted' in tier_df.columns:
            mask |= tier_df['t5_accepted']
        if "T10" in tier_filter and 't10_accepted' in tier_df.columns:
            mask |= tier_df['t10_accepted']
        if "T20" in tier_filter and 't20_accepted' in tier_df.columns:
            mask |= tier_df['t20_accepted']
        if "T50" in tier_filter and 't50_accepted' in tier_df.columns:
            mask |= tier_df['t50_accepted']
        filtered_df = filtered_df[mask]
    
    return filtered_df

# Apply filters based on session state (if filters applied, apply filters,Ik crazy)  
if st.session_state.filters_applied:
    filtered_df = apply_filters(df)
else:
    # On first load, show all data
    filtered_df = df.copy()

# Display total profiles count
st.sidebar.metric("Total Profiles in DB", len(df))
st.sidebar.metric("Filtered Profiles", len(filtered_df))

if not st.session_state.filters_applied:
    st.info("Set filters in the sidebar and click 'Apply Filters' to begin.")

# Profile Viewer Modal

def display_profile_modal(profile_idx, context=""):
    "Display a full profile in an expandable section."
    if profile_idx is None:
        return
    
    # profile_idx is the original index
    if profile_idx not in filtered_df.index:
        st.warning("Selected profile is no longer in the filtered view. Please re-apply filters.")
        st.session_state.selected_profile_idx = None
        return
    
    profile = filtered_df.loc[profile_idx]
    profile_id = str(profile.get('profile_id', 'N/A'))
    
    unique_suffix = f"{profile_idx}_{context}" if context else str(profile_idx)
    
    with st.expander(f"📋 Full Profile Details (ID: {profile_id[:10]}...)", expanded=True):
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
            majors = profile.get('majors', [])
            st.write(f"**Majors:** {', '.join(majors) if isinstance(majors, list) else majors}")
            st.write(f"**Gender:** {profile.get('gender', 'N/A')}")
            races = profile.get('race', [])
            st.write(f"**Race:** {', '.join(races) if isinstance(races, list) else races}")
            st.write(f"**STEM Major:** {'Yes' if profile.get('stem_major', False) else 'No'}")
        
        st.subheader("Extracurriculars")
        ecs = profile.get('extracurriculars', [])
        if isinstance(ecs, list) and ecs:
            for i, ec in enumerate(ecs, 1): st.write(f"{i}. {ec}")
        else: st.write("None listed")
        
        st.subheader("Awards")
        awards = profile.get('awards', [])
        if isinstance(awards, list) and awards:
            for i, award in enumerate(awards, 1): st.write(f"{i}. {award}")
        else: st.write("None listed")
        
        st.subheader("Acceptances")
        accepts = profile.get('acceptances', [])
        if isinstance(accepts, list) and accepts:
            for school in accepts: st.write(f"✅ {school}")
        else: st.write("None listed")
        
        st.subheader("Rejections")
        rejects = profile.get('rejections', [])
        if isinstance(rejects, list) and rejects:
            for school in rejects: st.write(f"❌ {school}")
        else: st.write("None listed")
        
        # Rating Section
        st.subheader("Rate This Profile")
        avg_rating = get_avg_rating(profile_id)
        if avg_rating:
            st.write(f"**Current Average Rating:** {avg_rating:.1f}/10")
        
        rating_col1, rating_col2 = st.columns([3, 1])
        with rating_col1:
            rating = st.slider("Impact Rating (1-10):", 1, 10, 5, key=f"rating_{unique_suffix}")
        with rating_col2:
            if st.button("Submit Rating", key=f"submit_{unique_suffix}"):
                save_rating(profile_id, rating)
                st.success(f"Rating of {rating}/10 submitted!")
                st.rerun()


# School Selection (for filtering graphs)

if all_schools:
    school_options = ["All Schools (from filter)"] + all_schools
    selected_school = st.selectbox(
        "**Optional:** View stats for a specific school (this overrides sidebar filters for charts)",
        options=school_options,
        index=0,
        key="graph_school_filter",
        help="Select a school to analyze all applicants in the DB, or select 'All Schools' to use the sidebar filters."
    )
    
    # Filter data based on selected school
    if selected_school == "All Schools (from filter)":
        graph_df = filtered_df.copy()
        graph_title_suffix = " (Filtered)"
    else:
        # If a specific school is chosen, IGNORE sidebar filters and pull from main DF
        school_indices = []
        for idx, row in df.iterrows(): # Note: using main 'df'
            acceptances = row.get('acceptances', [])
            rejections = row.get('rejections', [])
            
            if isinstance(acceptances, list) and selected_school in acceptances:
                school_indices.append(idx)
            elif isinstance(rejections, list) and selected_school in rejections:
                school_indices.append(idx)
        
        graph_df = df.loc[school_indices].copy() if school_indices else pd.DataFrame()
        graph_title_suffix = f" - All Applicants to {selected_school}"
else:
    graph_df = filtered_df.copy()
    graph_title_suffix = " (Filtered)"
    selected_school = "All Schools (from filter)"

# Main Page: Data Visualizations With Tabs

st.header("Data Visualizations")

if selected_school and selected_school != "All Schools (from filter)":
    st.info(f"Showing data for all **{len(graph_df)}** applicants to **{selected_school}** in the database.")
elif not st.session_state.filters_applied:
    st.info("Charts will appear here after you apply filters.")
else:
    st.info(f"Showing **{len(graph_df)}** filtered profiles.")

# Added tab5 and tab6 
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview Charts", 
    "📋 Applicant Browser", 
    "📈 Advanced Analytics", 
    "🎯 Acceptance Patterns", 
    "🔍 Find Similar Profiles", 
    "📊 Summary Statistics"
])

with tab1:
    col1, col2 = st.columns(2)
    
    # GPA vs. SAT Scatter Plot with Clickable Dots
    with col1:
        st.subheader(f"GPA vs. SAT Equivalent{graph_title_suffix}")
        
        if len(graph_df) > 0:
            # Use sat_equivalent 
            if 'gpa_unweighted' in graph_df.columns and 'sat_equivalent' in graph_df.columns:
                
                # Add acceptance status for the selected school (if selected school is not all schools, add acceptance status)
                if selected_school and selected_school != "All Schools (from filter)":
                    def get_school_status(row):
                        acceptances = row.get('acceptances', [])
                        if isinstance(acceptances, list) and selected_school in acceptances:
                            return 'Accepted'
                        return 'Rejected/Waitlisted' # Assume all others not accepted were rejected/WL
                    
                    graph_df['school_status'] = graph_df.apply(get_school_status, axis=1)
                    color_col = 'school_status'
                    color_map = {'Accepted': '#2E8B57', 'Rejected/Waitlisted': '#DC143C'}
                else:
                    color_col = 't20_accepted'
                    color_map = {True: '#2E8B57', False: '#DC143C'}
                
                required_cols = ['gpa_unweighted', 'sat_equivalent', color_col, 'majors', 'num_ecs', 'num_awards', 'profile_id']
                available_cols = [col for col in required_cols if col in graph_df.columns]
                plot_df = graph_df[available_cols].copy()
                # Dropna on sat_equivalent 
                plot_df = plot_df.dropna(subset=['gpa_unweighted', 'sat_equivalent'])
            else:
                plot_df = pd.DataFrame()
            
            if len(plot_df) > 0:
                # Create interactive scatter plot (plotly)
                fig = px.scatter(
                    plot_df,
                    x='gpa_unweighted',
                    y='sat_equivalent',     
                    color=color_col,
                    hover_data=['majors', 'num_ecs', 'num_awards'],
                    custom_data=[plot_df.index],  # Store original index for selection (for selection)  
                    labels={
                        'gpa_unweighted': 'Unweighted GPA',
                            'sat_equivalent': 'SAT Equivalent Score',   
                        color_col: 'Status' if selected_school and selected_school != "All Schools (from filter)" else 'T20 Accepted'
                    },
                    title=f"GPA vs. SAT Equivalent{graph_title_suffix}",
                    color_discrete_map=color_map
                )
                
                fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                fig.update_layout(clickmode='event+select')
                st.plotly_chart(fig, use_container_width=True, key="scatter_plot")
                
                st.caption("💡 Tip: Select a profile from the dropdown below to view full details")
                
                # Create profile selector
                profile_options = {idx: f"GPA: {plot_df.loc[idx, 'gpa_unweighted']:.2f}, SAT-Eq: {int(plot_df.loc[idx, 'sat_equivalent'])}, ECs: {plot_df.loc[idx, 'num_ecs']}, Awards: {plot_df.loc[idx, 'num_awards']}" 
                                  for idx in plot_df.index}
                
                if profile_options:
                    # Use a placeholder to allow clearing selection
                    options_list = [None] + list(profile_options.keys())
                    
                    selected_label = st.selectbox(
                        "Select a profile to view:",
                        options=options_list,
                        format_func=lambda x: "--- Select a Profile ---" if x is None else profile_options[x],
                        key="profile_selector_tab1"
                    )
                    
                    if selected_label is not None:
                        st.session_state.selected_profile_idx = selected_label
                    else:
                        st.session_state.selected_profile_idx = None
            else:
                st.info("No data points available for GPA vs. SAT plot (e.g., all profiles are Test Optional or missing GPA).")
        else:
            st.info("No profiles match the current filters or school selection.")

    # EC & Award Histograms
    with col2:
        st.subheader(f"Extracurricular & Award Counts{graph_title_suffix}")
        
        if len(graph_df) > 0:
            if 'num_ecs' in graph_df.columns:
                # Removed nbins=15 
                fig_ecs = px.histogram(
                    graph_df, x='num_ecs',
                    labels={'num_ecs': 'Number of Extracurriculars', 'count': 'Frequency'},
                    title=f"Distribution of EC Counts{graph_title_suffix}"
                )
                st.plotly_chart(fig_ecs, use_container_width=True)
            else: st.info("Extracurricular data not available.")
            
            if 'num_awards' in graph_df.columns:
                # Removed nbins=15 
                fig_awards = px.histogram(
                    graph_df, x='num_awards',
                    labels={'num_awards': 'Number of Awards', 'count': 'Frequency'},
                    title=f"Distribution of Award Counts{graph_title_suffix}"
                )
                st.plotly_chart(fig_awards, use_container_width=True)
            else: st.info("Award data not available.")
        else:
            st.info("No profiles to display.")

    # Display selected profile (if one was chosen from dropdown)
    if st.session_state.selected_profile_idx is not None:
        display_profile_modal(st.session_state.selected_profile_idx, context="tab1")

with tab2:
    st.subheader("Searchable Applicant Database")
    st.caption(f"Showing data for **{len(graph_df)}** profiles based on your selection.")
    
    if len(graph_df) > 0:
        # Prepare display dataframe
        display_cols = ['gpa_unweighted', 'sat_equivalent', 'test_optional', 'num_ecs', 'num_awards', 'ap_classes', 'stem_major']
        available_display_cols = [col for col in display_cols if col in graph_df.columns]
        
        display_df = graph_df[available_display_cols].copy()
        
        if 'majors' in graph_df.columns:
            display_df['Majors'] = graph_df['majors'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        if 'race' in graph_df.columns:
            display_df['Race'] = graph_df['race'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        if 'acceptances' in graph_df.columns:
            display_df['Acceptances'] = graph_df['acceptances'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        display_df['Profile ID'] = graph_df.index
        
        # Search functionality
        search_term = st.text_input("🔎 Search (GPA, SAT, Major, etc.):", key="applicant_search")
        
        if search_term:
            search_lower = search_term.lower()
            mask = pd.Series([False] * len(display_df), index=display_df.index)
            
            for col in display_df.columns:
                if col != 'Profile ID':
                    mask |= display_df[col].astype(str).str.lower().str.contains(search_lower, na=False)
            
            display_df = display_df[mask]
        
        # Pagination
        items_per_page = 25
        total_pages = max(1, (len(display_df) + items_per_page - 1) // items_per_page)
        current_page = st.number_input("Page:", 1, total_pages, 1, key="current_page")
        
        start_idx = (current_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(display_df))} of {len(display_df)} applicants")
        
        page_df = display_df.iloc[start_idx:end_idx].copy()
        display_cols_final = [col for col in page_df.columns if col != 'Profile ID']
        st.dataframe(page_df[display_cols_final], use_container_width=True, hide_index=True)
        
        # Profile selection
        st.subheader("View Full Profile")
        profile_options = {}
        for idx in page_df.index:
            profile_id = page_df.loc[idx, 'Profile ID']
            gpa = page_df.loc[idx, 'gpa_unweighted'] if 'gpa_unweighted' in page_df.columns else 'N/A'
            sat_eq = page_df.loc[idx, 'sat_equivalent'] if 'sat_equivalent' in page_df.columns else 'N/A'
            is_to = page_df.loc[idx, 'test_optional'] if 'test_optional' in page_df.columns else False
            
            gpa_str = f"{gpa:.2f}" if isinstance(gpa, (int, float)) and not pd.isna(gpa) else "N/A"
            sat_str = f"{int(sat_eq)}" if isinstance(sat_eq, (int, float)) and not pd.isna(sat_eq) else "N/A"
            if is_to: sat_str = "Test Optional"
            
            profile_options[profile_id] = f"GPA: {gpa_str}, Score: {sat_str}"
        
        if profile_options:
            selected_profile_id = st.selectbox(
                "Select a profile to view full details:",
                options=[None] + list(profile_options.keys()),
                format_func=lambda x: "--- Select a Profile ---" if x is None else profile_options[x],
                key="table_profile_selector"
            )
            
            if selected_profile_id is not None:
                st.session_state.selected_profile_idx = selected_profile_id
                display_profile_modal(selected_profile_id, context="tab2")
    else:
        st.info("No applicants to display with current filters.")

with tab3:
    st.subheader(f"Advanced Analytics{graph_title_suffix}")
    
    if len(graph_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'gpa_unweighted' in graph_df.columns:
                st.subheader("GPA Distribution")
                fig_gpa_box = px.box(graph_df, y='gpa_unweighted', title=f"GPA Distribution{graph_title_suffix}", labels={'gpa_unweighted': 'Unweighted GPA'})
                st.plotly_chart(fig_gpa_box, use_container_width=True)
            
            if 'sat_equivalent' in graph_df.columns:
                st.subheader("SAT Equivalent Distribution")
                sat_data = graph_df[graph_df['sat_equivalent'].notna()]
                if len(sat_data) > 0:
                    fig_sat_box = px.box(sat_data, y='sat_equivalent', title=f"SAT Equivalent Distribution{graph_title_suffix}", labels={'sat_equivalent': 'SAT Equivalent Score'})
                    st.plotly_chart(fig_sat_box, use_container_width=True)
        
        with col2:
            if 'stem_major' in graph_df.columns and selected_school and selected_school != "All Schools (from filter)":
                st.subheader("Acceptance Rate by Major Type")
                # Simplified logic for demonstration
                grouped = graph_df.groupby('stem_major').apply(lambda x: pd.Series({
                    'count': len(x),
                    'accepted': sum(1 for _, row in x.iterrows() if isinstance(row.get('acceptances', []), list) and selected_school in row.get('acceptances', []))
                })).reset_index()
                grouped['Acceptance Rate (%)'] = (grouped['accepted'] / grouped['count']) * 100
                grouped['stem_major'] = grouped['stem_major'].map({True: 'STEM', False: 'Non-STEM'})
                
                fig_major = px.bar(
                    grouped, x='stem_major', y='Acceptance Rate (%)',
                    title=f"Acceptance Rate by Major Type - {selected_school}",
                    labels={'stem_major': 'Major Type', 'y': 'Acceptance Rate (%)'},
                    color='stem_major', color_discrete_map={'STEM': '#4CAF50', 'Non-STEM': '#2196F3'}
                )
                st.plotly_chart(fig_major, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Matrix")
        numeric_cols = ['gpa_unweighted', 'sat_equivalent', 'ap_classes', 'num_ecs', 'num_awards']
        available_numeric = [col for col in numeric_cols if col in graph_df.columns]
        
        if len(available_numeric) > 1:
            corr_df = graph_df[available_numeric].corr()
            fig_corr = px.imshow(
                corr_df, labels=dict(color="Correlation"),
                title=f"Correlation Matrix{graph_title_suffix}",
                color_continuous_scale='RdBu', text_auto=True, aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No data available for advanced analytics.")

# Improved UI for tab4 
with tab4:
    st.subheader(f"🎯 Acceptance Patterns{graph_title_suffix}")
    
    if len(graph_df) > 0 and selected_school and selected_school != "All Schools (from filter)":
        
        # Overall Acceptance Rate Metric 
        total_applicants = len(graph_df)
        total_accepted = sum(1 for _, row in graph_df.iterrows() if isinstance(row.get('acceptances', []), list) and selected_school in row.get('acceptances', []))
        overall_rate = (total_accepted / total_applicants) * 100 if total_applicants > 0 else 0
        
        st.metric(
            f"Overall Acceptance Rate for {selected_school}",
            f"{overall_rate:.1f}%",
            f"({total_accepted} accepted out of {total_applicants} applicants)"
        )
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            # Acceptance rate by GPA range
            st.subheader("Acceptance Rate by GPA Range")
            gpa_ranges = [(0.0, 3.5), (3.5, 3.7), (3.7, 3.9), (3.9, 4.1)] # Extended to 4.1 to include 4.0
            gpa_summary = []
            
            for low, high in gpa_ranges:
                mask = (graph_df['gpa_unweighted'] >= low) & (graph_df['gpa_unweighted'] < high)
                range_df = graph_df[mask]
                label = f"{low:.1f}-{high:.1f}"
                
                if len(range_df) > 0:
                    accepted = sum(1 for _, row in range_df.iterrows() if isinstance(row.get('acceptances', []), list) and selected_school in row.get('acceptances', []))
                    rate = (accepted / len(range_df)) * 100
                    gpa_summary.append({"GPA Range": label, "Applicants": len(range_df), "Acceptance Rate (%)": rate})
            
            if gpa_summary:
                gpa_summary_df = pd.DataFrame(gpa_summary)
                fig_gpa_range = px.bar(
                    gpa_summary_df, x="GPA Range", y="Acceptance Rate (%)",
                    title=f"Acceptance Rate by GPA Range - {selected_school}",
                    text="Applicants", color="Acceptance Rate (%)", color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_gpa_range, use_container_width=True)
                st.dataframe(gpa_summary_df, hide_index=True, use_container_width=True)

        with col2:
            # Acceptance rate by SAT range
            st.subheader("Acceptance Rate by SAT Equivalent Range")
            sat_ranges = [(400, 1400), (1400, 1500), (1500, 1550), (1550, 1601)] # 1601 to include 1600
            sat_summary = []
            
            for low, high in sat_ranges:
                mask = (graph_df['sat_equivalent'].notna()) & (graph_df['sat_equivalent'] >= low) & (graph_df['sat_equivalent'] < high)
                range_df = graph_df[mask]
                label = f"{low}-{high-1}"
                
                if len(range_df) > 0:
                    accepted = sum(1 for _, row in range_df.iterrows() if isinstance(row.get('acceptances', []), list) and selected_school in row.get('acceptances', []))
                    rate = (accepted / len(range_df)) * 100
                    sat_summary.append({"SAT-Eq Range": label, "Applicants": len(range_df), "Acceptance Rate (%)": rate})

            # Add Test Optional category
            to_df = graph_df[graph_df['test_optional'] == True]
            if len(to_df) > 0:
                accepted = sum(1 for _, row in to_df.iterrows() if isinstance(row.get('acceptances', []), list) and selected_school in row.get('acceptances', []))
                rate = (accepted / len(to_df)) * 100
                sat_summary.append({"SAT-Eq Range": "Test Optional", "Applicants": len(to_df), "Acceptance Rate (%)": rate})

            if sat_summary:
                sat_summary_df = pd.DataFrame(sat_summary)
                fig_sat_range = px.bar(
                    sat_summary_df, x="SAT-Eq Range", y="Acceptance Rate (%)",
                    title=f"Acceptance Rate by SAT-Eq Range - {selected_school}",
                    text="Applicants", color="Acceptance Rate (%)", color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_sat_range, use_container_width=True)
                st.dataframe(sat_summary_df, hide_index=True, use_container_width=True)
    else:
        st.info("Select a specific school (not 'All Schools') to view acceptance patterns.")


# tab5 for Profile Matching 
with tab5:
    st.subheader("🔍 Find Similar Profiles")
    st.caption("This tool uses the currently filtered data. Broader filters will yield more diverse matches.")

    with st.expander("Enter your profile to find similar applicants", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            user_gpa = st.number_input("Your Unweighted GPA:", 0.0, 4.0, 3.5, 0.01, key="match_gpa")
            user_sat_eq = st.number_input("Your SAT Equivalent Score:", min_sat_eq, max_sat_eq, 1400, 10, key="match_sat")
            user_ap = st.number_input("Number of AP Classes:", 0, 30, 0, key="match_ap")
        
        with col2:
            user_ecs = st.number_input("Number of Extracurriculars:", 0, 50, 0, key="match_ecs")
            user_awards = st.number_input("Number of Awards:", 0, 50, 0, key="match_awards")
            user_stem = st.checkbox("STEM Major?", value=False, key="match_stem")
        
        if st.button("Find Similar Profiles"):
            if not SKLEARN_AVAILABLE:
                st.error("Profile matching requires scikit-learn. Please install it with: pip install scikit-learn")
            else:
                # Use sat_equivalent 
                numeric_cols = ['gpa_unweighted', 'sat_equivalent', 'ap_classes', 'num_ecs', 'num_awards']
                
                # Create feature vector
                user_features = np.array([[
                    user_gpa,
                    user_sat_eq,
                    user_ap,
                    user_ecs,
                    user_awards
                ]])
                
                # Prepare training data from *filtered* df (filtered df is the df with the filters applied)
                train_df = filtered_df.copy()
                # Dropna on sat_equivalent 
                train_df = train_df.dropna(subset=numeric_cols)
                
                if len(train_df) > 5: # Need enough profiles to match
                    X_train = train_df[numeric_cols].values
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    user_features_scaled = scaler.transform(user_features)
                    
                    n_neighbors = min(5, len(train_df))
                    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
                    nn.fit(X_train_scaled)
                    distances, indices = nn.kneighbors(user_features_scaled)
                    
                    st.subheader("Most Similar Profiles (from filtered data):")
                    
                    for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
                        similar_profile = train_df.iloc[idx]
                        profile_id = similar_profile.get('profile_id')
                        with st.container(border=True):
                            st.write(f"**Similar Profile #{i}** (Distance: {dist:.2f})")
                            st.write(f"**GPA:** {similar_profile.get('gpa_unweighted', 'N/A'):.2f} | **SAT-Eq:** {similar_profile.get('sat_equivalent', 'N/A')} | **APs:** {similar_profile.get('ap_classes', 0)} | **ECs:** {similar_profile.get('num_ecs', 0)} | **Awards:** {similar_profile.get('num_awards', 0)}")
                            majors = similar_profile.get('majors', [])
                            st.write(f"**Majors:** {', '.join(majors) if isinstance(majors, list) else majors}")
                            accepts = similar_profile.get('acceptances', [])
                            if isinstance(accepts, list) and accepts:
                                st.write(f"**Acceptances:** {', '.join(accepts[:5])}{'...' if len(accepts) > 5 else ''}")
                            
                            # Add a button to view the full profile
                            if st.button("View Full Profile", key=f"match_view_{profile_id}"):
                                st.session_state.selected_profile_idx = profile_id
                                # This re-run isn't ideal, but it's the simplest way
                                # to get the modal to appear outside the expander
                                st.rerun() 
                else:
                    st.warning("Not enough data for matching. Please adjust filters.")

# tab6 for Summary Statistics 
with tab6:
    st.subheader("Summary Statistics (Based on Filters)")

    if st.session_state.filters_applied and len(filtered_df) > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Filtered Profiles", len(filtered_df))
            if 'test_optional' in filtered_df.columns:
                to_count = filtered_df['test_optional'].sum()
                st.metric("Test Optional", f"{to_count} ({to_count/len(filtered_df)*100:.1f}%)")

        with col2:
            if 'gpa_unweighted' in filtered_df.columns:
                avg_gpa = filtered_df['gpa_unweighted'].mean()
                st.metric("Average GPA", f"{avg_gpa:.2f}" if not pd.isna(avg_gpa) else "N/A")
            
            if 'sat_equivalent' in filtered_df.columns:
                # Exclude TO (which are 0) from avg calc
                avg_sat = filtered_df[filtered_df['sat_equivalent'] > 0]['sat_equivalent'].mean()
                st.metric("Average SAT-Eq (testers)", f"{int(avg_sat)}" if not pd.isna(avg_sat) else "N/A")

        with col3:
            if 'num_ecs' in filtered_df.columns:
                avg_ecs = filtered_df['num_ecs'].mean()
                st.metric("Average ECs", f"{avg_ecs:.1f}" if not pd.isna(avg_ecs) else "N/A")
            
            if 'num_awards' in filtered_df.columns:
                avg_awards = filtered_df['num_awards'].mean()
                st.metric("Average Awards", f"{avg_awards:.1f}" if not pd.isna(avg_awards) else "N/A")

        with col4:
            if 'stem_major' in filtered_df.columns:
                stem_count = filtered_df['stem_major'].sum()
                st.metric("STEM Majors", f"{stem_count} ({stem_count/len(filtered_df)*100:.1f}%)")
            
            if 't20_accepted' in filtered_df.columns:
                t20_count = filtered_df['t20_accepted'].sum()
                st.metric("T20 Accepted", f"{t20_count} ({t20_count/len(filtered_df)*100:.1f}%)")
    elif st.session_state.filters_applied:
        st.info("No profiles match the current filter selection.")
    else:
        st.info("Apply filters to see summary statistics.")