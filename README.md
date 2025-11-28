Admissions Insights Dashboard

This project is a web-based, interactive dashboard for analyzing a dataset of real college applicant profiles. It is built using Streamlit for the frontend and Pandas for data manipulation.

The application loads applicant data from profiles.jsonl, processes and augments it, and then presents it in a multi-tab dashboard. This allows users to filter, visualize, and find patterns in college admissions results.

How This Dashboard Can Help You as an Applicant

This tool is designed to help you move from broad national statistics to specific, actionable data. Here are detailed ways you can use this dashboard to your advantage.

1. Build a Smarter, More Balanced School List

The biggest challenge for an applicant is knowing what is a "Reach," "Target," or "Safety" school. This tool helps you find out.

How to use:

Go to the Overview Charts tab.

In the "Focus analytics on a specific school" dropdown, select a college you're interested in (e.g., "Cornell University").

Look at the "GPA vs SAT Equivalent" scatter plot. You will instantly see all the applicants in the database who applied to that school.

Pay attention to the green "Accepted" dots versus the red "Rejected/Waitlisted" dots.

Ask yourself: Where do your own GPA and SAT scores fall? Are you comfortably in the middle of the green cluster (a good Target)? Are you on the lower edge or below it (a Reach)? Are you well above the entire cluster (a Safety)?

2. Decide on Your Test-Optional Strategy

This is one of the most confusing parts of applying, and this dashboard can provide data-driven answers.

How to use:

Go to the Acceptance Patterns tab.

Select a school you want to apply to.

Look at the "Acceptance Rate by SAT Equivalent Range" chart. You will see several bars.

Find the bar that matches your SAT score (e.g., "1400-1500"). Note its acceptance rate (e.g., "10%").

Now, look at the "Test Optional" bar. Note its acceptance rate (e.g., "15%").

In this example, the data shows that applicants who applied Test Optional were accepted at a higher rate than applicants who submitted a score in the 1400-1500 range. This is strong evidence that you might be better off not submitting your 1450 SAT score to this specific school.

3. Understand What "Holistic Review" Really Means

You always hear that "scores aren't everything." This tool lets you see why.

How to use:

On the Overview Charts scatter plot, look for "outliers." Find a green "Accepted" dot that has a much lower GPA or SAT score than the other accepted students.

Click on that dot.

A profile window will open. Look at their extracurriculars and awards. You will likely find the "why": maybe they had a 3.7 GPA, but they also founded a non-profit, were a nationally-ranked athlete, or had multiple international science awards.

This shows you how a strong profile in other areas can truly make a difference, and it de-mystifies the admissions process.

4. Find Your "Application Twins" to Predict Outcomes

The "Find Similar Profiles" tab is your most personal tool. It finds other students in the database who are most like you.

How to use:

Go to the Find Similar Profiles tab.

Enter your own stats: your unweighted GPA, your SAT/ACT equivalent score, and the number of APs, ECs, and Awards you have.

Click "Find Similar Profiles."

The app will show you the 5 students from the database who are your closest "application twins."

You can then look at their results to get a realistic preview of your own chances. If you see that 4 out of 5 of your "twins" were rejected from a certain school, that school is likely a "Reach" for you as well. This can also help you discover new "Target" schools you hadn't considered.

5. Get Ideas and Context for Your Own Profile

The "Applicant Browser" lets you see what successful applications look like for your specific goals.

How to use:

Use the Filters in the sidebar. Select the schools you want to get into (e.g., "T20 Accepted") and the major you're applying for (e.g., "Computer Science").

Go to the Applicant Browser tab.

You now have a list of every student who got into a T20 school for Computer Science.

Browse their profiles. What kinds of extracurriculars did they have? How many? What were their awards? This isn't about copying them, but about understanding the level of achievement and getting new ideas for how to frame your own "story" and activities.

Detailed Feature List

The dashboard is organized into several tabs, each providing a different analytical view:

Powerful Filtering: A sidebar allows users to filter the entire dataset by:

Unweighted GPA Range

SAT Equivalent Score Range

STEM Major status

Test Optional status

Specific Majors

Specific Races

Acceptance Tiers (T5, T10, T20, T50)

Overview Charts:

An interactive GPA vs. SAT Equivalent scatter plot.

Clickable Profiles: Click any point on the scatter plot to open a full detail view of that applicant.

Histograms showing the distribution of extracurriculars and award counts.

Applicant Browser:

A searchable, paginated table of all applicants that match the current filters.

Allows selection of any applicant from the table to open their full profile.

Advanced Analytics:

GPA and SAT score distributions (box plots).

A correlation matrix (heatmap) showing the relationship between numeric features like GPA, SAT, and number of AP classes.

Acceptance rate breakdown by major type (STEM vs. Non-STEM) when viewing a specific school.

Acceptance Patterns:

A powerful tool to analyze acceptance rates for a specific school or a filtered tier (e.g., T20).

Shows acceptance rate percentages broken down by GPA range (e.g., 3.5-3.7, 3.7-3.9) and SAT range (e.g., 1400-1500, 1500-1550).

Includes a "Test Optional" bucket to compare their acceptance rates.

Find Similar Profiles:

A profile matching tool using a k-Nearest Neighbors model.

Users can input their own stats (GPA, SAT, APs, etc.) and find the 5 most similar profiles from the filtered dataset.

Summary Statistics:

Key metrics (average GPA, average SAT, etc.) for the filtered data.

New charts showing the distribution of top majors and race.

Persistent Profile Ratings:

When viewing a profile's details, users can submit a 1-10 rating.

These ratings are saved permanently in a local SQLite database (profile_ratings.db).

How It Works

CollegeBase.py (Data Backend):

Loads the raw data from profiles.jsonl.

Cleans Data: Normalizes school names (e.g., "mit" -> "Massachusetts Institute of Technology") and major names.

Augments Data: Calculates new columns like sat_equivalent (from ACT), num_ecs, num_awards, stem_major, and acceptance tiers (t5_accepted, etc.).

Provides a command-line interface (CLI) for adding new profiles to the dataset.

app.py (Streamlit Frontend):

Loads the processed data from CollegeBase.py.

Uses st.session_state to manage filters and user selections.

Uses st.cache_data and st.cache_resource to speed up data loading and model training.

Renders all the tabs, charts (using Plotly), and tables.

Connects to a SQLite DB (profile_ratings.db) using st.connection to save and retrieve profile ratings.

profiles.jsonl:

The raw database. Each line is a JSON object representing one applicant.

How to Run

Option 1: Use the Batch File (Windows)

The simplest way to run the app is to use the provided batch file:

Double-click run_app.bat.

This script will automatically install all required Python packages and then start the Streamlit app.

It will open in your default web browser.

Option 2: Manual Installation (All Platforms)

If you aren't on Windows or prefer to run it manually:

Install dependencies:

pip install streamlit pandas plotly numpy sqlalchemy scikit-learn


Run the app:

streamlit run app.py


How to Add a New Profile

You can add new applicant profiles to the profiles.jsonl database by running the CollegeBase.py script directly in your terminal.

Open your terminal or command prompt.

Run the script:

python CollegeBase.py


Select Mode 1 ("Enter a New Profile").

Follow the prompts to enter the applicant's stats, ECs, awards, and results.

The script will automatically generate a unique ID, check for duplicates, and save the new profile to profiles.jsonl.

Relaunch the Streamlit app to see the new data.
