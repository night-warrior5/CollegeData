import pandas as pd
import json
import numpy as np
import hashlib


# Core Utility Functions

def prompt(msg):
    "Prompts the user and returns the stripped input."
    return input(msg + " ").strip()

def prompt_numbered_list(label):
    "Handles numbered list input (Awards, ECs, Majors)."
    items = []
    i = 1
    print(f"\nEnter {label}(s) (type 'stop' to finish):")
    while True:
        entry = prompt(f"{label}{i}:")
        if entry.lower() == "stop":
            break
        if entry:
            items.append(entry)
            i += 1
    # Sort the list to ensure consistent hashing later (for profile id)
    return sorted(items)

def normalize_gpa(x):
    "Normalizes GPA to a 4.0 scale if it appears to be on a 100-scale."
    if x is None:
        return None
    try:
        x_float = float(x)
        if x_float > 5:
            return (x_float / 100) * 4
        return x_float
    except ValueError:
        return None

def generate_profile_id(profile):
    "Creates a unique SHA256 hash based on core, sorted profile inputs."
    # Collect key inputs into a list of strings
    key_inputs = [
        str(profile.get('gpa_unweighted')),
        str(profile.get('gpa_weighted')),
        str(profile.get('sat')),
        str(profile.get('act')),
        str(profile.get('ap_classes')),
        str(profile.get('ib_classes')),
        str(profile.get('college_credit_classes')),
        '|'.join(profile.get('majors', [])),
        profile.get('gender', ''),
        '|'.join(profile.get('race', [])),
        # Sorted ECs/Awards ensure consistency
        '|'.join(profile.get('awards', [])),
        '|'.join(profile.get('extracurriculars', []))
    ]
    
    unique_string = "||".join(key_inputs)
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

# Data Input and Saving

def enter_by_prompt():
    "Gathers the complete college profile and generates a unique ID."
    print("\n=== ENTER PROFILE (PROMPT MODE) ===")
    
    # Data Collection (prompts) 
    raw_gpa = prompt("Unweighted GPA (4.0 or 100 scale):")
    gpa_unw = normalize_gpa(raw_gpa)
    
    wgpa_in = prompt("Weighted GPA (5.0 or 100 scale, or 0/none to skip):")
    gpa_weighted = None
    if wgpa_in.lower() not in ["0", "none", ""]:
        gpa_weighted = normalize_gpa(wgpa_in)

    sat_in = prompt("SAT (or 'none'):")
    sat = None
    if sat_in.lower() != "none" and sat_in.isdigit():
        sat = int(sat_in)

    act_in = prompt("ACT (or 'none'):")
    act = None
    if act_in.lower() != "none" and act_in.isdigit():
        act = int(act_in)

    ap_classes = int(prompt("Number of AP classes (0 if none):") or 0)
    ib_classes = int(prompt("Number of IB classes (0 if none):") or 0)
    college_credit_classes = int(prompt("College-credit / Dual Enrollment classes (0 if none):") or 0)

    majors = prompt_numbered_list("Major")
    gender = prompt("Gender:")
    race = prompt_numbered_list("Race")
    awards = prompt_numbered_list("Award")
    ecs = prompt_numbered_list("EC")
    accepts = prompt_numbered_list("Acceptance")
    rejects = prompt_numbered_list("Rejection")
    
    profile = {
        "gpa_unweighted": gpa_unw, "gpa_weighted": gpa_weighted, "sat": sat, "act": act,
        "ap_classes": ap_classes, "ib_classes": ib_classes, "college_credit_classes": college_credit_classes,
        "majors": majors, "gender": gender, "race": race, "awards": awards,
        "extracurriculars": ecs, "acceptances": accepts, "rejections": rejects
    }
    
    # Generate ID and add to profile
    profile['profile_id'] = generate_profile_id(profile)

    return profile

def save_profile(profile, path="profiles.jsonl"):
    "Saves the profile, checking for duplicates by ID."
    profile_id = profile['profile_id']
    
    # Check for duplicates
    duplicate_found = False
    try:
        with open(path, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    existing_profile = json.loads(stripped)
                    if existing_profile.get('profile_id') == profile_id:
                        print(f"\n⚠️ Duplicate profile detected (ID: {profile_id[:10]}...). Not saved.")
                        duplicate_found = True
                        break
                except json.JSONDecodeError:
                    # Skip malformed lines but continue checking (for invalid json)
                    continue
    except FileNotFoundError:
        pass # File doesn't exist, safe to save
    
    if duplicate_found:
        return
        
    # Save if unique
    try:
        with open(path, "a", encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False)
            f.write('\n')
        print(f"\n✔ Profile saved → {path} (ID: {profile_id[:10]}...)\n")
    except Exception as e:
        print(f"\n❌ Error saving profile: {e}\n")

# Data Loading and Augmentation

# Alias Maps For Normalization

SCHOOL_ALIASES = {
    # User's examples (for normalization)
    "mit": "Massachusetts Institute of Technology",
    "massachusetts": "Massachusetts Institute of Technology",
    "massachusetts institute of technology": "Massachusetts Institute of Technology",

    # Common aliases (for normalization)
    "caltech": "California Institute of Technology",
    "california institute of technology": "California Institute of Technology",
    "ucb": "University of California, Berkeley",
    "berkeley": "University of California, Berkeley",
    "university of california, berkeley": "University of California, Berkeley",
    "ucla": "University of California, Los Angeles",
    "university of california, los angeles": "University of California, Los Angeles",
    "ucsd": "University of California, San Diego",
    "university of california, san diego": "University of California, San Diego",
    "ucsb": "University of California, Santa Barbara",
    "university of california, santa barbara": "University of California, Santa Barbara",
    "uci": "University of California, Irvine",
    "university of california, irvine": "University of California, Irvine",
    "ucd": "University of California, Davis",
    "university of california, davis": "University of California, Davis",
    "upenn": "University of Pennsylvania",
    "penn": "University of Pennsylvania",
    "university of pennsylvania": "University of Pennsylvania",
    "cmu": "Carnegie Mellon University",
    "carnegie mellon university": "Carnegie Mellon University",
    "nyu": "New York University",
    "new york university": "New York University",
    "usc": "University of Southern California",
    "university of southern california": "University of Southern California",
    "gt": "Georgia Institute of Technology",
    "georgia tech": "Georgia Institute of Technology",
    "georgia institute of technology": "Georgia Institute of Technology",
    "uiuc": "University of Illinois, Urbana Champaign",
    "university of illinois, urbana champaign": "University of Illinois, Urbana Champaign",
    "uva": "University of Virginia",
    "university of virginia": "University of Virginia",
    "unc": "University of North Carolina, Chapel Hill",
    "unc chapel hill": "University of North Carolina, Chapel Hill",
    "university of north carolina, chapel hill": "University of North Carolina, Chapel Hill",
    "wustl": "Washington University in St. Louis",
    "washu": "Washington University in St. Louis",
    "washington university in st. louis": "Washington University in St. Louis",
    "jhu": "Johns Hopkins University",
    "johns hopkins university": "Johns Hopkins University",
}

MAJOR_ALIASES = {
    # Common aliases (for normalization)
    "math": "Mathematics",
    "mathematics": "Mathematics",
    "comp science": "Computer Science",
    "cs": "Computer Science",
    "computer science": "Computer Science",
    "bio": "Biology",
    "biology": "Biology",
    "biochem": "Biochemistry",
    "biochemistry": "Biochemistry",
    "chem": "Chemistry",
    "chemistry": "Chemistry",
    "econ": "Economics",
    "economics": "Economics",
    "poli sci": "Political Science",
    "polisci": "Political Science",
    "political science": "Political Science",
    "bme": "Biomedical Engineering",
    "biomedical engineering": "Biomedical Engineering",
    "aero": "Aerospace Engineering",
    "aerospace engineering": "Aerospace Engineering",
    "mech e": "Mechanical Engineering",
    "mechanical engineering": "Mechanical Engineering",
    "ee": "Electrical Engineering",
    "electrical engineering": "Electrical Engineering",
    "pre-med": "Pre-Med",
    "premed": "Pre-Med",
    "neuro": "Neuroscience",
    "neuroscience": "Neuroscience",
}

# ACT-to-SAT Concordance Table (Based on 2018 College Board) 
# This maps ACT Composite to the nearest SAT equivalent
ACT_TO_SAT_CONCORDANCE = {
    36: 1600, 35: 1570, 34: 1520, 33: 1490, 32: 1450, 31: 1420, 30: 1390,
    29: 1360, 28: 1330, 27: 1300, 26: 1260, 25: 1230, 24: 1200, 23: 1160,
    22: 1130, 21: 1100, 20: 1060, 19: 1020, 18: 980,  17: 940,  16: 900,
    15: 860,  14: 810,  13: 770,  12: 720,  11: 670,  10: 620,  9: 560
}


# Helper Functions For Normalization

def normalize_name(name, alias_map):
    "Converts a name to its canonical form using an alias map."
    if not isinstance(name, str):
        return name
    name_lower = name.lower().strip()
    return alias_map.get(name_lower, name)

def normalize_list_column(series, alias_map):
    "Applies normalization to a column containing lists of strings."
    def normalize_list(items):
        if not isinstance(items, list):
            return items
        normalized_set = {normalize_name(item, alias_map) for item in items}
        return sorted(list(normalized_set))
    
    return series.apply(normalize_list)


# Original functions

def load_profiles(path="profiles.jsonl"):
    "Loads profiles from a JSON Lines file into a Pandas DataFrame."
    rows = []
    try:
        with open(path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
    except FileNotFoundError:
        print(f"Error: File '{path}' not found. Returning empty DataFrame.")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Ensure scores are numeric (NaN for nulls)
    for col in ['gpa_unweighted', 'gpa_weighted', 'sat', 'act']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.fillna(np.nan)

def augment_dataframe(df):
    "Adds calculated columns like counts, STEM flag, and acceptance tiers."
    if df.empty:
        return df

    # Normalize data *before* augmentation 
    print("Normalizing majors and school names...")
    if 'majors' in df.columns:
        df['majors'] = normalize_list_column(df['majors'], MAJOR_ALIASES)
    
    if 'acceptances' in df.columns:
        df['acceptances'] = normalize_list_column(df['acceptances'], SCHOOL_ALIASES)
    
    if 'rejections' in df.columns:
        df['rejections'] = normalize_list_column(df['rejections'], SCHOOL_ALIASES)

        # Create sat_equivalent and test_optional columns 
    print("Calculating SAT equivalent scores and Test Optional flag...")

    # Create the sat_equivalent column
    def get_sat_equivalent(row):
        if pd.notna(row['sat']):
            return row['sat']
        if pd.notna(row['act']):
            return ACT_TO_SAT_CONCORDANCE.get(row['act'], np.nan)
        return np.nan

    df['sat_equivalent'] = df.apply(get_sat_equivalent, axis=1)

    # Create the test_optional flag
    df['test_optional'] = df['sat'].isnull() & df['act'].isnull()

    # Basic Counts 
    df['num_ecs'] = df['extracurriculars'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_awards'] = df['awards'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_acceptances'] = df['acceptances'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_rejections'] = df['rejections'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # STEM Major Flag 
    stem_keywords = [
        "Computer Science", "Mathematics", "Physics", "Biology", 
        "Chemistry", "Engineering", "Biochemistry", "Neuroscience",
        "Biomedical Engineering", "Aerospace Engineering", 
        "Mechanical Engineering", "Electrical Engineering"
    ]
    
    def is_stem_major(majors):
        if not isinstance(majors, list) or not majors:
            return False
        return any(major in stem_keywords for major in majors)

    df['stem_major'] = df['majors'].apply(is_stem_major)

    # Tiered Acceptance Flags 
    t5 = ["Harvard University", "Stanford University", "Yale University", "Princeton University", "Massachusetts Institute of Technology"]
    t10 = t5 + ["University of Chicago", "Columbia University", "University of Pennsylvania", "California Institute of Technology"]
    t20 = t10 + ["Dartmouth College", "Brown University", "Duke University", "Northwestern University", "Cornell University", "Johns Hopkins University", "Rice University"]
    t50 = t20 + ["University of California, Los Angeles", "University of California, Berkeley", "Carnegie Mellon University", "University of Michigan-Ann Arbor", "University of Southern California", "Emory University", "New York University", "Georgetown University", "University of Virginia", "Tufts University", "University of California, San Diego", "Wake Forest University", "Boston College", "University of Rochester", "Georgia Institute of Technology", "Brandeis University", "Case Western Reserve University", "William & Mary", "Northeastern University"]

    def accepted_in(acceptances, tier_list):
        if not isinstance(acceptances, list) or not acceptances:
            return False
        return any(school in tier_list for school in acceptances)

    df['t5_accepted'] = df['acceptances'].apply(lambda x: accepted_in(x, t5))
    df['t10_accepted'] = df['acceptances'].apply(lambda x: accepted_in(x, t10))
    df['t20_accepted'] = df['acceptances'].apply(lambda x: accepted_in(x, t20))
    df['t50_accepted'] = df['acceptances'].apply(lambda x: accepted_in(x, t50))

    return df

def load_data(path="profiles.jsonl"):
    "Loads, processes, and augments the college profile data."
    print(f"Loading and processing profiles from {path} ...")
    df = load_profiles(path)
    df = augment_dataframe(df)
    print("✔ Done! Data processed.")
    return df

    # Main Execution Menu

if __name__ == "__main__":
    
    print("\n\nCollegeBase Data Management System")
    print("1. Enter a New Profile (Input Mode)")
    print("2. Load and Analyze Existing Data (Analysis Mode)")
    
    mode = prompt("Select mode (1 or 2):")
    
    if mode == '1':
        profile = enter_by_prompt()
        save_profile(profile)
        
    elif mode == '2':
        df = load_data()
        
        if not df.empty:
            print("\nProcessed Data Snapshot")
            print(df[['profile_id', 'gpa_unweighted', 'sat_equivalent', 'test_optional', 'majors', 
                      'num_ecs', 'stem_major', 't5_accepted']].head())
            print(f"\nTotal Profiles Loaded: {len(df)}")
            
            # This is where future analysis features would hook in (future features)    
            print("\nReady for Trend Analysis and UI features...")
            
    else:
        print("Invalid selection. Exiting.")