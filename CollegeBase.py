import pandas as pd
import json
import numpy as np
import hashlib

# ====================================================================
# I. CORE UTILITY FUNCTIONS (From InputScript.py)
# ====================================================================

def prompt(msg):
    """Prompts the user and returns the stripped input."""
    return input(msg + " ").strip()

def prompt_numbered_list(label):
    """Handles numbered list input (Awards, ECs, Majors)."""
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
    # Sort the list to ensure consistent hashing later
    return sorted(items)

def normalize_gpa(x):
    """Normalizes GPA to a 4.0 scale if it appears to be on a 100-scale."""
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
    """Creates a unique SHA256 hash based on core, sorted profile inputs."""
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

# ====================================================================
# II. DATA INPUT AND SAVING (From InputScript.py)
# ====================================================================

def enter_by_prompt():
    """Gathers the complete college profile and generates a unique ID."""
    print("\n=== ENTER PROFILE (PROMPT MODE) ===")

    # --- Data Collection (prompts) ---
    raw_gpa = prompt("Unweighted GPA (4.0 or 100 scale):")
    gpa_unw = normalize_gpa(raw_gpa)
    # ... (rest of data collection prompts go here, similar to InputScript.py) ...
    
    # For brevity in this combined script, we'll define a minimal set of data collected:
    
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
    """Saves the profile, checking for duplicates by ID."""
    profile_id = profile['profile_id']
    
    # 1. Check for duplicates
    try:
        with open(path, "r") as f:
            for line in f:
                existing_profile = json.loads(line)
                if existing_profile.get('profile_id') == profile_id:
                    print(f"\n⚠️ Duplicate profile detected (ID: {profile_id[:10]}...). Not saved.")
                    return
    except FileNotFoundError:
        pass # File doesn't exist, safe to save
    except json.JSONDecodeError:
        print("\nError reading profiles.jsonl. Cannot guarantee duplicate check.")
        return 
        
    # 2. Save if unique
    with open(path, "a") as f:
        json.dump(profile, f)
        f.write('\n')
    print(f"\n✔ Profile saved → {path} (ID: {profile_id[:10]}...)\n")

# ====================================================================
# III. DATA LOADING AND AUGMENTATION (From LoadProfile.py)
# ====================================================================

def load_profiles(path="profiles.jsonl"):
    """Loads profiles from a JSON Lines file into a Pandas DataFrame."""
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
    # Ensure GPAs are floats (NaN for nulls)
    for col in ['gpa_unweighted', 'gpa_weighted']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.fillna(np.nan)

def augment_dataframe(df):
    """Adds calculated columns like counts, STEM flag, and acceptance tiers."""
    if df.empty:
        return df

    # --- Basic Counts ---
    df['num_ecs'] = df['extracurriculars'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_awards'] = df['awards'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_acceptances'] = df['acceptances'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_rejections'] = df['rejections'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # --- STEM Major Flag ---
    stem_keywords = ["CS", "Computer", "Math", "Physics", "Bio", "Chem", "Engineering"]
    
    def is_stem_major(majors):
        if not isinstance(majors, list) or not majors:
            return False
        for major in majors:
            for keyword in stem_keywords:
                if keyword.lower() in major.lower():
                    return True
        return False

    df['stem_major'] = df['majors'].apply(is_stem_major)

    # --- Tiered Acceptance Flags ---
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
    """Loads, processes, and augments the college profile data."""
    print(f"Loading and processing profiles from {path} ...")
    df = load_profiles(path)
    df = augment_dataframe(df)
    print("✔ Done! Data processed.")
    return df

# ====================================================================
# IV. MAIN EXECUTION MENU
# ====================================================================

if __name__ == "__main__":
    
    print("\n\n=== CollegeBase Data Management System ===")
    print("1. Enter a New Profile (Input Mode)")
    print("2. Load and Analyze Existing Data (Analysis Mode)")
    
    mode = prompt("Select mode (1 or 2):")
    
    if mode == '1':
        profile = enter_by_prompt()
        save_profile(profile)
        
    elif mode == '2':
        df = load_data()
        
        if not df.empty:
            print("\n--- Processed Data Snapshot ---")
            print(df[['profile_id', 'gpa_unweighted', 'sat', 'majors', 
                      'num_ecs', 'stem_major', 't5_accepted']].head())
            print(f"\nTotal Profiles Loaded: {len(df)}")
            
            # This is where future analysis features would hook in
            print("\nReady for Trend Analysis and UI features...")
            
    else:
        print("Invalid selection. Exiting.")