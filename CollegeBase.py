import pandas as pd
import json
import numpy as np
import hashlib
import re 

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
        str(profile.get('gender', '')),           
        '|'.join(profile.get('race', [])),
     
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
    if sat_in.lower() not in ["none", ""]:
        try:
            sat_float = float(sat_in)
            sat = int(sat_float) 
            
            if sat != sat_float or "." in sat_in:
                print(f"  -> Decimal detected. Saving score as {sat}.")
                
        except ValueError:
            print(f"  -> Invalid SAT input '{sat_in}'. Saving as 'none'.")

    act_in = prompt("ACT (or 'none'):")
    act = None
    if act_in.lower() not in ["none", ""]:
        try:
            act_float = float(act_in)
            act = int(act_float)
            
            if act != act_float or "." in act_in:
                 print(f"  -> Decimal detected. Saving score as {act}.")
                 
        except ValueError:
            print(f"  -> Invalid ACT input '{act_in}'. Saving as 'none'.")

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
        with open(path, "r", encoding='utf-8') as f:
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
    # Mathematics & Statistics
    "math": "Mathematics",
    "mathematics": "Mathematics",
    "applied math": "Applied Mathematics",
    "applied mathematics": "Applied Mathematics",
    "pure math": "Pure Mathematics",
    "pure mathematics": "Pure Mathematics",
    "statistics": "Statistics",
    "stats": "Statistics",
    "stat": "Statistics",
    "actuarial science": "Actuarial Science",
    "act sci": "Actuarial Science",
    "actuary": "Actuarial Science",
    "operations research": "Operations Research",
    "op res": "Operations Research",
    "or": "Operations Research",

    # Computer Science & IT
    "comp science": "Computer Science",
    "comp sci": "Computer Science",
    "cs": "Computer Science",
    "computer science": "Computer Science",
    "info tech": "Information Technology",
    "information technology": "Information Technology",
    "it": "Information Technology",
    "cybersecurity": "Cybersecurity",
    "cyber sec": "Cybersecurity",
    "infosec": "Cybersecurity",
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "data science": "Data Science",
    "data sci": "Data Science",
    "ds": "Data Science",
    "info systems": "Information Systems",
    "information systems": "Information Systems",
    "is": "Information Systems",
    "mis": "Information Systems",

    # Engineering
    "aero": "Aerospace Engineering",
    "aerospace engineering": "Aerospace Engineering",
    "aerospace": "Aerospace Engineering",
    "arch eng": "Architectural Engineering",
    "architectural engineering": "Architectural Engineering",
    "bme": "Biomedical Engineering",
    "biomedical engineering": "Biomedical Engineering",
    "biomed eng": "Biomedical Engineering",
    "biomed": "Biomedical Engineering",
    "chem e": "Chemical Engineering",
    "chem eng": "Chemical Engineering",
    "chemical engineering": "Chemical Engineering",
    "che": "Chemical Engineering",
    "ch e": "Chemical Engineering",
    "civil": "Civil Engineering",
    "civil engineering": "Civil Engineering",
    "ce": "Civil Engineering",
    "comp eng": "Computer Engineering",
    "computer engineering": "Computer Engineering",
    "ee": "Electrical Engineering",
    "electrical engineering": "Electrical Engineering",
    "env e": "Environmental Engineering",
    "env eng": "Environmental Engineering",
    "environmental engineering": "Environmental Engineering",
    "enviro eng": "Environmental Engineering",
    "ind e": "Industrial Engineering",
    "ind eng": "Industrial Engineering",
    "industrial engineering": "Industrial Engineering",
    "ie": "Industrial Engineering",
    "i e": "Industrial Engineering",
    "mech e": "Mechanical Engineering",
    "mechanical engineering": "Mechanical Engineering",
    "mse": "Materials Science and Engineering",
    "materials science": "Materials Science and Engineering",
    "materials engineering": "Materials Science and Engineering",
    "materials sci": "Materials Science and Engineering",
    "nuclear engineering": "Nuclear Engineering",
    "nuke e": "Nuclear Engineering",
    "ne": "Nuclear Engineering",
    "petroleum engineering": "Petroleum Engineering",
    "pet e": "Petroleum Engineering",
    "pet eng": "Petroleum Engineering",
    "robotics engineering": "Robotics Engineering",
    "robotics": "Robotics Engineering",
    "rob eng": "Robotics Engineering",
    "software engineering": "Software Engineering",
    "software eng": "Software Engineering",
    "se": "Software Engineering",
    "sw eng": "Software Engineering",
    "systems engineering": "Systems Engineering",
    "sys eng": "Systems Engineering",
    "eng physics": "Engineering Physics",
    "engineering physics": "Engineering Physics",
    "eng phys": "Engineering Physics",
    "mechatronics": "Mechatronics Engineering",
    "mechatronics engineering": "Mechatronics Engineering",

    # Biological Sciences
    "bio": "Biology",
    "biology": "Biology",
    "anatomy": "Anatomy",
    "biochem": "Biochemistry",
    "biochemistry": "Biochemistry",
    "botany": "Botany",
    "bot": "Botany",
    "cell bio": "Cell Biology",
    "cell biology": "Cell Biology",
    "ecology": "Ecology",
    "eco": "Ecology",
    "evolutionary biology": "Evolutionary Biology",
    "evo bio": "Evolutionary Biology",
    "genetics": "Genetics",
    "gen": "Genetics",
    "immunology": "Immunology",
    "immuno": "Immunology",
    "marine bio": "Marine Biology",
    "marine biology": "Marine Biology",
    "marine sci": "Marine Biology",
    "microbio": "Microbiology",
    "microbiology": "Microbiology",
    "micro": "Microbiology",
    "molecular bio": "Molecular Biology",
    "molecular biology": "Molecular Biology",
    "mol bio": "Molecular Biology",
    "neuro": "Neuroscience",
    "neuroscience": "Neuroscience",
    "pathology": "Pathology",
    "path": "Pathology",
    "physiology": "Physiology",
    "physio": "Physiology",
    "zoology": "Zoology",
    "zoo": "Zoology",

    # Physical Sciences
    "chem": "Chemistry",
    "chemistry": "Chemistry",
    "physics": "Physics",
    "phys": "Physics",
    "astro": "Astronomy",
    "astronomy": "Astronomy",
    "astrophysics": "Astrophysics",
    "astro physics": "Astrophysics",
    "earth sci": "Earth Science",
    "earth science": "Earth Science",
    "geo sci": "Earth Science",
    "geology": "Geology",
    "geol": "Geology",
    "geography": "Geography",
    "geog": "Geography",
    "meteorology": "Meteorology",
    "meteo": "Meteorology",
    "oceanography": "Oceanography",
    "ocean sci": "Oceanography",

    # Health & Medicine
    "pre-med": "Pre-Med",
    "premed": "Pre-Med",
    "nursing": "Nursing",
    "nurs": "Nursing",
    "pharmacy": "Pharmacy",
    "pharm": "Pharmacy",
    "dentistry": "Dentistry",
    "dental": "Dentistry",
    "public health": "Public Health",
    "publ hlth": "Public Health",
    "ph": "Public Health",
    "community health": "Public Health",
    "veterinary": "Veterinary Medicine",
    "vet med": "Veterinary Medicine",
    "vet": "Veterinary Medicine",
    "veterinary medicine": "Veterinary Medicine",
    "physician assistant": "Physician Assistant",
    "pa": "Physician Assistant",
    "physical therapy": "Physical Therapy",
    "phys therapy": "Physical Therapy",
    "pt": "Physical Therapy",
    "occupational therapy": "Occupational Therapy",
    "ot": "Occupational Therapy",
    "speech pathology": "Speech-Language Pathology",
    "speech lang pathology": "Speech-Language Pathology",
    "slp": "Speech-Language Pathology",
    "nutrition": "Nutrition",
    "nutr": "Nutrition",
    "dietetics": "Nutrition",
    "kinesiology": "Kinesiology",
    "kin": "Kinesiology",
    "exercise science": "Kinesiology",

    # Social Sciences
    "anthro": "Anthropology",
    "anthropology": "Anthropology",
    "communications": "Communications",
    "comm": "Communications",
    "comms": "Communications",
    "mass comm": "Communications",
    "crim justice": "Criminal Justice",
    "criminal justice": "Criminal Justice",
    "cj": "Criminal Justice",
    "econ": "Economics",
    "economics": "Economics",
    "education": "Education",
    "ed": "Education",
    "edu": "Education",
    "teaching": "Education",
    "ethnic studies": "Ethnic Studies",
    "ethn studies": "Ethnic Studies",
    "gender studies": "Gender Studies",
    "women's studies": "Gender Studies",
    "wms": "Gender Studies",
    "wgst": "Gender Studies",
    "geography": "Geography",
    "history": "History",
    "hist": "History",
    "international relations": "International Relations",
    "int'l relations": "International Relations",
    "ir": "International Relations",
    "global studies": "International Relations",
    "journalism": "Journalism",
    "journ": "Journalism",
    "legal studies": "Legal Studies",
    "liberal arts": "Liberal Arts",
    "lib arts": "Liberal Arts",
    "linguistics": "Linguistics",
    "ling": "Linguistics",
    "philosophy": "Philosophy",
    "phil": "Philosophy",
    "poli sci": "Political Science",
    "polisci": "Political Science",
    "poli science": "Political Science",
    "political science": "Political Science",
    "psychology": "Psychology",
    "psych": "Psychology",
    "psy": "Psychology",
    "public policy": "Public Policy",
    "pub pol": "Public Policy",
    "policy": "Public Policy",
    "social work": "Social Work",
    "soc work": "Social Work",
    "sw": "Social Work",
    "sociology": "Sociology",
    "soc": "Sociology",
    "urban planning": "Urban Planning",
    "city planning": "Urban Planning",
    "planning": "Urban Planning",
    "urban studies": "Urban Planning",

    # Business
    "accounting": "Accounting",
    "acct": "Accounting",
    "business": "Business Administration",
    "business admin": "Business Administration",
    "business administration": "Business Administration",
    "bus admin": "Business Administration",
    "ba": "Business Administration",
    "business analytics": "Business Analytics",
    "bus analytics": "Business Analytics",
    "entrepreneurship": "Entrepreneurship",
    "entrep": "Entrepreneurship",
    "finance": "Finance",
    "fin": "Finance",
    "hospitality management": "Hospitality Management",
    "hotel management": "Hospitality Management",
    "international business": "International Business",
    "int'l business": "International Business",
    "global business": "International Business",
    "management": "Management",
    "mgmt": "Management",
    "mgt": "Management",
    "marketing": "Marketing",
    "mktg": "Marketing",
    "mkt": "Marketing",
    "supply chain management": "Supply Chain Management",
    "scm": "Supply Chain Management",
    "logistics": "Supply Chain Management",
    "operations management": "Supply Chain Management",

    # Humanities & Arts
    "animation": "Animation",
    "anim": "Animation",
    "arabic": "Arabic",
    "ara": "Arabic",
    "architecture": "Architecture",
    "arch": "Architecture",
    "architect": "Architecture",
    "art": "Fine Arts",
    "fine arts": "Fine Arts",
    "fine art": "Fine Arts",
    "fa": "Fine Arts",
    "visual arts": "Visual Arts",
    "vis arts": "Visual Arts",
    "art history": "Art History",
    "arthist": "Art History",
    "ah": "Art History",
    "chinese": "Chinese",
    "chi": "Chinese",
    "mandarin": "Chinese",
    "classics": "Classics",
    "classical studies": "Classics",
    "class stud": "Classics",
    "creative writing": "Creative Writing",
    "creative writ": "Creative Writing",
    "cw": "Creative Writing",
    "dance": "Dance",
    "danc": "Dance",
    "english": "English",
    "engl": "English",
    "eng lit": "English",
    "film studies": "Film Studies",
    "film": "Film Studies",
    "cinema studies": "Film Studies",
    "french": "French",
    "fre": "French",
    "fashion design": "Fashion Design",
    "fashion": "Fashion Design",
    "game design": "Game Design",
    "gaming": "Game Design",
    "german": "German",
    "ger": "German",
    "graphic design": "Graphic Design",
    "graphics": "Graphic Design",
    "gd": "Graphic Design",
    "industrial design": "Industrial Design",
    "ind design": "Industrial Design",
    "id": "Industrial Design",
    "interior design": "Interior Design",
    "int design": "Interior Design",
    "int des": "Interior Design",
    "japanese": "Japanese",
    "jpn": "Japanese",
    "literature": "Literature",
    "lit": "Literature",
    "modern languages": "Modern Languages",
    "foreign languages": "Modern Languages",
    "languages": "Modern Languages",
    "music": "Music",
    "mus": "Music",
    "musicology": "Music",
    "photography": "Photography",
    "photo": "Photography",
    "spanish": "Spanish",
    "spa": "Spanish",
    "theater": "Theater",
    "theatre": "Theater",
    "thea": "Theater",
    "drama": "Theater",

    # Agriculture & Environmental
    "agriculture": "Agriculture",
    "ag": "Agriculture",
    "agri": "Agriculture",
    "agric": "Agriculture",
    "animal science": "Animal Science",
    "animal sci": "Animal Science",
    "ansci": "Animal Science",
    "animal husbandry": "Animal Science",
    "environmental science": "Environmental Science",
    "env sci": "Environmental Science",
    "enviro sci": "Environmental Science",
    "food science": "Food Science",
    "food sci": "Food Science",
    "forestry": "Forestry",
    "forest sci": "Forestry",
    "horticulture": "Horticulture",
    "hort": "Horticulture",
    "sustainability studies": "Sustainability Studies",
    "sustainability": "Sustainability Studies",
    "wildlife management": "Wildlife Management",
    "wildlife": "Wildlife Management",

    # Other
    "communications disorders": "Speech-Language Pathology",
    "comm disorders": "Speech-Language Pathology",
    "digital media": "Digital Media",
    "dig media": "Digital Media",
    "general studies": "General Studies",
    "gen studies": "General Studies",
    "library science": "Library Science",
    "lib sci": "Library Science",
    "lis": "Library Science",
    "museum studies": "Museum Studies",
    "museology": "Museum Studies",
    "nonprofit management": "Nonprofit Management",
    "nonprofit": "Nonprofit Management",
    "pre-law": "Pre-Law",
    "prelaw": "Pre-Law",
    "real estate": "Real Estate",
    "re": "Real Estate",
    "recreation management": "Recreation Management",
    "rec management": "Recreation Management",
    "sports management": "Sports Management",
    "sport management": "Sports Management",
    "sports admin": "Sports Management",
    "tourism management": "Tourism Management",
    "tourism": "Tourism Management",
}
# ACT-to-SAT Concordance Table 
# This maps ACT Composite to the nearest SAT equivalent
# Kinda redundant with the SAT-to-ACT map due to app.py having it as well
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


# NEW COMPREHENSIVE KEYWORD MAPS

EC_CATEGORY_MAP = {
    # Research & Internships
    "research": "Research",
    "intern": "Internship",
    "internship": "Internship",
    "shadow": "Shadowing/Medical",
    "hospital": "Shadowing/Medical",
    "clinic": "Shadowing/Medical",
    "medical": "Shadowing/Medical",
    "lab assistant": "Research",
    "research assistant": "Research",

    # Academic Teams
    "debate": "Speech/Debate",
    "speech": "Speech/Debate",
    "model un": "Model UN",
    "mun": "Model UN",
    "quiz bowl": "Quiz Bowl",
    "science olympiad": "Science Olympiad",
    "scioly": "Science Olympiad",
    "math team": "Math Team",
    "math club": "Math Team",
    "academic team": "Academic Team",
    "ethics bowl": "Academic Team",
    "mock trial": "Mock Trial",

    # Tech
    "robotics": "Robotics",
    "frc": "Robotics",
    "ftc": "Robotics",
    "vex": "Robotics",
    "coding": "Coding",
    "hackathon": "Hackathon",
    "cs": "Tech/CS",
    "computer science": "Tech/CS",
    "app": "Passion Project",
    "software": "Tech/CS",
    "website": "Passion Project",
    "usaco": "Olympiad (Computing)",

    # Leadership & Volunteering
    "founder": "Leadership",
    "president": "Leadership",
    "captain": "Leadership",
    "leader": "Leadership",
    "officer": "Leadership",
    "editor-in-chief": "Writing/Journalism",
    "director": "Leadership",
    "manager": "Leadership",
    "head": "Leadership",
    "vp": "Leadership",
    "vice president": "Leadership",
    "secretary": "Leadership",
    "treasurer": "Leadership",
    "student government": "Student Government",
    "student council": "Student Government",
    "volunteer": "Community Service",
    "service": "Community Service",
    "food bank": "Community Service",
    "nonprofit": "Non-Profit",
    "non-profit": "Non-Profit",
    "red cross": "Community Service",
    "key club": "Community Service",
    "nhs": "Community Service",

    # Arts
    "music": "Music/Arts",
    "band": "Music/Arts",
    "orchestra": "Music/Arts",
    "choir": "Music/Arts",
    "piano": "Music/Arts",
    "violin": "Music/Arts",
    "cello": "Music/Arts",
    "trumpet": "Music/Arts",
    "art": "Music/Arts",
    "theatre": "Music/Arts",
    "theater": "Music/Arts",
    "film": "Music/Arts",
    "dance": "Music/Arts",
    "photography": "Music/Arts",
    "writing": "Writing/Journalism",
    "newspaper": "Writing/Journalism",
    "journalism": "Writing/Journalism",
    "literary magazine": "Writing/Journalism",

    # Sports
    "varsity": "Sport",
    "jv": "Sport",
    "sport": "Sport",
    "soccer": "Sport",
    "tennis": "Sport",
    "track": "Sport",
    "swim": "Sport",
    "cross country": "Sport",
    "football": "Sport",
    "basketball": "Sport",
    "volleyball": "Sport",
    "fencing": "Sport",
    "rowing": "Sport",
    "wrestling": "Sport",
    "golf": "Sport",
    "hockey": "Sport",

    # Other
    "job": "Paid Job",
    "work": "Paid Job",
    "paid": "Paid Job",
    "cashier": "Paid Job",
    "barista": "Paid Job",
    "receptionist": "Paid Job",
    "lifeguard": "Paid Job",
    "tutor": "Tutoring/Teaching",
    "tutoring": "Tutoring/Teaching",
    "teacher": "Tutoring/Teaching",
    "teaching assistant": "Tutoring/Teaching",
    "ta": "Tutoring/Teaching",
    "instructor": "Tutoring/Teaching",
    "coach": "Tutoring/Teaching",
    "cultural": "Cultural Club",
    "chinese": "Cultural Club",
    "asian": "Cultural Club",
    "black student union": "Cultural Club",
    "jewish": "Cultural Club",
    "hispanic": "Cultural Club",
    "french": "Cultural Club",
    "summer program": "Summer Program",
    "summer school": "Summer Program",
    "governor's school": "Summer Program",
    "mostec": "Summer Program",
    "yygs": "Summer Program",
    "launchx": "Summer Program",
    "chess": "Hobby/Club",
    "baking": "Hobby/Club",
    "cooking": "Hobby/Club",
    "family responsibilit": "Family Responsibilities", # Partial match
}

AWARD_CATEGORY_MAP = {
    # Academic
    "ap scholar": "AP Scholar",
    "national merit": "National Merit",
    "semifinalist": "National Merit",
    "finalist": "National Merit",
    "commended": "National Merit",
    "honor roll": "School Honor Roll",
    "dean's list": "School Honor Roll",
    "cum laude": "School Honor Roll",
    "principal's list": "School Honor Roll",
    "nhs": "NHS",
    "national honor society": "NHS",
    "questbridge": "QuestBridge",

    # STEM Competitions
    "aime": "Amc/Aime",
    "amc": "Amc/Aime",
    "math competition": "Other Math Competition",
    "math olympiad": "Math Olympiad (USA)",
    "usaco": "Computing Olympiad (USA)",
    "usabo": "Biology Olympiad (USA)",
    "usnco": "Chemistry Olympiad (USA)",
    "usapho": "Physics Olympiad (USA)",
    "olympiad": "Olympiad (USA)", # This is the generic fallback
    "isef": "Isef/Other Science Fairs",
    "science fair": "Isef/Other Science Fairs",
    "regeneron": "Isef/Other Science Fairs",
    "sts": "Isef/Other Science Fairs",

    # Other Competitions
    "deca": "Deca/Fbla",
    "fbla": "Deca/Fbla",
    "hosa": "HOSA",
    "scholastic art": "Scholastic Art/Writing",
    "scholastic writing": "Scholastic Art/Writing",
    "science bowl": "Science Bowl",

    # Scouting
    "eagle scout": "Eagle Scout",
    "gold award": "Eagle Scout",

    # Other
    "pvsa": "PVSA",
    "president's volunteer service": "PVSA",
    "national latin exam": "Language Award",
    "national spanish exam": "Language Award",
    "national french exam": "Language Award",
    "seal of biliteracy": "Language Award",
    "all-state": "Music/Art Award",
    "all-region": "Music/Art Award",
    "youngarts": "Music/Art Award",
}

def extract_categories(text_list, keyword_map):
    """Iterates over a list of strings and extracts categories based on a keyword map."""
    categories = set()
    if not isinstance(text_list, list):
        return []
    
    # Compile regex patterns for word boundary matching
    # This prevents 'art' from matching 'startup'
    compiled_keys = {
        key: re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE) 
        for key in keyword_map.keys()
    }
    
    # Sort keys by length, descending.
    # This ensures "math olympiad" is checked before "olympiad".
    sorted_keys = sorted(keyword_map.keys(), key=len, reverse=True)
    
    for item in text_list:
        item_str = str(item)
        
        # Iterate using the sorted_keys list
        for key in sorted_keys: 
            if compiled_keys[key].search(item_str):
                # Add the category corresponding to the matched key
                categories.add(keyword_map[key])
                # Break from the inner loop to take the FIRST (most specific) match
                break
            
    if not categories:
        categories.add("Other")
        
    return sorted(list(categories))

 


# Original functions

def load_profiles(path="profiles.jsonl"):
    "Loads profiles from a JSON Lines file into a Pandas DataFrame."
    rows = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        rows.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        print(f"Skipping malformed line: {line[:100]}...") # Log error
    except FileNotFoundError:
        print(f"Error: File '{path}' not found. Returning empty DataFrame.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    
   
    # Some profiles from the user file have nested/malformed data.
    # We will try to flatten them.
    def clean_column(col_series, col_name):
        def extract_value(x):
            if isinstance(x, str) and '\n' in x:
                # Heuristic: This looks like a malformed entry
                # Example: "Politics\nRace:\nWhite\nGender:\nFemale..."
                # We'll try to find the first meaningful line.
                first_line = x.split('\n')[0].strip()
                # print(f"Cleaning malformed entry in '{col_name}': '{x[:50]}...' -> '{first_line}'") # DEBUG
                return [first_line] # Return as list for consistency
            elif isinstance(x, list):
                return x # Already a list, good.
            elif pd.isna(x):
                return [] # Use empty list for NaNs
            else:
                return [str(x)] # Convert other types to list of strings
        
        # Apply the cleaning function
        cleaned = col_series.apply(extract_value)
        
        # Handle cases where `majors` might be a single string
        if col_name == 'majors' and col_series.apply(type).eq(str).any():
             # print(f"Warning: Found string-types in '{col_name}', converting.") # DEBUG
             cleaned = col_series.apply(lambda x: [x] if isinstance(x, str) else (x if isinstance(x, list) else []))
             
        return cleaned

    # List of columns that should be lists
    list_cols = ['majors', 'race', 'awards', 'extracurriculars', 'acceptances', 'rejections']
    for col in list_cols:
        if col in df.columns:
            df[col] = clean_column(df[col], col)
        else:
            # Add empty list as default for missing columns
            df[col] = [[] for _ in range(len(df))] 
            
    # Normalize numeric and string columns
    for col in ['gpa_unweighted', 'gpa_weighted', 'sat', 'act', 'ap_classes', 'ib_classes']:
        if col in df.columns:
            # Forcibly clean GPA strings like '95.3' before numeric conversion
            if 'gpa' in col:
                 df[col] = df[col].apply(lambda x: str(x).split('\n')[0].strip() if isinstance(x, str) else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    if 'gender' in df.columns:
        df['gender'] = df['gender'].apply(lambda x: str(x).split('\n')[0].strip() if isinstance(x, str) else x)
        df['gender'] = df['gender'].replace('nan', np.nan)


    # Drop profiles that are mostly empty (e.g., just an ID)
    df = df.dropna(subset=['gpa_unweighted', 'sat', 'act'], how='all')
    

    # We will ALWAYS regenerate the profile_id from the content because
    # the IDs in the source file are unreliable and duplicated.
    print("Regenerating all profile IDs from content to ensure uniqueness...")
    df['profile_id'] = df.apply(
        lambda row: generate_profile_id(row.to_dict()), 
        axis=1
    )
    # --- END OF KEY CHANGE ---

    # Set profile_id as index
    df = df.set_index('profile_id', drop=False)
        
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

    print("Extracting EC and Award categories...")
    if 'extracurriculars' in df.columns:
        df['ec_categories'] = df['extracurriculars'].apply(
            lambda x: extract_categories(x, EC_CATEGORY_MAP)
        )
    
    if 'awards' in df.columns:
        df['award_categories'] = df['awards'].apply(
            lambda x: extract_categories(x, AWARD_CATEGORY_MAP)
        )
 

    # Tiered Acceptance Flags # Based on 2024-2025 US News National University Rankings
    
    t5 = [
        "Princeton University", "Massachusetts Institute of Technology", "Harvard University", "Stanford University", "California Institute of Technology"
    ]
    
    t10 = [
        "Princeton University", "Massachusetts Institute of Technology", "Harvard University", "Stanford University", "California Institute of Technology",
        "University of Chicago", "Yale University", "University of Pennsylvania", "Johns Hopkins University", "Brown University",
        "Columbia University", "Northwestern University"
    ] 

    t20 = [
        "Princeton University", "Massachusetts Institute of Technology", "Harvard University", "Stanford University", "California Institute of Technology",
        "University of Chicago", "Yale University", "University of Pennsylvania", "Johns Hopkins University", "Brown University",
        "Columbia University", "Northwestern University", "Cornell University", "University of California, Berkeley", "University of California, Los Angeles",
        "Dartmouth College", "Duke University", "University of Michigan-Ann Arbor", "Vanderbilt University"
    ] 
    
    t50 = [
        "Princeton University", "Massachusetts Institute of Technology", "Harvard University", "Stanford University", "California Institute of Technology",
        "University of Chicago", "Yale University", "University of Pennsylvania", "Johns Hopkins University", "Brown University",
        "Columbia University", "Northwestern University", "Cornell University", "University of California, Berkeley", "University of California, Los Angeles",
        "Dartmouth College", "Duke University", "University of Michigan-Ann Arbor", "Vanderbilt University", "Rice University",
        "University of Notre Dame", "Washington University in St. Louis", "Carnegie Mellon University", "Emory University", "Georgetown University",
        "University of Florida", "University of North Carolina, Chapel Hill", "University of Southern California", "University of Virginia", "University of California, Davis",
        "University of California, San Diego", "University of Wisconsin-Madison", "Boston College", "Georgia Institute of Technology", "New York University",
        "University of Illinois, Urbana Champaign", "University of Texas at Austin", "Boston University", "Rutgers University-New Brunswick", "Tufts University",
        "University of Washington", "Case Western Reserve University", "Florida State University", "Northeastern University", "University of California, Irvine",
        "University of California, Santa Barbara", "Purdue University-Main Campus", "University of Georgia", "University of Maryland-College Park", "University of Rochester"
    ]

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
    print("2. Load, Analyze, and Save Data (Processing Mode)")
    
    mode = prompt("Select mode (1 or 2):")
    
    if mode == '1':
        profile = enter_by_prompt()
        save_profile(profile)
        
    elif mode == '2':
    
        data_path = "profiles.jsonl"
        df = load_data(data_path)
        
        if not df.empty:
            print("\nProcessed Data Snapshot (Top 5 Rows)")
            # Show new category columns
            print(df[['profile_id', 'gpa_unweighted', 'sat_equivalent', 
                      'ec_categories', 'award_categories', 't20_accepted']].head())
            print(f"\nTotal Profiles Loaded: {len(df)}")
            
        
            try:
                
                # Now that we have re-generated all IDs, we can safely
                # drop duplicates. This will only drop *true content* duplicates,
                # not unique profiles that shared a bad ID.
                df_to_save = df.drop_duplicates(subset=['profile_id'], keep='last')
                
                # We must reset the index so the 'profile_id' (which is the index)
                # gets saved to the jsonl file.
                df_to_save = df_to_save.reset_index(drop=True)
                
                df_to_save.to_json(data_path, orient="records", lines=True, force_ascii=False)
                print(f"\n✔ Successfully saved {len(df_to_save)} fully analyzed profiles back to {data_path}.")
            except Exception as e:
                print(f"\n Error saving augmented data back to file: {e}")
            
            
            print("\nReady for Trend Analysis and UI features...")
            
    else:
        print("Invalid selection. Exiting.")