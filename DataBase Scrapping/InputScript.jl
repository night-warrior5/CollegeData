using JSON3

function prompt(msg)
    print(msg * " ")
    return readline()
end

# Numbered List Input (EC1, Award1, etc.)
function prompt_numbered_list(label)
    items = String[]
    i = 1
    println("Enter $label(s) (type 'stop' to finish):")
    while true
        print("$label$(i): ")
        entry = readline()

        if lowercase(strip(entry)) == "stop"
            break
        end

        if strip(entry) != ""
            push!(items, strip(entry))
            i += 1
        end
    end
    return items
end

# Normalize GPA (4.0 or 100 scale)
function normalize_gpa(x::Float64)
    if x > 5
        return (x / 100) * 4
    else
        return x
    end
end

# PROMPT MODE INPUT
function enter_by_prompt()
    println("\n=== ENTER PROFILE (PROMPT MODE) ===")

    raw_gpa = parse(Float64, prompt("Unweighted GPA (4.0 or 100 scale):"))
    gpa_unw = normalize_gpa(raw_gpa)

    wgpa_in = prompt("Weighted GPA (5.0 or 100 scale, or 0 to skip):")
    if wgpa_in == "0"
        gpa_weighted = nothing
    else
        gpa_weighted = normalize_gpa(parse(Float64, wgpa_in))
    end

    sat_in = prompt("SAT (or 'none'):")
    sat = sat_in == "none" ? nothing : parse(Int, sat_in)

    act_in = prompt("ACT (or 'none'):")
    act = act_in == "none" ? nothing : parse(Int, act_in)

    ap_classes = parse(Int, prompt("Number of AP classes (0 if none):"))
    ib_classes = parse(Int, prompt("Number of IB classes (0 if none):"))
    college_credit_classes = parse(Int, prompt("College-credit / Dual Enrollment classes (0 if none):"))

    majors = prompt_numbered_list("Major")
    gender = prompt("Gender:")
    race = prompt_numbered_list("Race")

    awards = prompt_numbered_list("Award")
    ecs = prompt_numbered_list("EC")

    accepts = prompt_numbered_list("Acceptance")
    rejects = prompt_numbered_list("Rejection")

    profile = Dict(
        "gpa_unweighted" => gpa_unw,
        "gpa_weighted" => gpa_weighted,
        "sat" => sat,
        "act" => act,
        "ap_classes" => ap_classes,
        "ib_classes" => ib_classes,
        "college_credit_classes" => college_credit_classes,
        "majors" => majors,
        "gender" => gender,
        "race" => race,
        "awards" => awards,
        "extracurriculars" => ecs,
        "acceptances" => accepts,
        "rejections" => rejects
    )

    return profile
end

# Save to profiles.jsonl
function save_profile(profile)
    open("profiles.jsonl", "a") do f
        write(f, JSON3.write(profile) * "\n")
    end
    println("\n✔ Profile saved → profiles.jsonl\n")
end

# Main Menu — only prompt mode
println("=== CollegeBase Data Input ===")

profile = enter_by_prompt()
save_profile(profile)
