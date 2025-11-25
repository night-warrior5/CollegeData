using JSON3
using DataFrames


function load_profiles(path="profiles.jsonl")
    rows = Vector{Dict}()

    open(path, "r") do f
        for line in eachline(f)
            stripped = strip(line)
            isempty(stripped) && continue
            push!(rows, JSON3.read(stripped))
        end
    end

    return rows
end


function profiles_to_dataframe(rows::Vector{Dict})
    df = DataFrame()

    # Find all distinct keys across all profiles
    cols = Set{Symbol}()
    for r in rows
        for k in keys(r)
            push!(cols, Symbol(k))
        end
    end

    # FIX: Initialize columns with Any instead of Missing
    for c in cols
        df[!, c] = Vector{Any}(undef, length(rows))
    end

    # Fill data row by row
    for (i, r) in enumerate(rows)
        for (k, v) in r
            df[i, Symbol(k)] = v
        end
    end

    return df
end


function augment_dataframe(df)

    df.num_ecs = map(x -> x === missing || x === nothing ? 0 : length(x), df.extracurriculars)
    df.num_awards = map(x -> x === missing || x === nothing ? 0 : length(x), df.awards)
    df.num_acceptances = map(x -> x === missing || x === nothing ? 0 : length(x), df.acceptances)
    df.num_rejections = map(x -> x === missing || x === nothing ? 0 : length(x), df.rejections)


    stem_keywords = ["CS", "Computer", "Math", "Physics", "Bio", "Chem", "Engineering"]

    df.stem_major = [
        # Check if majors list is valid first
        (m === missing || m === nothing) ? false :
        # If valid, check if ANY major (s) contains ANY keyword (k)
        any(occursin(k, s) for s in m, k in stem_keywords)
        for m in df.majors
    ]


    t5 = [
        "Harvard University", "Stanford University", "Yale University",
        "Princeton University", "Massachusetts Institute of Technology"
    ]

    t10 = vcat(t5, [
        "University of Chicago", "Columbia University", "University of Pennsylvania",
        "California Institute of Technology"
    ])

    t20 = vcat(t10, [
        "Dartmouth College", "Brown University", "Duke University", "Northwestern University",
        "Cornell University", "Johns Hopkins University", "Rice University"
    ])

    t50 = vcat(t20, [
        "University of California, Los Angeles", "University of California, Berkeley",
        "Carnegie Mellon University", "University of Michigan-Ann Arbor",
        "University of Southern California", "Emory University",
        "New York University", "Georgetown University", "University of Virginia",
        "Tufts University", "University of California, San Diego",
        "Wake Forest University", "Boston College", "University of Rochester",
        "Georgia Institute of Technology", "Brandeis University",
        "Case Western Reserve University", "William & Mary",
        "Northeastern University"
    ])

    function accepted_in(row, list)
        acc = row[:acceptances]
        acc isa Missing || acc === nothing && return false
        return any(s -> s in list, acc)
    end

    df.t5_accepted = [accepted_in(row, t5) for row in eachrow(df)]
    df.t10_accepted = [accepted_in(row, t10) for row in eachrow(df)]
    df.t20_accepted = [accepted_in(row, t20) for row in eachrow(df)]
    df.t50_accepted = [accepted_in(row, t50) for row in eachrow(df)]

    return df
end


function load_data(path="profiles.jsonl")
    println("Loading profiles from $path ...")
    rows = load_profiles(path)

    println("Converting to DataFrame ...")
    df = profiles_to_dataframe(rows)

    println("Adding derived metrics ...")
    df = augment_dataframe(df)

    println("✔ Done! Profiles loaded and processed.")
    return df
end

# -------- Run standalone --------
if abspath(PROGRAM_FILE) == @__FILE__
    df = load_data()
    println(df)

end
