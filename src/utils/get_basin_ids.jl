using DelimitedFiles

# load bucket_opt_init
bucket_opt_init = readdlm(joinpath("src/data/bucket_opt_init.csv"), ',')
basins_available = lpad.(string.(Int.(bucket_opt_init[:, 1])), 8, "0")
for i in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]
    # Filter basin IDs that start with "01"
    tmp_basins = filter(id -> startswith(id, i), basins_available)
    # Write basin IDs to text file, one per line
    open("src/data/basin_ids/basins_$i.txt", "w") do f
        for basin_id in tmp_basins
            println(f, basin_id)
        end
    end
end

open("src/data/basin_ids/basins_all.txt", "w") do f
    for basin_id in basins_available
        println(f, basin_id)
    end
end

open("src/data/basin_ids/basins_all_reverse.txt", "w") do f
    for basin_id in reverse(basins_available)
        println(f, basin_id)
    end
end
