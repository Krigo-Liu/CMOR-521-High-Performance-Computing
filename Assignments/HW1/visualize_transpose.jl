using CSV
using DataFrames
using Plots

# Read data and rename columns
naive_results     = CSV.read("./result/naive_results.csv", DataFrame)
block_results     = CSV.read("./result/block_results.csv", DataFrame)
recursive_results = CSV.read("./result/recursive_results.csv", DataFrame)

rename!(naive_results, Symbol("Time (s)") => :Time_s)
rename!(block_results, Symbol("Time (s)") => :Time_s)
rename!(recursive_results, Symbol("Time (s)") => :Time_s)

# Initialize an empty plot
plt = plot(
    title  = "Matrix Size vs. Time (Matrix transpose -03)",
    xlabel = "Matrix Size",
    ylabel = "Time (s)",
    legend = :left,
    size   = (800, 600),
)

# Plot Naive Implementation
plot!(
    plt,
    naive_results.Matrix_Size,
    naive_results.Time_s,
    label       = "Naive Implementation",
    seriestype  = :line,
    linestyle   = :solid,    
    lw          = 2,
    markershape = :circle,
    markersize  = 6,
    color       = :blue
)

# Plot Block Implementation
block_sizes = unique(block_results.Block_Size)
# Different colors for indentified different Block Size
block_palette = distinguishable_colors(length(block_sizes))

for (i, bs) in enumerate(block_sizes)
    subdf = block_results[block_results.Block_Size .== bs, :]
    plot!(
        plt,
        subdf.Matrix_Size,
        subdf.Time_s,
        label       = "Block Size = $bs",
        seriestype  = :line,
        linestyle   = :dot,     
        lw          = 2,
        markershape = :square,  
        markersize  = 6,
        color       = block_palette[i]
    )
end

# Plot Recursive Implementation

thresholds = unique(recursive_results.Threshold)
# Different colors for indentified different Threshold Size
threshold_palette = distinguishable_colors(length(thresholds))

for (j, thr) in enumerate(thresholds)
    subdf = recursive_results[recursive_results.Threshold .== thr, :]
    plot!(
        plt,
        subdf.Matrix_Size,
        subdf.Time_s,
        label       = "Threshold = $thr",
        seriestype  = :line,
        linestyle   = :dash,     # 虚线
        lw          = 2,
        markershape = :diamond,  # 菱形标记
        markersize  = 6,
        color       = threshold_palette[j]
    )
end

display(plt)
# savefig(plt, "matrix_transpose_02.png")
