using CSV
using DataFrames
using Plots

# 1. 读取数据并改列名
naive_results_mm     = CSV.read("./result/naive_results_mm.csv", DataFrame)
block_results_mm     = CSV.read("./result/block_results_mm.csv", DataFrame)
recursive_results_mm = CSV.read("./result/recursive_results_mm.csv", DataFrame)

rename!(naive_results_mm, Symbol("Time(s)") => :Time_s)
rename!(block_results_mm, Symbol("Time(s)") => :Time_s)
rename!(recursive_results_mm, Symbol("Time(s)") => :Time_s)

# 2. 初始化空图
plt = plot(
    title  = "Matrix Size vs. Time (Matrix multiplication O3)",
    xlabel = "Matrix Size",
    ylabel = "Time (s)",
    legend = :left,
    size   = (800, 600),
)

# --------------------
# 3. 绘制 Naive (单条线，实线、单色、圆圈)
# --------------------
plot!(
    plt,
    naive_results_mm.Matrix_Size,
    naive_results_mm.Time_s,
    label       = "Naive Implementation",
    seriestype  = :line,
    linestyle   = :solid,    # 实线
    lw          = 2,
    markershape = :circle,
    markersize  = 6,
    color       = :blue
)

# --------------------
# 4. 绘制 Block Results (点线，不同 Block Size 不同颜色、相同标记)
# --------------------
block_sizes = unique(block_results_mm.Block_Size)
# 准备一组颜色用于区分 Block Size
block_palette = distinguishable_colors(length(block_sizes))

for (i, bs) in enumerate(block_sizes)
    subdf = block_results_mm[block_results_mm.Block_Size .== bs, :]
    plot!(
        plt,
        subdf.Matrix_Size,
        subdf.Time_s,
        label       = "Block Size = $bs",
        seriestype  = :line,
        linestyle   = :dot,      # 点线
        lw          = 2,
        markershape = :square,   # 方形标记
        markersize  = 6,
        color       = block_palette[i]
    )
end

# --------------------
# 5. 绘制 Recursive Results (虚线，不同 Threshold 不同颜色、相同标记)
# --------------------
thresholds = unique(recursive_results_mm.Threshold)
# 准备一组颜色用于区分 Threshold
threshold_palette = distinguishable_colors(length(thresholds))

for (j, thr) in enumerate(thresholds)
    subdf = recursive_results_mm[recursive_results_mm.Threshold .== thr, :]
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
# savefig(plt, "all_methods_line_style.png")
