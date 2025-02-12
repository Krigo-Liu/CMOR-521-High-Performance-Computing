using CSV
using DataFrames
using Plots

# Load CSV data
block_data = CSV.read("./src/block_sizes_results.csv", DataFrame)
threshold_data = CSV.read("./src/threshold_sizes_results.csv", DataFrame)
naive_data = CSV.read("./src/naive_results.csv", DataFrame)

# Plot Block Size vs. Execution Time
plot(block_data.Block_Size, block_data."Time (s)",
    xlabel="Block Size", ylabel="Execution Time (s)",
    title="Cache-Blocked Transpose Performance",
    lw=2, marker=:circle, label="Execution Time",
    legend=:topright)

# savefig("block_size_plot.png")  # Save the plot

# Plot Threshold Size vs. Execution Time
plot(threshold_data.Threshold_Size, threshold_data."Time (s)",
    xlabel="Threshold Size", ylabel="Execution Time (s)",
    title="Recursive Transpose Performance",
    lw=2, marker=:square, label="Execution Time",
    legend=:topright)

savefig("threshold_size_plot.png")  # Save the plot

# Plot Matrix Size vs. Execution Time
plot(naive_data.Matrix_Size, naive_data."Time (s)",
    xlabel="Matrix Size", ylabel="Execution Time (s)",
    title="Naive Transpose Performance",
    lw=2, marker=:diamond, label="Execution Time",
    legend=:topright)

savefig("naive_size_plot.png")  # Save the plot

println("Plots saved as 'block_size_plot.png' and 'threshold_size_plot.png'")
