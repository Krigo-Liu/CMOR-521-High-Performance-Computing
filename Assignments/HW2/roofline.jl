using Plots

# --------------------------------------------------------------------------------
# Device and Performance Parameters
# --------------------------------------------------------------------------------

# According to your lscpu output, your system has 16 cores in total 
num_cores = 16
peak_performance = 352 / num_cores # GFLOPS/sec

# peak BW should be between 131.13 and 140 GB/second
memory_speed = 2933          # in MHz
num_memory_channels = 6      
data_width = 8               # in bytes

# Calculate the theoretical peak memory bandwidth (in GB/s)
peak_bandwidth = memory_speed * num_memory_channels * data_width / 1000  

# CI is defined as FLOPs per byte of data. Adjust the range if needed.
CI = LinRange(0, .25, 1000)
roofline = @. min(peak_performance, CI * peak_bandwidth)
plot(CI, roofline)
xlabel!("Computational Intensity (FLOPs/Byte)", fontsize=14)
ylabel!("Performance (GFLOPS/s)", fontsize=14)
title!("Roofline Model for 1, 2, and 8 Threads", fontsize=16)

CI_add_vec = 2 / (2 * 8)
n = [1000000000, 1000000000, 1000000000]
# We want to generate roofline plots for 1, 2, and 8 threads.
timings_v1 = [5.94938, 1.72952, 1.16535]# in seconds
timings_v2 = [2.67886, 1.45018, 0.464915]# in seconds
num_gflops = 2 * n * 1e-9
scatter!(CI_add_vec * ones(length(timings_v1)), num_gflops ./ timings_v1, label="AXPY_v1")
scatter!(CI_add_vec * ones(length(timings_v2)), num_gflops ./ timings_v2, label="AXPY_v2")


# Optionally, save the generated plot to a file.
savefig("./docs/roofline_plot.png")



# Calculate performance (GFLOPS/s)
performance_v1 = num_gflops ./ timings_v1
performance_v2 = num_gflops ./ timings_v2

# Calculate percentage of peak performance
percentage_v1 = performance_v1 ./ peak_performance .* 100
percentage_v2 = performance_v2 ./ peak_performance .* 100

# Create a DataFrame for clarity (using DataFrames.jl)
using DataFrames

performance_df = DataFrame(
    Threads = [1, 2, 8],
    Performance_v1_GFLOPS_s = performance_v1,
    Percentage_v1 = percentage_v1,
    Performance_v2_GFLOPS_s = performance_v2,
    Percentage_v2 = percentage_v2
)

println(performance_df)
