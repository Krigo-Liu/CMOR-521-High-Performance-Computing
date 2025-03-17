using Plots

# --------------------------------------------------------------------------------
# Device and Performance Parameters
# --------------------------------------------------------------------------------

# According to your lscpu output, your system has 16 cores in total 
# (1 core per socket, 16 sockets). Set the theoretical single-core peak performance.
# NOTE: Replace 35.2 with your actual measured or calculated GFLOPS per core if available.
peak_single = 35.2  # GFLOPS per core (example value)

# Memory parameters:
# These are based on your memory's operating frequency and architecture.
# memory_speed: Memory frequency in MHz (example: 2933 MHz)
# num_memory_channels: Number of memory channels (example: 6)
# data_width: The width of each memory channel in bytes (example: 8 bytes)
memory_speed = 2933          # in MHz
num_memory_channels = 6      
data_width = 8               # in bytes

# Calculate the theoretical peak memory bandwidth (in GB/s)
# The formula is: memory_speed * num_memory_channels * data_width / 1000
# Often it's better to use measured values (e.g., via STREAM benchmark). 
peak_bandwidth = memory_speed * num_memory_channels * data_width / 1000  
# Overwrite with a measured value if available (example value: 131.13 GB/s)

# --------------------------------------------------------------------------------
# Define the Computational Intensity (CI) Range
# --------------------------------------------------------------------------------
# CI is defined as FLOPs per byte of data. Adjust the range if needed.
CI = LinRange(0, 3.0, 1000)

# --------------------------------------------------------------------------------
# Define Thread Configurations and Plot the Roofline
# --------------------------------------------------------------------------------
# We want to generate roofline plots for 1, 2, and 8 threads.
thread_counts = [1, 2, 8]

# Initialize an empty plot.
plot()

# Loop through each thread configuration.
for t in thread_counts
    # Calculate the theoretical peak performance for t threads.
    # For example, for t threads, the peak performance is t times the single-core peak.
    peak_perf = peak_single * t

    # The roofline model for each CI value is the minimum of:
    # - The compute bound (peak_perf)
    # - The memory bound (CI * peak_bandwidth)
    roofline = @. min(peak_perf, CI * peak_bandwidth)

    # Plot the roofline curve for this thread count.
    plot!(CI, roofline, label="Roofline for $t thread(s)")
end

# Add plot labels and title.
xlabel!("Computational Intensity (FLOPs/Byte)", fontsize=14)
ylabel!("Performance (GFLOPS/s)", fontsize=14)
title!("Roofline Model for 1, 2, and 8 Threads", fontsize=16)

# Optionally, save the generated plot to a file.
savefig("./docs/roofline_plot.png")
