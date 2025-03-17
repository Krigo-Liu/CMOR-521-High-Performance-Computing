# calculates a roofline plot for axpy on an Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz

# peak performance in GFLOPS/sec
num_cores = 16
peak_performance = 352 / num_cores # GFLOPS/sec

# peak BW should be between 131.13 and 140 GB/second
memory_speed = 2933 # MHz; bits per second
num_memory_channels = 6 
data_width = 8 # size of memory bus in bytes; see also cache_alignment

# units of GB/seconds
peak_bandwidth = memory_speed * num_memory_channels * 
    data_width / 1000
peak_bandwidth = 131.13 # measured estimate GB/s

using Plots
CI = LinRange(0, .25, 1000)
roofline = @. min(peak_performance, CI * peak_bandwidth)
plot(CI, roofline)
xlabel!("Computational intensity (CI)", fontsize=14)
ylabel!("GFLOPS / second", fontsize=14)

CI_add_vec = 2 / (2 * 8)
n = [1000000000, 1000000000]
timings = [2218, 4419] * 1e-9 # in seconds
timings_blas = [1523,  2836] * 1e-9 # in seconds
num_gflops = 2 * n * 1e-9
scatter!(CI_add_vec * ones(length(timings)), num_gflops ./ timings, label="addvec")
scatter!(CI_add_vec * ones(length(timings)), num_gflops ./ timings_blas, label="BLAS")

# APP_peak_performance = .1056 * 1000 / .3 # .3 = weighting factor for non-vector process