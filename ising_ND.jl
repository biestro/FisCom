using GLMakie
using ThreadsX
using CircularArrays
using Random: MersenneTwister
using LinearAlgebra: diagm
using StatsBase: Weights, sample
using ProgressBars

using FisCom: Ising # main module

begin
GC.gc()
Ndim          = (2,2).^4  # N dimensions, can be any N-dim tuple
rng           = MersenneTwister(253)
space         = CircularArray(sample(rng,[-1,1], Weights([0.5, 0.5]),Ndim) )
interaction   = CircularArray(ones(Ndim...))
indices       = eachindex(space)
basis_vectors = eachcol(diagm(ones(Int8,length(Ndim))))
neighbours    = map(_n->CartesianIndex(_n...), [basis_vectors; -basis_vectors])
locs          = neighbours
BOLTZMANN     = 1.0
TEMP_ARR      = LinRange(1.5, 3.5, 40)
MAXITER       = 1_000_000
# MAXITER       = 50000
end

@time _mags = ThreadsX.map(_temp->Ising.updateSpace(copy(space), locs, 1.0 / (BOLTZMANN * _temp ), MAXITER), TEMP_ARR)
# _mags = [Ising.updateSpace(copy(space), locs, 1.0 / (BOLTZMANN * _temp ), MAXITER) for _temp in ProgressBar(TEMP_ARR)]

# Plot
let
fig = Figure()
plot(fig[1,1],TEMP_ARR, abs.(_mags))
fig
end


# begin
#   GC.gc()
#   fig = Figure()
#   sl = Slider(fig[2,1], range=range(1,MAXITER,step=1))
#   ax = Axis(fig[1,1])
#   # data = eachslice(S, dims=4)
#   # hm=volume!(ax,S[1],interpolate=false, algorithm=:absorption,absorption=7f0,colormap=:afmhot); ax.aspect=Ndim; ax.perspectiveness = 0.8
#   hm=heatmap!(ax,S[1],interpolate=false, colormap=:afmhot); ax.aspect=DataAspect()
#   # record(fig, "ising_3d_nz=5.mp4", 1:100:MAXITER, framerate=60) do _i
#   lift(sl.value) do _i
#     hm[3][] = S[_i % MAXITER + 1]
#     # ax.azimuth = _i / MAXITER 
#     # _i += 100
#     # sleep(0.00000000001)
#   end
#   fig
# end