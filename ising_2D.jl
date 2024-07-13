using GLMakie
# using ThreadsX
using CircularArrays
using Random: MersenneTwister
using StatsBase

include("./src/Ising.jl")

begin
GC.gc()

NRUNS       = 20
N           = 40
rng         = MersenneTwister(253)
space       = CircularArray(sample(rng,[-1,1], Weights([0.59, 0.41]),(N,N)) )
interaction = CircularArray(ones(N,N))
indices     = eachindex(space)
locs        = CartesianIndex.([-1,0,0,1],[0,1,-1,0])
BOLTZMANN   = 1.0
TEMP        = 1.0
BETA        = 1/(BOLTZMANN*TEMP)
MAXITER     = 10_000
# MAXITER     = 50000
β           = BETA

S,_ = Ising.updateSpace(MAXITER, space, indices, interaction,locs, BETA,true)
# @allocated M = ThreadsX.map((i)->last(update_space(MAXITER, space, indices, interaction,locs, BETA)), 1:NRUNS)
end

begin
  fig = Figure()
  sl = Slider(fig[2,1], range=range(1,MAXITER,step=1))
  ax = Axis(fig[1,1])
  ax.aspect=DataAspect()
  data = eachslice(S, dims=3)
  hm=heatmap!(ax,data[1],colormap=:bone)
  lift(sl.value) do _i
    hm[3][] = data[_i]
  end
  fig
end