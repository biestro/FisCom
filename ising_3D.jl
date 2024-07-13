using GLMakie
# using ThreadsX
using CircularArrays
using Random: MersenneTwister
using StatsBase
using FisCom: Ising

begin
GC.gc()
NRUNS       = 20
Nx, Ny, Nz  = 20,20,5
rng         = MersenneTwister(253)
space       = CircularArray(sample(rng,[-1,1], Weights([0.59, 0.41]),(Nx,Ny,Nz)) )
interaction = CircularArray(ones(Nx,Ny,Nz))
indices     = eachindex(space)
locs        = CartesianIndex.([-1,0,0,1,0,0],[0,1,-1,0,0,0],[0,0,0,0,1,-1]) # 3D grid neighbours
BOLTZMANN   = 1.0
TEMP        = 1.0
BETA        = 1/(BOLTZMANN*TEMP)
MAXITER     = 50_000
# MAXITER     = 50000
β           = BETA


S,_ = Ising.updateSpace(MAXITER, space, indices, interaction,locs, BETA, true)
GC.gc()
# @allocated M = ThreadsX.map((i)->last(update_space(MAXITER, space, indices, interaction,locs, BETA)), 1:NRUNS)
# volume(S.data, algorithm=:absorption,absorption=1f0, interpolate=false,colormap=:bone)
end

using JLD2

# @save "./data/ising_3d.jld" S

begin
  GC.gc()
  fig = Figure()
#   sl = Slider(fig[2,1], range=range(1,MAXITER,step=1))
  ax = Axis3(fig[1,1])
  # data = eachslice(S, dims=4)
  hm=volume!(ax,S[1],interpolate=false, algorithm=:absorption,absorption=7f0,colormap=:afmhot)
  ax.aspect=(Nx,Ny,Nz)
  ax.perspectiveness = 0.8
  record(fig, "ising_3d_nz=5.mp4", 1:100:MAXITER, framerate=60) do _i
  # lift(sl.value) do _i
    hm[4][] = S[_i % MAXITER + 1]
    ax.azimuth = _i / MAXITER 
    # _i += 100
    # sleep(0.00000000001)
  end
  fig
end