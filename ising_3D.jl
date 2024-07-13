using GLMakie
# using ThreadsX
using CircularArrays
using Random: MersenneTwister
using StatsBase
using FisCom: Ising

begin
GC.gc()
NRUNS       = 20
Nx, Ny, Nz  = 50,50,9
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
# @allocated M = ThreadsX.map((i)->last(update_space(MAXITER, space, indices, interaction,locs, BETA)), 1:NRUNS)
# volume(S.data, algorithm=:absorption,absorption=1f0, interpolate=false,colormap=:bone)
end

begin
  fig = Figure()
  sl = Slider(fig[2,1], range=range(1,MAXITER,step=1))
  pl = PointLight(Point3f(0,0,N), RGBf(4, 4, 4))
  lscene = LScene(fig[1, 1], show_axis=false, scenekw = (lights = [pl], backgroundcolor=:white, clear=true))
  # data = eachslice(S, dims=4)
  hm=volume!(lscene,S[1],interpolate=false, algorithm=:absorption,colormap=:bone)
  # hm=volume!(lscene,data[1],interpolate=false, algorithm=:iso,isovalue=1.0,colormap=:bone,colorrange=(-0,20))
  lift(sl.value) do _i
    hm[4][] = S[_i]
  end
  fig
end