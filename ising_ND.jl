using GLMakie
using ThreadsX
using CircularArrays
using Random: MersenneTwister
using LinearAlgebra
using StatsBase
using FisCom: Ising
using ProgressBars

begin
GC.gc()
NRUNS         = 20
Ndim          = (16,16)
rng           = MersenneTwister(253)
space         = CircularArray(sample(rng,[-1,1], Weights([0.59, 0.41]),Ndim) )
interaction   = CircularArray(ones(Ndim...))
indices       = eachindex(space)
basis_vectors = eachcol(diagm(ones(Int16,length(Ndim))))
neighbours    = map(_n->CartesianIndex(_n...), [basis_vectors; -basis_vectors])
end

begin
locs        = neighbours
BOLTZMANN   = 1.0
TEMP        = 1.91
BETA        = 1/(BOLTZMANN*TEMP)
TEMP_ARR    = LinRange(1.5, 3.5, 40)
MAXITER     = 5_000
# MAXITER     = 50000
β           = BETA
# S,M = Ising.updateSpace(MAXITER, space, indices, interaction,locs, BETA, false)

# _config = copy(space)
_energies = zeros(length(TEMP_ARR))
_mags = zeros(length(TEMP_ARR))
for (_ti,_tt) in enumerate(TEMP_ARR) # temperature
  E1 = M1 = 0.0
  _beta = 1.0 / _tt


  # for _ in ProgressBar(1:MAXITER*length(_config))# monte carlo steps
  #     _ind = rand(indices)
  #     _spin = _config[_ind]
  #     # calculate hamiltonian
  #     _neighbours = sum(_config[fill(_ind,length(locs)) + locs]) # get neighbouring spin
  #     _delta_energy = 2*_spin*_neighbours
  #     if _delta_energy < 0
  #       _spin *=-1
  #     elseif rand() < exp(-_delta_energy*_beta)
  #       _spin *=-1
  #     end
  #     _config[_ind] = _spin
  #     Mag = sum(_config) # can be inside or outside
  #     M1 = M1 + Mag
  #   # end
  # end
  M1, E1 = Ising.updateSpace(copy(space), locs, _beta, MAXITER)

  # _energy = 0.0
  # for _ind in eachindex()
  #   _spin = _config[_ind]
  #   _neighbours = sum(_config[fill(_ind,length(locs)) + locs]) # get neighbouring spin
  #   _energy += -_neighbours * _spin
  #   E1 += _energy / 4.0
  # end1
#   Mag = sum(_config)
#   M1 = M1 + Mag
  # end

  # _energies[_ti] = E1
  _mags[_ti] = M1
  # display(heatmap(_config))
end
end
begin

fig = Figure()
plot(fig[1,1],TEMP_ARR, abs.(_mags))
# plot(fig[2,1],TEMP_ARR , _energies/ MAXITER / length(space) / length(space))
fig
# @allocated M = ThreadsX.map((i)->last(update_space(MAXITER, space, indices, interaction,locs, BETA)), 1:NRUNS)
# volume(S.data, algorithm=:absorption,absorption=1f0, interpolate=false,colormap=:bone)
end


let fig = Figure()
  ax = Axis(fig[1,1])
  lines!(ax,last.(last.(result)))

  fig
end

lines(TEMP_ARR,last.(result))

let 
  fig = Figure()
  ax=[Axis(fig[1,1]), Axis(fig[1,2])]
  heatmap!(ax[1],result[end][1],colormap=:bone)
  fig
end
lines(M)

using JLD2

# @save "./data/ising_3d.jld" S

begin
  GC.gc()
  fig = Figure()
  sl = Slider(fig[2,1], range=range(1,MAXITER,step=1))
  ax = Axis(fig[1,1])
  # data = eachslice(S, dims=4)
  # hm=volume!(ax,S[1],interpolate=false, algorithm=:absorption,absorption=7f0,colormap=:afmhot); ax.aspect=Ndim; ax.perspectiveness = 0.8
  hm=heatmap!(ax,S[1],interpolate=false, colormap=:afmhot); ax.aspect=DataAspect()
  # record(fig, "ising_3d_nz=5.mp4", 1:100:MAXITER, framerate=60) do _i
  lift(sl.value) do _i
    hm[3][] = S[_i % MAXITER + 1]
    # ax.azimuth = _i / MAXITER 
    # _i += 100
    # sleep(0.00000000001)
  end
  fig
end

BETA