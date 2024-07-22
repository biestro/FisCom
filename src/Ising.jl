module Ising

# TODO: do a monte carlo step each time? or check all gridpoints per time
# TODO: add interaction
# TODO: add parametric types for CartesianIndex{N} and CircularArray{N} (for optimization)
# TODO: convolve the lattice to find its energy (set the mode to wrap as in python)
# TODO: use a lookup table or an OffsetArray to find energy (i.e. arr[-4] = exp(-4)) 
# TODO: change the conditional into branchless computation (not really that important, check the compiler)

using Graphs, SimpleWeightedGraphs, LinearAlgebra, Random
using CircularArrays, OffsetArrays, StatsBase
using OffsetArrays
using ProgressBars

global MAX_MEMORY = 5e9
global xoshiro_rng = Xoshiro(1111)

export updateSpace

# struct IntCartesianIndex{T<:Int64}
#   _ord::T
# end

""" function updateSpace

# Arguments:

- `_config`: configuration space of the problem (N-dimensional)
- `_locs`: location of nearest neighbours, given as a vector of `CartesianIndex`.
- `_beta`: inverse temperature parameter β = 1/k_B*T

"""
function updateSpace(_config::CircularArray, _locs::Vector{T}, _beta::Float64, _MAXITER::Int64) where T <: CartesianIndex

indices = eachindex(_config)
magnet  = 0.0 # will store all magnetization
spin = Int16
# dims = length(size(_config)) # dimensions of model
# delta_energy_vals = exp.(-4*(-dims:dims)*_beta)
# energy_table = Dict(zip(4*(-dims:dims),delta_energy_vals)) # value of neighbours
# println(energy_table)

# for _ in 1:_MAXITER*length(indices)# monte carlo steps times each axis dimension
for _ in 1:_MAXITER# monte carlo steps times each axis dimension

  ind = rand(indices)   # random index selection
  spin = _config[ind]  # spin at random index σᵢ
  
  neighbours = sum(_config[fill(ind,length(_locs)) + _locs]) # get nearest neighbours (locs) σᵢ * (σ₁+σ₂+...+σⱼ)
  delta_energy = 2*spin*neighbours

  spin *= 1 - (((delta_energy < 0) || (rand() < exp(-delta_energy*_beta))) * 2) # takes same time and allocations as using a lookup table
  # spin *= 1 - (((delta_energy < 0) || (rand() < energy_table[delta_energy])) * 2) # same as below
  # spin *= 1 - (((delta_energy < 0) || (rand() < delta_energy_vals[delta_energy÷4 + dims + 1])) * 2) # same as below

  #=
  if delta_energy < 0
    spin *=-1
  elseif rand() < exp(-delta_energy*_beta) # assert the rng to be xoshiro256
    spin *=-1
  end
  =#

  _config[ind] = spin  #  update spin at location
  magnet = magnet + sum(_config) # update magnetization
  # energy = fun(space)
end
return magnet / _MAXITER / length(indices) # average magnetization averaged over time (same as length(_config)) (normalized)
end


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


# """
# N-dimensional version of the ising hamiltonian with an N-dimensional location vector
# """
# function hamiltonian(selected_spin::T, space::CircularArray, interaction::CircularArray, locs::Vector{T}) where T<:CartesianIndex
# function hamiltonian(selected_spin::T, space::CircularArray, locs::Vector{T}) where T<:CartesianIndex
# # TODO: create type for locs for the function to not allocate anything
# # OR: make them "CartesianIndices"

# # interaction: weights (similar to graph version)
# neighbour_spin = view(space,fill(selected_spin,length(locs)) + locs)
# # neighbour_interaction = view(interaction,fill(selected_spin,length(locs)) + locs) # if interaction is the same, this is just an array of 1.0s
# neighbour_interaction = ones(length(locs))# (selected_spin,length(locs)) + locs) # if interaction is the same, this is just an array of 1.0s
# H = (-dot(neighbour_interaction, neighbour_spin) * view(space, selected_spin))[1] # can be either 0,±2, or ±4

# # μ = 0.1
# # H -= 
# # H += 

# # calculate energy of neighbours

# return H
# end

# """
# Runs Lenz-Ising model, *saving* each configuration at each step (for visualization)
# """
# function updateSpace(MAXITER, space::CircularArray, indices::CartesianIndices, interaction::CircularArray, locs::Vector{T},β::Float64,_store_data=true) where T <: CartesianIndex
# GC.gc()

# if _store_data == false
#   return updateSpace(MAXITER, space, indices,interaction,locs,β)
# end

# _space = copy(space)
# magnetization = zeros(MAXITER)

# selected_spin = CartesianIndex{3}
# println("Allocating...")
# evolution = fill(zeros(size(space)...), MAXITER)
# println("Allocated $(length(size(indices))+1)D array! (of size $(Base.format_bytes(sizeof(evolution))))")
# @assert sizeof(evolution) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"

# for ITER in ProgressBar(1:MAXITER)
#     # selected_spin = 
#   neighbours    = fill(rand(indices),length(locs)) + locs
#   selected_spin = rand(indices)
#   proposed_space = copy(_space)
#   proposed_space[selected_spin] *= -1
#   # energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(interaction), Ref(locs)))
#   # energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(interaction), Ref(locs)))
#   energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(locs)))
#   energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(locs)))

#   if energy_new < energy_old
#     _space = proposed_space
#   elseif exp(-β*(energy_new - energy_old)) > rand()
#     _space= proposed_space
#   else
#     nothing
#   end

#   magnetization[ITER] = mean(_space)
#   evolution[ITER] = _space.data
# end
# return evolution, magnetization
# end

# """

# Runs Lenz-Ising model, *without saving* each configuration at each step (for visualization)
# Still saves magnetization vector
# """
# function updateSpace(MAXITER, space::CircularArray, indices::CartesianIndices, interaction::CircularArray, locs::Vector{T},β::Float64) where T <: CartesianIndex
# println("Using General model (N-dim)")
# # GC.gc()

# # _magnetization = 0.0
# _space = copy(space)
# # _magnetization = zeros(MAXITER)

# selected_spin = CartesianIndex{3}
# # println("Allocating...")
# # evolution = fill(zeros(size(space)...), MAXITER)
# # println("Allocated $(length(size(indices))+1)D array! (of size $(Base.format_bytes(sizeof(evolution))))")
# # @assert sizeof(evolution) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"

# _mag = 0.0
# for ITER in ProgressBar(1:MAXITER)
# # for ITER in 1:MAXITER
#   for _ in axes(space,1)
#     # selected_spin = 
#   selected_spin = rand(indices)
#   # _neighbours    = sum(_space[fill(selected_spin,length(locs)) + locs]) # get neighbouring spin
#   neighbours    = fill(selected_spin,length(locs)) + locs
#   # _neighbours = _space[selected_spin + CartesianIndex(1,0)] + 
#   #               _space[selected_spin - CartesianIndex(1,0)] + 
#   #               _space[selected_spin + CartesianIndex(0,1)] + 
#   #               _space[selected_spin - CartesianIndex(0,1)];
#   # s = _space[selected_spin]
#   proposed_space = copy(_space)
#   proposed_space[selected_spin] *= -1
#   # nb = sum(space[neighbours])
#   # cost = 2 * s * _neighbours
#   # energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(interaction), Ref(locs)))
#   # energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(interaction), Ref(locs)))
#   energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(locs)))
#   energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(locs)))
#   # cost = energy_new - energy_old


#   if energy_new < energy_old
#     _space .= proposed_space
#   elseif exp(-β*(energy_new - energy_old)) > rand()
#     _space .= proposed_space
#   else
#     nothing
#   end
#   # end for loop
#   end
#   # 
#   _mag += sum(_space)
#   # end time for loop
# end
# return _space, _mag# netization# magnetization
# end


# function hamiltonian(selected_spin::CartesianIndex{3}, space::CircularArray, interaction::CircularArray, locs::Vector{CartesianIndex{3}})
# # 3d version
# # TODO: create type for locs for the function to not allocate anything
# # OR: make them "CartesianIndices"
# 
# neighbour_spin = view(space,fill(selected_spin,length(locs)) + locs)
# neighbour_interaction = view(interaction,fill(selected_spin,length(locs)) + locs)
# H = (-dot(neighbour_interaction, neighbour_spin) * view(space, selected_spin))[1] # can be either 0,±2, or ±4
# 
# # μ = 0.1
# # H -= 
# # H += 
# 
# # calculate energy of neighbours
# 
# return H
# end

# function hamiltonian(selected_spin::CartesianIndex{2}, space::CircularArray, interaction::CircularArray, locs::Vector{CartesianIndex{2}})
# # TODO: create type for locs for the function to not allocate anything
# # OR: make them "CartesianIndices"
# 
# neighbour_spin = view(space,fill(selected_spin,length(locs)) + locs)
# neighbour_interaction = view(interaction,fill(selected_spin,length(locs)) + locs)
# H = (-dot(neighbour_interaction, neighbour_spin) * view(space, selected_spin))[1] # can be either 0,±2, or ±4
# 
# # μ = 0.1
# # H -= 
# # H += 
# 
# # calculate energy of neighbours
# 
# return H
# end

# function updateSpace(MAXITER, space::CircularArray, indices::CartesianIndices, interaction::CircularArray, locs::Vector{CartesianIndex{2}},β::Float64,_store_data=true)
# 
# if _store_data == false
#   return updateSpace(MAXITER, space, indices,interaction,locs,β)
# end
# 
# _space = copy(space)
# magnetization = zeros(MAXITER)
# 
# selected_spin = CartesianIndex{2}
# println("Allocating...")
# evolution = zeros(Int16, size(space)..., MAXITER)
# println("Allocated 3D array! (of size $(Base.format_bytes(sizeof(evolution))))")
# @assert sizeof(evolution) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"
# 
# for ITER in ProgressBar(1:MAXITER)
#     # selected_spin = 
#   neighbours    = fill(rand(indices),length(locs)) + locs
#   selected_spin = rand(indices)
#   proposed_space = copy(_space)
#   proposed_space[selected_spin] *= -1
#   energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(interaction), Ref(locs)))
#   energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(interaction), Ref(locs)))
# 
#   if energy_new < energy_old
#     _space = proposed_space
#   elseif exp(-β*(energy_new - energy_old)) > rand()
#     _space= proposed_space
#   else
#     nothing
#   end
# 
#   magnetization[ITER] = mean(_space)
#   evolution[:,:,ITER] = _space.data
# end
# return evolution, magnetization
# end

# function updateSpace(MAXITER, space::CircularArray, indices::CartesianIndices, interaction::CircularArray, locs::Vector{CartesianIndex{3}},β::Float64,_store_data=true)
# 
# if _store_data == false
#   return updateSpace(MAXITER, space, indices,interaction,locs,β)
# end
# 
# _space = copy(space)
# magnetization = zeros(MAXITER)
# 
# selected_spin = CartesianIndex{3}
# println("Allocating...")
# evolution = fill(zeros(size(space)...), MAXITER)
# println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(evolution))))")
# @assert sizeof(evolution) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"
# 
# for ITER in ProgressBar(1:MAXITER)
#     # selected_spin = 
#   neighbours    = fill(rand(indices),length(locs)) + locs
#   selected_spin = rand(indices)
#   proposed_space = copy(_space)
#   proposed_space[selected_spin] *= -1
#   # energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(interaction), Ref(locs)))
#   # energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(interaction), Ref(locs)))
#   energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(locs)))
#   energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(locs)))
# 
#   if energy_new < energy_old
#     _space = proposed_space
#   elseif exp(-β*(energy_new - energy_old)) > rand()
#     _space= proposed_space
#   else
#     nothing
#   end
# 
#   magnetization[ITER] = mean(_space)
#   evolution[ITER] = _space.data
# end
# return evolution, magnetization
# end

# function updateSpace(MAXITER, space::CircularArray, indices::CartesianIndices, interaction::CircularArray, locs::Vector{CartesianIndex{2}},β::Float64)
# _space = copy(space)
# magnetization = zeros(MAXITER)
# selected_spin = CartesianIndex{2}
# 
# for ITER in ProgressBar(1:MAXITER)
#     # selected_spin = 
#   neighbours    = fill(rand(indices),length(locs)) + locs
#   selected_spin = rand(indices)
#   proposed_space = copy(_space)
#   proposed_space[selected_spin] *= -1
#   energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(interaction), Ref(locs)))
#   energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(interaction), Ref(locs)))
# 
#   if energy_new < energy_old
#     _space = proposed_space
#   elseif exp(-β*(energy_new - energy_old)) > rand()
#     _space= proposed_space
#   else
#     nothing
#   end
# 
#   magnetization[ITER] = mean(_space)
# end
# return _space, magnetization
# end
# 
# # 3D algorithm
# function updateSpace(MAXITER, space::CircularArray, indices::CartesianIndices, interaction::CircularArray, locs::Vector{CartesianIndex{3}},β::Float64)
# _space = copy(space)
# magnetization = zeros(MAXITER)
# selected_spin = CartesianIndex{3}
# 
# for ITER in ProgressBar(1:MAXITER)
#     # selected_spin = 
#   neighbours    = fill(rand(indices),length(locs)) + locs
#   selected_spin = rand(indices)
#   proposed_space = copy(_space)
#   proposed_space[selected_spin] *= -1
#   energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(interaction), Ref(locs)))
#   energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(interaction), Ref(locs)))
# 
#   if energy_new < energy_old
#     _space = proposed_space
#   elseif exp(-β*(energy_new - energy_old)) > rand()
#     _space= proposed_space
#   else
#     nothing
#   end
# 
#   magnetization[ITER] = mean(_space)
# end
# return _space, magnetization
# end

function neighbour_weights(g::SimpleWeightedGraph, inode::Int)
  # equivalent to w_ij of Ising model
  return view(g.weights,inode,neighbors(g,inode))
end

function hamiltonian(h::SimpleWeightedGraph, spin::Vector)
  # full hamiltonian of Ising lattice, used for testing only
  return -dot.(neighbour_weights.(Ref(h), vertices(h)), view.(Ref(spin), neighbors.(Ref(h),vertices(h))) .* spin[vertices(h)])
end

function efficient_hamiltonian(h::SimpleWeightedGraph, spin::Vector, idx::Int)
  # calculate hamiltonian for σ at idx
  # to use neighbors -> neighborhood, weights have to represent
  # distance, and one then can choose a geodesic radius
  return -dot(neighbour_weights(h, idx), view(spin, neighbors(h,idx)))*spin[idx]
end

function magnetization(spin::Vector)
  # calculate magnetization (number of "up" spins)
  return sum(spin) / length(spin)
end

function update_spin(β::Float64,  graph::SimpleWeightedGraph,orig_spin::Vector)
  # runs ising process
  spin_vec = copy(orig_spin)
  node_to_flip = rand(1:nv(graph))

  # β  = 0.1
  # old_energy_full = sum(hamiltonian(graph, spin_vector))
  old_energy = sum(efficient_hamiltonian.(Ref(graph), Ref(spin_vec), [neighbors(graph, node_to_flip); node_to_flip]))

  spin_vec[node_to_flip] *= -1

  # new_energy_full = sum(hamiltonian(graph, spin_vec))
  new_energy = sum(efficient_hamiltonian.(Ref(graph), Ref(spin_vec), [neighbors(graph, node_to_flip); node_to_flip]))

  if (new_energy - old_energy) < 0 # accept new configuration
    return spin_vec
  elseif exp(-β*(new_energy - old_energy)) > rand() # accept new configuration
    return spin_vec
  else # reject changes
    return orig_spin
  end
end


end # module ising
