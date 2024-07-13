module Ising

using Graphs, SimpleWeightedGraphs, LinearAlgebra, Random
using CircularArrays, OffsetArrays, StatsBase
using ProgressBars

global MAX_MEMORY = 5e9

export magnetization, updateSpin, updateSpace

function hamiltonian(selected_spin::CartesianIndex{3}, space::CircularArray, interaction::CircularArray, locs::Vector{CartesianIndex{3}})
# 3d version
# TODO: create type for locs for the function to not allocate anything
# OR: make them "CartesianIndices"

neighbour_spin = view(space,fill(selected_spin,length(locs)) + locs)
neighbour_interaction = view(interaction,fill(selected_spin,length(locs)) + locs)
H = (-dot(neighbour_interaction, neighbour_spin) * view(space, selected_spin))[1] # can be either 0,±2, or ±4

# μ = 0.1
# H -= 
# H += 

# calculate energy of neighbours

return H
end

function hamiltonian(selected_spin::CartesianIndex{2}, space::CircularArray, interaction::CircularArray, locs::Vector{CartesianIndex{2}})
# TODO: create type for locs for the function to not allocate anything
# OR: make them "CartesianIndices"

neighbour_spin = view(space,fill(selected_spin,length(locs)) + locs)
neighbour_interaction = view(interaction,fill(selected_spin,length(locs)) + locs)
H = (-dot(neighbour_interaction, neighbour_spin) * view(space, selected_spin))[1] # can be either 0,±2, or ±4

# μ = 0.1
# H -= 
# H += 

# calculate energy of neighbours

return H
end

function updateSpace(MAXITER, space::CircularArray, indices::CartesianIndices, interaction::CircularArray, locs::Vector{CartesianIndex{2}},β::Float64,_store_data=true)

if _store_data == false
  return updateSpace(MAXITER, space, indices,interaction,locs,β)
end

_space = copy(space)
magnetization = zeros(MAXITER)

selected_spin = CartesianIndex{2}
println("Allocating...")
evolution = zeros(Int16, size(space)..., MAXITER)
println("Allocated 3D array! (of size $(Base.format_bytes(sizeof(evolution))))")
@assert sizeof(evolution) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"

for ITER in ProgressBar(1:MAXITER)
    # selected_spin = 
  neighbours    = fill(rand(indices),length(locs)) + locs
  selected_spin = rand(indices)
  proposed_space = copy(_space)
  proposed_space[selected_spin] *= -1
  energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(interaction), Ref(locs)))
  energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(interaction), Ref(locs)))

  if energy_new < energy_old
    _space = proposed_space
  elseif exp(-β*(energy_new - energy_old)) > rand()
    _space= proposed_space
  else
    nothing
  end

  magnetization[ITER] = mean(_space)
  evolution[:,:,ITER] = _space.data
end
return evolution, magnetization
end

function updateSpace(MAXITER, space::CircularArray, indices::CartesianIndices, interaction::CircularArray, locs::Vector{CartesianIndex{3}},β::Float64,_store_data=true)

if _store_data == false
  return updateSpace(MAXITER, space, indices,interaction,locs,β)
end

_space = copy(space)
magnetization = zeros(MAXITER)

selected_spin = CartesianIndex{3}
println("Allocating...")
evolution = fill(zeros(size(space)...), MAXITER)
println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(evolution))))")
@assert sizeof(evolution) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"

for ITER in ProgressBar(1:MAXITER)
    # selected_spin = 
  neighbours    = fill(rand(indices),length(locs)) + locs
  selected_spin = rand(indices)
  proposed_space = copy(_space)
  proposed_space[selected_spin] *= -1
  energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(interaction), Ref(locs)))
  energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(interaction), Ref(locs)))

  if energy_new < energy_old
    _space = proposed_space
  elseif exp(-β*(energy_new - energy_old)) > rand()
    _space= proposed_space
  else
    nothing
  end

  magnetization[ITER] = mean(_space)
  evolution[ITER] = _space.data
end
return evolution, magnetization
end

function updateSpace(MAXITER, space::CircularArray, indices::CartesianIndices, interaction::CircularArray, locs::Vector{CartesianIndex{2}},β::Float64)
_space = copy(space)
magnetization = zeros(MAXITER)
selected_spin = CartesianIndex{2}

for ITER in ProgressBar(1:MAXITER)
    # selected_spin = 
  neighbours    = fill(rand(indices),length(locs)) + locs
  selected_spin = rand(indices)
  proposed_space = copy(_space)
  proposed_space[selected_spin] *= -1
  energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(interaction), Ref(locs)))
  energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(interaction), Ref(locs)))

  if energy_new < energy_old
    _space = proposed_space
  elseif exp(-β*(energy_new - energy_old)) > rand()
    _space= proposed_space
  else
    nothing
  end

  magnetization[ITER] = mean(_space)
end
return _space, magnetization
end

# 3D algorithm
function updateSpace(MAXITER, space::CircularArray, indices::CartesianIndices, interaction::CircularArray, locs::Vector{CartesianIndex{3}},β::Float64)
_space = copy(space)
magnetization = zeros(MAXITER)
selected_spin = CartesianIndex{3}

for ITER in ProgressBar(1:MAXITER)
    # selected_spin = 
  neighbours    = fill(rand(indices),length(locs)) + locs
  selected_spin = rand(indices)
  proposed_space = copy(_space)
  proposed_space[selected_spin] *= -1
  energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(interaction), Ref(locs)))
  energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(interaction), Ref(locs)))

  if energy_new < energy_old
    _space = proposed_space
  elseif exp(-β*(energy_new - energy_old)) > rand()
    _space= proposed_space
  else
    nothing
  end

  magnetization[ITER] = mean(_space)
end
return _space, magnetization
end

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
