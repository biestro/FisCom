using GLMakie
using Combinatorics: combinations
using ImageFiltering


function finite_diff_coefficient(_ord::Int64)
  if _ord == 2
    return [1,-2,1]
  elseif _ord == 4
    return [-1/12, 4/3, -5/2, 4/3, -1/12]
  elseif _ord == 6
    return [1/90, -3/20, 3/2, -49/18,3/2,-3/20,1/90]
  elseif _ord == 8
    return [-1/560,8/315,-1/5,8/5,-205/72,8/5,-1/5,8/315,-1/560]
  end
end

function get_laplace_kernel(_dim::Int64,_ord::Int64)
  base_ker = finite_diff_coefficient(_ord)
  kernel = zeros(fill(length(base_ker),_dim)...)
  mid_index = _ord÷2+1
  kernel[fill(mid_index,ndims(kernel)-1)...,:] .= base_ker # middle of middles
  
  perm_ind = collect(permutations(1:_dim))#[1:_dim-1:end]
  if _dim == 3
  # filter!(_i -> levicivita(_i) < 0, perm_ind)
  filter!(_i -> levicivita(_i) < 0, perm_ind)
  end
  kernel = mapreduce(_i -> permutedims(kernel,_i),+,perm_ind)
  

  #two_dim_ker += permutedims(two_dim_ker)
  return centered(kernel) # return centered version of kernel (0 = middle)
end


begin
  L          = 5.0
  Ndims      = (51,51,51);
  Coords     = LinRange.(-L,L,Ndims);
  speed      = 0.2;
  space_step = 1.0; # spatial width
  time_step  = 0.5; # time step width (remember, dt < dx^2/2)?
  velocities = ones(Ndims...) * speed # velocities 
  KAPPA      = copy(velocities) * time_step/space_step  # 
  ALPHA      = (velocities*time_step/space_step).^2; # wave equation coefficient (α^2)
  MAXITER    = 2500

  # ALPHA[nx÷3:nx÷3+10,ny÷3:ny÷3+2] .*= 0.0
  # ALPHA[2nx÷3:2nx÷3+10,2ny÷3:2ny÷3+2] .*= 0.0
  U = zeros(Ndims...); # old
  V = [exp(-5*((x-L/2)^2+(y-L/2)^2+(z-L/2)^2)) for x in Coords[1], y in Coords[2], z in Coords[3]] # actual
  # V += [exp(-5*((x+L/2)^2+(y+L/2)^2)) for x in Coords[1], y in Coords[2]] # actual
  W = zeros(Ndims...); # new 

  order = 2 
  wave_kernel = centered(get_laplace_kernel(length(Ndims),order))
  wave_array = fill(zeros(size(U)...), MAXITER)
end;

begin
wave_old = copy(U)
wave_new = copy(W)
wave_act = copy(V)
for _i in 1:MAXITER
  
  # update wave
 
  # wave_new = ALPHA .* imfilter(wave_act, wave_kernel, Fill(0,wave_kernel)) + 2*wave_act - wave_old  # specify Fill(0,kernel) for Dirichlett conditions (boundary = 0)
  wave_new = ALPHA .* imfilter(wave_act, wave_kernel, "circular") + 2*wave_act - wave_old

  wave_array[_i] = wave_new # store wave
  
  # update values

  wave_old = wave_act
  wave_act = wave_new
  
end
end

let
  GC.gc()
  fig = Figure()
  # sl = Slider(fig[2,1], range=range(1,MAXITER,step=1))
  ax = Axis(fig[1,1])
  ax.aspect=DataAspect()

  hm=heatmap!(ax,abs.(wave_array[1]),colormap=:turbo)#,colorrange=(0,10))
  display(fig)
  for _i in 1:10:MAXITER
  # record(fig, "ising_2d.mp4", 1:100:MAXITER, framerate=60) do _i
  # lift(sl.value) do _i
    hm[3][] = abs.(wave_array[_i])
    sleep(0.051)

  end
  fig
end

