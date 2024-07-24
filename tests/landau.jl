using GLMakie
using Combinatorics: permutations, levicivita
using ImageFiltering
using LinearAlgebra
using SparseArrays: spdiagm
using ProgressBars


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
  Ndims      = (51,51,51)#,51);
  Coords     = LinRange.(-L,L,Ndims);
  space_step = 1.0; # spatial width
  time_step  = 1e-3; # time step width (remember, dt < dx^2/2)?
  MAXITER    = 5000

  # ALPHA[nx÷3:nx÷3+10,ny÷3:ny÷3+2] .*= 0.0
  # ALPHA[2nx÷3:2nx÷3+10,2ny÷3:2ny÷3+2] .*= 0.0
  U = zeros(Ndims...); # old
  # V = [exp(-5*((x-L/2)^2+(y-L/2)^2+(z-L/2)^2)) for x in Coords[1], y in Coords[2], z in Coords[3]] # actual
  V = rand(Ndims...)

  kernel_order = 2
  laplacian = centered(get_laplace_kernel(length(Ndims),kernel_order))


  println("Allocating...")
  # mat = zeros(Float32,n, nx, ny,nz)
  wave_array     = fill(zeros(Float32, size(V)...), MAXITER)
  println("Allocated $(ndims(wave_array)+ndims(wave_array[1]))D array! (of size $(Base.format_bytes(sizeof(wave_array))))")
  #@assert sizeof(mat) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"
end;

begin
  GC.gc()
  wave = copy(V)
  for _i in ProgressBar(1:MAXITER)

    
    # update wave
  
    # wave_new = ALPHA .* imfilter(wave_act, wave_kernel, Fill(0,wave_kernel)) + 2*wave_act - wave_old  # specify Fill(0,kernel) for Dirichlett conditions (boundary = 0)
    wave .= wave + time_step * (imfilter(wave, laplacian, "circular") - wave .* wave .* wave)

    wave_array[_i] = wave # store wave
    
    
  end
end


# volume(wave_act, algorithm=:absorption, absorption=5f0)

begin
  GC.gc()
  fig = Figure(resolution=(400,500))
  # sl = Slider(fig[2,1], range=range(1,MAXITER,step=1))
  # ax = Axis(fig[1,1]); ax.aspect=DataAspect()
  ax = Axis3(fig[1,1]); ax.aspect=(1,1,1)

  # hm=heatmap!(ax,Coords...,abs.(wave_array[1]),colormap=:batlow)
  hm=volume!(ax,Coords..., wave_array[1], algorithm=:absorption, absorption=4.5f0, colormap=:balance)
  cb=Colorbar(fig[2,1], hm, vertical=false, flipaxis=false, label=L"|\psi|^2")
  cb.labelsize=24
  #hm=volume!(ax,wave_array[1], algorithm=:iso, isorange=-0.1)
  # hm = contour!(ax, wave_array[1], colormap=:delta, levels=2)
  
  display(fig)
  for _i in ProgressBar(2:10:MAXITER)
  # record(fig, "landau.mp4", 2:MAXITER, framerate=60) do _i
  # lift(sl.value) do _i
    hm[4][] = abs.(wave_array[_i])
    sleep(0.01)
    # save("./images/landau"*lpad(_i, 4, '0')*".png", fig)


  end
  # fig
end

