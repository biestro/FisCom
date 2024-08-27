using GLMakie
using ProgressBars: ProgressBar
using FisCom: FDTD
using ImageFiltering: imfilter
using Images: load, Gray, Fill

init_field = Float32.(Gray.(load("./tests/cat.jpeg")));
init_field = transpose(init_field[1:20:end, 1:20:end]) / maximum(init_field)

heatmap(init_field)

begin
  GC.gc()
  L          = 2.0
  Ndims      = size(init_field)
  Coords     = LinRange.(-L,L,Ndims);
  speed      = 0.5;
  space_step = 0.5; # spatial width
  time_step  = 0.10; # time step width (remember, dt < dx^2/2)?
  MAXITER    = 1000
  velocities = ones(Ndims...) * speed # velocities 
  KAPPA      = copy(velocities) * time_step/space_step  # 
  # KAPPA      = copy(init_field) * time_step/space_step  # 
  ALPHA      = (velocities*time_step/space_step).^2; # wave equation coefficient (α^2)

  U = zeros(Ndims...)
  # V = copy(init_field); # initial field
  # V = 5*[exp(-2(x^2+y^2)) for x in Coords[1], y in Coords[2]]; # initial field
  
  W = zeros(Ndims...)
  # potential = [(x^2)/L* 5 for x in Coords]
  # V = [sin((x+0im)*pi/L) * cos(2(y+0im)*pi/L) for x in Coords[1], y in Coords[2]]

  kernel_order = 4 # O(h^4)
  laplacian    = FDTD.get_laplace_kernel(length(Ndims),kernel_order)

  println("Allocating...")
  wave_array     = fill(zeros(Float32, size(V)...), MAXITER)
  println("Allocated $(ndims(wave_array)+ndims(wave_array[1]))D array! (of size $(Base.format_bytes(sizeof(wave_array))))")
end;

heatmap(ALPHA)

# heatmap(potential)

begin
  GC.gc()
  wave_old = copy(U)
  wave_new = copy(W)
  wave_act = copy(V)
  for _i in ProgressBar(1:MAXITER)
    
    # update wave
  
    # wave_new .= ALPHA .* imfilter(wave_act, laplacian, "circular") + 2*wave_act - wave_old
    wave_new .= ALPHA .* imfilter(wave_act, laplacian, Fill(0,laplacian)) + 2*wave_act - wave_old
    # wave_new = ALPHA .* imfilter(wave_act, ImageFiltering.Laplacian((1,2),2), "circular") + 2*wave_act - wave_old
    # wave_new = ALPHA .* (sparse_mat .* wave_act) + 2*wave_act - wave_old # tridiagonal

    wave_array[_i] = wave_new # store wave
    
    # update values
    wave_old .= wave_act
    wave_act .= wave_new
    
  end
end



let
  GC.gc()
  fig = Figure()
  # sl = Slider(fig[2,1], range=range(1,MAXITER,step=1))
  ax = Axis(fig[1,1],yreversed=true); ax.aspect=DataAspect()

  hm=heatmap!(ax,abs2.(wave_array[1]),colormap=:cubehelix)#, colorrange=(-10,10))
  # hm=volume!(ax,wave_act, algorithm=:absorption, absorption=2f0)
  # hm=volume!(ax,wave_act, algorithm=:iso, absorption=2f0)
  display(fig)
  for _i in 1:10:MAXITER
  # record(fig, "ising_2d.mp4", 1:100:MAXITER, framerate=60) do _i
  # lift(sl.value) do _i
    hm[3][] = abs2.(wave_array[_i])
    sleep(0.05)

  end
  fig
end
