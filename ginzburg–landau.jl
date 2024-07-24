using GLMakie
using ProgressBars: ProgressBar
using FisCom: FDTD

begin
  L          = 2.0
  Ndims      = (51,51)#,51);
  Coords     = LinRange.(-L,L,Ndims);
  space_step = 0.5; # spatial width
  time_step  = 1e-3; # time step width (remember, dt < dx^2/2)?
  MAXITER    = 5000

  V = rand(Ndims...); # initial field

  kernel_order = 4 # O(h^4)
  laplacian    = centered(get_laplace_kernel(length(Ndims),kernel_order))

  println("Allocating...")
  wave_array     = fill(zeros(Float32, size(V)...), MAXITER)
  println("Allocated $(ndims(wave_array)+ndims(wave_array[1]))D array! (of size $(Base.format_bytes(sizeof(wave_array))))")
end;

# visualize
begin
  GC.gc()
  wave = copy(V)
  # Euler step
  for _i in ProgressBar(1:MAXITER)
    wave = wave + time_step * (imfilter(wave, laplacian, "circular")/space_step^2 - abs2.(wave) .* wave + wave) # convolution
    wave_array[_i] = wave # store wave
  end
end

begin
  GC.gc()
  fig = Figure(size= (400,600))
  sl = Slider(fig[0,1], range=range(1,MAXITER,step=1))
  ax = Axis(fig[1,1]); ax.aspect=DataAspect(); hm=heatmap!(ax,Coords...,abs.(wave_array[1]),colormap=:batlow)
  # ax = Axis3(fig[1,1]); ax.aspect=(1,1,1); hm=volume!(ax,Coords..., wave_array[1], algorithm=:absorption, absorption=4.5f0, colormap=:balance)
  cb=Colorbar(fig[2,1], hm, vertical=false, flipaxis=false, label=L"|\psi|^2"); cb.labelsize=24
  #hm=volume!(ax,wave_array[1], algorithm=:iso, isorange=-0.1)
  # hm = contour!(ax, wave_array[1], colormap=:delta, levels=2)
  
  # display(fig)
  # for _i in ProgressBar(2:10:MAXITER)
  # record(fig, "landau.mp4", 2:MAXITER, framerate=60) do _i
  lift(sl.value) do _i
    hm[3][] = abs.(wave_array[_i])
    # sleep(0.01)
    # save("./images/landau"*lpad(_i, 4, '0')*".png", fig)


  end
  fig
end
