using GLMakie
using ProgressBars: ProgressBar
using FisCom: FDTD
using ImageFiltering: imfilter

begin
  GC.gc()
  L          = 2.0
  Ndims      = (91)#,91)#,51);
  Coords     = LinRange.(-L,L,Ndims);
  space_step = 0.08; # spatial width
  time_step  = 0.0001; # time step width (remember, dt < dx^2/2)?
  MAXITER    = 2000
  KAPPA      = 1.0

  V = [exp(-5(x^2)/L) for x in Coords]; # initial field
  # V = 5*[sech(5(x)/L) for x in Coords]; # initial field
  # potential = [(x^2)/L* 5 for x in Coords]
  # V = [sin((x+0im)*pi/L) * cos(2(y+0im)*pi/L) for x in Coords[1], y in Coords[2]]

  kernel_order = 4 # O(h^4)
  laplacian    = FDTD.get_laplace_kernel(length(Ndims),kernel_order)

  println("Allocating...")
  wave_array     = fill(zeros(ComplexF32, size(V)...), MAXITER)
  println("Allocated $(ndims(wave_array)+ndims(wave_array[1]))D array! (of size $(Base.format_bytes(sizeof(wave_array))))")
end;

# heatmap(potential)

# visualize
begin
  GC.gc()
  wave = copy(V)
  # Euler step
  for _i in ProgressBar(1:MAXITER)
    wave = wave - time_step * 1im * (imfilter(wave, laplacian, "circular")/space_step^2 - KAPPA * abs2.(wave) .* wave) # convolution
    wave_array[_i] = wave # store wave
  end
end

heatmap(eachindex(wave_array), Coords, abs2.(mapreduce(permutedims,vcat,wave_array)))

begin
  display_fun(x) = real(x)
  _colormap = :balance
  GC.gc()
  fig = Figure(size= (400,600))
  sl = Slider(fig[0,1], range=range(1,MAXITER,step=1))
  ax = Axis(fig[1,1]); ax.aspect=DataAspect(); hm=heatmap!(ax,Coords...,display_fun.(wave_array[1]),colormap=_colormap, interpolate=true)
  # ax = Axis3(fig[1,1]); ax.aspect=(1,1,1); hm=volume!(ax,Coords..., wave_array[1], algorithm=:absorption, absorption=4.5f0, colormap=:balance)
  cb=Colorbar(fig[2,1], hm, vertical=false, flipaxis=false, label=L"|\psi|^2"); cb.labelsize=24
  
  lift(sl.value) do _i
    hm[3][] = display_fun.(wave_array[_i])
  end
  fig
end
