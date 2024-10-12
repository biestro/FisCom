using GLMakie, ProgressBars
using FisCom: FDTD


# init procedure
begin
L = 2.0
nx,ny,nz   = 91,91,91;
X,Y,Z      = LinRange(-L,L,nx),LinRange(-L,L,ny), LinRange(-1L,1L,nz);
speed      = 0.3;
space_step = 1.0;# spatial width
time_step  = 1.0; # time step width (remember, dt < dx^2/2)?
velocities = ones(nx,ny,nz) * speed # velocities 
KAPPA      = copy(velocities) * time_step/space_step  # 
ALPHA      = (velocities*time_step/space_step).^2; # wave equation coefficient 
# ALPHA[1:nx÷4, 1:ny÷4, 1:nz÷4] .= 0.0
# ALPHA[nx-nx÷4:nx, 1:ny÷4, 1:nz÷4] .= 0.0
# ALPHA[nx-nx÷4:nx, ny-ny÷4:ny, 1:nz÷4] .= 0.0
# ALPHA[nx-nx÷4:nx, ny-ny÷4:ny, nz-nz÷4:nz] .= 0.0
# ALPHA[1:nx÷4, ny-ny÷4:ny, nz-nz÷4:nz] .= 0.0
# ALPHA[1:nx÷4, 1:ny÷4, nz-nz÷4:nz] .= 0.0
# ALPHA[nx-nx÷4:nx, 1:ny÷4, nz-nz÷4:nz] .= 0.0
# ALPHA[1:nx÷4, 1:ny÷4, nz-nz÷4:nz] .= 0.0
# ALPHA[1:nx÷4, ny-ny÷4:ny, 1:nz÷4] .= 0.0
#ALPHA[1:nx÷4,1:ny÷4,1:nz÷4] .*= 0.;
#ALPHA[nx÷3:nx÷3+20,ny÷3:ny÷3+20,nz÷3:nz÷3+5] .*= 0.0
#ALPHA[2nx÷3:2nx÷3+20,2ny÷3:2ny÷3+20,2nz÷3:2nz÷3+5] .*= 0.0
U = zeros(nx, ny, nz); # actual #[exp(- 0.1 * (x-5)^2 - 0.1 * (y-5)^2) for x in xs, y in ys]
V = zeros(nx, ny, nz); # new
# V[:,:,4] .= [exp(-40*(x^2+y^2)) for x in X, y in Y]
# # V = [exp(-2*((x)^2+(y)^2+(z)^2)) for x in X, y in Y, z in Z]; # new
# V .+= [exp(-40*((x)^2+(y-2L/3)^2+(z-2L/3)^2)) for x in X, y in Y, z in Z]
# V .+= [exp(-40*((x+L/2)^2+(y+L/2)^2+(z+L/3)^2)) for x in X, y in Y, z in Z]
W = zeros(nx, ny, nz); # old

SIM_STEPS = 551;
wave_oh6 = FDTD.oh6_drops(U,copy(V),ALPHA, KAPPA, X,Y,Z, SIM_STEPS;_absorbing=true); GC.gc()
volumes = eachslice(wave_oh6,dims=1)
end

#=
# wave_oh6 = FDTD.oh6(U,V, ALPHA, KAPPA, X,Y,Z, SIM_STEPS); GC.gc()
# volumes = eachslice(wave_oh6, dims=1)
# wave_oh2 = FDTD.oh2(U,V, ALPHA, KAPPA, X,Y,Z, SIM_STEPS); GC.gc()

begin
  GC.gc()
  MAX_VAL = maximum(abs.(wave_oh6))
#  sl = SliderGrid(fig[2,1],(label="t", range=2:SIM_STEPS,startvalue=2))
#  hm = lift(sl.sliders[1].value) do _i
  fig = Figure()
  ax = Axis3(fig[1,1])
  ax.aspect=(1,1,1)
  vol_data = volume!(ax,X,Y,Z,abs.(volumes[2]), algorithm=:mip,transparency=false, colormap=:turbo)
  #  vol_data = contour!(ax,X,Y,Z,volumes[2], transparency=false, levels=[-0.1, 0.1], colormap=:balance)
  # custom_cmap=[RGBAf(c.r, c.g, c.b, 1.0) for c in to_colormap(:grays)]
  # contour!(ax,X,Y,Z,ALPHA, transparency=false, colorrange=(0, maximum(ALPHA)), levels=[0, 0.17], colormap=:grays)
#   volume!(ax, X, Y, Z, ALPHA, transparency=true, colormap=custom_cmap)

#  vol_data = contour!(ax,X,Y,Z,volumes[2], transparency=false, colormap=:turbo, levels=0.1:0.1:0.2)
  for _i in 3:SIM_STEPS
  #  text!(ax, 0.0, 0.0, text="$_i")
  vol_data[4][]=abs.(volumes[_i])
  # vol_data[4][]=(volumes[_i])
    sleep(0.00005)
    println(_i)
    display(fig)
  end
  #volume!(ax, X,Y,Z,volumes[_i])
#  end
  fig
end
=#
begin
  GC.gc()
#  sl = SliderGrid(fig[2,1],(label="t", range=2:SIM_STEPS,startvalue=2))
#  hm = lift(sl.sliders[1].value) do _i
  fig = Figure(size= (400,400))
  # pl = PointLight(Point3f(0,L+0.5,L+1.5), RGBf(3, 3, 3))
  pl = PointLight(Point3f(0,0,0), RGBf(1, 1, 1))
  #al = AmbientLight(RGBf(0.5, 0.5, 0.5))
  lscene = LScene(fig[1, 1], show_axis=false, scenekw = (lights = [pl], backgroundcolor=:white, clear=true))
  center_vec = [0.0, 1.5L]
  # meshscatter!(lscene, [Point3f(x,y,z) for x in center_vec, y in center_vec, z in center_vec][:] .- Point3f(L,L,L), marker = my_marker, color=RGBAf(1.0, 1.0, 1.0, 1.0), markersize=1.0)
  # vol_data = volume!(lscene,X,Y,Z,abs.(volumes_non_abs[2]), algorithm=:iso,isorange=0.4,transparency=true, colormap=[RGBAf(1.0,1.0,0.0, 1.0)])
  #
   #vol_data = contour!(ax,X,Y,Z,volumes[2], transparency=false, levels=[-0.1, 0.1], colormap=:balance)
  # custom_cmap=[RGBAf(c.r, c.g, c.b, 1.0) for c in to_colormap(:grays)]
#   contour!(lscene, X,Y,Z,ALPHA, transparency=false, colorrange=(0, maximum(ALPHA)), levels=[0, 0.17], colormap=:grays )

#   vol_data = volume!(lscene,X,Y,Z,abs.(volumes[2]), transparency=true, colormap=:cubehelix
#  vol_data = contour!(lscene,X,Y,Z,abs.(volumes[2]), transparency=false, colormap=:turbo, levels=0.01:0.001:0.05, fxaa=true, overdraw=true)
  vol_data = volume!(lscene,X,Y,Z,abs.(volumes[2]), algorithm = :absorption, absorption=3f0, transparency=false, colormap=:turbo, fxaa=true, colorrange=(0,1))
  # vol_data = contour!(lscene,X,Y,Z,abs.(volumes_non_abs[2]), transparency=false, colorm1p=:turbo, levels=0.1:0.02:0.65 )
#   vol_data = contour!(lscene,X,Y,Z,volumes[2], transparency=false, colormap=:grays,levels=[0.01, 0.1] )#levels=-0.1:0.01:0.1)

  for _i in ProgressBar(3:SIM_STEPS)
  #  text!(ax, 0.0, 0.0, text="$_i")
    # vol_data[4][]=abs.(volumes[_i])
    vol_data[4][]=abs.(volumes[_i])
    # sleep(0.00005) 
    display(fig)
    save("./images/fig"*lpad(_i, 4, '0')*".png", fig)
  end
  #volume!(ax, X,Y,Z,volumes[_i])
#  end
  fig
end
