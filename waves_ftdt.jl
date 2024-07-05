using LinearAlgebra
using GLMakie
using BenchmarkTools

function init(nx::Int64, ny::Int64)
  # initial values
  L = 10  
  #  calculations
  #dx, dy = L/nx, L/ny
  xs = LinRange(-L,L,nx);
  ys = LinRange(-L,L,ny);
  
  
  
      
  c = 0.3;
  h = .4;# spatial width
  k = .2; # time step width (remember, dt < dx^2/2)?
  α = ones(nx, ny) .* ((c*k) / h)^2; # alpha squared ()
  #α[144:148,  1:55] .= 0;
  #α[144:148, 60:90] .= 0;
  #α[144:148, 95:123] .= 0;
  #α[194:198, 143:170] .= 0;

  #α[div(nx,3):nx-div(nx,3),div(ny,3):ny-div(ny,3)] .= 0


  κ = sqrt.(α) .* (k/h) .* 9; # no tengo idea por qué hay que multiplicar por 9 para que no refleje

 
  Zₙ₊₁ = zeros(nx, ny);#[exp(- 0.1 * (x-5)^2 - 0.1 * (y-5)^2) for x in xs, y in ys]
  Zₙ   = zeros(nx, ny);
  Zₙ₋₁ = zeros(nx, ny);
  #data = zeros(N, nx+1, ny+1)
  
  
  return Zₙ, Zₙ₊₁, Zₙ₋₁, α, κ, xs, ys;
end

function oh4(vₙ::Matrix{Float64}, vₙ₊₁::Matrix{Float64}, α::Matrix{Float64}, κ::Matrix{Float64}, nx::Int64, ny::Int64,n::Int64)
  # better than the vectorized format
  mat = zeros(n, nx, ny)
  
  bndry = 2
  for ii in 1:n
    mat[ii,:,:] = vₙ
      # you NEED to use copy(), this isn't python
    ω = 2* π * 0.02;
    #if ii < 40
    #  vₙ₊₁[nx-bndry-1:nx-bndry,div(ny,3):ny-div(ny,3)] .= 10 .* cos(ω * ii);
    #end
    #vₙ₊₁[nx-bndry-2:nx-bndry,ny-bndry-2:ny-bndry] .= 10 .* cos(ω * ii);
    vₙ₊₁[div(nx,2)-1:div(nx,2)+1,div(ny,2)-1:div(ny,2)+1] .= 10 .* cos(ω * ii);
    #M[ii,:,:] = copy(vₙ₊₁);
    vₙ₋₁ = copy(vₙ);
    vₙ = copy(vₙ₊₁);
    
    for ii in 3:nx-2
      for jj in 3:ny-2
        vₙ₊₁[ii, jj]  = α[ii,jj] * (  -vₙ[ii,jj-2]+
                                    16*vₙ[ii,jj-1]-
                                       vₙ[ii-2,jj]+
                                    16*vₙ[ii-1,jj]-
                                    60*vₙ[ii,  jj]+
                                    16*vₙ[ii+1,jj]-
                                       vₙ[ii+2,jj]+
                                    16*vₙ[ii,jj+1]-
                                       vₙ[ii,jj+2]
                                    ) + 
                            2 * vₙ[ii, jj] - vₙ₋₁[ii, jj];
        
          # absorbing boundaries
          # don't mix for loops and vectorization, bad results
          # make this only work where the boudnaries are? would it be betters?
        for kk in 1:bndry
          vₙ₊₁[kk, jj] =     vₙ[kk+1,   jj]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[kk+1,   jj]-vₙ[kk,     jj]);# x = 0
          vₙ₊₁[nx-2+kk,jj] = vₙ[nx-3+kk,jj]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[nx-3+kk,jj]-vₙ[nx-2+kk,jj]);# x = N
          vₙ₊₁[ii, kk] =     vₙ[ii,   kk+1]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[ii,   kk+1]-vₙ[ii,     kk]);# y = 0
          vₙ₊₁[ii,ny-2+kk] = vₙ[ii,ny-3+kk]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[ii,ny-3+kk]-vₙ[ii,ny-2+kk]);# y = N
        end
      end
    end
  end

  return mat;
end

function get_plot(data,medium, xs, ys)
  custom_cmap=[RGBAf(0, 0, 0, 1) for c in to_colormap(:Greys)]
  fig=Figure()
  ax=Axis(fig[1,1])
  ax.aspect = DataAspect() 
  hm=Makie.heatmap!(ax,xs, ys, data,colormap=Reverse(:Spectral), interpolate=false)
  
  mask = ones(size(medium))
  mask[medium .> minimum(medium)] .= NaN;
  
  #bl=Makie.heatmap!(ax, xs, ys, mask; colormap=custom_cmap, interpolate=false)
  Colorbar(fig[1,2],hm)
  rowsize!(fig.layout, 1, ax.scene.px_area[].widths[2]) # set colorbar height
  fig
end

begin
  N = 300;
  nx, ny = 251,251;
  U, U_new, U_old, Alpha,Kappa, xs, ys = init(nx,ny);
  #@benchmark data = oh4(U, U_new, U_old, A, K, zeros(N, nx, ny),nx, ny, N)
  data = oh4(U, U_new, Alpha, Kappa, nx, ny, N);
  get_plot(norm.(data[end,:,:]),Alpha,xs,ys)
end

begin
  set_theme!(theme_black())
  custom_cmap=[RGBAf(c.r, c.g, c.b, 0.05) for c in to_colormap(:grays)]
  f = Figure(resolution=(500,500))   
  ax = Axis(f[1,1])
  ax.aspect=DataAspect()
  rowsize!(f.layout, 1, ax.scene.px_area[].widths[2]) # set colorbar height
  framerate = 30
  record(f, "wave_4_showoff_4.mp4", 1:2:N;
          framerate = framerate) do idx
      hm = heatmap!(ax, xs, ys, (data[idx,:,:]); colormap=Reverse(:Spectral), colorrange=(-10,10), interpolate=true)
      
      #bl = heatmap!(ax,alpha[diagind(alpha)],colormap = :grays,colorrange=(0,0), transparency=true)
      #bl = heatmap!(ax, xs, ys, A; colormap=custom_cmap, interpolate=true)
      
  end
end