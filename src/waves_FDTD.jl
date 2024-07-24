module FDTD

# TODO: change Array{4} to Vec{Array{3}} for 3D data

using LinearAlgebra
using StaticArrays
using SIMD
using ProgressBars
using Combinatorics: permutations, levicivita



export oh2, oh4, oh6

global MAX_MEMORY = 5e9

#= === 1D algorithms === =#

"""
  oh2(_vₙ, _vₙ₊₁, _alpha, _kappa, _xs, _nt)

Solves the 1D wave equatio_nt with error O(h^2)
"""
function oh2(_vₙ::Array{Float64, 1}, 
             _vₙ₊₁::Array{Float64, 1}, 
             _alpha::Array{Float64, 1}, 
             _kappa::Array{Float64, 1}, 
             _xs::LinRange, 
             _nt::Int64;
             _absorbing=true)


  ##############################################################################
  println()
  println("Running O(h²) approximation...")
  println("==============================")
  nx = length(_xs)
  println("Allocating...")
  mat = zeros(Float32,_nt, nx)
  println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(mat))))")
  @assert sizeof(mat) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"
  ##############################################################################
  loss = 1.0
  bndry = 1

  @inbounds @fastmath for tt in ProgressBar(1:_nt)
    mat[tt,:] = _vₙ
    _vₙ₋₁ = copy(_vₙ);
    _vₙ = copy(_vₙ₊₁);
    @inbounds @fastmath @simd for jj in 2:ny-1
      @inbounds for ii in 2:nx-1
        _vₙ₊₁[ii, jj]  = _alpha[ii,jj] * (
                                    _vₙ[ii,jj-1]+  
                                    _vₙ[ii,jj+1]+
                                    _vₙ[ii-1,jj]+
                                    _vₙ[ii+1,jj]+
                                  -4_vₙ[ii, jj]) + 
                                  2 * _vₙ[ii,jj] - _vₙ₋₁[ii,jj];
        _vₙ₊₁[ii, jj] *= loss
        
        # absorbing bou_ntdaries
        if _absorbing
          _vₙ₊₁[1,jj]  = _vₙ[2 ,jj ]  + (_kappa[ii,jj]-1)/(_kappa[ii,jj]+1) * (_vₙ₊₁[2, jj] - _vₙ[1, jj]);# x = 0
          _vₙ₊₁[ii,1]  = _vₙ[ii, 2 ]  + (_kappa[ii,jj]-1)/(_kappa[ii,jj]+1) * (_vₙ₊₁[ii, 2] - _vₙ[ii, 1]);# y = 0
          _vₙ₊₁[nx,jj] = _vₙ[nx-1,jj] + (_kappa[ii,jj]-1)/(_kappa[ii,jj]+1) * (_vₙ₊₁[nx-1,jj] - _vₙ[nx,jj]);# x = N
          _vₙ₊₁[ii,ny] = _vₙ[ii,ny-1] + (_kappa[ii,jj]-1)/(_kappa[ii,jj]+1) * (_vₙ₊₁[ii,ny-1] - _vₙ[ii,ny]);# y = N
        end
      end
    end
  end
  return mat;
end

"""
  oh4(vₙ, vₙ₊₁, α, κ, nx, ny, n)

Solves the 1D wave equation with error O(h^4)
"""
function oh4(vₙ::Array{Float64,1}, 
             vₙ₊₁::Array{Float64,1}, 
             α::Array{Float64,1}, 
             κ::Array{Float64,1}, 
             _xs::LinRange, 
             n::Int64;
             _absorbing=true)
  # 3D
  println()
  println("Running O(h⁴) approximation...")
  println("==============================")
  nx,ny = length(_xs), length(_ys)
  println("Allocating...")
  mat = zeros(Float32,n, nx,ny)
  println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(mat))))")
  @assert sizeof(mat) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"

  # envelope_field = [exp(-0.5*(x^2)) for x in _xs]
  loss = 1.0
  bndry = 2
  ω     = 2π * 0.02
  # vₙ₊₁[:,:,nz-3] .= envelope_field
  @inbounds @fastmath for tt in ProgressBar(1:n)
    mat[tt,:,:] = copy(vₙ)
    vₙ₋₁ = copy(vₙ);
    # vₙ₊₁[:,:,nz-3] .= cos(ω*tt) .* envelope_field
    vₙ = copy(vₙ₊₁);
    @inbounds @fastmath @simd for jj in 3:ny-2
      @inbounds for ii in 3:nx-2
        vₙ₊₁[ii, jj]  = α[ii,jj] * (
                                      -vₙ[ii,jj-2]+
                                     16vₙ[ii,jj-1]+
                                     16vₙ[ii,jj+1]+  
                                      -vₙ[ii,jj+2]+

                                      -vₙ[ii-2,jj]+
                                     16vₙ[ii-1,jj]+
                                     16vₙ[ii+1,jj]+  
                                      -vₙ[ii+2,jj]+
                                      
                                    -60vₙ[ii,  jj]
                                  ) / 12 + 

                                  2 * vₙ[ii, jj] - vₙ₋₁[ii, jj]; # 
          vₙ₊₁[ii, jj] *= loss
        
        # absorbing boundaries
        # don't mix for loops and vectorization, bad results
        # make this only work where the boudnaries are? would it be betters?
        if _absorbing
        @inbounds for bb in 1:bndry
          vₙ₊₁[bb,jj] = vₙ[bb+1 ,jj ] + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[bb+1, jj]-vₙ[bb, jj]);# x = 0
          vₙ₊₁[ii,bb] = vₙ[ii, bb+1 ] + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[ii, bb+1]-vₙ[ii, bb]);# y = 0
          vₙ₊₁[nx-2+bb,jj] = vₙ[nx-3+bb,jj]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[nx-3+bb,jj]-vₙ[nx-2+bb,jj]);# x = N
          vₙ₊₁[ii,ny-2+bb] = vₙ[ii,ny-3+bb]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[ii,ny-3+bb]-vₙ[ii,ny-2+bb]);# y = N
        end
        end
        end
      end
    end

  return mat;
end

"""
  oh6(vₙ, vₙ₊₁, α, κ, nx, ny, n)

Solves the 1D wave equation with error O(h^6)
"""
function oh6(vₙ::Array{Float64,1}, 
             vₙ₊₁::Array{Float64,1}, 
             α::Array{Float64,1}, 
             κ::Array{Float64,1}, 
             _xs::LinRange, 
             n::Int64;
             _absorbing=true)
  
  ##############################################################################
  println()
  println("Running O(h⁶) approximation...")
  println("==============================")
  nx,ny = length(_xs), length(_ys)
  println("Allocating...")
  mat = zeros(Float32,n, nx, ny)
  println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(mat))))")
  @assert sizeof(mat) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"
  ##############################################################################

  # envelope_field = copy(vₙ₊₁)
  loss = 1.0
  bndry = 3
  ω     = 2π * 0.02
  # vₙ₊₁[:,:,nz-4] .= envelope_field
  @inbounds @fastmath for tt in ProgressBar(1:n)
    mat[tt,:,:] = copy(vₙ)
    vₙ₋₁ = copy(vₙ);
    #vₙ₊₁[:,:,nz-4] .= cos(ω*tt) .* envelope_field[:,:,nz÷2+1]
    vₙ = copy(vₙ₊₁);
    @inbounds @fastmath @simd for jj in 4:ny-3
      @inbounds for ii in 4:nx-3
        vₙ₊₁[ii, jj]  = α[ii,jj] * (
                                        2vₙ[ii,jj-3]+
                                      -27vₙ[ii,jj-2]+
                                      270vₙ[ii,jj-1]+
                                      270vₙ[ii,jj+1]+
                                      -27vₙ[ii,jj+2]+
                                        2vₙ[ii,jj+3]+

                                        2vₙ[ii-3,jj]+
                                      -27vₙ[ii-2,jj]+
                                      270vₙ[ii-1,jj]+
                                      270vₙ[ii+1,jj]+
                                      -27vₙ[ii+2,jj]+
                                        2vₙ[ii+3,jj]+
                                        
                                   -2*490vₙ[ii,  jj]
                                    ) / 180 + 

                                    2 * vₙ[ii, jj] - vₙ₋₁[ii, jj]; # 
          vₙ₊₁[ii, jj] *= loss
        
        if _absorbing
        @inbounds for bb in 1:bndry
          vₙ₊₁[bb,jj] = vₙ[bb+1 ,jj] + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[bb+1, jj]-vₙ[bb, jj]);# x = 0
          vₙ₊₁[ii,bb] = vₙ[ii, bb+1] + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[ii, bb+1]-vₙ[ii, bb]);# y = 0
          vₙ₊₁[nx-3+bb,jj] = vₙ[nx-4+bb,jj]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[nx-4+bb,jj]-vₙ[nx-3+bb,jj]);# x = N
          vₙ₊₁[ii,ny-3+bb] = vₙ[ii,ny-4+bb]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[ii,ny-4+bb]-vₙ[ii,ny-3+bb]);# y = N
        end
        end
      end
  end

  end
  return mat;
end
#= === 2D algorithms === =#

"""
  oh2(_vₙ, _vₙ₊₁, _alpha, _kappa, _nx, _ny,_nz, _nt)

Solves the 2D wave equatio_nt with error O(h^2)
"""
function oh2(_vₙ::Array{Float64, 2}, 
             _vₙ₊₁::Array{Float64, 2}, 
             _alpha::Array{Float64, 2}, 
             _kappa::Array{Float64, 2}, 
             _xs::LinRange, 
             _ys::LinRange, 
             _nt::Int64;
             _absorbing=true)


  ##############################################################################
  println()
  println("Running O(h²) approximation...")
  println("==============================")
  nx,ny = length(_xs), length(_ys)
  println("Allocating...")
  mat = zeros(Float32,_nt, nx, ny)
  println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(mat))))")
  @assert sizeof(mat) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"
  ##############################################################################
  loss = 1.0
  bndry = 1

  @inbounds @fastmath for tt in ProgressBar(1:_nt)
    mat[tt,:,:] = _vₙ
    _vₙ₋₁ = copy(_vₙ);
    _vₙ = copy(_vₙ₊₁);
    @inbounds @fastmath @simd for jj in 2:ny-1
      @inbounds for ii in 2:nx-1
        _vₙ₊₁[ii, jj]  = _alpha[ii,jj] * (
                                    _vₙ[ii,jj-1]+  
                                    _vₙ[ii,jj+1]+
                                    _vₙ[ii-1,jj]+
                                    _vₙ[ii+1,jj]+
                                  -4_vₙ[ii, jj]) + 
                                  2 * _vₙ[ii,jj] - _vₙ₋₁[ii,jj];
        _vₙ₊₁[ii, jj] *= loss
        
        # absorbing bou_ntdaries
        if _absorbing
          _vₙ₊₁[1,jj]  = _vₙ[2 ,jj ]  + (_kappa[ii,jj]-1)/(_kappa[ii,jj]+1) * (_vₙ₊₁[2, jj] - _vₙ[1, jj]);# x = 0
          _vₙ₊₁[ii,1]  = _vₙ[ii, 2 ]  + (_kappa[ii,jj]-1)/(_kappa[ii,jj]+1) * (_vₙ₊₁[ii, 2] - _vₙ[ii, 1]);# y = 0
          _vₙ₊₁[nx,jj] = _vₙ[nx-1,jj] + (_kappa[ii,jj]-1)/(_kappa[ii,jj]+1) * (_vₙ₊₁[nx-1,jj] - _vₙ[nx,jj]);# x = N
          _vₙ₊₁[ii,ny] = _vₙ[ii,ny-1] + (_kappa[ii,jj]-1)/(_kappa[ii,jj]+1) * (_vₙ₊₁[ii,ny-1] - _vₙ[ii,ny]);# y = N
        end
      end
    end
  end
  return mat;
end

"""
  oh4(vₙ, vₙ₊₁, α, κ, nx, ny, n)

Solves the 2D wave equation with error O(h^4)
"""
function oh4(vₙ::Array{Float64,2}, 
             vₙ₊₁::Array{Float64,2}, 
             α::Array{Float64,2}, 
             κ::Array{Float64,2}, 
             _xs::LinRange, 
             _ys::LinRange,
             n::Int64;
             _absorbing=true)
  # 3D
  println()
  println("Running O(h⁴) approximation...")
  println("==============================")
  nx,ny = length(_xs), length(_ys)
  println("Allocating...")
  mat = zeros(Float32,n, nx,ny)
  println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(mat))))")
  @assert sizeof(mat) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"

  # envelope_field = [exp(-0.5*(x^2)) for x in _xs]
  loss = 1.0
  bndry = 2
  ω     = 2π * 0.02
  # vₙ₊₁[:,:,nz-3] .= envelope_field
  @inbounds @fastmath for tt in ProgressBar(1:n)
    mat[tt,:,:] = copy(vₙ)
    vₙ₋₁ = copy(vₙ);
    # vₙ₊₁[:,:,nz-3] .= cos(ω*tt) .* envelope_field
    vₙ = copy(vₙ₊₁);
    @inbounds @fastmath @simd for jj in 3:ny-2
      @inbounds for ii in 3:nx-2
        vₙ₊₁[ii, jj]  = α[ii,jj] * (
                                      -vₙ[ii,jj-2]+
                                     16vₙ[ii,jj-1]+
                                     16vₙ[ii,jj+1]+  
                                      -vₙ[ii,jj+2]+

                                      -vₙ[ii-2,jj]+
                                     16vₙ[ii-1,jj]+
                                     16vₙ[ii+1,jj]+  
                                      -vₙ[ii+2,jj]+
                                      
                                    -60vₙ[ii,  jj]
                                  ) / 12 + 

                                  2 * vₙ[ii, jj] - vₙ₋₁[ii, jj]; # 
          vₙ₊₁[ii, jj] *= loss
        
        # absorbing boundaries
        # don't mix for loops and vectorization, bad results
        # make this only work where the boudnaries are? would it be betters?
        if _absorbing
        @inbounds for bb in 1:bndry
          vₙ₊₁[bb,jj] = vₙ[bb+1 ,jj ] + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[bb+1, jj]-vₙ[bb, jj]);# x = 0
          vₙ₊₁[ii,bb] = vₙ[ii, bb+1 ] + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[ii, bb+1]-vₙ[ii, bb]);# y = 0
          vₙ₊₁[nx-2+bb,jj] = vₙ[nx-3+bb,jj]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[nx-3+bb,jj]-vₙ[nx-2+bb,jj]);# x = N
          vₙ₊₁[ii,ny-2+bb] = vₙ[ii,ny-3+bb]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[ii,ny-3+bb]-vₙ[ii,ny-2+bb]);# y = N
        end
        end
        end
      end
    end

  return mat;
end

"""
  oh6(vₙ, vₙ₊₁, α, κ, nx, ny, n)

Solves the 2D wave equation with error O(h^6)
"""
function oh6(vₙ::Array{Float64,2}, 
             vₙ₊₁::Array{Float64,2}, 
             α::Array{Float64,2}, 
             κ::Array{Float64,2}, 
             _xs::LinRange, 
             _ys::LinRange,
             n::Int64;
             _absorbing=true)
  
  ##############################################################################
  println()
  println("Running O(h⁶) approximation...")
  println("==============================")
  nx,ny = length(_xs), length(_ys)
  println("Allocating...")
  mat = zeros(Float32,n, nx, ny)
  println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(mat))))")
  @assert sizeof(mat) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"
  ##############################################################################

  # envelope_field = copy(vₙ₊₁)
  loss = 1.0
  bndry = 3
  ω     = 2π * 0.02
  # vₙ₊₁[:,:,nz-4] .= envelope_field
  @inbounds @fastmath for tt in ProgressBar(1:n)
    mat[tt,:,:] = copy(vₙ)
    vₙ₋₁ = copy(vₙ);
    #vₙ₊₁[:,:,nz-4] .= cos(ω*tt) .* envelope_field[:,:,nz÷2+1]
    vₙ = copy(vₙ₊₁);
    @inbounds @fastmath @simd for jj in 4:ny-3
      @inbounds for ii in 4:nx-3
        vₙ₊₁[ii, jj]  = α[ii,jj] * (
                                        2vₙ[ii,jj-3]+
                                      -27vₙ[ii,jj-2]+
                                      270vₙ[ii,jj-1]+
                                      270vₙ[ii,jj+1]+
                                      -27vₙ[ii,jj+2]+
                                        2vₙ[ii,jj+3]+

                                        2vₙ[ii-3,jj]+
                                      -27vₙ[ii-2,jj]+
                                      270vₙ[ii-1,jj]+
                                      270vₙ[ii+1,jj]+
                                      -27vₙ[ii+2,jj]+
                                        2vₙ[ii+3,jj]+
                                        
                                   -2*490vₙ[ii,  jj]
                                    ) / 180 + 

                                    2 * vₙ[ii, jj] - vₙ₋₁[ii, jj]; # 
          vₙ₊₁[ii, jj] *= loss
        
        if _absorbing
        @inbounds for bb in 1:bndry
          vₙ₊₁[bb,jj] = vₙ[bb+1 ,jj] + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[bb+1, jj]-vₙ[bb, jj]);# x = 0
          vₙ₊₁[ii,bb] = vₙ[ii, bb+1] + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[ii, bb+1]-vₙ[ii, bb]);# y = 0
          vₙ₊₁[nx-3+bb,jj] = vₙ[nx-4+bb,jj]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[nx-4+bb,jj]-vₙ[nx-3+bb,jj]);# x = N
          vₙ₊₁[ii,ny-3+bb] = vₙ[ii,ny-4+bb]  + (κ[ii,jj]-1)/(κ[ii,jj]+1) * (vₙ₊₁[ii,ny-4+bb]-vₙ[ii,ny-3+bb]);# y = N
        end
        end
      end
  end

  end
  return mat;
end
#= === 3D algorithms === =#

"""
  Utility function for initializing a 3D wave problem.
"""
function init(_nx::Int64, _ny::Int64, _nz::Int64)
  # initial values
  L = 5 
  #  calculatio_nts
  #dx, dy = L/_nx, L/_ny
  xs = LinRange(-L,L,_nx);  ys = LinRange(-L,L,_ny);
  zs = LinRange(-1.5L,1.5L,_nz);
      
  c = 0.2;
  h = 1.0;# spatial width
  k = 1.0; # time step width (remember, dt < dx^2/2)?
  v = ones(_nx,_ny,_nz) * c # velocities 
  κ = copy(v) * k/h 
  alpha = (v*k/h).^2; # alpha squared 

  alpha[1:_nx,
    1:div(_ny,3),
    div(_nz,3):div(_nz,3)+5] .= 0

  Zₙ₊₁ = zeros(_nx, _ny, _nz);#[exp(- 0.1 * (x-5)^2 - 0.1 * (y-5)^2) for x in xs, y in ys]
  Zₙ   = zeros(_nx, _ny, _nz);
  # Zₙ =  [exp(-2.0*(x^2 +y^2 + (z)^2)) for x in xs, y in ys, z in zs]
  # Zₙ +=  [exp(-2.0*(x^2 +y^2 + (z-1.5L)^2)) for x in xs, y in ys, z in zs]
  Zₙ₋₁ = zeros(_nx, _ny,_nz);
  #data = zeros(_nt, _nx+1, _ny+1)
  # Zₙ₊₁[div(_nx,2)-2:div(_nx,2)+3, 
  return Zₙ, Zₙ₊₁, Zₙ₋₁, alpha, κ, xs, ys, zs;
end


"""
  oh2(_vₙ, _vₙ₊₁, _alpha, _kappa, _nx, _ny,_nz, _nt)

Solves the 3D wave equatio_nt with error O(h^2)
"""
function oh2(_vₙ::Array{Float64, 3}, 
             _vₙ₊₁::Array{Float64, 3}, 
             _alpha::Array{Float64, 3}, 
             _kappa::Array{Float64, 3}, 
             _xs::LinRange, 
             _ys::LinRange, 
             _zs::LinRange,
             _nt::Int64;
             _absorbing=true)


  ##############################################################################
  println()
  println("Running O(h²) approximation...")
  println("==============================")
  nx,ny,nz = length(_xs), length(_ys), length(_zs)
  println("Allocating...")
  mat = zeros(Float32,n, nx, ny,nz)
  println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(mat))))")
  @assert sizeof(mat) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"
  ##############################################################################
  loss = 1.0
  bndry = 1

  @inbounds @fastmath for tt in ProgressBar(1:_nt)
    mat[tt,:,:,:] = _vₙ
    _vₙ₋₁ = copy(_vₙ);
    _vₙ = copy(_vₙ₊₁);
    for kk in 2:_nz-1
      for jj in 2:_ny-1
        for ii in 2:_nx-1
          _vₙ₊₁[ii, jj]  = _alpha[ii,jj] * (
                                      _vₙ[ii,jj,kk-1]+
                                      _vₙ[ii,jj,kk+1]+  
                                      _vₙ[ii,jj-1,kk]+  
                                      _vₙ[ii,jj+1,kk]+
                                      _vₙ[ii-1,jj,kk]+
                                      _vₙ[ii+1,jj,kk]+
                                    -6_vₙ[ii, jj, kk]) + 
                                    2 * _vₙ[ii, jj,kk] - _vₙ₋₁[ii, jj,kk];
          _vₙ₊₁[ii, jj, kk] *= loss
          
          # absorbing bou_ntdaries
          if _absorbing
            _vₙ₊₁[1,jj,kk] = _vₙ[2 ,jj,kk ] + (_kappa[ii,jj,kk]-1)/(_kappa[ii,jj,kk]+1) * (_vₙ₊₁[2, jj,kk] - _vₙ[1, jj, kk]);# x = 0
            _vₙ₊₁[ii,1,kk] = _vₙ[ii, 2 ,kk] + (_kappa[ii,jj,kk]-1)/(_kappa[ii,jj,kk]+1) * (_vₙ₊₁[ii, 2,kk] - _vₙ[ii, 1, kk]);# y = 0
            _vₙ₊₁[ii,jj,1] = _vₙ[ii,jj, 2 ] + (_kappa[ii,jj,kk]-1)/(_kappa[ii,jj,kk]+1) * (_vₙ₊₁[ii,jj, 2] - _vₙ[ii, jj, 1]);# z = 0
            _vₙ₊₁[nx,jj,kk] = _vₙ[nx-1,jj,kk]  + (_kappa[ii,jj,kk]-1)/(_kappa[ii,jj,kk]+1) * (_vₙ₊₁[nx-1,jj,kk] - _vₙ[nx,jj,kk]);# x = N
            _vₙ₊₁[ii,ny,kk] = _vₙ[ii,ny-1,kk]  + (_kappa[ii,jj,kk]-1)/(_kappa[ii,jj,kk]+1) * (_vₙ₊₁[ii,ny-1,kk] - _vₙ[ii,ny,kk]);# y = N
            _vₙ₊₁[ii,jj,nz] = _vₙ[ii,jj,nz-1]  + (_kappa[ii,jj,kk]-1)/(_kappa[ii,jj,kk]+1) * (_vₙ₊₁[ii,jj,nz-1] - _vₙ[ii,jj,nz]);# z = N
          end
        end
      end
    end
  end
  return mat;
end


"""
  oh4(vₙ, vₙ₊₁, α, κ, nx, ny, n)

Solves the 3D wave equation with error O(h^4)
"""
function oh4(vₙ::Array{Float64,3}, 
             vₙ₊₁::Array{Float64,3}, 
             α::Array{Float64,3}, 
             κ::Array{Float64,3}, 
             _xs::LinRange, 
             _ys::LinRange,
             _zs::LinRange ,
             n::Int64;
             _absorbing=true)
  # 3D
  println()
  println("Running O(h⁴) approximation...")
  println("==============================")
  nx,ny,nz = length(_xs), length(_ys), length(_zs)
  println("Allocating...")
  mat = zeros(Float32,n, nx, ny,nz)
  println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(mat))))")
  @assert sizeof(mat) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"

  envelope_field = [exp(-0.5*(x^2 +y^2)) for x in _xs, y in _ys]
  loss = 1.0
  bndry = 2
  ω     = 2π * 0.02
  # vₙ₊₁[:,:,nz-3] .= envelope_field
  @inbounds @fastmath for tt in ProgressBar(1:n)
    mat[tt,:,:,:] = copy(vₙ)
    vₙ₋₁ = copy(vₙ);
    # vₙ₊₁[:,:,nz-3] .= cos(ω*tt) .* envelope_field
    vₙ = copy(vₙ₊₁);
    @inbounds @fastmath @simd for kk in 3:nz-2
      @inbounds for jj in 3:ny-2
        @inbounds for ii in 3:nx-2
          vₙ₊₁[ii, jj,kk]  = α[ii,jj,kk] * (
                                        -vₙ[ii,jj,kk-2]+
                                       16vₙ[ii,jj,kk-1]+
                                       16vₙ[ii,jj,kk+1]+  
                                        -vₙ[ii,jj,kk+2]+

                                        -vₙ[ii,jj-2,kk]+
                                       16vₙ[ii,jj-1,kk]+
                                       16vₙ[ii,jj+1,kk]+  
                                        -vₙ[ii,jj+2,kk]+

                                        -vₙ[ii-2,jj,kk]+
                                       16vₙ[ii-1,jj,kk]+
                                       16vₙ[ii+1,jj,kk]+  
                                        -vₙ[ii+2,jj,kk]+
                                        
                                      -90vₙ[ii,  jj,kk]
                                    ) / 12 + 

                                    2 * vₙ[ii, jj,kk] - vₙ₋₁[ii, jj,kk]; # 
          vₙ₊₁[ii, jj,kk] *= loss
        
        # absorbing boundaries
        # don't mix for loops and vectorization, bad results
        # make this only work where the boudnaries are? would it be betters?
        if _absorbing
        @inbounds for bb in 1:bndry
          vₙ₊₁[bb,jj,kk] = vₙ[bb+1 ,jj,kk ] + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[bb+1, jj,kk]-vₙ[bb, jj, kk]);# x = 0
          vₙ₊₁[ii,bb,kk] = vₙ[ii, bb+1 ,kk] + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii, bb+1,kk]-vₙ[ii, bb, kk]);# y = 0
          vₙ₊₁[ii,jj,bb] = vₙ[ii,jj, bb+1 ] + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii,jj, bb+1]-vₙ[ii, jj, bb]);# z = 0
          vₙ₊₁[nx-2+bb,jj,kk] = vₙ[nx-3+bb,jj,kk]  + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[nx-3+bb,jj,kk]-vₙ[nx-2+bb,jj,kk]);# x = N
          vₙ₊₁[ii,ny-2+bb,kk] = vₙ[ii,ny-3+bb,kk]  + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii,ny-3+bb,kk]-vₙ[ii,ny-2+bb,kk]);# y = N
          vₙ₊₁[ii,jj,nz-2+bb] = vₙ[ii,jj,nz-3+bb]  + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii,jj,nz-3+bb]-vₙ[ii,jj,nz-2+bb]);# z = N
        end
        end
        end
      end
    end
  end

  return mat;
end


"""
  oh6(vₙ, vₙ₊₁, α, κ, nx, ny, n)

Solves the 3D wave equation with error O(h^6)
"""
function oh6(vₙ::Array{Float64,3}, 
             vₙ₊₁::Array{Float64,3}, 
             α::Array{Float64,3}, 
             κ::Array{Float64,3}, 
             _xs::LinRange, 
             _ys::LinRange,
             _zs::LinRange ,
             n::Int64;
             _absorbing=true)
  
  ##############################################################################
  println()
  println("Running O(h⁶) approximation...")
  println("==============================")
  nx,ny,nz = length(_xs), length(_ys), length(_zs)
  println("Allocating...")
  mat = zeros(Float32,n, nx, ny,nz)
  println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(mat))))")
  @assert sizeof(mat) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"
  ##############################################################################

  envelope_field = copy(vₙ₊₁)
  loss = 1.0
  bndry = 3

  # ω     = 2π * 0.02
  # vₙ₊₁[:,:,nz-4] .= envelope_field
  @inbounds @fastmath for tt in ProgressBar(1:n)
    mat[tt,:,:,:] = copy(vₙ)
    
    vₙ₋₁ = copy(vₙ);
    #vₙ₊₁[:,:,nz-4] .= cos(ω*tt) .* envelope_field[:,:,nz÷2+1]
    #vₙ₊₁[:,:,:] .= cos(ω*tt) .* copy(envelope_field)
    vₙ = copy(vₙ₊₁);
    # vₙ[nx-4,ny-4,nz-4] = cos(ω*tt)
    @inbounds @fastmath @simd for kk in 4:nz-3
      @inbounds for jj in 4:ny-3
        @inbounds for ii in 4:nx-3
          vₙ₊₁[ii, jj,kk]  = α[ii,jj,kk] * (
                                        2vₙ[ii,jj,kk-3]+
                                      -27vₙ[ii,jj,kk-2]+
                                      270vₙ[ii,jj,kk-1]+
                                      270vₙ[ii,jj,kk+1]+
                                      -27vₙ[ii,jj,kk+2]+
                                        2vₙ[ii,jj,kk+3]+

                                        2vₙ[ii,jj-3,kk]+
                                      -27vₙ[ii,jj-2,kk]+
                                      270vₙ[ii,jj-1,kk]+
                                      270vₙ[ii,jj+1,kk]+
                                      -27vₙ[ii,jj+2,kk]+
                                        2vₙ[ii,jj+3,kk]+

                                        2vₙ[ii-3,jj,kk]+
                                      -27vₙ[ii-2,jj,kk]+
                                      270vₙ[ii-1,jj,kk]+
                                      270vₙ[ii+1,jj,kk]+
                                      -27vₙ[ii+2,jj,kk]+
                                        2vₙ[ii+3,jj,kk]+
                                        
                                   -3*490vₙ[ii,  jj,kk]
                                    ) / 180 + 

                                    2 * vₙ[ii, jj,kk] - vₙ₋₁[ii, jj,kk]; # 
          vₙ₊₁[ii, jj,kk] *= loss
        
        if _absorbing
        @inbounds for bb in 1:bndry
          vₙ₊₁[bb,jj,kk] = vₙ[bb+1 ,jj,kk ] + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[bb+1, jj,kk]-vₙ[bb, jj, kk]);# x = 0
          vₙ₊₁[ii,bb,kk] = vₙ[ii, bb+1 ,kk] + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii, bb+1,kk]-vₙ[ii, bb, kk]);# y = 0
          vₙ₊₁[ii,jj,bb] = vₙ[ii,jj, bb+1 ] + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii,jj, bb+1]-vₙ[ii, jj, bb]);# z = 0
          vₙ₊₁[nx-3+bb,jj,kk] = vₙ[nx-4+bb,jj,kk]  + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[nx-4+bb,jj,kk]-vₙ[nx-3+bb,jj,kk]);# x = N
          vₙ₊₁[ii,ny-3+bb,kk] = vₙ[ii,ny-4+bb,kk]  + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii,ny-4+bb,kk]-vₙ[ii,ny-3+bb,kk]);# y = N
          vₙ₊₁[ii,jj,nz-3+bb] = vₙ[ii,jj,nz-4+bb]  + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii,jj,nz-4+bb]-vₙ[ii,jj,nz-3+bb]);# z = N
        end
        end
        end
      end
    end
  end

  return mat;
end

function oh6_drops(vₙ::Array{Float64,3}, 
             vₙ₊₁::Array{Float64,3}, 
             α::Array{Float64,3}, 
             κ::Array{Float64,3}, 
             _xs::LinRange, 
             _ys::LinRange,
             _zs::LinRange ,
             n::Int64;
             _absorbing=true)
  
  ##############################################################################
  println()
  println("Running O(h⁶) approximation...")
  println("==============================")
  nx,ny,nz = length(_xs), length(_ys), length(_zs)
  println("Allocating...")
  mat = zeros(Float32,n, nx, ny,nz)
  println("Allocated 4D array! (of size $(Base.format_bytes(sizeof(mat))))")
  @assert sizeof(mat) < MAX_MEMORY "MAX_MEMORY variable exceeded, check waves_FDTD for array memory limit"
  ##############################################################################

  envelope_field = copy(vₙ₊₁)#[exp(-60 * (x^2+y^2+z^2))]
  loss = 1.0
  bndry = 3

  to_choose_from = CartesianIndices((4:nx-4,4:ny-4, 4:nz-4))
  

  # ω     = 2π * 0.02
  # vₙ₊₁[:,:,nz-4] .= envelope_field
  @inbounds @fastmath for tt in ProgressBar(1:n)
    mat[tt,:,:,:] = copy(vₙ)
    vₙ₋₁ = copy(vₙ);
    #vₙ₊₁[:,:,nz-4] .= cos(ω*tt) .* envelope_field[:,:,nz÷2+1]
    #vₙ₊₁[:,:,:] .= cos(ω*tt) .* copy(envelope_field)
    vₙ = copy(vₙ₊₁);
    if (tt-1) % 25 == 0
      rand_center = (rand(3)*2 .- 1) * _xs[end]
      vₙ += exp.( -60 .*( 
               reshape(_xs .- rand_center[1],(nx,1,1)).^2 
            .+ reshape(_ys .- rand_center[2],(1,ny,1)).^2 
            .+ reshape(_zs .- rand_center[3],(1,1,nz)).^2)
      ) 


      # vₙ[rand(to_choose_from)] += [exp(-60 * (x))]
    end
    @inbounds @fastmath @simd for kk in 4:nz-3
      @inbounds for jj in 4:ny-3
        @inbounds for ii in 4:nx-3
          vₙ₊₁[ii, jj,kk]  = α[ii,jj,kk] * (
                                        2vₙ[ii,jj,kk-3]+
                                      -27vₙ[ii,jj,kk-2]+
                                      270vₙ[ii,jj,kk-1]+
                                      270vₙ[ii,jj,kk+1]+
                                      -27vₙ[ii,jj,kk+2]+
                                        2vₙ[ii,jj,kk+3]+

                                        2vₙ[ii,jj-3,kk]+
                                      -27vₙ[ii,jj-2,kk]+
                                      270vₙ[ii,jj-1,kk]+
                                      270vₙ[ii,jj+1,kk]+
                                      -27vₙ[ii,jj+2,kk]+
                                        2vₙ[ii,jj+3,kk]+

                                        2vₙ[ii-3,jj,kk]+
                                      -27vₙ[ii-2,jj,kk]+
                                      270vₙ[ii-1,jj,kk]+
                                      270vₙ[ii+1,jj,kk]+
                                      -27vₙ[ii+2,jj,kk]+
                                        2vₙ[ii+3,jj,kk]+
                                        
                                   -3*490vₙ[ii,  jj,kk]
                                    ) / 180 + 

                                    2 * vₙ[ii, jj,kk] - vₙ₋₁[ii, jj,kk]; # 
          vₙ₊₁[ii, jj,kk] *= loss
        
        if _absorbing
        @inbounds for bb in 1:bndry
          vₙ₊₁[bb,jj,kk] = vₙ[bb+1 ,jj,kk ] + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[bb+1, jj,kk]-vₙ[bb, jj, kk]);# x = 0
          vₙ₊₁[ii,bb,kk] = vₙ[ii, bb+1 ,kk] + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii, bb+1,kk]-vₙ[ii, bb, kk]);# y = 0
          vₙ₊₁[ii,jj,bb] = vₙ[ii,jj, bb+1 ] + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii,jj, bb+1]-vₙ[ii, jj, bb]);# z = 0
          vₙ₊₁[nx-3+bb,jj,kk] = vₙ[nx-4+bb,jj,kk]  + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[nx-4+bb,jj,kk]-vₙ[nx-3+bb,jj,kk]);# x = N
          vₙ₊₁[ii,ny-3+bb,kk] = vₙ[ii,ny-4+bb,kk]  + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii,ny-4+bb,kk]-vₙ[ii,ny-3+bb,kk]);# y = N
          vₙ₊₁[ii,jj,nz-3+bb] = vₙ[ii,jj,nz-4+bb]  + (κ[ii,jj,kk]-1)/(κ[ii,jj,kk]+1) * (vₙ₊₁[ii,jj,nz-4+bb]-vₙ[ii,jj,nz-3+bb]);# z = N
        end
        end
        end
      end
    end
  end

  return mat;
end
end

#= === kernels === =#
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