module sp

using DifferentialEquations

const v0      = 0.353
const u0      = 11.7
const vn      = 0.1
const nu      = 0.1826
const muInv   = 491.0
const Dv      = 0.0001

const Nkx = 63
const Nky = 63
const Nx  = 2*Nkx+1
const Ny  = 2*Nky+1

const Lx    = 8*pi
const Ly    = 8*pi

const tspan    = (0.0,1.0)
#const dt       = 0.001
const outputFr = 100

indexToK(x :: Int,Nk :: Int) = x <= Nk + 1 ? x - 1 : x - 2*Nk -2
kToIndex(k :: Int,Nk :: Int) = k >= 0      ? k + 1 : 2*Nk + 2 + k

const ddx    = Complex{Float64}[ 2*pi*im*indexToK(x,Nkx)/Lx for x in 1:Nx, y in 1:Ny]
const ddy    = Complex{Float64}[ 2*pi*im*indexToK(y,Nky)/Ly for x in 1:Nx, y in 1:Ny]
const Lap    = ddx.*ddx .+ ddy.*ddy
const invLap = Complex{Float64}[ abs(Lap[x,y]) == 0 ? 0.0+0.0im : 1.0/Lap[x,y]  for x in 1:Nx, y in 1:Ny]
const deal   = [sqrt(indexToK(x,Nkx)^2 + indexToK(y,Nky)^2 ) > (2/3)*min(Nkx,Nky) ? 0 : 1 for x in 1:Nx, y in 1:Ny]

function funLin(t ,state ::Array{Complex{Float64},3},dState ::Array{Complex{Float64},3})
    n     = @view state[:,:,1]
    theta = @view state[:,:,2]
    eta   = @view state[:,:,3]
    # do I need pre-allocation?
    phi = muInv.*(eta .- n).*invLap
    chi = theta.*invLap
    #nl1 = (1/(Nx*Ny))*fft( bfft(deal.*n).*bfft(deal.*theta) .+ bfft(deal.*ddx.*n).*bfft(deal.*ddx.*chi) .+ bfft(deal.*ddy.*n).*bfft(deal.*ddy.*chi) )
    #nl2 = (1/(Nx*Ny))*fft( bfft(deal.*ddx.*theta).*bfft(deal.*ddx.*chi) .+ bfft(deal.*ddy.*theta).*bfft(deal.*ddy.*chi) )
    nl3 = (1/(Nx*Ny))*fft( bfft(deal.*ddx.*phi).*bfft(deal.*ddy.*n) .- bfft(deal.*ddy.*phi).*bfft(deal.*ddx.*n) )

    dState[:,:,1] .= -v0.*ddx.*n     .+ theta                           .- Dv.*Lap.*Lap.*n      #.+ nl1
    dState[:,:,2] .= -v0.*ddx.*theta .+ eta .- n                        .- Dv.*Lap.*Lap.*theta  #.+ nl2
    dState[:,:,3] .= -u0.*ddy.*eta   .- nu.*(eta .- n) .- vn.*ddy.*phi  .- Dv.*Lap.*Lap.*eta    .+ nl3

    return nothing
end

#initial conditions
state0  = zeros(Complex{Float64},Nx,Ny,3)
for ky = 4:10, kx = -10:10
    x  = kToIndex( kx,Nkx)
    y  = kToIndex( ky,Nky)
    my = kToIndex(-ky,Nky)
    state0[x,y,1]  = 0.0001*rand(Complex128)
    state0[x,my,1] = conj(state0[x,y,1])
end



prob = ODEProblem(funLin,state0,tspan)

#sol = solve(prob,Δt=dt,adaptive=false,maxiters=10^10,timeseries_steps = outputFr)
sol = solve(prob,RK4)

end

import sp

using Plots
using CurveFit

t   = sp.sol.t
sol = sp.sol[:]

logen(kx,ky) = [ log(abs(s[sp.kToIndex(kx,sp.Nkx),sp.kToIndex(ky,sp.Nky),1])) for s in sol]
logen()      = [ log(sum(abs(s))) for s in sol ]
en = logen()

plot(t,en)

#anim = @animate for i=1:129
#       plot(fftshift(abs(sol[i][:,:,1])),st=:contourf)
#       end

#gamas = zeros(Float64,1+sp.Nkx,1+sp.Nky)
#
#tMin = 50
#for kx = 0:sp.Nkx, ky = 0:sp.Nky
#    gamas[kx+1,ky+1] = linear_fit(t[tMin:end],logen(kx,ky)[tMin:end])[2]
#end
#
#kxs = 2*pi*collect(0:sp.Nkx)/sp.Lx
#kys = 2*pi*collect(0:sp.Nky)/sp.Ly
#plot(kxs,kys,gamas,st=:contourf)



julia> include("src/spectral.jl")
ERROR: LoadError: MethodError: no method matching Array{Complex{Float64},3}()
Closest candidates are:
  Array{Complex{Float64},3}{T,N}(::Tuple{Vararg{Int64,N}}) at boot.jl:310
  Array{Complex{Float64},3}{T}(::Int64, ::Int64, ::Int64) at boot.jl:316
  Array{Complex{Float64},3}{T}(::Any) at sysimg.jl:53
 in macro expansion at /home/kosh/.julia/v0.5/OrdinaryDiffEq/src/integrators/integrator_utils.jl:79 [inlined]
 in ode_solve(::OrdinaryDiffEq.ODEIntegrator{DiffEqBase.RK4,Array{Complex{Float64},3},Complex{Float64},4,Float64,Complex{Float64},Float64,Array{Complex{Float64},3},Array{Complex{Float64},3},sp.#funLin,OrdinaryDiffEq.#ODE_DEFAULT_NORM,OrdinaryDiffEq.#ODE_DEFAULT_CALLBACK,OrdinaryDiffEq.#ODE_DEFAULT_ISOUTOFDOMAIN}) at /home/kosh/.julia/v0.5/OrdinaryDiffEq/src/integrators/fixed_timestep_integrators.jl:116
 in #solve#47(::Float64, ::Bool, ::Int64, ::DiffEqBase.ExplicitRKTableau, ::Bool, ::Void, ::Symbol, ::Bool, ::Bool, ::Array{Float64,1}, ::Array{Float64,1}, ::Bool, ::Float64, ::Rational{Int64}, ::Rational{Int64}, ::Void, ::Void, ::Rational{Int64}, ::Bool, ::Void, ::Void, ::Int64, ::Float64, ::Float64, ::Bool, ::OrdinaryDiffEq.#ODE_DEFAULT_NORM, ::OrdinaryDiffEq.#ODE_DEFAULT_ISOUTOFDOMAIN, ::Bool, ::Int64, ::String, ::Void, ::Array{Any,1}, ::DiffEqBase.#solve, ::DiffEqBase.ODEProblem{Array{Complex{Float64},3},Float64,true,sp.#funLin}, ::Type{DiffEqBase.RK4}, ::Array{Any,1}, ::Array{Any,1}, ::Array{Any,1}) at /home/kosh/.julia/v0.5/OrdinaryDiffEq/src/solve.jl:182
 in solve(::DiffEqBase.ODEProblem{Array{Complex{Float64},3},Float64,true,sp.#funLin}, ::Type{DiffEqBase.RK4}, ::Array{Any,1}, ::Array{Any,1}, ::Array{Any,1}) at /home/kosh/.julia/v0.5/OrdinaryDiffEq/src/solve.jl:19 (repeats 2 times)
 in eval_user_input(::Any, ::Base.REPL.REPLBackend) at ./REPL.jl:64
 in macro expansion at ./REPL.jl:95 [inlined]
 in (::Base.REPL.##3#4{Base.REPL.REPLBackend})() at ./event.jl:68
while loading /home/kosh/jul/spectralJul/src/spectral.jl, in expression starting on line 66
