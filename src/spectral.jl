module sp

using DifferentialEquations

const v0      = 0.353
const u0      = 11.7
const vn      = 0.1
const nu      = 0.1826
const muInv   = 491.0
const Dv      = 0.0001

const Nkx = 16
const Nky = 16
const Nx  = 2*Nkx+1
const Ny  = 2*Nky+1

const Lx    = 10*pi
const Ly    = 10*pi

indexToK(x :: Int,Nk :: Int) = x <= Nk + 1 ? x - 1 : x - 2*Nk -2
kToIndex(k :: Int,Nk :: Int) = k >= 0      ? k + 1 : 2*Nk + 2 + k

const ddx    = Complex{Float64}[ 2*pi*im*indexToK(x,Nkx)/Lx for x in 1:Nx, y in 1:Ny]
const ddy    = Complex{Float64}[ 2*pi*im*indexToK(y,Nky)/Ly for x in 1:Nx, y in 1:Ny]
const Lap    = ddx.*ddx .+ ddy.*ddy
const invLap = Complex{Float64}[ abs(Lap[x,y]) == 0 ? 0.0+0.0im : 1.0/Lap[x,y]  for x in 1:Nx, y in 1:Ny]

function funLin(t ,state ::Array{Complex{Float64},3},dState ::Array{Complex{Float64},3})
    n     = @view state[:,:,1]
    theta = @view state[:,:,2]
    eta   = @view state[:,:,3]
    # do I need pre-allocation?
    phi = muInv.*(eta .- n).*invLap
    chi = theta.*invLap
    nl1 = fft( ifft(n).*ifft(theta) .+ ifft(ddx.*n).*ifft(ddx.*chi) .+ ifft(ddy.*n).*ifft(ddy.*chi) )
    nl2 = fft( ifft(ddx.*theta).*ifft(ddx.*chi) .+ ifft(ddy.*theta).*ifft(ddy.*chi) )
    nl3 = fft( ifft(ddx.*phi).*ifft(ddy.*n) .- ifft(ddy.*phi).*ifft(ddx.*n) )

    dState[:,:,1] .= -v0.*ddx.*n     .+ theta                           .+ nl1 .- Dv.*n
    dState[:,:,2] .= -v0.*ddx.*theta .+ eta .- n                        .+ nl2 .- Dv.*theta
    dState[:,:,3] .= -u0.*ddy.*eta   .- nu.*(eta .- n) .- vn.*ddy.*phi  .+ nl3 .- Dv.*eta

    return nothing
end

#initial conditions
state0  = zeros(Complex{Float64},Nx,Ny,3)
state0 .= 0.01

prob = ODEProblem(funLin,state0)
tspan = [0,10]

sol =solve(prob,tspan,Δt=0.001,adaptive=false,maxiters=10^5,timeseries_steps = 100)
#`save_timeseries=false` will turn off all intermediate saving and just give you the end.
#sol =solve(prob,tspan,internalnorm=mynorm)

end

import sp

using Plots
using CurveFit

t   = sp.sol.t
sol = sp.sol[:]

logen(kx,ky) = [ log(abs(s[sp.kToIndex(kx,sp.Nkx),sp.kToIndex(ky,sp.Nky),1])) for s in sol]


gamas = zeros(Float64,1+sp.Nkx,1+sp.Nky)

tMin = 50
for kx = 0:sp.Nkx, ky = 0:sp.Nky
    gamas[kx+1,ky+1] = linear_fit(t[tMin:end],logen(kx,ky)[tMin:end])[2]
end

kxs = 2*pi*collect(0:sp.Nkx)/sp.Lx
kys = 2*pi*collect(0:sp.Nky)/sp.Ly
plot(kxs,kys,gamas,st=:contourf)
