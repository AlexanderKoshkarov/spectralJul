module sp

using DifferentialEquations

const v0      = 0.353
const u0      = 11.7
const vn      = 0.1
const nu      = 0.1826
const muInv   = 491.0
const Dv      = 0.0001

const Nkx = 31
const Nky = 31
const Nx  = 2*Nkx+1
const Ny  = 2*Nky+1

const Lx    = 5*pi
const Ly    = 10*pi

const dt       = 0.0001
const outputFr = 1000

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
    nl1 = (1/(Nx*Ny))*fft( bfft(deal.*n).*bfft(deal.*theta) .+ bfft(deal.*ddx.*n).*bfft(deal.*ddx.*chi) .+ bfft(deal.*ddy.*n).*bfft(deal.*ddy.*chi) )
    nl2 = (1/(Nx*Ny))*fft( bfft(deal.*ddx.*theta).*bfft(deal.*ddx.*chi) .+ bfft(deal.*ddy.*theta).*bfft(deal.*ddy.*chi) )
    nl3 = (1/(Nx*Ny))*fft( bfft(deal.*ddx.*phi).*bfft(deal.*ddy.*n) .- bfft(deal.*ddy.*phi).*bfft(deal.*ddx.*n) )

    dState[:,:,1] .= -v0.*ddx.*n     .+ theta                           .+ nl1 .- Dv.*Lap.*Lap.*n
    dState[:,:,2] .= -v0.*ddx.*theta .+ eta .- n                        .+ nl2 .- Dv.*Lap.*Lap.*theta
    dState[:,:,3] .= -u0.*ddy.*eta   .- nu.*(eta .- n) .- vn.*ddy.*phi  .+ nl3 .- Dv.*Lap.*Lap.*eta

    return nothing
end

#initial conditions
state0  = zeros(Complex{Float64},Nx,Ny,3)
kx0s = kToIndex.(vcat(1:2,-(1:2)),Nkx)
ky0s = kToIndex.(vcat(1:2,-(1:2)),Nky)
state0[kx0s,ky0s,1] .= 0.0001

prob = ODEProblem(funLin,state0)

sol = solve(prob,[0,30],Δt=dt,adaptive=false,maxiters=10^6,timeseries_steps = outputFr)

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
