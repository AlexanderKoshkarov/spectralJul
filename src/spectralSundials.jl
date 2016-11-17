module sp


using Sundials

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

const Lx    = 4*pi
const Ly    = 4*pi

const tspan    = collect(linspace(0,1,11))

indexToK(x :: Int,Nk :: Int) = x <= Nk + 1 ? x - 1 : x - 2*Nk -2
kToIndex(k :: Int,Nk :: Int) = k >= 0      ? k + 1 : 2*Nk + 2 + k

const ddx    = Complex128[ 2*pi*im*indexToK(x,Nkx)/Lx for x in 1:Nx, y in 1:Ny]
const ddy    = Complex128[ 2*pi*im*indexToK(y,Nky)/Ly for x in 1:Nx, y in 1:Ny]
const Lap    = ddx.*ddx .+ ddy.*ddy
const invLap = Complex128[ abs(Lap[x,y]) == 0 ? 0.0+0.0im : 1.0/Lap[x,y]  for x in 1:Nx, y in 1:Ny]
const deal   = [sqrt(indexToK(x,Nkx)^2 + indexToK(y,Nky)^2 ) > (2/3)*min(Nkx,Nky) ? 0 : 1 for x in 1:Nx, y in 1:Ny]

#fromVec(u :: Array{Float64,1}   ) :: Array{Complex128,3} = reshape(reinterpret(Complex128,u),Nx,Ny,3)
#toVec(  u :: Array{Complex128,3}) :: Array{Float64,1}    = reinterpret(Float64,vec(u))

fromVec(u)  = reshape(reinterpret(Complex128,u),Nx,Ny,3)
toVec(  u)  = reinterpret(Float64,vec(u))

function fun(t ,u ,du )

    state  = fromVec(u)
    dState = fromVec(du)

    n     = @view state[:,:,1]
    theta = @view state[:,:,2]
    eta   = @view state[:,:,3]
    # do I need pre-allocation?
    phi = muInv.*(eta .- n).*invLap
    chi = theta.*invLap
    nl1 = (1/(Nx*Ny))*fft( bfft(deal.*n).*bfft(deal.*theta) .+ bfft(deal.*ddx.*n).*bfft(deal.*ddx.*chi) .+ bfft(deal.*ddy.*n).*bfft(deal.*ddy.*chi) )
    nl2 = (1/(Nx*Ny))*fft( bfft(deal.*ddx.*theta).*bfft(deal.*ddx.*chi) .+ bfft(deal.*ddy.*theta).*bfft(deal.*ddy.*chi) )
    nl3 = (1/(Nx*Ny))*fft( bfft(deal.*ddx.*phi).*bfft(deal.*ddy.*n) .- bfft(deal.*ddy.*phi).*bfft(deal.*ddx.*n) )

    dState[:,:,1] .= -v0.*ddx.*n     .+ theta                           .- Dv.*Lap.*Lap.*n      .+ nl1
    dState[:,:,2] .= -v0.*ddx.*theta .+ eta .- n                        .- Dv.*Lap.*Lap.*theta  .+ nl2
    dState[:,:,3] .= -u0.*ddy.*eta   .- nu.*(eta .- n) .- vn.*ddy.*phi  .- Dv.*Lap.*Lap.*eta    .+ nl3

    return nothing
end


#initial conditions
state0  = zeros(Complex128,Nx,Ny,3)
for ky = 2:3, kx = -3:3
    x  = kToIndex( kx,Nkx)
    y  = kToIndex( ky,Nky)
    my = kToIndex(-ky,Nky)
    state0[x,y,1]  = 0.0001*rand(Complex128)
    state0[x,my,1] = conj(state0[x,y,1])
end

@show length(toVec(state0))

res = Sundials.cvode(fun,toVec(state0),tspan)

end

import sp
