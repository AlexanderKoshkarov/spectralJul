module sp

using DifferentialEquations

const v₀  = 0.353
const u₀  = 11.7
const vₙ  = 0.1
const ν   = 0.1826
const μ   = 1/491.0
const Dᵥ  = 0.0001

const Nkx = 15
const Nky = 15
const Nx  = 2*Nkx+1
const Ny  = 2*Nky+1

const Lx    = 5*pi
const Ly    = 5*pi

indexToK(x :: Int,Nk :: Int) = x <= Nk + 1 ? x - 1 : x - 2*Nk -2
kToIndex(k :: Int,Nk :: Int) = k >= 0      ? k + 1 : 2*Nk + 2 + k

const ddx    = Complex{Float64}[ 2*pi*im*indexToK(x,Nkx)/Lx for x in 1:Nx, y in 1:Ny]
const ddy    = Complex{Float64}[ 2*pi*im*indexToK(y,Nky)/Ly for x in 1:Nx, y in 1:Ny]
const Lap    = ddx.*ddx .+ ddy.*ddy
const invLap = Complex{Float64}[ abs(Lap[x,y]) == 0 ? 0.0+0.0im : 1.0/Lap[x,y]  for x in 1:Nx, y in 1:Ny]

function funLin(t ,state ::Array{Complex{Float64},3},dState ::Array{Complex{Float64},3})
    n = @view state[:,:,1]
    θ = @view state[:,:,2]
    η = @view state[:,:,3]
    # do I need pre-allocation?
    ϕ = (1/μ).*(η .- n).*invLap
    χ = θ.*invLap

    dState[:,:,1] .= -v₀.*ddx.*n .+ θ
    dState[:,:,2] .= -v₀.*ddx.*θ .+ η .- n
    dState[:,:,3] .= -u₀.*ddy.*θ .- ν.*(η .- n) .- vₙ.*ddy.*ϕ

    return nothing
end

#initial conditions
state0  = zeros(Complex{Float64},Nx,Ny,3)
state0 .= 0.01

prob = ODEProblem(funLin,state0)
tspan = [0,5]

sol =solve(prob,tspan,Δt=0.001,adaptive=false)

end

import sp
