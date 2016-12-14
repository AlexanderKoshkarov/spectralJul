module sp


using DifferentialEquations, Plots


# Constants
const v0    = 0.18
const u0    = 11.7
const LnInv = 2.36
const nu    = 0.28
const muInv = 489.0
const mu    = 1.0/muInv
const Dv    = 1e-4

const Nkx = 40
const Nky = 40
const Nx  = 2*Nkx+1
const Ny  = 2*Nky+1

const NxNl = 128
const NyNl = 128
const nrm  = 1/(NxNl*NyNl)
const rngX = vcat(1:Nkx+1,NxNl-Nkx+1:NxNl)
const rngY = vcat(1:Nky+1,NyNl-Nky+1:NyNl)



const Lx    = 2*pi
const Ly    = 2*pi

const dx = Lx/Nx
const dy = Ly/Ny

const dt = 0.01
const nt = 10000

# end constants

# multithreading for fftw, maybe need a plan
FFTW.set_num_threads(Sys.CPU_CORES)

# hellper functions
indexToK(x :: Int,Nk :: Int) = x <= Nk + 1 ? x - 1 : x - 2*Nk -2
kToIndex(k :: Int,Nk :: Int) = k >= 0      ? k + 1 : 2*Nk + 2 + k
fromVec(u)  = reshape(reinterpret(Complex128,u),Nx,Ny,3)
toVec(u)    = reinterpret(Float64,vec(u))

immutable Rhs
    phi :: Array{Complex128,2}
    chi :: Array{Complex128,2}

    nl1 :: Array{Complex128,2}
    nl2 :: Array{Complex128,2}
    nl3 :: Array{Complex128,2}

    temp1 :: Array{Complex128,2}
    temp2 :: Array{Complex128,2}
    temp3 :: Array{Complex128,2}
    temp4 :: Array{Complex128,2}
    temp5 :: Array{Complex128,2}
    temp6 :: Array{Complex128,2}


    Rhs() = let tp = Complex128
                new(zeros(tp,Nx,Ny)
                    ,zeros(tp,Nx,Ny)
                    ,zeros(tp,NxNl,NyNl)
                    ,zeros(tp,NxNl,NyNl)
                    ,zeros(tp,NxNl,NyNl)
                    ,zeros(tp,NxNl,NyNl)
                    ,zeros(tp,NxNl,NyNl)
                    ,zeros(tp,NxNl,NyNl)
                    ,zeros(tp,NxNl,NyNl)
                    ,zeros(tp,NxNl,NyNl)
                    ,zeros(tp,NxNl,NyNl))
            end
end

const ddx    = Complex128[ 2*pi*im*indexToK(x,Nkx)/Lx for x in 1:Nx, y in 1:Ny]
const ddy    = Complex128[ 2*pi*im*indexToK(y,Nky)/Ly for x in 1:Nx, y in 1:Ny]
const Lap    = ddx.*ddx .+ ddy.*ddy
const hv     = Dv.*Lap.*Lap
const invLap = Complex128[ abs(Lap[x,y]) == 0 ? 0.0+0.0im : 1.0/Lap[x,y]  for x in 1:Nx, y in 1:Ny]

fun = Rhs()
#function (fun :: Rhs)(t,u :: Array{Float64,1},du :: Array{Float64,1})
function (fun :: Rhs)(t,u,du)

    state  = fromVec(u)
    dState = fromVec(du)

    n     = @view state[:,:,1]
    theta = @view state[:,:,2]
    eta   = @view state[:,:,3]
    phi = fun.phi; phi  .= n .+ muInv.*(eta .- n).*invLap
    chi = fun.chi; chi  .= theta.*invLap

    nl1 = fun.nl1; nl1 .= 0
    nl2 = fun.nl2; nl2 .= 0
    nl3 = fun.nl3; nl3 .= 0
    temp1 = fun.temp1;
    temp2 = fun.temp2;
    temp3 = fun.temp3;
    temp4 = fun.temp4;
    temp5 = fun.temp5;
    temp6 = fun.temp6;

    temp1 .= 0; temp1[rngX,rngY] .= n;         bfft!(temp1)
    temp2 .= 0; temp2[rngX,rngY] .= theta;     bfft!(temp2)
    temp3 .= 0; temp3[rngX,rngY] .= ddx.*n;    bfft!(temp3)
    temp4 .= 0; temp4[rngX,rngY] .= ddx.*chi;  bfft!(temp4)
    temp5 .= 0; temp5[rngX,rngY] .= ddy.*n;    bfft!(temp5)
    temp6 .= 0; temp6[rngX,rngY] .= ddy.*chi;  bfft!(temp6)

    nl1 .= nrm .* ( temp1.*temp2 .+ temp3.*temp4 .+ temp5.*temp6 )
    fft!(nl1)

    nl2 .= (0.5.*nrm) .* ( temp4.^2 .+ temp6.^2 )
    fft!(nl2)
    nl2[rngX,rngY] .*= Lap

    temp1 .= 0; temp1[rngX,rngY] .= ddx.*phi
    temp2 .= 0; temp2[rngX,rngY] .= ddy.*phi
    temp3 .= 0; temp3[rngX,rngY] .= ddx.*eta
    temp4 .= 0; temp4[rngX,rngY] .= ddy.*eta

    nl3 .= -nrm .* ( temp1.*temp4 .- temp2.*temp3)
    fft!(nl3)

    #nl4 = mu*(1/(Nx*Ny)).*fft( bfft(deal.*ddx.*ddx.*phi)  .*bfft(deal.*ddy.*ddx.*n)   .- bfft(deal.*ddy.*ddx.*phi)  .*bfft(deal.*ddx.*ddx.*n) .+
    #                           bfft(deal.*ddx.*ddy.*phi)  .*bfft(deal.*ddy.*ddy.*n)   .- bfft(deal.*ddy.*ddy.*phi)  .*bfft(deal.*ddx.*ddy.*n) )

    dState[:,:,1] .= -v0.*ddx.*n     .+ theta                              .- hv.*n       .+ nl1[rngX,rngY]
    dState[:,:,2] .= -v0.*ddx.*theta .+ mu.*Lap.*phi                       .- hv.*theta   .+ nl2[rngX,rngY]
    dState[:,:,3] .=  u0.*ddy.*eta   .- nu.*(eta .- n) .+ LnInv.*ddy.*phi  .- hv.*eta     .+ nl3[rngX,rngY]


    return nothing
end

#initial conditions
state = zeros(Complex128,Nx,Ny,3)
state[1,1,1] = 0.001
state[1,1,2] = 0.001


tspan = (0.0,1.0)
prob = ODEProblem(fun,toVec(state),tspan)

#sol = solve(prob)
sol = solve(prob,CVODE_BDF(method=:Functional))



end
