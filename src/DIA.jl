module DIA

using SparseArrays
using LinearAlgebra
using CuArrays
using CUDAnative
using AlgebraicMultigrid

include("base.jl")
include("convert.jl")
include("linalg.jl")
include("gs.jl")
include("amg.jl")


end # end module
