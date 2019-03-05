module DIA

using SparseArrays
using LinearAlgebra

struct SparseMatrixDIA{Tv,Ti,N} <: AbstractSparseMatrix{Tv,Ti}
    diags::NTuple{N, Pair{Ti,Vector{Tv}}}
    m::Ti
    n::Ti
end
Base.size(a::SparseMatrixDIA) = (a.m, a.n)
function SparseArrays.nnz(a::SparseMatrixDIA)
    l = 0
    for x in a.diags
        l += length(x.second)
    end
    l
end


function Base.getindex(a::SparseMatrixDIA{Tv,Ti,N}, i, j) where {Tv,Ti,N}
    if i == j
        for x in a.diags
            if x.first == 0
                return x.second[i]
            end
        end
    else 
        return zero(Tv)
    end
end

Base.show(io::IO, S::SparseMatrixDIA) = Base.show(convert(IOContext, io), S::SparseMatrixDIA)
function Base.show(io::IOContext, S::SparseMatrixDIA{Tv,Ti,N}) where {Tv,Ti,N}
    
    println(io, summary(S))
    for x in S.diags
        print(io, "Diagonal $(x.first): ")
        print(io, x.second)
        print(io, '\n')
    end
    
end
Base.display(S::SparseMatrixDIA) = Base.show(S)

function Base.summary(S::SparseMatrixDIA{Tv,Ti,N}) where {Tv,Ti,N} 
    "$(S.m)×$(S.n) SparseMatrixDIA{$Tv,$Ti,$N} with $(length(S.diags)) diagonals: "
end

function Base.:*(S::SparseMatrixDIA{Tv,Ti,N}, b::Vector{Tv}) where {Tv,Ti,N}
    mul!(zeros(Tv, length(b)), S, b)
end

function LinearAlgebra.mul!(ret::Vector{Tv}, S::SparseMatrixDIA{Tv,Ti,N}, b::Vector{Tv}) where {Tv,Ti,N}
    @assert S.n == length(b) || throw(DimensionMismatch("Matrix - vector sizes do not match"))
    d = S.diags
    fill!(ret, zero(Tv))
    for x in d
        s = x.second
        offset = x.first
        l = length(s)
        for j = 1:l
            @inbounds ret[j] += s[j] * b[j + offset] 
        end
    end
    ret
end

Base.Matrix(s::SparseMatrixDIA) = diagm(s.diags...)

end # end module

        
