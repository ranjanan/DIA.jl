module DIA

using SparseArrays
using LinearAlgebra

export SparseMatrixDIA

struct SparseMatrixDIA{Tv,Ti,N,V<:AbstractArray{Tv}} <: AbstractSparseMatrix{Tv,Ti}
    diags::NTuple{N, Pair{Ti,V}}
    m::Ti
    n::Ti
end
# SparseMatrixDIA(x::Pair{Ti,V<:AbstractVector{Tv}}) where {Ti,Tv,V} = SparseMatrixDIA((x,), length(x.second), length(x.second))
Base.size(a::SparseMatrixDIA) = (a.m, a.n)
function SparseArrays.nnz(a::SparseMatrixDIA)
    l = 0
    for x in a.diags
        per = length(x.second[1])
        l += length(x.second) * per
    end
    l
end


function Base.getindex(a::SparseMatrixDIA{Tv,Ti,N}, i, j) where {Tv,Ti,N}
    diff = j - i
    ind = ifelse(diff > 0, i, j)
    for x in a.diags
        if x.first == diff
            return x.second[ind]
        end
    end
    return zero(Tv)
end

Base.show(io::IO, S::SparseMatrixDIA) = Base.show(convert(IOContext, io), S::SparseMatrixDIA)
function Base.show(io::IOContext, S::SparseMatrixDIA)
    
    println(io, summary(S))
    for x in S.diags
        print(io, "Diagonal $(x.first): ")
        print(io, x.second)
        print(io, '\n')
    end
    
end
Base.display(S::SparseMatrixDIA) = Base.show(S)

function Base.summary(S::SparseMatrixDIA{Tv,Ti,N,V}) where {Tv,Ti,N,V} 
    "$(S.m)×$(S.n) SparseMatrixDIA{$Tv,$Ti,$N} with $(length(S.diags)) diagonals: "
end

function Base.:*(S::SparseMatrixDIA{Tv1,Ti,N,V}, b::Vector{Tv2}) where {Tv1,Tv2,Ti,N,V}
    mul!(zeros(Tv2, length(b)), S, b)
end

# Matrix Vector product 
function LinearAlgebra.mul!(ret::Vector{Tv2}, S::SparseMatrixDIA{Tv1,Ti,N,V}, 
                            b::Vector{Tv2}) where {Tv1,Tv2, Ti,N,V<:DenseVector}
    @assert S.n == length(b) || throw(DimensionMismatch("Matrix - vector sizes do not match"))
    d = S.diags
    fill!(ret, zero(Tv2))
    for x in d
        s = x.second
        offset = x.first
        l = length(s)
        if offset >= 0 
            for j = 1:l
                @inbounds ret[j] += s[j] * b[j + offset] 
            end
        else 
            for j = 1:l
                @inbounds ret[j-offset] += s[j] * b[j] 
            end
        end
    end
    ret
end
function LinearAlgebra.mul!(ret::Vector{Tv}, S::SparseMatrixDIA{Tv,Ti,N,V}, 
                            b::Vector{Tv}) where {Tv,Ti,N,V<:SparseVector}
    @assert S.n == length(b) || throw(DimensionMismatch("Matrix - vector sizes do not match"))
    d = S.diags
    fill!(ret, zero(Tv))
    for x in d
        s = x.second
        nzval = s.nzval
        nzind = s.nzind
        offset = x.first
        l = length(s)
        if offset >= 0 
            for (idx,j) in enumerate(nzind)
                @inbounds ret[j] += nzval[idx] * b[j + offset] 
            end
        else
            for (idx,j) in enumerate(nzind)
                @inbounds ret[j-offset] += nzval[idx] * b[j] 
            end
        end
    end
    ret
end

function BLAS.gemv!(tA, alpha, A::SparseMatrixDIA{Tv1,Ti,N,V}, b::Vector{Tv2}, beta, ret::Vector{Tv2}) where {Tv1,Tv2,Ti,N,V}
    @assert S.n == length(b) || throw(DimensionMismatch("Matrix - vector sizes do not match"))
    d = S.diags
    fill!(ret, zero(Tv2))
    for x in d 
        s = x.second
        offset = x.first
        l = length(s)
        if offset >= 0 
            for j = 1:l
                @inbounds ret[j] = beta * ret[j] + alpha * s[j] * b[j + offset] 
            end
        else 
            for j = 1:l
                @inbounds ret[j-offset] = beta * ret[j-offset] + alpha * s[j] * b[j] 
            end
        end
    end
    ret
end

### Conversion 
Base.Matrix(s::SparseMatrixDIA) = diagm(s.diags...)

# TODO: Speed this up
function SparseMatrixDIA(S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    m,n = size(S)
    s = Vector{Pair{Ti,Vector{Tv}}}()
    # Add diagonals above
    for i = 1:n
        d = diag(S, i-1)
        if nnz(d) != 0
            push!(s, i-1 => Vector(d))
        end
    end
    for i = -1:-1:-n
        d = diag(S, i)
        if nnz(d) != 0
            push!(s, i => Vector(d))
        end
    end
    SparseMatrixDIA(tuple(s...), m, n)
end

end # end module

        
