module DIA

using SparseArrays
using LinearAlgebra
using CuArrays
using CUDAnative

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

function Base.:*(S::SparseMatrixDIA{Tv,Ti,N,V}, b::Vector{Tv}) where {Tv,Ti,N,V}
    mul!(zeros(Tv, length(b)), S, b)
end

# Matrix Vector product 
function LinearAlgebra.mul!(ret::Vector{Tv}, S::SparseMatrixDIA{Tv,Ti,N,V}, 
                            b::Vector{Tv}) where {Tv,Ti,N,V<:DenseVector}
    @assert S.n == length(b) || throw(DimensionMismatch("Matrix - vector sizes do not match"))
    d = S.diags
    fill!(ret, zero(Tv))
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

# GPU Matvec
function gpumatvec!(ret::CuVector, S::SparseMatrixDIA{Tv,Ti,N,V}, 
                            b::CuVector) where {Tv,Ti,N,V}
    @assert S.n == length(b) || throw(DimensionMismatch("Matrix - vector sizes do not match"))
    d = S.diags
    fill!(ret, zero(Tv))
    for x in d
        s = x.second
        offset = x.first
        l = length(s)
        if offset >= 0 
            @cuda threads=256 dot_add_with_offset1!(ret, s, b, offset) 
        else 
            @cuda threads=256 dot_add_with_offset2!(ret, s, b, offset) 
        end
    end
    ret
end
function dot_add_with_offset1!(y, x, z, c)
    index = threadIdx().x 
    stride = blockDim().x
    for i = index:stride:length(x)
        @inbounds y[i] += x[i] * z[i+c]
    end
    return nothing
end
function dot_add_with_offset2!(y, x, z, c)
    index = threadIdx().x 
    stride = blockDim().x
    for i = index:stride:length(x)
        @inbounds y[i-c] += x[i] * z[i]
    end
    return nothing
end

# If you change the variable names, it gives me compilation error. Strange. 
#=function dot_add_with_offset1!(ret, s, b)
    index = threadIdx().s    # this example only requires linear indexing, so just use `x`
    stride = blockDim().s
    for i = index:stride:length(s)
        @inbounds ret[i] += s[i] * b[i]
    end
    return nothing
end
function dot_add_with_offset2!(ret, s, b, offset)::nothing
    index = (blockIdx().s - 1) * blockDim().s + threadIdx().s
    stride = blockDim().s * gridDim().s
    for i = index:stride:length(s)
        @inbounds ret[i-offset] += s[i] * b[i]
    end
    return nothing
end=#


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
function CuArrays.cu(S::SparseMatrixDIA{Tv,Ti,N,V}) where {Tv,Ti,N,V}
	m, n = size(S)
	s = Vector{Pair{Ti,CuVector{Float32}}}(undef, N)
	for (i,x) in enumerate(S.diags)
		first = x.first
		second = x.second
		s[i] = first => cu(x.second)
	end
	SparseMatrixDIA(tuple(s...), m, n)
end
		

end # end module

        
