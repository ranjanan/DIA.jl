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

function Base.:*(S::SparseMatrixDIA{Tv,Ti,N,V}, b::Vector{Tv}) where {Tv,Ti,N,V}
    mul!(similar(b), S, b)
end

# Matvec
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
# GPU mul!
function LinearAlgebra.mul!(ret::CuVector, S::SparseMatrixDIA{Tv1,Ti,N,V},
                            b::CuVector{Tv2}) where {Tv1,Tv2,Ti,N,V}
    @assert S.n == length(b) || throw(DimensionMismatch("Matrix - vector sizes do not match"))
    d = S.diags
    fill!(ret, zero(Tv2))
    function kernel_1(ret, strip, b, offset)  ## Case of offset >=0
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i <= length(strip) @inbounds ret[i] += strip[i] * b[i+offset] end
        return
    end
    function kernel_2(ret, strip, b, offset) ## Case of offset < 0
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i <= length(strip) @inbounds ret[i-offset] += strip[i] * b[i] end
        return
    end

    for x in d
        s = x.second
        offset = x.first
        if offset >= 0
            @cuda threads=256 blocks=ceil(Int, length(s)/256) kernel_1(ret, s, b, offset)
        else
            @cuda threads=256 blocks=ceil(Int, length(s)/256) kernel_2(ret, s, b, offset)
        end
    end
    ret
end

#=function dot_add_with_offset1!(y, x, z, c, alpha=1., beta=1.)
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(x)
        @inbounds y[i] = beta * y[i] + alpha * x[i] * z[i+c]
    end
    return nothing
end
function dot_add_with_offset2!(y, x, z, c, alpha=1., beta=1.0)
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(x)
        @inbounds y[i-c] = beta * y[i-c] + alpha* x[i] * z[i]
    end
    return nothing
end

# If you change the variable names, it gives me compilation error. Strange.
function dot_add_with_offset1!(ret, s, b)
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

function BLAS.gemv!(tA::Char, alpha::Number, S::SparseMatrixDIA{Tv1,Ti,N,V}, b::Vector{Tv2}, beta::Number, ret::Vector{Tv2}) where {Tv1,Tv2,Ti,N,V}
    @assert S.n == length(b) || throw(DimensionMismatch("Matrix - vector sizes do not match"))
    d = S.diags
    rmul!(ret, beta)
    for x in d
        s = x.second
        offset = x.first
        l = length(s)
        if offset >= 0
            for j = 1:l
                @inbounds ret[j] += alpha * s[j] * b[j + offset]
            end
        else
            for j = 1:l
                @inbounds ret[j-offset] += alpha * s[j] * b[j]
            end
        end
    end
    ret
end
function BLAS.gemv!(tA::Char, alpha, S::SparseMatrixDIA{Tv,Ti,N,V},
                    b::CuVector, beta, ret::CuVector) where {Tv,Ti,N,V}
    @assert S.n == length(b) || throw(DimensionMismatch("Matrix - vector sizes do not match"))
    d = S.diags
    rmul!(ret, beta)
    function kernel_1(α, strip, b, ret, offset)  ## Case of offset >=0
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i <= length(strip) @inbounds ret[i] += α * strip[i] * b[i+offset] end
        return
    end
    function kernel_2(α, strip, b, ret, offset) ## Case of offset < 0
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i <= length(strip) @inbounds ret[i-offset] += α * strip[i] * b[i] end
        return
    end
    for x in d
        s = x.second
        offset = x.first
        l = length(s)
        if offset >= 0
            @cuda threads=256 blocks=ceil(Int, length(s)/256) kernel_1(alpha, s, b, ret, offset)
        else
            @cuda threads=256 blocks=ceil(Int, length(s)/256) kernel_2(alpha, s, b, ret, offset)
        end
    end
    ret
end

function BLAS.gemv!(tA::Char, alpha, S::SparseMatrixDIA{Tv,Ti,N,V},
                    b::CuVector, beta, ret::CuVector) where {Tv,Ti,N,V}
    @assert S.n == length(b) || throw(DimensionMismatch("Matrix - vector sizes do not match"))
    d = S.diags
    rmul!(ret, beta)
    function kernel(α, strip, b, ret)  ## Case of offset >=0
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i <= length(strip) @inbounds ret[i] += α * strip[i] * b[i] end
	return
    end
    for x in d
        s = x.second
        offset = x.first
        l = length(s)
        if offset >= 0
            @cuda threads=256 blocks=ceil(Int, length(s)/256) kernel(alpha, s, view(b, offset+1:S.n), view(ret, 1:S.n-offset))
        else
            @cuda threads=256 blocks=ceil(Int, length(s)/256) kernel(alpha, s, view(b, 1:S.n+offset), view(ret, 1-offset:S.n))
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
    R = figure_out_type(V)
        s = Vector{Pair{Ti,R}}(undef, N)
        for (i,x) in enumerate(S.diags)
                first = x.first
                second = x.second
                s[i] = first => cu(x.second)
        end
        SparseMatrixDIA(tuple(s...), m, n)
end
figure_out_type(::Type{Vector{Float64}}) = CuVector{Float32}
figure_out_type(::Type{Vector{S}}) where S = CuVector{S}

end # end module
