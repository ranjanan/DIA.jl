# Matvec
import LinearAlgebra: mul!

function mul!(ret::Vector{Tv2}, S::SparseMatrixDIA{Tv1,Ti,V}, 
				b::Vector{Tv2}) where {Tv1,Tv2, Ti,N, V<:Vector}
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
function mul!(ret::CuVector, S::SparseMatrixDIA{Tv,Ti,V},
                            b::CuVector) where {Tv,Ti,V <: CuVector}
    @assert S.n == length(b) || throw(DimensionMismatch("Matrix - vector sizes do not match"))
    d = S.diags
    fill!(ret, zero(Tv))
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

# Sparse Vector
function LinearAlgebra.mul!(ret::Vector{Tv}, S::SparseMatrixDIA{Tv,Ti,V},
                            b::Vector{Tv}) where {Tv,Ti,V<:SparseVector}
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

function BLAS.gemv!(tA::Char, alpha::Float64, S::SparseMatrixDIA{Tv1,Ti,V}, b::Vector{Tv2}, beta::Float64, ret::Vector{Tv2}) where {Tv1,Tv2,Ti,V}
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
#=function BLAS.gemv!(tA::Char, alpha, S::SparseMatrixDIA{Tv,Ti,N,V},
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
end=#
## Which is better? above or below? or is it just the same?

function BLAS.gemv!(tA::Char, alpha::Float64, S::SparseMatrixDIA{Float64,Ti,V},
                    b::CuVector{Float64}, beta::Float64, ret::CuVector{Float64}) where {Ti,V}
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
function BLAS.gemv!(tA::Char, alpha::Float32, S::SparseMatrixDIA{Float32,Ti,V},
                    b::CuVector{Float32}, beta::Float32, ret::CuVector{Float32}) where {Ti,V}
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

function Base.:*(S::SparseMatrixDIA{Tv,Ti,V}, b::Vector{Tv}) where {Tv,Ti,V}
    mul!(similar(b), S, b)
end
