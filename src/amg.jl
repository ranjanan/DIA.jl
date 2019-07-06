import AlgebraicMultigrid: gs!

function _gs_diag!(offset, diag, x, i)
    if   offset<0 x[i] -= diag[i+offset] * x[i+offset] ## i+offset should be >0
    else x[i] -= diag[i] * x[i+offset] end             ## i+offset should be <=length(n)
end
function _gs_kernel!(offset, diag, x, ind)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i<=length(ind) && ind[i]+offset>0 && ind[i]+offset<=length(x) _gs_diag!(offset, diag, x, ind[i]) end
    return
end
function _gs!(A::SparseMatrixDIA, b, x, ind) ## Performs GS on subset ind ⊂ 1:length(x), ind must be CuArray
    n = length(ind)
    _copy_cuind!(b, x, ind)
    for i in 1:length(A.diags)
            if A.diags[i].first != 0
                    @cuda threads=64 blocks=ceil(Int, n/64) _gs_kernel!(A.diags[i].first, A.diags[i].second, x, ind)
            end
    end
    _div_cuind!(x, A.diags[length(A.diags)>>1 + 1].second, ind) ### temp because length(A.diags)>>1+1 is the main diagonal
end
function gs!(A::SparseMatrixDIA, b::CuVector, x::CuVector; ind = CuVector{Float32}())
	_gs(A, b, x, ind)
end


### Copy and Divide with CuArray index (red or black)
function _copy_cuind!(from, to, ind)
    function kernel(from, to, ind)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i<=length(ind) to[ind[i]] = from[ind[i]] end
        return
    end
    @cuda threads=64 blocks=ceil(Int, length(ind)/64) kernel(from, to, ind)
end
function _div_cuind!(x, y, ind) ## x /= y for ind
    function kernel(x, y, ind)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if i<=length(ind) x[ind[i]] /= y[ind[i]] end
        return
    end
    @cuda threads=64 blocks=ceil(Int, length(ind)/64) kernel(x, y, ind)
end
