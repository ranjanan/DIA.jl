import AlgebraicMultigrid: gs!, Sweep, Smoother, GaussSeidel, Level, extend_heirarchy!


struct RedBlackSweep <: Sweep
	ind1::CuVector{Int64}
	ind2::CuVector{Int64}
end
GaussSeidel(rb::RedBlackSweep) = GaussSeidel(rb, 1)
function (s::GaussSeidel{RedBlackSweep})(A::SparseMatrixDIA{T}, x::CuVector{T}, b::CuVector{T}) where {T}
	# @assert eltype(A.diags)==Pair{Int64, CuVector{T}} || ArgumentError("only CuDIA allowed") 
	for i in 1:s.iter
		gs!(A, b, x, s.sweep.ind1, s.sweep.ind2)
	end
end

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

function gs!(A::SparseMatrixDIA, b::CuVector, x::CuVector; ind1=CuArray(1:2:length(b)), ind2=CuArray(2:2:length(b)))
	_gs!(A, b, x, ind1)
	_gs!(A, b, x, ind2)
end



###### Multilevel construction
struct PR_op{T}
	ind_from::AbstractVector{Int64} 
	ind_to::AbstractVector{Int64}
	weights::AbstractVector{T} ## All three vectors should have same length(length of finer grid)
end
function mul!(to::CuVector{T}, P::PR_op, from::CuVector{T}) where {T}
	function kernel(from, to, indf, indt, w)
		i = (blockIdx().x-1) * blockDim().x + threadIdx().x
		if i<=length(indf)
			@atomic to[indt[i]] += w[i] * from[indf[i]]
		end
		return
	end
	@cuda threads=64 blocks=ceil(Int, length(P.ind_from)/64) kernel(from, to, P.ind_from, P.ind_to, P.weights)
end		

function gmg_interpolation(A::SparseMatrixDIA{T,TF,CuVector}, gridsize, divunit, indexing) where {T,TF}
	coarse_size = ceil.(Int64, gridsize ./ divunit)
	ind_f, ind_t, weights = cuzeros(Int64, prod(gridsize)), cuzeros(Int64, prod(gridsize)), cuzeros(T, prod(gridsize))
	
	function kernel(indexing, ind_f, ind_t, weights, gridsize, divunit, coarse_size)
		i = (blockIdx().x-1) * blockDim().x + threadIdx().x
		j = (blockIdx().y-1) * blockDim().y + threadIdx().y
		k = (blockIdx().z-1) * blockDim().z + threadIdx().z
		if i<=gridsize[1] && j<=gridsize[2] && k<=gridsize[3]
			i2, j2, k2 = ceil.(Int, (i, j, k)./divunit)
			nd = indexing(gridsize..., i, j, k)
			nd_coarse = indexing(coarse_size..., i2, j2, k2)
			ind_f[nd] = nd
			ind_t[nd] = nd_coarse
			weights[nd] = 1.0 # https://calcul.math.cnrs.fr/attachments/spip/IMG/pdf/aggamgtut_notay.pdf page 56
		end
		return nothing
	end

	max_threads = 256
	threads_x   = min(max_threads, gridsize[1])
	threads_y   = min(max_threads ÷ threads_x, gridsize[2])
   	threads_z   = min(max_threads ÷ threads_x ÷ threads_y, gridsize[3])
   	threads     = (threads_x, threads_y, threads_z)
   	blocks      = ceil.(Int, gridsize ./ threads)
	@cuda threads=threads blocks=blocks kernel(indexing, ind_f, ind_t, weights, gridsize, divunit, coarse_size)

	P = PR_op(ind_t, ind_f, weights)
	R = PR_op(ind_f, ind_t, weights)

	return P, R
end

function gmg_PAP_diag(indexing, rev_indexing, offset_from, diag_from, diag_offdiag, diag_main, gridsize, divunit)
	coarse_size = ceil.(Int64, gridsize ./ divunit)		
	i = (blockIdx().x-1) * blockDim().x + threadIdx().x
	j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    k = (blockIdx().z-1) * blockDim().z + threadIdx().z
	nd = indexing(gridsize..., i, j, k) ### row number

	#  Coeff nd->nd_nb becomes nd_coarse->nd_nb_coarse, and in ijk (i,j,k)->ind_nb becomes ind_nd_coarse->ind_nb_coarse
	if nd>max(-offset_from, 0) && nd<min(prod(gridsize), prod(gridsize)-offset_from)
		nd_nb = nd + offset_from
		ind_nb = rev_indexing(gridsize..., nd_nb) 
		ind_nd_coarse = ceil.(Int, (i, j, k)./divunit)
		ind_nb_coarse = ceil.(Int, ind_nb ./ divunit)
		nd_coarse = indexing(coarse_size..., ind_nd_coarse)
		nd_nb_coarse = indexing(coarse_size..., ind_nb_coarse)
		value_diag = offset_from > 0 ? diag_from[nd] : diag_from[nd+offset_from]
		if nd_coarse==nd_nb_coarse ## Me and neighbor in the same cell in coarse level
			@atomic diag_main[nd_coarse] += value_diag
		elseif nd_coarse>nd_nb_coarse ## subdiagonal
			@atomic diag_offdiag[nd_nb_coarse] += value_diag
		else 
			@atomic diag_offdiag[nd_coarse] += value_diag
		end
	end
end	

function extend_heirarchy!(levels, A::SparseMatrixDIA{T, TF, CuVector}, gridsize, divunit, indexing, rev_indexing) where {T, TF}
	P, R = gmg_interpolation(A, gridsize, divunit, indexing)
	c_length  = prod(ceil.(gridsize ./ divunit))
	c_offsets = [rev_indexing(gridsize..., A.diags[i].first) for i in 1:length(A.diags)]
	A_c = SparseMatrixDIA([c_offsets[i]->cuzero(T, c_length - abs(c_offsets[i])) for i in 1:length(A.diags)])
	d   = length(A.diags)>>1+1 # main diag index
	
	max_threads = 256
    threads_x   = min(max_threads, gridsize[1])
    threads_y   = min(max_threads ÷ threads_x, gridsize[2])
    threads_z   = min(max_threads ÷ threads_x ÷ threads_y, gridsize[3])
    threads     = (threads_x, threads_y, threads_z)
    blocks      = ceil.(Int, gridsize ./ threads)

	for i in 1:length(A.diags)
		@cuda threads=threads blocks=blocks gmg_PAP_diag(indexing, rev_indexing, A.diags[i].first, A.diags[i].second, A_c.diags[i].second, A_c.diags[d].second, gridsize, divunit)
	end
	
	push!(levels, Level(A, P, R))	
	return A_c
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
