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
	if   offset<0 
		x[i] -= diag[i+offset] * x[i+offset] ## i+offset should be >0
    else 
		x[i] -= diag[i] * x[i+offset] 
	end   ## i+offset should be <=length(n)
end

function _gs_kernel!(offset, diag, x, ind)
	i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i<=length(ind) && ind[i]+offset>0 && ind[i]+offset<=length(x) 
		_gs_diag!(offset, diag, x, ind[i]) 
	end

    return nothing
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

function gmg_interpolation(A::SparseMatrixDIA{T,TF,CuVector{T}}, fdim, agg, ind) where {T,TF}
	
	cdim  = ceil.(Int64, fdim ./ agg)
	ind_f = cuzeros(Int64, prod(fdim))
	ind_t = cuzeros(Int64, prod(fdim))
	w     = cuzeros(T, prod(fdim))
	
	function kernel(ind, ind_f, ind_t, w, fdim, agg, cdim)
		i = (blockIdx().x-1) * blockDim().x + threadIdx().x
		j = (blockIdx().y-1) * blockDim().y + threadIdx().y
		k = (blockIdx().z-1) * blockDim().z + threadIdx().z

		if prod((i, j, k) .< (fdim))
			i2 = ceil.(Int, (i, j, k)./agg)
			nd = ind(fdim..., i, j, k)
			nd_c = ind(cdim..., i2...)
			ind_f[nd] = nd
			ind_t[nd] = nd_c
			w[nd] = 1.0 # https://calcul.math.cnrs.fr/attachments/spip/IMG/pdf/aggamgtut_notay.pdf page 56
		end

		return nothing
	end
	
	# thread/block setup
	max_threads = 256
	threads_x   = min(max_threads, fdim[1])
	threads_y   = min(max_threads ÷ threads_x, fdim[2])
   	threads_z   = min(max_threads ÷ threads_x ÷ threads_y, fdim[3])
   	threads     = (threads_x, threads_y, threads_z)
   	blocks      = ceil.(Int, fdim ./ threads)
	@cuda threads=threads blocks=blocks kernel(ind, ind_f, ind_t, w, fdim, agg, cdim)

	P = PR_op(ind_t, ind_f, w)
	R = PR_op(ind_f, ind_t, w)

	return P, R
end

function gmg_PAP_diag(ind, rev_ind, o_from, d_from, d_c_off, d_c_main, fdim, agg)
	
	cdim = ceil.(Int64, fdim ./ agg)		
	i = (blockIdx().x-1) * blockDim().x + threadIdx().x
	j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    k = (blockIdx().z-1) * blockDim().z + threadIdx().z
	nd = ind(fdim..., i, j, k) ### row number

	#  Coeff nd->nd_nb becomes nd_coarse->nd_nb_coarse, and in ijk (i,j,k)->ind_nb becomes ind_nd_coarse->ind_nb_coarse
	if nd>max(-o_from, 0) && nd<min(prod(fdim), prod(fdim)-o_from)
		nd_nb   = nd + o_from
		i_nb    = rev_ind(fdim..., nd_nb) 
		i_nd_c  = ceil.(Int, (i, j, k)./agg)
		i_nb_c  = ceil.(Int, i_nb ./ agg)
		nd_c    = ind(cdim..., i_nd_c)
		nd_nb_c = ind(cdim..., i_nb_c)

		val = o_from > 0 ? d_from[nd] : d_from[nd+o_from]
		if nd_c==nd_nb_c ## Me and neighbor in the same cell in coarse level
			@atomic d_c_main[nd_c] += val
		elseif nd_c>nd_nb_c ## subdiagonal
			@atomic d_c_off[nd_nb_c] += val
		else 
			@atomic d_c_off[nd_c] += val
		end
	end

	return nothing
end	

function extend_heirarchy!(levels, A::SparseMatrixDIA{T,TF,CuVector{T}}, fdim, agg, ind, rev_ind) where {T, TF}
	
	P, R = gmg_interpolation(A, fdim, agg, ind)
	
	cdim = ceil.(fdim ./ agg)
	c_l  = prod(cdim)
	c_o  = [-cdim[2]*cdim[3], -cdim[3], -1, 0, 1, cdim[3], cdim[2]*cdim[3]] ## Can't figure out how to do this generic
	A_c  = SparseMatrixDIA([c_o[i]=>cuzeros(T, round(Int, c_l - abs(c_o[i]))) for i in 1:length(c_o)], c_l, c_l)
	d    = length(A.diags)>>1+1 # main diag index
	
	# threads/blocks setup
	max_threads = 256
    threads_x   = min(max_threads, fdim[1])
    threads_y   = min(max_threads ÷ threads_x, fdim[2])
    threads_z   = min(max_threads ÷ threads_x ÷ threads_y, fdim[3])
    threads     = (threads_x, threads_y, threads_z)
    blocks      = ceil.(Int, fdim ./ threads)

	for i in 1:length(A.diags)
		@cuda threads=threads blocks=blocks gmg_PAP_diag(ind, rev_ind, A.diags[i].first, A.diags[i].second, 
															A_c.diags[i].second, A_c.diags[d].second, fdim, agg)
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
