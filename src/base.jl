export SparseMatrixDIA

struct SparseMatrixDIA{Tv,Ti,V<:AbstractArray{Tv}} <: AbstractSparseMatrix{Tv,Ti}
    diags::Vector{Pair{Ti,V}}
    m::Ti
    n::Ti
end

function SparseMatrixDIA(x::NTuple{N,Pair{Ti,V}}, 
				m::Ti, n::Ti) where {Ti,Tv,N,V<:AbstractArray{Tv}}
	SparseMatrixDIA(collect(x), m, n)
end


Base.size(a::SparseMatrixDIA) = (a.m, a.n)

function SparseArrays.nnz(a::SparseMatrixDIA)
    l = 0
    for x in a.diags
        per = length(x.second)
        l += per
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

# Base.show(io::IO, S::SparseMatrixDIA) = Base.show(convert(IOContext, io), S::SparseMatrixDIA)

function Base.show(io::IO, p::Pair{Ti, Vector{Tv}}) where {Ti,Tv}
	print(IOContext(io, :limit=>true), "Diagonal $(p.first): ", p.second)
end 
Base.summary(io::IO, v::Vector{Pair{Ti,Vector{Tv}}}) where {Ti,Tv} = ""

#=function Base.show(io::IO, diags::Vector{Pair{Ti,Vector{Tv}}}) where {Ti,Tv}
#	for i = 1:length(diags)
#		print(io, diags[i], '\n')
#	end
	print(io, summary(diags), '\n')
	show(IOContext(io, :compact=>true, :limit=>true), "text/plain", diags)
end=#
function Base.show(io::IO, S::SparseMatrixDIA)
	print(io, summary(S))
	print(io, '\n')
	println(IOContext(io, :limit=>true), S.diags)
end

Base.display(S::SparseMatrixDIA) = Base.show(S)

function Base.summary(S::SparseMatrixDIA{Tv,Ti,V}) where {Tv,Ti,V}
    "$(S.m)×$(S.n) SparseMatrixDIA{$Tv,$Ti} with $(length(S.diags)) diagonals: "
end

