### Conversion
Base.Matrix(s::SparseMatrixDIA) = diagm(s.diags...)
SparseMatrixCSC(S::SparseMatrixDIA{Tv,Ti,V}) where {Tv,Ti,V} = spdiagm(S.diags...)

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
function CuArrays.cu(S::SparseMatrixDIA{Tv,Ti,V}) where {Tv,Ti,V}
        m, n = size(S)
    R = figure_out_type(V)
        s = Vector{Pair{Ti,R}}(undef, length(S.diags))
        for (i,x) in enumerate(S.diags)
                first = x.first
                second = x.second
                s[i] = first => cu(x.second)
        end
        SparseMatrixDIA(s, m, n)
end
figure_out_type(::Type{Vector{Float64}}) = CuVector{Float32}
figure_out_type(::Type{Vector{S}}) where S = CuVector{S}

