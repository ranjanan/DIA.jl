using DIA
using Test

x = SparseMatrixDIA((0=>[1,2,3,4,5], 2=>[1,2,3]), 5, 5)
v = [1,2,3,4,5]
@assert x * v == Matrix(x) * v
