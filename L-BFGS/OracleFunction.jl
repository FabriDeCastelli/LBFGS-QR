module OracleFunction
export OracleF, LeastSquaresF, tomography

struct OracleF{T, F<:Function, G<:Function}
    starting_point::AbstractArray{T}
    eval::F
    grad::G
end


struct LeastSquaresF{T, F<:Function, G<:Function}
    oracle::OracleF{T, F, G}
    X::AbstractMatrix{T}
    y::AbstractArray{T}
    symm::AbstractMatrix{T}
    yX::AbstractArray{T}
end

using LinearAlgebra: norm

function LeastSquaresF(starting_point::AbstractArray{T}, X::AbstractMatrix{T}, y::AbstractArray{T}) where T
    f(x)  = norm(X * x - y)
    df(x) = X' * (X * x - y)
    symm  = X' * X
    yX    = y' * X

    o = OracleF(starting_point, f, df)
    LeastSquaresF(o, X, y, symm, yX)
end

function tomography(l::LeastSquaresF{T, F, G}, w::AbstractArray{T}, p::AbstractArray{T}) where {T, F, G}
    (l.yX * p - w' * l.symm * p) * inv(p' * l.symm * p)
end

function Base.getproperty(l::LeastSquaresF{T, F, G}, name::Symbol) where {T, F, G}
    if name === :eval
        return l.oracle.eval
    elseif name === :grad
        return l.oracle.grad
    elseif name === :starting_point
        return l.oracle.starting_point
    else
        getfield(l, name)
    end
end

end ## module