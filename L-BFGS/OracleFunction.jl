module OracleFunction

using LinearAlgebra: norm

export OracleF, LeastSquaresF, tomography

@doc """
﻿OracleF{T, F<:Function, G<:Function}

Struct that holds a generic function to evaluate.
`eval` is the function that evaluates a point, `grad` is the gradient of the function and
`starting_point` is the point from which minimization should start.
"""
struct OracleF{T, F<:Function, G<:Function}
    starting_point::AbstractArray{T}
    eval::F
    grad::G
end

@doc """
﻿LeastSquaresF{T, F<:Function, G<:Function}

Struct that holds an instance of a least squares problem. The interface is similar to the `OracleF` struct.
`eval` is the function that evaluates a point, `grad` is the gradient of the function and
`starting_point` is the point from which minimization should start.

See also [`OracleF`](@ref).
"""
struct LeastSquaresF{T, F<:Function, G<:Function}
    oracle::OracleF{T, F, G}
    X::AbstractMatrix{T}
    y::AbstractArray{T}
    symm::AbstractMatrix{T}
    yX::AbstractArray{T}
end

function LeastSquaresF(starting_point::AbstractArray{T}, X::AbstractMatrix{T}, y::AbstractArray{T}) where T
    f(x)  = norm(X * x - y)
    df(x) = X' * (X * x - y)
    symm  = X' * X
    yX    = y' * X

    o = OracleF(starting_point, f, df)
    LeastSquaresF(o, X, y, symm, yX)
end

@doc """
﻿tomography(l::LeastSquaresF{T, F, G}, w::AbstractArray{T}, p::AbstractArray{T})

Function that returns the minimum of the function `l` along the plane in `w` and with direction `p`.

See also [`LeastSquaresF`](@ref).
"""
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

end ## module OracleFunction
