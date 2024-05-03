module housQR

import Base: size, show, getproperty, getfield, propertynames, \, *
using LinearAlgebra: norm, I, triu, diagm, dot, mul!

export QRhous, qrfact

@doc raw"""
```julia
QRhous{T <: Real}
```

Struct that holds a QR factorization. The R factor is stored in
the upper triangular part of the matrix A. The diagonal of R is stored in the
vector d. The Q factor is computed lazily.
"""
mutable struct QRhous{T <: Real}
    raw"""
    `A` holds the R factor in the upper triangular part minus the diagonal
    and the housholder vectors in the lower triangular.
    """
    QR::AbstractVecOrMat{T} # R after the QR factorization

    raw"""
    `d` holds the diagonal of the R factor.
    """
    d::AbstractArray{T} # diagonal of R

    raw"""
    `AQ` holds the cached value for Q, is `nothing` if not yet computed
    """
    AQ::Union{Nothing, AbstractVecOrMat{T}} # Q

    raw"""
    `AR` holds the cached value for R, is `nothing` if not yet computed
    """
    AR::Union{Nothing, AbstractVecOrMat{T}} # R0

    QRhous(QR, d, AQ=nothing, AR=nothing) = new{eltype(QR)}(QR, d, AQ, AR)
end


@doc raw"""
```julia
householder_vector(x::AbstractVecOrMat{T})::Tuple{AbstractVecOrMat{T}, T}
```

Computes a normalized vector u such that ``Hx = se_1, H = I - 2uu^T``.

### Input

- `x` -- the input vector

### Output

A tuple [u, s], where u is the householder vector of x and ``s = \big | x \big |``

"""
function householder_vector(x::AbstractVecOrMat{T})::Tuple{AbstractVecOrMat{T}, T} where T
    s = norm(x)
    if x[1] ≥ 0
        s = -s
    end
    u = copy(x)
    u[1] -= s
    u ./= norm(u)
    return u, s
end

@doc raw"""
```julia
qrfact(A::Matrix{T})::QRhous{T}
```

Computes the QR factorization of A: A = QR.

### Input

- 'A' -- the input matrix; ``A \in \mathbb{R}^{m \times n}``.

### Output

A `QRhous` object containing the QR factorization of A.

See also [`QRhous`](@ref).
"""
function qrfact(A::Matrix{T})::QRhous{T} where T
    (m, n) = size(A)
    R = deepcopy(A)
    d = zeros(min(n, m))

    tmp = similar(R)

    for k ∈ 1:min(n, m)
        @views (u, s) = householder_vector(R[k:m, k])
        # construct R
        d[k] = s
        @views mul!(tmp[k:m, k+1:n], 2*u, u' * R[k:m, k+1:n])
        @views R[k:m, k+1:n] .-= tmp[k:m, k+1:n]
        R[k:m, k] .= u
    end
    return QRhous(R, d)
end

@doc raw"""
```julia
qyhoust(A::QRhous{T}, y::AbstractArray{T})
```

Computes the product ``Q_0^Ty,`` where ``Q_0`` is the ``n \times n`` (upper) part of the matrix Q of the QR factorization of A.

### Input

- `A` -- the QR factorization of A.
- `y` -- the vector to be multiplied by ``Q_0^T``.

### Output

The product ``Q_0^Ty``.

See also [`qyhous`](@ref).
"""
function qyhoust(A::QRhous{T}, y::AbstractArray{T})::AbstractArray{T} where T
    m, n = size(A.QR)
    z = deepcopy(y)
    tmp = similar(z)

    for j ∈ 1:n
        @views tmp[j:m] .= 2 * dot(A.QR[j:m, j], z[j:m]) * A.QR[j:m, j]
        @views z[j:m] .-= tmp[j:m]
    end
    return z
end

@doc raw"""
```julia
qyhous(A::QRhous{T}, y::AbstractArray{T})
```

Computes the product ``Q_0y,`` where ``Q_0`` is the ``n \times n`` (upper) part of the matrix Q of the QR factorization of A.

### Input

- `A` -- the QR factorization of A.
- `y` -- the vector to be multiplied by ``Q_0``.

### Output

The product ``Q_0y``.

See also [`qyhoust`](@ref).
"""
function qyhous(A::QRhous{T}, y::AbstractArray{T}) where T
    m, n = size(A.QR)
    z = deepcopy(y)
    tmp = similar(z)
    
    for j ∈ n:-1:1
        @views tmp[j:m] .= 2 * dot(A.QR[j:m, j], z[j:m]) * A.QR[j:m, j]
        @views z[j:m] .-= tmp[j:m]
    end
    return z
end

@doc raw"""
```julia
multiplybyr(A::QRhous{T}, y::AbstractArray{T})
```

Computes the product ``Ry,`` where ``R`` is the upper part of the matrix A, already transformed by the QR factorization.

"""
function multiplybyr(A::QRhous{T}, y::AbstractArray{T}) where T
    m, n = size(A.QR)
    min_size = min(m, n)

    vcat(A.R * y[1:min_size], zeros(max(0, m - n)))
end

@doc raw"""
```julia
calculateQ(A::QRhous{T})
```

Computes the Q factor of the QR factorization of A. The result is cached.

### Input

- `A` -- the QR factorization of `A`.

### Output

The Q factor of the QR factorization of `A`.

"""
function calculateQ(A::QRhous{T}) where T
    if !isnothing(A.AQ)
        return A.AQ
    end
    m, n = size(A.QR)

    A.AQ = zeros(m, 0)
    id = Matrix{eltype(A.QR)}(I, m, n)
    for i ∈ eachcol(id)
        A.AQ = [A.AQ qyhous(A, i)]
    end
    return A.AQ
end

@doc raw"""
```julia
calculateR(A::QRhous{T})
```

Computes the R factor of the QR factorization of `A`. 
The R factor is the upper part of the matrix A, already transformed by the QR factorization. 
The result is cached.

### Input

- `A` -- the QR factorization of `A`.

### Output

The R factor of the QR factorization of `A`.

"""
function calculateR(A::QRhous{T}) where T
    if !isnothing(A.AR)
        return A.AR
    end
    m, n = size(A.QR)
    min_size = min(m, n)
    # @show triu(A.QR[1:n, :], 1)
    A.AR = triu(A.QR[1:min_size, 1:min_size], 1) + diagm(A.d)
    return A.AR
end

@doc """
```julia
(\\)(A::QRhous{T}, b::AbstractVector{T})
```

Solves the linear system ``Ax = b`` using the QR factorization of A. 
First, it computes the product ```Q^t b``` and then solves the triangular system ```R x = Q^t b``` via backsubstitution.

### Input

- `A` -- the QR factorization of A.
- `b` -- the right-hand side vector.

### Output

The solution vector x.

"""
function (\)(A::QRhous{T}, b::AbstractVector{T}) where T
    m, n = size(A)
    v = qyhoust(A, b)
    x = zeros(n)
    tmp = x[1]

    for j ∈ min(m, n):-1:1
        @views tmp = dot(x[j+1:n], A.QR[j, j+1:n])
        x[j] = (v[j] - tmp) * inv(A.d[j])
    end
    return x
end

@doc raw"""
```julia
(*)(A::QRhous{T}, x::AbstractVecOrMat{T})
```

Computes the product ``Ax,`` where ``A`` is the QR factorization of A.

### Input

- `A` -- the QR factorization of A.
- `x` -- the vector to be multiplied by ``A``.

### Output

The product ``Ax``.

"""
function (*)(A::QRhous{T}, x::AbstractVecOrMat{T}) where T
    return qyhous(A, multiplybyr(A, x))
end

@doc raw"""
```julia
show(io::IO, mime, A::QRhous{T})
```

Pretty printing for the factoried matrix A. Computes explicitly Q and R.
"""
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, A::QRhous{T}) where T
    summary(io, A)
    println(io)
    print(io, "Q factor: ")
    show(io, mime, A.Q)
    println(io)
    print(io, "R factor: ")
    show(io, mime, A.R)
end

@doc raw"""
```julia
getproperty(A::QRhous{T}, name::Symbol)
```

Calculates explicitely Q and R if required.
"""
function Base.getproperty(A::QRhous{T}, name::Symbol) where T
    if name === :R
        return calculateR(A)
    elseif name === :Q
        return calculateQ(A)
    else
        getfield(A, name)
    end
end

@doc raw"""
```julia
propertynames(A::QRhous, [private::Bool = false])
```

Returns a tuple of all the properties of a QRhous matrix.

```jldoctest
julia> :R ∈ propertynames(qrfact([1 0; 0 1]))
true
```
"""
Base.propertynames(A::QRhous, private::Bool=false) = (:R, :Q, (private ? fieldnames(typeof(A)) : ())...)

@doc raw"""
```julia
size(A::QRhous)
```

Returns the size of the original matrix A. It is also the size of the support matrix.
"""
Base.size(A::QRhous) = size(getfield(A, :QR))

end # module housQR
