module housQR

import Base: size, show, getproperty, getfield, propertynames, \, *
using LinearAlgebra: norm, I, triu, diagm, dot

export QRhous, qrfact

mutable struct QRhous{T <: Real}
    A::AbstractVecOrMat{T} # R after the QR factorization
    d::AbstractArray{T} # diagonal of R
    AQ::Union{Nothing, AbstractVecOrMat{T}} # Q
    AR::Union{Nothing, AbstractVecOrMat{T}} # R0

    QRhous(A, d, AQ=nothing, AR=nothing) = new{eltype(A)}(A, d, AQ, AR)
end


@doc raw"""
    householder_vector(x::AbstractVecOrMat{T})::Tuple{AbstractVecOrMat{T}, T} where T

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
    qrfact(A::Matrix{T})::QRhous where T

Computes the QR factorization of A: A = QR.

### Input

- 'A' -- the input matrix; ``A \in \mathbb{R}^{m \times n}``.

### Output

A QRhous object containing the QR factorization of A. The R factor is stored in
the upper triangular part of the matrix A. The diagonal of R is stored in the
vector d. The Q factor is computed lazily.

"""
function qrfact(A::Matrix{T})::QRhous where T
    (m, n) = size(A)
    R = deepcopy(A)
    d = zeros(min(m, n))

    for k ∈ 1:min(n, m)
        @views (u, s) = householder_vector(R[k:m, k])
        # construct R
        d[k] = s
        @views R[k:m, k+1:n] -= 2 * u * (u' * R[k:m, k+1:n])
        R[k:m, k] .= u
    end
    return QRhous(R, d)
end

@doc raw"""
    qyhoust(A::QRhous{T}, y::AbstractArray{T}) where T

Computes the product ``Q_0^Ty,`` where ``Q_0`` is the ``n \times n`` (upper) part of the matrix Q of the QR factorization of A.

### Input

- `A` -- the QR factorization of A.
- `y` -- the vector to be multiplied by ``Q_0^T``.

### Output

The product ``Q_0^Ty``.


"""
function qyhoust(A::QRhous{T}, y::AbstractArray{T}) where T
    m, n = size(A.A)
    z = deepcopy(y)

    for j ∈ 1:n
        @views z[j:m] -= 2 * A.A[j:m, j] .* dot(A.A[j:m, j], z[j:m])
    end
    return z
end

@doc raw"""
    qyhous(A::QRhous{T}, y::AbstractArray{T}) where T

Computes the product ``Q_0y,`` where ``Q_0`` is the ``n \times n`` (upper) part of the matrix Q of the QR factorization of A.

### Input

- `A` -- the QR factorization of A.
- `y` -- the vector to be multiplied by ``Q_0``.

### Output

The product ``Q_0y``.

"""
function qyhous(A::QRhous{T}, y::AbstractArray{T}) where T
    m, n = size(A.A)
    z = deepcopy(y)
    for j ∈ n:-1:1
        @views z[j:m] -= 2 * A.A[j:m, j] .* dot(A.A[j:m, j], z[j:m])
    end
    return z
end


@doc raw"""
    multiplybyr(A::QRhous{T}, y::AbstractArray{T}) where T

Computes the product ``Ry,`` where ``R`` is the upper part of the matrix A, already transformed by the QR factorization.

"""
function multiplybyr(A::QRhous{T}, y::AbstractArray{T}) where T
    m, n = size(A.A)
    min_size = min(m, n)

    vcat(A.R * y[1:min_size], zeros(max(0, m - n)))
end

@doc raw"""
    calculateQ(A::QRhous{T}) where T

Computes the Q factor of the QR factorization of A.

### Input

- `A` -- the QR factorization of A.

### Output

The Q factor of the QR factorization of A.

"""
function calculateQ(A::QRhous{T}) where T
    if A.AQ !== nothing
        return A.AQ
    end
    m, n = size(A.A)

    A.AQ = zeros(m, 0)
    id = Matrix{eltype(A.A)}(I, m, n)
    for i ∈ eachcol(id)
        A.AQ = [A.AQ qyhous(A, i)]
    end
    return A.AQ
end

@doc raw"""
    calculateR(A::QRhous{T}) where T

Computes the R factor of the QR factorization of A. 
The R factor is the upper part of the matrix A, already transformed by the QR factorization. 

### Input

- `A` -- the QR factorization of A.

### Output

The R factor of the QR factorization of A.

"""
function calculateR(A::QRhous{T}) where T
    if A.AR !== nothing
        return A.AR
    end
    m, n = size(A.A)
    min_size = min(m, n)
    @show triu(A.A[1:n, :], 1)
    A.AR = triu(A.A[1:min_size, 1:min_size], 1) + diagm(A.d)
    return A.AR
end

"""
    (\\)(A::QRhous{T}, b::AbstractVector{T}) where T

Solves the linear system ``Ax = b`` using the QR factorization of A.

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

    for j ∈ min(m, n):-1:1
        @views x[j] = (v[j] - dot(x[j+1:n], A.A[j, j+1:n])) * A.d[j]^-1
    end
    return x
end

@doc raw"""
    (*)(A::QRhous{T}, x::AbstractVecOrMat{T}) where T

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


function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, A::QRhous{T}) where T
    summary(io, A); println(io)
    print(io, "Q factor: ")
    show(io, mime, A.Q)
    print(io, "\nR factor: ")
    show(io, mime, A.R)
end

# ------------------ Overloading ------------------

function Base.getproperty(A::QRhous{T}, d::Symbol) where T
    if d === :R
        return calculateR(A)
    elseif d === :Q
        return calculateQ(A)
    else
        getfield(A, d)
    end
end

Base.propertynames(A::QRhous, private::Bool=false) = (:R, :Q, (private ? fieldnames(typeof(A)) : ())...)

Base.size(A::QRhous) = size(getfield(A, :A))


end

