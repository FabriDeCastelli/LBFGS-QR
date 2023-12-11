module housQR

import Base: size, show, getproperty, getfield, propertynames, \, *
using LinearAlgebra: norm, I, triu, diagm, dot

export QRhous, qrfact

mutable struct QRhous{T <: Real}
    A::AbstractVecOrMat{T}
    d::AbstractArray{T}
    AQ::Union{Nothing, AbstractVecOrMat{T}}
    AR::Union{Nothing, AbstractVecOrMat{T}}

    QRhous(A, d, AQ=nothing, AR=nothing) = new{eltype(A)}(A, d, AQ, AR)
end



function householder_vector(x::AbstractVecOrMat{T})::Tuple{AbstractVecOrMat{T}, T} where T
    # returns the normalized vector u such that H*x is a multiple of e_1
    s = norm(x)
    if x[1] ≥ 0
        s = -s
    end
    u = copy(x)
    u[1] -= s
    u ./= norm(u)
    return u, s
end

function qrfact(A::Matrix{T})::QRhous where T
    (m, n) = size(A)
    R = deepcopy(A)
    d = zeros(n)

    for k ∈ 1:min(n, m)
        @views (u, s) = householder_vector(R[k:m, k])
        # construct R
        d[k] = s
        @views R[k:m, k+1:n] -= 2 * u * (u' * R[k:m, k+1:n])
        R[k:m, k] .= u
    end
    return QRhous(R, d)
end


function qyhous(A::QRhous{T}, y::AbstractArray{T}) where T
    m, n = size(A)
    z = deepcopy(y)
    for j ∈ n:-1:1
        # z[j:m] = z[j:m] - 2 * A.A[j:m, j] * (A.A[j:m, j]' * z[j:m])
        @views z[j:m] -= 2 * A.A[j:m, j] .* dot(A.A[j:m, j], z[j:m])
    end
    return z
end

function qyhoust(A::QRhous{T}, y::AbstractArray{T}) where T
    m, n = size(A.A)
    z = deepcopy(y)
    for j ∈ 1:n
        # z[j:m] = z[j:m] - A.A[j:m, j] * (A.A[j:m, j]' * z[j:m])
        @views z[j:m] -= 2 * A.A[j:m, j] .* dot(A.A[j:m, j], z[j:m])
    end
    return z
end

function calculateQ(A::QRhous{T}) where T
    if A.AQ != nothing
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

function calculateR(A::QRhous{T}) where T
    if A.AR != nothing
        return A.AR
    end
    m, n = size(A.A)
    A.AR = triu(A.A[1:n, :], 1) + diagm(A.d)
    return A.AR
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, A::QRhous{T}) where T
    summary(io, A); println(io)
    print(io, "Q factor: ")
    show(io, mime, A.Q)
    print(io, "\nR factor: ")
    show(io, mime, A.R)
end

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

function (\)(A::QRhous{T}, b::AbstractVector{T}) where T
    n, m = size(A)
    v = qyhoust(A, b)
    x = zeros(m)
    for j ∈ m:-1:1
        @views x[j] = (v[j] - dot(x[j+1:m], A.A[j, j+1:m])) * A.d[j]^-1
    end
    return x
end

function (*)(A::QRhous{T}, x::AbstractVecOrMat{T}) where T
    return qyhous(A, (A.R * x))
end

end