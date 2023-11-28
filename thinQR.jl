module thinQR

import Base: size, show, getproperty, getfield, propertynames, \, *
using LinearAlgebra: norm, I, triu, diagm, ldiv!, dot

export QRthin, qrhous!, qyhous, qyhoust

mutable struct QRthin{T <: Real}
    A::AbstractVecOrMat{T}
    d::AbstractArray{T}
    AQ::Union{Nothing, AbstractVecOrMat{T}}
    AR::Union{Nothing, AbstractVecOrMat{T}}

    QRthin(A, d, AQ=nothing, AR=nothing) = new{eltype(A)}(A, d, AQ, AR)
end

function qrhous!(A::AbstractMatrix{T})::QRthin{T} where T
    m, n = size(A)
    d = zeros(n)
    @inbounds begin
        for j ∈ 1:n
            s = norm(A[j:m, j])
            # iszero(s) && throw(ArgumentError("The matrix A is singular"))
            d[j]= copysign(s, -A[j,j])
            fak = sqrt(s * (s + abs(A[j,j])))
            A[j,j] -= d[j]

            A[j:m, j] ./= fak

            if j < n
                A[j:m, j+1:n] -= A[j:m, j] * (A[j:m, j]' * A[j:m, j+1:n])
                #for k ∈ j+1:n
                #    A[j:m, k] -= A[j:m, j] .* dot(A[j:m, j], A[j:m, k])
                #end
            end
        end
    end

    return QRthin(A, d)
end

function qyhous(A::QRthin{T}, y::AbstractArray{T}) where T
    m, n = size(A.A)
    z = deepcopy(y)
    for j ∈ n:-1:1
        # z[j:m] = z[j:m] - A.A[j:m, j] * (A.A[j:m, j]' * z[j:m])
        z[j:m] -= A.A[j:m, j] .* dot(A.A[j:m, j], z[j:m])
    end
    return z
end

function qyhoust(A::QRthin{T}, y::AbstractArray{T}) where T
    m, n = size(A.A)
    z = deepcopy(y)
    for j ∈ 1:n
        # z[j:m] = z[j:m] - A.A[j:m, j] * (A.A[j:m, j]' * z[j:m])
        z[j:m] -= A.A[j:m, j] .* dot(A.A[j:m, j], z[j:m])
    end
    return z
end

function calculateQ(A::QRthin{T}) where T
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

function calculateR(A::QRthin{T}) where T
    if A.AR != nothing
        return A.AR
    end
    m, n = size(A.A)
    A.AR = vcat(triu(A.A[1:n, :], 1) + diagm(A.d), zeros(m-n, n))
    return A.AR
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, A::QRthin{T}) where T
    summary(io, A); println(io)
    print(io, "Q factor: ")
    show(io, mime, A.Q)
    print(io, "\nR factor: ")
    show(io, mime, A.R)
end

function Base.getproperty(A::QRthin{T}, d::Symbol) where T
    if d === :R
        return calculateR(A)
    elseif d === :Q
        return calculateQ(A)
    else
        getfield(A, d)
    end
end

Base.propertynames(A::QRthin, private::Bool=false) = (:R, :Q, (private ? fieldnames(typeof(A)) : ())...)

Base.size(A::QRthin) = size(getfield(A, :A))

function (\)(A::QRthin{T}, b::AbstractVector{T}) where T
    n, m = size(A)
    v = qyhoust(A, b)
    x = zeros(m)
    for j ∈ m:-1:1
        x[j] = (v[j] - dot(x[j+1:m], A.A[j, j+1:m])) * A.d[j]^-1
    end
    return x
end

function (*)(A::QRthin{T}, x::AbstractVecOrMat{T}) where T
    return qyhous(A, (A.R * x))
end

end # module