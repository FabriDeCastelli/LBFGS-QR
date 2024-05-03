module SR1

using LinearAlgebra: norm, I, dot, diagm, mul!

using ..OracleFunction

export SymmetricRank1


function SymmetricRank1(
        f::Union{LeastSquaresF, OracleF};
        x::Union{Nothing, AbstractVector{T}}=nothing,
        ϵ::T=1e-6,
        η::T=1e-4, # threshold for ignoring direction
        r::T=1e-8, # skipping rule for updating B and H
        MaxEvaluations::Integer=10000
    )::NamedTuple where {T}

    Δ = 1 # initial size of trust region
    smallestΔ = 1e-4 # smallest size where linear aproximation is applied

    if isnothing(x)
        x = f.starting_point
    end

    gradient = f.grad(x)
    evalx = f.eval(x)
    nextevalx = 0
    MaxEvaluations -= 2
    normgradient0 = norm(gradient)
    normgradient = normgradient0
    B = diagm(ones(length(x)))
    H = diagm(ones(length(x)))
    tmp1 = similar(x)
    tmp2 = similar(H)

    local s

    while MaxEvaluations > 0 && normgradient > ϵ * normgradient0
        # compute s by solving the subproblem min_s grad' * s + 0.5 s' * B * s with norm(s) ≤ Δ
        CauchyPoint = - (Δ/normgradient) * gradient
        τ = if gradient' * B * gradient ≤ 0
                1
            else
                min((normgradient^3)/(Δ * dot(gradient, B, gradient)), 1)
            end

        if Δ ≤ smallestΔ || B == I
            # the Cauchy point is enought for small regions (linear aproximation)
            s = τ * CauchyPoint
        else
            pB = -H * gradient
            pU = -dot(gradient, gradient)/dot(gradient, B, gradient) * gradient

            if norm(pB) ≤ Δ
                # the region is larger than the dogleg
                s = pB
            elseif Δ ≤ norm(pU)
                # the region is smaller than the first step
                s = Δ/norm(pU) * pU
            else
                # solve the quadratic sistem for the dogleg
                one = dot(pU, (pB - pU))
                two = dot((pB - pU), (pB - pU))
                three = dot(pU, pU)

                τ = (-one+two + sqrt(one^2 - three * two + two * Δ^2))/two
                s = pU + (τ - 1) * (pB - pU)
            end
        end

        # ------
        y = f.grad(x + s) - gradient

        nextevalx = f.eval(x + s)
        ared = evalx - nextevalx # actual reduction
        pred = -(dot(gradient, s) + 0.5 * dot(s, B, s)) # predicted reduction

        MaxEvaluations -= 2

        if ared/pred > η
            x = x + s
            evalx = nextevalx
            gradient = f.grad(x)
            normgradient = norm(gradient)
            MaxEvaluations -= 1
        end

        # expand or contract the region
        if (0.75 < ared/pred) && (0.8 * Δ < norm(s))
            Δ = 2 * Δ
        elseif (ared/pred < 0.1)
            Δ = 0.5 * Δ
        end
        if abs(s' * (y - B * s)) ≥ r * norm(s) * norm(y - B * s) # if the denominator is not too small
            # B = B + ((y - B * s)*(y - B * s)')/((y - B * s)' * s)
            mul!(tmp1, B, -s)
            tmp1 .+= y
            mul!(tmp2, tmp1, tmp1')
            tmp2 ./= dot(tmp1, s)
            B .+= tmp2

            # H = H + ((s - H * y) * (s - H * y)')/((s - H * y)' * y)
            mul!(tmp1, H, -y)
            tmp1 += s
            mul!(tmp2, tmp1, tmp1')
            tmp2 ./= dot(tmp1, y)
            H .+= tmp2
        end
    end

    return (;
        :x => x,
        :eval => f.eval(x),
        :grad => gradient,
        :RemainingEvaluations => MaxEvaluations)
end

end # module SR1