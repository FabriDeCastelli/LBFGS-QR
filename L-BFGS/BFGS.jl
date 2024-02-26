using Plots: Plot, plot, plot!
using LinearAlgebra: norm, I, eigvals, dot
using Printf

function BFGS(f;
              x::Union{Nothing, Vector}=nothing,
              delta::Real=1,
              eps::Real=1e-6,
              MaxFeval::Integer=1000,
              m1::Real=1e-3,
              m2::Real=0.9,
              tau::Real=0.9,
              sfgrd::Real=0.20,
              MInf::Real=-Inf,
              mina::Real=1e-16,
              plt::Union{Plot, Nothing}=nothing,
              plotatend::Bool=true,
              Plotf::Integer=0,
              printing::Bool=true
              )::Tuple{Vector, String, Real}

    # DOCUMENTATION for the matlab function, functionality may differ
    #function [ x , status , v ] = BFGS( f , x , delta , eps , MaxFeval , ...
    #                                    m1 , m2 , tau , sfgrd , MInf , mina )
    #
    # Apply a Quasi-Newton approach, in particular using the celebrated
    # Broyden-Fletcher-Goldfarb-Shanno (BFGS) formula, for the minimization of
    # the provided function f, which must have the following interface:
    #
    #   [ v , g ] = f( x )
    #
    # Input:
    #
    # - x is either a [ n x 1 ] real (column) vector denoting the input of
    #   f(), or [] (empty).
    #
    # Output:
    #
    # - v (real, scalar): if x == [] this is the best known lower bound on
    #   the unconstrained global optimum of f(); it can be -Inf if either f()
    #   is not bounded below, or no such information is available. If x ~= []
    #   then v = f(x).
    #
    # - g (real, [ n x 1 ] real vector): this also depends on x. if x == []
    #   this is the standard starting point from which the algorithm should
    #   start, otherwise it is the gradient of f() at x (or a subgradient if
    #   f() is not differentiable at x, which it should not be if you are
    #   applying the gradient method to it).
    #
    # The other [optional] input parameters are:
    #
    # - x (either [ n x 1 ] real vector or [], default []): starting point.
    #   If x == [], the default starting point provided by f() is used.
    #
    # - delta (real scalar, optional, default value 1): the initial
    #   approximation of the Hesssian is taken as delta * I if delta > 0;
    #   otherwise, the initial Hessian is approximated by finite differences
    #   with - delta as the step, and inverted just the once.
    #
    # - eps (real scalar, optional, default value 1e-6): the accuracy in the
    #   stopping criterion: the algorithm is stopped when the norm of the
    #   gradient is less than or equal to eps. If a negative value is provided,
    #   this is used in a *relative* stopping criterion: the algorithm is
    #   stopped when the norm of the gradient is less than or equal to
    #   (- eps) * || norm of the first gradient ||.
    #
    # - MaxFeval (integer scalar, optional, default value 1000): the maximum
    #   number of function evaluations (hence, iterations will be not more than
    #   MaxFeval because at each iteration at least a function evaluation is
    #   performed, possibly more due to the line search).
    #
    # - m1 (real scalar, optional, must be in ( 0 , 1 ), default value 1e-3):
    #   parameter of the Armijo condition (sufficient decrease) in the line
    #   search
    #
    # - m2 (real scalar, optional, default value 0.9): typically the parameter
    #   of the Wolfe condition (sufficient derivative increase) in the line
    #   search. It should to be in ( 0 , 1 ); if not, it is taken to mean that
    #   the simpler Backtracking line search should be used instead
    #
    # - tau (real scalar, optional, default value 0.9): scaling parameter for
    #   the line search. In the Armijo-Wolfe line search it is used in the
    #   first phase to identify a point where the Armijo condition is not
    #   satisfied or the derivative is positive by divding the current
    #   value (starting with astart, see above) by tau (which is < 1, hence it
    #   is increased). In the Backtracking line search, each time the step is
    #   multiplied by tau (hence it is decreased).
    #
    # - sfgrd (real scalar, optional, default value 0.20): safeguard parameter
    #   for the line search. to avoid numerical problems that can occur with
    #   the quadratic interpolation if the derivative at one endpoint is too
    #   large w.r.t. the one at the other (which leads to choosing a point
    #   extremely near to the other endpoint), a *safeguarded* version of
    #   interpolation is used whereby the new point is chosen in the interval
    #   [ as * ( 1 + sfgrd ) , am * ( 1 - sfgrd ) ], being [ as , am ] the
    #   current interval, whatever quadratic interpolation says. If you
    #   experiemce problems with the line search taking too many iterations to
    #   converge at "nasty" points, try to increase this
    #
    # - MInf (real scalar, optional, default value -Inf): if the algorithm
    #   determines a value for f() <= MInf this is taken as an indication that
    #   the problem is unbounded below and computation is stopped
    #   (a "finite -Inf").
    #
    # - mina (real scalar, optional, default value 1e-16): if the algorithm
    #   determines a stepsize value <= mina, this is taken as an indication
    #   that something has gone wrong (the gradient is not a direction of
    #   descent, so maybe the function is not differentiable) and computation
    #   is stopped. It is legal to take mina = 0, thereby in fact skipping this
    #   test.
    #
    # Output:
    #
    # - x ([ n x 1 ] real column vector): the best solution found so far.
    #
    # - status (string): a string describing the status of the algorithm at
    #   termination
    #
    #   = 'optimal': the algorithm terminated having proven that x is a(n
    #     approximately) optimal solution, i.e., the norm of the gradient at x
    #     is less than the required threshold
    #
    #   = 'unbounded': the algorithm has determined an extrenely large negative
    #     value for f() that is taken as an indication that the problem is
    #     unbounded below (a "finite -Inf", see MInf above)
    #
    #   = 'stopped': the algorithm terminated having exhausted the maximum
    #     number of iterations: x is the bast solution found so far, but not
    #     necessarily the optimal one
    #
    #   = 'error': the algorithm found a numerical error that prevents it from
    #     continuing optimization (see mina above)
    #
    # - v (scalar real): the best function value found so far (that of x)
    #
    #{
    # =======================================
    # Author: Antonio Frangioni
    # Date: 05-12-23
    # Version 1.50
    # Copyright Antonio Frangioni
    # =======================================
    #}

    #Plotf = 0
    # 0 = nothing is plotted
    # 1 = the level sets of f and the trajectory are plotted (when n = 2)
    # 2 = the function value / gap are plotted, iteration-wise
    # 3 = the function value / gap are plotted, function-evaluation-wise

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # inner functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    function f2phi(alpha, derivate=false)
        #
        # computes and returns the value of the tomography at alpha
        #
        #    phi( alpha ) = f( x + alpha * d )
        #
        # if Plotf > 2 saves the data in gap() for plotting
        #
        # if the second output parameter is required, put there the derivative
        # of the tomography in alpha
        #
        #    phi'( alpha ) = < \nabla f( x + alpha * d ) , d >
        #
        # saves the point in lastx, the gradient in lastg and increases feval
        lastx = x + alpha * d
        (phi, lastg, _) = f(lastx)

        if Plotf > 2
            if fStar > - Inf
                push!(gap, (phi - fStar) / max(abs(fStar), 1))
            else
                push!(gap, phi)
            end
        end

        feval += 1

        if derivate
            return (phi, dot(d, lastg))
        end
        return (phi, nothing)
    end

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    function ArmijoWolfeLS(phi0, phip0, as, m1, m2, tau)
        # performs an Armijo-Wolfe Line Search.
        #
        # phi0 = phi( 0 ), phip0 = phi'( 0 ) < 0
        #
        # as > 0 is the first value to be tested: if phi'( as ) < 0 then as is
        # divided by tau < 1 (hence it is increased) until this does not happen
        # any longer
        #
        # m1 and m2 are the standard Armijo-Wolfe parameters; note that the strong
        # Wolfe condition is used
        #
        # returns the optimal step and the optimal f-value

        lsiter = 1  # count iterations of first phase
        local phips, phia
        while feval ≤ MaxFeval
            phia, phips = f2phi(as, true)

            if (phia ≤ phi0 + m1 * as * phip0) && (abs(phips) ≤ - m2 * phip0)
                if printing
                    @printf("\t%2d", lsiter)
                end
                a = as
                return (a, phia)  # Armijo + strong Wolfe satisfied, we are done
            end
            if phips ≥ 0
                break
            end
            as = as / tau
            lsiter += 1
        end

        if printing
            @printf("\t%2d ", lsiter)
        end
        lsiter = 1  # count iterations of second phase

        am = 0
        a = as
        phipm = phip0
        while (feval ≤ MaxFeval ) && (as - am) > mina && (phips > 1e-12)

            # compute the new value by safeguarded quadratic interpolation
            a = (am * phips - as * phipm) / (phips - phipm)
            a = max(am + ( as - am ) * sfgrd, min(as - ( as - am ) * sfgrd, a))

            # compute phi(a)
            phia, phip = f2phi(a, true)

            if (phia ≤ phi0 + m1 * a * phip0) && (abs(phip) ≤ -m2 * phip0)
                break  # Armijo + strong Wolfe satisfied, we are done
            end

            # restrict the interval based on sign of the derivative in a
            if phip < 0
                am = a
                phipm = phip
            else
                as = a
                if as ≤ mina
                    break
                end
                phips = phip
            end
            lsiter += 1
        end

        if printing
            @printf("%2d", lsiter)
        end
        return (a, phia)
    end

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    function BacktrackingLS( phi0 , phip0 , as , m1 , tau )

        # performs a Backtracking Line Search.
        #
        # phi0 = phi( 0 ), phip0 = phi'( 0 ) < 0
        #
        # as > 0 is the first value to be tested, which is decreased by
        # multiplying it by tau < 1 until the Armijo condition with parameter
        # m1 is satisfied
        #
        # returns the optimal step and the optimal f-value

        lsiter = 1  # count ls iterations
        while feval ≤ MaxFeval && as > mina
            phia, _ = f2phi(as)
            if phia ≤ phi0 + m1 * as * phip0 # Armijo satisfied
                break                        # we are done
            end
            as *= tau
            lsiter += 1
        end

        if printing
            @printf("\t%2d", lsiter)
        end
        return (as, phia)
    end

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    Interactive = false # if we pause at every iteration
    local gap

    if Plotf > 1
        if Plotf == 2
            MaxIter = 50 # expected number of iterations for the gap plot
        else
            MaxIter = 70 # expected number of iterations for the gap plot
        end
        gap = []
    end

    # reading and checking input- - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if isnothing(x)
        fStar, x = f(nothing)
    else
        fStar, _ = f(nothing)
    end

    n = size(x, 1)

    if m1 ≤ 0 || m1 ≥ 1
        error("m1 is not in ( 0 , 1 )")
    end

    AWLS = (m2 > 0 && m2 < 1)

    if tau ≤ 0 || tau ≥ 1
        error("tau is not in (0, 1)")
    end

    if sfgrd ≤ 0 || sfgrd ≥ 1
        error("sfgrd is not in ( 0 , 1 )")
    end

    if mina < 0
        error("mina is < 0")
    end

    # "global" variables- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    lastx = zeros(n)  # last point visited in the line search
    lastg = zeros(n)  # gradient of lastx
    d = zeros(n)      # quasi-Newton's direction
    feval = 0         # f() evaluations count ("common" with LSs)
    bestx = zeros(n)  # best point found ever: the method is a descent
                      # one, but only if the LS is not interrupted
    bestf = Inf       # best f-value found ever (that of bestx)

    PXY = Matrix{Real}(undef, 2, 0)
    status = "error"

    # initializations - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if printing
        @printf("BFGS method\n")
        if fStar > - Inf
            @printf("feval\trel gap")
        else
            @printf("feval\tf(x)")
        end
        @printf("\t\t|| g(x) ||\tls fev\ta*\t rho\n\n")
    end

    if Plotf > 1 && isnothing(plt)
        if Plotf == 2
            plt = plot(xlims=(0, MaxIter), ylims=(1e-15, 1e+1), yscale=:log10)
        else
            plt = plot(xlims=(0, MaxIter), ylims=(1e-15, 1e+4), yscale=:log10)
        end
    elseif isnothing(plt)
        plt = plot()
    end

    v, _ = f2phi(0)
    g = lastg
    ng = norm(g)
    if eps < 0
        ng0 = -ng # norm of first subgradient: why is there a "-"? ;-)
    else
        ng0 = 1   # un-scaled stopping criterion
    end

    if delta > 0
        # initial approximation of inverse of Hessian = scaled identity
        B = delta * I
    else
        # initial approximation of inverse of Hessian computed by finite
        # differences of gradient
        smallsetp = max(-delta, 1e-8)
        B = zeros(n, n)
        for i ∈ 1:n
            xp = copy(x)
            xp[i] = xp[i] + smallsetp
            _, gp = f(xp)
            @views B[i, :] = (gp - g) ./ smallsetp
        end
        B = (B + B')/2       # ensure it is symmetric
        lambdan = minimum(abs.(eigvals(B)))  # smallest eigenvalue
        if lambdan < 1e-6
            B = B + (1e-6 - lambdan) * I
        end
        B = inv(B)
    end

    # main loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    while true

        # output statistics - - - - - - - - - - - - - - - - - - - - - - - - - -

        if fStar > -Inf
            gapk = (v - fStar) / max(abs(fStar), 1)

            if printing
                @printf("%4d\t%1.4e\t%1.4e", feval, gapk, ng)
            end

            if Plotf > 1
                if Plotf == 2
                    push!(gap, gapk)
                end
            end
        else
            if printing
                @printf("%4d\t%1.8e\t%1.4e", feval, v, ng)
            end

            if Plotf > 1
                if Plotf == 2
                    push!(gap, v)
                end
            end
        end

        # stopping criteria - - - - - - - - - - - - - - - - - - - - - - - - - -

        if ng ≤ eps * ng0
            status = "optimal"
            if printing
                @printf("\n")
            end
            break
        end

        if feval > MaxFeval
            status = "stopped"
            if printing
                @printf("\n")
            end
            break
        end

        # compute approximation to Newton's direction - - - - - - - - - - - - -

        d = -B * g

        # compute step size - - - - - - - - - - - - - - - - - - - - - - - - - -
        # as in Newton's method, the default initial stepsize is 1

        phip0 = dot(g, d)

        if AWLS
            a, v = ArmijoWolfeLS(v, phip0, 1, m1, m2, tau)
        else
            a, v = BacktrackingLS(v, phip0, 1, m1, tau)
        end

        # output statistics - - - - - - - - - - - - - - - - - - - - - - - - - -

        if printing
            @printf("\t%1.2e", a)
        end

        if a ≤ mina
            status = "error"
            if printing
                @printf("\n")
            end
            break
        end

        if v ≤ MInf
            status = "unbounded"
            if printing
                @printf("\n")
            end
            break
        end

        # update approximation of the Hessian - - - - - - - - - - - - - - - - -
        # warning: magic at work! Broyden-Fletcher-Goldfarb-Shanno formula

        s = lastx - x   # s^i = x^{i + 1} - x^i
        y = lastg - g   # y^i = \nabla f( x^{i + 1} ) - \nabla f( x^i )

        rho = dot(y, s)
        if rho < 1e-16
            if printing
                @printf("\nError: y^i s^i = %1.2e\n\n", rho)
            end
            status = "error"
            break
        end

        rho = 1 / rho

        if printing
            @printf(" %1.2e\n", rho)
        end

        D = B * y * s'
        B = B + rho * ((1 + rho * y' * B * y) * (s * s') - D - D')

        # compute new point - - - - - - - - - - - - - - - - - - - - - - - - - -

        # possibly plot the trajectory
        if n == 2 && Plotf == 1
            PXY = hcat(PXY, hcat(x, lastx))
        end

        x = lastx

        # update gradient - - - - - - - - - - - - - - - - - - - - - - - - - - -

        g = lastg
        ng = norm(g)

        # iterate - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if Interactive
            readline()
        end
    end

    # end of main loop- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if bestf < v   # the algorithm was not monotone (the LS was interrupted)
        x = bestx
        v = bestf
    end

    if Plotf ≥ 2
        plot!(plt, gap)
    elseif Plotf == 1 && n == 2
        plot!(plt, PXY[1, :], PXY[2, :])
    end
    if plotatend
        display(plt)
    end

    return (x, status, v)
end  # the end- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
