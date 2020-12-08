using Distributions
using Random

"""
    pnorm(x)

Julia equivalent of R's `pnorm`
"""
function pnorm(x)
    cdf(Normal(), x)
end


"""
    dnorm(x)

Julia equivalent of R's `dnorm`
"""
function dnorm(x)
    pdf(Normal(), x)
end


# """
#     cond_mean_on_LR(L::T, R::T, a1::T, a2::T, a3::T, μ1::T, μ2::T, μ3::T) where T <: Real

# Conditional mean on (L, R), defined as μ_{L,R} = (∫_L^R f(x) dx)^(-1) / (∫_L^R x f(x) dx)
# """
# function cond_mean_on_LR(L::T, R::T, a1::T, a2::T, a3::T, μ1::T, μ2::T, μ3::T) where T <: Real
#     pL = a1 * pnorm(L-μ1) + a2 * pnorm(L-μ2) + a3 * pnorm(L-μ3)
#     pR = a1 * pnorm(R-μ1) + a2 * pnorm(R-μ2) + a3 * pnorm(R-μ3)

#     ∫_L = -(a1 * dnorm(L-μ1) + a2 * dnorm(L-μ2) + a3 * dnorm(L-μ3)) +
#           a1 * μ1 * pnorm(L-μ1) + a2 * μ2 * pnorm(L-μ2) + a3 * μ3 * pnorm(L-μ3)
#     ∫_R = -(a1 * dnorm(R-μ1) + a2 * dnorm(R-μ2) + a3 * dnorm(R-μ3)) +
#           a1 * μ1 * pnorm(R-μ1) + a2 * μ2 * pnorm(R-μ2) + a3 * μ3 * pnorm(R-μ3)

#     μ_LR = (∫_R - ∫_L)/(pR - pL)

#     (pL, pR, ∫_L, ∫_R, μ_LR)
# end


"""
    cond_mean_on_LR(L, R, a1, a2, a3, μ1, μ2, μ3)

Conditional mean on (L, R), defined as μ_{L,R} = (∫_L^R f(x) dx)^(-1) / (∫_L^R x f(x) dx)
"""
function cond_mean_on_LR(L, R, a1, a2, a3, μ1, μ2, μ3)
    # (∫_l^r f(x) dx) = pR - pL
    pL = a1 * pnorm(L.-μ1) + a2 * pnorm(L.-μ2) #+ a3 * pnorm(L.-μ3)
    pR = a1 * pnorm(R.-μ1) + a2 * pnorm(R.-μ2) #+ a3 * pnorm(R.-μ3)

    # (∫_l^r x f(x) dx) = ∫_R - ∫_L
    ∫_L = -(a1 * dnorm(L.-μ1) + a2 * dnorm(L.-μ2)) + # + a3 * dnorm(L.-μ3)) +
          a1 * μ1 * pnorm(L.-μ1) + a2 * μ2 * pnorm(L.-μ2) #+ a3 * μ3 * pnorm(L.-μ3)
    ∫_R = -(a1 * dnorm(R-μ1) + a2 * dnorm(R-μ2)) + # + a3 * dnorm(R-μ3)) +
          a1 * μ1 * pnorm(R-μ1) + a2 * μ2 * pnorm(R-μ2) #+ a3 * μ3 * pnorm(R-μ3)
    ## adding a3/μ3 terms yields inaccurate result.

    # population conditional mean on (L, R)
    μ_LR = (∫_R .- ∫_L) ./ (pR .- pL)

    (pL, pR, ∫_L, ∫_R, μ_LR)
end


"""
    find_split(d, a1, a2, μ1, μ2, L, R;
               a3=0., μ3=0., find_deltas=false, find_split=!find_deltas)

Split the cluster (L, R) into subclusters that maximizes G_{L,R}.
Returns centered conditional mean (deltas) of the left and right subcluster
if `find_deltas` is true. Otherwise return the split point.
"""
function find_split(d, a1, a2, μ1, μ2, L, R;
                    a3=0., μ3=0., find_deltas=false, find_split=!find_deltas)
    # p, int, int_en evaluated at `d`
    ∫_d = -(a1 * dnorm(d.-μ1) + a2 * dnorm(d.-μ2)) + # + a3 * dnorm(d.-μ3)) +
          a1 * μ1 * pnorm(d.-μ1) + a2 * μ2 * pnorm(d.-μ2) #+ a3 * μ3 * pnorm(d.-μ3)
    p_d = a1 * pnorm(d.-μ1) + a2 * pnorm(d.-μ2) #+ a3 * pnorm(d.-μ3)

    pL, pR, ∫_L, ∫_R, μ_LR = cond_mean_on_LR(L, R, a1, a2, a3, μ1, μ2, μ3)

    # find split point with normlized G'_{L,R} (=h)(supp. p.32)
    h = μ_LR .* (p_d .- pL).^2 + (pL + pR .- 2p_d) .* (∫_d .- ∫_L) - d .* (p_d .- pL) .* (pR .- p_d)
    split = d[2:end][h[2:end] .* h[1:end-1] .< 0]

    if find_deltas && (sum(μ1 .< split .< μ2) == 0)
        return(1., 1.)
    end

    # split point
    s = max(split[μ1 .< split .< μ2]...)

    if find_split
        return(s)
    else
        _, _, _, _, δ_1 = cond_mean_on_LR(L, s, a1, a2, a3, μ1, μ2, μ3)
        δ_1 -= (L + s)/2  # centering

        _, _, _, _, δ_2 = cond_mean_on_LR(s, R, a1, a2, a3, μ1, μ2, μ3)
        δ_2 -= (s + R)/2  # centering

        return(δ_1, δ_2)
    end

end


"""
    find_truncation(a1, a2, μ1, μ2,
                    R_ub=(μ2 + 8), L_lb=(μ1 - 8),
                    tuning_par1=0.01, tuning_par2=0.005,
                    a3=0, μ3=0, σ=1)

Find population right-truncation points for bimodal Normal mixtures.
Inputs:
    a1, a2, a3: weights for Normal distributions. a1 + a2 + a3 should be == 1
    μ1, μ2, μ3: means for normals μ1 = -μ2

Possible returns:
    0: negligible result (not a symmetric problem)
    -1: no truncation
"""
function find_truncation(a2, μ2;
                         a1=1-a2, μ1=-μ2, R_ub=μ2+8., L_lb=μ1-8.,
                         tuning_par1=0.01, tuning_par2=0.005,
                         a3=0., μ3=0., σ=1.)

    r = Array(R_ub:(-tuning_par1):μ1)

    # if the bimodal distn is symmetric, and a1 == a2,
    # then truncate both ends equally
    if a1 == a2 && μ2 == -μ1
        for i=1:length(r)

            # truncate both left and right with the same amount
            R = r[i]
            L = -R

            d = Array(L:0.01:R)
            δ_1, δ_2 = find_split(d, a1, a2, μ1, μ2, L, R; find_deltas=true)

            # L^* is the smallest L that satisfies δ_1 < 0 < δ_2. (supp. p.30)
            # Thus returns right-trunctation point
            if δ_1 < 0 < δ_2
                return(R)
            end
        end
    end

    # possible cases:
    #   a1 != a2 && μ2 != -μ1
    #   a1 != a2 && μ2 == -μ1
    if a1 != a2
        L_old = -1000
        for i=1:length(r)
            R = r[i]
            L = Array{Float64,1}(L_old:tuning_par2:(R-0.05))

            pL, pR, ∫_L, ∫_R, μ_LR = cond_mean_on_LR(L, R, a1, a2, a3, μ1, μ2, μ3)

            # locate R_L using the equation μ_{L,R_L} - (L + R_L)/2.  (supp. p.30)
            temp = sign.(μ_LR .- (L .+ R)./2)

            if sum(temp[2:end] .* temp[1:end-1] .< 0) ≠ 0
                # this is L^* from (supp. p.28)
                L = L[2:end][temp[2:end] .* temp[1:end-1] .< 0][1]

                d = Array(L:0.01:R)
                δ_1, δ_2 = find_split(d, a1, a2, μ1, μ2, L, R; find_deltas=true)

                pL = a1 * pnorm(L-μ1) + a2 * pnorm(L-μ2) + a3 * pnorm(L-μ3)
                pR = a1 * pnorm(R-μ1) + a2 * pnorm(R-μ2) + a3 * pnorm(R-μ3)

                if sign((R-L)/(pR-pL) * (a1 * dnorm(L-μ1) + a2 * dnorm(L-μ2) + a3 * dnorm(L-μ3)) - 1) ≠ -1
                    println("🤔🤔")
                end

                # L^* is the smallest L that satisfies δ_1 < 0 < δ_2. (supp. p.30)
                # Thus returns right-trunctation point
                if δ_1 < 0 < δ_2 && L_old ≤ L

                    # if L^* is achieved at i==1, it means there is no truncation.
                    if i == 1
                        return(-1)
                    end

                    return(R)
                end

                # increment L
                if L_old ≤ L
                    L_old = L
                end

                # if L > 0, it means roughly that L > m_ (m_≤0)  (supp. p.29 table and p.28)
                if L > 0
                    return(-1)
                end

            end
        end
    end

    # If a1 == a2 and μ1 != -μ2, then return 0
    # : do not need to assess in this case, since it can WLOG be reduced to a symmetric case)
    return(0)
end


"""
    clusterpath_pop(a2::T, μ2::T; a1=1-a2, μ1=-μ1, a3=0., μ3=0, σ=1) where T <: Real

Inputs:
    a2: weight on 2nd normal distribution
    μ2: mean of 2nd normal distribution
Returns:
    A dictionary of weights (`a1`, `a2`), means (`μ1`, `μ2`) and
    description of truncation (`L*`, `R*`) and split point (`s`).
"""
function clusterpath_pop(a2::T, μ2::T; a1=1-a2, μ1=-μ2, a3=0., μ3=0, σ=1.) where T <: Real
    # search right truncation point
    R = find_truncation(a2, μ2; a1=a1, μ1=μ1, a3=a3, μ3=μ3, σ=σ)

    # R == 0 or R == -1
    if R == 0 || R == -1
        d = Array(μ1:1e-04:μ2)
        # d[argmin(abs.(a1 * dnorm(d.-μ1) .- a2 * dnorm(d.-μ2)))]  ## ???????
        # d[argmin(a1 * dnorm(d.-μ1) .+ a2 * dnorm(d.-μ2))]

        return(Dict("a1" => a1, "a2" => a2,
                    "mu1" => μ1, "mu2" => μ2,
                    "s" => NaN, "L*" => NaN, "R*" => NaN))
    end

    # If R is a proper right-truncation point, then find split point
    L_lb = -3 * max(R, 3)  # lower bound of searching grid of L
    L_ub = R - 0.01  # upper bound
    L = Array(L_lb:5e-04:L_ub)

    _, _, _, _, μ_LR = cond_mean_on_LR(L, R, a1, a2, a3, μ1, μ2, μ3)
    δs = sign.(μ_LR .- (L .+ R)/2)

    # L^* is the smallest L that satisfies δ_1 < 0 < δ_2
    L = L[2:end][δs[2:end] .* δs[1:end-1] .< 0][1]

    d = Array(L:0.001:R)
    split = find_split(d, a1, a2, μ1, μ2, L, R)

    return(Dict("a1" => a1, "a2" => a2,
                "mu1" => μ1, "mu2" => μ2,
                "s" => split, "L*" => L, "R*" => R))
end
