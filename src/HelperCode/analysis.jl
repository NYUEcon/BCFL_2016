# ------------------------------------------------------------------- #
# Useful helper functions
# ------------------------------------------------------------------- #
"Use the FOC wrt c1, c2, a1, a2 to comptue the exchange ratio p2/p1"
function get_exchangerate(m::AbstractModel, c1, c2, a1, a2)

    # Unpack parameters
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)

    # Get numerator and denominator
    p2 = c1.^(1-σ1) .* (1-ω1) .* a1.^(σ1-1)
    p1 = c2.^(1-σ2) .* ω2 .* a2.^(σ2-1)

    # fxr = p2/p1 = num/den
    return p2/p1
end

"Use the FOC wrt c1, c2, b1, b2 to comptue the exchange ratio p2/p1"
function get_exchangerateb(m::AbstractModel, c1, c2, b1, b2)

    # Unpack parameters
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)

    # Get numerator and denominator
    p2 = c1.^(1-σ1) .* ω1 .* b1.^(σ1-1)
    p1 = c2.^(1-σ2) .* (1-ω2) .* b2.^(σ2-1)

    # fxr = p2/p1 = num/den
    return p2/p1
end

"""
Given the model and a matrix `Nshocks × Ngrid` matrix of optimal policy choices,
construct an interpolant for lλ☆p as a function of (lλ☆, lzh, v, ϵ1, ϵ2, ϵ3)

(NOTE if `m` is a `ModelCV` then the variables `v` and `ϵ3` do not apply.)
"""
function get_lλ☆p_itp(m::AbstractModel, lλ☆p::Matrix{Float64})
    # Get the unique shocks for each shock
    eps_grids = Vector[unique(m.ϵ[:, i]) for i in 1:n_exog(m.exog)]

    # Build a gridded interpolant (so we can have uneven grids)
    linspace_grids = map(collect, m.m_ls)
    knots = (linspace_grids..., eps_grids...)
    Nknots = map(length, knots)
    lλ☆p_rs = reshape(lλ☆p', Nknots...)

    # Create interpolant
    lλ☆p_itp = interpolate(knots, lλ☆p_rs, Gridded(Linear()))

    return lλ☆p_itp
end

# ------------------------------------------------------------------- #
# Simulation functions
# ------------------------------------------------------------------- #
"""
Simulate the model given an interpolant for lλ☆p, a sequence of shocks for
`ϵ` and an initial level of lλ☆p.

Note that ϵ is assumed to be `Nshocks × T`, where `Nshocks` is the number of
shocks in the model (2 for CV, 3 for SV) and `T` is the desired simulation
length.

The function returns simulated paths of a1, b1, c1, a2, b2, c2, lλ☆, exogvals,
c2_c1_ratio, fxr,

where

- `exogvals` = `(lzh, lg)`` in the CV model and `(lzh, v, lg)` in the SV model
- `fxr = p2/p1`
"""
function simulate(m::AbstractModel, lλ☆p_itp::AbstractInterpolation,
                  ϵ::Matrix{Float64}; lλ0=0.0)

    # Simulate the exogenous parameters
    nshocks, capT = size(ϵ)
    initvals = init(m.exog)
    exogvals = simulate(m.exog, initvals, ϵ)

    # Allocate space for remaining info
    a1 = Array(Float64, capT-1)
    b1 = Array(Float64, capT-1)
    c1 = Array(Float64, capT-1)
    a2 = Array(Float64, capT-1)
    b2 = Array(Float64, capT-1)
    c2 = Array(Float64, capT-1)
    lλ☆ = Array(Float64, capT-1)
    fxr = Array(Float64, capT-1)

    # Initial condition and simulate
    lλ☆[1] = lλ0
    for t=1:capT-1
        # Get period t allocation
        lλ☆_t = lλ☆[t]

        # Get allocation values
        a1[t], b1[t], c1[t], a2[t], b2[t], c2[t] = get_allocation(m, lλ☆_t, exogvals[1, t])
        fxr[t] = get_exchangerate(m, c1[t], c2[t], a1[t], a2[t])

        if t<capT-1
            lλ☆[t+1] = lλ☆p_itp[lλ☆[t], exogvals[1:end-1, t]..., ϵ[:, t+1]...]
        end
    end
    c2_c1_ratio = c2 ./ c1

    return a1, b1, c1, a2, b2, c2, lλ☆, exogvals, c2_c1_ratio, fxr
end

# API method above
simulate(m::AbstractModel, lλ☆p_itp::AbstractInterpolation; capT=10_000, lλ0=0.0) =
    simulate(m, lλ☆p_itp, randn(n_exog(m), capT), lλ0=lλ0)

"""
Approximate the stochastic steady state of lλ☆ by simulating the stochastic
version of the model and computing the observed mean of lλ☆.
"""
function find_steadystate_lλ(m::AbstractModel, lλ☆p_itp::AbstractInterpolation;
                             capT=50_000)

    return mean(simulate(m, lλ☆p_itp; capT=capT)[7][end-10_000:end])
end

# ------------------------------------------------------------------- #
# Euler Errors
# ------------------------------------------------------------------- #
"""
Compute euler errors given the model, a level of lλ☆, the exogenous state,
and interpolands for the value functions and policy function. `exog_state` will
contain `(lzh, v)`
"""
function euler_err_t(m::ModelSV, lλ☆::Float64, exog_state::Array{Float64},
                     vf_itps::VFITP, lλ☆p_itp::AbstractInterpolation)

    # Unpack parameters
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)
    Nϵ = length(m.Π)

    # Evaluate time t allocations
    lzh, v = exog_state[1], exog_state[2]
    a1, b1, c1, a2, b2, c2 = get_allocation(m, lλ☆, lzh)

    # Get all possible lλ☆p values for tomorrow
    lλ☆p = Float64[lλ☆p_itp[lλ☆, exog_state..., m.ϵ[i, :]...] for i in 1:Nϵ]

    # Get t+1 possible exogenous values
    exog_p = exog_step(m.exog, exog_state, m.ϵ)
    lgp = exog_p[end]
    gp = exp(lgp)

    # Get t+1 value functions
    ljp, lup = map(n->Array(Float64, n), fill(Nϵ, 2))
    for j=1:Nϵ
        curr_exogp_state = map(x->x[j], exog_p[1:end-1])
        ljp[j] = vf_itps[1][lλ☆p[j], curr_exogp_state...]
        lup[j] = vf_itps[2][lλ☆p[j], curr_exogp_state...]
    end

    # Evaluate μ values
    lμ1 = log(_CE(m.agent1, gp, exp(ljp), m.Π))
    lμ2 = log(_CE(m.agent2, gp, exp(lup), m.Π))

    # Evaluate current value functions
    lj, lu = eval_vfs(m, c1, c2, exp(lμ1), exp(lμ2))

    # Evaluate Euler Equation
    lk = lk_residual(m, lμ1, lμ2, 0.0)
    ee = exp(up_residual(m, ljp, lup, lλ☆p, lgp, lλ☆, lk)) - 1.0

    return ee
end

"Compute the Euler errors for the model along a random simulation path"
function euler_err_sim(m::AbstractModel, vf_itps::VFITP,
                       lλ☆p_itp::AbstractInterpolation; capT=10_000)

    # Allocate memory
    eul_err = Array(Float64, length(m.Π), capT)

    # Simulate economy
    simvals = simulate(m, lλ☆p_itp, randn(3, capT+1))
    lλ☆ = simvals[7]
    exogvals = simvals[8]

    for t=1:capT
        eul_err[:, t] = euler_err_t(m, lλ☆[t], exogvals[1:end-1, t], vf_itps, lλ☆p_itp)
    end

    return eul_err
end

