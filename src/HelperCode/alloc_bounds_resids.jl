# ------------------------------------------------------------------- #
# Allocations
# ------------------------------------------------------------------- #
"""
Computes `b2` given `a1` and `zhat` for model where `σ1 = σ2`

Utilitzes the planner's FOC wrt a1, a2, b1, and b2

The condition `σ1 = σ2` is not checked or enforced in this function. Thus,
unless you are certain that the σ's are the same, you should use `eval_b2(m, a1,
zh)` instead
"""
function analytical_b2(m::AbstractModel, a1, zh)
    ω1, ω2, σ1 = m.agent1.ω, m.agent2.ω, m.agent1.σ
    foo = (((1-ω1)*(1-ω2))/(ω1*ω2))^(1/(σ1-1))*(a1./(zh-a1))
    B = 1 ./ zh
    b2 = B ./ (1 + foo)
end

"Given `(a1, lzh)`, solve for b2"
function eval_b2(m::AbstractModel, a1, lzh)
    if is_same_sigma(m)
        analytical_b2(m, a1, exp(lzh))
    else
        # Could allow for possibility of different σ
        error("Need the values of sigma to be the same")
    end
end

"Given (lλ☆, lzh), solve for allocations `a1, b1, c1, a2, b2, c2`"
function get_allocation(m::AbstractModel, lλ☆, lzh)

    # Get a1 and b2 from interpolation
    a1 = m.a1_itp[lλ☆, lzh]
    b2 = m.b2_itp[lλ☆, lzh]

    # Now we can get everything else
    zh = exp(lzh)
    a2 = zh - a1
    b1 = 1./zh - b2
    c1 = _agg_c(m.agent1, a1, b1)
    c2 = _agg_c(m.agent2, b2, a2)

    return a1, b1, c1, a2, b2, c2
end

"Given `curr_state=(lλ☆, lzh)`, solve for allocations `a1, b1, c1, a2, b2, c2`"
get_allocations(m::AbstractModel, curr_state::Array{Float64, 1}) =
    get_allocations(m, curr_state[1], curr_state[2])


# ------------------------------------------------------------------- #
# Bounds
# ------------------------------------------------------------------- #
"""
Compute bounds on grids for exogenous variables, given their process

It is hard to pin down the exact bounds we care about for the exogenous
processes, so, instead of guessing, we will simulate for a large number
of periods and then add a little extra space on either end to get a
good idea of where the bounds should live for these processes.
"""
function get_exog_bounds(exog::AbstractExog)
    # Get number of exogenous states
    n_exog_states = n_exog(exog)

    # Use same seed and simulate for a long time
    srand(42)
    exog_sim = simulate(exog, init(exog), randn(n_exog_states, 2_500_000))

    # Get the exogenous state (es) bounds
    es_bounds = map(i->extrema(exog_sim[i, :]), collect(1:n_exog_states))

    # Expand bounds
    exp_es_bounds = Array(Tuple{Float64, Float64}, n_exog_states)
    for i=1:n_exog_states
        x_lb, x_ub = es_bounds[i][1], es_bounds[i][2]
        exp_bds = 0.025*abs(x_ub - x_lb)
        exp_es_bounds[i] = x_lb - exp_bds, x_ub + exp_bds
    end

    return exp_es_bounds
end

"""
Compute reasonable bounds to begin bisection on 'Lars k'

Note that

    log(Lars k) = (ρ2-α2) log(μ2) - (ρ1-α1) log(μ1)

Also notice that if

* (ρi - αi) > 0.
* μ2 is increasing in lλ☆ (μ1 is decreasing in lλ☆)
* μ2 is decreasing in lzh (μ1 is increasing in lzh) <- Don't think we use

We will have that  `Lars k` is increasing in lλ☆, which implies

* upper bound to lars k is when lλ☆ is big (and lzh is small)
* lower bound to lars k is when lλ☆ is small (and lzh is big)

## Parameters

- `i::Int` an integer specifying which state in `m.grid` we are at for time t
  variables
- `vf_itps::VFITP`: A scheme for interpolating over the time t+1 value functions

## Returns
- `lb, ub`: The lower and upper bounds for Lars' k at state `i`.
"""
function log_lars_k_bounds(m::AbstractModel, i::Int, vf_itps::VFITP)

    # unpack
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)
    Nϵ, Nshocks = size(m.ϵ)

    # current states
    lλ☆_i = m.grid[i, 1]

    # t+1 states
    lgp_i = get_exog_tomorrow(m, i)[end]
    gp_i = exp(lgp_i)

    # Build a function that gets the pieces we care about
    function get_k_K(lλ☆p)
        ljp, lup = Array(Float64, Nϵ), Array(Float64, Nϵ)

        for j=1:Nϵ
            # `curr_exog_statep` will be ordered `[lzh, others..., lg]`,
            # where others is empty for cosntant volatility model and equal to
            # `v` for the stochastic volatility model.
            curr_exog_statep = get_exog_tomorrow(m, i, j)
            ljp[j] = vf_itps[1][lλ☆p[j], curr_exog_statep[1:end-1]...]
            lup[j] = vf_itps[2][lλ☆p[j], curr_exog_statep[1:end-1]...]
        end
        # evaluate certainty equivalents
        lμ1 = eval_lμ1(m, gp_i, exp(ljp))
        lμ2 = eval_lμ2(m, gp_i, exp(lup))

        # now get the lars k implied from both the FOC wrt U' _and_ the
        # definition of lars' k from in the docstring
        k_upres = up_residual(m, ljp, lup, lλ☆p, lgp_i, lλ☆_i, 0.0)
        k_defn = (ρ2-α2)*lμ2 - (ρ1-α1)*lμ1

        return [k_upres; k_defn]
    end

    # TODO: Chase will you comment about what this stuff does? Maybe add it
    #       to docstring (please check the docstring)
    lλ_step = max(m.m_ls[1][end] / 12, .1)
    lλ☆_big, lλ☆_small = m.grid[i, 1] + lλ_step, m.grid[i, 1] - lλ_step

    div_val = 2.5
    bstep = round(Int, Nϵ/div_val)

    lb = sort(get_k_K(fill(lλ☆_small, Nϵ)))[end-bstep]
    ub = sort(get_k_K(fill(lλ☆_big, Nϵ)))[bstep]

    return lb, ub
end


# ------------------------------------------------------------------- #
# Residuals
# ------------------------------------------------------------------- #
"""
Evaluate the residual for Lars' k:

    lk = (ρ2-α2) log(μ2) - (ρ1-α1) log(μ1)
"""
function lk_residual(m::AbstractModel, lμ1, lμ2, lk)

    # Unpack
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)

    return ((ρ2-α2)*lμ2 - (ρ1-α1)*lμ1) - lk
end

"""
Compute the residual for the FOC wrt `U_{t+1}` at a particular t+1 state:



Note that here we take `log Lars k` as an argument, but when the argument is
set to 0.0 then the output of this function is the implied `log Lars k`.
"""
function up_residual(m::AbstractModel, ljp, lup, lλ☆p, lgp, lλ☆, lk)

    # Unpack
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)

    log(β1/β2) + (α1-α2).*lgp + (α1-ρ1).*ljp - (α2-ρ2).*lup + lλ☆p - lλ☆ - lk
end

