# ------------------------------------------------------------------- #
# Stochastic Volatility Exogenous Process
# ------------------------------------------------------------------- #
"""
This is the type that holds the parameters for a constant volatility
exogenous process from the BCFL paper. Namely the process follows:

```
\log(\hat{z}_{t+1}) = (1 - 2γ) \log(\hat{z}_{t}) + (σz1/2) ϵ1 - (σz2/2) ϵ2
\log(g_{t+1}) = (σz1/2) ϵ1 + (σz2/2) ϵ2
v_{t+1} = (1-ϕv) vbar + ϕv v_{t} + τ ϵ3
```

The number of epsilons we use for quadrature in each i is given by nϵi
"""
immutable ExogSV <: AbstractExog
    γ::Float64
    σz2::Float64
    vbar::Float64
    φv::Float64
    τ::Float64
    nϵ1::Int
    nϵ2::Int
    nϵ3::Int
end


# Small self-explanatory helper functions
init(e::ExogSV) = [0.0, e.vbar, 0.0]
n_exog(e::ExogSV) = 3
_unpack(e::ExogSV) = (e.γ, e.σz2, e.vbar, e.φv, e.τ, e.nϵ1, e.nϵ2, e.nϵ3)

const _sv_exog_ordering_docstr_extra ="""
Because lzhp and vp are time t+1 state variables, but lgp is not, the ordering
of variables here is significant. Specifically, at points in the code it is
assumed that lzhp comes first and lgp comes last. This is also true in the
constant volatiltiy version of the model where there is
"""

"""
Takes one step for exogenous process, given current levels and innovations

$_sv_exog_ordering_docstr_extra
"""
function exog_step(exog::ExogSV, lzh, v, ϵ1, ϵ2, ϵ3)
    γ, σz2, vbar, φv, τ, nϵ1, nϵ2, nϵ3 = _unpack(exog)
    lzhp = (1-2γ)*lzh .+ 0.5*((sqrt(v).*ϵ1) .- σz2*ϵ2)
    lgp = 0.0.*lzh .+ 0.5*sqrt(v).*ϵ1 .+ 0.5*σz2*ϵ2
    vp = (1-φv)*vbar + φv*v .+ τ*ϵ3

    lzhp, vp, lgp
end

# API methods on above function
exog_step(exog::ExogSV, current::Vector, ϵ::Vector) =
    exog_step(exog, current[1], current[2], ϵ[1], ϵ[2], ϵ[3])

exog_step(exog::ExogSV, current::Vector, ϵ::Matrix) =
    exog_step(exog, current[1], current[2], ϵ[:, 1], ϵ[:, 2], ϵ[:, 3])


# ------------------------------------------------------------------- #
# Stochastic Volatility Model
# ------------------------------------------------------------------- #
type ModelSV <: AbstractModel
    agent1::Agent
    agent2::Agent
    exog::ExogSV
    m_ls::NTuple{3,LinSpace{Float64}}
    grid::Matrix{Float64}
    grid_transpose::Matrix{Float64}
    lλ☆p_pol::Matrix{Float64}
    ϵ::Matrix{Float64}
    Π::Vector{Float64}
    lzhp::Matrix{Float64}
    vp::Matrix{Float64}
    lgp::Matrix{Float64}
    a1_itp::ScaledInterpolation
    b2_itp::ScaledInterpolation
    other::OtherItems

    """
    Given the paramters of the model as well as information on the coarseness
    of the grids, do initial setup work to define the model. This involves
    the following steps:

    - Construct instances of `Agent` to hold preference paramters for
      both agents
    - Construct an object to hold the paramters of the exogenous process
    - Define grids over state variables (lλ☆, lzh, v)
    - Define an initial guess for the policy of lλ☆p
    - Build quadrature weights and nodes to approximate the expectations
    - Precompute solutions the allocation in terms of lλ☆ and lzh (these
      decisions are given by the associated FOC and are independent of v)

    Note that all parameters match the notation from the paper and are settable
    one at a time using keyword argument syntax (`ModelSV(β2=0.99)`) changes
    only β2 relative to the default values. The only notational exception is
    that the standard deviation of the shock to `z2` is `σz2` here and
    `sqrt(v)` in the paper.

    Other arguments to this constructor are:

    - `lλ☆bds`: bound on grid for lλ☆. The grid is assumed to be symmetric,
    extending from `-lλ☆bds` to `lλ☆bds`.
    - `nX` for various `X`: the number of grid points for variable `X`
    - `σz2`: standard deviation of shock to `lz2` (see above)
    """
    function ModelSV(;ρ1::Real=-1.0, α1::Real=-9.0, β1::Real=0.98,
                      ω1::Real=0.1, σ1::Real=0.0,
                      ρ2::Real=-1.0, α2::Real=-9.0, β2::Real=0.98,
                      ω2::Real=0.1, σ2::Real=0.0,
                      γ::Real=0.10, vbar::Real=0.015^2, σz2::Real=0.015,
                      φv::Real=0.9, τ::Real=0.74e-5, lλ☆bds=4.6,
                      nλ☆::Int=451, nlzh::Int=13, nv::Int=13,
                      nϵ1::Int=5, nϵ2::Int=5, nϵ3::Int=5)

        # pack up agents and exog
        agent1 = Agent(ρ1, α1, β1, ω1, σ1)
        agent2 = Agent(ρ2, α2, β2, ω2, σ2)
        exog = ExogSV(γ, σz2, vbar, φv, τ, nϵ1, nϵ2, nϵ3)

        # Get linspaces that define grids  for lλ☆, zh, vh
        lλ☆_grid = linspace(-lλ☆bds, lλ☆bds, nλ☆)
        lλ☆_bds = (-lλ☆bds, lλ☆bds)

        # Get exogenous grid bounds from simulation of 5,000,000 periods
        lzh_bds, v_bds, _ = get_exog_bounds(exog)
        lzh_grid = linspace(lzh_bds[1], lzh_bds[2], nlzh)
        v_grid = linspace(v_bds[1], v_bds[2], nv)
        m_ls = (lλ☆_grid, lzh_grid, v_grid)

        # Create a grid
        grid = gridmake(collect(lλ☆_grid), collect(lzh_grid), collect(v_grid))
        grid_transpose = grid'

        # Initial guess at policy for lλ☆_{t+1} at each exog state in t+1 is
        # the time t level on the grid
        lλ☆_pol = repeat(grid_transpose[1, :], inner=[nϵ1*nϵ2*nϵ3, 1])

        # build exog grid for next period.  Both lgp, ξp are Neps × Ngrid
        ϵ, Π = qnwnorm([nϵ1, nϵ2, nϵ3], [0.0, 0.0, 0.0], eye(3))
        lzhp, vp, lgp = exog_step(exog, grid[:, 2]', grid[:, 3]',
                                  ϵ[:, 1], ϵ[:, 2], ϵ[:, 3])

        # incompletely initialize the ModelSV so we can pass it to routines that
        # construct the `OtherItems` for us
        m = new(agent1, agent2, exog, m_ls, grid, grid_transpose, lλ☆_pol,
                ϵ, Π, lzhp, vp, lgp)

        # Precompute solutions for a1 and b2 on a very fine grid for lλ☆ and
        # lzh. Then store coefficients of corresponding interpoland
        a1_itp, b2_itp = solve_for_a1b2(m, lλ☆_bds, lzh_bds)
        m.a1_itp = a1_itp
        m.b2_itp = b2_itp

        # Use interpolands for a1, b2 to find these variables on the grid we
        # will be solving the rest of the model on
        a1 = Float64[a1_itp[grid_transpose[1:2, i]...] for i=1:nλ☆*nlzh*nv]
        b2 = Float64[b2_itp[grid_transpose[1:2, i]...] for i=1:nλ☆*nlzh*nv]

        # now extract the rest of the allocation and package into the
        # `OtherItems` object, attching it to the model
        a2 = exp(grid[:, 2]) - a1
        b1 = 1.0 ./ exp(grid[:, 2]) - b2
        c1 = _agg_c(m.agent1, a1, b1)
        c2 = _agg_c(m.agent2, b2, a2)
        m.other = OtherItems(a1, b2, c1, c2)

        m
    end
end

"""
Extracts time t+1 levels of lzh, v, and lgp on the grid in state i at time t for
all time t+1 states

$_sv_exog_ordering_docstr_extra
"""
function get_exog_tomorrow(m::ModelSV, i::Int)
    lzhp_i = m.lzhp[:, i]
    lgp_i = m.lgp[:, i]
    vp_i = m.vp[:, i]
    lzhp_i, vp_i, lgp_i
end

"""
Extracts time t+1 levels of lzh, v, and lgp on the grid in state j at time t+1
and i at time t

$_sv_exog_ordering_docstr_extra
"""
function get_exog_tomorrow(m::ModelSV, i::Int, j::Int)
    lzhp_ij = m.lzhp[j, i]
    lgp_ij = m.lgp[j, i]
    vp_ij = m.vp[j, i]
    lzhp_ij, vp_ij, lgp_ij
end

jld_prefix(::ModelSV) = "curr_soln_cv"
jld_prefix(::Type{ModelSV}) = "curr_soln_cv"

