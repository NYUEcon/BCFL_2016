# ------------------------------------------------------------------- #
# Constant Volatility Exogenous Process
# ------------------------------------------------------------------- #
"""
This is the type that holds the parameters for a constant volatility
exogenous process from the BCFL paper. Namely the process follows:

```
\log(\hat{z}_{t+1}) = (1 - 2γ) \log(\hat{z}_{t}) + (σz1/2) ϵ1 - (σz2/2) ϵ2
\log(g_{t+1}) = (σz1/2) ϵ1 + (σz2/2) ϵ2
```

The number of epsilons we use for quadrature in each i is given by nϵi
"""
immutable ExogCV <: AbstractExog
    γ::Float64
    σz1::Float64
    σz2::Float64
    nϵ1::Int
    nϵ2::Int
end

# Small self-explanatory helper functions
init(::ExogCV) = [0.0, 0.0]
n_exog(::ExogCV) = 2
_unpack(e::ExogCV) = (e.γ, e.σz1, e.σz2, e.nϵ1, e.nϵ2)

const _cv_exog_ordering_docstr_extra = """
Because lzhp is a time t+1 state variable, but lgp is not, the ordering of
variables here is significant. Specifically, at points in the code it is assumed
that lzhp comes first and lgp comes last. This is also true in the stochastic
volatiltiy version of the model where there there is an additional state `vp`
placed between the other two.
"""

"""
Takes one step for exogenous process, given current levels and innovations

$_cv_exog_ordering_docstr_extra
"""
function exog_step(exog::ExogCV, lzh, ϵ1, ϵ2)
    γ, σz1, σz2, nϵ1, nϵ2 = _unpack(exog)
    lzhp = (1-2γ)*lzh .+ 0.5*(σz1.*ϵ1 .- σz2*ϵ2)
    lgp = 0.0*lzh .+ 0.5*σz1.*ϵ1 .+ 0.5*σz2*ϵ2

    lzhp, lgp
end

# API method on above
exog_step(exog::ExogCV, current::Vector, ϵ::Vector) =
    exog_step(exog, current[1], ϵ[1], ϵ[2])

# ------------------------------------------------------------------- #
# Constant Volatility Model
# ------------------------------------------------------------------- #
type ModelCV <: AbstractModel
    agent1::Agent
    agent2::Agent
    exog::ExogCV
    m_ls::Tuple{LinSpace{Float64}, LinSpace{Float64}}
    grid::Matrix{Float64}
    grid_transpose::Matrix{Float64}
    lλ☆p_pol::Matrix{Float64}
    ϵ::Matrix{Float64}
    Π::Vector{Float64}
    lzhp::Matrix{Float64}
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
    - Define grids over state variables (lλ☆, lzh)
    - Define an initial guess for the policy of lλ☆p
    - Build quadrature weights and nodes to approximate the expectations
    - Precompute solutions the allocation in terms of lλ☆ and lzh

    Note that all parameters match the notation from the paper and are settable
    one at a time using keyword argument syntax (`ModelCV(β2=0.99)`) changes
    only β2 relative to the default values. The only notational exception is
    that the standard deviation of the shock to `zi` is `σzi` here and
    `sqrt(vi)` in the paper.

    Other arguments to this constructor are:

    - `lλ☆bds`: bound on grid for lλ☆. The grid is assumed to be symmetric,
    extending from `-lλ☆bds` to `lλ☆bds`.
    - `nX` for various `X`: the number of grid points for variable `X`
    - `σz1` and `σz2`: standard deviation of shocks to `lz1` and `lz2` (see above)
    """
    function ModelCV(;ρ1::Real=-1.0, α1::Real=-9.0, β1::Real=0.98,
                      ω1::Real=0.1, σ1::Real=0.0,
                      ρ2::Real=-1.0, α2::Real=-9.0, β2::Real=0.98,
                      ω2::Real=0.1, σ2::Real=0.0,
                      γ::Real=0.10, σz1::Real=0.015, σz2::Real=0.015,
                      lλ☆bds=4.6, nλ☆::Int=451, nlzh::Int=15,
                      nϵ1::Int=5, nϵ2::Int=5)

        # pack up agents and exog
        agent1 = Agent(ρ1, α1, β1, ω1, σ1)
        agent2 = Agent(ρ2, α2, β2, ω2, σ2)
        exog = ExogCV(γ, σz1, σz2, nϵ1, nϵ2)

        # Get linspaces that define grids  for lλ☆, zh
        lλ☆_grid = linspace(-lλ☆bds, lλ☆bds, nλ☆)

        # Get exogenous grid bounds from simulation of 5,000,000 periods
        lzh_bds, _ = get_exog_bounds(exog)
        lzh_grid = linspace(lzh_bds[1], lzh_bds[2], nlzh)
        m_ls = (lλ☆_grid, lzh_grid)

        # Create a grid
        grid = gridmake(collect(lλ☆_grid), collect(lzh_grid))
        grid_transpose = grid'
        lλ☆_pol = repeat(grid_transpose[1, :], inner=[nϵ1*nϵ2, 1])

        # build exog grid for next period.  Both lgp, ξp are Neps × Ngrid
        ϵ, Π = qnwnorm([nϵ1, nϵ2], [0.0, 0.0], eye(2))
        lzhp, lgp = exog_step(exog, grid[:, 2]', ϵ[:, 1], ϵ[:, 2])

        # incompletely initialize the ModelCV so we can pass it to routines that
        # construct the `OtherItems` for us
        m = new(agent1, agent2, exog, m_ls, grid, grid_transpose, lλ☆_pol,
                ϵ, Π, lzhp, lgp)

        # Solve for b2 on the grid and get the coefficients for its approximant
        a1_itp, b2_itp = solve_for_a1b2(m, (-lλ☆bds, lλ☆bds), lzh_bds)
        m.a1_itp = a1_itp
        m.b2_itp = b2_itp

        # # solve for rest of allocation and fill in other items field of ModelCV
        a1 = Float64[a1_itp[grid_transpose[:, i]...] for i=1:nλ☆*nlzh]
        b2 = Float64[b2_itp[grid_transpose[:, i]...] for i=1:nλ☆*nlzh]

        a2 = exp(grid[:, 2]) - a1
        b1 = 1.0 ./ exp(grid[:, 2]) - b2

        c1 = _agg_c(m.agent1, a1, b1)
        c2 = _agg_c(m.agent2, b2, a2)

        m.other = OtherItems(a1, b2, c1, c2)

        m
    end

end

"""
Extracts time t+1 levels of lzh and lgp on the grid in state i at time t for all
time t+1 states

$_cv_exog_ordering_docstr_extra
"""
function get_exog_tomorrow(m::ModelCV, i::Int)
    lzhp_i = m.lzhp[:, i]
    lgp_i = m.lgp[:, i]
    lzhp_i, lgp_i
end

"""
Extracts time t+1 levels of lzh and lgp on the grid in state j at time t+1 and i
at time t

$_cv_exog_ordering_docstr_extra
"""
function get_exog_tomorrow(m::ModelCV, i::Int, j::Int)
    lzhp_ij = m.lzhp[j, i]
    lgp_ij = m.lgp[j, i]
    lzhp_ij, lgp_ij
end

jld_prefix(::ModelCV) = "curr_soln_cv"
jld_prefix(::Type{ModelCV}) = "curr_soln_cv"

