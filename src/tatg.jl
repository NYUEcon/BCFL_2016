module Exchange

#
# Need these packages to run the code
#
using CompEcon: brent, gridmake, qnwnorm
using Interpolations
using JLD

#
# Abstract and Typle alias declarations
#
typealias VFITP Tuple{ScaledInterpolation, ScaledInterpolation}

# Abstract Model
abstract AbstractModel

Base.writemime{TM<:AbstractModel}(io::IO, ::MIME"text/plain", m::TM) =
    print(io, "The model of type $TM")

"Identifies how many exogenous variables the model has"
n_exog(m::AbstractModel) = n_exog(m.exog)

"Evaluates certainty equivalents for agent 1"
eval_lμ1(m::AbstractModel, gp::Vector, jp::Vector) =
    log(_CE(m.agent1, gp, jp, m.Π))

"Evaluates certainty equivalents for agent 1"
eval_lμ2(m::AbstractModel, gp::Vector, up::Vector) =
    log(_CE(m.agent2, gp, up, m.Π))

"Evaluate value function for agent 1 given consumption and cert equiv"
eval_lj(m::AbstractModel, c1, μ1) = _agg_lv(m.agent1, c1, μ1)

"Evaluate value function for agent 2 given consumption and cert equiv"
eval_lu(m::AbstractModel, c2, μ2) = _agg_lv(m.agent2, c2, μ2)

"This holds allocation values at each state on our grid"
type OtherItems
    a1::Vector{Float64}
    b2::Vector{Float64}
    c1::Vector{Float64}
    c2::Vector{Float64}
end

# Abstract Exog
abstract AbstractExog

"""
Simulates the model's exogenous variables. Needs an `exog` type that
describes the exogenous processes, `init` which is an initial vector of
levels for exogenous variables, and `ϵ` which are the standard normal
i.i.d. shocks.

Note: In order for this function to work, a method named `exog_step` must
be defined for the `exog::AbstractExog` type that is passed in.
"""
function simulate(exog::AbstractExog, init::Vector, ϵ::Matrix)
    capT = size(ϵ, 2)

    # allocate
    out = similar(ϵ)
    out[:, 1] = init

    N = length(init)

    # step forward
    for t in 2:capT
        next = exog_step(exog, out[:, t-1], ϵ[:, t])
        for i in 1:N
            out[i, t] = next[i]
        end
    end
    out
end

"""
Simulates exogenous variables. Calls other `simulate` method but
initializes the state of exogenous variables.
"""
simulate(exog::AbstractExog, ϵ::Matrix) = simulate(exog, init(exog), ϵ)

"""
Given a model, construct an intitial guess for the value functions on the grid.

If the keyword arugment `infile` it should point to a `.jld` file containing at
least three top level (not in HDF5 groups) variables: `lj`, `lu`, `m_ls`
specifying log J, log U and linspaces forming the grid over (lλ☆, lzh, v);
respectively.

If the file is not valid or not given, then the guess that `lj = c1` and `lu=c2`
is used.
"""
function initialize_vf(m::AbstractModel;
                       infile="solutions/$(jld_prefix(m))_1.jld")

    # Pull out useful data
    Ngrid, Nst = size(m.grid)

    # Check for the file
    try  # try to use infile. If it fails, just set l, j to c
        # Read in solution data
        # Read in solution data
        f = jldopen(infile, "r")
        lj = read(f, "lj")
        lu = read(f, "lu")
        m_ls = read(f, "m_ls")
        close(f)

        # Create interpolants
        lj_itp = scale(interpolate(lj, BSpline(Linear()), OnGrid()), m.m_ls...)
        lu_itp = scale(interpolate(lu, BSpline(Linear()), OnGrid()), m.m_ls...)

        # Allocate space to fill
        lj_grid, lu_grid = Array(Float64, Ngrid), Array(Float64, Ngrid)

        # Fill grids
        for i=1:Ngrid
            lj_grid[i] = lj_itp[m.grid_transpose[:, i]...]
            lu_grid[i] = lu_itp[m.grid_transpose[:, i]...]
        end
        lj_grid, lu_grid
    catch e
        # Otherwise guess it is the same as consumption
        warn("Failed to use infile with error:\n$e")
        lj_grid = eval_lj(m, m.other.c1, m.other.c1)
        lu_grid = eval_lu(m, m.other.c2, m.other.c2)

        lj_grid, lu_grid
    end
end


#
# Bring in our files
#
include("HelperCode/alloc_bounds_resids.jl")
include("HelperCode/agents.jl")
include("HelperCode/model_cv.jl")
include("HelperCode/model_sv.jl")
include("HelperCode/analysis.jl")

# ------------------------------------------------------------------- #
# Helper Functions
# ------------------------------------------------------------------- #
"Checks if agents have same sigma parameter"
is_same_sigma(m::AbstractModel) = abs(m.agent1.σ-m.agent1.σ) < 1e-14

"""
Finds a level of (a1, b2) for each possible pair of (lλ☆, lzh)

Takes a lower and upper bound for both lλ☆ and lzh and builds a very
fine grid over both of them. It then uses various first order conditions
to get the allocations (a1, b1, c1, a2, b2, c2) for each possible
combination of lλ☆ and lzh.

## Parameters

- `m::AbstractModel` : The type that describes our model
- `lλ_bds::Tuple{Float64, Float64}` : A lower and upper bound for lλ☆
- `lzh_bds::Tuple{Float64, Float64}` : A lower and upper bound for lzh
- `nlλ☆::Int` : The number of points to include on lλ☆ grid
- `nlzh::Int` : The number of points to include on lzh grid

## Returns

- `a1_itp::Interpolant` : An interpolating function that takes (lλ☆, lzh)
    and returns a value for a1.
- `b2_itp::Interpolant` : An interpolating function that takes (lλ☆, lzh)
    and returns a value for b2.

"""
function solve_for_a1b2(m::AbstractModel, lλ_bds, lzh_bds; nlλ☆=1500, nlzh=250)

    # Unpack parameters
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)

    # Create our fine grids
    lλ☆_fg = linspace(lλ_bds[1]*1.08, lλ_bds[2]*1.08, nlλ☆)
    lzh_fg = linspace(lzh_bds[1], lzh_bds[2], nlzh)

    # Allocate space
    a1_vals = Array(Float64, nlλ☆, nlzh)
    b2_vals = Array(Float64, nlλ☆, nlzh)


    for (i_λ, lλ_val) in enumerate(lλ☆_fg)
        for (i_zh, lzh_val) in enumerate(lzh_fg)
            zh = exp(lzh_val)

            # This function evaluates the residual from combination of
            # focs
            function a1b2_resid(a1)
                # Get allocations corresponding to a1
                b2 = eval_b2(m, a1, lzh_val)
                b1 = 1/zh - b2
                a2 = zh - a1

                # Get consumption
                c1 = _agg_c(m.agent1, a1, b1)
                c2 = _agg_c(m.agent2, b2, a2)

                rhs = log((1-β1)/(1-β2)) + (ρ1-σ1)*log(c1) - (ρ2-σ2)*log(c2)
                rhs += log((1-ω1)/ω2) + (σ1-1)*log(a1) - (σ2-1)*log(a2)

                return lλ_val - rhs
            end

            # Bisect over possible values of a1 and get corresponding b2
            a1_vals[i_λ, i_zh] = brent(a1b2_resid, 1e-6, zh-1e-6)
            b2_vals[i_λ, i_zh] = eval_b2(m, a1_vals[i_λ, i_zh], lzh_val)
        end
    end

    # Build linear interpolants for a1 and b2
    a1_itp = scale(interpolate(a1_vals, BSpline(Linear()), OnGrid()), lλ☆_fg, lzh_fg)
    b2_itp = scale(interpolate(b2_vals, BSpline(Linear()), OnGrid()), lλ☆_fg, lzh_fg)

    return a1_itp, b2_itp
end

"""
Gives us the omega that corresponds with a changed sigma

Takes a value for σ (and optionally the share of imports, s) and returns a
value for omega that should keep the import share to be about the
specified level.
"""
function given_sigma_return_omega(σ::Float64; s::Float64=0.1)
    ratio = ((1-s)/s)^(1-σ)
    ω = 1 / (1 + ratio)
    return ω
end


"""
Takes consumptions and certainty equivalents and evaluates the log of
the value function for those values
"""
function eval_vfs(m::AbstractModel, c1, c2, μ1, μ2)
    lj_new = eval_lj(m, c1, μ1)
    lu_new = eval_lu(m, c2, μ2)
    return lj_new, lu_new
end

"""
Takes two vectors of certainty equivalents (one element for each state
on the grid) and evaluates the value functions using the consumption
values on the grid that are stored in the model
"""
eval_vfs(m::AbstractModel, μ1, μ2) =
    eval_vfs(m, m.other.c1, m.other.c2, μ1, μ2)


"""
Takes value functions at points on the grid and returns a tuple of
interpolants that can evaluate the value function
"""
function update_vf_itps(m::AbstractModel, lj, lu; itptype=Linear())
    Nss = map(length, m.m_ls)
    lj_rs = reshape(lj, Nss...)
    lu_rs = reshape(lu, Nss...)
    lj_itp = scale(interpolate(lj_rs, BSpline(itptype), OnGrid()), m.m_ls...)
    lu_itp = scale(interpolate(lu_rs, BSpline(itptype), OnGrid()), m.m_ls...)
    itps = (lj_itp, lu_itp)
end

# ------------------------------------------------------------------- #
# Solution methods
# ------------------------------------------------------------------- #
"""
Outputs the implied residual when we give it an lk and a lλ☆p

The lowest level of our solution method. It takes a state today (i), a
state tomorrow (j), a value for ratio of certainty equivalents (lk), and
a value for the ratio of Pareto weights tomorrow (lλ☆). It uses these to
evaluate the residual of the foc w.r.t. U_{t+1}.

## Parameters

- `m::AbstractModel` : The model
- `i::Int` : The current period t state we are solving for on grid
- `j::Int` : The current period t+1 we are evaluating residual at
- `vf_itps` : A tuple of interpolants that can evaluate value functions
- `lk::Float64` : Value for log(μ2^(ρ2-α2) / μ1^(ρ1-α1))
- `lλ☆p::Float64` : Value for Pareto weight ratio tomorrow

## Returns

- `::Float64` : The residual of foc when lk and lλ☆p take the given values
"""
function solve_given_lλ☆p(m::AbstractModel, i::Int, j::Int, vf_itps::VFITP,
                          lk::Float64, lλ☆p::Float64)

    # Unpack parameters
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)

    # Get period t state
    lλ☆ = m.grid[i, 1]

    # t+1 exogenous states
    exogp = get_exog_tomorrow(m, i, j)
    lgp = exogp[end]

    # Evaluate value function at t+1
    ljp = vf_itps[1][lλ☆p, exogp[1:end-1]...]
    lup = vf_itps[2][lλ☆p, exogp[1:end-1]...]

    return up_residual(m, ljp, lup, lλ☆p, lgp, lλ☆, lk)
end


"""
Solves for lλ☆ given a specific lk and t+1 value functions

We solve for the policy rules, lλ☆p, as if we already knew what the
value for Lars' k (`lk`) was. We bisect over the `solve_given_lλ☆p`
function in order to find the correct state by state policy choice.
"""
function solve_given_lk(m::AbstractModel, i::Int, vf_itps::VFITP, lk::Float64)

    # Unpack parameters
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)
    Nϵ, Nshocks = size(m.ϵ)

    # Allocate space
    ljp, lup, lλ☆p = map(n->Array(Float64, n), fill(Nϵ, 3))

    # Need to decide how far to go out for bisection. We can't go too
    # far or we move to very inaccurate regions because of extrapolation.
    # The values below seem to do well enough.
    lλ_step = max(0.15, m.m_ls[1][end] / 3)
    lλ☆_big, lλ☆_small = m.grid[i, 1] + lλ_step, m.grid[i, 1] - lλ_step

    # Get relevant state
    exogp = get_exog_tomorrow(m, i)
    lgp_i = exogp[end]
    gp_i = exp(lgp_i)

    # Iterate over every possible state. Create an object that allows
    # us to know whether bisection failed at any state
    it_warn = fill(false, Nϵ)
    for j=1:Nϵ

        vals = map(x->solve_given_lλ☆p(m, i, j, vf_itps, lk, x),
                   (lλ☆_small, lλ☆_big))

        # Check if bisection will work. If so, bisect
        if prod(vals) .<= 0
            it_warn[j] = false
            lλ☆p[j] = brent(x->solve_given_lλ☆p(m, i, j, vf_itps, lk, x),
                            lλ☆_small, lλ☆_big)

        # If not, then choose a bogus policy and change state to a warning
        else
            it_warn[j] = true
            lλ☆p[j] = m.grid[i, 1]
        end

        # Fill in values for value function
        curr_exogp_state = map(x->x[j], exogp[1:end-1])
        ljp[j] = vf_itps[1][lλ☆p[j], curr_exogp_state...]
        lup[j] = vf_itps[2][lλ☆p[j], curr_exogp_state...]
    end

    if any(it_warn)
        warn("Brent warning $(sum(it_warn)) at $i")
    end

    # Get certainty equivalents
    lμ1 = eval_lμ1(m, gp_i, exp(ljp))
    lμ2 = eval_lμ2(m, gp_i, exp(lup))

    return lk_residual(m, lμ1, lμ2, lk), lλ☆p, lμ1, lμ2
end

"""
Updates value functions using a specified value function

We want to be able to use Howard Policy Iteration so we must be able to
update value functions for a given policy. This method updates for a
specific state `i` and manages to skip any expensive bisection and
root-finding which makes it significantly faster than calling
`solve_recursiv_case`. We return the Lars k and two certainty
equivalents
"""
function solve_given_policy(m::AbstractModel, i::Int, lλ☆p::Vector{Float64},
                            vf_itps::VFITP)
    # Unpack parameters
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)
    Nϵ, Nshocks = size(m.ϵ)

    # Allocate space for t+1 value functions
    ljp, lup = map(n->Array(Float64, n), fill(Nϵ, 2))

    # Get relevant states
    exogp = get_exog_tomorrow(m, i)
    lgp_i = exogp[end]
    gp_i = exp(lgp_i)

    for j=1:Nϵ
        curr_exogp_state = map(x->x[j], exogp[1:end-1])
        ljp[j] = vf_itps[1][lλ☆p[j], curr_exogp_state...]
        lup[j] = vf_itps[2][lλ☆p[j], curr_exogp_state...]
    end

    # Get certainty equivalents
    lμ1_i = eval_lμ1(m, gp_i, exp(ljp))
    lμ2_i = eval_lμ2(m, gp_i, exp(lup))

    # If we use lk=0.0 then lk_residual gives the value for lk
    lk_i = lk_residual(m, lμ1_i, lμ2_i, 0.0)

    return lk_i, lμ1_i, lμ2_i
end


"""
Finds an optimally given policy and gives necessary output to update
value functions for recursive utility

If an agent has recursive utility then the planner will find it optimal
to give him varying Pareto weights. This function solves for these
policies at a specific state (i) and for given value functions (vf_itps).
It returns Lars log k, updated policy, and certainty equivalents.
"""
function solve_recursiv_case(m::AbstractModel, i::Int, vf_itps::VFITP)

    # Unpack parameters
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)
    Nϵ, Nshocks = size(m.ϵ)

    lk_lb, lk_ub = log_lars_k_bounds(m, i, vf_itps)

    # Use brent to get solution of lk
    lk = brent(x->solve_given_lk(m, i, vf_itps, x)[1], lk_lb, lk_ub)
    # Grab remaining solution items
    _, lλ☆p, lμ1, lμ2 = solve_given_lk(m, i, vf_itps, lk)

    return lk, lλ☆p, lμ1, lμ2
end

"""
Finds an optimally given policy and gives necessary output to update
value functions for additive utility

It turns out that if the agents have additive utility then we are able
to analytically solve for the optimal policy (constant Pareto weights).
This function uses these policies to call the `solve_given_policy`
function and returns Lars log k, the (constant) policies, and the
certainty equivalents
"""
function solve_additive_case(m::AbstractModel, i::Int, vf_itps::VFITP)
    # Unpack parameters
    ρ1, α1, β1, ω1, σ1 = _unpack(m.agent1)
    ρ2, α2, β2, ω2, σ2 = _unpack(m.agent2)
    Nϵ, Nshocks = size(m.ϵ)

    # Know policy function (constant relative Pareto weights)
    lλ☆p = fill(m.grid[i, 1], Nϵ)

    # Can get lk, lμ1, lμ2 from `solve_given_policy`
    lk, lμ1, lμ2 = solve_given_policy(m, i, lλ☆p, vf_itps)

    return lk, lλ☆p, lμ1, lμ2
end

"""
Solves a specified model with various options.

## Parameters

- `m::AbstractModel` : Model we want to solve
- `capT::Int` : Maximum number of periods to recurse backwards
- `tol::Float64` : Stopping tolerance for value functions
- `infile::String` : File with starting guess for value functions
- `outfile::String` : Where to save the value funtion and policies
- `print_skip::Int` : How many periods between each print (or save)
- `mapper::Function` : Function that specifies how to solve the states
    `pmap` is for parallel and `map` is for sequential

## Returns

- `lj::Vector` : Agent 1 value function at each grid point
- `lu::Vecto` : Agent 2 value function at each grid point
- `lλ☆p::Matrix` : Policy at each grid point and possible state tomorrow
- `lμ1::Vector` : Certainty equivalent for agent 1
- `lμ2::Vector` : Certainty equivalent for agent 2
- `t::Int` : Number of backward steps taken while solving
- `err::Float64` : The difference in value functions between t+1 and t.
"""
function solve_model(m::AbstractModel; capT::Int=500, tol::Float64=1e-7,
                     infile="solutions/$(jld_prefix(m))_1.jld",
                     outfile="solutions/$(jld_prefix(m))_1.jld",
                     print_skip::Int=1, mapper::Function=pmap)

    # Decide if the model has time/state additive or recursive preferences
    if (m.agent1.α ≈ m.agent1.ρ) && (m.agent2.α ≈ m.agent2.ρ)
        state_solver = solve_additive_case
    else
        state_solver = solve_recursiv_case
    end

    # make sure the directories for the solutions exist
    for fn in (infile, outfile)
        # if we have a directory prefix of fn and it isn't already a directory,
        # create one
        dir = dirname(fn)
        !isempty(dir) && !isdir(dir) && mkdir(dir)
    end

    # Get sizes
    Nss = map(length, m.m_ls)
    Ngrid, Nstates = size(m.grid)

    # Allocate Space
    lk = zeros(Float64, Ngrid)
    lμ1 = zeros(Float64, Ngrid)
    lμ2 = zeros(Float64, Ngrid)
    lλ☆p = zeros(Float64, length(m.Π), Ngrid)
    lλ☆p_old = zeros(Float64, length(m.Π), Ngrid)
    err_info = Array(Float64, 3, capT)

    # Initialize VF by setting μ = c or reading in old solution
    lj, lu = initialize_vf(m; infile=infile)

    local t, err
    t_start = time()
    itptype = Linear()
    for t in capT:-1:1

        # Update coefficients
        itptype =  t > capT+1 ? Linear() : Quadratic(Line())
        itps = update_vf_itps(m, lj, lu, itptype=itptype)

        # Update the policies once
        solns = mapper(i -> state_solver(m, i, itps), 1:Ngrid)
        for i=1:Ngrid
            lk[i], lλ☆p[:, i], lμ1[i], lμ2[i] = solns[i]
        end

        # Update value functions
        lj_new, lu_new = eval_vfs(m, exp(lμ1), exp(lμ2))

        # Compute error and copy new vfs into the old vfs (Need to change vf here! otherwise is the same as previous)
        err = max(maxabs(lj-lj_new), maxabs(lu-lu_new))
        pol_err = maxabs(lλ☆p - lλ☆p_old)
        copy!(lj, lj_new)
        copy!(lu, lu_new)
        copy!(lλ☆p_old, lλ☆p)
        err_info[:, t] = [t, err, pol_err]

        if mod(t, print_skip) == 0
            msg = @sprintf("%-3i %-2.5e  %-2.5e %-16s  %11.05f\n",
                           t, err, pol_err, "Total Time", time()-t_start)
            print(msg)

            # Get relevant info from model
            m_ls = m.m_ls
            Nss = map(length, m_ls)

            # Want to save objects needed to create an interpolant
            jldopen(outfile, "w") do file
                write(file, "lj", reshape(lj, Nss...))
                write(file, "lu", reshape(lu, Nss...))
                write(file, "llambdastar", lλ☆p)
                write(file, "err_info", err_info)
                write(file, "m_ls", m_ls)
                write(file, "m", m)
            end
        end

        # Iterate for at least hpi_maxit steps or until close enough
        if abs(pol_err) > 1e-15
            hpi_steps = abs(round(Int, min(0.0, log10(pol_err)))) * 2
        else
            hpi_steps = 25
        end

        for i=1:hpi_steps
            # Update vfs from last step
            vf_itps = update_vf_itps(m, lj, lu; itptype=itptype)

            # Use policy to update lk and CE values
            foo = mapper(i -> solve_given_policy(m, i, lλ☆p[:, i], vf_itps), 1:Ngrid)
            for i=1:Ngrid
                lk[i], lμ1[i], lμ2[i] = foo[i]
            end

            # Update value functions
            lj_new, lu_new = eval_vfs(m, exp(lμ1), exp(lμ2))

            # Check the HPI conditions
            copy!(lj, lj_new)
            copy!(lu, lu_new)

        end

        # Check for whether to break or print stuff
        if err < tol
            println(err)
            break
        end

    end

    return lj, lu, lλ☆p, lμ1, lμ2, t, err
end

function run_all_models{MT<:AbstractModel}(::Type{MT},
                                           #           α      ρ     σ
                                           params = [(-1.0,   -2.0,  0.0),
                                                     (-1.0,  -1.0,  0.0),
                                                     (-9.0,  -1.0,  0.0),
                                                     (-49.0, -1.0,  0.0),
                                                     (-9.0,  -0.01, 0.0),
                                                     (-9.0,   1/3,  0.0),
                                                     (-9.0,  -1.0,  -0.5),
                                                     (-9.0,  -1.0,  0.5)]
                                          )

    for (i, (α, ρ, σ)) in enumerate(params)
        ω = given_sigma_return_omega(σ)

        # Create model
        m = MT(α1=α, α2=α, ρ1=ρ, ρ2=ρ, σ1=σ, σ2=σ, ω1=ω, ω2=ω)

        # Create file names
        # infile = outfile  # This sets the infile to the previous solution
        outfile = "../solutions/$(jld_prefix(m))_$(i).jld"
        infile = "../solutions/$(jld_prefix(m))_$(i).jld"

        try
            println("Solving model with α=$α, ρ=$ρ, σ=$σ")
            Exchange.solve_model(m;infile=infile, outfile=outfile, mapper=map)
        catch e
            rethrow(e)
        end

        break

    end
end

end  # Module end

