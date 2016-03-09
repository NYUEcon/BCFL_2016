include("../tatg.jl")
using CompEcon: qnwnorm
using JLD
using KernelDensity
using PyCall
using PyPlot
using StatsBase
@pyimport seaborn as sb


# ------------------------------------------------------------------- #
# File Helper Functions
# ------------------------------------------------------------------- #
"Reads lj, lu, lλ☆p, m_ls from a specific file"
function read_solutions(filename::AbstractString)
    file = jldopen(filename, "r")
    lj, lu, lλ☆p, m_ls =
        map(var->read(file, var), ["lj", "lu", "llambdastar", "m_ls"]);
    close(file)

    return lj, lu, lλ☆p, m_ls
end

"""
Takes model parameters and a filename and returns a model with that has
stochastic volatility with the correct parmeters and all relevant
solution output
"""
function build_modelSV_interpolants(α::Float64, ρ::Float64, σ::Float64;
                                    file="solutions/curr_soln_sv_9.jld")
    # Get solutions
    lj, lu, lλ☆p, m_ls = read_solutions(file)

    # Get inputs to model creation
    nλ☆ = length(m_ls[1])
    nlzh = length(m_ls[2])
    nv = length(m_ls[3])
    lλ☆bds = m_ls[1][end]
    neps = Int(size(lλ☆p, 1)^(1/3))

    # Create model
    m = Exchange.ModelSV(α1=α, α2=α, lλ☆bds=lλ☆bds,
                         nλ☆=nλ☆, nlzh=nlzh, nv=nv,
                         nϵ1=neps, nϵ2=neps, nϵ3=neps)

    # Get interpolants
    vf_itp = Exchange.update_vf_itps(m, lj, lu);
    lλ☆p_itp = Exchange.get_lλ☆p_itp(m, lλ☆p);

    return m, vf_itp, lλ☆p_itp, lj, lu, lλ☆p
end

"""
Takes model parameters and a filename and returns a model with that has
constant volatility with the correct parmeters and all relevant
solution output
"""
function build_modelCV_interpolants(α::Float64, ρ::Float64, σ::Float64;
                                    filename="solutions/curr_soln_cv_9.jld")
    # Get solutions
    lj, lu, lλ☆p, m_ls = read_solutions(filename)

    # Get inputs to model creation
    nλ☆ = length(m_ls[1])
    nlzh = length(m_ls[2])
    lλ☆bds = m_ls[1][end]
    neps = Int(size(lλ☆p, 1)^(1/2))

    # Create model
    m = Exchange.ModelCV(α1=α, α2=α, lλ☆bds=lλ☆bds,
                         nλ☆=nλ☆, nlzh=nlzh,
                         nϵ1=neps, nϵ2=neps)

    # Get interpolants
    vf_itp = Exchange.update_vf_itps(m, lj, lu);
    lλ☆p_itp = Exchange.get_lλ☆p_itp(m, lλ☆p);

    return m, vf_itp, lλ☆p_itp, lj, lu, lλ☆p
end

# ------------------------------------------------------------------- #
# General Helper Functions
# ------------------------------------------------------------------- #
"""
Simulates for a long time with no shocks in order to go towards the
steady state
"""
function find_steadystate_lλ(m::Exchange.AbstractModel,
                             lλ☆p_itp::Exchange.AbstractInterpolation;
                             capT=2_000_000)

    return Exchange.simulate(m, lλ☆p_itp, zeros(3, capT))[7][end]
end

"""
Takes the ratio of consumptions, an exchange rate, and an alpha and returns
slope and correlations that are implied
"""
function get_slope_corr(c2_c1_ratio, fxr, alpha, npers=5000)
    x = log(c2_c1_ratio[1:npers])
    y = log(fxr[1:npers])
    slope = x \ y
    corrr = round(cor(log(c2_c1_ratio[1:npers]), log(fxr[1:npers])), 2)

    return slope, corrr
end

# ------------------------------------------------------------------- #
# Pareto Frontiers
# ------------------------------------------------------------------- #
"""
Plots the Pareto frontiers for both consumption and value functions
"""
function plot_c_vf_frontiers!(lj, lu, c1, c2, ax)

    # Set colors and line style
    colors = ("#852C64", "#AA6039")
    ls = ("-", "-.")

    # Get how many of the exogenous states are and find mean
    i = map(x->div(x, 2)+1, size(lj)[2:end])

    ax[:plot](exp(lu[:, i...]), exp(lj[:, i...]), color=colors[1], ls=ls[1])
    ax[:plot](c2[:, i...], c1[:, i...], color=colors[2], ls=ls[2])

    ax[:set_ylabel]("Agent 1")
    ax[:set_xlabel]("Agent 2")
    return ax
end

"""
Plots the frontier for value functions only
"""
function plot_vf_frontiers!(lj, lu, c, ax)

    ax[:plot](exp(lu), exp(lj), color=c)
    ax[:set_xlabel](L"log U")
    ax[:set_ylabel](L"log J")

    return ax
end


# ------------------------------------------------------------------- #
# Consumption v Exchange Rate Functions
# ------------------------------------------------------------------- #
"""
Takes a vectors of `c2_c1_ratio` and a vector of `fxr` and plots them
on a scatter plot.
"""
function plot_c2c1ratio_fxr_single!(alpha, c2_c1_ratio, fxr, npers, ax)
    # Get slope and correlation
    s, c = get_slope_corr(c2_c1_ratio, fxr, alpha, npers);

    palpha = npers < 10000 ? .75 : npers < 25000 ? .5 : .25

    ax[:scatter](log(c2_c1_ratio[1:npers]), log(fxr[1:npers]);
                 s=0.75, alpha=1.00, label="Correlation is $c", plotkwargs[alpha]...)

    return ax
end



# ------------------------------------------------------------------- #
# Pareto Weight Plots
# ------------------------------------------------------------------- #
"""
Plots the history of Pareto weights. Has option of plotting a random
walk of the same standard deviation next to history.
"""
function plot_pareto_history!(alpha, lλ☆, ax; rw=false, T::Int=1_000_000,
                              stsz::Int=10000, c="red")

    ax[:plot](collect(1:stsz:T), lλ☆[1:stsz:T]; color=c)

    if rw
        srand(4251989)
        lλ☆_std = std(lλ☆[1:end-1] - lλ☆[2:end])
        ranwalk = cumsum(randn(T) .* lλ☆_std)
        ax[:plot](collect(1:stsz:T), ranwalk[1:stsz:T]; color="k")
    end

    ax[:set_ylim](-3.75, 3.75)

    return ax
end

# ------------------------------------------------------------------- #
# Exchange Rate
# ------------------------------------------------------------------- #
"""
Plots the autocorrelation function for a specified number of periods
"""
function acorr_plot_one_ax!(data, α, ax, k_max=100, nshow=30; c="red")
    ks = collect(1:k_max)
    acf = autocor(data, ks)
    lab = string(L"$\rho^k$ ($\rho=", round(acf[1], 4), L")$")
    ax[:plot](ks, acf; label=lab, color=c)
    ax
end

# ------------------------------------------------------------------- #
# Impulse Response Function
# ------------------------------------------------------------------- #
"""
Plots linear impulse response functions. In particular, given a history
of shocks, ϵ, it plots the response of (zhat, v, c1, c2, lλ☆, e) in %
deviations
"""
function irf_plotter(m::Exchange.AbstractModel,
                     lλ☆p_itp::Exchange.Interpolations.AbstractInterpolation,
                     ϵ::Array{Float64, 2};
                     capT=50, lλ0=0.0, c="red")

    # Get irf data
    simuls = Exchange.simulate(m, lλ☆p_itp, ϵ; lλ0=lλ0)
    a1, b1, c1, a2, b2, c2, lλ☆, exogvals, c2_c1_ratio, fxr = simuls


    # Create a t array
    irf_per = collect(1:capT)
    fig_irf, ax_irf = subplots(3, 2)
    map(x->x[:locator_params](nbins=4), ax_irf)

    # Put zhat in log deviations and plot
    ld_zhat = 100*(exogvals[1, 1:end-1]' - exogvals[1, 1])
    ax_irf[1, 1][:plot](irf_per, ld_zhat; color="k")
    ax_irf[1, 1][:set_title]("Productivity")

    # Put v in log deviations and plot
    ld_v = 100*(log(exogvals[2, 1:end-1]') - log(exogvals[2, 1]))
    ax_irf[1, 2][:plot](irf_per, ld_v, color="k")
    ax_irf[1, 2][:set_title]("Volatility in Country 1")

    # Put c1 in log deviations and plot
    ld_c1 = 100*(log(c1) - log(c1[1]))
    ax_irf[2, 1][:plot](irf_per, ld_c1; color=c)
    ax_irf[2, 1][:set_title]("Consumption: Country 1")

    # Put c2 in log deviations and plot
    ld_c2 = 100*(log(c2) - log(c2[1]))
    ax_irf[2, 2][:plot](irf_per, ld_c2; color=c)
    ax_irf[2, 2][:set_title]("Consumption: Country 2")

    # Put P.W. ratio in log deviations and plot
    ld_λ☆ = 100*(lλ☆ - lλ☆[1])
    ax_irf[3, 1][:plot](irf_per, ld_λ☆; color=c)
    ax_irf[3, 1][:set_title]("Pareto Weight Ratio")
    ax_irf[3, 1][:set_xlabel]("Time (Quarters)")

    # Put exchange rate in log deviations and plot
    ld_fxr = 100*(log(fxr) - log(fxr[1]))
    ax_irf[3, 2][:plot](irf_per, ld_fxr; color=c)
    ax_irf[3, 2][:set_title]("Exchange Rate")
    ax_irf[3, 2][:set_xlabel]("Time (Quarters)")

    fig_irf[:tight_layout]()
    # fig_irf[:savefig]("images/irf_wrt_v.png", dpi=400)
    fig_irf[:show]()

    return fig_irf, ax_irf
end

# ------------------------------------------------------------------- #
# Policies
# ------------------------------------------------------------------- #
"""
Plots the policy function over the possible values for vt. All other
variables are held at their ss.
"""
function plot_policy_wrt_v!(m, itp, ax=nothing; c="red")
    lλ☆t = 0.0
    lzh = 0.0
    v = linspace(extrema(m.grid[:, 3])..., 50)

    lλ☆p = Array(Float64, 50)

    for i=1:50
        lλ☆p[i] = itp[lλ☆t, lzh, v[i], zeros(3)...]
    end

    if ax === nothing
        fig, ax = subplots()
    end

    ax[:set_ylabel](L"$\log \lambda_{t+1}^*$")
    ax[:set_xlabel](L"$v_{t}$")
    ax[:plot](v, lλ☆p; color=c)
    ax
end

"""
Plots the expected change in the Pareto weight across various values of
lλ☆. All other variables are held at their steady states.
"""
function plot_policy_at_ss!(m, itp, ss, ax=nothing; c="red")

    lλ☆t = linspace(ss-3.5, ss+3.5, 50)
    lzh = 0.0
    v = m.exog.vbar

    ϵ, Π = Exchange.qnwnorm([7, 7, 7], zeros(3), eye(3))
    nshocks = length(Π)
    lλ☆p = Array(Float64, nshocks, 50)

    for i=1:50
        for j=1:nshocks
            lλ☆p[j, i] = itp[lλ☆t[i], lzh, v, ϵ[j, :]...]
        end
    end

    Elλ☆p = vec(Π' * lλ☆p)
    ΔElλ☆p = Elλ☆p - collect(lλ☆t)

    if ax === nothing
        fig, ax = subplots()
    end

    ax[:set_ylabel](L"$E[\log \lambda_{t+1}^*] - \log \lambda_{t}^*$")
    ax[:set_xlabel](L"$\log \lambda_{t}^*$")
    ax[:plot](lλ☆t, ΔElλ☆p; color=c)
    ax
end

