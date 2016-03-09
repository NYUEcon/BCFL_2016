
include("HelperCode/plot_helpers.jl")
sb.set_style("white")

# # Create alpha we work with
# α1 = -1.0
# α9 = -9.0
# α49 = -49.0

# # Create all the models and interps we care about
# m1, vf_itp1, lλ☆p_itp1, lj1, lu1, lλ☆p1 =
#     build_model_interpolants(α1);
# m9, vf_itp9, lλ☆p_itp9, lj9, lu9, lλ☆p9 =
#     build_model_interpolants(α9);
# m49, vf_itp49, lλ☆p_itp49, lj49, lu49, lλ☆p49 =
#     build_model_interpolants(α49);

# # Find all steady state lambdas
# lλ1_ss = 0.0
# lλ9_ss = find_steadystate_lλ(m9, lλ☆p_itp9; capT=6_000_000)
# lλ49_ss = find_steadystate_lλ(m49, lλ☆p_itp49; capT=6_000_000)

# # Simulate all economies
# srand(61089)
# sim_1 = Exchange.simulate(m1, lλ☆p_itp1; capT=500_000, lλ0=lλ1_ss);
# a1_1, b1_1, c1_1, a2_1, b2_1, c2_1, lλ☆_1, exogvals_1, c2_c1_ratio_1, fxr_1 = sim_1;

# srand(61089)
# sim_9 = Exchange.simulate(m9, lλ☆p_itp9; capT=500_000, lλ0=lλ9_ss);
# a1_9, b1_9, c1_9, a2_9, b2_9, c2_9, lλ☆_9, exogvals_9, c2_c1_ratio_9, fxr_9 = sim_9;

# srand(61089)
# sim_49 = Exchange.simulate(m49, lλ☆p_itp49; capT=500_000, lλ0=lλ49_ss);
# a1_49, b1_49, c1_49, a2_49, b2_49, c2_49, lλ☆_49, exogvals_49, c2_c1_ratio_49, fxr_49 = sim_49;

# @save "plot_data.jld"

@load "plot_data.jld"

get_c1_c2(m) = map(x-> reshape(x, map(length, m.m_ls)), (m.other.c1, m.other.c2))
(c₁1, c₂1), (c₁9, c₂9), (c₁49, c₂49) = map(get_c1_c2, (m1, m9, m49));

function plot_frontiers(lj, lu, c1, c2, α, inds=[7, 10], var=:zhat)
    fig, ax = subplots()
    sb.despine(fig)
    colors = ("#852C64", "#AA6039")
    ls = ("-", "-.")

    # plot baseline
    i = var == :zhat ? (inds[1], 7) : (7, inds[1])
    ax[:plot](exp(getindex(lu, :, i...)), exp(getindex(lj, :, i...)), color=colors[1], ls=ls[1])
    ax[:plot](getindex(c2, :, i...,), getindex(c1, :, i...), color=colors[2], ls=ls[1])

    # plot experiment
    i = var == :zhat ? (inds[2], 7) : (7, inds[2])
    ax[:plot](exp(getindex(lu, :, i...)), exp(getindex(lj, :, i...)), color=colors[1], ls=ls[2], lw=3)
    ax[:plot](getindex(c2, :, i...,), getindex(c1, :, i...), color=colors[2], ls=ls[2], lw=3)

    ax[:set_ylabel]("Agent 1")
    ax[:set_xlabel]("Agent 2")
    ax[:legend](["Value Function", "Consumption"], loc=0)
    fig[:savefig]("images/frontiers_alpha$(α).pdf", dpi=1000)
    fig[:show]()

    fig, ax
end

plot_frontiers(lj9, lu9, c₁9, c₂9, "9_zhat", [length(m9.m_ls[2]), 1])
plot_frontiers(lj9, lu9, c₁9, c₂9, "9_v", [1, length(m9.m_ls[3])], :v)

fig, ax = subplots()
sb.despine(fig)
ax[:plot](exp(lu9[:, 7, 7]), exp(lj9[:, 7, 7]), label=L"v")
ax[:plot](c₁9[:, 7, 7], c₂9[:, 7, 7], label=L"c")

ax[:set_ylabel]("Agent 1")
ax[:set_xlabel]("Agent 2")

ax[:legend](loc=0)

fig[:savefig]("images/frontiers_alpha9.pdf", dpi=1000)
fig[:show]()

fig, ax = subplots()
sb.despine(fig)
ax[:plot](exp(lu49[:, 7, 7]), exp(lj49[:, 7, 7]), label=L"v")
ax[:plot](c₁49[:, 7, 7], c₂49[:, 7, 7], label=L"c")

ax[:set_ylabel]("Agent 1")
ax[:set_xlabel]("Agent 2")

ax[:legend](loc=0)

fig[:savefig]("images/frontiers_alpha49.pdf", dpi=1000)
fig[:show]()

fig, ax = subplots()
sb.despine(fig)
ax[:plot](exp(lu1[:, 7, 7]), exp(lj1[:, 7, 7]), label=L"$\alpha = -1.0$")
# ax[:plot](exp(lu9[:, 7, 7]), exp(lj9[:, 7, 7]), label=L"$\alpha = -9.0$")
ax[:plot](exp(lu49[:, 7, 7]), exp(lj49[:, 7, 7]), label=L"$\alpha = -49.0$")

ax[:set_xlabel](L"$U$")
ax[:set_ylabel](L"$J$")

ax[:legend](loc=0)

fig[:savefig]("images/ValueFunctions_alpha49.pdf", dpi=1000)
fig[:show]()

fig, ax = subplots()
sb.despine(fig)

fig[:suptitle]("Consumption vs Exchange Rate")
ax[:set_ylabel]("Log Exchange Rate")
ax[:set_xlabel]("Log Consumption Ratio")
# Note: We could also make this into multiple subplots and have
# a round robin type comparsion (1v9), (1v49), (9v49)...
plotslice = 3000:15000
plot_c2c1ratio_fxr_single!(α9, c2_c1_ratio_9[plotslice], fxr_9[plotslice], 2500, ax)
# plot_c2c1ratio_fxr_single!(α49, c2_c1_ratio_49[plotslice], fxr_49[plotslice], 2500, ax)
plot_c2c1ratio_fxr_single!(α1, c2_c1_ratio_1[plotslice], fxr_1[plotslice], 2500, ax)
ax[:legend](markerscale=5, framealpha=1.0, loc=2)

fig[:savefig]("images/con_v_fxr_alpha9.pdf", dpi=1000)
fig[:show]()

T = 500_000
st_sz = 1000

fig, ax = plt[:subplots]()
sb.despine(fig)

plot_pareto_history!(α1, lλ☆_1, ax; T=T, stsz=st_sz)
plot_pareto_history!(α9, lλ☆_9, ax; rw=false, T=T, stsz=st_sz)

# plot_pareto_history!(α1, lλ☆_1, ax[1, 1]; T=T, stsz=st_sz)
# plot_pareto_history!(α9, lλ☆_9, ax[1, 1]; rw=false, T=T, stsz=st_sz)

# plot_pareto_history!(α1, lλ☆_1, ax[2, 1]; T=T, stsz=st_sz)
# plot_pareto_history!(α49, lλ☆_49, ax[2, 1]; rw=false, T=T, stsz=st_sz)

# add x/y labels to figure
# fig[:text](0.0, 0.5, "Ratio of Pareto Weights",
#                ha="center", va="center", rotation="vertical")
# fig[:text](0.5, 0.0, "t", ha="center", va="center")
# fig[:tight_layout]()
ax[:set_xlabel]("Time")
ax[:set_ylabel]("Ratio of Pareto Weights")

fig[:savefig]("images/paretoweightstability_alpha9.pdf", dpi=1000)
fig[:show]()

fig, axs = subplots(2, 1)
sb.despine(fig)
acorr_plot!(fxr_1, -1, axs[1], 100, 50)
acorr_plot!(fxr_9, -9, axs[2], 100, 50)
fig[:text](0.05, 0.5, "Autocorrelation Function",
           ha="center", va="center", rotation="vertical")
axs[2][:set_xlabel]("Time")
# map(ax->ax[:grid](alpha=0.4, lw=0.75), axs)
fig[:savefig]("images/acorr_fx_alpha1_9.pdf", dpi=1000)

fig, ax = subplots()
sb.despine(fig)
acorr_plot_one_ax!(fxr_1, -1, ax, 100, 50)
acorr_plot_one_ax!(fxr_9, -9, ax, 100, 50)
# fig[:text](0.05, 0.5, "Autocorrelation Function",
#            ha="center", va="center", rotation="vertical")
ax[:set_ylabel]("Autocorrelation Function")
ax[:set_xlabel]("Lags")
ax[:set_ylim](-0.05, 1.05)
fig[:savefig]("images/acorr_fx_alpha1_9_one_axis.pdf", dpi=1000)

# kde_plot(lλ☆_9, N=50)

# kde_plot(lλ☆_49, N=10)

# Create a matrix of all zeros except for a one
# time shock to volatility in period 2
T_irf = 50
irf_per = collect(1:T_irf)
vshock = zeros(3, T_irf+1)
vshock[3, 2] = 2.5

# Simulate both economies
fig, ax = irf_plotter(m9, lλ☆p_itp9, vshock; capT=T_irf, lλ0=lλ9_ss)
sb.despine(fig)
map(x->x[:set_xlim](0, T_irf), ax)
map(x->x[:set_xticks](0:10:T_irf), ax)

fig[:savefig]("images/irf_wrt_v_alpha9.pdf", dpi=1000)
fig[:show]()

# Create a matrix of all zeros except for a one
# time shock to volatility in period 2
T_irf = 50
irf_per = collect(1:T_irf)
vshock = zeros(3, T_irf+1)
vshock[1, 2] = 0.5
vshock[2, 2] = -0.5

# Simulate both economies
fig, ax = irf_plotter(m9, lλ☆p_itp9, vshock; capT=T_irf, lλ0=lλ9_ss)
sb.despine(fig)
map(x->x[:set_xlim](0, T_irf), ax)
map(x->x[:set_xticks](0:10:T_irf), ax)

fig[:savefig]("images/irf_wrt_zhat_alpha9.pdf", dpi=1000)
fig[:show]()

# Choose time
T_irf = 100
irf_per = collect(1:T_irf)

# Choose how big v3 should be and
v2 = m49.exog.vbar
v3 = 1.5*m49.exog.vbar
v4 = 1.5*m49.exog.vbar
v5 = m49.exog.vbar
ϵ = zeros(3, T_irf+1)
ϵ[3, 2:10:T_irf] = (v3 - (1-m49.exog.φv)*m49.exog.vbar - m49.exog.φv*v2)/m49.exog.τ
ϵ[3, 3:10:T_irf] = (v5 - (1-m49.exog.φv)*m49.exog.vbar - m49.exog.φv*v4)/m49.exog.τ

fig, ax = irf_plotter(m49, lλ☆p_itp49, ϵ; capT=T_irf, lλ0=lλ49_ss)
sb.despine(fig)
fig[:show]()

# Choose time
T_irf = 100
irf_per = collect(1:T_irf)

# Choose how big v3 should be and
lzh2 = 0.01
lzh3 = 0.0

ϵ = zeros(3, T_irf+1)
ϵ[1, 2:10:T_irf] = lzh2 / (sqrt(m49.exog.vbar)/2 + m49.exog.σz2/2)
ϵ[2, 2:10:T_irf] = -lzh2 / (sqrt(m49.exog.vbar)/2 + m49.exog.σz2/2)
ϵ[1, 3:10:T_irf] = -(1-2*m49.exog.γ)*lzh2 / (sqrt(m49.exog.vbar)/2 + m49.exog.σz2/2)
ϵ[2, 3:10:T_irf] = (1-2*m49.exog.γ)*lzh2 / (sqrt(m49.exog.vbar)/2 + m49.exog.σz2/2)

fig, ax = irf_plotter(m49, lλ☆p_itp49, ϵ; capT=T_irf, lλ0=lλ49_ss)
sb.despine(fig)
fig[:show]()

function plot_policy_wrt_v!(m, itp, ax=nothing)
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

    ax[:set_ylabel](L"$\lambda_{t+1}^*$")
    ax[:set_xlabel](L"$v_{t}$")
    ax[:plot](v, lλ☆p; plotkwargs[m.agent1.α]...)
    ax
end

fig, (ax1, ax2, ax3) = subplots(3, 1)
sb.despine(fig)
m_itp = [(m1, m9, m49), (lλ☆p_itp1, lλ☆p_itp9, lλ☆p_itp49)]
map(plot_policy_wrt_v!, m_itp..., (ax1, ax2, ax3))
fig[:savefig]("images/policy_subplots.pdf", dpi=1000)

fig, ax = subplots()
sb.despine(fig)
map(plot_policy_wrt_v!, m_itp..., (ax, ax, ax))
ax[:set_title]("Policy with resepct to volatility")
fig[:savefig]("images/policy_one_axis.pdf", dpi=1000)

function plot_policy_at_ss!(m, itp, ss, ax=nothing)

    lλ☆t = linspace(ss-2.5, ss+2.5, 50)
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

    ax[:set_ylabel](L"$E[\lambda_{t+1}^*] - \lambda_{t}^*$")
    ax[:set_xlabel](L"$\lambda_{t}^*$")
    ax[:plot](lλ☆t, ΔElλ☆p; plotkwargs[m.agent1.α]...)
    ax
end

lλ49_ss

# lλ49_ss = find_steadystate_lλ(m49, lλ☆p_itp49; capT=6_000_000)

fig, ax = subplots()
sb.despine(fig)
m_itp2 = [(m1, m9, m49), (lλ☆p_itp1, lλ☆p_itp9, lλ☆p_itp49), (lλ1_ss, lλ9_ss, lλ49_ss)]
map(plot_policy_at_ss!, m_itp2..., (ax, ax, ax))
fig[:savefig]("images/policy_at_ss_one_axis.pdf", dpi=1000)

