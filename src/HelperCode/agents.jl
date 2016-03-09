"""
This is the main type which holds agent specific parameters.

In particular,

* (ρ, α, β) are preference parameters for EZ preferences
* (ω, σ) are parameters used in consumption aggregation over goods
"""
immutable Agent
    ρ::Float64
    α::Float64
    β::Float64
    ω::Float64
    σ::Float64
end

_unpack(a::Agent) = (a.ρ, a.α, a.β, a.ω, a.σ)

"""
Takes the two goods (x, y) and combines them according to the agents consumption
aggregator (CES)
"""
function _agg_c(a::Agent, x, y)

    # If σ = 0 then use Cobb-Douglas to aggregate goods
    if abs(a.σ) < 1e-14
        return x.^(1-a.ω) .* y.^(a.ω)
    # Otherwise use the general CES aggregator
    else
        return ((1-a.ω)*x.^(a.σ) + a.ω*y.^(a.σ)).^(1/a.σ)
    end
end

"""
Takes consumption and the certainty equivalent of the value functions next
period and returns this period's log value function
"""
_agg_lv(a::Agent, c, μ) = (ρ=a.ρ; (1/ρ) .* log((1-a.β)*c.^ρ + a.β*μ.^ρ))

"""
Takes a vector of growth rates, a vector of period t+1 value functions and
computes the certainty equivalent for an agent
"""
_CE(a::Agent, gp::Vector, vp::Vector, Π::Vector) =
    dot(Π, (gp.*vp).^(a.α)).^(1.0/a.α)

