include(joinpath(@__DIR__, "..", "src", "AlgebraicOptimization.jl"))
using .AlgebraicOptimization
using .AlgebraicOptimization: FinSet, FinFunction, Open, PrimalObjective

# linear cost functions
f(x) = x[1] + 2*x[2]
g(x) = 3*x[1] + x[2]

# Define composition for Open{PrimalObjective}
function compose(f::Open{PrimalObjective}, g::Open{PrimalObjective}, mapping::Dict{Int,Int})
    n_self = length(f.o.decision_space)
    n_g = length(g.o.decision_space)
    self_only = setdiff(1:n_self, collect(keys(mapping)))
    shared = collect(keys(mapping))
    other_only = setdiff(1:n_g, collect(values(mapping)))
    new_len = length(self_only) + length(shared) + length(other_only)
    S = FinSet(new_len)
    function obj(x)
        xs = zeros(Float64, n_self)
        ys = zeros(Float64, n_g)
        # assign self-only variables
        for (i, idx) in enumerate(self_only)
            xs[idx] = x[i]
        end
        # assign shared variables
        for (j, var) in enumerate(shared)
            val = x[length(self_only) + j]
            xs[var] = val
            ys[mapping[var]] = val
        end
        # assign other-only variables
        base = length(self_only) + length(shared)
        for (k, var) in enumerate(other_only)
            ys[var] = x[base + k]
        end
        return f.o(xs) + g.o(ys)
    end
    # identity portmap
    m = FinFunction(collect(1:new_len), new_len)
    return Open{PrimalObjective}(S, obj, m)
end

# build open problems and compose
open_f = Open{PrimalObjective}(FinSet(2), f, FinFunction([1,2], 2))
open_g = Open{PrimalObjective}(FinSet(2), g, FinFunction([1,2], 2))
comp = compose(open_f, open_g, Dict(1=>2))

# input and evaluation
x = [1.0, 2.0, 5.0]
println(comp(x))
