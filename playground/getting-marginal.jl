import Pkg; Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate();

using RxInfer, Test

@model function beta_bernoulli(n)
    θ ~ Beta(1, 1)
    y = datavar(Float64, n)
    y .~ Bernoulli(θ)
end

p = 0.5
n = 1_000
y = float.(rand(Bernoulli(p), n))

# So this playground shows 3 different ways to get the marginal posteriors during the execution 
# of the inference procedure 

# The first way subscribes on the posterior marginal updates before the inference
# and unsubscribes when the inference is completed
test_marginals_from_after_inference_callback = []
subscription_reference = Ref{Any}(nothing)

function before_inference(model)
    subscription_reference[] = subscribe!(RxInfer.ReactiveMP.getmarginal(model[:θ], IncludeAll()), (update) -> begin 
        push!(test_marginals_from_after_inference_callback, RxInfer.ReactiveMP.getdata(update))
    end)
end

function after_inference(model)
    if !isnothing(subscription_reference[])
        unsubscribe!(subscription_reference[])
        subscription_reference[] = nothing 
    end
end

# The second way simply listens to the `on_marginal_update` (the preffered way)
test_marginals_from_on_marginal_update_callback = []

function on_marginal_update(model, name, update)
    push!(test_marginals_from_on_marginal_update_callback, RxInfer.ReactiveMP.getdata(update))
end

# The third way fetches the recent update from the `after_iteration` callbacks
# This approach is less preferred
test_marginals_from_after_iteration_callback = []

function after_iteration(model, iteration_index)
    update = RxInfer.Rocket.getrecent(RxInfer.ReactiveMP.getmarginal(model[:θ], IncludeAll()))
    push!(test_marginals_from_after_iteration_callback, RxInfer.ReactiveMP.getdata(update))
end

# The example uses global arrays, so we reset them every time we execute the inference procedure
function before_model_creation()
    
    empty!(test_marginals_from_after_inference_callback)
    empty!(test_marginals_from_on_marginal_update_callback)
    empty!(test_marginals_from_after_iteration_callback)
    
    if !isnothing(subscription_reference[])
        unsubscribe!(subscription_reference[])
        subscription_reference[] = nothing 
    end
end

results = infer(
    model = beta_bernoulli(n),
    data  = (y = y, ),
    iterations = 10,
    callbacks = (
        before_model_creation = before_model_creation,
        before_inference = before_inference,
        after_inference = after_inference,
        on_marginal_update = on_marginal_update,
        after_iteration = after_iteration
    )
)

# The example does not use VI, so all the posteriors are the same 
@show results.posteriors[:θ]

# Test that all 3 approaches give the same results as the inference procedure itself
@test all(test_marginals_from_after_inference_callback .== results.posteriors[:θ])
@test all(test_marginals_from_on_marginal_update_callback .== results.posteriors[:θ])
@test all(test_marginals_from_after_iteration_callback .== results.posteriors[:θ])
