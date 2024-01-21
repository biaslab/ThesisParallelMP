using RxInfer

# A `PromisedMessage` holds a promise of the message computation
# Use `as_message` function to block and wait for the result.
struct PromisedMessage{T} <: ReactiveMP.AbstractMessage
	task::T
end

# The `as_message` for the `PromisedMessage` blocks until available
ReactiveMP.as_message(promised::PromisedMessage) = fetch(promised.task)

# This function takes an event and casts it to a message
# In a separate thread
function as_promised(event)
	task = Threads.@spawn begin
		as_message(event)
	end
	return PromisedMessage(task)
end

function custom_prod(strategy, _, _)
	× = (left, right) -> ReactiveMP.multiply_messages(strategy, left, right)
	return (messages) -> mapreduce(as_message, ×, messages)
end

function after_iteration_cb(model, iteration)
	return nothing
end

# Threads pipeline stage uses the `PromisedMessage` to compute messages
# in separate threads
struct ThreadsPipelineStage <: ReactiveMP.AbstractPipelineStage end

function ReactiveMP.apply_pipeline_stage(::ThreadsPipelineStage, factornode, tag, stream)
	return stream |> map(eltype(stream), as_promised)
end