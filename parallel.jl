using RxInfer, BenchmarkTools

# A `PromisedMessage` holds a promise of the message computation
# Use `as_message` function to block and wait for the result.
struct PromisedMessage{T} <: ReactiveMP.AbstractMessage
	task::T
end

# Split the mapreduce work over the available threads.
function parallelmapreduce(f, op, x)
	if length(x) < Threads.nthreads() * 2
		return mapreduce(f, op, x)
	end

	results = Array{eltype(x)}(undef, Threads.nthreads())
	@sync for tid in 1:Threads.nthreads()
		Threads.@spawn :interactive begin
			len = div(length(x), Threads.nthreads())

			if tid == 1:Threads.nthreads()
				len += length(x) % Threads.nthreads()
			end

			domain = ((tid-1)*len +1):tid*len
			results[tid] = mapreduce(f, op, view(x, domain))
		end
	end
	reduce(op, results)
end

function parallelreduce(op, x)
	if length(x) < Threads.nthreads() * 2
		return reduce(op, x)
	end

	results = Array{eltype(x)}(undef, Threads.nthreads())
	@sync for tid in 1:Threads.nthreads()
		Threads.@spawn begin
			len = div(length(x), Threads.nthreads())

			if tid == 1:Threads.nthreads()
				len += length(x) % Threads.nthreads()
			end

			domain = ((tid-1)*len +1):tid*len
			results[tid] = reduce(op, view(x, domain))
		end
	end
	reduce(op, results)
end

# The `as_message` for the `PromisedMessage` blocks until available
ReactiveMP.as_message(promised::PromisedMessage) = fetch(promised.task)
ReactiveMP.as_marginal(promised::PromisedMessage) = as_marginal(fetch(promised.task))
ReactiveMP.setcache!(::ReactiveMP.EqualityLeftOutbound, node::ReactiveMP.EqualityNode, cache::PromisedMessage)  = node.cache_left = as_message(cache)
ReactiveMP.setcache!(::ReactiveMP.EqualityRightOutbound, node::ReactiveMP.EqualityNode, cache::PromisedMessage) = node.cache_right = as_message(cache)

# This function takes an event and casts it to a message
# In a separate thread
function as_promised(event)
	task = Threads.@spawn begin
		as_message(event)
	end
	return PromisedMessage(task)
end

function as_promised_interactive(event)
	task = Threads.@spawn :interactive begin
		as_message(event)
	end
	return PromisedMessage(task)
end

function blocking_parallel_prod(strategy, _, _)
	× = (left, right) -> ReactiveMP.multiply_messages(strategy, left, right)
	return (messages) -> parallelmapreduce(as_message, ×, messages)
end

function promised_parallel_prod(strategy, _, _)
	× = (left, right) -> ReactiveMP.multiply_messages(strategy, left, right)
	return (messages) -> begin
		task = Threads.@spawn :interactive begin
			iparallelmapreduce(as_message, ×, messages)
		end
		return PromisedMessage(task)
	end
end

function done_first_parallel_prod(strategy, _, _)
	× = (left, right) -> ReactiveMP.multiply_messages(strategy, left, right)
	return (messages) -> begin
		if messages isa Tuple
			return parallelmapreduce(as_message, ×, messages)
		end
	 	ispromised = map((msg)->msg isa PromisedMessage, messages)
		variational = view(messages, .!ispromised)
		promised = view(messages, ispromised)
		isdone = map(istaskdone, getfield.(promised, :task))
		done = view(promised, isdone)
		notdone = view(promised, .!isdone)
		first = true
		toreduce = vcat(variational, done)

		if length(toreduce) > 0
			res = parallelmapreduce(as_message, ×, toreduce)
			first = false
		end

		while length(notdone) > 0
			isdone = map(istaskdone, getfield.(notdone, :task))
			done = view(notdone, isdone)
			notdone = view(notdone, .!isdone)
			if first & length(done) > 0
				res = parallelmapreduce(as_message, ×, vcat(variational, done))
				first = false
			else
				if length(done) > 0
					res = vcat(res, parallelmapreduce(as_message, ×, done))
				end
			end
		end

		if res isa Message
			return res
		end

		return parallelreduce(×, res)
	end
end

function after_iteration_cb(model, iteration)
	return nothing
end

# Threads pipeline stage uses the `PromisedMessage` to compute messages
# in separate threads
struct ThreadsPipelineStage <: ReactiveMP.AbstractPipelineStage end
struct IThreadsPipelineStage <: ReactiveMP.AbstractPipelineStage end

function ReactiveMP.apply_pipeline_stage(::ThreadsPipelineStage, factornode, tag, stream)
	return stream |> map(eltype(stream), as_promised)
end

function ReactiveMP.apply_pipeline_stage(::IThreadsPipelineStage, factornode, tag, stream)
	return stream |> map(eltype(stream), as_promised_interactive)
end