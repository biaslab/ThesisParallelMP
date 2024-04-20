using RxInfer, BenchmarkTools, LinearAlgebra

# A `PromisedMessage` holds a promise of the message computation
# Use `as_message` function to block and wait for the result.
struct PromisedMessage{T} <: ReactiveMP.AbstractMessage
	task::T
end

mutable struct ChanneledMessage{T} <: ReactiveMP.AbstractMessage
	channel::T
	cached::Any
end

mutable struct ChanneledMessages{T} <: ReactiveMP.AbstractMessage
	channel::T
	cached::Any
end

# Split the mapreduce work over the available threads.
function parallelmapreduce(f, op, x)
	m = length(x)
	if m < 2 || x isa Tuple
		return mapreduce(f, op, x)
	end
	N = min(Threads.nthreads(), div(m, 2))

	len = div(m, N)
	results = Vector{Message}(undef, N)

	@sync for tid in 1:N
		Threads.@spawn begin
			if tid == N
				domain = ((tid - 1) * len + 1):m
			else
				domain = ((tid - 1) * len + 1):(tid * len)
			end
			results[tid] = mapreduce(f, op, view(x, domain))
		end
	end
	reduce(op, results)
end

function parallelreduce(op, x)
	m = length(x)
	if m < 2
		return reduce(op, x)
	end
	N = min(Threads.nthreads(:interactive), div(m, 2))

	len = div(m, N)
	results = Array{eltype(x)}(undef, N)

	@sync for tid in 1:N
		Threads.@spawn :interactive begin
			if tid == N
				domain = ((tid - 1) * len + 1):m
			else
				domain = ((tid - 1) * len + 1):(tid * len)
			end
			results[tid] = reduce(op, view(x, domain))
		end
	end
	reduce(op, results)
end

# The `as_message` for the `PromisedMessage` blocks until available
ReactiveMP.as_message(promised::PromisedMessage) = fetch(promised.task)

function ReactiveMP.as_message(channeled::ChanneledMessage)
	if ismissing(channeled.cached)
		channeled.cached = take!(channeled.channel)
	end
	return channeled.cached
end

function ReactiveMP.as_message(channeled::ChanneledMessages)
	if ismissing(channeled.cached)
		if Base.isready(channeled.channel)
			while Base.isready(channeled.channel)
				channeled.cached = take!(channeled.channel)
			end
		else
			channeled.cached = take!(channeled.channel)
		end
	end
	return channeled.cached
end

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

function fifo_prod(strategy, _, _)
	× = (left, right) -> ReactiveMP.multiply_messages(strategy, left, right)
	return (messages) -> parallelmapreduce(as_message, ×, messages)
end

function ismessageready(message)
	if !(message isa PromisedMessage)
		return true
	end
	return istaskdone(message.task)
end

function check!(ready, messages)
	for i in eachindex(ready)
		if !ready[i]
			ready[i] = ismessageready(messages[i])
		end
	end
end

function frfo_prod(strategy, _, _)
	× = (left, right) -> ReactiveMP.multiply_messages(strategy, left, right)
	return (messages) -> begin
		if messages isa Tuple
			return mapreduce(as_message, ×, messages)
		end

		s = length(messages)
		res = Message(missing, false, false, nothing)
		ready = Vector{Bool}(undef, s)
		used = falses(s)
		check!(ready, messages)

		while sum(used) < s
			if sum(ready) > sum(used)
				res = ×(res, parallelmapreduce(as_message, ×, view(messages, ready .& .!used)))
				used = ready
			end
			check!(ready, messages)
		end

		return res
	end
end

function after_iteration_cb(model, iteration)
	return nothing
end

struct MyCustomProd
    size::Int
end

# Threads pipeline stage uses the `PromisedMessage` to compute messages
# in separate threads
struct ThreadsPipelineStage <: ReactiveMP.AbstractPipelineStage end
struct IThreadsPipelineStage <: ReactiveMP.AbstractPipelineStage end
struct ThreadsReusingPipelineStage{I} <: ReactiveMP.AbstractPipelineStage
	initial::I
end
struct ThreadsSoftReusingPipelineStage{I} <: ReactiveMP.AbstractPipelineStage
	initial::I
end

function ReactiveMP.apply_pipeline_stage(::ThreadsPipelineStage, factornode, tag, stream)
	return stream |> map(eltype(stream), as_promised)
end

function ReactiveMP.apply_pipeline_stage(::IThreadsPipelineStage, factornode, tag, stream)
	return stream |> map(eltype(stream), as_promised_interactive)
end

function ReactiveMP.apply_pipeline_stage(pipeline::ThreadsReusingPipelineStage, factornode, tag, stream)
	channel = Channel{Any}(1)
	put!(channel, Message(pipeline.initial, false, false, nothing))
	callback = (event) -> begin
		Threads.@spawn begin
			put!(channel, as_message(event))
		end
		return ChanneledMessage(channel, missing)
	end
	return stream |> map(eltype(stream), callback)
end

function ReactiveMP.apply_pipeline_stage(pipeline::ThreadsSoftReusingPipelineStage, factornode, tag, stream)
	channel = Channel{Any}(Inf)
	put!(channel, Message(pipeline.initial, false, false, nothing))
	callback = (event) -> begin
		Threads.@spawn begin
			put!(channel, as_message(event))
		end
		return ChanneledMessages(channel, missing)
	end
	return stream |> map(eltype(stream), callback)
end