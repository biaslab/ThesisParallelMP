using RxInfer, Distributed, SharedArrays

struct FutureMessage{T} <: ReactiveMP.AbstractMessage
	future::T
end

ReactiveMP.as_message(message::FutureMessage) = fetch(message.future)

function as_promised(event)
	future = @spawnat :any as_message(event)
	return FutureMessage(future)
end

struct DistributedPipelineStage <: ReactiveMP.AbstractPipelineStage end

function ReactiveMP.apply_pipeline_stage(::DistributedPipelineStage, factornode, tag, stream)
	return stream |> map(eltype(stream), as_promised)
end


function parallelmapreduce(f, op, x)
	m = length(x)

	if m < 2 || x isa Tuple
		return mapreduce(f, op, x)
	end

	N = min(length(workers()), div(m, 2))
	len = div(m, N)
	futures = RemoteChannel(()->Channel{Message}(N))

	@distributed for tid in 1:N
        if tid == N
            domain = ((tid - 1) * len + 1):m
        else
            domain = ((tid - 1) * len + 1):(tid * len)
        end
        put!(futures, mapreduce(f, op, view(x, domain)))
	end

    results = Vector{Message}(undef, N)

    for tid in 1:N
        results[tid] = fetch(take!(futures))
    end

	reduce(op, results)
end

function dist_prod(strategy, _, _)
	× = (left, right) -> ReactiveMP.multiply_messages(strategy, left, right)
	return (messages) -> parallelmapreduce(as_message, ×, messages)
end