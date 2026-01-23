# TODO: implement various operations on abstract polydisc functions and re-implement this using these operations

# "Naive" loss function, using absolute polynomials to try and capture the distance between nodes. This does not work well!
function init_rational_loss(D::Matrix{Bool}, K)
    # TODO: do we also want some kind of method for sparse evaluation?
    # For now type is hard-coded. Should be changed later on...
    linear_polynomials = Matrix{LinearPolynomial{PadicFieldElem}}(undef, size(D, 1), size(D, 2)) #([], K(0))
    for i in axes(D, 1)
        for j in axes(D, 2)
            linear_polynomials[i, j] = LinearPolynomial([
                    u == i ? K(1) : K(0) for u in axes(D, 1)
                ] - [u == j ? K(1) : K(0) for u in axes(D, 1)], K(0))
        end
    end
    linear_polynomials = map(batch_evaluate_init, linear_polynomials)

    function eval_matrix(params::ValuationPolydisc{S,T})::Matrix{Float64} where S where T
        return map(f -> f(params), linear_polynomials)
    end

    # This can be vectorised!
    function eval(params::ValuationPolydisc{S,T})::Float64 where S where T
        pre_computed = eval_matrix(params)
        result = Float64(0)
        for i in axes(D, 1)
            den = Float64(0)
            num = Float64(0)
            for j in axes(D, 2)
                if !D[i, j]
                    den += pre_computed[i, j]
                else
                    num += pre_computed[i, j]
                end
            end
            result += num / den
        end
        return result
    end
    return Loss(p -> map(eval, p), p -> 0)
end

# TODO: we may want to add a term to penalise lack of injectivity in the embedding! But maybe this is just something to be solved by initialising in a
# smarter way?

# The loss function used in the Meta paper.
function init_distance_loss(D::Matrix{Bool}, K::Ring)
    # Quick and dirty implementation for now: we'll vectorise and optimise later.

    function eval(params::ValuationPolydisc{S,T})::Float64 where S where T
        result = Float64(0)
        # TODO: implement iterate for polydiscs
        discs = components(params)
        for i in axes(D, 1)
            den = Float64(0)
            num = Float64(0)
            for j in axes(D, 2)
                if !D[i, j]
                    den += exp(-dist(discs[i], discs[j]))
                else
                    num += exp(-dist(discs[i], discs[j]))
                end
            end
            if num == 0
                continue
            else
                result += log(num / den)
            end
        end
        return result
    end
    return Loss(p -> map(eval, p), p -> 0)
end

function mkComponentwiseDistance(i::Int, j::Int, K::Ring)
    return Lambda{PadicFieldElem}(p -> dist(components(p)[i], components(p)[j]))
end