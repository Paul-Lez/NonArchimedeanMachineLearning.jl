"""
Generic valuation, absolute-value, and unit interfaces for p-adic-style scalar
types used throughout the package.
"""

@doc raw"""
    valuation(a::padic)

Compute the p-adic valuation of a p-adic number.

For a p-adic number ``a = p^v \cdot u`` where ``u`` is a p-adic unit,
returns the integer ``v``.

# Arguments
- `a::padic`: A p-adic number

# Returns
The p-adic valuation ``v(a)``

# Note
This wraps the Nemo/Oscar implementation. To support custom p-adic
implementations, define a method `valuation(a::YourType)` for your type.
"""
function valuation(a::padic)
    return Nemo.valuation(a)
end

@doc raw"""
    Base.abs(a::padic)

Compute the p-adic absolute value of a p-adic number.

For a p-adic number ``a`` with valuation ``v(a)``, returns ``p^{-v(a)}``.

# Arguments
- `a::padic`: A p-adic number

# Returns
`Float64`: The p-adic absolute value ``|a|_p = p^{-v(a)}``

# Note
This extends `Base.abs` for `padic` types, making the codebase generic.
To support custom p-adic implementations, define `Base.abs` for your type.
"""
function Base.abs(a::padic)
    v = valuation(a)
    return Float64(Nemo.prime(a.parent))^(-v)
end

@doc raw"""
    unit(a::padic)

Extract the unit part of a p-adic number.

For a p-adic number ``a = p^v \cdot u`` where ``u`` is a p-adic unit,
returns the unit ``u``.

# Arguments
- `a::padic`: A p-adic number

# Returns
The unit part ``u`` (accessed via `a.u` for `PadicFieldElem`)

# Note
This provides a generic interface to the unit part of p-adic numbers.
For custom p-adic implementations, define `unit(a::YourType)` for your type.
"""
function unit(a::padic)
    return a.u
end
