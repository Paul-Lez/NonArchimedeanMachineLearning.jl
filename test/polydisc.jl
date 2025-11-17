# Test file for basic polydisc operations.
#
# This file demonstrates and tests the fundamental polydisc operations
# including creation, iteration, joining, and generating children.

include("../src/basic/polydisc.jl")

prec = 20
K = PadicField(3, prec)

a1 = [K(1), K(2)]
r1 = [1, 2]
a2 = [K(2), K(2)]
r2 = [2, 2]

p1 = ValuationPolydisc(a1, r1)
p2 = ValuationPolydisc(a2, r2)

# Test: Iterate over polydisc indices
for i in Base.eachindex(p1)
    println("i-th center is ", p1.center[i], " and i-th radius is ", p1.radius[i])
end

# Test: Join (smallest common ancestor) of two polydiscs
println("The join of p1 and p2 is ", join(p1, p2))



# TODO: write more tests for the children function

# Test: Generate children of a polydisc
L = PadicField(2, prec)
gauss = ValuationPolydisc([L(0)], [2])
@show children(gauss)




