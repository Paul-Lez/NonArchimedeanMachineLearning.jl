include("src/NAML.jl")
using .NAML
using Oscar

K = PadicField(2, 20)
R, (x, a) = K["x", "a"]
g = AbsolutePolynomialSum([x - a])
f = AbstractModel(g, [true, false])
p1 = ValuationPolydisc([K(5)], [20])
data = [(p1, 0)]
ell = MPE_loss_init(f, data, 2)
param = ValuationPolydisc([K(11)], [0])

println("Param type: ", typeof(param))
println("Param in vector type: ", typeof([param]))

try
    result = ell.eval([param])
    println("Success! Result: ", result)
catch e
    println("Error: ", e)
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
