include("../src/statistics/frechet.jl")

prec = 20
K = PadicField(3,prec)

p1 = ValuationPolydisc([K(1)], [0])
p2 = ValuationPolydisc([K(2)], [0])
p3 = ValuationPolydisc([K(29)], [0])

frechet_mean([[K(1), K(2)], [K(2), K(5)], [K(29), K(32)]])
frechet_mean([p1, p2, p3], 20)