## Test file for geodesic interpolation

include("../src/NonArchimedeanMachineLearning.jl")
using .NonArchimedeanMachineLearning
using Oscar

println("=== Testing Geodesic Interpolation ===\n")

# Setup: 2-adic field with precision 20
prec = 20
K = PadicField(2, prec)
p = 2

println("Test 1: Conversion functions (valuation <-> radius)")
println("-" ^ 40)
# Test that conversion functions are inverses
test_vals = [0, 1, 2, 5, 10]
for val in test_vals
    r = valuation_to_radius(val, p)
    val_back = radius_to_valuation(r, p)
    println("  valuation=$val -> radius=$r -> valuation=$val_back")
    @assert abs(val - val_back) < 1e-10 "Conversion should be invertible"
end
println("✓ Conversion functions are inverses\n")

println("Test 2: Endpoint behavior (x=0 and x=1)")
println("-" ^ 40)
# Create two nested discs
d1 = ValuationPolydisc([K(0)], [5])  # Small disc
d2 = ValuationPolydisc([K(0)], [2])  # Large disc (contains d1)

# Test x = 0 (should give d1)
d_0 = geodesic_interpolation(d1, d2, 0.0)
r1_actual = valuation_to_radius(5, p)
println("  x=0: radius=$(d_0.radius[1]), expected=$(r1_actual)")
@assert abs(d_0.radius[1] - r1_actual) < 1e-10 "x=0 should give d1's radius"

# Test x = 1 (should give d2)
d_1 = geodesic_interpolation(d1, d2, 1.0)
r2_actual = valuation_to_radius(2, p)
println("  x=1: radius=$(d_1.radius[1]), expected=$(r2_actual)")
@assert abs(d_1.radius[1] - r2_actual) < 1e-10 "x=1 should give d2's radius"
println("✓ Endpoints are correct\n")

println("Test 3: Midpoint (x=0.5)")
println("-" ^ 40)
d_mid = geodesic_interpolation(d1, d2, 0.5)
r_mid_expected = 0.5 * r1_actual + 0.5 * r2_actual
println("  x=0.5: radius=$(d_mid.radius[1]), expected=$(r_mid_expected)")
@assert abs(d_mid.radius[1] - r_mid_expected) < 1e-10 "Midpoint should be average of radii"
println("✓ Midpoint is correct\n")

println("Test 4: Monotonicity (radius increases as x increases)")
println("-" ^ 40)
x_values = [0.0, 0.25, 0.5, 0.75, 1.0]
radii = []
for x in x_values
    d = geodesic_interpolation(d1, d2, x)
    push!(radii, d.radius[1])
    println("  x=$x: radius=$(d.radius[1])")
end
# Check that radii are increasing
for i in 1:(length(radii) - 1)
    @assert radii[i] <= radii[i + 1] "Radii should be non-decreasing"
end
println("✓ Radii are monotonically increasing\n")

println("Test 5: Multi-dimensional polydiscs")
println("-" ^ 40)
# Create 2-dimensional polydiscs
d1_2d = ValuationPolydisc([K(0), K(1)], [4, 3])
d2_2d = ValuationPolydisc([K(0), K(1)], [1, 1])

d_mid_2d = geodesic_interpolation(d1_2d, d2_2d, 0.5)
r1_coord1 = valuation_to_radius(4, p)
r2_coord1 = valuation_to_radius(1, p)
r1_coord2 = valuation_to_radius(3, p)
r2_coord2 = valuation_to_radius(1, p)
r_mid_coord1 = 0.5 * r1_coord1 + 0.5 * r2_coord1
r_mid_coord2 = 0.5 * r1_coord2 + 0.5 * r2_coord2

println("  Coordinate 1: radius=$(d_mid_2d.radius[1]), expected=$(r_mid_coord1)")
println("  Coordinate 2: radius=$(d_mid_2d.radius[2]), expected=$(r_mid_coord2)")
@assert abs(d_mid_2d.radius[1] - r_mid_coord1) < 1e-10 "Coord 1 should be correct"
@assert abs(d_mid_2d.radius[2] - r_mid_coord2) < 1e-10 "Coord 2 should be correct"
println("✓ Multi-dimensional interpolation works\n")

println("Test 6: Different prime (p=3)")
println("-" ^ 40)
K3 = PadicField(3, prec)
d1_3 = ValuationPolydisc([K3(0)], [4])
d2_3 = ValuationPolydisc([K3(0)], [1])

d_mid_3 = geodesic_interpolation(d1_3, d2_3, 0.5)
r1_3 = valuation_to_radius(4, 3)
r2_3 = valuation_to_radius(1, 3)
r_mid_3 = 0.5 * r1_3 + 0.5 * r2_3
println("  p=3, x=0.5: radius=$(d_mid_3.radius[1]), expected=$(r_mid_3)")
@assert abs(d_mid_3.radius[1] - r_mid_3) < 1e-10 "Should work for p=3"
println("✓ Works for different primes\n")

println("Test 7: Center preservation")
println("-" ^ 40)
# Test that centers are preserved
d1_center = ValuationPolydisc([K(7)], [5])
d2_center = ValuationPolydisc([K(7)], [2])

d_interp = geodesic_interpolation(d1_center, d2_center, 0.3)
println("  Input center: $(d1_center.center[1])")
println("  Output center: $(d_interp.center[1])")
@assert NonArchimedeanMachineLearning.valuation(d_interp.center[1] - d1_center.center[1]) > 15 "Center should be preserved"
println("✓ Center is preserved\n")

println("Test 8: Error handling - invalid containment")
println("-" ^ 40)
# Create two discs where d1 is NOT contained in d2
d_small = ValuationPolydisc([K(0)], [5])
d_not_containing = ValuationPolydisc([K(0)], [6])  # Even smaller!

try
    geodesic_interpolation(d_small, d_not_containing, 0.5)
    println("✗ Should have thrown an error")
    @assert false "Should have detected invalid containment"
catch e
    println("✓ Correctly detected invalid containment: $(e)")
end
println()

println("Test 9: Error handling - different centers")
println("-" ^ 40)
d_center1 = ValuationPolydisc([K(0)], [5])
d_center2 = ValuationPolydisc([K(1)], [1])  # Different center: val(0-1)=0 < radius=1

try
    geodesic_interpolation(d_center1, d_center2, 0.5)
    println("✗ Should have thrown an error")
    @assert false "Should have detected different centers"
catch e
    println("✓ Correctly detected different centers: $(e)")
end
println()

println("Test 10: Error handling - x out of bounds")
println("-" ^ 40)
d1_bounds = ValuationPolydisc([K(0)], [5])
d2_bounds = ValuationPolydisc([K(0)], [2])

try
    geodesic_interpolation(d1_bounds, d2_bounds, -0.1)
    println("✗ Should have thrown an error for x < 0")
    @assert false "Should have detected x < 0"
catch e
    println("✓ Correctly detected x < 0: $(e)")
end

try
    geodesic_interpolation(d1_bounds, d2_bounds, 1.1)
    println("✗ Should have thrown an error for x > 1")
    @assert false "Should have detected x > 1"
catch e
    println("✓ Correctly detected x > 1: $(e)")
end
println()

println("=== All Tests Passed ===")
