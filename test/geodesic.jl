## Test file for geodesic interpolation

using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Geodesic Interpolation" begin
    # Setup: 2-adic field with precision 20
    prec = 20
    K = PadicField(2, prec)
    p = 2

    # Test that conversion functions are inverses
    test_vals = [0, 1, 2, 5, 10]
    for val in test_vals
        r = NonArchimedeanMachineLearning.valuation_to_radius(val, p)
        val_back = NonArchimedeanMachineLearning.radius_to_valuation(r, p)
        @test abs(val - val_back) < 1e-10
    end

    # Create two nested discs
    d1 = ValuationPolydisc([K(0)], [5])
    d2 = ValuationPolydisc([K(0)], [2])

    d_0 = NonArchimedeanMachineLearning.geodesic_interpolation(d1, d2, 0.0)
    r1_actual = NonArchimedeanMachineLearning.valuation_to_radius(5, p)
    @test abs(d_0.radius[1] - r1_actual) < 1e-10

    d_1 = NonArchimedeanMachineLearning.geodesic_interpolation(d1, d2, 1.0)
    r2_actual = NonArchimedeanMachineLearning.valuation_to_radius(2, p)
    @test abs(d_1.radius[1] - r2_actual) < 1e-10

    d_mid = NonArchimedeanMachineLearning.geodesic_interpolation(d1, d2, 0.5)
    r_mid_expected = 0.5 * r1_actual + 0.5 * r2_actual
    @test abs(d_mid.radius[1] - r_mid_expected) < 1e-10

    x_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    radii = Float64[]
    for x in x_values
        d = NonArchimedeanMachineLearning.geodesic_interpolation(d1, d2, x)
        push!(radii, d.radius[1])
    end
    for i in 1:(length(radii) - 1)
        @test radii[i] <= radii[i + 1]
    end

    d1_2d = ValuationPolydisc([K(0), K(1)], [4, 3])
    d2_2d = ValuationPolydisc([K(0), K(1)], [1, 1])

    d_mid_2d = NonArchimedeanMachineLearning.geodesic_interpolation(d1_2d, d2_2d, 0.5)
    r1_coord1 = NonArchimedeanMachineLearning.valuation_to_radius(4, p)
    r2_coord1 = NonArchimedeanMachineLearning.valuation_to_radius(1, p)
    r1_coord2 = NonArchimedeanMachineLearning.valuation_to_radius(3, p)
    r2_coord2 = NonArchimedeanMachineLearning.valuation_to_radius(1, p)
    r_mid_coord1 = 0.5 * r1_coord1 + 0.5 * r2_coord1
    r_mid_coord2 = 0.5 * r1_coord2 + 0.5 * r2_coord2

    @test abs(d_mid_2d.radius[1] - r_mid_coord1) < 1e-10
    @test abs(d_mid_2d.radius[2] - r_mid_coord2) < 1e-10

    K3 = PadicField(3, prec)
    d1_3 = ValuationPolydisc([K3(0)], [4])
    d2_3 = ValuationPolydisc([K3(0)], [1])

    d_mid_3 = NonArchimedeanMachineLearning.geodesic_interpolation(d1_3, d2_3, 0.5)
    r1_3 = NonArchimedeanMachineLearning.valuation_to_radius(4, 3)
    r2_3 = NonArchimedeanMachineLearning.valuation_to_radius(1, 3)
    r_mid_3 = 0.5 * r1_3 + 0.5 * r2_3
    @test abs(d_mid_3.radius[1] - r_mid_3) < 1e-10

    d1_center = ValuationPolydisc([K(7)], [5])
    d2_center = ValuationPolydisc([K(7)], [2])

    d_interp = NonArchimedeanMachineLearning.geodesic_interpolation(d1_center, d2_center, 0.3)
    @test NonArchimedeanMachineLearning.valuation(d_interp.center[1] - d1_center.center[1]) > 15

    d_small = ValuationPolydisc([K(0)], [5])
    d_not_containing = ValuationPolydisc([K(0)], [6])
    @test_throws Exception NonArchimedeanMachineLearning.geodesic_interpolation(d_small, d_not_containing, 0.5)

    d_center1 = ValuationPolydisc([K(0)], [5])
    d_center2 = ValuationPolydisc([K(1)], [1])
    @test_throws Exception NonArchimedeanMachineLearning.geodesic_interpolation(d_center1, d_center2, 0.5)

    d1_bounds = ValuationPolydisc([K(0)], [5])
    d2_bounds = ValuationPolydisc([K(0)], [2])
    @test_throws Exception NonArchimedeanMachineLearning.geodesic_interpolation(d1_bounds, d2_bounds, -0.1)
    @test_throws Exception NonArchimedeanMachineLearning.geodesic_interpolation(d1_bounds, d2_bounds, 1.1)
end
