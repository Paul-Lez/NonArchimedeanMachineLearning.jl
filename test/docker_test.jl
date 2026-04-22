#!/usr/bin/env julia
# Quick test to verify Docker setup works correctly

println("=== NonArchimedeanMachineLearning Docker Environment Test ===\n")

# Test 1: Check Julia version
println("1. Julia version: $(VERSION)")

# Test 2: Check if we can activate the project
println("2. Activating project...")
using Pkg
Pkg.activate(".")

# Test 3: Check if Oscar is available
println("3. Loading Oscar...")
try
    using Oscar
    println("   ✓ Oscar loaded successfully")
catch e
    println("   ✗ Failed to load Oscar: $e")
    exit(1)
end

# Test 4: Check if LinearAlgebra is available
println("4. Loading LinearAlgebra...")
try
    using LinearAlgebra
    println("   ✓ LinearAlgebra loaded successfully")
catch e
    println("   ✗ Failed to load LinearAlgebra: $e")
    exit(1)
end

# Test 5: Try to include NonArchimedeanMachineLearning module
println("5. Loading NonArchimedeanMachineLearning module...")
try
    include("../src/NonArchimedeanMachineLearning.jl")
    using .NonArchimedeanMachineLearning
    println("   ✓ NonArchimedeanMachineLearning module loaded successfully")
catch e
    println("   ✗ Failed to load NonArchimedeanMachineLearning: $e")
    exit(1)
end

# Test 6: Create a simple p-adic polydisc
println("6. Testing p-adic polydisc creation...")
try
    K = PadicField(2, 10)
    center = [K(1), K(2)]
    radius = [0, 1]
    p = NonArchimedeanMachineLearning.ValuationPolydisc(center, radius)
    println("   ✓ Created polydisc: center=$(center), radius=$(radius)")
    println("   ✓ Dimension: $(NonArchimedeanMachineLearning.dim(p))")
    println("   ✓ Prime: $(NonArchimedeanMachineLearning.prime(p))")
catch e
    println("   ✗ Failed to create polydisc: $e")
    exit(1)
end

println("\n=== All tests passed! ✓ ===")
println("Docker environment is ready for NonArchimedeanMachineLearning development.")
