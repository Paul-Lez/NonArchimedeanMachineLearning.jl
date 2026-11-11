using Test
using Oscar
using NonArchimedeanMachineLearning

@testset "Creating DAGs and nodes" begin
    dag = mk_empty_flattened_dag(Int, 5)
    @test length(dag.nodes) == 5
    @test all(x -> x === nothing, dag.nodes) 

    # create a node with value 1
    b = mk_node(dag, 1)
    @test b.state == 1
    @test b.index == 1
    @test dag.nodes[1] === b
    @test b.parent == Int[]
    @test b.children == Int[]

    # create a node with value 2
    c = mk_node(dag, 2, parent=[1])
    @test c.state == 2
    @test c.index == 2
    @test dag.nodes[2] === c
    @test c.parent == Int[1]
    @test c.children == Int[]
    @test b.children == Int[2]

    # try to create a node with value 3 and child 5: fails because 5 does not exist yet
    @test_throws Exception d = mk_node_with_index(dag, 3, 4, parent=[1, 2], children=[5])

    e = mk_node_with_index(dag, 3, 5, parent=[1, 2], children=Int[])
    @test e.parent == Int[1, 2]
    d = mk_node_with_index(dag, 3, 4, parent=[1, 2], children=[5])
    @test e.parent == Int[1, 2, 3]
    @test d.parent == Int[1, 2]
    @test d.children == Int[5]

    @testset "Span" begin
        dagCopy1 = deepcopy(dag)
        dagCopy2 = deepcopy(dag)
        b1 = get_node(dagCopy1, 1)
        c2 = get_node(dagCopy2, 2)
        span1 = restrict_to_span!(dagCopy1, b1)
        span2 = restrict_to_span!(dagCopy2, c2)
        
        @test length(span1.nodes) == 5
        @test length(span2.nodes) == 5
    end
end
