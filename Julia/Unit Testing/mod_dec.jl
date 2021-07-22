include("test.jl")

@testset "RBL with moderate eigenvalue decay arrays" begin
    for i in 100:200:1000
        @test norm(moderate_decay(i,5,5)) < 1e-13
    end
end;