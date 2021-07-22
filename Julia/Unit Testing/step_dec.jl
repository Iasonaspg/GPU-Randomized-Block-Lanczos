include("test.jl")

@testset "RBL with step eigenvalue decay arrays" begin
    for i in 100000:200000:1000000
        @test norm(step_decay(i,5,5)) < 1e-13
    end
end;