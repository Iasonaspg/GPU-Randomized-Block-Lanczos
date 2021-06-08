using LinearAlgebra
using BenchmarkTools
using ProfileView
using TimerOutputs
include("./RBL.jl")

const to = TimerOutput();

function bench()
    BLAS.set_num_threads(6)
    file = matopen("C:/Users/Iasonas/Desktop/Git Repositories/randomized-block-lanczos/matrices/Serena.mat");
    problem = read(file,"Problem");
    A = problem["A"];
    println("Lets start!");

    @timeit to "RBL" D,_ = RBL_timed(A,50,4); 
    # @time D,_ = RBL(A,50,4); 
    # @time D,_ = RBL(A,50,4); 
    # @time D,_ = RBL(A,50,4); 
    println(D);
end

bench();
show(to)
