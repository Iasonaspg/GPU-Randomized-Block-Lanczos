using TimerOutputs
using MAT
using MatrixMarket
# using Arpack
# using MKL

include("./RBL.jl")
include("./RBL_gpu.jl")

function bench()
    # file = matopen("/home/iasonas/Desktop/randomized-block-lanczos/Matrix/ldoor.mat")
    # Problem = read(file,"Problem");
    # A::SparseMatrixCSC{DOUBLE} = Problem["A"];
    A::SparseMatrixCSC{DOUBLE} = mmread("/home/iasonas/Desktop/randomized-block-lanczos/Matrix/hood.mtx");

    @timeit to "RBL" @time d,v = RBL_restarted(A,100,2,10);
    # @timeit to "RBL" @time d,v = RBL(A,100,4);
    # @timeit to "RBL_gpu" CUDA.@time d,_ = RBL_gpu(A,50,1);
    # @timeit to "RBL_gpu" CUDA.@profile d,_ = RBL_gpu(A,50,1);
    # println(d);
    # println("\n\n");

    # @timeit to "Arpack" @time d,v = eigs(A,nev=10);
    # println(d);
    # res = norm(A*v[:,10] - d[10]*v[:,10]);
    # println("Norm: $res");
    # println("\n\n");
end

BLAS.set_num_threads(1);
# CUDA.math_mode!(CUDA.DEFAULT_MATH);

to = TimerOutput();
d,_ = RBL(sprandn(DOUBLE,200,200,0.5),1,1);
# d,_ = RBL_gpu(sprandn(DOUBLE,50,50,0.5),1,10);
to = TimerOutput();
bench();
show(to);
println();
println("\n\n");
