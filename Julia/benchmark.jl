using TimerOutputs
using MAT
using MatrixMarket
using Arpack


include("./RBL.jl")
include("./RBL_gpu.jl")

function bench()
    # file = matopen("/home/iasonas/Desktop/randomized-block-lanczos/Matrix/ldoor.mat")
    # Problem = read(file,"Problem");
    # A::SparseMatrixCSC{DOUBLE} = Problem["A"];
    A::SparseMatrixCSC{DOUBLE} = mmread("/home/iasonas/Desktop/randomized-block-lanczos/Matrix/ldoor.mtx");

    bs = [1,2,4,8];
    for i in bs
        @timeit to "RBL $i" @time d,v = RBL(A,25,i);
        println(d);
    end

    # @timeit to "RBL_gpu" CUDA.@time d,v = RBL_gpu(A,25,8);
    # @timeit to "RBL_gpu" CUDA.@time d,_ = RBL_gpu_restarted(A,10);
    # @timeit to "RBL" @time d,v = RBL_restarted(A,25);
    # @timeit to "RBL_gpu" CUDA.@profile d,v = RBL_gpu(A,50,1);
    # println(d);
    # println("\n\n");

    # @timeit to "Arpack" @time d,v = eigs(A,nev=100,tol=1e-7);
    # @timeit to "Arpack" @time d,v = eigs(A,nev=100,tol=1e-7,ncv=780);
    # println(d);
    # println("\n\n");
end

BLAS.set_num_threads(1);
# CUDA.math_mode!(CUDA.DEFAULT_MATH);

# to = TimerOutput();
d,_ = RBL(sprandn(DOUBLE,200,200,0.5),1,1);
# d,_ = RBL_gpu(sprandn(DOUBLE,50,50,0.5),1,1);
# d,v = eigs(sprandn(DOUBLE,50,50,0.5),nev=1);
# d,_ = RBL_restarted(sprandn(DOUBLE,200,200,0.5),1);
# d,_ = RBL_gpu_restarted(sprandn(DOUBLE,50,50,0.5),1);
to = TimerOutput();
bench();
show(to);
println();
println("\n\n");
