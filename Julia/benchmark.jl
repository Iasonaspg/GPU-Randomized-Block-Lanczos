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
    A::SparseMatrixCSC{DOUBLE} = mmread("../Matrix/ldoor.mtx");

    # bs = [1,2,4,8];
    # for i in bs
        # @timeit to "RBL $i" @time d,v = RBL(A,100,i);
        # println("Largest: $(d[1]) and smallest $(d[end])");
        # println(d);
    # end
    @timeit to "RBL" @time d,v = RBL(A,100,4);

    # @timeit to "RBL_gpu" CUDA.@time d,v = RBL_gpu(A,100,4);
    # @timeit to "RBL" @time d,v = RBL_restarted(A,25);
    # @timeit to "RBL_gpu" CUDA.@profile d,v = RBL_gpu(A,100,4);
    # println("\n\n");
    
    # @timeit to "Arpack" @time d,v = eigs(A,nev=100,tol=1e-7,which=:LM);
    # @timeit to "Arpack" @time d,v = eigs(A,nev=100,tol=1e-7,ncv=576);
    # println(d);
    println("Largest: $(d[1]) and smallest $(d[end])");
    println("\n");
end

BLAS.set_num_threads(1);
println("Number of threads: $(BLAS.get_num_threads())");
CUDA.math_mode!(CUDA.DEFAULT_MATH);

to = TimerOutput();
d,_ = RBL(sprandn(DOUBLE,200,200,0.5),1,1);
#  d,_ = RBL_gpu(sprandn(DOUBLE,50,50,0.5),1,1);
# d,v = eigs(sprandn(DOUBLE,50,50,0.5),nev=1);
to = TimerOutput();
bench();
show(to);
println();
println("\n\n");
