using TimerOutputs
using MAT
using MatrixMarket
using Arpack
using LinearMaps
using CSV
using Tables

include("./RBL.jl")
include("./RBL_gpu.jl")

function Ax_gpu(x)
    xg = CuArray(x);
    return Vector(Ag*xg);
end

function Ax(x)
    return A*x;
end

const A = mmread("../Matrix/audi.mtx");
# const Ag = adapt(CuArray,A);

function bench()
    # file = matopen("/home/iasonas/Desktop/randomized-block-lanczos/Matrix/ldoor.mat")
    # Problem = read(file,"Problem");
    # A::SparseMatrixCSC{DOUBLE} = Problem["A"];
    # global A::SparseMatrixCSC{DOUBLE} = mmread("../Matrix/hood.mtx");
    nd = 100;

    # bs = [1,2,4,8];
    # for i in bs
        # @timeit to "RBL $i" @time d,v = RBL(A,100,i);
        # println("Largest: $(d[1]) and smallest $(d[end])");
    # end
    @timeit to "RBL" @time d,v = RBL(A,nd,4);
    # @timeit to "RBL_gpu" CUDA.@time d,v = RBL_gpu(A,nd,4);
    # println("Largest: $(d[1]) and smallest $(d[nd])");

    # @timeit to "RBL_gpu" CUDA.@profile CUDA.@time d,v = RBL_gpu(A,10,4);
    
    #  @timeit to "Arpack" @time d,v = eigs(A,nev=nd,tol=1e-7,which=:LM);
    # @timeit to "Arpack" @time d,v = eigs(A_map_gpu,nev=nd,tol=1e-7,which=:LM);
    # println(d);
    println("Largest: $(d[1]) and smallest $(d[nd])");
    println("\n");
end

BLAS.set_num_threads(1);
println("Number of threads: $(BLAS.get_num_threads())");
CUDA.math_mode!(CUDA.DEFAULT_MATH);

A_map = LinearMap(Ax,size(A,1),issymmetric=true);
A_map_gpu = LinearMap(Ax_gpu,size(A,1),issymmetric=true);

to = TimerOutput();
d,_ = RBL(sprandn(DOUBLE,200,200,0.5),1,1);
#  d,_ = RBL_gpu(sprandn(DOUBLE,50,50,0.5),1,1);
# d,v = eigs(sprandn(DOUBLE,50,50,0.5),nev=1);
# d,v = eigs(A_map_gpu,nev=1);
to = TimerOutput();
bench();
show(to);
println();
println("\n\n");
