using LinearAlgebra
using MAT
using TimerOutputs

include("RBL_gpu.jl")

# Require square A
function insertA!(A::Matrix{DOUBLE}, b::Int64)
    T = zeros(DOUBLE,b+1,b);
    n = size(A,2);
    for j = 1:n
        size = n - j + 1;
        T[1:size,j] = A[j:n,j];
    end
    return T;
end

# Require square B
function insertB!(B::Matrix{DOUBLE}, T::Matrix{DOUBLE}, b::Int64, iter::Int64)
    n = size(B,2);
    start = (iter-1)*b;
    for j = 1:n
        T[end-j+1:end,start+j] = B[1:j,j];
    end
end

function dsbev_lapack(jobz::Char, uplo::Char, n::Int64, bw::Int64, a::Matrix{Float32}, lda::Int64, D::Matrix{Float32}, V::Matrix{Float32}, ldz::Int64, work::Vector{Float32}, info::Int64)
    ccall((:ssbev_, Base.liblapack_name), Nothing, (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ptr{Float32}, Ref{Int64}, Ptr{Float32}, Ptr{Float32}, Ref{Int64}, Ptr{Float32}, Ref{Int64}), jobz, uplo, n, bw, a, lda, D, V, ldz, work, info)
end

function dsbev_lapack(jobz::Char, uplo::Char, n::Int64, bw::Int64, a::Matrix{Float64}, lda::Int64, D::Matrix{Float64}, V::Matrix{Float64}, ldz::Int64, work::Vector{Float64}, info::Int64)
    ccall((:dsbev_, Base.liblapack_name), Nothing, (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}), jobz, uplo, n, bw, a, lda, D, V, ldz, work, info)
end

function dsbev(jobz::Char, uplo::Char, A::Matrix{DOUBLE})
    n = size(A,2);
    lda = size(A,1);
    bw = lda - 1;
    ldz = n;
    a = copy(A);
    work = zeros(DOUBLE,3*n-2);
    info = 0;
    D = zeros(DOUBLE,n,1);
    V = zeros(DOUBLE,ldz,n);
    dsbev_lapack(jobz,uplo,n,bw,a,lda,D,V,ldz,work,info);
    return D,V;
end

# orthogonalize U1 against U2
function loc_reorth!(U1::Matrix{FLOAT},U2::Matrix{FLOAT})
    temp = BLAS.gemm('T','N',FLOAT(1.0),U2,U1);
    BLAS.gemm!('N','N',FLOAT(-1.0),U2,temp,FLOAT(1.0),U1);
    U1[:,:] = Matrix(qr(U1).Q);
    return nothing;
end

# orthogonalize the two latest blocks against all the previous
function part_reorth!(U::Vector{Matrix{FLOAT}})
    i = size(U,1);
    temp = Array{FLOAT}(undef,size(U[1],2),size(U[1],2));
    for j=1:i-2
        Uj = U[j];
        @timeit to "Uj_t*U1" mul!(temp,transpose(Uj),U[i]);
        @timeit to "U1" mul!(U[i],Uj,temp,FLOAT(-1.0),FLOAT(1.0));
        @timeit to "Uj_t*U2" mul!(temp,transpose(Uj),U[i-1]);
        @timeit to "U2" mul!(U[i-1],Uj,temp,FLOAT(-1.0),FLOAT(1.0));
    end
    return nothing
end

function RBL(A::Union{SparseMatrixCSC{DOUBLE}},k::Int64,b::Int64)
#=
Input parameters
A - n by n Real Symmetric Matrix whose eigenvalues we seek
k - number of largest eigenvalues
b - block size

Output parameters
D - k by 1 vector with k largest eigenvalues of A (by magnitude)
V - n by k matrix with eigenvectors associated with the k largest eigenvalues of A

This routine uses the randomized block Lanczos algorithm to compute the k 
largest eigenvalues of a matrix A.
=#

    n = size(A,2);
    Q = Matrix{FLOAT}[];
    Qi = randn(DOUBLE,n,b);
    Qi = Matrix{FLOAT}(qr(A*Qi).Q);
    V = zeros(DOUBLE);
    D = zeros(DOUBLE);
   
    # first loop
    push!(Q,Qi);
    @timeit to "A*Qi" U::Matrix{DOUBLE} = A*Qi;
    Ai::Matrix{DOUBLE} = transpose(Qi)*U;
    mul!(U,Qi,Ai,-1.0,1.0);
    U = Matrix{FLOAT}(U);
    fact = qr(U);
    Qi = Matrix{FLOAT}(fact.Q);
    Bi = Matrix{DOUBLE}(fact.R);
    T = insertA!(Ai,b);
    insertB!(Bi,T,b,1);
    i = 2;
    while i*b < 1000
        push!(Q,Qi);
        if mod(i,3) == 0
            @timeit to "part_reorth" part_reorth!(Q);
        end
        @timeit to "loc_reorth" loc_reorth!(Q[i],Q[i-1]);
        @timeit to "A*Qi" mul!(U,A,Q[i]);
        @timeit to "A*Qi" mul!(U,Q[i-1],transpose(Bi),-1.0,1.0);  # U = A*Q[i] - Q[i-1]*transpose(Bi)
        @timeit to "Ai" mul!(Ai,transpose(Q[i]),U,1.0,0.0);
        U = Matrix{FLOAT}(U);
        @timeit to "U-QiAi" mul!(U,Qi,Ai,-1.0,1.0);
        @timeit to "qr" fact = qr(U);
        @timeit to "qr" Qi = Matrix{FLOAT}(fact.Q);
        Bi = Matrix{DOUBLE}(fact.R);
        T = [T insertA!(Ai,b)];
        if i*b > k
            @timeit to "dsbev" D,V = dsbev('V','L',T);
            if norm(Bi*V[end-b+1:end,end-k+1]) < 1.0e-6
               break;
            end
        end
        insertB!(Bi,T,b,i);
        i = i + 1;
    end
    println("Iterations: $i");
    D = D[end:-1:end-k+1];
    #V = Q*V(:,1:k);
    return D,V;
end

function bench()
    file = matopen("/home/iasonas/Desktop/randomized-block-lanczos/Matrix/Serena.mat")
    Problem = read(file,"Problem");
    A::SparseMatrixCSC{DOUBLE} = Problem["A"];
    @timeit to "RBL" @time d,_ = RBL(A,25,10);
    # @timeit to "RBL_gpu" CUDA.@time d,_ = RBL_gpu(A,25,10);
    # @timeit to "RBL_gpu" CUDA.@profile d,_ = RBL_gpu(A,25,10);
    println(d);
end

BLAS.set_num_threads(1);
# CUDA.math_mode!(CUDA.DEFAULT_MATH);
to = TimerOutput();
d,_ = RBL(sprandn(DOUBLE,50,50,0.5),1,10);
# d,_ = RBL_gpu(sprandn(DOUBLE,50,50,0.5),1,10);
to = TimerOutput();
bench();
show(to);
println();
