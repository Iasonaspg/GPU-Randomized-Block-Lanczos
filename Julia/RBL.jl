using LinearAlgebra
using MAT
using TimerOutputs

include("RBL_gpu.jl")

# Require square A
function insertA!(A::Matrix{FLOAT}, b::Int64)
    T = zeros(FLOAT,b+1,b);
    n = size(A,2);
    for j = 1:n
        size = n - j + 1;
        T[1:size,j] = A[j:n,j];
    end
    return T;
end

# Require square B
function insertB!(B::Matrix{FLOAT}, T::Matrix{FLOAT}, b::Int64, iter::Int64)
    n = size(B,2);
    start = (iter-1)*b;
    for j = 1:n
        T[end-j+1:end,start+j] = B[1:j,j];
    end
end

function dsbev(jobz::Char, uplo::Char, A::Matrix{Float32})
    n = size(A,2);
    lda = size(A,1);
    bw = lda - 1;
    ldz = n;
    a = copy(A);
    work = zeros(Float32,3*n-2);
    info = 0;
    D = zeros(Float32,n,1);
    V = zeros(Float32,ldz,n);
    ccall((:ssbev_64_, Base.liblapack_name), Nothing, (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ptr{Float32}, Ref{Int64}, Ptr{Float32}, Ptr{Float32}, Ref{Int64}, Ptr{Float32}, Ref{Int64}), jobz, uplo, n, bw, a, lda, D, V, ldz, work, info)
    return D,V;
end

function dsbev(jobz::Char, uplo::Char, A::Matrix{Float64})
    n = size(A,2);
    lda = size(A,1);
    bw = lda - 1;
    ldz = n;
    a = copy(A);
    work = zeros(3*n-2);
    info = 0;
    D = zeros(n,1);
    V = zeros(ldz,n);
    ccall((:dsbev_64_, Base.liblapack_name), Nothing, (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}), jobz, uplo, n, bw, a, lda, D, V, ldz, work, info)
    return D,V;
end

# orthogonalize U1 against U2
function loc_reorth!(U1::Union{Matrix{FLOAT},CuArray{FLOAT}},U2::Union{Matrix{FLOAT},CuArray{FLOAT}})
    temp = transpose(U2)*U1;
    temp = U1 - U2*temp;
    U1[:,:] = Matrix(qr(temp).Q);
    return nothing;
end

# orthogonalize the two latest blocks against all the previous
function part_reorth!(U::Vector{Matrix{FLOAT}})
    i = size(U,1);
    for j=1:i-2
        Uj = U[j];
        Uj_T = transpose(Uj);
        # @timeit to "Uj*U2" temp = Uj_T*U[i];
        # @timeit to "U1" U[i] = U[i] - Uj*temp;
        # @timeit to "UjU2" temp = Uj_T*U[i-1];
        # @timeit to "U2" U[i-1] = U[i-1] - Uj*temp;
        temp = Uj_T*U[i];
        U[i] = U[i] - Uj*temp;
        temp = Uj_T*U[i-1];
        U[i-1] = U[i-1] - Uj*temp;
    end
    return nothing
end

function RBL(A::Union{SparseMatrixCSC{FLOAT}},k::Int64,b::Int64)
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
    Qi = randn(FLOAT,n,b);
    Qi = Matrix{FLOAT}(qr(A*Qi).Q);
    V = zeros(FLOAT);
    D = zeros(FLOAT);
   
    # first loop
    push!(Q,Qi);
    @timeit to "A*Qi" U = A*Qi;
    Ai = transpose(Qi)*U;
    R = U - Qi*Ai;
    fact = qr(R);
    Qi = Matrix{FLOAT}(fact.Q);
    Bi = fact.R;
    T = insertA!(Ai,b);
    insertB!(Bi,T,b,1);
    i = 2;
    while i*b < 1000
        push!(Q,Qi);
        if mod(i,3) == 0
            @timeit to "part_reorth" part_reorth!(Q);
        end
        @timeit to "loc_reorth" loc_reorth!(Q[i],Q[i-1]);
        @timeit to "A*Qi" U = A*Q[i] - Q[i-1]*transpose(Bi);
        @timeit to "Ai" Ai = transpose(Q[i])*U;
        @timeit to "U-QiAi" R = U - Q[i]*Ai;
        @timeit to "qr" fact = qr(R);
        @timeit to "qr" Qi = Matrix{FLOAT}(fact.Q);
        Bi = fact.R;
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
    # println("Iterations: $i");
    D = D[end:-1:end-k+1];
    #V = Q*V(:,1:k);
    return D,V;
end

function bench()
    file = matopen("/home/iasonas/Desktop/randomized-block-lanczos/ldoor.mat")
    Problem = read(file,"Problem");
    A::SparseMatrixCSC{FLOAT} = Problem["A"];
    @timeit to "RBL" d,_ = RBL_gpu(A,25,30);
    println(d);
end

# BLAS.set_num_threads(6);
# to = TimerOutput();
# bench()
# show(to);
# println();