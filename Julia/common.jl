using LinearAlgebra
using SparseArrays
using TimerOutputs

const FLOAT = Float64;
const DOUBLE = Float64;

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
    ccall((:ssbev_64_, Base.liblapack_name), Nothing, (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ptr{Float32}, Ref{Int64}, Ptr{Float32}, Ptr{Float32}, Ref{Int64}, Ptr{Float32}, Ref{Int64}), jobz, uplo, n, bw, a, lda, D, V, ldz, work, info)
end

function dsbev_lapack(jobz::Char, uplo::Char, n::Int64, bw::Int64, a::Matrix{Float64}, lda::Int64, D::Matrix{Float64}, V::Matrix{Float64}, ldz::Int64, work::Vector{Float64}, info::Int64)
    ccall((:dsbev_64_, Base.liblapack_name), Nothing, (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ptr{Float64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}), jobz, uplo, n, bw, a, lda, D, V, ldz, work, info)
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