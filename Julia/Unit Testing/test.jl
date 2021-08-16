using LinearAlgebra
using SparseArrays
using Test

include("../RBL.jl");

# Function that computes the k largest eigenvalues of A and returns the residual based on eig
function RBL_residual(A,eig::Vector{FLOAT},k::Int64,b::Int64)
    d,_ = RBL(A,k,b);
    # println((d - eig) ./ eig);
    return (d - eig) ./ eig;
end

# Create array with moderate eigenvalue decay
function moderate_decay(n::Int64,k::Int64,b::Int64)
    a = Vector{FLOAT}();
    sum = 0;
    for i in 1:n
        sum += i;
        push!(a,sum);
    end
    eig = a[end:-1:end-k+1];
    A = sparse(Diagonal(a));

    return RBL_residual(A,eig,k,b);
end

# Create array with slow eigenvalue decay
function slow_decay(n::Int64,k::Int64,b::Int64)
    a::Vector{FLOAT} = [1.0:n;];
    eig = a[end:-1:end-k+1];
    A = sparse(Diagonal(a));

    return RBL_residual(A,eig,k,b);
end

# Create array with step eigenvalue decay
function step_decay(n::Int64,k::Int64,b::Int64)
    a = ones(FLOAT,n);
    sz = 2*k;
    for i in 1:sz
        a[sz+1-i] = i*n;
    end
    eig = a[1:k];
    A = sparse(Diagonal(a));

    return RBL_residual(A,eig,k,b);
end

