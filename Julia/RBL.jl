include("common.jl")

# orthogonalize U1 against U2
function loc_reorth!(U1::Matrix{FLOAT},U2::Matrix{FLOAT})
    p = size(U1,2);
    for i=1:2*p
        temp = BLAS.gemm('T','N',FLOAT(1.0),U2,U1);
        BLAS.gemm!('N','N',FLOAT(-1.0),U2,temp,FLOAT(1.0),U1);
        U1 = Matrix(qr(U1).Q);
    end
    U1[:,:] = U1;
    return nothing;
end

# orthogonalize the two latest blocks against all the previous
function part_reorth!(nblocks::Int64,U::Vector{Matrix{FLOAT}},U1::Matrix{FLOAT},U2::Matrix{FLOAT})
    temp = Array{FLOAT}(undef,size(U1,2),size(U1,2));
    for j=1:nblocks
        Uj = U[j];
        mul!(temp,transpose(Uj),U1);
        mul!(U1,Uj,temp,FLOAT(-1.0),FLOAT(1.0)); # FLOAT is needed to prevent cast to Float64 and copy of data!
        mul!(temp,transpose(Uj),U2);
        mul!(U2,Uj,temp,FLOAT(-1.0),FLOAT(1.0));
    end
    U1[:,:] = U1;
    U2[:,:] = U2;
    return nothing
end

function part_reorth!(U::Vector{Matrix{FLOAT}})
    i = size(U,1);
    temp1 = Array{FLOAT}(undef,size(U[1],2),size(U[1],2));
    temp2 = Array{FLOAT}(undef,size(U[1],2),size(U[1],2));
    for j=1:i-2
        Uj = U[j];
        @sync begin
            Threads.@spawn begin
                mul!(temp1,transpose(Uj),U[i]);
                mul!(U[i],Uj,temp1,FLOAT(-1.0),FLOAT(1.0));
            end
            Threads.@spawn begin
                mul!(temp2,transpose(Uj),U[i-1]);
                mul!(U[i-1],Uj,temp2,FLOAT(-1.0),FLOAT(1.0));
            end
        end
    end
    return nothing
end

function restart_reorth!(U::Vector{Matrix{FLOAT}},Q::Matrix{FLOAT})
    len = length(U);
    temp = Array{FLOAT}(undef,size(Q,2),size(Q,2));
    for j=1:len
        Uj = U[j];
        mul!(temp,transpose(Uj),Q);
        mul!(Q,Uj,temp,FLOAT(-1.0),FLOAT(1.0));
    end
    return nothing
end

function recover_eigvec(Q::Vector{Matrix{FLOAT}},V_trunc::Matrix{FLOAT},k::Int64)
    n = size(Q[1],1);
    b = size(Q[1],2);
    V = zeros(FLOAT,n,k);

    tot_size = length(Q);
    for i=1:tot_size
        mul!(V,Q[i],V_trunc[(i-1)*b+1:i*b,:],1.0,1.0);
    end
    return V;
end


function lanczos_iteration(A::Union{SparseMatrixCSC{DOUBLE},Matrix{DOUBLE}}, k::Int64, b::Int64, kryl_sz::Int64, Qi::Matrix{FLOAT}, Q::Vector{Matrix{FLOAT}})
    D = zeros(DOUBLE);
    V = zeros(DOUBLE);
    
    # first loop
    push!(Q,Qi);
    @timeit to "A*Q" U::Matrix{DOUBLE} = A*Qi;
    @timeit to "3-term" Ai::Matrix{DOUBLE} = transpose(Qi)*U;
    @timeit to "3-term" mul!(U,Qi,Ai,-1.0,1.0);
    U = Matrix{FLOAT}(U);
    @timeit to "QR" fact = qr(U);
    Qi = Matrix{FLOAT}(fact.Q);
    Bi = Matrix{DOUBLE}(fact.R);
    T = insertA!(Ai,b);
    insertB!(Bi,T,b,1);
    i = 1;
    while i*b < kryl_sz
        i = i + 1;
        push!(Q,Qi);
        if mod(i,2) == 0
            @timeit to "Part reorth" part_reorth!(Q);
        end
        @timeit to "Loc reorth" loc_reorth!(Q[i],Q[i-1]);
        @timeit to "A*Q" mul!(U,A,Q[i]);
        @timeit to "3-term" mul!(U,Q[i-1],transpose(Bi),-1.0,1.0);  # U = A*Q[i] - Q[i-1]*transpose(Bi)
        @timeit to "3-term" mul!(Ai,transpose(Q[i]),U,1.0,0.0);
        U = Matrix{FLOAT}(U);
        @timeit to "3-term" mul!(U,Qi,Ai,-1.0,1.0);
        @timeit to "QR" fact = qr(U);
        Qi = Matrix{FLOAT}(fact.Q);
        Bi = Matrix{DOUBLE}(fact.R);
        T = [T insertA!(Ai,b)];
        if (i*b > k) && (mod(i,4) == 0)
            @timeit to "eig" D,V = dsbev('V','L',T);
            D,V = sort_eig_abs(D,V,k);
            if check_convergence(Bi,V,b,k,1e-7);
               break;
            end
        end
        insertB!(Bi,T,b,i);
    end
    println("Iterations: $i and kryl_sz: $(length(Q)*b)");
    return D[end:-1:1],V[:,end:-1:1];
end

function RBL(A::Union{SparseMatrixCSC{DOUBLE},Matrix{DOUBLE}},k::Int64,b::Int64)
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
    max_kryl_sz = 1400;
    n = size(A,2);
    Q = Matrix{FLOAT}[];
    Qi = randn(DOUBLE,n,b);
    Qi = Matrix{FLOAT}(qr(A*Qi).Q);
   
    D,V = lanczos_iteration(A,k,b,max_kryl_sz,Qi,Q);
    @timeit to "Ritz vectors" V = recover_eigvec(Q,Matrix{FLOAT}(V),k);
    return D,V;
end
