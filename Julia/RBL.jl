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


function lanczos_iteration(A::Union{SparseMatrixCSC{DOUBLE},Matrix{DOUBLE}},k::Int64,b::Int64,kryl_sz::Int64,Qi::Matrix{FLOAT},Q::Vector{Matrix{FLOAT}})
    D = zeros(DOUBLE);
    V = zeros(DOUBLE);
    
    # first loop
    push!(Q,Qi);
    U::Matrix{DOUBLE} = A*Qi;
    Ai::Matrix{DOUBLE} = transpose(Qi)*U;
    mul!(U,Qi,Ai,-1.0,1.0);
    U = Matrix{FLOAT}(U);
    fact = qr(U);
    Qi = Matrix{FLOAT}(fact.Q);
    Bi = Matrix{DOUBLE}(fact.R);
    T = insertA!(Ai,b);
    insertB!(Bi,T,b,1);
    i = 2;
    while i*b < kryl_sz
        push!(Q,Qi);
        if mod(i,2) == 0
            part_reorth!(Q);
        end
        loc_reorth!(Q[i],Q[i-1]);
        mul!(U,A,Q[i]);
        mul!(U,Q[i-1],transpose(Bi),-1.0,1.0);  # U = A*Q[i] - Q[i-1]*transpose(Bi)
        mul!(Ai,transpose(Q[i]),U,1.0,0.0);
        U = Matrix{FLOAT}(U);
        mul!(U,Qi,Ai,-1.0,1.0);
        fact = qr(U);
        Qi = Matrix{FLOAT}(fact.Q);
        Bi = Matrix{DOUBLE}(fact.R);
        T = [T insertA!(Ai,b)];
        if (i*b > k) && (mod(i,4) == 0)
            D,V = dsbev('V','L',T);
            if norm(Bi*V[end-b+1:end,end-k+1]) < 1.0e-7
               break;
            end
        end
        insertB!(Bi,T,b,i);
        i = i + 1;
    end
    println("Iterations: $i");
    return D[end:-1:end-k+1],V[:,end:-1:end-k+1];
end

function new_lanczos_iteration(
    A::Union{SparseMatrixCSC{DOUBLE},Matrix{DOUBLE}},b::Int64,kryl_sz::Int64,Qi::Matrix{FLOAT},
    Q::Vector{Matrix{FLOAT}},Qlock::Vector{Matrix{FLOAT}}
)
    D = zeros(DOUBLE);
    V = zeros(DOUBLE);
    B = Matrix{DOUBLE}[];
    
    # first loop
    restart_reorth!(Qlock,Qi);
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
    while i*b < kryl_sz
        push!(Q,Qi);
        if mod(i,3) == 0
            part_reorth!(length(Qlock),Qlock,Q[i],Q[i-1]);
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
        insertB!(Bi,T,b,i);
        i = i + 1;
    end
    part_reorth!(length(Qlock),Qlock,Q[end],Q[end-1]);
    @timeit to "part_reorth" part_reorth!(Q);
    println("Iterations: $i");
    @timeit to "dsbev" D,V = dsbev('V','L',T);
    res_bounds = Bi*V[end-b+1:end,end:-1:1]; # residual bounds in descending order
    return D[end:-1:1],V[:,end:-1:1],res_bounds;
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
    V = recover_eigvec(Q,Matrix{FLOAT}(V),k);
    return D,V;
end

function RBL_restarted(A::Union{SparseMatrixCSC{DOUBLE},Matrix{DOUBLE}},k::Int64)
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
    max_kryl_sz = 80;
    n = size(A,2);
    D = zeros(FLOAT,k,1);
    V = zeros(FLOAT,n,k);
    
    Q = Matrix{FLOAT}[];
    Qlock = Matrix{FLOAT}[];
    Qi = randn(DOUBLE,n,1);
    Qi = Matrix{FLOAT}(qr(A*Qi).Q);
    
    count = 0;
    while (count < k)
        d,v,conv = new_lanczos_iteration(A,1,max_kryl_sz,Qi,Q,Qlock);
        ncomp = 0;
        for i=1:length(conv)
            if (count + ncomp < k)
                if norm(conv[i]) < 1e-7
                    ncomp = ncomp + 1;
                    println("val: $(d[i])");
                    @timeit to "recovery" qv = recover_eigvec(Q,Matrix{FLOAT}(v[:,i:i]),1);
                    push!(Qlock,qv);
                    D[count+ncomp] = d[i];
                else
                    @timeit to "recovery" Qi = recover_eigvec(Q,Matrix{FLOAT}(v[:,i:i]),1);
                    break;
                end
            else
                break;
            end
        end
        
        Q = Matrix{FLOAT}[];
        count = count + ncomp;
        max_kryl_sz = max_kryl_sz + 10;
    end
    return D,V;
end
