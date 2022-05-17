# orthogonalize the two latest blocks against all the previous
function part_reorth_gpu!(U::Vector{Matrix{FLOAT}})
    i = size(U,1);
    U1 = CuArray{FLOAT}(U[i]);
    U2 = CuArray{FLOAT}(U[i-1]);
    Uj = CUDA.zeros(size(U1,1),size(U1,2));
    for j=1:i-2
        copyto!(Uj,U[j]);
        temp = transpose(Uj)*U1;
        U1 = U1 - Uj*temp;
        temp = transpose(Uj)*U2;
        U2 = U2 - Uj*temp;
    end
    copyto!(U[i],U1);
    copyto!(U[i-1],U2);
    return nothing
end

function RBL_gpu_old(A::SparseMatrixCSC{FLOAT},k::Int64,b::Int64)
    n = size(A,2);
    V = zeros(FLOAT);
    D = zeros(FLOAT);
    Ag = adapt(CuArray,A);
    
    Qi = randn(FLOAT,n,b);
    Qg = CuArray(Qi);
    Qg = CuArray(qr(Ag*Qg).Q);
    copyto!(Qi,Qg);
    
    Q = Matrix{FLOAT}[];

    # first loop
    push!(Q,Qi);
    U = Array(Ag*Qg);
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
            # NVTX.@range "part reorth" begin
                part_reorth_gpu!(Q);
            # end
        end
        loc_reorth!(Q[i],Q[i-1]);
        copyto!(Qg,Q[i]);
        U = Array(Ag*Qg) - Q[i-1]*transpose(Bi);
        Ai = transpose(Q[i])*U;
        R = U - Q[i]*Ai;
        fact = qr(R);
        Qi = Matrix{FLOAT}(fact.Q);
        Bi = fact.R;
        T = [T insertA!(Ai,b)];
        if i*b > k
            D,V = dsbev('V','L',T);
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
