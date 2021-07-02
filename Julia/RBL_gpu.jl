using CUDA

# orthogonalize the two latest blocks against all the previous
function part_reorth_gpu!(U::Vector{Matrix{Float64}})
    i = size(U,1);
    U1 = CuArray(U[i]);
    U2 = CuArray(U[i-1]);
    CUDA.synchronize();
    Uj = CUDA.zeros(size(U1,1),size(U1,2));
    Uj_T = CUDA.zeros(size(U1,2),size(U1,1));
    for j=1:i-2
        @timeit to "Load Uj" copyto!(Uj,U[j]);
        CUDA.synchronize();
        @timeit to "transpose" transpose!(Uj_T,Uj);
        CUDA.synchronize();
        @timeit to "Uj*U1" temp = Uj_T*U1;
        CUDA.synchronize();
        @timeit to "U1" U1 = U1 - Uj*temp;
        CUDA.synchronize();
        @timeit to "Uj*U2" temp = Uj_T*U2;
        CUDA.synchronize();
        @timeit to "U2" U2 = U2 - Uj*temp;
        CUDA.synchronize();
    end
    copyto!(U[i],U1);
    copyto!(U[i-1],U2);
    CUDA.synchronize();
    return nothing
end


function RBL_gpu(A,k::Int64,b::Int64)
    n = size(A,2);
    Q = Matrix{Float64}[];
    Qi = randn(n,b);
    Qi = Matrix(qr(A*Qi).Q);
    V = zeros();
    D = zeros();
    
    # first loop
    push!(Q,Qi);
    U = A*Qi;
    Ai = transpose(Qi)*U;
    R = U - Qi*Ai;
    fact = qr(R);
    Qi = Matrix(fact.Q);
    Bi = fact.R;
    T = insertA!(Ai,b);
    insertB!(Bi,T,b,1);
    i = 2;
    while i*b < 500
        push!(Q,Qi);
        if mod(i,4) == 0
            @timeit to "part_reorth" part_reorth_gpu!(Q);
        end
        loc_reorth!(Q[i],Q[i-1]);
        U = A*Q[i] - Q[i-1]*transpose(Bi);
        Ai = transpose(Q[i])*U;
        R = U - Q[i]*Ai;
        fact = qr(R);
        Qi = Matrix(fact.Q);
        Bi = fact.R;
        T = [T insertA!(Ai,b)];
        if i*b > k
            D,V = dsbev('V','L',T);
            if norm(Bi*V[end-b+1:end,end-k+1]) < 1.0e-5
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
