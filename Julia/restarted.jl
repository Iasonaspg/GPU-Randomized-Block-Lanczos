function restart_reorth_gpu!(Q::Vector{Matrix{FLOAT}},Qgpu::Vector{CuArray{FLOAT}},Qg::CuArray{FLOAT})
    i = length(Qgpu);
    temp = CuArray{FLOAT}(undef,size(Qg,2),size(Qg,2));
    for j=1:i
        mul!(temp,transpose(Qgpu[j]),Qg);
        mul!(Qg,Qgpu[j],temp,FLOAT(-1.0),FLOAT(1.0));
    end
    
    cpu_locked = length(Q);
    if cpu_locked > 0
        Qgj = CuArray{FLOAT}(undef,size(Qg,1),size(Qg,2));
        for j=1:cpu_locked
            copyto!(Qgj,Q[j]);
            mul!(temp,transpose(Qgj),Qg);
            mul!(Qg,Qgj,temp,FLOAT(-1.0),FLOAT(1.0));
        end
    end

    Qg[:,:] = Qg;
    return nothing
end

function lanczos_iteration_res(
    Ag::CUSPARSE.CuSparseMatrixCSC{DOUBLE},k::Int64,b::Int64,kryl_sz::Int64,Qg_d::CuArray{DOUBLE},Q::Vector{Matrix{FLOAT}},
    Qgpu::Vector{CuArray{FLOAT}},Qlock::Vector{Matrix{FLOAT}},Qlock_gpu::Vector{CuArray{FLOAT}}
)
    n = size(Ag,2);
    # buffer_size = gpu_buffer_size(Ag,n,b);
    buffer_size = 150;

    Qg = CuArray{FLOAT}(Qg_d);
    Qg1_d = CuArray{DOUBLE}(undef,n,b);
    U = CuArray{DOUBLE}(undef,n,b);
    D = zeros(FLOAT);
    V = zeros(FLOAT);

    # first loop
    restart_reorth_gpu!(Qlock,Qlock_gpu,Qg);
    push!(Q,Array(Qg));
    push!(Qgpu,copy(Qg));
    @timeit to "A*Qi" CUDA.@sync mul!(U,Ag,Qg_d);
    Ai::CuArray{DOUBLE} = transpose(Qg_d)*U;
    mul!(U,Qg_d,Ai,-1.0,1.0);   # U = U - Qg*Ai;
    fact = qr(U);
    Qg1 = CuArray{FLOAT}(copy(Qg_d));
    Qg_d = CuArray(fact.Q);
    CUDA.copyto!(Qg,Qg_d);
    Bi = Array{DOUBLE}(fact.R);
    T = insertA!(Array(Ai),b);
    insertB!(Bi,T,b,1);
    i = 2;
    while i*b < kryl_sz
        if mod(i,3) == 0
            restart_reorth_gpu!(Qlock,Qlock_gpu,Qg1);
            restart_reorth_gpu!(Qlock,Qlock_gpu,Qg);
            hybrid_part_reorth!(i,buffer_size,Qgpu,Q,Qg1,Qg);
        end
        @timeit to "loc_reorth" CUDA.@sync loc_reorth_gpu!(Qg,Qg1);
        push!(Q,Array(Qg));
        if i <= buffer_size
            push!(Qgpu,copy(Qg));
        end
        CUDA.copyto!(Qg_d,Qg);
        CUDA.copyto!(Qg1_d,Qg1);
        Big = CuArray{DOUBLE}(Bi);
        @timeit to "A*Qi" CUDA.@sync mul!(U,Ag,Qg_d);
        @timeit to "A*Qi" CUDA.@sync mul!(U,Qg1_d,transpose(Big),FLOAT(-1.0),FLOAT(1.0));  # U = A*Q[i] - Q[i-1]*transpose(Bi)
        mul!(Ai,transpose(Qg_d),U);
        @timeit to "U-QiAi" CUDA.@sync mul!(U,Qg_d,Ai,-1.0,1.0);
        @timeit to "qr" CUDA.@sync fact = qr(U);
        CUDA.copyto!(Qg1,Qg_d);
        @timeit to "qr" CUDA.@sync Qg_d = CuArray(fact.Q);
        CUDA.copyto!(Qg,Qg_d);
        Bi = Array{DOUBLE}(fact.R);
        T = [T insertA!(Array(Ai),b)];
        if (i+1)*b < kryl_sz
            insertB!(Bi,T,b,i);
        end
        i = i + 1;
    end
    println("Iterations: $i");
    Qg = nothing;
    Qg_d = nothing;
    Qg1 = nothing;
    Qg1_d = nothing;
    U = nothing;
    CUDA.reclaim();

    restart_reorth_gpu!(Qlock,Qlock_gpu,Qgpu[end-1]);
    restart_reorth_gpu!(Qlock,Qlock_gpu,Qgpu[end]);
    hybrid_part_reorth!(i-1,buffer_size,Qgpu,Q,Qgpu[end-1],Qgpu[end]);
    @timeit to "dsbev" D,V = dsbev('V','L',T);
    res_bounds = Bi*V[end-b+1:end,end:-1:1]; # residual bounds in descending order
    return D[end:-1:1],V[:,end:-1:1],res_bounds;
end

function RBL_gpu_restarted(A::SparseMatrixCSC{DOUBLE},k::Int64)
    n = size(A,2);
    D = zeros(FLOAT,k,1);
    V = zeros(FLOAT,n,k);
    Ag = adapt(CuArray, A);
    
    max_kryl_sz = 100;

    Qg_d = CUDA.randn(DOUBLE,n,1);
    Qg_d = CuArray(qr(Ag*Qg_d).Q);
    Q = Matrix{FLOAT}[];
    Qgpu = CuArray{FLOAT}[];
    Qlock = Matrix{FLOAT}[];
    Qlock_gpu = CuArray{FLOAT}[];
    count = 0;
    while (count < k)
        d,v,conv = lanczos_iteration_res(Ag,k,1,max_kryl_sz,Qg_d,Q,Qgpu,Qlock,Qlock_gpu);
        buff_size = 100;
        ncomp = 0;
        for i=1:length(conv)
            if (count + ncomp < k)
                if norm(conv[i]) < 1e-7
                    ncomp = ncomp + 1;
                    println("val: $(d[i])");
                    @timeit to "recovery" qv = recover_eigvec(Q,Qgpu,Matrix{FLOAT}(v[:,i:i]),1);
                    
                    if (count + ncomp <= buff_size)
                        push!(Qlock_gpu,CuArray{FLOAT}(qv));
                    else
                        push!(Qlock,qv);
                    end

                    D[count+ncomp] = d[i];
                else
                    @timeit to "recovery" Qi = recover_eigvec(Q,Qgpu,Matrix{FLOAT}(v[:,i:i]),1);
                    copyto!(Qg_d,Qi);
                    break;
                end
            else
                break;
            end
        end
        
        Q = Matrix{FLOAT}[];
        Qgpu = CuArray{FLOAT}[];
        max_kryl_sz = max_kryl_sz + 10;
        count = count + ncomp;
    end
    return D,V;
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