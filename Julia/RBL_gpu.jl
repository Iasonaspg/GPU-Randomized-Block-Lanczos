using CUDA
using Adapt

include("common.jl")

CUDA.allowscalar(false);

function sparse_size(A::Union{SparseMatrixCSC{Float32,Int64},CUSPARSE.CuSparseMatrixCSC{Float32,Int32}})
    nnz = SparseArrays.nnz(A);
    n = size(A,2);
    return nnz*(sizeof(Float32) + sizeof(Int64)) + (n+1)*sizeof(Int64);
end

function sparse_size(A::Union{SparseMatrixCSC{Float64,Int64},CUSPARSE.CuSparseMatrixCSC{Float64,Int32}})
    nnz = SparseArrays.nnz(A);
    n = size(A,2);
    return nnz*(sizeof(Float64) + sizeof(Int64)) + (n+1)*sizeof(Int64);
end

function blocksize(nrows::Int64,::Core.Type{T}) where T
    avail_mem = 0.9*CUDA.available_memory();
    return Int64( floor(avail_mem / (nrows * sizeof(T))) );
end

# orthogonalize the two latest blocks against all the previous
function part_reorth_gpu!(U::Vector{Matrix{FLOAT}})
    i = size(U,1);
    U1 = CuArray{FLOAT}(U[i]);
    U2 = CuArray{FLOAT}(U[i-1]);
    Uj = CUDA.zeros(size(U1,1),size(U1,2));
    # Uj_T = CUDA.zeros(size(U1,2),size(U1,1));
    for j=1:i-2
        copyto!(Uj,U[j]);
        # @timeit to "transpose" transpose!(Uj_T,Uj);
        temp = transpose(Uj)*U1;
        U1 = U1 - Uj*temp;
        temp = transpose(Uj)*U2;
        U2 = U2 - Uj*temp;
    end
    copyto!(U[i],U1);
    copyto!(U[i-1],U2);
    return nothing
end

function part_reorth_gpu_block!(U1::CuArray{FLOAT},U2::CuArray{FLOAT},Ug::CuArray{FLOAT})
    temp = transpose(Ug)*U1;
    mul!(U1,Ug,temp,FLOAT(-1.0),FLOAT(1.0));
    mul!(temp,transpose(Ug),U2);
    mul!(U2,Ug,temp,FLOAT(-1.0),FLOAT(1.0));
    U1[:,:] = U1;
    U2[:,:] = U2;
    return nothing
end

function restart_reorth_gpu!(Q::Vector{Matrix{FLOAT}},Qgpu::Vector{CuArray{FLOAT}},Qg::CuArray{FLOAT})
    i = length(Qgpu);
    temp = CuArray{FLOAT}(undef,size(Qg,2),size(Qg,2));
    for j=1:i
        mul!(temp,transpose(Qgpu[j]),Qg);
        mul!(Qg,Qgpu[j],temp,FLOAT(-1.0),FLOAT(1.0));
    end
    
    cpu_locked = length(Q);
    if cpu_locked > 0
        println("Enter\n");
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

function loc_reorth_gpu!(U1::CuArray{FLOAT},U2::CuArray{FLOAT})
    p = size(U1,2);
    for i=1:2*p
        temp = transpose(U2)*U1;
        mul!(U1,U2,temp,-1.0,1.0);
        U1 = CuArray(qr(U1).Q);
    end
    U1[:,:] = U1;
    return nothing;
end

function recover_eigvec(Qcpu::Vector{Matrix{FLOAT}},Qgpu::Vector{CuArray{FLOAT}},Vk::Matrix{FLOAT},k::Int64)
    n = size(Qcpu[1],1);
    b = size(Qcpu[1],2);
    V = zeros(FLOAT,n,k);
    blsz = blocksize(n,FLOAT);
    # temp = CuArray{FLOAT}(undef,n,min(blsz,k));
    println("Blocksize: $blsz, buff_size: $(length(Qgpu)) and tot_size: $(length(Qcpu))");;
    
    buff_size = length(Qgpu);
    i = 1;
    while i <= k
        blsz_i = min(blsz,k-i+1);
        temp = CUDA.zeros(FLOAT,n,blsz_i)
        V_trunc = cu(Vk[:, i:i+blsz_i-1]);
        for j=1:buff_size
            mul!(temp,Qgpu[j],V_trunc[(j-1)*b+1:j*b,:],1.0,1.0);
        end
        V[:,i:i+blsz_i-1] = Matrix{FLOAT}(temp);
        i = i + blsz_i;
    end

    tot_size = length(Qcpu);
    for i = buff_size+1 : tot_size
        mul!(V,Qcpu[i],Vk[(i-1)*b+1:i*b,:],1.0,1.0);
    end
    return V;
end

function gpu_buffer_size(A::Union{SparseMatrixCSC{DOUBLE,Int64},CUSPARSE.CuSparseMatrixCSC{DOUBLE,Int32}},n::Int64,b::Int64)
    avail_mem = 0.8*CUDA.available_memory();
    bl_sz_f = n*b*sizeof(FLOAT);
    bl_sz_d = n*b*sizeof(DOUBLE);
    avail_mem = avail_mem - 6*bl_sz_f - 5*bl_sz_d - sparse_size(A);
    buffer_size::Int64 = floor(avail_mem/bl_sz_f);
    println("buffer_size: $buffer_size");
    return buffer_size;
end

function lanczos_iteration(
    Ag::CUSPARSE.CuSparseMatrixCSC{DOUBLE},k::Int64,b::Int64,kryl_sz::Int64,Qg_d::CuArray{DOUBLE},Q::Vector{Matrix{FLOAT}},
    Qgpu::Vector{CuArray{FLOAT}},Qlock::Vector{Matrix{FLOAT}},Qlock_gpu::Vector{CuArray{FLOAT}}
)
    n = size(Ag,2);
    buffer_size = gpu_buffer_size(Ag,n,b);

    Qg = CuArray{FLOAT}(Qg_d);
    Qgj = CuArray{FLOAT}(undef,n,b);
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
            if i > buffer_size
                last = min(buffer_size,i-2);
                for j=1:last
                    @timeit to "part_reorth" CUDA.@sync part_reorth_gpu_block!(Qg,Qg1,Qgpu[j]);
                end
                for j=last+1:i-2
                    copyto!(Qgj,Q[j]);
                    @timeit to "part_reorth" CUDA.@sync part_reorth_gpu_block!(Qg,Qg1,Qgj);
                end
            else
                for j=1:i-2
                    @timeit to "par_reorth" CUDA.@sync part_reorth_gpu_block!(Qg,Qg1,Qgpu[j]);
                end
                copyto!(Qgpu[i-1],Qg1);
            end
            copyto!(Q[i-1],Qg1);
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
        if i*b > k
            @timeit to "dsbev" D,V = dsbev('V','L',T);
            if norm(Bi*V[end-b+1:end,end-k+1]) < 1.0e-5
               break;
            end
        end
        insertB!(Bi,T,b,i);
        i = i + 1;
    end
    println("Iterations: $i");
    Qg = nothing;
    Qg_d = nothing;
    Qg1 = nothing;
    Qg1_d = nothing;
    U = nothing;
    CUDA.reclaim();
    # @timeit to "Recovery" V = recover_eigvec(Q,Qgpu,Matrix{FLOAT}(V[:,end:-1:end-k+1]),k); # V = Q*V(:,1:k);
    return D,V;
end

function RBL_gpu(A::SparseMatrixCSC{DOUBLE},k::Int64,b::Int64)
    n = size(A,2);
    V = zeros(FLOAT);
    D = zeros(FLOAT);
    Ag = adapt(CuArray, A);
    
    max_kryl_sz = 1000;

    Qg_d = CUDA.randn(DOUBLE,n,b);
    Qg_d = CuArray(qr(Ag*Qg_d).Q);
    Q = Matrix{FLOAT}[];
    Qgpu = CuArray{FLOAT}[];
    Qlock = Matrix{FLOAT}[];
    Qlock_gpu = CuArray{FLOAT}[];
    D,V = lanczos_iteration(Ag,k,b,max_kryl_sz,Qg_d,Q,Qgpu,Qlock,Qlock_gpu);
    return D,V;
end

function RBL_gpu_restarted(A::SparseMatrixCSC{DOUBLE},k::Int64,b::Int64,step::Int64)
    if (mod(k,step) != 0)
        throw(ArgumentError("number of desired eigenvalues should be multiple of step size"))
    end

    if (mod(step,b) != 0)
        throw(ArgumentError("step size should be multiple of block size"))
    end

    n = size(A,2);
    V = zeros(FLOAT);
    D = zeros(FLOAT);
    Ag = adapt(CuArray, A);
    
    max_kryl_sz = 1000;

    Qg_d = CUDA.randn(DOUBLE,n,b);
    Qg_d = CuArray(qr(Ag*Qg_d).Q);
    Q = Matrix{FLOAT}[];
    Qgpu = CuArray{FLOAT}[];
    Qlock = Matrix{FLOAT}[];
    Qlock_gpu = CuArray{FLOAT}[];
    count = 0;
    while (count < k)
        D,V = lanczos_iteration(Ag,step,b,max_kryl_sz,Qg_d,Q,Qgpu,Qlock,Qlock_gpu);
        println("D1: $(D[end:-1:end-step+1])");
        QV = recover_eigvec(Q,Qgpu,Matrix{FLOAT}(V[:,end:-1:end-k+1]),k);
        
        Qi = recover_eigvec(Q,Qgpu,Matrix{FLOAT}(V[:,1:b]),b);
        copyto!(Qg_d,Qi);
        Q = Matrix{FLOAT}[];
        Qgpu = CuArray{FLOAT}[];
        buff_size = gpu_buffer_size(Ag,n,b);
        blks = Int64(step/b);
        for i=1:blks
            if i <= buff_size
                push!(Qlock_gpu,CuArray{FLOAT}(QV[:,(i-1)*b+1:i*b]));
            else
                push!(Qlock,QV[:,(i-1)*b+1:i*b]);
            end
        end
        count = count + step;
    end
    return D,V;
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
