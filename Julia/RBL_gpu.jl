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
    avail_mem = 0.7*CUDA.available_memory();
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

function hybrid_part_reorth!(i::Int64,buffer_size::Int64,Qgpu::Vector{CuArray{FLOAT}},Q::Vector{Matrix{FLOAT}},Qg1::CuArray{FLOAT},Qg::CuArray{FLOAT})
    n = size(Q[1],1);
    b = size(Q[1],2);
    Qgj = CuArray{FLOAT}(undef,n,b);
    if i > buffer_size
        last = min(buffer_size,i-2);
        for j=1:last
            CUDA.@sync part_reorth_gpu_block!(Qg,Qg1,Qgpu[j]);
        end
        for j=last+1:i-2
            copyto!(Qgj,Q[j]);
            CUDA.@sync part_reorth_gpu_block!(Qg,Qg1,Qgj);
        end
    else
        for j=1:i-2
            CUDA.@sync part_reorth_gpu_block!(Qg,Qg1,Qgpu[j]);
        end
        copyto!(Qgpu[i-1],Qg1);
    end
    copyto!(Q[i-1],Qg1);
    Qg[:,:] = Qg;
    Qg1[:,:] = Qg1;
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
    # println("Blocksize: $blsz, buff_size: $(length(Qgpu)) and tot_size: $(length(Qcpu))");;
    
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
    println("avail_mem :$avail_mem");
    bl_sz_f = n*b*sizeof(FLOAT);
    bl_sz_d = n*b*sizeof(DOUBLE);
    avail_mem = avail_mem - 6*bl_sz_f - 5*bl_sz_d - sparse_size(A);
    buffer_size::Int64 = floor(avail_mem/bl_sz_f);
    println("buffer_size: $buffer_size");
    return max(buffer_size,0);
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
            @timeit to "part_reorth" hybrid_part_reorth!(i,buffer_size,Qgpu,Q,Qg1,Qg);
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
        if (i*b > k) && (mod(i,4) == 0)
            @timeit to "dsbev" D,V = dsbev('V','L',T);
            if norm(Bi*V[end-b+1:end,end-k+1]) < 1.0e-7
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
    return D[end:-1:end-k+1],V[:,end:-1:end-k+1];
end

function new_lanczos_iteration(
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
    @timeit to "Recover eigevec" V = recover_eigvec(Q,Qgpu,Matrix{FLOAT}(V),k);
    return D,V;
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
        d,v,conv = new_lanczos_iteration(Ag,k,1,max_kryl_sz,Qg_d,Q,Qgpu,Qlock,Qlock_gpu);
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
