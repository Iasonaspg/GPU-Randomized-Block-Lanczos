using CUDA
using Adapt

include("common.jl")

CUDA.allowscalar(false);

function matrix_size(A::Union{SparseMatrixCSC{Float32,Int64},CUSPARSE.CuSparseMatrixCSC{Float32,Int32}})
    nnz = SparseArrays.nnz(A);
    n = size(A,2);
    return nnz*(sizeof(Float32) + sizeof(Int64)) + (n+1)*sizeof(Int64);
end

function matrix_size(A::Union{SparseMatrixCSC{Float64,Int64},CUSPARSE.CuSparseMatrixCSC{Float64,Int32}})
    nnz = SparseArrays.nnz(A);
    n = size(A,2);
    return nnz*(sizeof(Float64) + sizeof(Int64)) + (n+1)*sizeof(Int64);
end

function matrix_size(A::CuArray{Float64})
    return size(A,1)*size(A,2)*sizeof(Float64);
end

function blocksize(nrows::Int64,::Core.Type{T}) where T
    avail_mem = 0.7*CUDA.available_memory();
    return Int64( floor(avail_mem / (nrows * sizeof(T))) );
end

function part_reorth_gpu_async!(U1::CuArray{FLOAT},U2::CuArray{FLOAT},Ug::CuArray{FLOAT})
    synchronize();
    @sync begin
        @async begin
            temp = transpose(Ug)*U1;
            mul!(U1,Ug,temp,FLOAT(-1.0),FLOAT(1.0));
            nothing;
        end

        @async begin
            temp2 = transpose(Ug)*U2;
            mul!(U2,Ug,temp2,FLOAT(-1.0),FLOAT(1.0));
            nothing;
        end
    end
    U1[:,:] = U1;
    U2[:,:] = U2;
    return nothing
end

function part_reorth_gpu!(U1::CuArray{FLOAT},U2::CuArray{FLOAT},Ug::CuArray{FLOAT})
    temp = transpose(Ug)*U1;
    mul!(U1,Ug,temp,FLOAT(-1.0),FLOAT(1.0));
    temp = transpose(Ug)*U2;
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
            CUDA.@sync part_reorth_gpu_async!(Qg,Qg1,Qgpu[j]);
        end
        for j=last+1:i-2
            copyto!(Qgj,Q[j]);
            CUDA.@sync part_reorth_gpu!(Qg,Qg1,Qgj);
        end
    else
        for j=1:i-2
            CUDA.@sync part_reorth_gpu_async!(Qg,Qg1,Qgpu[j]);
        end
        copyto!(Qgpu[i-1],Qg1);
    end
    copyto!(Q[i-1],Qg1);
    Qg[:,:] = Qg;
    Qg1[:,:] = Qg1;
end

function loc_reorth_gpu!(U1::CuArray{FLOAT},U2::CuArray{FLOAT})
    p = size(U1,2);
    temp = CuArray{FLOAT}(undef,size(U2,2),size(U1,2));
    for i=1:2*p
        mul!(temp,transpose(U2),U1);
        mul!(U1,U2,temp,-1.0,1.0);
        U1 = CuArray(qr(U1).Q);
    end
    U1[:,:] = U1;
    return nothing;
end

function gpu_buffer_size(A::Union{SparseMatrixCSC{DOUBLE,Int64},CUSPARSE.CuSparseMatrixCSC{DOUBLE,Int32},CuArray{DOUBLE}}, n::Int64, b::Int64)
    avail_mem = 0.8*CUDA.available_memory();
    # println("avail_mem :$avail_mem");
    bl_sz_f = n*b*sizeof(FLOAT);
    bl_sz_d = n*b*sizeof(DOUBLE);
    avail_mem = avail_mem - 6*bl_sz_f - 5*bl_sz_d - matrix_size(A);
    buffer_size::Int64 = floor(avail_mem/bl_sz_f);
    println("buffer_size: $buffer_size");
    return max(buffer_size,0);
end

function recover_eigvec(Qcpu::Vector{Matrix{FLOAT}}, Qgpu::Vector{CuArray{FLOAT}}, Vk::Matrix{FLOAT}, k::Int64)
    n = size(Qcpu[1],1);
    b = size(Qcpu[1],2);
    V = zeros(FLOAT,n,k);
    blsz = blocksize(n,FLOAT);
    println("Blocksize: $blsz, buff_size: $(length(Qgpu)) and tot_size: $(length(Qcpu))");
    # blsz = 5;
    
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

function lanczos_iteration(
    Ag::Union{CUSPARSE.CuSparseMatrixCSC{DOUBLE},CuArray{DOUBLE}}, k::Int64, b::Int64, kryl_sz::Int64, Qg_d::CuArray{DOUBLE},
    Q::Vector{Matrix{FLOAT}}, Qgpu::Vector{CuArray{FLOAT}}
)
    n = size(Ag,2);
    buffer_size = gpu_buffer_size(Ag,n,b);
    # buffer_size = 50;

    Qg = CuArray{FLOAT}(Qg_d);
    Qg1_d = CuArray{DOUBLE}(undef,n,b);
    U = CuArray{DOUBLE}(undef,n,b);
    D = zeros(FLOAT);
    V = zeros(FLOAT);

    # first loop
    i = 1;
    push!(Q,Array(Qg));
    push!(Qgpu,copy(Qg));
    @timeit to "AQ" CUDA.@sync mul!(U,Ag,Qg_d);
    @timeit to "3-term" CUDA.@sync Ai::CuArray{DOUBLE} = transpose(Qg_d)*U;
    @timeit to "3-term" CUDA.@sync mul!(U,Qg_d,Ai,-1.0,1.0);   # U = U - Qg*Ai;
    @timeit to "qr" CUDA.@sync fact = qr(U);
    Qg1 = CuArray{FLOAT}(copy(Qg_d));
    Qg_d = CuArray(fact.Q);
    CUDA.copyto!(Qg,Qg_d);
    Bi = Array{DOUBLE}(fact.R);
    T = insertA!(Array(Ai),b);
    insertB!(Bi,T,b,1);
    while i*b < kryl_sz
        i = i + 1;
        if mod(i,2) == 0
            @timeit to "part reorth" CUDA.@sync hybrid_part_reorth!(i,buffer_size,Qgpu,Q,Qg1,Qg);
        end
        @timeit to "loc reorth" CUDA.@sync loc_reorth_gpu!(Qg,Qg1);
        @timeit to "push" CUDA.@sync push!(Q,Array(Qg));
        Q[end] = Mem.pin(Q[end]);
        if i <= buffer_size
            push!(Qgpu,copy(Qg));
        end
        CUDA.copyto!(Qg_d,Qg);
        CUDA.copyto!(Qg1_d,Qg1);
        Big = CuArray{DOUBLE}(Bi);
        @timeit to "AQ" CUDA.@sync mul!(U,Ag,Qg_d);
        @timeit to "3-term" CUDA.@sync mul!(U,Qg1_d,transpose(Big),FLOAT(-1.0),FLOAT(1.0));  # U = A*Q[i] - Q[i-1]*transpose(Bi)
        @timeit to "3-term" CUDA.@sync mul!(Ai,transpose(Qg_d),U);
        @timeit to "3-term" CUDA.@sync mul!(U,Qg_d,Ai,-1.0,1.0);
        @timeit to "qr" CUDA.@sync fact = qr(U);
        CUDA.copyto!(Qg1,Qg_d);
        Qg_d = CuArray(fact.Q);
        CUDA.copyto!(Qg,Qg_d);
        Bi = Array{DOUBLE}(fact.R);
        T = [T insertA!(Array(Ai),b)];
        if (i*b > k) && (mod(i,4) == 0)
            @timeit to "eig" CUDA.@sync D,V = dsbev('V','L',T);
            D,V = sort_eig_abs(D,V,k);
            if check_convergence(Bi,V,b,k,1e-7);
               break;
            end
        end
        insertB!(Bi,T,b,i);
    end
    println("Iterations: $i and kryl_sz: $(length(Q)*b)");
    Qg = nothing;
    Qg_d = nothing;
    Qg1 = nothing;
    Qg1_d = nothing;
    U = nothing;
    #CUDA.reclaim();
    return D[end:-1:1],V[:,end:-1:1];
end

function RBL_gpu(A::Union{SparseMatrixCSC{DOUBLE},Matrix{DOUBLE}},k::Int64,b::Int64)
    n = size(A,2);
    V = zeros(FLOAT);
    D = zeros(FLOAT);
    Ag = adapt(CuArray, A);
    
    max_kryl_sz = 1200;

    Qg_d = CUDA.randn(DOUBLE,n,b);
    Qg_d = CuArray(qr(Ag*Qg_d).Q);
    Q = Matrix{FLOAT}[];
    Qgpu = CuArray{FLOAT}[];
    D,V = lanczos_iteration(Ag,k,b,max_kryl_sz,Qg_d,Q,Qgpu);
    Ag = nothing;
    @timeit to "Ritz vectors" CUDA.@sync V = recover_eigvec(Q,Qgpu,Matrix{FLOAT}(V),k);
    return D,V;
end
