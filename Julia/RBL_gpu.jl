using CUDA
using Adapt
using SparseArrays

const FLOAT = Float32;
CUDA.allowscalar(false);

function sparse_size(A::SparseMatrixCSC{Float32,Int64})
    nnz = SparseArrays.nnz(A);
    n = size(A,2);
    return nnz*(sizeof(Float32) + sizeof(Int64)) + (n+1)*sizeof(Int64);
end

function sparse_size(A::SparseMatrixCSC{Float64,Int64})
    nnz = SparseArrays.nnz(A);
    n = size(A,2);
    return nnz*(sizeof(Float64) + sizeof(Int64)) + (n+1)*sizeof(Int64);
end

# orthogonalize the two latest blocks against all the previous
function part_reorth_gpu!(U::Vector{Matrix{FLOAT}})
    i = size(U,1);
    @timeit to "CuArray allocation" U1 = CuArray{FLOAT}(U[i]);
    @timeit to "CuArray allocation" U2 = CuArray{FLOAT}(U[i-1]);
    Uj = CUDA.zeros(size(U1,1),size(U1,2));
    # Uj_T = CUDA.zeros(size(U1,2),size(U1,1));
    for j=1:i-2
        @timeit to "Load Uj" copyto!(Uj,U[j]);
        # @timeit to "transpose" transpose!(Uj_T,Uj);
        @timeit to "Uj^T*U1" temp = transpose(Uj)*U1;
        @timeit to "U1" U1 = U1 - Uj*temp;
        @timeit to "Uj^T*U2" temp = transpose(Uj)*U2;
        @timeit to "U2" U2 = U2 - Uj*temp;
    end
    @timeit to "copy to CPU" copyto!(U[i],U1);
    @timeit to "copy to CPU" copyto!(U[i-1],U2);
    return nothing
end

function part_reorth_gpu_block!(U1::CuArray{FLOAT},U2::CuArray{FLOAT},Ug::CuArray{FLOAT})
    @timeit to "Uj^T*U1" temp = transpose(Ug)*U1;
    @timeit to "U1" U1[:,:] = U1 - Ug*temp;
    @timeit to "Uj^T*U2" temp = transpose(Ug)*U2;
    @timeit to "U2" U2[:,:] = U2 - Ug*temp;
    return nothing
end

function loc_reorth_gpu!(U1::CuArray{FLOAT},U2::CuArray{FLOAT})
    @timeit to "transpose" temp = transpose(U2)*U1;
    @timeit to "u1-u2*temp" temp = U1 - U2*temp;
    @timeit to "qr" U1[:,:] = CuArray(qr(temp).Q);
    return nothing;
end

function RBL_gpu(A::SparseMatrixCSC{FLOAT},k::Int64,b::Int64)
    n = size(A,2);
    V = zeros(FLOAT);
    D = zeros(FLOAT);
    @timeit to "A gpu allocation" Ag = adapt(CuArray, A);
    
    Qi = randn(FLOAT,n,b);
    Qg = CuArray(Qi);
    Qg = CuArray(qr(Ag*Qg).Q);
    
    Q = Matrix{FLOAT}[];
    Qgpu = CuArray{FLOAT}[];
    Qgj = CuArray{FLOAT}(undef,n,b);

    # GPU buffer size
    avail_mem = CUDA.available_memory();
    bl_sz = n*b*sizeof(FLOAT);
    avail_mem = avail_mem - 11*bl_sz - sparse_size(A);
    buffer_size::Int32 = floor(avail_mem/bl_sz);
    println("buffer_size: $buffer_size");
    buffer_size = 45;

    # first loop
    push!(Q,Array(Qg));
    push!(Qgpu,copy(Qg));
    @timeit to "A*Qi" U = Ag*Qg;
    Ai = transpose(Qg)*U;
    R = U - Qg*Ai;
    fact = qr(R);
    Qg1 = CuArray(Qg);
    Qg = CuArray(fact.Q);
    Bi = Array(fact.R);
    T = insertA!(Array(Ai),b);
    insertB!(Bi,T,b,1);
    i = 2;
    while i*b < 1000
        if mod(i,3) == 0
            if i > buffer_size
                @timeit to "part_reorth_hybrid" begin
                    last = min(buffer_size,i-2);
                    for j=1:last
                        part_reorth_gpu_block!(Qg,Qg1,Qgpu[j]);
                    end
                    for j=last+1:i-2
                        copyto!(Qgj,Q[j]);
                        part_reorth_gpu_block!(Qg,Qg1,Qgj);
                    end
                end
            else
                @timeit to "part_reorth_on" begin
                    for j=1:i-2
                        part_reorth_gpu_block!(Qg,Qg1,Qgpu[j]);
                    end
                end
                copyto!(Qgpu[i-1],Qg1);
            end
            copyto!(Q[i-1],Qg1);
        end
        @timeit to "loc_reorth" loc_reorth_gpu!(Qg,Qg1);
        push!(Q,Array(Qg));
        if i <= buffer_size
            push!(Qgpu,copy(Qg));
        end
        Big = CuArray{FLOAT}(Bi);
        @timeit to "A*Qi" U = Ag*Qg - Qg1*transpose(Big);
        @timeit to "Ai" Ai = transpose(Qg)*U;
        @timeit to "U-QiAi" R = U - Qg*Ai;
        @timeit to "qr" fact = qr(R);
        copyto!(Qg1,Qg);
        @timeit to "qr" Qg = CuArray(fact.Q);
        Bi = Array(fact.R);
        T = [T insertA!(Array(Ai),b)];
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

function RBL_gpu_old(A::SparseMatrixCSC{FLOAT},k::Int64,b::Int64)
    n = size(A,2);
    V = zeros(FLOAT);
    D = zeros(FLOAT);
    @timeit to "A gpu allocation" Ag = adapt(CuArray, A);
    
    Qi = randn(FLOAT,n,b);
    Qg = CuArray(Qi);
    Qg = CuArray(qr(Ag*Qg).Q);
    
    Q = Matrix{FLOAT}[];
    Qgpu = CuArray{FLOAT}[];
    Qgj = CuArray{FLOAT}(undef,n,b);

    # GPU buffer size
    avail_mem = CUDA.available_memory();
    bl_sz = n*b*sizeof(FLOAT);
    avail_mem = avail_mem - 4*bl_sz - sparse_size(A); # 4 blocks needed at least for part_reorth
    buffer_size::Int32 = floor(avail_mem/bl_sz);
    println("buffer_size: $buffer_size");

    # first loop
    push!(Q,Array(Qg));
    push!(Qgpu,copy(Qg));
    @timeit to "A*Qi" U = Ag*Qg;
    Ai = transpose(Qg)*U;
    R = Array(U - Qg*Ai);
    fact = qr(R);
    Qi = Matrix(fact.Q);
    Bi = fact.R;
    T = insertA!(Array(Ai),b);
    insertB!(Bi,T,b,1);
    Qg1 = CuArray(Qg);
    copyto!(Qg,Qi);
    i = 2;
    while i*b < 1000
        if mod(i,3) == 0
            # Ag = nothing;
            if i > buffer_size
                @timeit to "part_reorth_hybrid" begin
                    last = min(buffer_size,i-2);
                    for j=1:last
                        part_reorth_gpu_block!(Qg,Qg1,Qgpu[j]);
                    end
                    for j=last+1:i-2
                        copyto!(Qgj,Q[j]);
                        part_reorth_gpu_block!(Qg,Qg1,Qgj);
                    end
                end
            else
                @timeit to "part_reorth_on" begin
                    for j=1:i-2
                        # copyto!(Qgj,Q[j]);
                        part_reorth_gpu_block!(Qg,Qgpu[i-1],Qgpu[j]);
                    end
                end
                # @timeit to "copyto" copyto!(Qgpu[i-1],Qgpu[i-1]);
                # @timeit to "copyto" copyto!(Qgpu[i],Qgpu[i]);
                # copyto!(Q[i],Qgpu[i]);
                # copyto!(Qgpu[i-1],Qgpu[i-1]);
                copyto!(Q[i-1],Qgpu[i-1]);
            end
        end
        @timeit to "loc_reorth" loc_reorth_gpu!(Qg,Qgpu[i-1]);
        push!(Q,Qi);
        if i <= buffer_size
            push!(Qgpu,copy(Qg));
        end
        # copyto!(Q[i],Qg);
        # copyto!(Qgpu[i],Qg);
        # copyto!(Qg,Qg);
        Big = CuArray{FLOAT}(Bi);
        # @timeit to "copyto" copyto!(Q[i],Qg);
        @timeit to "A*Qi" U = Ag*Qg - Qgpu[i-1]*transpose(Big);
        @timeit to "Ai" Ai = transpose(Qg)*U;
        @timeit to "U-QiAi" R = Array(U - Qg*Ai);
        @timeit to "qr" fact = qr(R);
        @timeit to "qr" Qg = Matrix(fact.Q);
        Bi = fact.R;
        T = [T insertA!(Array(Ai),b)];
        copyto!(Qg1,Qg);
        if i*b > k
            D,V = dsbev('V','L',T);
            if norm(Bi*V[end-b+1:end,end-k+1]) < 1.0e-7
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
