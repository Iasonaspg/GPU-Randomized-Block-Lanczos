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
    U1[:,:] = U1 - Ug*temp;
    temp = transpose(Ug)*U2;
    U2[:,:] = U2 - Ug*temp;
    return nothing
end

function loc_reorth_gpu!(U1::CuArray{FLOAT},U2::CuArray{FLOAT})
    temp = transpose(U2)*U1;
    temp = U1 - U2*temp;
    U1[:,:] = CuArray(qr(temp).Q);
    return nothing;
end

function RBL_gpu(A::SparseMatrixCSC{FLOAT},k::Int64,b::Int64)
    n = size(A,2);
    V = zeros(FLOAT);
    D = zeros(FLOAT);
    Ag = adapt(CuArray, A);
    
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
    # buffer_size = 1;

    # first loop
    push!(Q,Array(Qg));
    push!(Qgpu,copy(Qg));
    U = Ag*Qg;
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
                    last = min(buffer_size,i-2);
                    for j=1:last
                        part_reorth_gpu_block!(Qg,Qg1,Qgpu[j]);
                    end
                    for j=last+1:i-2
                        copyto!(Qgj,Q[j]);
                        part_reorth_gpu_block!(Qg,Qg1,Qgj);
                end
            else
                    for j=1:i-2
                        part_reorth_gpu_block!(Qg,Qg1,Qgpu[j]);
                end
                copyto!(Qgpu[i-1],Qg1);
            end
            copyto!(Q[i-1],Qg1);
        end
        loc_reorth_gpu!(Qg,Qg1);
        push!(Q,Array(Qg));
        if i <= buffer_size
            push!(Qgpu,copy(Qg));
        end
        Big = CuArray{FLOAT}(Bi);
        U = Ag*Qg - Qg1*transpose(Big);
        Ai = transpose(Qg)*U;
        R = U - Qg*Ai;
        fact = qr(R);
        copyto!(Qg1,Qg);
        Qg = CuArray(fact.Q);
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
    Qi = Matrix(fact.Q);
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
        Qi = Matrix(fact.Q);
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
