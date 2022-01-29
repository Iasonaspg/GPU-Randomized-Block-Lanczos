using LinearAlgebra, LowRankApprox, Images, Plots

include("./RBL.jl")
include("./RBL_gpu.jl")
using Arpack

# I = load("./collis.jpg")
# G = Gray.(I)
# B = Array{Float64}(G)
B = Array{Float64}(Gray.(load("./collis.jpg")));

to1 = TimerOutput();

k = 300;

psvd(randn(50,50));
@timeit to1 "PSVD" U, S, V = psvd(B);
Blr= U[:,1:k]*diagm(S[1:k])*transpose(V[:,1:k]);

RBL(sprandn(DOUBLE,200,200,0.5),1,1);
@timeit to1 "RBL" begin
    D,V = RBL(transpose(B)*B,k,1);
end
U = (B*V) ./ transpose(D);
Blr = U*diagm(D)*transpose(V);


RBL_gpu(sprandn(DOUBLE,50,50,0.5),1,1);
@timeit to1 "RBL_gpu" begin
    D,V = RBL_gpu(transpose(B)*B,k,1);
end
    U = (B*V) ./ transpose(D);
Blr = U*diagm(D)*transpose(V);


svds(sprandn(DOUBLE,200,200,0.5),nsv=1,tol=1e-7);
@timeit to1 "svds" begin
    obj = svds(B,nsv=k,tol=1e-7)[1];
    U = obj.U;
    S = obj.S;
    V = obj.V;
end
Blr = U*diagm(D)*transpose(V);

show(to1);
println();


plot(Gray.(Blr),
    xaxis=false, 
    xticks=false, 
    yaxis=false, 
    yticks=false, 
    grid=false, 
    title="Using $k largest singular values. Full rank 4912"
)