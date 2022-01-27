using LinearAlgebra, LowRankApprox, Images, Plots

include("./RBL.jl")

I = load("./collis.jpg")
G = Gray.(I)
size(G)
B = Array{Float64}(G)
# B = Array{Float64}(Gray.(load("./collis.jpg")))

@timeit to "PSVD" U, S, V = psvd(B)
B100= U[:,1:100]*diagm(S[1:100])*transpose(V[:,1:100]);

plot(Gray.(B), 
    xaxis=false, 
    xticks=false, 
    yaxis=false, 
    yticks=false, 
    grid=false, 
    title="B=B(365) 365 by 548 full rank matrix"
)

@timeit to "Eig" begin
    D,V = RBL(transpose(B)*B,300,8);
    U = (B*V) ./ transpose(D);
end
Blr = U*diagm(D)*transpose(V);
plot(Gray.(Blr),
    xaxis=false, 
    xticks=false, 
    yaxis=false, 
    yticks=false, 
    grid=false, 
    title="B=B(365) 365 by 548 full rank matrix"
)