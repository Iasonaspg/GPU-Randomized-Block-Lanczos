function [V,D] = RBL(A,k,b)
% Input parameters
% A - n by n Real Symmetric Matrix whose eigenvalues we seek
% k - number of largest eigenvalues
% b - block size
%
%
% Output parameters
% D - k by 1 vector with k largest eigenvalues of A (by magnitude)
% V - n by k matrix with eigenvectors associated with the k largest eigenvalues of A
%
% This routine uses the randomized block Lanczos algorithm to compute the k 
% largest eigenvalues of a matrix A.
%
    n = size(A,2);
    T = zeros(k*b,k*b);
    Q = zeros(n,k*b);
    Qi = randn(n,b);
    [Qi,~] = qr(A*Qi,0);
   
    % first loop
    Q(:,1:b) = Qi;
    U = A*Qi;
    M = Qi.'*U;
    R = U - Qi*M;
    [Qi,B] = qr(R,0);
    T(1:b,1:b) = M;
    T(b+1:2*b,1:b) = B;
    T(1:b,b+1:2*b) = B.';
    i = 2;
    while i*b < 500
        Q(:,(i-1)*b+1:i*b) = Qi;
        if mod(i,4) == 0
            [Q(:,(i-1)*b+1:i*b), Q(:,(i-2)*b+1:(i-1)*b)] = part_reorth(Q,i,b);
        end
        Q(:,(i-1)*b+1:i*b) = loc_reorth(Q(:,(i-1)*b+1:i*b), Q(:,(i-2)*b+1:(i-1)*b));
        Qi = Q(:,(i-1)*b+1:i*b);
        U = A*Qi - Q(:,(i-2)*b+1:(i-1)*b)*B.';
        M = Qi.'*U;
        R = U - Qi*M;
        [Qi,B] = qr(R,0);
        T((i-1)*b+1:i*b,(i-1)*b+1:i*b) = M;
        if i > k
            [V,D,~] = svd(T);
            if norm(B*V(end-b+1:end,k)) < 1.0e-5
                break;
            end
        end
        T(i*b+1:(i+1)*b,(i-1)*b+1:i*b) = B;
        T((i-1)*b+1:i*b,i*b+1:(i+1)*b) = B.';
        i = i + 1;
    end
    disp(i)
    D = diag(D);
    D = D(1:k);
    %V = Q*V(:,1:k);
end


% orthogonalize U1 against U2
function V  = loc_reorth(U1,U2)
    temp = U2.'*U1;
    V = U1 - U2*temp;
    [V,~] = qr(V,0);
end


% orthogonalize the two latest blocks against all the previous
function [V1,V2] = part_reorth(U,i,b)
    V1 = U(:,(i-1)*b+1:i*b);
    V2 = U(:,(i-2)*b+1:(i-1)*b);
    for j=1:i-2
        Uj = U(:,(j-1)*b+1:j*b);
        temp = Uj.'*V1;
        V1 = V1 - Uj*temp;
        temp = Uj.'*V2;
        V2 = V2 - Uj*temp;
    end
end