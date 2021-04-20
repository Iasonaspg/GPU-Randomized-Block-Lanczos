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
    Q0 = randn(n,b);
    [Q1,~] = qr(A*Q0,0);
   
    % first loop
    Q(:,1:b) = Q1;
    U = A*Q1;
    M = Q1.'*U;
    R = U - Q1*M;
    Q0 = Q1;
    [Q1,B] = qr(R,0);
    T(1:b,1:b) = M;
    T(b+1:2*b,1:b) = B;
    T(1:b,b+1:2*b) = B.';
    i = 2;
    while true
        Q(:,(i-1)*b+1:i*b) = Q1;
        [Q(:,(i-2)*b+1:i*b),~] = qr( Q(:,(i-2)*b+1:i*b),0 );
        if mod(i,2) == 0   
            [Q(:,1:i*b),~] = qr( Q(:,1:i*b),0 );
        end
        U = A*Q1 - Q0*B.';
        M = Q1.'*U;
        R = U - Q1*M;
        Q0 = Q1;
        [Q1,B] = qr(R,0);
        T((i-1)*b+1:i*b,(i-1)*b+1:i*b) = M;
        if i > k
        [V,D,~] = svd(T);
            if norm(B*V(end-b+1:end,k)) < 1.0e-2
                break;
            end
        end
        T(i*b+1:(i+1)*b,(i-1)*b+1:i*b) = B;
        T((i-1)*b+1:i*b,i*b+1:(i+1)*b) = B.';
        i = i + 1;
    end
    D = diag(D);
    D = D(1:k);
    V = Q*V(:,1:k);
end