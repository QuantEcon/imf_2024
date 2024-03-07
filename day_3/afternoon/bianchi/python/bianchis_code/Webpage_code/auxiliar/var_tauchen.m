function [Z,Zprob] = var_tauchen(N,A,vcv,m)

%Purpose:    Finds a Markov chain whose sample paths
%            approximate those of the VAR(1) process
%                z(t+1) = rho * z(t) + eps(t+1)
%            where eps are normal with vcv matrix
%
%Format:     {Z, Zprob} = Tauchen(N,mu,rho,sigma,m)
%
%Input:      N       vector, number of nodes for Z
%            A       matrix of autocorrelations
%            vcv     matrix, var-cov matrix of epsilons
%            m       max +- std. devs.
%
%Output:     Z       N*1 vector, nodes for Z
%            Zprob   N*N matrix, transition probabilities
%
%    Sofia Bauducco
%    December 2013
%
%    This procedure is an implementation of George Tauchen's algorithm
%    described in Fabrizio Perri's notes on numerical methods for income
%    fluctuation problems, for bivariate processes (for higher-dimensional 
%    processes, check computation of variance in general form).
%

%% Diagonalization of var-cov matrix of epsilons

n_proc = 2;

[Q,Lambda] = eig(vcv);

A_tilde = Q'*A*Q;

%% Computation of matrix of var-cov of transformed process

vcv_ytilde = vcv_var(A_tilde,Lambda);

std_ytilde = [vcv_ytilde(1,1).^0.5; vcv_ytilde(2,2).^0.5];

%% Approximation of the diagonalized system

for i=1:n_proc
    
    [Z_tilde(:,1,i),Z_step(i,1)] = step_tauchen(N(i,1),std_ytilde(i,1),m(i,1));
    
end

aux_Z_tilde1 = kron(ones(N(2,1),1),Z_tilde(:,1,1));
aux_Z_tilde2 = kron(Z_tilde(:,1,2),ones(N(1,1),1));

Z_tilde = [aux_Z_tilde1 aux_Z_tilde2];

Zprob = tauchen_mult(N(1,1)*N(2,1),A_tilde,Lambda,Z_tilde,Z_step);

%% Computation of original process

Q_Z1_aux = repmat(Q(1,:),size(Z_tilde,1),1);
aux_Z1 = Q_Z1_aux.*Z_tilde;
Z1 = sum(aux_Z1,2);

Q_Z2_aux = repmat(Q(2,:),size(Z_tilde,1),1);
aux_Z2 = Q_Z2_aux.*Z_tilde;
Z2 = sum(aux_Z2,2);

Z = [Z1 Z2];

%% Additional functions


% function vcv = vcv_var(A,vcv_eps)
% 
% a11 = A(1,1);
% a12 = A(1,2);
% a21 = A(2,1);
% a22 = A(2,2);
% 
% aux_mat1 = [a11.^2 2.*a11.*a12 a12.^2; a11.*a21 a12.*a21 + a22.*a11 a12.* a22; a21.^2 2.*a21.*a22 a22.^2];
% aux_mat2 = eye(3);
% 
% aux_mat = aux_mat2-aux_mat1;
% 
% vcv_aux1 = [vcv_eps(1,1) vcv_eps(1,2) vcv_eps(2,2)]';
% 
% vcv_aux = inv(aux_mat)*vcv_aux1;
% 
% vcv = [vcv_aux(1,1) vcv_aux(2,1);vcv_aux(2,1) vcv_aux(3,1)];

function vcv = vcv_var(A,vcv_eps)

vcv_guess = eye(size(vcv_eps,1));

d = 1;
maxiter= 10000;

for i=1:maxiter
    
    vcv_guess_new = A*vcv_guess*A' + vcv_eps;
    d = max(max(abs(vcv_guess_new-vcv_guess)));
    
    vcv_guess = vcv_guess_new;
    
    if d < 1e-5
        break;
    end
    
end

vcv = vcv_guess;


function [Z,zstep] = step_tauchen(N,sigma,m)

Z     = zeros(N,1);

Z(N)  = m * sigma;
Z(1)  = -Z(N);
zstep = (Z(N) - Z(1)) / (N - 1);

for i=2:(N-1)
    Z(i) = Z(1) + zstep * (i - 1);
end 

function Zprob = tauchen_mult(N,A,sigma,Z,Z_step) 

Z1_step = Z_step(1,1);
Z2_step = Z_step(2,1);

Z1_min = Z(1,1);
Z2_min = Z(1,2);
Z1_max = Z(end,1);
Z2_max = Z(end,2);

for j = 1:N
    for k = 1:N
        Z1_aux = Z(k,1);
        Z2_aux = Z(k,2);
        
        if Z1_aux == Z1_min
            Z1prob = cdf_normal((Z1_aux - A(1,1)*Z(j,1) - A(1,2)*Z(j,2) + Z1_step/2)/(sigma(1,1).^0.5));
        elseif Z1_aux == Z1_max
            Z1prob = 1 - cdf_normal((Z1_aux - A(1,1)*Z(j,1) - A(1,2)*Z(j,2) - Z1_step/2)/(sigma(1,1).^0.5));
        else
            Z1prob = cdf_normal((Z1_aux - A(1,1)*Z(j,1) - A(1,2)*Z(j,2) + Z1_step/2)/(sigma(1,1).^0.5)) ...
                - cdf_normal((Z1_aux - A(1,1)*Z(j,1) - A(1,2)*Z(j,2) - Z1_step/2)/(sigma(1,1).^0.5));
        end
        
        if Z2_aux == Z2_min
            Z2prob = cdf_normal((Z2_aux - A(2,1)*Z(j,1) - A(2,2)*Z(j,2) + Z2_step/2)/(sigma(2,2).^0.5));
        elseif Z2_aux == Z2_max
            Z2prob = 1 - cdf_normal((Z2_aux - A(2,1)*Z(j,1) - A(2,2)*Z(j,2) - Z2_step/2)/(sigma(2,2).^0.5));
        else
            Z2prob = cdf_normal((Z2_aux - A(2,1)*Z(j,1) - A(2,2)*Z(j,2) + Z2_step/2)/(sigma(2,2).^0.5)) ...
                - cdf_normal((Z2_aux - A(2,1)*Z(j,1) - A(2,2)*Z(j,2) - Z2_step/2)/(sigma(2,2).^0.5));
        end
        
        Zprob(j,k) = Z1prob*Z2prob;
        
    end        
end


function c = cdf_normal(x)
    c = 0.5 * erfc(-x/sqrt(2));

