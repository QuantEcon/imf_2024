function [g,T] = var_markov(x,Px,y,Py,varargin)
% This function computes the grid and transition matrix of two discrete
% processes (x and y)
%
% Damian Romero
% November, 2016

% Length of each shock
Nx = length(x);
Ny = length(y);

g = gridmake(x,y);
T = repmat(Px,Ny,Ny).*kron(Py,ones(Nx));

if nargin>4
    % Variance decomposition
    S = varargin{:};
    [Q,~] = eig(S);
    
    % Modified grid
    g = g*Q';
end