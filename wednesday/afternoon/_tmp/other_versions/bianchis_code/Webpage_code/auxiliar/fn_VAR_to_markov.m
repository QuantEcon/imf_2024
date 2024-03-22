function [Pr_mat,Pr_mat_key,zbar,Pr_mat_key_pos] =...
    fn_VAR_to_markov(A0,A1,A2,SIGMA,N,random_draws,method_input)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% GENERALIZED MARKOV APPROXIMATIONS TO VAR PROCESSES
%
% This function converts a VAR to a discretized Markov process,
% generalizing the approach in Tauchen (1986) by allowing for more general
% var./cov. structure in the error term.  The required multivariate normal
% probabilities are calculated using a Monte Carlo-type technique
% implemented in the function qscmvnv.m, developed by and available on the
% website of Alan Genz: http://www.math.wsu.edu/faculty/genz/homepage.
%
% Original VAR: A0*Z(t) = A1 + A2*Z(t-1) + e(t), e(t) ~ N(0,SIGMA)
%
% INPUTS:
% 1 - A0, A1, A2 are the VAR coefficients, as indicated above, with
%     A0 assumed non-singular
% 2 - N is n x 1, where n = # of vars, N(i) = # grid points for ith var.
% 3 - SIGMA is the arbitrary positive semi-definite error var./cov. matrix
% 4 - random_draws is the number of random draws used in the required
%     Monte Carlo-type integration of the multivariate normal (note: set a
%     rand seed prior to calling this function)
% 5 - method_input is _POTENTIALLY_ a 2x1 vector
%       - method_input=[1] follows the grid spacing strategy 
%         proposed in Tauchen (1986), a uniformly spaced grid,
%         covering 2 std. dev. of the relevant component variables.
%       - method_input=[1;m] constructs a uniformly spaced grid as in Tauchen
%         (1986), but it covers m std. dev. of the relevant component
%         variables.
%       - method=[2] selects grid points based on approximately equal
%         weighting from the UNIVARIATE normal cdf.  This method is adapted
%         from code written by Jonathan Willis.  (Note that method = 2
%         requires the use of the MATLAB statistics toolbox.)
%
% OUTPUTS:
% 1 - Pr_mat is the Prod(N) x Prod(N) computed transition probability matrix
%     for the discretized Markov chain
% 2 - Pr_mat_key is n x Prod(N) matrix s.t. if Z* is the discretized Markov
%     approximation to the VAR Z, then Z*(state i) = Pr_mat_key(:,i)
% 3 - zbar is the n x max(N) matrix st. zbar(i,1:N(i)) is the univariate
%     grid for the ith component of Z*
% 4 - Pr_mat_key_pos is the n x Prod(N) matrix containing the discretized
%     grid point positions of each state in Pr_mat_key
%
% HISTORY
% 03/17/08 - Stephen Terry, function adapted from earlier code by Ed Knotek
% 03/18/08 - Stephen Terry, "method=2" capability added, adapting earlier
%            functions from Jonathan Willis
% 03/21/08 - Stephen Terry, rounding correction and zbar output added
% 03/26/08 - Ed Knotek, unified the multiple functions into one; rewrote 
%            the qsmcvnv.m code for speed and renamed it qsmcvnv_fast.m
% 04/07/08 - Ed Knotek, embedded all functions and fully integrated the
%            qsmcvnv.m code for maximum speed
% 08/04/10 - Stephen Terry, added Pr_mat_key_pos output
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = size(N,1);  %number of variables in VAR

%compute reduced form parameters & steady-state mean
A1bar = inv(A0)*A1;
A2bar = inv(A0)*A2;
SIGMAbar = inv(A0)*SIGMA*(inv(A0)');

sstate_mean = inv(eye(n)-A2bar)*A1bar;

method=method_input(1);
if length(method_input)>1
  m=method_input(2);
else
  m=2;  %default number of std. deviations of the VAR covered by grid
end;

%iterate to obtain var./cov. structure of the PROCESS (not error term)
SIGMAprocess = SIGMAbar;
SIGMAprocess_last = SIGMAprocess;
dif = 1;
while dif>0.00000001
    SIGMAprocess = A2bar*SIGMAprocess_last*(A2bar') + SIGMAbar;
    dif = max(max(SIGMAprocess-SIGMAprocess_last));
    SIGMAprocess_last = SIGMAprocess;
end;

%This block equally spaces grid points bounded by m*(std.deviation of
%process) on either side of the unconditional mean of the process.  Any
%more sophisticated spacing of the grid points could be implemented by
%changing the definition of zbar below.
zbar = zeros(n,max(N)); 
grid_stdev = diag(SIGMAprocess).^0.5;
if method==1;
    grid_increment = zeros(n,1);
    for i = 1:n;
         grid_increment(i) = 2*m*grid_stdev(i)/(N(i)-1);
         zbar(i,1) = -m*grid_stdev(i) + sstate_mean(i);
        for j = 1:N(i)-1;
            zbar(i,j+1) = zbar(i,j) + grid_increment(i);
        end;
    end;
elseif method==2;
    d = zeros(n,max(N));
    b = -4:.005:4;
    c = normcdf(b,0,1);
    for i = 1:n;
        a = (1/(2*N(i))):(1/N(i)):1;
        for j = 1:N(i);
            [d1,d(i,j)] = min((a(j)-c).^2);
        end;
        zbar(i,1:N(i)) = grid_stdev(i)*b(d(i,:))+sstate_mean(i);
    end;
end;

%compute key matrix & pos matrix
Pr_mat_key = zeros(length(N),prod(N));
Pr_mat_key_pos = zeros(length(N),prod(N));
Pr_mat_key(length(N),:) = repmat(zbar(length(N),1:N(length(N))),[1 prod(N)/N(length(N))]);
Pr_mat_key_pos(length(N),:) = repmat(1:N(length(N)),[1 prod(N)/N(length(N))]);
for i=length(N)-1:-1:1;
    Pr_mat_key(i,:) = repmat(kron(zbar(i,1:N(i)),ones(1,prod(N(i+1:length(N))))),[1 prod(N)/prod(N(i:length(N)))]);
    Pr_mat_key_pos(i,:) = repmat(kron(1:N(i),ones(1,prod(N(i+1:length(N))))),[1 prod(N)/prod(N(i:length(N)))]);
end;

nstate = prod(N);
Pr_mat_intervals = zeros(n,nstate,2);   %this will store the unadjusted limits of integration for each variable in each state, for input into the Genz code
if method==1;
    for i = 1:nstate;  %number of states
        for j = 1:n;    %number of variables
            if Pr_mat_key_pos(j,i)==1;
                Pr_mat_intervals(j,i,1) = -inf;
                Pr_mat_intervals(j,i,2) = zbar(j,Pr_mat_key_pos(j,i)) + (grid_increment(j)/2);
            elseif Pr_mat_key_pos(j,i)==N(j);
                Pr_mat_intervals(j,i,1) = zbar(j,Pr_mat_key_pos(j,i)) - (grid_increment(j)/2);
                Pr_mat_intervals(j,i,2) = inf;
            else
                Pr_mat_intervals(j,i,1) = zbar(j,Pr_mat_key_pos(j,i)) - (grid_increment(j)/2);
                Pr_mat_intervals(j,i,2) = zbar(j,Pr_mat_key_pos(j,i)) + (grid_increment(j)/2);
            end;
        end;
    end;
elseif method==2;
    for i = 1:nstate;  %number of states
        for j = 1:n;    %number of variables
            if Pr_mat_key_pos(j,i)==1;
                Pr_mat_intervals(j,i,1) = -inf;
                Pr_mat_intervals(j,i,2) = zbar(j,Pr_mat_key_pos(j,i)) + (zbar(j,Pr_mat_key_pos(j,i)+1)-zbar(j,Pr_mat_key_pos(j,i)))/2;
            elseif Pr_mat_key_pos(j,i)==N(j);
                Pr_mat_intervals(j,i,1) = zbar(j,Pr_mat_key_pos(j,i)) - (zbar(j,Pr_mat_key_pos(j,i))-zbar(j,Pr_mat_key_pos(j,i)-1))/2;
                Pr_mat_intervals(j,i,2) = inf;
            else
                Pr_mat_intervals(j,i,1) = zbar(j,Pr_mat_key_pos(j,i)) - (zbar(j,Pr_mat_key_pos(j,i))-zbar(j,Pr_mat_key_pos(j,i)-1))/2;
                Pr_mat_intervals(j,i,2) = zbar(j,Pr_mat_key_pos(j,i)) + (zbar(j,Pr_mat_key_pos(j,i)+1)-zbar(j,Pr_mat_key_pos(j,i)))/2;
            end;
        end;
    end;
end;

error_est = zeros(nstate,nstate);
Pr_mat_intervals_adjusted = zeros(n,nstate,2);
Pr_mat = zeros(nstate,nstate);
n_keep=n;
for i_keep = 1:nstate; %rows of Pr_mat
    Pr_mat_intervals_adjusted(:,:,1) = Pr_mat_intervals(:,:,1) - repmat((A1bar + A2bar*Pr_mat_key(:,i_keep)),1,nstate);
    Pr_mat_intervals_adjusted(:,:,2) = Pr_mat_intervals(:,:,2) - repmat((A1bar + A2bar*Pr_mat_key(:,i_keep)),1,nstate);
    for j_keep = 1:nstate;   %columns of Pr_mat
        %Pr_mat(i,j) = P(state j|state i)
%          [Pr_mat(i_keep,j_keep), error_est(i_keep,j_keep)] = ...
%              qscmvnv(random_draws,SIGMAbar,Pr_mat_intervals_adjusted(:,j_keep,1),eye(n),Pr_mat_intervals_adjusted(:,j_keep,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Embed function in code
c=SIGMAbar; 
a=Pr_mat_intervals_adjusted(:,j_keep,1); 
cn=eye(n_keep); 
b=Pr_mat_intervals_adjusted(:,j_keep,2);
%  Computes permuted lower Cholesky factor ch for covariance SIGMAbar which 
%   may be singular, combined with contraints a < cn*x < b, to
%   form revised lower triangular constraint set ap < ch*x < bp; 
%   clg contains information about structure of ch: clg(1) rows for 
%   ch with 1 nonzero, ..., clg(np) rows with np nonzeros.
%
ep = 1e-10; % singularity tolerance;
%
[m1 n] = size(cn); ch = cn; np = 0;
ap = a; bp = b; y = zeros(n,1); sqtp = sqrt(2*pi);
d = sqrt(max(diag(c),0));
for i = 1 : n, di = d(i);
  if di > 0
    c(:,i) = c(:,i)/di; c(i,:) = c(i,:)/di; ch(:,i) = ch(:,i)*di;
  end
end;
%
% determine (with pivoting) Cholesky factor for SIGMAbar 
%  and form revised constraint matrix ch
%
clg=zeros(n,1);
for i = 1 : n
  epi = ep*i^2; j = i; 
  for l = i+1 : n, if c(l,l) > c(j,j), j = l; end, end
  if j > i
    t = c(i,i); c(i,i) = c(j,j); c(j,j) = t;
    t = c(i,1:i-1); c(i,1:i-1) = c(j,1:i-1); c(j,1:i-1) = t;
    t = c(i+1:j-1,i); c(i+1:j-1,i) = c(j,i+1:j-1)'; c(j,i+1:j-1) = t';
    t = c(j+1:n,i); c(j+1:n,i) = c(j+1:n,j); c(j+1:n,j) = t;
    t = ch(:,i); ch(:,i) = ch(:,j); ch(:,j) = t;
  end
  if c(i,i) < epi, break, end, cvd = sqrt( c(i,i) ); c(i,i) = cvd;
  for l = i+1 : n
    c(l,i) = c(l,i)/cvd; c(l,i+1:l) = c(l,i+1:l) - c(l,i)*c(i+1:l,i)';
  end
  ch(:,i) = ch(:,i:n)*c(i:n,i); np = np + 1;
end
%
% use right reflectors to reduce ch to lower triangular
%
for i = 1 : min( np-1, m1 )
  epi = ep*i*i; vm = 1; lm = i;
  %
  % permute rows so that smallest variance variables are first.
  %
  v=ch(i:m1,1:np); 
  s=v(:,1:i-1)*y(1:i-1);
  ss=max(sqrt(sum(v(:,i:np).^2,2)),epi);
  al=(ap-s)./ss;
  bl=(bp-s)./ss;
  dna=zeros(m1-i+1,1);
  dsa=zeros(m1-i+1,1);
  dnb=zeros(m1-i+1,1);
  dsb=ones(m1-i+1,1);
  ind1=(al>-9);
  dna(ind1)=exp(-al(ind1).*al(ind1)/2)/sqtp;
  ind2=(bl<9);
  dnb(ind2)=exp(-bl(ind2).*bl(ind2)/2)/sqtp;
  ds=erfc(-([al(ind1); bl(ind2)])/sqrt(2))/2;
  dsa(ind1)=ds(1:length(al(ind1)));
  dsb(ind2)=ds(1+length(al(ind1)):length(ds));
  mn=zeros(m1-i+1,1);
  vr=zeros(m1-i+1,1);
  ind3=(dsb-dsa>epi);
  ind4=(al<=-9);
  ind5=(bl>=9);  
  mn(ind3&ind4)=-dnb(ind3&ind4);
  vr(ind3&ind4)=-bl(ind3&ind4).*dnb(ind3&ind4);
  mn(ind3&ind5)=dna(ind3&ind5);
  vr(ind3&ind5)=al(ind3&ind5).*dna(ind3&ind5);
  mn(ind3&~ind4&~ind5)=dna(ind3&~ind4&~ind5)-dnb(ind3&~ind4&~ind5);
  vr(ind3&~ind4&~ind5)=al(ind3&~ind4&~ind5).*dna(ind3&~ind4&~ind5)-...
      bl(ind3&~ind4&~ind5).*dnb(ind3&~ind4&~ind5);
  mn(ind3)=mn(ind3)./(dsb(ind3)-dsa(ind3));
  vr(ind3)=1+vr(ind3)./(dsb(ind3)-dsa(ind3))-mn(ind3).^2;
  mn(~ind3&ind4)=bl(~ind3&ind4);
  mn(~ind3&ind5)=al(~ind3&ind5);
  mn(~ind3&~ind4&~ind5)=(al(~ind3&~ind4&~ind5)+bl(~ind3&~ind4&~ind5))/2;
  for l=i:m1;
      if vr(l)<=vm;
          lm=l; vm=vr(l); y(i)=mn(l);
      end;
  end;
  v = ch(lm,1:np);
  if lm > i 
    ch(lm,1:np) = ch(i,1:np); ch(i,1:np) = v;
    tl = ap(i); ap(i) = ap(lm); ap(lm) = tl;
    tl = bp(i); bp(i) = bp(lm); bp(lm) = tl;
  end
  ch(i,i+1:np) = 0; ss = sum( v(i+1:np).^2 );
  if ( ss > epi )
    ss = sqrt( ss + v(i)^2 ); if v(i) < 0, ss = -ss; end
    ch(i,i) = -ss; v(i) = v(i) + ss; vt = v(i:np)'/( ss*v(i) );
    ch(i+1:m1,i:np) = ch(i+1:m1,i:np) - ch(i+1:m1,i:np)*vt*v(i:np); 
  end
end
%
% scale and sort constraints
%
clm=zeros(m1,1);
for i = 1 : m1
  v = ch(i,1:np); clm(i) = min(i,np); 
  jm = 1; for j = 1 : clm(i), if abs(v(j)) > ep*j*j, jm = j; end, end 
  if jm < np, v(jm+1:np) = 0; end, clg(jm) = clg(jm) + 1; 
  at = ap(i); bt = bp(i); j = i;
  for l = i-1 : -1 : 1
    if jm >= clm(l), break, end
    ch(l+1,1:np) = ch(l,1:np); j = l;
    ap(l+1) = ap(l); bp(l+1) = bp(l); clm(l+1) = clm(l);
  end
  clm(j) = jm; vjm = v(jm); ch(j,1:np) = v/vjm; 
  ap(j) = at/vjm; bp(j) = bt/vjm;
  if vjm < 0, tl = ap(j); ap(j) = bp(j); bp(j) = tl; end
end
j = 0; for i = 1 : np, if clg(i) > 0, j = i; end, end, np = j;
%
% combine constraints for first variable
%
if clg(1) > 1 
  ap(1) = max( ap(1:clg(1)) ); bp(1) = max( ap(1), min( bp(1:clg(1)) ) ); 
  ap(2:m1-clg(1)+1) = ap(clg(1)+1:m1); bp(2:m1-clg(1)+1) = bp(clg(1)+1:m1);
  ch(2:m1-clg(1)+1,:) = ch(clg(1)+1:m1,:); 
  clg(1) = 1;
end
%
% end chlsrt
%
as=ap; bs=bp; n=np;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cci=erfc(-([as(1);bs(1)])/sqrt(2))/2; ci=cci(1); dci=cci(2)-ci; p=0; e=0;
ns = 8; nv = fix( max( [random_draws/(2*ns) 1] ) ); 
q = 2.^( [1:n-1]'/n) ; % Niederreiter point set generators
%
% Randomization loop for ns samples
%
on=ones(ns,nv);
on1=ones(1,nv*ns);
xx=reshape(abs(2*mod(repmat(q,ns,1)*[1:nv]+rand((n-1)*ns,1)*ones(1,nv),1)-1),n-1,ns,nv);
xx=permute(xx,[2 3 1]);
mv_y=zeros(ns,nv,n-1);
mv_c=ci*on;
mv_dc=dci*on;
mv_p=mv_dc;
li=2;
lf=1;
mv_y1=zeros(ns,nv,n-1);
xx1=1-xx;
mv_c1=ci*on;
mv_dc1=dci*on;
mv_p1=mv_dc1;
for i1=2:n;
    yyy=zeros(ns,nv,2);
    yyy(:,:,1)=mv_c+xx(:,:,i1-1).*mv_dc;
    yyy(:,:,2)=mv_c1+xx1(:,:,i1-1).*mv_dc1;
    mv_yy=-sqrt(2)*erfcinv(2*[yyy]);
    lf=lf+clg(i1);
    mv_y(:,:,i1-1)=mv_yy(:,:,1);
    mv_y1(:,:,i1-1)=mv_yy(:,:,2);
    if lf<li;
        mv_c=0;
        mv_dc=1;
        mv_c1=0;
        mv_dc1=1;
    else;
        mv_y=permute(mv_y,[2,1,3]);
        mv_y1=permute(mv_y1,[2,1,3]);
        mv_s=ch(li:lf,1:i1-1)*reshape(mv_y,nv*ns,1)';
        mv_s1=ch(li:lf,1:i1-1)*reshape(mv_y1,nv*ns,1)';
        mv_ai=max(max(as(li:lf)*on1-mv_s,[],1),-9);
        mv_ai1=max(max(as(li:lf)*on1-mv_s1,[],1),-9);
        mv_bi=max(mv_ai,min(min(bs(li:lf)*on1-mv_s,[],1),9)); 
        mv_bi1=max(mv_ai1,min(min(bs(li:lf)*on1-mv_s1,[],1),9));        
        mv_cc=erfc(-[mv_ai;mv_ai1;mv_bi;mv_bi1]/sqrt(2))/2;
        mv_c=reshape(mv_cc(1,:),nv,ns)'; 
        mv_c1=reshape(mv_cc(2,:),nv,ns)'; 
        mv_dc=reshape(mv_cc(3,:),nv,ns)'-mv_c; 
        mv_dc1=reshape(mv_cc(4,:),nv,ns)'-mv_c1; 
        mv_p=mv_p.*mv_dc; 
        mv_p1=mv_p1.*mv_dc1;
    end; 
    li=li+clg(i1);
end;
vp1=mv_p; 
vp2=mv_p1;
vp=(vp1+vp2)/2;
md=mean(vp,2);
for i=1:ns;
    d=(md(i)-p)./i; p=p+d;
    if abs(d)>0; e=abs(d)*sqrt(1+(e/d)^2*(i-1)/i); 
    else; if i>1; e=e*sqrt((i-2)/i); end;
    end;
end;
%
e = 3*e; % error estimate is 3 x standard error with ns samples.
%
% end qscmvnv_fast
%

Pr_mat(i_keep,j_keep)=p;
error_est(i_keep,j_keep)=e;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end;
end;

%rounding error adjustment
round_sum = sum(Pr_mat,2);
for i = 1:size(Pr_mat,2);
    Pr_mat(i,:) = Pr_mat(i,:)/round_sum(i);
end