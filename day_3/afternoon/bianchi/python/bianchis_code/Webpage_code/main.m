 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%                                                              %%%%%%
%%%%%%                           %%%%%%
%%%%%%                                                              %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Overborrowing and Systemic Externalities
% American Economic Review, 2011
% Javier Bianchi

%% 0. Housekeeping
clear all;
clc;
close all

addpath('auxiliar')

%% 1. Model Parameter Values

% MODEL PARAMETERS

sigma    = 2;
ita      = (1/0.83)-1; % elasticity parameter (elasticity is 0.83)
beta     = 0.906;       % discount factor
omega    = 0.3070;       % share for tradables
kappa    = 0.3235;
r        = 0.04;  R=1+r;

sep_prefs = 0;
if sep_prefs==1  % allows analytical solution of b*
    sigma =ita+1;  
end

NT_shock = 1;  % to use T-NT process for paper

str = sprintf('MODEL PARAMETERS: sigma %2.1f  elasticity %2.2f  beta %2.2f omega %2.2f kappa %2.2f  %r',sigma, 1/(1+ita),beta,omega,kappa,r); disp(str)

%%  2.  Numerical Parameters

if NT_shock ==1
    
    load proc_shock yT yN Prob 
    
    NSS =length(yT);
    
else
    
    yn          = 1   ;
    sigma_y     = 0.058; % stdev  yT
    rho         = 0.5;
    sigma_eps   = sqrt(sigma_y^2*(1-rho^2)); % stdev of innovation to yT
    
    NSS          = 9;
%     [Z,Prob]    = tauchenhussey(NSS,0,rho,sigma_eps ,sigma_eps );
     m=2;   
    [Z,Zprob] = tauchen(NSS,0,rho,sigma_eps,m);
    yT          = exp(Z);
    yN          = repmat(yn,NSS,1);
  
end

NB          = 100;
 

bmin_NDL    = -kappa*(1+min(yT(1)))+0.01;  % maximum debt consistent with feasibility
bmin        = -1.02;
bmax        = -0.2000;

B           = linspace(bmin,bmax,NB)';


SR          = ones(NB,NSS)*(1+r);
YT          = repmat(yT,1,NB )';
YN          = repmat(yN,1,NB )';
KAPPAS      = kappa*ones(NB,NSS);
b           = repmat(B,1,NSS);
 

% ininitialize
bp          = b;
c           = b.*SR  +YT  -bp;
 


price       = (1-omega)/omega*c.^(1+ita);
EE          = zeros(NB,NSS);
bmax_collat = -kappa*(price.*YN +YT);
cbind       = b.*SR  +YT  -bmax_collat;
emu         = zeros(NB,NSS)*nan;
Ibind       = ones(NB,NSS);
 
 
uptd        = 0.2;   % Change to 0.1 if policy function fluctuate
uptdSP      = 0.02;

outfreq     = 50;    % Display frequency
iter_tol    = 50000;   % Iteration tolerance
Tol_eulerDE = 1.0e-7;% when to look for unconstrained solution
Tol_eulerSP = 1.0e-7;
tol         = 1.0e-5; %Numerical tolerance


T   = 79000;  % Number of simulations
cut = 0;  % Initial condition dependence


options = optimset('Display','off');
str     = sprintf('tol %10.2e  bmin %2.2f bmax %2.2f  NB %3i NS %2i',tol,bmin,bmax,NB,NSS); disp(str)

disp(' ')
%% 3) Decentralized Equilibrium

iter = 0;
d2   = 100;

tic

figure('name','bp')

disp('DE Iter      Norm         Updt');

while d2>tol && iter <iter_tol
    
    oldp   = price;
    oldc   = c;
    oldbp  = bp;
    
    totalc = (omega*c.^(-ita)+(1-omega)*YN.^(-ita));
    mup    = real(omega*totalc.^(sigma/ita).*(totalc.^(-1/ita-1)).*(c.^(-ita-1)));  % marginal utility today
    
          if sep_prefs==1
                 mup    = omega*c.^(-sigma);
          end
    
    for i=1:NB
        for j=1:NSS
            emu(i,j) = beta*(SR(i,j))*interp1(B,mup,bp(i,j),'linear','extrap')*Prob(j,:)'; %EMU is expected marginal utility tomorrow in today's grid.
 
        end
    end
    
    
    for i=1:NB
        for j=1:NSS
            EE(i,j) = (omega*cbind(i,j)^(-ita)+(1-omega)*YN(i,j)^(-ita))^(sigma/ita -1/ita -1 )*omega*cbind(i,j)^(-ita-1)-emu(i,j); 
           
            
            if EE(i,j)>Tol_eulerDE     % BORROWING CONSTRAINT BINDS
                bp(i,j)    = bmax_collat(i,j);
                c(i,j)     = cbind(i,j);
                Ibind(i,j) = 1; 
      
            else                        % EULER EQUATION
                
                
                if sep_prefs ==0   
                    f                = @(cc) (omega*cc^(-ita)+(1-omega)*YN(i,j)^(-ita))^(sigma/ita -1/ita -1)*omega*cc^(-ita-1)-emu(i,j);
                    c0               = c(i,j);
 
                    [c(i,j),EE(i,j)] = fzero(f,c0,options);
                else
                    c(i,j)           = (emu(i,j)/omega) ^(-1/sigma);
                    
                    EE(i,j)          = 0;
                end
               
                Ibind(i,j)       = 0;
            end
                  
            
        end
    end
    
    bp    = SR.*b+YT-c;
    bp    = min(bmax,max(bp,bmin));
    
    price = (1-omega)/omega*(c./YN).^(1+ita);
    
    c     = SR.*b+YT-max(bp,-KAPPAS.*(price.*YN+YT)); % update consumption based on update for pN
    
    d2    = max([max(max(abs(c-oldc))),max(max(abs(bp-oldbp)))]); % metric
    
    bmax_collat                   = -KAPPAS.*(price.*YN+YT);
    bmax_collat(bmax_collat>bmax) = bmax;
    bmax_collat(bmax_collat<bmin) = bmin;
    cbind                         = SR.*b+YT - bmax_collat;
    cbind(cbind<0)                = nan;
    
    
    %=====================Updating rule.Must be slow, important==============
    bp    = uptd*bp+(1-uptd)*oldbp;
    c     = uptd*c+(1-uptd)*oldc;
 
    %========================================================================
    

    iter              = iter+1;
    
    D2(iter)          = d2;
    
    if mod(iter, outfreq)==0 | iter==1
     fprintf('%3i          %10.2e   %2.2f \n',iter,d2, uptd);
       plot(B,bp); xlabel('B'); ylabel('B''');  drawnow
    end
end

     fprintf('No. Iterations: %3i  Metric  %10.2e \n',iter,d2 );

LagrangeDE            = EE;
LagrangeDE(Ibind ==0) = 0;

plots_de
 
bpDE             = bp;
cDE              = c;
totalcDE         = totalc;
priceDE          = price;
IbindDE          = Ibind;
clear bp c totalc  price mup Ibind
%% 4) Planner problem

bp         = bpDE;
c          = cDE;
psi        = (1-omega)/omega*(ita+1)*KAPPAS.*(c./YN).^(ita);
price      = priceDE;
LagrangeSP = LagrangeDE./(1-psi);  % guess for SP
IbindSP    = IbindDE;

EESP       = zeros(NB,NSS)*nan;

uptd = uptdSP;
d2   = 100;
iter = 0;
disp('SPP Iter      Norm       Uptd');

while d2>tol && iter <iter_tol*2
    
 
    oldc          = c;
    oldbp         = bp;
    oldLagrangeSP = LagrangeSP;
    
    totalc        = (omega*c.^(-ita)+(1-omega)*YN.^(-ita));
    mup           = (omega*totalc.^(sigma/ita).*(totalc.^(-1/ita-1)).*(c.^(-ita-1)));  %mup(b,y)
    
          if sep_prefs==1
                 mup    = omega*c.^(-sigma);
          end
    
    psi      = (1-omega)/omega*(ita+1)*KAPPAS.*(c./YN).^(ita);
    LFH_SP   = mup+LagrangeSP.*psi;
    
    for i=1:NB
        for j=1:NSS
            RHS_SP(i,j) = beta*SR(i,j)*interp1(B,LFH_SP,bp(i,j),'linear','extrap')*Prob(j,:)'; %Expected marginal utility tomorrow in today's grid.
        end
    end
    
    LagrangeSP = LFH_SP-beta*SR.*RHS_SP; % THIS IS EQUATION u_T +mu*psi= beta*R E [(u_T(t+1)+mu_t+1 ]+mu_t
    
    LagrangeSP(IbindSP ==0) = 0;
    
    for i=1:NB
        for j=1:NSS
       
            if LagrangeSP(i,j)>=Tol_eulerSP          % BORROWING CONSTRAINT BINDS
        
                c(i,j) = cbind(i,j);
                IbindSP (i,j) = 1;
                EESP(i,j)= LagrangeSP(i,j);
            else
                 
                if sep_prefs==0                           % EULER EQUATION   
                    f          = @(cc) (omega*cc^(-ita)+(1-omega)*YN(i,j)^(-ita))^(sigma/ita-1/ita-1)*omega*cc^(-ita-1)-RHS_SP(i,j);
                    x0         = c(i,j);
                    [c(i,j),EESP(i,j)] = fzero(f,c0,options);
                else
                    c(i,j)=   (RHS_SP(i,j)/omega) ^(-1/sigma);
                    EESP(i,j)  = 0;
                end
  
                IbindSP (i,j) = 0;
           
            end
        end
    end
    bp          = (SR.*b+YT-c);
    bp(bp>bmax) = bmax;
    bp(bp<bmin) = bmin;
    price       = ((1-omega)/omega*(c./YN).^(1+ita));
    
    %===============Check collateral constraint==============================
    c           = SR.*b+YT-max(bp,-KAPPAS.*(price.*YN+YT));
    price       = ((1-omega)/omega*(c./YN).^(1+ita));
    bp          = (SR.*b+YT-c);
    %========================================================================
 
    d2          = max(max(abs(c-oldc)));
    
  
 
    
    %=====================Updating rule.Must be slow, important==============
    bp          = uptd*bp+(1-uptd)*oldbp;
    c           = uptd*c+(1-uptd)*oldc;
    %========================================================================
    
    bmax_collat = -KAPPAS.*(price.*YN+YT);
    cbind       = SR.*b+YT-bmax_collat;
    
    iter=iter+1;
   if mod(iter, outfreq) == 0 | iter==1
        fprintf('%3i          %10.2e   %2.2f \n',iter,d2, uptd);
      
   end
 
end

    fprintf('No. Iterations: %3i  Metric  %10.2e \n ',iter,d2);


toc
totalcSP = totalc;
 mupSP    = mup;
 bpSP     = bp;
 priceSP  = price;
 cSP      = c;

clear bp c totalc  price mup
%% 5. Welfare Calculation
V    = zeros(NB,NSS);
VSP  = zeros(NB,NSS);

U    = (totalcDE.^(-1/ita)).^(1-sigma)./(1-sigma);
USP  = (totalcSP.^(-1/ita)).^(1-sigma)./(1-sigma);

EV   = U;
EVSP = USP;

d3   = 100;
iter = 0;
disp('Value Iter     Norm');
while d3>tol && iter <iter_tol
    for i=1:NB
        for j=1:NSS
            EV(i,j)   = beta*interp1(B,V,bpDE(i,j),'linear','extrap')*Prob(j,:)'; %EV is expected value tomorrow in today's grid.
            EVSP(i,j) = beta*interp1(B,VSP,bpSP(i,j),'linear','extrap')*Prob(j,:)';
        end
    end
    
    V_new   = U+EV;
    VSP_new = USP+EVSP;
    
    d3      = max([max(max(abs((V_new-V)./V))),max(max(abs((VSP_new-VSP)./VSP)))]);
    
    iter    = iter+1;
    
    V       = V_new;
    VSP     = VSP_new;
    if mod(iter, outfreq) == 0
        fprintf('%d          %1.7f \n',iter,d3);
    end
    
end
Wf=(VSP./V).^(1/(1-sigma))-1;

%% 6. Optimal Tax
 
% mupSP    = (omega*totalcSP.^(sigma/ita).*(totalcSP.^(-1/ita-1)).*(cSP.^(-ita-1))); %mupSP-EESP.*psi;
for i=1:NB
    for j=1:NSS
        emuSP(i,j) = beta*(SR(i,j))*interp1(B,mupSP,bpSP(i,j),'linear','extrap')*Prob(j,:)';
    end
end
tau           = mupSP./emuSP-1;


% tau2= 

tau(IbindSP ==1) = 0;
tau(tau<0)    = 0;

figure('name','tax as a function of debt')
plot(B,tau); legend('1','2','3','4','5','6','7','8','9')
xlabel('debt')
figure('name','tax as a function of income')
for IB=1:10:50
    plot(yT,tau(IB,:)); xlabel('income'); title('tax');     hold on
end

plots_comp

 

%% 7. Simulation

% load shocks_initials_aer.mat  b0_DE      b0_SP      state_sim
% S_index=state_sim;
disp('simulating the model')
% cut=0;
b0_DE     = -0.9443;
b0_SP     = -0.9443;
[S_index] = markov(Prob,T+cut+1,1,1:length(Prob));

% SHOCKS
ySIM     = YT(1,S_index)';
yNSIM    = YN(1,S_index)';
RSIM     = SR(1,S_index)';
KAPPASIM = KAPPAS(1,S_index)';

% INITIALIZING MATRIXE
bpSIM       = zeros(T,1);
WfSIM       = zeros(T,1);
bindSIM     = zeros(T,1);
cSIM        = zeros(T,1);
pSIM        = zeros(T,1);
CA_SIM      = zeros(T,1);
bpSP_SIM    = zeros(T,1);
bindSP_SIM  = zeros(T,1);
cSP_SIM     = zeros(T,1);
pSP_SIM     = zeros(T,1);
CASP_SIM    = zeros(T,1);
Tau_SIM     = zeros(T,1);
Tregion_SIM = zeros(T,1);

%Initial Conditions
bpSIM(1)    = b0_DE;
bpSP_SIM(1) = b0_SP;

for i=1:T
    
    bptemp     = interp1(B,bpDE(:,S_index(i)),bpSIM(i),'linear','extrap');
    cSIM(i)    = bpSIM(i)*RSIM(i)+ySIM(i)-bptemp;
    
    WfSIM(i)   = interp1(B,Wf(:,S_index(i)),bpSIM(i),'linear','extrap');
    
    bptempSP   = interp1(B,bpSP(:,S_index(i)),bpSP_SIM(i),'linear','extrap');
    cSP_SIM(i) = bpSP_SIM(i)*RSIM(i)+ySIM(i)-bptempSP;
    Tau_SIM(i) = interp1(B,tau(:,S_index(i)),bpSP_SIM(i),'linear','extrap');
    if i<T
        bpSIM(i+1)    = bptemp;
        bpSP_SIM(i+1) = bptempSP;
    end
end

fprintf('welfare gain is: ');
fprintf('%.6f \n ',  mean(WfSIM)*100);
disp(' ')

pSIM               = (1-omega)/omega*(cSIM./yNSIM).^(1+ita);
pSP_SIM            = (1-omega)/omega*(cSP_SIM./yNSIM).^(1+ita);

bplim_SIM0         = -  KAPPASIM.*(pSIM.*yNSIM+ySIM);
bplimSP_SIM0       = -  KAPPASIM.*(pSP_SIM.*yNSIM+ySIM);

bplim_SIM          = bplim_SIM0;
bplimSP_SIM        = bplimSP_SIM0;

bplim_SIM(2:end)   = bplim_SIM0(1:end-1);
bplimSP_SIM(2:end) = bplimSP_SIM0(1:end-1);

%%%%% IMPORTANT TO USE TOLERANCE NOT <= %%%%%%%%%%%%
bindSIM((bpSIM-bplim_SIM)<=1e-5) = 1;
bindSP_SIM((bpSP_SIM-bplimSP_SIM)<=1e-5) = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bindSIM(1)    = 0; % Not binding the first perdiod
bindSP_SIM(1) = 0; % Not binding the first perdiod

bpSIM(bindSIM==1)       = bplim_SIM(bindSIM==1);
bpSP_SIM(bindSP_SIM==1) = bplimSP_SIM(bindSP_SIM==1);

Tregion_SIM(Tau_SIM>0) = 1;

CA_SIM(1:end-1)        = bpSIM(2:end)-bpSIM(1:end-1);  % >0 capital outflow
CASP_SIM(1:end-1)      = bpSP_SIM(2:end)-bpSP_SIM(1:end-1);

CA_SIM(T)              = interp1(B,bpDE(:,S_index(T)),bpSIM(T),'linear','extrap')-bpSIM(T);
CASP_SIM(T)            = interp1(B,bpSP(:,S_index(T)),bpSP_SIM(T),'linear','extrap')-bpSP_SIM(T);

Y_SIM                  = pSIM.*yNSIM+ySIM;   % Income
YSP_SIM                = pSP_SIM.*yNSIM+ySIM;

CAY_SIM                = CA_SIM./Y_SIM;  % Current Account
CAYSP_SIM              = CASP_SIM./YSP_SIM;

CAchg_SIM              = CAY_SIM(2:end)-CAY_SIM(1:end-1);
CAchg_SIM              = [0;CAchg_SIM];

Lev_SIM                = bpSIM./Y_SIM;  % Leverage
LevSP_SIM              = bpSP_SIM./YSP_SIM;

CtSIM                  = (omega.*cSIM.^(-ita)+(1-omega).*yNSIM.^(-ita)).^(-1/ita);
CtSPSIM                = (omega.*cSP_SIM.^(-ita)+(1-omega).*yNSIM.^(-ita)).^(-1/ita);

RER_SIM                = (omega^(1/(1+ita))+(1-omega)^(1/(1+ita)).*pSIM.^(ita/(1+ita))).^((1+ita)/ita); % Real Exchange Rate
RERSP_SIM              = (omega^(1/(1+ita))+(1-omega)^(1/(1+ita)).*pSP_SIM.^(ita/(1+ita))).^((1+ita)/ita);


%%

%% 2. Event Analysis
%%%%%%% Parameter %%%%%%%%%%%%%%%%%%%%
nbd=3; % periods before and after crisis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Event analysis')

clear SS
SS                   = zeros(T,1);
SS_SP                = zeros(T,1);
bindSIM0             = bindSIM;
bindSIM0(T)          = 0;
bindSIM0(1:end-1)    = bindSIM(2:T);
bindSP_SIM0          = bindSP_SIM;
bindSP_SIM0(T)       = 0;
bindSP_SIM0(1:end-1) = bindSP_SIM(2:T);

SS(CA_SIM>(mean(CA_SIM)+std(CA_SIM)) & bindSIM0>0)         = 1; % Crisis is defined as current account goes 2 sd away and collateral constraint binds in decentralized economy
SS_SP(CASP_SIM>(mean(CA_SIM)+std(CA_SIM)) & bindSP_SIM0>0) = 1;

fprintf(' \n ');
fprintf('The long-run probability of crisis in DE is: ');
fprintf('%5.4f \n ',  mean(SS));
fprintf('The long-run probability of crisis in SP is: ');
fprintf('%5.4f \n ',  mean(SS_SP));

fprintf('std of CA/Y DE is: ');
fprintf('%5.4f \n ',  std(CAY_SIM));

fprintf('std of CA/Y SP is: ');
fprintf('%5.4f \n ',  std(CAYSP_SIM));

fprintf('mean debt/Y level in DE is: ');
fprintf('%5.4f \n ',  mean(Lev_SIM));

fprintf('mean debt/Y level in SP is: ');
fprintf('%5.4f \n ',  mean(LevSP_SIM));

fprintf('welfare gain is: ');
fprintf('%.6f \n ',  mean(WfSIM)*100);



Cdev     = (CtSIM-mean(CtSIM))./mean(CtSIM);
CSPdev   = (CtSPSIM-mean(CtSPSIM))./mean(CtSPSIM);

RERdev   = (RER_SIM-mean(RER_SIM))./mean(RER_SIM);
RERSPdev = (RERSP_SIM-mean(RERSP_SIM))./mean(RERSP_SIM);
Ydev     = (Y_SIM-mean(Y_SIM))./mean(Y_SIM);
YSPdev   = (YSP_SIM-mean(YSP_SIM))./mean(YSP_SIM);
yTdev    = (ySIM-mean(ySIM))./mean(ySIM);

clear SS_Index SScontrol
SS_Index        = find(SS==1);
SSSP_Index      = find(SS_SP==1);
SS_Index        = SS_Index(1+nbd:end-nbd); %Take out some events that might cause before/after periods to go out of bounds
SSSP_Index      = SSSP_Index(1+nbd:end-nbd);

SScontrol       = ismember(S_index,S_index(SS_Index));  % Track all the exogeneous states that occuered in SSs
SScontrol_Index = find(SScontrol==1);

SSomega         = (SScontrol==1 & SS==0); %longer step
SSomega_Index   = find(SSomega==1);

CdevSS          = Cdev(SS_Index);
RERdevSS        = RERdev(SS_Index);
CSPdevSS        = CSPdev(SS_Index);
RERSPdevSS      = RERSPdev(SS_Index);

fprintf('C drop in DE is: ');
fprintf('%5.4f \n ',  mean(CdevSS));
fprintf('C drop in SP is: ');
fprintf('%5.4f \n ',  mean(CSPdevSS));
fprintf('RER depreciation in DE is: ');
fprintf('%5.4f \n ',  mean(RERdevSS));
fprintf('RER depreciation in SP is: ');
fprintf('%5.4f \n ',  mean(RERSPdevSS));
fprintf('CA/Y | SS in DE is: ');
fprintf('%5.4f \n ',  mean(CAY_SIM(SS_Index)));
fprintf('CA/Y | SS in SP is: ');
fprintf('%5.4f \n ',  mean(CAYSP_SIM(SS_Index)));

% scatter(ySIM,Tau_SIM)


plotspaper

toc
