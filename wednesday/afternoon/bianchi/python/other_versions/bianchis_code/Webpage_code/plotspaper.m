%% Variable Counterparts
close all
CONSTRAINTp=zeros(NB,NSS);
CONSTRAINTp(LagrangeSP>=1e-2)=1;
 

h=length(Y_SIM);
[f,xi] = ksdensity(bpSIM);
densityb=f/h;

h=length(Y_SIM);
[fSP,xip] = ksdensity(bpSP_SIM);
densitybp=fSP/h;


% TIGHTening of the Margins
ktightp=bpSP./(-KAPPAS.*(priceSP.*YN+YT));

simC=CtSIM;
simCp=CtSPSIM;
simSS2=SS;
simSS2p=SS_SP;



%% Figure 1


de=[0.9333 0.1 0] ;
sp=[0 0.5 1];
de1=2.5;
sp1=1.5;

shok=6;

lim11=B(find(CONSTRAINTp(:,shok)<1,1))-0.0001;
lim12=  -0.8138 ;
lim21=-1.05;
lim22=-0.26;

figure(1)
patch([lim11 lim11 lim12 lim12],[lim21 lim22 lim22 lim21],[0.8 0.8 0.8],'EdgeColor','none','LineStyle','none')
hold on
plot(B,bpDE(:,shok),'LineWidth',3.5,'color',de)
hold on
plot(B,bpSP(:,shok),'LineWidth',2.5,'LineStyle','-.','color',sp)
hold on
plot(B,B,'color','black','LineStyle','-','LineWidth',0.5)
box off
annotation(gcf,'textarrow',[0.389 0.3617],[0.4604 0.3914],...
    'TextEdgeColor','none',...
    'String',{'Social Planner'},'FontName','AvantGarde','Fontsize',10);

% Create textarrow
annotation(gcf,'textarrow',[0.5637 0.4625],[0.3113 0.3257],...
    'TextEdgeColor','none',...
    'String',{'Decentralized Equilibrium'},'FontName','AvantGarde','Fontsize',10);

axis([-1.05 -0.6 -1.05 -0.6]);

%% Figure 2


inde= find(xi>xip(1),1);

figure(2)
plot(xi,densityb,'LineWidth',de1,'LineStyle','-','color',de); hold on;
plot(xip,densitybp,'LineWidth',sp1,'LineStyle','-.','color',sp);   legend('Decentralized Equilibrium','Social Planner')
legend boxoff
hXLabel= xlabel('Bond Holdings');
hYLabel= ylabel('Probability');

%% Figure 3


shok=6;
colorpol= [0 0 .5]    ;
aa=find(CONSTRAINTp(:,shok)<1,1);

lim11=B(find(CONSTRAINTp(:,shok)<1,1))+0.0017;
lim12=  -0.8198 ;
lim21=-1.05+0.0000000000001;
lim22=0.26;

figure(3)
subplot(1,2,1);
patch([lim11 lim11 lim12 lim12],[lim21 lim22 lim22 lim21],[0.8 0.8 0.8],'EdgeColor','none','LineStyle','none')
hold on;
plot(B(1:aa),zeros(1,aa),'-','LineWidth',2.,'color', colorpol); hold on
plot(B(aa+1:end),tau(aa+1:end,shok),'-','LineWidth',2.5,'color', colorpol); hold on

hXLabel =xlabel('Current Bond Holdings');
hYLabel= ylabel('Percentage');
hTitle=title('Implied Tax on Debt');
axis([-1.05 -0.6 0 0.25])


subplot(1,2,2);
plot(B,(1-ktightp(:,shok))*100,'-','LineWidth',2.5,'color' ,colorpol)
hTitle=title('Tightening of Margins');
hXLabel =xlabel('Current Bond Holdings');
hYLabel= ylabel('Percentage');
axis([-1.05 -0.6 0 40])

%% Figure 4

Ccrisis=simC(simSS2>0);
Ccrisisp=simCp(simSS2p>0);
[f,xic] = ksdensity(Ccrisis);
hold on
[fp,xicp] = ksdensity(Ccrisisp);
xic=(xic-mean(simC))/mean(simC);
xicp=(xicp-mean(simCp))/mean(simCp);

figure(4)
plot(xic*100,f/h,'LineWidth',de1,'LineStyle','-','color',de); hold on;
plot(xicp*100,fp/h,'LineWidth',sp1,'LineStyle','-.','color',sp);
legend('Decentralized Equilibrium','Social Planner');
hXLabel=xlabel('Percentage change in consumption ');
hYLabel=ylabel('Probability');


%% Figure 5

WELFDIF=(VSP./V).^(1/(1-sigma))-1;


lim11=B(find(CONSTRAINTp(:,shok)<1,1));
lim12=  -0.8138 ;
lim21=-1.05+0.0000000000001;
lim22=0.26;

figure(5)
patch([lim11 lim11 lim12 lim12],[lim21 lim22 lim22 lim21],[0.8 0.8 0.8],'EdgeColor','none','LineWidth',1) %,...
hold on
plot(B,WELFDIF(:,shok)*100,'LineWidth',2.5,'LineStyle','-','color', colorpol)
hold on
axis([-1.05 -0.6 0.05 0.20])



