
% June 11, 2015

% This program sovles the spatial equilibrium for <fj_June 10 2015.pdf>

clear all; clc;

%% Initialize Parameters

N       = 3;      % number of locations
Nbar_H  = 100;    % total measure of high skill workers
Nbar_L  = 100;    % total measure of low skill workers
sigma   = 4;      % elasticity of substitution across goods
epsilon = 2;      % elasticity of substitution across types of workers
gamma_U = -0.2;   % parameter of congestion effect. note: gamma_U<0
gamma_A = 0.1;    % parameter of common agglomeration effect. note: gamma_A>0
phi_H   = 0.1;    % additional agglomeration force for high skill workers
phi_L   = 0.0;    % additional agglomeration force for low skill workers
theta   = 15;     % dispersion parameter of Frechet distribution for unobserved preferences

% Matrix of trade costs (to be symmetric) 
tc = [1.0, 1.2, 1.2;
      1.2, 1.0, 1.2;
      1.2, 1.2, 1.0];
  
% exogenous factor intesity
beta_H = [0.50, 0.50, 0.50]';
beta_L = 1-beta_H;

% Amenity levels. note: after adjusting for congestion,
% U(i) = Ubar(i)*(n_H(i)+n_L(i)).^gamma_U
Ubar = [1.0, 1.0, 1.0]';

% Productivity levels. note: after adjusting for common agglomeration,
% A(i) = Abar(i)*(n_H(i)+n_L(i)).^gamma_A
Abar = [1.0, 1.0, 1.0]';

%% Graph settings and outer loop
point_num = 100;                     % number of skill intensities to plot
skill_share = zeros(N,point_num);    % initialize skill share matrix
skill_premium = zeros(N,point_num);  % initialize skill premium matrix
population = zeros(N,point_num);     % initialize population matrix
relative_W_margin = zeros(point_num,1); % initialize relative marginal workers' welafare for matrix
%W_core_H_vector = zeros(:,point_num);
%W_core_L_vector = zeros(:,point_num);
expected_utility_H_vector = zeros(point_num,1);
expected_utility_L_vector = zeros(point_num,1);
W_H_vector = zeros(point_num,1);
W_L_vector = zeros(point_num,1);
n_H_vector = zeros(N,point_num,1);
n_L_vector = zeros(N,point_num,1);

graph_iter = 0;                      % iteration number for outer loop

for k=1e-12:1/point_num:1-1e-12

    %Abar(1) = 1+k;
    d = .8 - .6*k; % from .8 to .2 between the middle city (SC) and corner cities (NYU and Chicago)

% Matrix of trade costs (to be symmetric) 

%   In my example, where one city gets closer to the center of gravity
%    tc_BC = .2;
%     tc = [1.0 ,      1+d,     1+d+tc_BC;
%          1+d ,      1.0,     1+tc_BC;
%          1+d+tc_BC, 1+tc_BC, 1.0  ];

% In David example for NYU, SC, and Chicago
     tc0 = .2; % between NYU and Chicago
     tc = [1.0 , 1+d, 1+tc0;
          1+d ,  1.0, 1+d;
          1+tc0, 1+d, 1.0  ];
    %% Solution Algorithm

    % step 1: guess initial distribution of high-skilled labor
    rho_H = [1/3, 1/3, 1/3]';   % density of high skill labor

    error_H = 1; eps_n_H = 1e-9; count=0; max_count = 1000; no_conv = 0;
    while error_H>eps_n_H && count<max_count

        n_H = rho_H.*Nbar_H;   % population of high skill workers

        % check for non-convergence
        if count + 1 == max_count
            no_conv = 1;
        end
        
        % step 2-5
        tn_H = (beta_H./beta_L).^(-theta*epsilon/(theta+epsilon)) .* ...
               n_H.^( (theta*(1-epsilon*phi_H)+epsilon)/(theta+epsilon) ); % eq. 16
        W_H2L = Nbar_L.^( 1/epsilon ) .* Nbar_H.^( 1/theta ) .* ...
                sum(tn_H).^( -(theta+epsilon)/(theta*epsilon) );           % eq. 17 (but use n_H instead of \rho_H)
        n_L = tn_H./sum(tn_H) .* Nbar_L;                                   % by plugginh eq. 17 in eq. 16
        omega = W_H2L.^(theta/(theta+epsilon)) .* ...
                (Nbar_H./Nbar_L).^(-1/(theta+epsilon)) .* ...
                (beta_H./beta_L).^(epsilon/(theta+epsilon)) .* ...
                n_H.^((phi_H*epsilon)/(theta+epsilon));                    % skill_premium, eq. 15
        b = 1./(1+n_L./n_H./omega);                                        % share of spending on high-skilled labor
        ctilde = (beta_H.*n_H.^phi_H).^(epsilon./(1-epsilon)) .* b.^(-1/(1-epsilon));   % eq. 17

        % step 6-7: calculate wages of high-skilled labor, define system of 
        % equations, and update iterations.

        % define U and A
        U = Ubar.*(n_H+n_L).^gamma_U;
        A = Abar.*(n_H+n_L).^gamma_A;

        w_H = ( (A./U./ctilde).^(sigma-1) .* b .* ...
                n_H.^((sigma-1)/theta -1) ).^( 1/(2*sigma-1) );            % derived by eq. 21 (set lambda=1)
        f = w_H.^(1-sigma);
        kernel = ( repmat(U' .* ( (n_H').^(-1/theta) ),[N,1]) .* ...
                   repmat(A./ctilde,[1,N]) ./ tc ).^(sigma-1);
        integ = kernel.*repmat(f,[1,N]);
        relative_f_new = sum(integ,1)' ./ sum(sum(integ));   % note: sum(f_new)=1 by construction
        relative_w_H = relative_f_new.^(1/(1-sigma));        % updated wages up to a scale 

        % updated guess of high-skilled labor density
        xx = ( (A./U./ctilde).^(sigma-1) .* b .* ...
               relative_w_H.^(1-2*sigma) ) .^ (theta/(theta-sigma+1));     % eq. 20
        rho_H_new = xx./sum(xx);   

        error_H = max( abs((rho_H_new-rho_H)./rho_H) );
        rho_H = rho_H_new;
        count=count+1;
    end
    
%    w_H_tilde = ( (A./U./ctilde).^(sigma-1) .* b .* ...
%                n_H.^((sigma-1)/theta -1) ).^( 1/(2*sigma-1) );            % derived by eq. 21 (set lambda=1)
%    lambda = ( Nbar_H ./ sum(xx) ).^((theta-sigma+1)/theta); % scale parameter
    lambda = sum(relative_w_H.^(1/(1-sigma)))^( (sigma-1)*(2*sigma-1) );
    w_H = lambda.^( 1/(2*sigma-1) ) .* relative_w_H;
    n_H = rho_H .* Nbar_H;
    kernel = ( repmat(U' .* ( (n_H').^(-1/theta) ),[N,1]) .* ...
               repmat(A./ctilde,[1,N]) ./ tc ).^(sigma-1);
    kappa = w_H.^(1-sigma) ./ sum(kernel.*repmat(w_H.^(1-sigma),[1,N]),1)'; % kappa as defined in step 7 
    % check that kappa(i) = kappa(j) for all i and j
    if sum(kappa-kappa(1)) > 1e-9   % check with 1e-9 instead of zero, (as integration has numerical errors) 
        display('error in integration');
    end
    W_H = kappa(1).^(1/(1-sigma)) .* Nbar_H.^(1/theta);
    
    % By normalizing sum_i w_H(i)=1
    % W_H = 1./sum( (sum(integ,1)').^(1/(1-sigma)) );
       
    W_L = W_H./W_H2L;
    
%    W_core_H = W_H.*n_H.^(1/theta);
%    W_core_L = W_L.*n_L.^(1/theta);    
%    expected_utility_H = gamma((theta-1)/theta) .* sum(W_core_H.^theta).^(1/theta);
%    expected_utility_L = gamma((theta-1)/theta) .* sum(W_core_L.^theta).^(1/theta);
    expected_utility_H = gamma((theta-1)/theta) .* W_H;
    expected_utility_L = gamma((theta-1)/theta) .* W_L;
    %{
    w_H = w_H./sum(w_H);
    w_L = w_H./omega;
    unit_cost = ctilde .* w_H;
    price_matrix = repmat(unit_cost./A,[1,N]).*tc;
    price_index = sum(price_matrix.^(1-sigma),2).^(1/(1-sigma));
    welfare_core_H = w_H ./ price_index .* U;
    welfare_core_L = w_L ./ price_index .* U;
    %}
    % inner loop output
    count
    error_H
    [n_H, n_L]
    omega
    skill_share = n_H./n_L
    W_H2L
    
    % read into graph data
    graph_iter = graph_iter + 1;
    skill_share(:,graph_iter) = n_H./n_L;
    skill_premium(:,graph_iter) = omega;
    population(:,graph_iter) = n_H + n_L;
    relative_W_margin(graph_iter) = W_H2L;
    % core welfare of middle city
    %W_core_H_vector(graph_iter) = W_core_H;
    %W_core_L_vector(graph_iter) = W_core_L;
    expected_utility_H_vector(graph_iter) = expected_utility_H;
    expected_utility_L_vector(graph_iter) = expected_utility_L;
    W_H_vector(graph_iter) = W_H;
    W_L_vector(graph_iter) = W_L;
    n_H_vector(:,graph_iter) = n_H;
    n_L_vector(:,graph_iter) = n_L;
    
    if no_conv == 1
        display('Warning: Did not converge')
        skill_share(:,graph_iter) = [0;0;0];
        skill_premium(:,graph_iter) = [0;0;0];
        population(:,graph_iter) = [0;0;0];
    end

end

%% Create graphs

figure(1)
plot(skill_premium','LineWidth',2);
set(gca,'xtick',[])
xlabel('Mountain road ---> Highway','FontName', 'Calibri', 'FontSize', 19);
ylabel('skill premium','FontName', 'Calibri', 'FontSize', 19);
h_legend=legend('New York','State College','Chicago');
set(h_legend,'FontSize',16);
print('../pics/mr_skillpremium','-dpng')

figure(2)
plot(population'/(Nbar_H + Nbar_L),'LineWidth',2);
set(gca,'xtick',[])
xlabel('Mountain road ---> Highway','FontName', 'Calibri', 'FontSize', 19);
ylabel('total population share','FontName', 'Calibri', 'FontSize', 19);
h_legend=legend('New York','State College','Chicago');
set(h_legend,'FontSize',16);
print('../pics/mr_populationshare','-dpng')

figure(3)
plot(expected_utility_H_vector'./expected_utility_L_vector','LineWidth',2);
set(gca,'xtick',[])
xlabel('Mountain road ---> Highway','FontName', 'Calibri', 'FontSize', 19);
ylabel('Welfare of High to Low skill workers','FontName', 'Calibri', 'FontSize', 19);
h_legend=legend('All cities');
set(h_legend,'FontSize',16);
print('../pics/mr_welfareH2L','-dpng')

figure(4)
plot(expected_utility_H_vector','LineWidth',2);
%plot(W_L_vector,'LineWidth',2);
h_legend=legend('High Skill');
set(h_legend,'FontSize',16);
set(gca,'xtick',[])
xlabel('Mountain road ---> Highway','FontName', 'Calibri', 'FontSize', 19);
ylabel('Expected Utility of High Skill','FontName', 'Calibri', 'FontSize', 19);
print('../pics/mr_welfarehigh','-dpng')

figure(5)
plot(expected_utility_L_vector','LineWidth',2);
h_legend=legend('Low Skill');
set(h_legend,'FontSize',16);
set(gca,'xtick',[])
xlabel('Mountain road ---> Highway','FontName', 'Calibri', 'FontSize', 19);
ylabel('Expected Utility of Low Skill','FontName', 'Calibri', 'FontSize', 19);
print('../pics/mr_welfarelow','-dpng')

figure(6)
plot(n_H_vector'./n_L_vector','LineWidth',2);
h_legend=legend('New York','State College','Chicago');
set(h_legend,'FontSize',16);
set(gca,'xtick',[])
xlabel('Mountain road ---> Highway','FontName', 'Calibri', 'FontSize', 19);
ylabel('Skill Share','FontName', 'Calibri', 'FontSize', 19);
print('../pics/mr_skillshare','-dpng')
