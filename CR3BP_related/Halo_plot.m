% Halo Orbit Of Earth-Moon



%% Data
clc;clear;close all
global mu
R_E       = 6378.1363 ;
m_moon    = 7.34767309e22;  %kg
m_earth   = 5.9742e24;      %kg
mu        = m_moon/(m_moon + m_earth);
DU        = 384402 ; %km
TU        = 27.3216*3600*24/(2*pi) ; %sec

%% Finding Lagrange Points

[L1, L2, L3, L4, L5]  = lagrangepoints(mu);


%% Initial condition


r0 = [1.171590840448506 ; 0 ;0.088096915825144];
v0 = [0; -0.189065015297496;0]; %L2 initial condition
x0 = [r0' v0'];


T_halo =3.348; 

%% Plot the Halo orbit

ode_options     = odeset('RelTol',1e-13,'AbsTol',1e-22);
[t_halo,X_halo] = ode45(@CR3BP, [0:0.0001:T_halo], x0 , ode_options); 

plot3((X_halo(:,1)), X_halo(:,2), X_halo(:,3), 'k', 'LineWidth', 1.5); 
hold on
plot(1-mu, 0, 'ok', 'markerfacecolor', 'blue', 'markersize', 8); 
text(1-mu, 0, '   Moon', 'FontSize', 12, 'FontWeight', 'bold'); 
plot3(L2(1), L2(2), L2(3), 'r*', 'MarkerSize', 8);
text(L2(1), L2(2), L2(3), '   L2', 'FontSize', 12, 'FontWeight', 'bold'); 
plot(-mu, 0, 'ok', 'markerfacecolor', 'blue', 'markersize', 8); 
text(-mu, 0, '   Earth', 'FontSize', 12, 'FontWeight', 'bold'); 
xlabel('X', 'FontSize', 12, 'FontWeight', 'bold'); 
ylabel('Y', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('Z', 'FontSize', 12, 'FontWeight', 'bold');
grid on
view(3); % Adjust 3D view


