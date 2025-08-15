clc;clear
m_moon    = 7.34767309e22;  %kg
m_earth   = 5.9742e24;      %kg
mu        = m_moon/(m_moon + m_earth);
[L1, L2, L3, L4, L5]  = lagrangepoints(mu);
A = compute_A([L2' 0 0 0]');
B = [0 0 0;0 0 0;0 0 0;1 0 0;0 1 0;0 0 1] ;
co = ctrb(A,B) ;
disp('rank of the controbality matrix:')
rankco = rank(co) 