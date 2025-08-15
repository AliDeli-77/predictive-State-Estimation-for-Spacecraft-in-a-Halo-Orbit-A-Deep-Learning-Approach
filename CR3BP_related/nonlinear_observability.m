clc;clear

syms x y z dx dy dz u1 u2 u3

R_E       = 6378.1363 ;
m_moon    = 7.34767309e22;  %kg
m_earth   = 5.9742e24;      %kg
mu        = m_moon/(m_moon + m_earth);

r1 = [x+mu , y , z];
r2 = [x-1+mu , y , z];
nr1 = norm(r1) ;
nr2 = norm(r2) ;

f1 = dx;
f2 = dy;
f3 = dz;
f4 = 2*dy + x - ((1-mu)/nr1^3)*r1(1) - (mu/nr2^3) * r2(1) + u1 ;
f5 = -2*dx + y - ((1-mu)/nr1^3)*r1(2) - (mu/nr2^3) * r2(2) + u2;
f6 =  - ((1-mu)/nr1^3)*r1(3) - (mu/nr2^3) * r2(3) + u3;

%sensor outputs
h1 = x;
h2 = y;
h3 = z;

h = [h1 h2 h3]';
x=[x;y;z;dx;dy;dz];
f=[f1;f2;f3;f4;f5;f6];

lie1=jacobian(h, x)*f;
lie2=jacobian(lie1, x)*f;
lie3=jacobian(lie2, x)*f;
lie4=jacobian(lie3, x)*f;
lie5=jacobian(lie4, x)*f;

G=[h;lie1;lie2;lie3;lie4;lie5];

O=jacobian(G,x);
disp('Rank of the observability matrix:')
rank(O)
