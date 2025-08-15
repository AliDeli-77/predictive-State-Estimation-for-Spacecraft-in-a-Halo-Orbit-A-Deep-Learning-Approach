
function [result] = CR3BP(times,r)
global mu
% Position and Velocity
x  = r(1) ;
y  = r(2) ;
z  = r(3) ;
dx = r(4) ; 
dy = r(5) ;
dz = r(6) ;

r1 = [x+mu , y , z];
r2 = [x-1+mu , y , z];
nr1 = norm(r1) ;
nr2 = norm(r2) ;

%Equations
ddx =  2*dy + x - ((1-mu)/nr1^3)*r1(1) - (mu/nr2^3) * r2(1) ;
ddy = -2*dx + y - ((1-mu)/nr1^3)*r1(2) - (mu/nr2^3) * r2(2) ;
ddz =           - ((1-mu)/nr1^3)*r1(3) - (mu/nr2^3) * r2(3) ;

%Result 
result = [r(4:6); [ddx ;ddy; ddz] ];

end