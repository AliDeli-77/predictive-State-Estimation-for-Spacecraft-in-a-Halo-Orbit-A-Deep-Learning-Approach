function [L1, L2, L3, L4, L5]  = lagrangepoints(mu)

c = 1 - mu;

%% L1
f_L1 = [1, 2*(2*mu-1), 1-(6*mu*c), 2*mu*c*(1-2*mu)+(2*mu)-1,(mu^2*c^2)+2*(mu^2+c^2), mu^3-c^3];
roots_L1 = roots(f_L1);
L1 = 0;
for i=1:5
   if roots_L1(i) > -mu && roots_L1(i) < 1-mu 
       L1 = roots_L1(i);
   end
end 

%% L2
f_L2 = [1, 2*(2*mu-1), 1-6*mu*c, 2*mu*c*(1-2*mu)-1, mu^2*c^2+2*(1-2*mu), -mu^3-c^3];
roots_L2 = roots(f_L2);
for i=1:5
   if roots_L2(i) > 1-mu
       L2 = roots_L2(i);
   end
end

%% L3
f_L3 = [1, 2*(2*mu-1), 1-6*mu*c, 2*mu*c*(1-2*mu)+1, mu^2*c^2-2*(1-2*mu), mu^3+c^3];
roots_L3 = roots(f_L3);
for i=1:5
   if roots_L3(i) < -mu
       L3 = roots_L3(i);
   end
end 

%% L4
L4 = [-mu+0.5 ;sqrt(3)/2];

%% L5
L5 = [-mu+0.5 ;-sqrt(3)/2];

%% Output
L1 = [L1 ; 0 ; 0];
L2 = [L2 ; 0 ; 0];
L3 = [L3 ; 0 ; 0];
L4 = [L4 ; 0];
L5 = [L5 ; 0];

end