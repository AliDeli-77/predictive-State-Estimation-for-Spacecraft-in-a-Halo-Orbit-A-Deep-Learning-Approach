%% Save 6-DoF state (x,y,z,vx,vy,vz) from Simulink run to Excel
% Assumptions:
%   • Variable 'out' is either a timeseries OR a SimulationOutput
%   • The underlying signal has 6 columns: [x y z vx vy vz]

% Choose the Excel file name in the current folder
excelFile = fullfile(pwd, 'sim_state.xlsx');

%% 1) Extract time and data from whichever format we received
if isa(out, 'timeseries')
    % ----- Case 1: 'out' is already a timeseries -------------------------
    timeVec = out.Time;          % column vector (Nx1)
    dataMat = out.Data;          % Nx6  [x y z vx vy vz]
    
elseif isa(out, 'Simulink.SimulationOutput')
    % ----- Case 2: 'out' is a SimulationOutput --------------------------
    % -- adapt 'simout' below if you named the To-Workspace variable else
    ts      = out.get('simout'); % returns a timeseries object
    timeVec = ts.Time;
    dataMat = ts.Data;
    
else
    error('Variable "out" is neither timeseries nor SimulationOutput.');
end

%% 2) Split the 6-column matrix into position and velocity for clarity
pos = dataMat(:, 1:3);   % x, y, z
vel = dataMat(:, 4:6);   % vx, vy, vz

%% 3) Assemble a nicely-labeled table and write it to Excel
T = table( timeVec, pos(:,1), pos(:,2), pos(:,3), ...
                     vel(:,1), vel(:,2), vel(:,3), ...
          'VariableNames', {'Time','x','y','z','vx','vy','vz'} );

writetable(T, excelFile);

fprintf('✓ State history written to "%s"\n', excelFile);



%% 3-D plot of the position components stored in “out”
% Handles either a timeseries or a SimulationOutput object.

if isa(out,'timeseries')
    pos = out.Data(:,1:3);                       % [x y z] columns
elseif isa(out,'Simulink.SimulationOutput')
    ts  = out.get('simout');                     % change name if needed
    pos = ts.Data(:,1:3);
else
    error('Variable "out" must be a timeseries or SimulationOutput.');
end

figure;
plot3(pos(:,1), pos(:,2), pos(:,3), 'k', 'LineWidth', 1.2);
grid on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('3-D Trajectory (first three components of out)');
view(35,25);            % nice angled view; tweak as desired

