% Mass of the system
m = 1;

% Transfer function of the plant: G(s) = 1/(m*s)
s = tf('s');
G = 1/(m * s);

% Define the disturbance transfer function (G_d(s))
% Here we assume the disturbance affects the output directly, so G_d(s) = G(s)
G_d = G;

% Define time vector for simulation
t = 0:0.01:10;

% Define reference trajectory (e.g., a step input)
r = ones(size(t));  % Step reference input

% Define disturbance (e.g., a step disturbance)
disturbance = 0.5 * ones(size(t));  % A constant disturbance of 0.5

%% 1. P Controller
Kp = 10;  % Proportional gain (tune this value)
C_P = Kp;  % P controller transfer function

% Closed-loop transfer function with P controller (feedback for reference input)
T_P = feedback(C_P * G, 1);

% Response to reference input and disturbance (open-loop disturbance path)
y_P = lsim(T_P, r, t) + lsim(G_d, disturbance, t);

%% 2. PI Controller
Kp = 10;  % Proportional gain (tune this value)
Ki = 5;   % Integral gain (tune this value)
C_PI = Kp + Ki/s;  % PI controller transfer function

% Closed-loop transfer function with PI controller (feedback for reference input)
T_PI = feedback(C_PI * G, 1);

% Response to reference input and disturbance
y_PI = lsim(T_PI, r, t) + lsim(G_d, disturbance, t);

%% 3. PID Controller
Kp = 10;  % Proportional gain (tune this value)
Ki = 5;   % Integral gain (tune this value)
Kd = 1;   % Derivative gain (tune this value)
C_PID = Kp + Ki/s + Kd*s;  % PID controller transfer function

% Closed-loop transfer function with PID controller (feedback for reference input)
T_PID = feedback(C_PID * G, 1);

% Response to reference input and disturbance
y_PID = lsim(1/(1 + C_PID * G), r, t);

%% Plot the Results
figure;
plot(t, y_P, 'r', 'LineWidth', 1.5); hold on;
plot(t, y_PI, 'b', 'LineWidth', 1.5);
plot(t, y_PID, 'g', 'LineWidth', 1.5);
plot(t, r, 'k--', 'LineWidth', 1.5);  % Reference signal
plot(t, disturbance, 'm--', 'LineWidth', 1.5);  % Disturbance signal

xlabel('Time (s)');
ylabel('Response');
legend('P Controller', 'PI Controller', 'PID Controller', 'Reference', 'Disturbance');
title('Tracking Performance with Disturbance on Output');
grid on;
