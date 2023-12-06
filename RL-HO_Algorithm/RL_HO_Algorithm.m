%%% Implementing the Orientation-based Random Waypoint (ORWP) algorithm
%%% described in [1]: "Modeling the Random Orientation of Mobile Devices:
%%% Measurement, Analysis and LiFi Use Case" https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8540452.

% For WiFi Channel:
% This code was implemented based on the following references:
% [2] "Mobility-aware load balancing for hybrid LiFi and WiFi networks" https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8863828
% [3] "Next Generation Wireless LANs 802.11n and 802.11ac" https://www.cambridge.org/core/books/next-generation-wireless-lans/1C3DF09331104E23D48599AE1D6373D4

clear all
% close all
addpath(genpath('./geom3d')) % path for determining blockage (intersection between line and cylinder)
rng(3,'twister'); % Fix the random number generator

%%% ML Model

%%% Loading the dataset

% dataset = csvread('dataset_v3.csv');
% X = dataset(:,1:length(dataset(1,:))-1);
% Y = dataset(:,length(dataset(1,:)));

% dataset = csvread('dataset_v7.csv');
% dataset = csvread('dataset_v8.csv');
% dataset = csvread('dataset_v9.csv');
dataset = csvread('dataset_v10.csv');
X = dataset(:,[1 6:length(dataset(1,:))-1]); % Using all SNRs (LiFi and WiFi) + Host AP ID
% X = dataset(:,1:length(dataset(1,:))-1); % Using all data
Y = dataset(:,length(dataset(1,:)));

Y(Y==1)= 0;
Y(Y==2)= 1;
Y(Y==3)= 2;

%%% Splitting dataset in traning and test

% Cross varidation (train: 70%, test: 30%)
cv = cvpartition(Y,'HoldOut',0.3,'Stratify',true);
idx = cv.test;

% Separate to training and test data
Xtrain = X(~idx,:);
XTest  = X(idx,:);
ytrain = Y(~idx,:);
ytest = Y(idx,:);

%%% Dataset processing

%%% Hyperparameters tuning

eta_vector = 0:0.1:0.2; % learning rate
gamma_vector = 0:0.1:1; %  minimum loss reduction required to make a split
max_depth_vector = 3:1:10; % to control over-fitting as higher depth will allow model to learn very specific relations 
min_child_weight_vector = 1:1:5; % to control over-fitting as higher values prevent a model from learning very specific relations
subsample_vector = 0.5:0.1:1; % to control over-fitting as lower values make the algorithm more conservative and prevents overfitting 

parameters_combination_matrix = combvec(eta_vector, gamma_vector, max_depth_vector, min_child_weight_vector,subsample_vector);

%%% Best model

% idx = 1494; % dataset_v8 -> RSS values
% idx = 4167; % dataset_v9 -> RSS values
idx = 6051; % dataset_v9 -> RSS values
best_eta = parameters_combination_matrix(1,idx);
best_gamma = parameters_combination_matrix(2,idx);
best_max_depth = parameters_combination_matrix(3,idx);
best_min_child_weight = parameters_combination_matrix(4,idx);
best_subsample = parameters_combination_matrix(5,idx);

params_best_model = struct;

% Define the type of model to run at each iteration usig the hyperparameter booster
params_best_model.booster           = 'gbtree'; % use tree-based models. Another option is using linear models
% params.booster           = 'dart'; % use tree-based models. Another option is using linear models
% params.booster           = 'gblinear'; % use linear models BAD PERFORMANCE

% Booster parameters (for tree-based models).
params_best_model.eta               = best_eta; % learning rate. [0,1]. Typical final values : 0.01-0.2  It is the step size shrinkage used in update to prevent overfitting
params_best_model.gamma               = best_gamma; % [0,Inf] specifies the minimum loss reduction required to make a split.
params_best_model.max_depth         = best_max_depth; % [3,10] The maximum depth of a tree. It is used to control over-fitting. Increasing this value will make the model more complex and more likely to overfit. should be tuned using CV
params_best_model.min_child_weight  = best_min_child_weight; % [0,Inf] It is used to control over-fitting. Too high values can lead to under-fitting. should be tuned using CV
params_best_model.subsample         = best_subsample; % [0,1] It denotes the fraction of observations to be randomly samples for each tree. Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
params_best_model.colsample_bytree  = 0.5; % i(0, 1] s the subsample ratio of columns when constructing each tree
params_best_model.num_parallel_tree = 1; % Number of parallel trees constructed during each iteration

% Define the loss function to be minimized (objective)
params_best_model.objective         = 'multi:softmax'; % set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
params_best_model.num_class = 3;
eval_metric = 'Accuracy'; % metric to be used for validation data
% eval_metric = 'AUC'; % metric to be used for validation data WORSE THAN Accuracy

%%% Command Line Parameters
num_round                = 1000; % The number of rounds for boosting

best_model = xgboost_train(Xtrain,ytrain,params_best_model,num_round,eval_metric,[]);


%%% Simulation parameters

speed = 1.5; % user speed [m/s]

% LiFi
T_c_theta = 130e-3; % Coherence time of the polar angle. 134 ms -> walking. 373 ms -> sitting
T_HHO = 200e-3; % Horizontal Handover Overhead
T_VHO = 500e-3; % Vertical Handover Overhead
mu_theta_deg = 29.67; % mean of the Gaussian RP [degrees]
sigma_sqr_theta_deg = (7.78)^2; % variance onf the Gaussian RP [degrees]
sigma_theta_deg = 7.78; % variance onf the Gaussian RP [degrees]

N_users = 5; % It includes the main user that we are tracking, plus N_users - 1 users with random location and rotation angles

%%% Channel parameters (as in [1])

% LiFi TX
phi_half_deg = 60;
phi_half = phi_half_deg*pi/180;
m = -1/log2(cos(phi_half));

% LiFi RX
Gf = 1;
FoV_deg = 60;
FoV = FoV_deg*pi/180;
% nn = 1.5;
% Gc = nn^2/(sin(FoV)^2);
Gc = 1;
Aapd_PD1 = 1*10^-4; % 1cm2
% Aapd_PD2 = 0; % 1cm2
Aapd_PD2 = 1*10^-4; % 1cm2
% Aapd = 0.5*10^-4; % 1cm2

% WiFi channel

% d_BP = 10; % breakpoint distance in meters. According to [3], Table 3.4 Path loss model parameters (Model D, IEEE 802.11n, typical office enviroment)
d_BP = 5; % Channel model C (small office)
% freq = 2400000000; % operation frequency
freq = 5000000000; % operation frequency

%%% LiFi SNR calculation parameters

% Transmitters
Popt = 3.5; % [Watt] Transmitted optical power
B_LiFi = 20*10^6; %[MHz] Bandwidth of the LiFi APs

% Receivers
Rpd = 0.53; % [A/W] detector responsivity
k_conv_eff_opt_elect = 1; % optical to electric power conversion effciency
N_LiFi = 1*10^-21; % [A^2/Hz] power spectral density of noise in LiFi (includes shot noise and thermal noise)

%%% WiFi SNR calculation parameters

P_WiFi_AP_dBm = 0; % TX power at WiFi AP in dBm
P_WiFi_AP_Watt = (10^(P_WiFi_AP_dBm/10))/1000;

P_WiFi_AP_interf_dBm = 0; % TX power at WiFi AP in dBm
P_WiFi_AP_Watt_interf = (10^(P_WiFi_AP_interf_dBm/10))/1000;

N_WiFi_dBm = -174; % PSD of noise at the RX in dBm/Hz
N_WiFi_AP_Watt = (10^(N_WiFi_dBm/10))/1000;
B_WiFi = 40*10^6; % Bandwidth at the WiFI AP in Hz

%%% Scenario parameters

% Room
len = 10; % x
width = 10; % y
height = 3;

% User positions for the RWP mobility model (this is for the main user we are tracking)

resolution = 15;               % points per meter in the room
N_ue_pos_x = 1 + len*resolution;            % number of grid samples in the receiver plane
N_ue_pos_y = 1+ width*resolution;          
ue_pos_x = linspace(0, len, N_ue_pos_x);                % space samples on x axis (linearly spaced)
ue_pos_y = linspace(0, width, N_ue_pos_y);              % space samples on y axis (linearly spaced)

%%% APs location 

% LiFi

inter_lifi_ap_dist = 2.5;

pos_lifi_ap_aux = combvec(1.25:inter_lifi_ap_dist:len, 1.25:inter_lifi_ap_dist:width); % Create all combinations of vectors. [x,y] of APs
pos_lifi_ap = [pos_lifi_ap_aux(2,:).' pos_lifi_ap_aux(1,:).' ones(size(pos_lifi_ap_aux,2),1)*height]; % [x,y,z] of APs. APs at the ceiling z = height

N_lifi_ap = size(pos_lifi_ap_aux,2); % number of LiFi APs

% Optical channel assigment and ajacent APs

% For a LiFi network with 16 APs we are considering 7 different optical
% channels, so adjacent APs do not share the same optical channel to avoid
% ICI

ap_channels_vector = [1 2 3 1 3 6 5 5 4 5 7 2 1 2 3 1];
adjacent_aps_matrix = [2 5 6 0 0 0 0 0;
                        1 3 5 6 7 0 0 0;
                        2 4 6 7 8 0 0 0;
                        3 7 8 0 0 0 0 0;
                        1 2 6 9 10 0 0 0;
                        1 2 3 5 7 9 10 11;
                        2 3 4 6 8 10 11 12;
                        3 4 7 11 12 0 0 0;
                        5 6 10 13 14 0 0 0;
                        5 6 7 9 11 13 14 15;
                        6 7 8 10 12 14 15 16;
                        7 8 12 15 16 0 0 0;
                        9 10 14 0 0 0 0 0;
                        9 10 11 13 15 0 0 0;
                        10 11 12 14 16 0 0 0;
                        11 12 15 0 0 0 0 0;
                        11 12 15 16 0 0 0 0];

% WiFi

% pos_wifi_ap = [width/2 len/2 height]; % AP in the ceiling
% pos_wifi_ap = [width/2 len/2 3]; % AP in the ground
pos_wifi_ap = [10 10 height]; % AP in the ceiling in a corner
% pos_wifi_ap = [10 10 height; 5 15 height; 15 5 height; 20 20 height];
% pos_wifi_ap = [10 10 height; 5 15 height; 15 5 height];
% pos_wifi_ap = [10 10 height; 20 20 height];
% pos_wifi_ap = [15 15 height];
N_wifi_ap = size(pos_wifi_ap,1); % number of WiFi APs

%%% Preparing the iterations

N_ite = 10;

% speeds_vector = 0.5:0.5:3;
speeds_vector = 0.5:0.5:1.5;
% lamda_vector = 15;
% speeds_vector = 0.5;
% lamda_vector = [0:0.05:0.2]; % this controls the number of users
lamda_vector = 0; % this controls the number of users
Threshold_coverage = 15; % [dB]

Throughput_matrix_STD_LTE = zeros(length(speeds_vector),length(lamda_vector));
Avr_HHO_rate_matrix_STD_LTE = zeros(length(speeds_vector),length(lamda_vector));
Avr_VHO_rate_matrix_STD_LTE = zeros(length(speeds_vector),length(lamda_vector));
Percent_time_conect_WiFi_STD_LTE_matrix = zeros(length(speeds_vector),length(lamda_vector));
Avr_N_block_STD_LTE_matrix = zeros(length(speeds_vector),length(lamda_vector));


Percent_time_conect_LiFi_STD_LTE_matrix = zeros(length(speeds_vector),length(lamda_vector));
Percent_time_conect_LiFi_Smart_HO_matrix = zeros(length(speeds_vector),length(lamda_vector));
Percent_time_conect_LiFi_STD_LTE_ML_block_matrix = zeros(length(speeds_vector),length(lamda_vector));

Percent_time_blockage_STD_LTE_matrix = zeros(length(speeds_vector),length(lamda_vector));
Percent_time_blockage_Smart_HO_matrix = zeros(length(speeds_vector),length(lamda_vector));
Percent_time_blockage_STD_LTE_ML_block_matrix = zeros(length(speeds_vector),length(lamda_vector));

Blockage_rate_STD_LTE_matrix = zeros(length(speeds_vector),length(lamda_vector));
Blockage_rate_Smart_HO_matrix = zeros(length(speeds_vector),length(lamda_vector));
Blockage_rate_STD_LTE_ML_block_matrix = zeros(length(speeds_vector),length(lamda_vector));


Throughput_matrix_Smart_HO = zeros(length(speeds_vector),length(lamda_vector));
Avr_HHO_rate_matrix_Smart_HO = zeros(length(speeds_vector),length(lamda_vector));
Avr_VHO_rate_matrix_Smart_HO = zeros(length(speeds_vector),length(lamda_vector));
Percent_time_conect_WiFi_Smart_HO_matrix = zeros(length(speeds_vector),length(lamda_vector));
Avr_N_block_Smart_HO_matrix = zeros(length(speeds_vector),length(lamda_vector));


Throughput_matrix_STD_LTE_ML_block = zeros(length(speeds_vector),length(lamda_vector));
Avr_HHO_rate_matrix_STD_LTE_ML_block = zeros(length(speeds_vector),length(lamda_vector));
Avr_VHO_rate_matrix_STD_LTE_ML_block = zeros(length(speeds_vector),length(lamda_vector));
Percent_time_conect_WiFi_STD_LTE_ML_block_matrix = zeros(length(speeds_vector),length(lamda_vector));
Avr_N_block_STD_LTE_ML_block_matrix = zeros(length(speeds_vector),length(lamda_vector));

Avr_Outage_prob_STD_LTE_matrix = zeros(length(speeds_vector),length(lamda_vector));
Avr_Outage_prob_Smart_HO_matrix = zeros(length(speeds_vector),length(lamda_vector));
Avr_Outage_prob_STD_LTE_ML_block_matrix = zeros(length(speeds_vector),length(lamda_vector));


for spd = 1:length(speeds_vector)
    for lmd = 1:length(lamda_vector)
        speed = speeds_vector(1,spd);
%         Obj_function_coeficient = lamda_vector(1,lmd);
        lambda_b = lamda_vector(1,lmd); %unit/m^2 % controls the number of blocking elements

        N_HHO_STD_LTE_per_ite = zeros(1,N_ite);
        N_VHO_STD_LTE_per_ite = zeros(1,N_ite); 
        N_HHO_Smart_HO_per_ite = zeros(1,N_ite);
        N_VHO_Smart_HO_per_ite = zeros(1,N_ite);
        N_HHO_STD_LTE_ML_block_per_ite = zeros(1,N_ite);
        N_VHO_STD_LTE_ML_block_per_ite = zeros(1,N_ite);
        Thr_STD_LTE_per_ite = zeros(1,N_ite);
        Thr_Smart_HO_per_ite = zeros(1,N_ite);
        Thr_STD_LTE_ML_block_per_ite = zeros(1,N_ite);
        Percent_time_conect_WiFi_STD_LTE_ite = zeros(1,N_ite);
        Percent_time_conect_WiFi_Smart_HO_ite = zeros(1,N_ite);
        Percent_time_conect_WiFi_STD_LTE_ML_block_ite = zeros(1,N_ite);
        Percent_time_conect_LiFi_STD_LTE_ite = zeros(1,N_ite);
        Percent_time_conect_LiFi_Smart_HO_ite = zeros(1,N_ite);
        Percent_time_conect_LiFi_STD_LTE_ML_block_ite = zeros(1,N_ite);
        Percent_time_blockage_STD_LTE_ite = zeros(1,N_ite);
        Percent_time_blockage_Smart_HO_ite = zeros(1,N_ite);
        Percent_time_blockage_STD_LTE_ML_block_ite = zeros(1,N_ite);
        Ave_N_steps_per_ite = zeros(1,N_ite);
        Outage_prob_STD_LTE_per_ite = zeros(1,N_ite);
        Outage_prob_Smart_HO_per_ite = zeros(1,N_ite);
        Outage_prob_STD_LTE_ML_block_per_ite = zeros(1,N_ite);
        N_block_STD_LTE_per_ite = zeros(1,N_ite);
        N_block_Smart_HO_per_ite = zeros(1,N_ite);
        N_block_STD_LTE_ML_block_per_ite = zeros(1,N_ite);
        
        actions_vector_STD_LTE_ML_block = [];
        usr_satisfaction_datarate_matrix = [];
        reward_vector = [];
        actions_vector_random = [];
        
                    % Initialize Q table
%             q_inital_value = 0.5;
            q_inital_value = 0;
            q_table(1,2) = q_inital_value; % satisfied user connected to LiFi walking to a wall -> A2 (Select the best WiFi AP)
            q_table(2,1) = q_inital_value; % satisfied user connected to LiFi walking to cell center -> A1 (Select the best LiFi AP)
            q_table(3,3) = q_inital_value; % satisfied user connected to LiFi crossing cell edge -> A3 (Do nothing)
            q_table(4,2) = q_inital_value; % non satisfied user connected to LiFi walking to a wall -> A2 (Select the best WiFi AP)
            q_table(5,1) = q_inital_value; % non satisfied user connected to LiFi walking to cell center -> A1 (Select the best LiFi AP)
            q_table(6,3) = q_inital_value; % non satisfied user connected to LiFi crossing cell edge -> A3 (Do nothing)
            q_table(7,3) = q_inital_value; % satisfied user connected to WiFi walking to a wall -> A3 (Do nothing)
%             q_table(8,3) = 0.35; % satisfied user connected to WiFi walking to cell center -> A3 (Do nothing)
%             q_table(9,3) = q_inital_value; % satisfied user connected to WiFi crossing cell edge -> A3 (Do nothing)
%             q_table(7,1) = 0.35; % satisfied user connected to WiFi walking to a wall -> A1 (Select the best LiFi AP)
            q_table(8,1) = q_inital_value; % satisfied user connected to WiFi walking to cell center -> A1 (Select the best LiFi AP)
            q_table(9,1) = q_inital_value; % satisfied user connected to WiFi crossing cell edge -> A1 (Select the best LiFi AP)
            q_table(10,3) = q_inital_value; % non satisfied user connected to WiFi walking to a wall -> A3 (Do nothing)
            q_table(11,1) = q_inital_value; % non satisfied user connected to WiFi walking to cell center -> A1 (Select the best LiFi AP)
            q_table(12,1) = q_inital_value; % non satisfied user connected to WiFi crossing cell edge -> A1 (Select the best LiFi AP)

                    % RL algorithm
            q_table = zeros(12,3);    % #states , # actions = 3
            q_table_story_S1 = [];
            q_table_story_S2 = [];
            q_table_story_S3 = [];
            q_table_story_S4 = [];
            q_table_story_S5 = [];
            q_table_story_S6 = [];
            q_table_story_S7 = [];
            q_table_story_S8 = [];
            q_table_story_S9 = [];
            q_table_story_S10 = [];
            q_table_story_S11 = [];
            q_table_story_S12 = [];

        for ite = 1 : N_ite


            %%% Initialization

            pos_ue_0 = [0 0]; % Initial position of main user
            pos_ue_k_1 = pos_ue_0;
            pos_ue_n_1 = pos_ue_k_1;
            n = 1;
            flag_HO_decision = 0; % Indicates if the first HO decision has been made
%             lambda_b = 0.1; %unit/m^2 % controls the number of blocking elements
            N_steps_per_path = [];

            % Generating a random tajectory for each iteration

            switch speed
                case 0.5 
                    N_waypoints = 5;
                    Obj_function_coeficient = 15;
%                     learning_rate = 0.5;
%                     Threshold_t_TTT_STD_LTE_ML_block = 4;
                case 1 
                    N_waypoints = 50;
                    Obj_function_coeficient = 15;
%                     learning_rate = 0.5;
%                     Threshold_t_TTT_STD_LTE_ML_block = 4;
                case 1.5 
                    N_waypoints = 100;
                    Obj_function_coeficient = 15;
%                     learning_rate = 0.5;
%                     Threshold_t_TTT_STD_LTE_ML_block = 3;
                 case 2
                    N_waypoints = 150;
                    Obj_function_coeficient = 15;
%                     learning_rate = 0.5;
%                     Threshold_t_TTT_STD_LTE_ML_block = 3;
                case 2.5 
                    N_waypoints = 200;
                    Obj_function_coeficient = 15;
%                     learning_rate = 0.5;
%                     Threshold_t_TTT_STD_LTE_ML_block = 2;
                case 3
                    N_waypoints = 250;
                    Obj_function_coeficient = 15;
%                     learning_rate = 0.5;
%                     Threshold_t_TTT_STD_LTE_ML_block = 2;
                otherwise
                    N_waypoints = 50;
            end

            pos_ue_trajectory = zeros(N_waypoints,2);
            axis_points = [0.625:0.625:len];

            for waypoint = 1 : N_waypoints
                x_pos_ue = axis_points(randi([1,numel(axis_points)]));
                y_pos_ue = axis_points(randi([1,numel(axis_points)]));
                pos_ue_trajectory(waypoint,:) = [x_pos_ue y_pos_ue];
            end

            N_r = length(pos_ue_trajectory(:,1)); % number of runs = number of random waypoints

            %%% Vectors to store the parameters of interest on each iteration

            pos_ue_vector = pos_ue_0; % position of main user on each iteration
            % theta_deg_vetor = 0; % inclination angle of main user on each ietartion
            % omega_deg_vector = 0; % angle of direction of main user on each ietartion
            % blocking_matrix = zeros(1,N_lifi_ap); % stores the blockages between the main user and all deployed APs for each iteration
            % FoV_indicator_matrix = zeros(1,N_lifi_ap); % stores the FoV indicator between the main user and all deployed APs for each iteration
            % snr_lifi_matrix_dB_ite_total_v1 = zeros(1,N_lifi_ap);
            % sinr_lifi_matrix_dB_ite_total_v1 = zeros(1,N_lifi_ap); % stores the sinr between the main user and all deployed LiFi APs for each iteration
            % sinr_lifi_matrix_watt_ite_total_v1 = zeros(1,N_lifi_ap);
            % snr_wifi_matrix_dB_ite = zeros(1,N_wifi_ap); % stores the snr between the main user and all deployed WiFi APs for each iteration
            % sinr_wifi_matrix_dB_ite = zeros(1,N_wifi_ap);
            % sinr_wifi_matrix_watt_ite = zeros(1,N_wifi_ap);
            theta_deg_vetor = []; % inclination angle of main user on each ietartion
            omega_deg_vector = []; % angle of direction of main user on each ietartion
            blocking_matrix = []; % stores the blockages between the main user and all deployed APs for each iteration
            FoV_indicator_matrix = []; % stores the FoV indicator between the main user and all deployed APs for each iteration
            snr_lifi_matrix_dB_ite_total_v1 = [];
            sinr_lifi_matrix_dB_ite_total_v1 = []; % stores the sinr between the main user and all deployed LiFi APs for each iteration
            sinr_lifi_matrix_watt_ite_total_v1 = [];
            snr_wifi_matrix_dB_ite = []; % stores the snr between the main user and all deployed WiFi APs for each iteration
            sinr_wifi_matrix_dB_ite = [];
            sinr_wifi_matrix_watt_ite = [];
            sinr_lifi_wifi_matrix_watt_ite_total = [];



            %%% Parameter HO Algorithm STD-LTE
            Host_AP_STD_LTE = 1; % Initially, the main user is connected to LiFi AP1
            prev_Host_AP_STD_LTE = 1;
            best_AP_STD_LTE = 1; % Indicates the best target AP that it is being tracked. If it stays the best for 3 t_TTT, will become Host_AP
            flag_t_TTT_STD_LTE = 0; % Flag to indicate if t_TTT was already triggered
            delta_to_trigger_t_TTT = 1; % at least 1 dB higher 
            t_TTT_STD_LTE = 0; % counter time to trigger HO algorithm
            N_VHO_STD_LTE = 0;
            N_HHO_STD_LTE = 0;
            N_block_STD_LTE = 0;
            flag_blockage = 0;
            t_TTT_vector_STD_LTE = []; % stores the value of t_TTT counter for each iteration
            HO_algorithm_triggers_STD_LTE = []; % for each iteration, it stores 1 -> HO algorithm was triggered or 0 -> HO algorithm wasn't triggered
            best_AP_vector_STD_LTE = []; % stores the best AP
            best_AP_vector_STD_LTE_SINR_value = []; % stores the best AP
            SINR_Host_AP_vector_STD_LTE = [];
            Host_AP_vector_STD_LTE = [1]; % stores the host AP
            SINR_Host_AP_vector_STD_LTE = []; % stores the SINR of the host AP
            Blockage_vector_STD_LTE = []; % stores the blockages per iteration. 0 -> blockage, 1 -> no blockage

            %%% Parameter Smart HO Algorithm 
            Host_AP_Smart_HO = 1; % Initially, the main user is connected to LiFi AP1
            prev_Host_AP_Smart_HO = 1;
            best_AP_Smart_HO = 1; % Indicates the best target AP that it is being tracked. If it stays the best for 3 t_TTT, will become Host_AP
            flag_t_TTT_Smart_HO = 0; % Flag to indicate if t_TTT was already triggered
            t_TTT_Smart_HO = 0; % counter time to trigger HO algorithm
            N_VHO_Smart_HO = 0;
            N_HHO_Smart_HO = 0;
            N_block_Smart_HO = 0;
            t_TTT_vector_Smart_HO = []; % stores the value of t_TTT counter for each iteration
            HO_algorithm_triggers_Smart_HO = []; % for each iteration, it stores 1 -> HO algorithm was triggered or 0 -> HO algorithm wasn't triggered
            best_AP_vector_Smart_HO = []; % stores the best AP
            best_AP_vector_Smart_HO_SINR_value = []; % stores the best AP
            SINR_Host_AP_vector_Smart_HO = [];
            Host_AP_vector_Smart_HO= [1]; % stores the best AP
            Blockage_vector_Smart_HO = []; % stores the blockages per iteration. 0 -> blockage, 1 -> no blockage
            
            %%% Parameters of the proposed HO algorithm (ML-based)
            best_AP_STD_LTE_ML_block = 1; % Indicates the best target AP that it is being tracked. If it stays the best for 3 t_TTT, will become Host_AP
            Host_AP_STD_LTE_ML_block = 1; % Initially, the main user is connected to LiFi AP1
            prev_Host_AP_STD_LTE_ML_block = 1;
            flag_t_TTT_STD_LTE_ML_block = 0; % Flag to indicate if t_TTT was already triggered
            delta_to_trigger_t_TTT_STD_LTE_ML_block = 1; % at least 1 dB higher 
            t_TTT_STD_LTE_ML_block = 0; % counter time to trigger HO algorithm
            N_VHO_STD_LTE_ML_block = 0;
            N_HHO_STD_LTE_ML_block = 0;
            N_block_STD_LTE_ML_block = 0;
            N_blockages_STD_LTE_ML_block = 0;
            Blockage_vector_STD_LTE_ML_block = []; % stores the blockages per iteration. 0 -> blockage, 1 -> no blockage

            
            t_TTT_vector_STD_LTE_ML_block = []; % stores the value of t_TTT counter for each iteration
            HO_algorithm_triggers_STD_LTE_ML_block = []; % for each iteration, it stores 1 -> HO algorithm was triggered or 0 -> HO algorithm wasn't triggered
            best_AP_vector_STD_LTE_ML_block = []; % stores the best AP
            Host_AP_vector_STD_LTE_ML_block = [1];
            user_traj_predict_vector_STD_LTE_ML_block = [];
            SNR_Host_AP_vector_STD_LTE_ML_block = [];
            SINR_Host_AP_vector_STD_LTE_ML_block = [];
            LiFi_blockages_STD_LTE_ML_block = [];
            HO_failure_vector_STD_LTE_ML_block = [];
            added_q_value_vector = [];
            

            SINR_user_requirement = 10; % [dB] data rate required by current user. This is a threshold that should be adjusted
            prev_state = 2; % Initial state, Connected to LiFi, satisfied, walking to cell center
            prev_action = 3; % Initial action, do nothing
            beta = 0.5; % for reward definition
            epsilon = 0.1;
            learning_rate = 0.5;
            discount_rate = 0.2;
            Threshold_t_TTT_STD_LTE_ML_block = 3;
            prev_datarate = 1000000; % for reward calculation. It ensures that initially the user is satisfied
            SINR_HO_failure_STD_LTE_ML_block = 15; % [dB]
            HO_failure_vector_STD_LTE_ML_block = [];
            user_traj_predict_vector_STD_LTE_ML_block = [];
            
%             % Initialize Q table
%             q_inital_value = 0.5;
% %             q_inital_value = 0;
%             q_table(1,2) = q_inital_value; % satisfied user connected to LiFi walking to a wall -> A2 (Select the best WiFi AP)
%             q_table(2,1) = q_inital_value; % satisfied user connected to LiFi walking to cell center -> A1 (Select the best LiFi AP)
%             q_table(3,3) = q_inital_value; % satisfied user connected to LiFi crossing cell edge -> A3 (Do nothing)
%             q_table(4,2) = q_inital_value; % non satisfied user connected to LiFi walking to a wall -> A2 (Select the best WiFi AP)
%             q_table(5,1) = q_inital_value; % non satisfied user connected to LiFi walking to cell center -> A1 (Select the best LiFi AP)
%             q_table(6,3) = q_inital_value; % non satisfied user connected to LiFi crossing cell edge -> A3 (Do nothing)
%             q_table(7,3) = q_inital_value; % satisfied user connected to WiFi walking to a wall -> A3 (Do nothing)
% %             q_table(8,3) = 0.35; % satisfied user connected to WiFi walking to cell center -> A3 (Do nothing)
% %             q_table(9,3) = q_inital_value; % satisfied user connected to WiFi crossing cell edge -> A3 (Do nothing)
% %             q_table(7,1) = 0.35; % satisfied user connected to WiFi walking to a wall -> A1 (Select the best LiFi AP)
%             q_table(8,1) = q_inital_value; % satisfied user connected to WiFi walking to cell center -> A1 (Select the best LiFi AP)
%             q_table(9,1) = q_inital_value; % satisfied user connected to WiFi crossing cell edge -> A1 (Select the best LiFi AP)
%             q_table(10,3) = q_inital_value; % non satisfied user connected to WiFi walking to a wall -> A3 (Do nothing)
%             q_table(11,1) = q_inital_value; % non satisfied user connected to WiFi walking to cell center -> A1 (Select the best LiFi AP)
%             q_table(12,1) = q_inital_value; % non satisfied user connected to WiFi crossing cell edge -> A1 (Select the best LiFi AP)

            %% ORWP algorithm

            for k = 1 : N_r
                %%% Choose a point from the RWP mobility model 

                pos_ue_k = pos_ue_trajectory(k, :); % get the waypoint for current excursion. Note this is the next point. The current one is denoted k-1. For first excursion k-1 is the (0,0)

                if isequal(pos_ue_k_1,pos_ue_k)
                    pos_ue_k = pos_ue_k - 0.1;
                end

                %%% Compute transition length D_k = |Pk - Pk-1|

                dist_vect = [pos_ue_k_1;pos_ue_k];
                D_k = pdist(dist_vect,'euclidean');

                %%% Compute angle of direction omega_d = tan-1(yk - yk-1/xk -xk-1)

                delta_x = pos_ue_k(1,1) - pos_ue_k_1(1,1);
                delta_y = pos_ue_k(1,2) - pos_ue_k_1(1,2);

                angle = atand(abs(delta_y/delta_x));

                %%% Initialize parameters of current excursion from Pk-1 to Pk
                t_move = 0;


                while t_move <= D_k/speed

                    %%% Compute Pn = (xn,yn). xn = xn-1 + v*T_c_theta * cos(omega_d). yn = yn-1 + v*T_c_theta * sin(omega_d)
                    % Note this calculation depends on the location of Pk with respect
                    % to Pk-1 in the XY plane

                    if delta_x >= 0
                        if delta_y >= 0
                            x_n = pos_ue_n_1(1,1) + speed* T_c_theta * cosd(angle); % case 1
                            y_n = pos_ue_n_1(1,2) + speed* T_c_theta * sind(angle);
                            omega_d_deg = angle;
                        else
                            x_n = pos_ue_n_1(1,1) + speed* T_c_theta * cosd(360 - angle); % case 4
                            y_n = pos_ue_n_1(1,2) + speed* T_c_theta * sind(360 - angle);
                            omega_d_deg = 360 - angle;
                        end
                    else
                        if delta_y >= 0
                            x_n = pos_ue_n_1(1,1) + speed* T_c_theta * cosd(180 - angle); % case 2
                            y_n = pos_ue_n_1(1,2) + speed* T_c_theta * sind(180 - angle);
                            omega_d_deg = 180 - angle;
                        else
                            x_n = pos_ue_n_1(1,1) + speed* T_c_theta * cosd(180 + angle); % case 3
                            y_n = pos_ue_n_1(1,2) + speed* T_c_theta * sind(180 + angle);
                            omega_d_deg = 180 + angle;
                        end
                    end

                    %%% UEs' position

                    % Random UEs
                    pos_rnd_ue = zeros(N_users - 1, 2); % vector of random user position
                    for u = 1:N_users - 1
                        pos_rnd_ue(u, :) = [ue_pos_x(randi(length(ue_pos_x))), ue_pos_y(randi(length(ue_pos_y)))];
                    end

                    z_ue = 1; % UEs height
                    pos_UEs = zeros(N_users,3);
                    pos_UEs(1,:) = [x_n y_n z_ue]; % the location of main user is on the first element of this vector
                    pos_UEs(2:N_users,:) = [ pos_rnd_ue z_ue*ones(1,N_users-1)'];

                    %%% Rotation angles of UEs

                    % Azimuth angle
                    omega_rnd_ue_deg = rand(N_users-1,1)*360; % Azimuth angle
                    omega_ue_deg = [omega_d_deg; omega_rnd_ue_deg]; % the azimuth angle for all UEs. The one for main user is on the first element of this vector
                    omega_UEs = omega_ue_deg*pi/180;

                    % Polar angle

                    % Option 1 (Borja's code)

                    pd = makedist('Normal','mu',mu_theta_deg,'sigma',sigma_theta_deg);
                    pd_trunc = truncate(pd,0,90);
                    theta_n_deg_UEs = random(pd_trunc,N_users,1); % the rotation angle for all UEs. The one for main user is on the first element of this vector
                    theta_n_UEs = theta_n_deg_UEs*pi/180;
                    theta_n_UEs_PD2 = pi/2 - theta_n_UEs;

            %         theta_n_deg_UEs = 90*ones(N_users,1);
            %         theta_n_deg_UEs = zeros(N_users,1);
            %         theta_n_UEs = theta_n_deg_UEs*pi/180;
            %         theta_n_UEs_PD2 = pi/2 - theta_n_UEs;

                    %%% Blocking elements

                    cyl_radius = 0.15; % meters
                    cyl_height = 1.75; % m standing up. Sitting up = 1 m
                    d_p_ue = 0.3; % distance between UE and person holding it

                    % users holding the UEs
                    blocking_elements(1:N_users,:) = [pos_UEs(:,1)-d_p_ue*cos(omega_UEs) pos_UEs(:,2)-d_p_ue*sin(omega_UEs)];

                    % people without UEs (just blocking elements)
                    blocking_elements_b = get_poisson_positions(lambda_b, len, width); % this is the location of blocking elements
                    N_b = size(blocking_elements_b,1); %number of people without device

                    blocking_elements(N_users+1:N_users+N_b,:) = blocking_elements_b;

                    % Check channel blockages

                    % Cheng's code
                    L_room=len;
                    AP_loc_mtx=pos_lifi_ap(:,1:2);
                    N_UE=N_users;
                    UE_loc_mtx=pos_UEs(:,1:2);
                    B_loc_mtx=blocking_elements; % blocking elements
                    N_B=length(B_loc_mtx(:,1));

                    r_body=cyl_radius;
                    H_a=height;
                    H_u=z_ue;
                    H_b=cyl_height;

                    [los_clear_index_mtx]=cylinder_blocker_status_fun_v2(r_body,H_a,H_u,H_b,B_loc_mtx,AP_loc_mtx,UE_loc_mtx);

                    blocking_matrix_aux = los_clear_index_mtx(:,:,1);
                    channel_blockage_matrix = prod(blocking_matrix_aux,2)';

                    %%% LiFi channel modeling

                    % Calculating Euclidian distance between users and APs and the cos of the irradiance angles

                    cos_irradiance_angle = zeros(N_users,N_lifi_ap);
                    distance_users_aps = zeros(N_users,N_lifi_ap);
                    distance_users_aps_x = zeros(N_users,N_lifi_ap);
                    distance_users_aps_y = zeros(N_users,N_lifi_ap);
                    distance_users_aps_z = zeros(N_users,N_lifi_ap);

                    for api = 1:N_lifi_ap
                        distance_users_aps(:,api) = sqrt(sum((pos_lifi_ap(api,:) - pos_UEs).^2,2)); % distance matrix between users and APs
                        distance_users_aps_x(:,api) = pos_lifi_ap(api,1)- pos_UEs(:,1);
                        distance_users_aps_y(:,api) = pos_lifi_ap(api,2)- pos_UEs(:,2);
                        distance_users_aps_z(:,api) = pos_lifi_ap(api,3)- pos_UEs(:,3);

                        cos_irradiance_angle(:,api) = (pos_lifi_ap(api,3)-pos_UEs(:,3))./distance_users_aps(:,api); % cos a = adjacent/h
                    end

                    cos_irradiance_angle_deg = cos_irradiance_angle*180/pi;

                    % Calculating the cos of incidence angles

                    cos_incidence_angle = zeros(N_users,N_lifi_ap);
                    cos_incidence_angle_PD2 = zeros(N_users,N_lifi_ap);
                    for usr = 1:N_users
                        % cosine of the incidence angle according to [1]
                        cos_incidence_angle(usr,:) = (distance_users_aps_x(usr,:)*sin(theta_n_UEs(usr))*cos(omega_UEs(usr)) + distance_users_aps_y(usr,:)*sin(theta_n_UEs(usr))*sin(omega_UEs(usr)) + distance_users_aps_z(usr,:)*cos(theta_n_UEs(usr)))./distance_users_aps(usr,:);
                        cos_incidence_angle_PD2(usr,:) = (distance_users_aps_x(usr,:)*sin(theta_n_UEs_PD2(usr))*cos(omega_UEs(usr)) + distance_users_aps_y(usr,:)*sin(theta_n_UEs_PD2(usr))*sin(omega_UEs(usr)) + distance_users_aps_z(usr,:)*cos(theta_n_UEs_PD2(usr)))./distance_users_aps(usr,:);
                    end

                    cos_incidence_angle_deg = cos_incidence_angle*180/pi;        
                    cos_incidence_angle_PD2_deg = cos_incidence_angle_PD2*180/pi;

                    % Calculating the gain of LiFi channel

                    % LoS

                    FoV_indicator = acos(cos_incidence_angle) < FoV;
                    FoV_indicator_PD2 = acos(cos_incidence_angle_PD2) < FoV;
                    H_LiFi = ((m+1)*Gf*Gc*Aapd_PD1./(2*pi*distance_users_aps.^2)).*(cos_irradiance_angle.^m).*cos_incidence_angle.*FoV_indicator.*channel_blockage_matrix;
                    H_LiFi_PD2 = ((m+1)*Gf*Gc*Aapd_PD2./(2*pi*distance_users_aps.^2)).*(cos_irradiance_angle.^m).*cos_incidence_angle.*FoV_indicator_PD2.*channel_blockage_matrix;

                    %%% Calculating SNR and SINR

                    % SNR        
                    snr_lifi_matrix = zeros(N_users,N_lifi_ap); % For PD1
                    snr_lifi_matrix_PD2 = zeros(N_users,N_lifi_ap); % For PD2
                    snr_lifi_matrix_total = zeros(N_users,N_lifi_ap); % Total PPD1 and PD2, (H1 + H2)^2
                    snr_lifi_matrix_total_v1 = zeros(N_users,N_lifi_ap); % Total PPD1 and PD2, H1^2 + H2^2
                    for uu = 1: N_users
                        snr_lifi_matrix(uu,:) = Rpd^2*Popt^2*(1/k_conv_eff_opt_elect)^2*(H_LiFi(uu,:)).^2/(N_LiFi*B_LiFi); % From this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7925839 Eq.5
                        snr_lifi_matrix_PD2(uu,:) = Rpd^2*Popt^2*(1/k_conv_eff_opt_elect)^2*(H_LiFi_PD2(uu,:)).^2/(N_LiFi*B_LiFi);
                        snr_lifi_matrix_total(uu,:) = Rpd^2*Popt^2*(1/k_conv_eff_opt_elect)^2*(H_LiFi(uu,:) + H_LiFi_PD2(uu,:)).^2/(N_LiFi*B_LiFi);
                        snr_lifi_matrix_total_v1(uu,:) = Rpd^2*Popt^2*(1/k_conv_eff_opt_elect)^2*(H_LiFi(uu,:).^2 + H_LiFi_PD2(uu,:).^2)/(N_LiFi*B_LiFi);
                    end

                    snr_lifi_matrix_dB = 10*log10(snr_lifi_matrix);
                    snr_lifi_matrix_dB(snr_lifi_matrix_dB==-Inf)=-30;

                    snr_lifi_matrix_dB_PD2 = 10*log10(snr_lifi_matrix_PD2);
                    snr_lifi_matrix_dB_PD2(snr_lifi_matrix_dB_PD2==-Inf)=-30;

                    snr_lifi_matrix_dB_total = 10*log10(snr_lifi_matrix_total);
                    snr_lifi_matrix_dB_total(snr_lifi_matrix_dB_total==-Inf)=-30;

                    snr_lifi_matrix_dB_total_v1 = 10*log10(snr_lifi_matrix_total_v1);
                    snr_lifi_matrix_dB_total_v1(snr_lifi_matrix_dB_total_v1==-Inf)=-30;

                    % SINR        
                    sinr_lifi_matrix = zeros(N_users,N_lifi_ap);
                    sinr_lifi_matrix_total = zeros(N_users,N_lifi_ap);
                    for uu = 1: N_users
                        for aa = 1: N_lifi_ap
                            H_LiFi_aux_interf = H_LiFi(uu,:);
                            H_LiFi_aux_interf(1,aa) = 0; % this is eliminate the current AP from interfernce calculation
                            H_LiFi_aux_interf_PD2 = H_LiFi_PD2(uu,:);
                            H_LiFi_aux_interf_PD2(1,aa) = 0; % this is eliminate the current AP from interfernce calculation

                            current_ap_channel = ap_channels_vector(1,aa);
                            index_non_interfering_aps = find(ap_channels_vector ~= current_ap_channel);
                            H_LiFi_aux_interf(index_non_interfering_aps) = 0; % this is to eliminate all non-interfering APs from interfernce calculation
                            H_LiFi_aux_interf_PD2(index_non_interfering_aps) = 0;
                            sinr_lifi_matrix(uu,aa) = Rpd^2*Popt^2*(1/k_conv_eff_opt_elect)^2*(H_LiFi(uu,aa)).^2 /((N_LiFi*B_LiFi) + sum((Rpd*Popt*(1/k_conv_eff_opt_elect)^2*ones(1,N_lifi_ap)).*(H_LiFi_aux_interf.^2))); % From this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8863828 Eq.4
                            sinr_lifi_matrix_total(uu,aa) = (Rpd^2*Popt^2*(1/k_conv_eff_opt_elect)^2*(H_LiFi(uu,aa)).^2 + Rpd^2*Popt^2*(1/k_conv_eff_opt_elect)^2*(H_LiFi_PD2(uu,aa)).^2)/((N_LiFi*B_LiFi) + sum((Rpd*Popt*(1/k_conv_eff_opt_elect)^2*ones(1,N_lifi_ap)).*(H_LiFi_aux_interf.^2)) + sum((Rpd*Popt*(1/k_conv_eff_opt_elect)^2*ones(1,N_lifi_ap)).*(H_LiFi_aux_interf_PD2.^2)));
                        end
                    end

                    sinr_lifi_matrix_dB = 10*log10(sinr_lifi_matrix);
                    sinr_lifi_matrix_dB(sinr_lifi_matrix_dB==-Inf)=-30;

                    sinr_lifi_matrix_dB_total = 10*log10(sinr_lifi_matrix_total);
                    sinr_lifi_matrix_dB_total(sinr_lifi_matrix_dB_total==-Inf)=-30;

                    %%% WiFi Channel modeling

                    % Calculating euclidian distance from UEs to WiFi APs

                    dist_UEs_WiFi_AP = zeros(N_users,N_wifi_ap);  % This is a vector with the euclidian distance from each UE to the WiFi TXs

                    for aaa = 1: N_wifi_ap
                        dist_UEs_WiFi_AP(:,aaa) = sqrt(sum((pos_wifi_ap(aaa,:) - pos_UEs).^2,2)); % distance matrix between users and APs
                    end

                    % Free-space loss
                    L_FS_WiFi_matrix = zeros(N_users,N_wifi_ap);

                    for ap_w = 1: N_wifi_ap
                        L_FS_WiFi_matrix(:,ap_w) = 20*log10(dist_UEs_WiFi_AP(:,ap_w)) + 20*log10(freq) - 147.5; % Free-space loss for each user in dB (Eqs. 3.28 in [3] and Eq. 6 in [2])
                    end

                    % Total losses considering the breack point (Eqs. 3.26 and 3.27 in [3] and Eq. 5 in [2])

                    mu_SF_WiFi = 0; % mean 
                %             L_WiFi = zeros(1,length(dist_UEs_WiFi_AP)); 
                    L_WiFi_matrix = zeros(N_users,N_wifi_ap); 

                    for kk = 1 : N_users
                        for ap_w1 = 1: N_wifi_ap
                            if dist_UEs_WiFi_AP(kk,ap_w1) <= d_BP
                                sigma_SF_WiFi = 3; % shadow fading standard deviation in dB (Table 3.4 in [3])
                                pd_SF_WiFi = makedist('Normal','mu',mu_SF_WiFi,'sigma',sigma_SF_WiFi);
                                SF_WiFi = random(pd_SF_WiFi,1,1);  % Shadow fading loss in dB
                                L_WiFi_matrix(kk,ap_w1) = L_FS_WiFi_matrix(kk,ap_w1) + SF_WiFi;
                            else
                                sigma_SF_WiFi = 5; 
                                pd_SF_WiFi = makedist('Normal','mu',mu_SF_WiFi,'sigma',sigma_SF_WiFi);
                                SF_WiFi = random(pd_SF_WiFi,1,1);  % Shadow fading loss in dB
                        %         L_WiFi(1,kk) = L_FS_WiFi_d_BP + 35*log10(dist_UEs_WiFi_AP(1,kk)/d_BP) + SF_WiFi;
                                L_WiFi_matrix(kk,ap_w1) = L_FS_WiFi_matrix(kk,ap_w1) + 35*log10(dist_UEs_WiFi_AP(kk,ap_w1)/d_BP) + SF_WiFi;
                            end
                        end
                    end

                    % WiFi channel (According to Eq.7 in [2])

                    mu_X_H_WiFi = 0; % mean 
                    sigma_X_H_WiFi = 1; % variance

                    pd_X_H_WiFi = makedist('Normal','mu',mu_X_H_WiFi,'sigma',sigma_X_H_WiFi);
                    X_H_WiFi_real = random(pd_X_H_WiFi,N_users,N_wifi_ap); 
                    X_H_WiFi_imag = random(pd_X_H_WiFi,N_users,N_wifi_ap);
                    X_H_WiFi = X_H_WiFi_real + 1i*X_H_WiFi_imag;

                    H_WiFi_matrix = zeros(N_users,N_wifi_ap);
                    phi_WiFi = 45; % [degrees] This is derived from the angle of arrival/departure of the LoS component. It's 45 degrees for IEEE802.1n according to [3] page 42. top.
                    phi_WiFi_rad = phi_WiFi*pi/180;

                    for kkk = 1 : N_users
                        for ap_w2 = 1: N_wifi_ap
                            if dist_UEs_WiFi_AP(kkk,ap_w2) <= d_BP
                        %         K_factor = 2; % Ricean K-factor for LoS (d <= d_BP). It is equal to 2 for LoS in typical office environments (Channel type D in Table 3.3 in [3]. Notice in [1] they use K-factor = 1.)
                                K_factor = 1; % Channel model C (small office)
                            else
                                K_factor = 0; % Ricean K-factor for NLoS (d > d_BP)
                            end
                            H_WiFi_matrix(kkk,ap_w2) = sqrt(K_factor/(K_factor + 1))*(exp(1)^(1i*phi_WiFi_rad)) + sqrt(1/(K_factor + 1))*X_H_WiFi(kkk,ap_w2);
                        end
                    end

                    % WiFi channel gain (according to Eq.8 in [2])

                    G_WiFi_matrix = zeros(N_users,N_wifi_ap);

                    for kkkk = 1: N_users
                        for ap_w3 = 1: N_wifi_ap
                            G_WiFi_matrix(kkkk,ap_w3) = (abs(H_WiFi_matrix(kkkk,ap_w3))^2)*10^(-1*L_WiFi_matrix(kkkk,ap_w3)/10);
                        end
                    end

                    % WiFi SNR
                    snr_wifi_matrix = (P_WiFi_AP_Watt*G_WiFi_matrix)./(N_WiFi_AP_Watt*B_WiFi); 
                    snr_wifi_matrix_dB = 10*log10(snr_wifi_matrix);

                    % SINR

                    sinr_wifi_matrix = zeros(N_users,N_wifi_ap);
                    for uuu = 1: N_users
                        for aaaa = 1: N_wifi_ap
                            G_WiFi_aux_interf = G_WiFi_matrix(uuu,:);
                            G_WiFi_aux_interf(1,aaaa) = 0; % this is to eliminate the current AP from interfernce calculation
                            sinr_wifi_matrix(uuu,aaaa) =  (P_WiFi_AP_Watt*G_WiFi_matrix(uuu,aaaa))/(N_WiFi_AP_Watt*B_WiFi + sum(P_WiFi_AP_Watt*G_WiFi_aux_interf)); % From this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8863828 Eq.4
            %                     sinr_wifi_matrix(uuu,aaaa) =  P_WiFi_AP_Watt*G_WiFi_matrix(uuu,aaaa)/(N_WiFi_AP_Watt*B_WiFi); % From this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8863828 Eq.4
                        end
                    end

                    sinr_wifi_matrix_dB = 10*log10(sinr_wifi_matrix);
                    sinr_wifi_matrix_dB(sinr_wifi_matrix_dB==-Inf)=-30;

                    %%% Returning the parameters of interest for main user (position (Pn), rotation angles (theta and omega), blocking_matrix_n, LiFi and WiFI channels, RSSI from all LiFI and WiFI APs)

                    pos_ue_n = [x_n y_n];  % position
                    pos_ue_vector = [pos_ue_vector; pos_ue_n]; % position
                    theta_deg_vetor = [theta_deg_vetor theta_n_deg_UEs(1,:)]; % polar angle
                    omega_deg_vector = [omega_deg_vector omega_d_deg]; % azimuth angle
                    blocking_matrix = [blocking_matrix; channel_blockage_matrix(1,:)]; % Blockinng matrix
                    FoV_indicator_matrix = [FoV_indicator_matrix; FoV_indicator(1,:)];
                    snr_lifi_matrix_dB_ite_total_v1 = [snr_lifi_matrix_dB_ite_total_v1; snr_lifi_matrix_dB_total_v1(1,:)];
                    sinr_lifi_matrix_dB_ite_total_v1 = [sinr_lifi_matrix_dB_ite_total_v1; sinr_lifi_matrix_dB_total(1,:)];
                    sinr_lifi_matrix_watt_ite_total_v1 = [sinr_lifi_matrix_watt_ite_total_v1; sinr_lifi_matrix_total(1,:)];
                    snr_wifi_matrix_dB_ite = [snr_wifi_matrix_dB_ite; snr_wifi_matrix_dB(1,1)];
                    sinr_wifi_matrix_dB_ite = [sinr_wifi_matrix_dB_ite; sinr_wifi_matrix_dB(1,1)];
                    sinr_wifi_matrix_watt_ite = [sinr_wifi_matrix_watt_ite; sinr_wifi_matrix(1,1)];
                    sinr_lifi_wifi_matrix_watt_ite_total = [sinr_lifi_wifi_matrix_watt_ite_total; [sinr_lifi_matrix_total(1,:),sinr_wifi_matrix(1,1)]];

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
                    %%% HO Algorithm STD-LTE 

                    SINR_all_APs_dB = [sinr_lifi_matrix_dB_total(1,:) sinr_wifi_matrix_dB(1,1)];
                    [current_best_SINR_value current_best_AP_SINR_indx] = max(SINR_all_APs_dB);
                    SINR_all_APs_Watt = [sinr_lifi_matrix_total(1,:) sinr_wifi_matrix(1,1)];
                    
                    SNR_all_APs_dB = [snr_lifi_matrix_dB_total_v1(1,:) snr_wifi_matrix_dB(1,1)];
                    [current_best_SNR_value current_best_AP_SNR_indx] = max(SNR_all_APs_dB);
                    SNR_all_APs_Watt = [snr_lifi_matrix_dB_total_v1(1,:) snr_wifi_matrix(1,1)];

                    best_AP_vector_STD_LTE = [best_AP_vector_STD_LTE current_best_AP_SINR_indx]; % list of best AP for each iteration
                    best_AP_vector_STD_LTE_SINR_value = [best_AP_vector_STD_LTE_SINR_value current_best_SINR_value]; % list of best AP for each iteration

                    % Counting blockages
                    if (Host_AP_STD_LTE ~= 17) && (SINR_all_APs_dB(1,Host_AP_STD_LTE) == -30) % blockage   
                        Blockage_vector_STD_LTE = [Blockage_vector_STD_LTE 0];
                    else
                        Blockage_vector_STD_LTE = [Blockage_vector_STD_LTE 1];
                    end

                    if SINR_all_APs_dB(1,current_best_AP_SINR_indx) >= SINR_all_APs_dB(1,Host_AP_STD_LTE) + delta_to_trigger_t_TTT % at least 1 dB of difference
                        
                       if (Host_AP_STD_LTE ~= 17) && (SINR_all_APs_dB(1,Host_AP_STD_LTE) == -30) % blockage
                            prev_Host_AP_STD_LTE = Host_AP_STD_LTE;
                            Host_AP_STD_LTE = 17;    % Switch to WiFi
                            N_VHO_STD_LTE = N_VHO_STD_LTE + 1;
                            HO_algorithm_triggers_STD_LTE = [HO_algorithm_triggers_STD_LTE; 1]; % list of HOs for each iteration 1-> VHO
                            t_TTT_STD_LTE = 0;
                            flag_t_TTT_STD_LTE = 0; 
                            N_block_STD_LTE = N_block_STD_LTE + 1;
                        
                       else 
                            if flag_t_TTT_STD_LTE == 1 % t_TTT was already triggered
                                if best_AP_STD_LTE == current_best_AP_SINR_indx % if best_AP_STD_LTE is still the best
                                    t_TTT_STD_LTE = t_TTT_STD_LTE + 1;
                                else % rest the counters otherwise
                                    t_TTT_STD_LTE = 0;
                                    flag_t_TTT_STD_LTE = 0;
                                end
                            else % t_TTT has not been triggered before
                                flag_t_TTT_STD_LTE = 1;
                                t_TTT_STD_LTE = 1;
                                best_AP_STD_LTE = current_best_AP_SINR_indx; % best_AP_STD_LTE is the AP under evaluation. I want to know if it will keep being the best for TTT seconds.
                            end

                            t_TTT_vector_STD_LTE = [t_TTT_vector_STD_LTE; t_TTT_STD_LTE]; % values of TTT counter for each iteration

                            if t_TTT_STD_LTE == 2 % The AP is the best for 130ms (TTT)
                               prev_Host_AP_STD_LTE = Host_AP_STD_LTE;
                               Host_AP_STD_LTE = best_AP_STD_LTE;    % make the HO decision
                               % Counting HOs
                               if prev_Host_AP_STD_LTE ~= 17 && Host_AP_STD_LTE == 17 %LiFi to WiFi
                                   N_VHO_STD_LTE = N_VHO_STD_LTE + 1;
                                   HO_algorithm_triggers_STD_LTE = [HO_algorithm_triggers_STD_LTE; 1]; % list of HOs for each iteration 1-> VHO
                               elseif prev_Host_AP_STD_LTE == 17 && Host_AP_STD_LTE ~= 17 % WiFi to LiFi
                                   N_VHO_STD_LTE = N_VHO_STD_LTE + 1;
                                   HO_algorithm_triggers_STD_LTE = [HO_algorithm_triggers_STD_LTE; 1]; % list of HOs for each iteration 1-> VHO
                               elseif prev_Host_AP_STD_LTE ~= 17 && Host_AP_STD_LTE ~= 17 % LiFi to LiFi
                                   N_HHO_STD_LTE = N_HHO_STD_LTE + 1;
                                   HO_algorithm_triggers_STD_LTE = [HO_algorithm_triggers_STD_LTE; 2]; % list of HOs for each iteration 2-> HHO
                               end

    %                            prev_Host_AP_STD_LTE = Host_AP_STD_LTE;
                               t_TTT_STD_LTE = 0;
                               flag_t_TTT_STD_LTE = 0;
                            else
                               HO_algorithm_triggers_STD_LTE = [HO_algorithm_triggers_STD_LTE; 0]; % list of HOs for each iteration 0-> No HO
                            end
                       end
                    else
                        t_TTT_vector_STD_LTE = [t_TTT_vector_STD_LTE; 0];
                        HO_algorithm_triggers_STD_LTE = [HO_algorithm_triggers_STD_LTE; 0]; % list of HOs for each iteration 0-> No HO
                        flag_t_TTT_STD_LTE = 0;
                        t_TTT_STD_LTE = 0;
                    end 

                    Host_AP_vector_STD_LTE = [Host_AP_vector_STD_LTE Host_AP_STD_LTE];
                    SINR_Host_AP_vector_STD_LTE = [SINR_Host_AP_vector_STD_LTE SINR_all_APs_dB(1,Host_AP_STD_LTE)];

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
                    %%% Smart HO Algorithm, implemented according to https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9186819, "Smart Handover for Hybrid LiFi and WiFi Networks"

                    % Counting blockages
                    if (Host_AP_Smart_HO ~= 17) && (SINR_all_APs_dB(1,Host_AP_Smart_HO) == -30) % blockage   
                        Blockage_vector_Smart_HO = [Blockage_vector_Smart_HO 0];
                    else
                        Blockage_vector_Smart_HO = [Blockage_vector_Smart_HO 1]; 
                    end

                    if SINR_all_APs_dB(1,current_best_AP_SINR_indx) >= SINR_all_APs_dB(1,Host_AP_Smart_HO) + delta_to_trigger_t_TTT % at least 1 dB of difference
                        if (Host_AP_Smart_HO ~= 17) && (SINR_all_APs_dB(1,Host_AP_Smart_HO) == -30) % blockage
%                              disp("something")
%                             prob2 = unifrnd(0,1);
%                             if prob2 < 0.1
                                prev_Host_AP_Smart_HO = Host_AP_Smart_HO;
                                Host_AP_Smart_HO = 17;    % Switch to WiFi
                                N_VHO_Smart_HO = N_VHO_Smart_HO + 1;
                                HO_algorithm_triggers_Smart_HO = [HO_algorithm_triggers_Smart_HO; 1]; % list of HOs for each iteration 1-> VHO
                                t_TTT_Smart_HO = 0;
                                flag_t_TTT_Smart_HO = 0; 
                                N_block_Smart_HO = N_block_Smart_HO + 1;
%                             else
%                                 prev_Host_AP_Smart_HO = Host_AP_Smart_HO;
%                                 HO_algorithm_triggers_Smart_HO = [HO_algorithm_triggers_Smart_HO; 0]; % list of HOs for each iteration 0-> NHO
%                                 t_TTT_Smart_HO = 0;
%                                 flag_t_TTT_Smart_HO = 0;  
%                             end

                        else % no blockage -> use the HO mechinism of LTE + Obejective Function of Smart HO
                            if flag_t_TTT_Smart_HO == 1 % t_TTT was already triggered
                                if best_AP_Smart_HO == current_best_AP_SINR_indx % if best_AP_Smart_HO is still the best
                                    t_TTT_Smart_HO = t_TTT_Smart_HO + 1;
                                else % rest the counters otherwise
                                    t_TTT_Smart_HO = 0;
                                    flag_t_TTT_Smart_HO = 0;
                                end
                            else % t_TTT has not been triggered before
                                flag_t_TTT_Smart_HO = 1;
                                t_TTT_Smart_HO = 1;
                                best_AP_Smart_HO = current_best_AP_SINR_indx; % best_AP_Smart_HO is the AP under evaluation. I want to know if it will keep being the best for TTT seconds.
                                SINR_when_t_TTT_was_triggered = SINR_all_APs_Watt;
                            end

                            t_TTT_vector_Smart_HO = [t_TTT_vector_Smart_HO; t_TTT_Smart_HO]; % values of TTT counter for each iteration

                            if t_TTT_Smart_HO == 2 % The AP is the best for 130ms (TTT)

                               % Find new target AP and calculate Objective Function (OF)
                               SINR_difference_Watt = (SINR_all_APs_Watt - SINR_when_t_TTT_was_triggered)./T_c_theta*2;
                               Obj_function_vector = zeros(1,length(SINR_difference_Watt));
                               for ite_smart_HO = 1 : length(SINR_difference_Watt)
                                   if ite_smart_HO <= 16  % LiFi
                                       Obj_function_vector(1,ite_smart_HO) = 1*(SINR_when_t_TTT_was_triggered(1,ite_smart_HO) + SINR_difference_Watt(1,ite_smart_HO));
                                   else % WiFi
                                       Obj_function_vector(1,ite_smart_HO) = Obj_function_coeficient * (SINR_when_t_TTT_was_triggered(1,ite_smart_HO) + SINR_difference_Watt(1,ite_smart_HO));
                                   end
                               end
                               [best_value_Obj_Funct Target_AP_Smart_HO] = max(Obj_function_vector); % make the HO decision

                               if SINR_all_APs_dB(1,Target_AP_Smart_HO) > SINR_all_APs_dB(1,Host_AP_Smart_HO)
                                   prev_Host_AP_Smart_HO = Host_AP_Smart_HO;
                                   Host_AP_Smart_HO = Target_AP_Smart_HO;
                                   % Counting HOs
                                   if prev_Host_AP_Smart_HO ~= 17 && Host_AP_Smart_HO == 17 %LiFi to WiFi
                                       N_VHO_Smart_HO = N_VHO_Smart_HO + 1;
                                       HO_algorithm_triggers_Smart_HO = [HO_algorithm_triggers_Smart_HO; 1]; % list of HOs for each iteration 1-> VHO
                                   elseif prev_Host_AP_Smart_HO == 17 && Host_AP_Smart_HO ~= 17 % WiFi to LiFi
                                       N_VHO_Smart_HO = N_VHO_Smart_HO + 1;
                                       HO_algorithm_triggers_Smart_HO = [HO_algorithm_triggers_Smart_HO; 1]; % list of HOs for each iteration 1-> VHO
                                   elseif prev_Host_AP_Smart_HO ~= 17 && Host_AP_Smart_HO ~= 17 % LiFi to LiFi
                                       N_HHO_Smart_HO = N_HHO_Smart_HO + 1;
                                       HO_algorithm_triggers_Smart_HO = [HO_algorithm_triggers_Smart_HO; 2]; % list of HOs for each iteration 2-> HHO
                                   elseif prev_Host_AP_Smart_HO == 17 && Host_AP_Smart_HO == 17 % WiFi to WiFi
                                       HO_algorithm_triggers_Smart_HO = [HO_algorithm_triggers_Smart_HO; 0]; % list of HOs for each iteration 0-> No HO                           
                                   end
%                                    prev_Host_AP_Smart_HO = Host_AP_Smart_HO;
                                   t_TTT_Smart_HO = 0;
                                   flag_t_TTT_Smart_HO = 0;
                               else
                                   HO_algorithm_triggers_Smart_HO = [HO_algorithm_triggers_Smart_HO; 0]; % list of HOs for each iteration 0-> No HO 
                                   t_TTT_Smart_HO = 0;
                                   flag_t_TTT_Smart_HO = 0;
                               end
                            else
                               HO_algorithm_triggers_Smart_HO = [HO_algorithm_triggers_Smart_HO; 0]; % list of HOs for each iteration 0-> No HO
                            end
                        end

                    else
                        t_TTT_vector_Smart_HO = [t_TTT_vector_Smart_HO; 0];
                        HO_algorithm_triggers_Smart_HO = [HO_algorithm_triggers_Smart_HO; 0]; % list of HOs for each iteration 0-> No HO
                        flag_t_TTT_Smart_HO = 0;
                        t_TTT_Smart_HO = 0;
                    end 

                    Host_AP_vector_Smart_HO = [Host_AP_vector_Smart_HO Host_AP_Smart_HO];
                    SINR_Host_AP_vector_Smart_HO = [SINR_Host_AP_vector_Smart_HO SINR_all_APs_dB(1,Host_AP_Smart_HO)];
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
                    %%% Proposed Algorithm (ML-based)
                    
                    % Counting blockages
                    if (Host_AP_STD_LTE_ML_block ~= 17) && (SINR_all_APs_dB(1,Host_AP_STD_LTE_ML_block) == -30) % blockage   
                        Blockage_vector_STD_LTE_ML_block = [Blockage_vector_STD_LTE_ML_block 0];
                    else
                        Blockage_vector_STD_LTE_ML_block = [Blockage_vector_STD_LTE_ML_block 1];
                    end
                    
                    %%% Cheking if it is necessary to make a HO decision
                    if SINR_all_APs_dB(1,current_best_AP_SINR_indx) >= SINR_all_APs_dB(1,Host_AP_STD_LTE_ML_block) + delta_to_trigger_t_TTT_STD_LTE_ML_block % at least 1 dB of difference
%                         if (Host_AP_STD_LTE_ML_block~=17) && (SINR_all_APs_dB(1,Host_AP_STD_LTE_ML_block) == -30) 
                        if not(Host_AP_STD_LTE_ML_block==17) && (SINR_all_APs_dB(1,Host_AP_STD_LTE_ML_block) == -30) 
                            % If blockage switch to WiFi
                            if Host_AP_STD_LTE_ML_block==17
                                disp("wrong")
                                break;
                            else
                                disp("correct")
                                % update q table for previous action
                                % Check HO failure
                                SINR_current_host_AP = SINR_all_APs_dB(1,Host_AP_STD_LTE_ML_block);
                                if SINR_current_host_AP > SINR_HO_failure_STD_LTE_ML_block
                                    HO_failure_vector_STD_LTE_ML_block = [HO_failure_vector_STD_LTE_ML_block; 1]; 
                                    HO_failure = 1; % no failure
                                else
                                    HO_failure_vector_STD_LTE_ML_block = [HO_failure_vector_STD_LTE_ML_block; 0];
                                    HO_failure = 0;
                                end
                                
                                % User satisfaction degree in terms of data rate
                                if Host_AP_STD_LTE_ML_block == 17 % WiFi link
                                    new_datarate = B_WiFi * log2(1 + sinr_wifi_matrix(1,1));
                                    new_datarate_Mbps = new_datarate/1000000;
                                else % LiFi link
                                    new_datarate = B_LiFi * log2(1 + (exp(1)/2*pi)* sinr_lifi_matrix_total(1,Host_AP_STD_LTE_ML_block));
                                    new_datarate_Mbps = new_datarate/1000000;
                                end 
%                                 usr_satisfaction_datarate = new_datarate/prev_datarate;
%                                 if usr_satisfaction_datarate > 1 % truncating it to max 1
%                                     usr_satisfaction_datarate = 1;
%                                 end
                                
                                if new_datarate == 0 || prev_datarate == 0
                                    usr_satisfaction_datarate = 0;
                                else
                                    usr_satisfaction_datarate = new_datarate/prev_datarate;
                                    if usr_satisfaction_datarate > 1 % truncating it to max 1
                                        usr_satisfaction_datarate = 1;
                                    end
                                end
                                
                                % The HO Cost depends on the consequnces of
                                % previous action in previous state
                                
                                if prev_action == 3 % Do Nothing
                                    HO_cost = 0;
                                else
                                    if prev_Host_AP_STD_LTE_ML_block ~= 17 && Host_AP_STD_LTE_ML_block == 17 %LiFi to WiFi
                                       HO_cost = 0.9;
                                    elseif prev_Host_AP_STD_LTE_ML_block == 17 && Host_AP_STD_LTE_ML_block ~= 17 % WiFi to LiFi
                                       HO_cost = 0.9;
                                    elseif prev_Host_AP_STD_LTE_ML_block ~= 17 && Host_AP_STD_LTE_ML_block ~= 17 % LiFi to LiFi
                                       HO_cost = 0.2;
                                    elseif prev_Host_AP_STD_LTE_ML_block == 17 && Host_AP_STD_LTE_ML_block == 17 % WiFi to WiFi
                                       HO_cost = 0.2;
                                    end 
                                end
     
                                reward = HO_failure * (usr_satisfaction_datarate*beta + (1-beta)*(1-HO_cost));
%                                 reward = HO_failure * usr_satisfaction_datarate;
                                reward_vector = [reward_vector reward];
                                
                                % Define current state (st+1)
                                
                                %%% Defining the States of the RL algorithm

                                % state: serving networks (1 -> LiFi, 0 -> WiFi) 
                                % direction of movement (0 -> to wall, 1 -> to cell center, 2 -> crossing cell edge
                                % user satisfaction degree (1 -> satisfied, 0 -> not satisfied)

                                % User satisfaction degree in terms of data rate (only considers a threshold)
                                if SINR_all_APs_dB(1,Host_AP_STD_LTE_ML_block) > SINR_user_requirement 
                                    user_satisfaction = 1; % user is satisfied
                                else
                                    user_satisfaction = 0; % user not satisfied
                                end
                                
                                % Defining the input parameters of the ML algotirhm to determine user trajectory
                                if n>3                                
                                    SNR_all_APs_dB_prev_block = [snr_lifi_matrix_dB_ite_total_v1(n-2,:) snr_wifi_matrix_dB_ite(n-2,1)];
                                else
                                    SNR_all_APs_dB_prev_block = [snr_lifi_matrix_dB_ite_total_v1(n,:) snr_wifi_matrix_dB_ite(n,1)];
                                end
                                Current_SNR_LiFi_WiFi = [snr_lifi_matrix_dB_ite_total_v1(n,:) snr_wifi_matrix_dB_ite(n,1)];
                                delta_SNR = SNR_all_APs_dB - SNR_all_APs_dB_prev_block;
                                input_ML = [Host_AP_STD_LTE_ML_block SNR_all_APs_dB(1,1:16) delta_SNR];

                                % Call ML algorithm to make prediction
                                user_traj_predict = xgboost_test(input_ML,best_model,0);
                                user_traj_predict_vector_STD_LTE_ML_block = [user_traj_predict_vector_STD_LTE_ML_block user_traj_predict];
                                
                                switch user_traj_predict
                                    case 0 % walking to the wall
                                        if user_satisfaction == 1 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 1;
                                           disp('LiFi, walking to wall, user satisfied')
                                        elseif user_satisfaction == 0 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 4; 
                                           disp('LiFi, walking to wall, user not satisfied')
                                        elseif user_satisfaction == 1 && Host_AP_STD_LTE_ML_block == 17 
                                           current_state = 7;
                                           disp('WiFi, walking to wall, user satisfied')
                                        else
                                           current_state = 10;
                                           disp('WiFi, walking to wall, user not satisfied')
                                        end

                                    case 1 % walking to cell center
                                        if user_satisfaction == 1 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 2;
                                           disp('LiFi, walking to cell center, user satisfied')
                                        elseif user_satisfaction == 0 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 5; 
                                           disp('LiFi, walking to cell center, user not satisfied')
                                        elseif user_satisfaction == 1 && Host_AP_STD_LTE_ML_block == 17 
                                           current_state = 8;
                                           disp('WiFi, walking to cell center, user satisfied')
                                        else
                                           current_state = 11;
                                           disp('WiFi, walking to cell center, user not satisfied')
                                        end

                                    case 2 % crosssing cell edge
                                        if user_satisfaction == 1 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 3;
                                           disp('LiFi, crosssing cell edge, user satisfied')
                                        elseif user_satisfaction == 0 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 6; 
                                           disp('LiFi, crosssing cell edge, user not satisfied')
                                        elseif user_satisfaction == 1 && Host_AP_STD_LTE_ML_block == 17 
                                           current_state = 9;
                                           disp('WiFi, crosssing cell edge, user satisfied')
                                        else
                                           current_state = 12;
                                           disp('WiFi, crosssing cell edge, user not satisfied')
                                        end
                                    otherwise
                                        current_state = 2;
                                end
                         
                                % update the Q-table for previous action and state
                                % greedy policy (Q-Learning)
                                [max_q_value_current_state,index_max_q_value] = max(q_table(current_state,:));
                                q_table(prev_state,prev_action) = q_table(prev_state,prev_action) + learning_rate*(reward + discount_rate*max_q_value_current_state - q_table(prev_state,prev_action));
                                current_q_value = q_table(prev_state,prev_action) + learning_rate*(reward + discount_rate*max_q_value_current_state - q_table(prev_state,prev_action));
                                if isnan(current_q_value)
                                    disp('NaN')
                                end
                                added_value = learning_rate*(reward + discount_rate*max_q_value_current_state - q_table(prev_state,prev_action));
                                if added_value < 0
                                    disp('Negative added value')
                                end
                                
                                added_q_value_vector = [added_q_value_vector added_value];
                                
                                switch prev_state
                                    case 1
                                        q_table_story_S1 = [q_table_story_S1 q_table(1,:)'];
                                    case 2
                                        q_table_story_S2 = [q_table_story_S2 q_table(2,:)'];
                                    case 3
                                        q_table_story_S3 = [q_table_story_S3 q_table(3,:)'];
                                    case 4
                                        q_table_story_S4 = [q_table_story_S4 q_table(4,:)'];
                                    case 5
                                        q_table_story_S5 = [q_table_story_S5 q_table(5,:)'];
                                    case 6
                                        q_table_story_S6 = [q_table_story_S6 q_table(6,:)'];
                                    case 7
                                        q_table_story_S7 = [q_table_story_S7 q_table(7,:)'];
                                    case 8
                                        q_table_story_S8 = [q_table_story_S8 q_table(8,:)'];
                                    case 9
                                        q_table_story_S9 = [q_table_story_S9 q_table(9,:)'];
                                    case 10
                                        q_table_story_S10 = [q_table_story_S10 q_table(10,:)'];
                                    case 11
                                        q_table_story_S11 = [q_table_story_S11 q_table(11,:)'];
                                    case 12
                                        q_table_story_S12 = [q_table_story_S12 q_table(12,:)'];
                                end
                                
                                % Make a HO decision
                                Host_AP_STD_LTE_ML_block = 17;    % Switch to WiFi
                                prev_Host_AP_STD_LTE_ML_block = Host_AP_STD_LTE_ML_block;
%                                 prev_action = 3;
%                                 prev_state = 10;

                                prev_state = current_state;
                                prev_action = 2;
                                prev_datarate = new_datarate;                              
                                actions_vector_STD_LTE_ML_block = [actions_vector_STD_LTE_ML_block 2];
                                N_VHO_STD_LTE_ML_block = N_VHO_STD_LTE_ML_block + 1;
                                t_TTT_STD_LTE_ML_block = 0;
                                flag_t_TTT_STD_LTE_ML_block = 0;
                                HO_algorithm_triggers_STD_LTE_ML_block = [HO_algorithm_triggers_STD_LTE_ML_block; 1]; % list of HOs for each iteration 1-> VHO
                                N_block_STD_LTE_ML_block = N_block_STD_LTE_ML_block + 1;
                            end   
%                             prev_Host_AP_STD_LTE_ML_block = Host_AP_STD_LTE_ML_block;  
%                             else
%                                 prev_Host_AP_STD_LTE_ML_block = Host_AP_STD_LTE_ML_block;
%                                 [current_best_SINR_LiFi current_best_LiFi_AP_indx] = max(SINR_all_APs_dB(1,1:16));
%                                 Host_AP_STD_LTE_ML_block = current_best_LiFi_AP_indx;
%                                 actions_vector_STD_LTE_ML_block = [actions_vector_STD_LTE_ML_block 1];
%                                 N_HHO_STD_LTE_ML_block = N_HHO_STD_LTE_ML_block + 1;
%                                 t_TTT_STD_LTE_ML_block = 0;
%                                 flag_t_TTT_STD_LTE_ML_block = 0;
%                                 HO_algorithm_triggers_STD_LTE_ML_block = [HO_algorithm_triggers_STD_LTE_ML_block; 2]; % list of HOs for each iteration 2-> HHO
%     %                             prev_Host_AP_STD_LTE_ML_block = Host_AP_STD_LTE_ML_block;  
%                             end
                            
                           
%                             prev_Host_AP_STD_LTE_ML_block = Host_AP_STD_LTE_ML_block;
%                             Host_AP_STD_LTE_ML_block = 17;    % Switch to WiFi
%                             actions_vector_STD_LTE_ML_block = [actions_vector_STD_LTE_ML_block 2];
%                             N_VHO_STD_LTE_ML_block = N_VHO_STD_LTE_ML_block + 1;
%                             t_TTT_STD_LTE_ML_block = 0;
%                             flag_t_TTT_STD_LTE_ML_block = 0;
%                             HO_algorithm_triggers_STD_LTE_ML_block = [HO_algorithm_triggers_STD_LTE_ML_block; 1]; % list of HOs for each iteration 1-> VHO
% %                             prev_Host_AP_STD_LTE_ML_block = Host_AP_STD_LTE_ML_block;  
                        
                        
                        else   
                            % Mechanism to trigger the t_TTT counter (as in LTE)
                            if flag_t_TTT_STD_LTE_ML_block == 1 % t_TTT was already triggered
                                if best_AP_STD_LTE_ML_block == current_best_AP_SINR_indx
                                    t_TTT_STD_LTE_ML_block = t_TTT_STD_LTE_ML_block + 1;
                                else
                                    t_TTT_STD_LTE_ML_block = 0;
                                    flag_t_TTT_STD_LTE_ML_block = 0;
                                end
                            else % t_TTT has not been triggered before
                                flag_t_TTT_STD_LTE_ML_block = 1;
                                t_TTT_STD_LTE_ML_block = 1;
                                best_AP_STD_LTE_ML_block = current_best_AP_SINR_indx;
                                SNR_all_APs_dB_when_t_TTT_triggered = SNR_all_APs_dB;
                            end
                            t_TTT_vector_STD_LTE_ML_block = [t_TTT_vector_STD_LTE_ML_block; t_TTT_STD_LTE_ML_block];
                            
                            if t_TTT_STD_LTE_ML_block == Threshold_t_TTT_STD_LTE_ML_block
                                t_TTT_STD_LTE_ML_block = 0;
                                flag_t_TTT_STD_LTE_ML_block = 0;
                                
                                % Make a HO decision using RL
                                % First calculate the reward for previous action (a) and state(s)

                                % Check HO failure
                                SINR_current_host_AP = SINR_all_APs_dB(1,Host_AP_STD_LTE_ML_block);
                                if SINR_current_host_AP > SINR_HO_failure_STD_LTE_ML_block
                                    HO_failure_vector_STD_LTE_ML_block = [HO_failure_vector_STD_LTE_ML_block; 1]; 
                                    HO_failure = 1; % no failure
                                else
                                    HO_failure_vector_STD_LTE_ML_block = [HO_failure_vector_STD_LTE_ML_block; 0];
                                    HO_failure = 0;
                                end
                                
                                % User satisfaction degree in terms of data rate
                                if Host_AP_STD_LTE_ML_block == 17 % WiFi link
                                    new_datarate = B_WiFi * log2(1 + sinr_wifi_matrix(1,1));
                                    new_datarate_Mbps = new_datarate/1000000;
                                else % LiFi link
                                    new_datarate = B_LiFi * log2(1 + (exp(1)/2*pi)* sinr_lifi_matrix_total(1,Host_AP_STD_LTE_ML_block));
                                    new_datarate_Mbps = new_datarate/1000000;
                                end 
%                                 usr_satisfaction_datarate = new_datarate/prev_datarate;
%                                 if usr_satisfaction_datarate > 1 % truncating it to max 1
%                                     usr_satisfaction_datarate = 1;
%                                 end
                                
                                if new_datarate == 0 || prev_datarate == 0
                                    usr_satisfaction_datarate = 0;
                                else
                                    usr_satisfaction_datarate = new_datarate/prev_datarate;
                                    if usr_satisfaction_datarate > 1 % truncating it to max 1
                                        usr_satisfaction_datarate = 1;
                                    end
                                end
                                
                                % The HO Cost depends on the consequnces of
                                % previous action in previous state
                                
                                if prev_action == 3 % Do Nothing
                                    HO_cost = 0;
                                else
                                    if prev_Host_AP_STD_LTE_ML_block ~= 17 && Host_AP_STD_LTE_ML_block == 17 %LiFi to WiFi
                                       HO_cost = 0.9;
                                    elseif prev_Host_AP_STD_LTE_ML_block == 17 && Host_AP_STD_LTE_ML_block ~= 17 % WiFi to LiFi
                                       HO_cost = 0.9;
                                    elseif prev_Host_AP_STD_LTE_ML_block ~= 17 && Host_AP_STD_LTE_ML_block ~= 17 % LiFi to LiFi
                                       HO_cost = 0.2;
                                    elseif prev_Host_AP_STD_LTE_ML_block == 17 && Host_AP_STD_LTE_ML_block == 17 % WiFi to WiFi
                                       HO_cost = 0.2;
                                    end 
                                end
     
                                reward = HO_failure * (usr_satisfaction_datarate*beta + (1-beta)*(1-HO_cost));
%                                 reward = HO_failure * usr_satisfaction_datarate;
                                reward_vector = [reward_vector reward];
                                
                                % Define current state (st+1)
                                
                                %%% Defining the States of the RL algorithm

                                % state: serving networks (1 -> LiFi, 0 -> WiFi) 
                                % direction of movement (0 -> to wall, 1 -> to cell center, 2 -> crossing cell edge
                                % user satisfaction degree (1 -> satisfied, 0 -> not satisfied)

                                % User satisfaction degree in terms of data rate (only considers a threshold)
                                if SINR_all_APs_dB(1,Host_AP_STD_LTE_ML_block) > SINR_user_requirement 
                                    user_satisfaction = 1; % user is satisfied
                                else
                                    user_satisfaction = 0; % user not satisfied
                                end
                                
                                % Defining the input parameters of the ML algotirhm to determine user trajectory
                                Current_SNR_LiFi_WiFi = [snr_lifi_matrix_dB_ite_total_v1(n,:) snr_wifi_matrix_dB_ite(n,1)];
                                delta_SNR = SNR_all_APs_dB - SNR_all_APs_dB_when_t_TTT_triggered;
                                input_ML = [Host_AP_STD_LTE_ML_block SNR_all_APs_dB(1,1:16) delta_SNR];

                                % Call ML algorithm to make prediction
                                user_traj_predict = xgboost_test(input_ML,best_model,0);
                                user_traj_predict_vector_STD_LTE_ML_block = [user_traj_predict_vector_STD_LTE_ML_block user_traj_predict];
                                
                                switch user_traj_predict
                                    case 0 % walking to the wall
                                        if user_satisfaction == 1 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 1;
                                           disp('LiFi, walking to wall, user satisfied')
                                        elseif user_satisfaction == 0 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 4; 
                                           disp('LiFi, walking to wall, user not satisfied')
                                        elseif user_satisfaction == 1 && Host_AP_STD_LTE_ML_block == 17 
                                           current_state = 7;
                                           disp('WiFi, walking to wall, user satisfied')
                                        else
                                           current_state = 10;
                                           disp('WiFi, walking to wall, user not satisfied')
                                        end

                                    case 1 % walking to cell center
                                        if user_satisfaction == 1 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 2;
                                           disp('LiFi, walking to cell center, user satisfied')
                                        elseif user_satisfaction == 0 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 5; 
                                           disp('LiFi, walking to cell center, user not satisfied')
                                        elseif user_satisfaction == 1 && Host_AP_STD_LTE_ML_block == 17 
                                           current_state = 8;
                                           disp('WiFi, walking to cell center, user satisfied')
                                        else
                                           current_state = 11;
                                           disp('WiFi, walking to cell center, user not satisfied')
                                        end

                                    case 2 % crosssing cell edge
                                        if user_satisfaction == 1 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 3;
                                           disp('LiFi, crosssing cell edge, user satisfied')
                                        elseif user_satisfaction == 0 && Host_AP_STD_LTE_ML_block ~= 17 
                                           current_state = 6; 
                                           disp('LiFi, crosssing cell edge, user not satisfied')
                                        elseif user_satisfaction == 1 && Host_AP_STD_LTE_ML_block == 17 
                                           current_state = 9;
                                           disp('WiFi, crosssing cell edge, user satisfied')
                                        else
                                           current_state = 12;
                                           disp('WiFi, crosssing cell edge, user not satisfied')
                                        end
                                    otherwise
                                        current_state = 2;
                                end
                         
                                % update the Q-table for previous action and state
                                % greedy policy (Q-Learning)
                                [max_q_value_current_state,index_max_q_value] = max(q_table(current_state,:));
                                q_table(prev_state,prev_action) = q_table(prev_state,prev_action) + learning_rate*(reward + discount_rate*max_q_value_current_state - q_table(prev_state,prev_action));
                                current_q_value = q_table(prev_state,prev_action) + learning_rate*(reward + discount_rate*max_q_value_current_state - q_table(prev_state,prev_action));
                                if isnan(current_q_value)
                                    disp('NaN')
                                end
                                added_value = learning_rate*(reward + discount_rate*max_q_value_current_state - q_table(prev_state,prev_action));
                                if added_value < 0
                                    disp('Negativeadded value')
                                end
                                added_q_value_vector = [added_q_value_vector added_value];
                                
%                                 q_table_story_S1 = [q_table_story_S1 q_table(1,:)'];
%                                 q_table_story_S2 = [q_table_story_S2 q_table(2,:)'];
%                                 q_table_story_S3 = [q_table_story_S3 q_table(3,:)'];
%                                 q_table_story_S4 = [q_table_story_S4 q_table(4,:)'];
%                                 q_table_story_S5 = [q_table_story_S5 q_table(5,:)'];
%                                 q_table_story_S6 = [q_table_story_S6 q_table(6,:)'];
%                                 q_table_story_S7 = [q_table_story_S7 q_table(7,:)'];
%                                 q_table_story_S8 = [q_table_story_S8 q_table(8,:)'];
%                                 q_table_story_S9 = [q_table_story_S9 q_table(9,:)'];
%                                 q_table_story_S10 = [q_table_story_S10 q_table(10,:)'];
%                                 q_table_story_S11 = [q_table_story_S11 q_table(11,:)'];
%                                 q_table_story_S12 = [q_table_story_S12 q_table(12,:)'];
                                switch prev_state
                                    case 1
                                        q_table_story_S1 = [q_table_story_S1 q_table(1,:)'];
                                    case 2
                                        q_table_story_S2 = [q_table_story_S2 q_table(2,:)'];
                                    case 3
                                        q_table_story_S3 = [q_table_story_S3 q_table(3,:)'];
                                    case 4
                                        q_table_story_S4 = [q_table_story_S4 q_table(4,:)'];
                                    case 5
                                        q_table_story_S5 = [q_table_story_S5 q_table(5,:)'];
                                    case 6
                                        q_table_story_S6 = [q_table_story_S6 q_table(6,:)'];
                                    case 7
                                        q_table_story_S7 = [q_table_story_S7 q_table(7,:)'];
                                    case 8
                                        q_table_story_S8 = [q_table_story_S8 q_table(8,:)'];
                                    case 9
                                        q_table_story_S9 = [q_table_story_S9 q_table(9,:)'];
                                    case 10
                                        q_table_story_S10 = [q_table_story_S10 q_table(10,:)'];
                                    case 11
                                        q_table_story_S11 = [q_table_story_S11 q_table(11,:)'];
                                    case 12
                                        q_table_story_S12 = [q_table_story_S12 q_table(12,:)'];
                                end
                                
                                % Then select a new action to make the HO decision
                                % A1 -> Select the best LiFi AP 
                                % A2 -> Select the best WiFi AP
                                % A3 -> Do nothing
                                
                                % Select the action of highest value
%                                 [val, action] = max(q_table(current_state,:));% greedy policy

                                % e-greedy policy

                                prob = unifrnd(0,1);
                                if prob > epsilon
                                    [val, action] = max(q_table(current_state,:));
                                    actions_vector_random = [actions_vector_random 1];  % 1 -> Exploitation
                                else
                                    action = randi(3);
                                    actions_vector_random = [actions_vector_random 0]; % 0 -> Exploration
                                end

                                actions_vector_STD_LTE_ML_block = [actions_vector_STD_LTE_ML_block action]; 
                                prev_Host_AP_STD_LTE_ML_block = Host_AP_STD_LTE_ML_block;
                                if action == 1 % Select the best LiFi AP
                                    [current_best_SINR_LiFi current_best_LiFi_AP_indx] = max(SINR_all_APs_dB(1,1:16));
                                    Host_AP_STD_LTE_ML_block = current_best_LiFi_AP_indx;
                                  
                                elseif action == 2 % Select the best WiFi AP, in our case we have only one
                                    Host_AP_STD_LTE_ML_block = 17; % WiFi
                                else % Do nothing
                                    Host_AP_STD_LTE_ML_block = Host_AP_STD_LTE_ML_block; 
                                end
                                
                                % Counting the number of HOs
                                if prev_Host_AP_STD_LTE_ML_block ~= 17 && Host_AP_STD_LTE_ML_block == 17 %LiFi to WiFi
                                   N_VHO_STD_LTE_ML_block = N_VHO_STD_LTE_ML_block + 1;
                                   HO_algorithm_triggers_STD_LTE_ML_block = [HO_algorithm_triggers_STD_LTE_ML_block; 1]; % list of HOs for each iteration 1-> VHO
                                elseif prev_Host_AP_STD_LTE_ML_block == 17 && Host_AP_STD_LTE_ML_block ~= 17 % WiFi to LiFi
                                   N_VHO_STD_LTE_ML_block = N_VHO_STD_LTE_ML_block + 1;
                                   HO_algorithm_triggers_STD_LTE_ML_block = [HO_algorithm_triggers_STD_LTE_ML_block; 1]; % list of HOs for each iteration 1-> VHO
                                elseif prev_Host_AP_STD_LTE_ML_block ~= 17 && Host_AP_STD_LTE_ML_block ~= 17 % LiFi to LiFi
                                   N_HHO_STD_LTE_ML_block = N_HHO_STD_LTE_ML_block + 1;
                                   HO_algorithm_triggers_STD_LTE_ML_block = [HO_algorithm_triggers_STD_LTE_ML_block; 2]; % list of HOs for each iteration 2-> HHO
                                elseif prev_Host_AP_STD_LTE_ML_block == 17 && Host_AP_STD_LTE_ML_block == 17 % WiFi to WiFi
                                   HO_algorithm_triggers_STD_LTE_ML_block = [HO_algorithm_triggers_STD_LTE_ML_block; 0]; % list of HOs for each iteration 0-> No HO                           
                                end                                    
%                                 prev_Host_AP_STD_LTE_ML_block = Host_AP_STD_LTE_ML_block;
                                prev_state = current_state;
                                prev_action = action;
                                prev_datarate = new_datarate;
                                
                            else
                                HO_algorithm_triggers_STD_LTE_ML_block = [HO_algorithm_triggers_STD_LTE_ML_block; 0];
                                actions_vector_STD_LTE_ML_block = [actions_vector_STD_LTE_ML_block 3];
                            end
                        end
                    else
                        t_TTT_vector_STD_LTE_ML_block = [t_TTT_vector_STD_LTE_ML_block; 0];
                        HO_algorithm_triggers_STD_LTE_ML_block = [HO_algorithm_triggers_STD_LTE_ML_block; 0];
                        flag_t_TTT_STD_LTE_ML_block = 0;
                        t_TTT_STD_LTE_ML_block = 0;
                        actions_vector_STD_LTE_ML_block = [actions_vector_STD_LTE_ML_block 3];
                    end

                    Host_AP_vector_STD_LTE_ML_block = [Host_AP_vector_STD_LTE_ML_block Host_AP_STD_LTE_ML_block];
                    SINR_Host_AP_vector_STD_LTE_ML_block = [SINR_Host_AP_vector_STD_LTE_ML_block SINR_all_APs_dB(1,Host_AP_STD_LTE_ML_block)];                                      
                    N_HO_failures_STD_LTE_ML_block = length(HO_failure_vector_STD_LTE_ML_block) - sum(HO_failure_vector_STD_LTE_ML_block);

                    %%% Preparing the next iteration whithin the while loop

                    pos_ue_n_1 = pos_ue_n;        
                    t_move = t_move + T_c_theta;  
                    n = n + 1;
                    blocking_elements = [];
                end 
                if t_move ~= (D_k/speed) - T_c_theta

                % This is to get ready for the next excursion. It adjusts the Pn-1 and Pk-1 in case Pn is beyond Pk.
                    pos_ue_n_1 = pos_ue_k;
                    pos_ue_k_1 = pos_ue_k;
                    pos_ue_vector(n,:) = pos_ue_n_1;
                end 
                N_steps_per_path = [N_steps_per_path n-1];
            end

            %%% Throughput calculation
            
            %STD-LTE

            Thr_vector_Mbps_STD_LTE = zeros(1,n-1);

            for k_thr = 1: n-1
                if Host_AP_vector_STD_LTE(1,k_thr) == 17  % WiFi link
                    new_datarate = B_WiFi * log2(1 + sinr_wifi_matrix_watt_ite(k_thr,1)*Blockage_vector_STD_LTE(1,k_thr));
                    new_datarate_Mbps = new_datarate/1000000;
                    Thr_vector_Mbps_STD_LTE(1,k_thr) = new_datarate_Mbps;

                else  % LiFi link
                    new_datarate = B_LiFi * log2(1 + (exp(1)/2*pi)* sinr_lifi_matrix_watt_ite_total_v1(k_thr,Host_AP_vector_STD_LTE(1,k_thr))*Blockage_vector_STD_LTE(1,k_thr));
                    new_datarate_Mbps = new_datarate/1000000;
                    Thr_vector_Mbps_STD_LTE(1,k_thr) = new_datarate_Mbps;
                end 

                if isnan(new_datarate_Mbps)
                        disp('NaN')
                end
            end

            Avr_thr_STD_LTE = sum(Thr_vector_Mbps_STD_LTE)/length(Thr_vector_Mbps_STD_LTE);


            for kk_thr = 1: n-1
                if kk_thr + 3 < n-1
                    if HO_algorithm_triggers_STD_LTE(kk_thr,1) == 1
                        Thr_vector_Mbps_STD_LTE(1,kk_thr:kk_thr+3) = 0;
                    elseif HO_algorithm_triggers_STD_LTE(kk_thr,1) == 2
                        Thr_vector_Mbps_STD_LTE(1,kk_thr:kk_thr+1) = 0;
                    end   
                end
            end

            Avr_thr_STD_LTE_with_Overhead = sum(Thr_vector_Mbps_STD_LTE)/length(Thr_vector_Mbps_STD_LTE);

            % Smart HO
            Thr_vector_Mbps_Smart_HO = zeros(1,n-1);

            for k_thr = 1: n-1
                if Host_AP_vector_Smart_HO(1,k_thr) == 17  % WiFi link
                    new_datarate = B_WiFi * log2(1 + sinr_wifi_matrix_watt_ite(k_thr,1)*Blockage_vector_Smart_HO(1,k_thr));
                    new_datarate_Mbps = new_datarate/1000000;
                    Thr_vector_Mbps_Smart_HO(1,k_thr) = new_datarate_Mbps;

                else  % LiFi link
                    new_datarate = B_LiFi * log2(1 + (exp(1)/2*pi)* sinr_lifi_matrix_watt_ite_total_v1(k_thr,Host_AP_vector_Smart_HO(1,k_thr))*Blockage_vector_Smart_HO(1,k_thr));
                    new_datarate_Mbps = new_datarate/1000000;
                    Thr_vector_Mbps_Smart_HO(1,k_thr) = new_datarate_Mbps;
                end 

                if isnan(new_datarate_Mbps)
                        disp('NaN')
                end
            end

            Avr_thr_Smart_HO = sum(Thr_vector_Mbps_Smart_HO)/length(Thr_vector_Mbps_Smart_HO);

            for kk_thr = 1: n-1
                if kk_thr + 3 < n-1
                    if HO_algorithm_triggers_Smart_HO(kk_thr,1) == 1
                        Thr_vector_Mbps_Smart_HO(1,kk_thr:kk_thr+3) = 0;
                    elseif HO_algorithm_triggers_Smart_HO(kk_thr,1) == 2
                        Thr_vector_Mbps_Smart_HO(1,kk_thr:kk_thr+1) = 0;
                    end   
                end
            end

            Avr_thr_Smart_HO_with_Overhead = sum(Thr_vector_Mbps_Smart_HO)/length(Thr_vector_Mbps_Smart_HO);
            
            %Proposed RL-based HO scheme
            Thr_vector_Mbps_STD_LTE_ML_block = zeros(1,n-1);

            for k_thr = 1: n-1
                if Host_AP_vector_STD_LTE_ML_block(1,k_thr) == 17  % WiFi link
                    new_datarate = B_WiFi * log2(1 + sinr_wifi_matrix_watt_ite(k_thr,1)*Blockage_vector_STD_LTE_ML_block(1,k_thr));
                    new_datarate_Mbps = new_datarate/1000000;
                    Thr_vector_Mbps_STD_LTE_ML_block(1,k_thr) = new_datarate_Mbps;

                else  % LiFi link
                    new_datarate = B_LiFi * log2(1 + (exp(1)/2*pi)* sinr_lifi_matrix_watt_ite_total_v1(k_thr,Host_AP_vector_STD_LTE_ML_block(1,k_thr))*Blockage_vector_STD_LTE_ML_block(1,k_thr));
                    new_datarate_Mbps = new_datarate/1000000;
                    Thr_vector_Mbps_STD_LTE_ML_block(1,k_thr) = new_datarate_Mbps;
                end 

                if isnan(new_datarate_Mbps)
                        disp('NaN')
                end
            end

            Avr_thr_STD_LTE_ML_block = sum(Thr_vector_Mbps_STD_LTE_ML_block)/length(Thr_vector_Mbps_STD_LTE_ML_block);


            for kk_thr = 1: n-1
                if kk_thr + 3 < n-1
                    if HO_algorithm_triggers_STD_LTE_ML_block(kk_thr,1) == 1
                        Thr_vector_Mbps_STD_LTE_ML_block(1,kk_thr:kk_thr+3) = 0;
                    elseif HO_algorithm_triggers_STD_LTE_ML_block(kk_thr,1) == 2
                        Thr_vector_Mbps_STD_LTE_ML_block(1,kk_thr:kk_thr+1) = 0;
                    end   
                end
            end

            Avr_thr_STD_LTE_ML_block_with_Overhead = sum(Thr_vector_Mbps_STD_LTE_ML_block)/length(Thr_vector_Mbps_STD_LTE_ML_block);
            
            % Calculating the % of time connected to WiFi
          
            Index_Blockage_vector_STD_LTE = find(Blockage_vector_STD_LTE==0);
            Host_AP_vector_STD_LTE_aux = Host_AP_vector_STD_LTE;
            Host_AP_vector_STD_LTE_aux(1,Index_Blockage_vector_STD_LTE) = 0;
            
            SINR_Host_AP_vector_STD_LTE_aux = SINR_Host_AP_vector_STD_LTE;
            SINR_Host_AP_vector_STD_LTE_aux(1,Index_Blockage_vector_STD_LTE) = -30;

%             N_Conect_WiFi_STD_LTE = length(find(Host_AP_vector_STD_LTE==17)); % finds how many times it was connected to WiFi
            N_Conect_WiFi_STD_LTE = length(find(Host_AP_vector_STD_LTE_aux ==17)); % finds how many times it was connected to WiFi
            Percent_time_conect_WiFi_STD_LTE = N_Conect_WiFi_STD_LTE/(n-1);
            Percent_time_blockage_STD_LTE = length(Index_Blockage_vector_STD_LTE )/(n-1);
            Percent_time_conect_LiFi_STD_LTE = 1 - Percent_time_conect_WiFi_STD_LTE - Percent_time_blockage_STD_LTE;
            
            
            Index_Blockage_vector_Smart_HO = find(Blockage_vector_Smart_HO==0);
            Host_AP_vector_Smart_HO_aux = Host_AP_vector_Smart_HO;
            Host_AP_vector_Smart_HO_aux(1,Index_Blockage_vector_Smart_HO) = 0;
            
            SINR_Host_AP_vector_Smart_HO_aux = SINR_Host_AP_vector_Smart_HO;
            SINR_Host_AP_vector_Smart_HO_aux(1,Index_Blockage_vector_Smart_HO) = -30;
            
%             N_Conect_WiFi_Smart_HO = length(find(Host_AP_vector_Smart_HO==17)); % finds how many times it was connected to WiFi
            N_Conect_WiFi_Smart_HO = length(find(Host_AP_vector_Smart_HO_aux==17)); % finds how many times it was connected to WiFi
            Percent_time_conect_WiFi_Smart_HO = N_Conect_WiFi_Smart_HO/(n-1);
            Percent_time_blockage_Smart_HO = length(Index_Blockage_vector_Smart_HO )/(n-1);
            Percent_time_conect_LiFi_Smart_HO = 1 - Percent_time_conect_WiFi_Smart_HO - Percent_time_blockage_Smart_HO;
            
            Index_Blockage_vector_STD_LTE_ML_block = find(Blockage_vector_STD_LTE_ML_block==0);
            Host_AP_vector_STD_LTE_ML_block_aux = Host_AP_vector_STD_LTE_ML_block;
            Host_AP_vector_STD_LTE_ML_block_aux(1,Index_Blockage_vector_STD_LTE_ML_block) = 0;
            
            
            SINR_Host_AP_vector_STD_LTE_ML_block_aux = SINR_Host_AP_vector_STD_LTE_ML_block;
            SINR_Host_AP_vector_STD_LTE_ML_block_aux(1,Index_Blockage_vector_STD_LTE_ML_block) = -30;
            
%             N_Conect_WiFi_STD_LTE_ML_block = length(find(Host_AP_vector_STD_LTE_ML_block==17)); % finds how many times it was connected to WiFi
            N_Conect_WiFi_STD_LTE_ML_block = length(find(Host_AP_vector_STD_LTE_ML_block_aux==17)); % finds how many times it was connected to WiFi
            Percent_time_conect_WiFi_STD_LTE_ML_block = N_Conect_WiFi_STD_LTE_ML_block/(n-1);
            Percent_time_blockage_STD_LTE_ML_block = length(Index_Blockage_vector_STD_LTE_ML_block )/(n-1);
            Percent_time_conect_LiFi_STD_LTE_ML_block = 1 - Percent_time_conect_WiFi_STD_LTE_ML_block - Percent_time_blockage_STD_LTE_ML_block;
            
            % Coverage analysis
            
            Outage_prob_STD_LTE = length(find(SINR_Host_AP_vector_STD_LTE < Threshold_coverage))/(n-1);
            Outage_prob_Smart_HO = length(find(SINR_Host_AP_vector_Smart_HO < Threshold_coverage))/(n-1);
            Outage_prob_STD_LTE_ML_block = length(find(SINR_Host_AP_vector_STD_LTE_ML_block < Threshold_coverage))/(n-1);


            N_HHO_STD_LTE_per_ite(1,ite) = N_HHO_STD_LTE;
            N_VHO_STD_LTE_per_ite(1,ite) = N_VHO_STD_LTE; 
            Thr_STD_LTE_per_ite(1,ite) = Avr_thr_STD_LTE_with_Overhead;
            N_HHO_Smart_HO_per_ite(1,ite) = N_HHO_Smart_HO;
            N_VHO_Smart_HO_per_ite(1,ite) = N_VHO_Smart_HO; 
            Thr_Smart_HO_per_ite(1,ite) = Avr_thr_Smart_HO_with_Overhead;
            N_HHO_STD_LTE_ML_block_per_ite(1,ite) = N_HHO_STD_LTE_ML_block;
            N_VHO_STD_LTE_ML_block_per_ite(1,ite) = N_VHO_STD_LTE_ML_block; 
            Thr_STD_LTE_ML_block_per_ite(1,ite) = Avr_thr_STD_LTE_ML_block_with_Overhead;
            Percent_time_conect_WiFi_STD_LTE_ite(1,ite) = Percent_time_conect_WiFi_STD_LTE;
            Percent_time_conect_WiFi_Smart_HO_ite(1,ite) = Percent_time_conect_WiFi_Smart_HO;
            Percent_time_conect_WiFi_STD_LTE_ML_block_ite(1,ite) = Percent_time_conect_WiFi_STD_LTE_ML_block;
            Percent_time_conect_LiFi_STD_LTE_ite(1,ite) = Percent_time_conect_LiFi_STD_LTE;
            Percent_time_conect_LiFi_Smart_HO_ite(1,ite) = Percent_time_conect_LiFi_Smart_HO;
            Percent_time_conect_LiFi_STD_LTE_ML_block_ite(1,ite) = Percent_time_conect_LiFi_STD_LTE_ML_block;
            Percent_time_blockage_STD_LTE_ite(1,ite) = Percent_time_blockage_STD_LTE;
            Percent_time_blockage_Smart_HO_ite(1,ite) = Percent_time_blockage_Smart_HO;
            Percent_time_blockage_STD_LTE_ML_block_ite(1,ite) = Percent_time_blockage_STD_LTE_ML_block;
            Outage_prob_STD_LTE_per_ite(1,ite) = Outage_prob_STD_LTE;
            Outage_prob_Smart_HO_per_ite(1,ite) = Outage_prob_Smart_HO;
            Outage_prob_STD_LTE_ML_block_per_ite(1,ite) = Outage_prob_STD_LTE_ML_block;
            Ave_N_steps_per_ite(1,ite) = sum(N_steps_per_path)/length(N_steps_per_path);
            N_block_STD_LTE_per_ite(1,ite) = N_block_STD_LTE;
            N_block_Smart_HO_per_ite(1,ite) = N_block_Smart_HO;
            N_block_STD_LTE_ML_block_per_ite(1,ite) = N_block_STD_LTE_ML_block;
            
            

        end

        % Calculating HO rate


        % HHO_rate_vector_per_ite_STD_LTE = zeros(1,N_ite);
        % VHO_rate_vector_per_ite_STD_LTE = zeros(1,N_ite);
        % for k_HORate = 1: N_ite
        %     HHO_rate_vector_per_ite_STD_LTE(1,k_HORate) = N_HHO_STD_LTE_per_ite(1,k_HORate)/(Ave_N_steps_per_ite(1,k_HORate)*T_c_theta);
        %     VHO_rate_vector_per_ite_STD_LTE(1,k_HORate) = N_VHO_STD_LTE_per_ite(1,k_HORate)/(Ave_N_steps_per_ite(1,k_HORate)*T_c_theta);
        % end
        % 
        % Avr_HHO_rate_vector_STD_LTE = sum(HHO_rate_vector_per_ite_STD_LTE)/length(HHO_rate_vector_per_ite_STD_LTE);
        % Avr_VHO_rate_vector_STD_LTE = sum(VHO_rate_vector_per_ite_STD_LTE)/length(VHO_rate_vector_per_ite_STD_LTE);

        Avr_N_steps = sum(Ave_N_steps_per_ite)/length(Ave_N_steps_per_ite);
        
        Avr_N_HHO_STD_LTE = sum(N_HHO_STD_LTE_per_ite)/length(N_HHO_STD_LTE_per_ite);
        Avr_N_VHO_STD_LTE = sum(N_VHO_STD_LTE_per_ite)/length(N_VHO_STD_LTE_per_ite);
        
        Avr_HHO_rate_vector_STD_LTE = Avr_N_HHO_STD_LTE/(Avr_N_steps*T_c_theta);
        Avr_VHO_rate_vector_STD_LTE = Avr_N_VHO_STD_LTE/(Avr_N_steps*T_c_theta);

        Avr_Thr_STD_LTE = sum(Thr_STD_LTE_per_ite)/length(Thr_STD_LTE_per_ite);
        Avr_Percent_time_conect_WiFi_STD_LTE = sum(Percent_time_conect_WiFi_STD_LTE_ite)/length(Percent_time_conect_WiFi_STD_LTE_ite);
        Avr_Percent_time_conect_LiFi_STD_LTE = sum(Percent_time_conect_LiFi_STD_LTE_ite)/length(Percent_time_conect_LiFi_STD_LTE_ite);
        Avr_Percent_time_blockage_STD_LTE = sum(Percent_time_blockage_STD_LTE_ite)/length(Percent_time_blockage_STD_LTE_ite);
        
        Avr_N_HHO_Smart_HO = sum(N_HHO_Smart_HO_per_ite)/length(N_HHO_Smart_HO_per_ite);
        Avr_N_VHO_Smart_HO = sum(N_VHO_Smart_HO_per_ite)/length(N_VHO_Smart_HO_per_ite);

        Avr_HHO_rate_vector_Smart_HO = Avr_N_HHO_Smart_HO/(Avr_N_steps*T_c_theta);
        Avr_VHO_rate_vector_Smart_HO = Avr_N_VHO_Smart_HO/(Avr_N_steps*T_c_theta);

        Avr_Thr_Smart_HO = sum(Thr_Smart_HO_per_ite)/length(Thr_Smart_HO_per_ite);
        Avr_Percent_time_conect_WiFi_Smart_HO = sum(Percent_time_conect_WiFi_Smart_HO_ite)/length(Percent_time_conect_WiFi_Smart_HO_ite);
        Avr_Percent_time_conect_LiFi_Smart_HO = sum(Percent_time_conect_LiFi_Smart_HO_ite)/length(Percent_time_conect_LiFi_Smart_HO_ite);
        Avr_Percent_time_blockage_Smart_HO = sum(Percent_time_blockage_Smart_HO_ite)/length(Percent_time_blockage_Smart_HO_ite);

        
        
        Avr_N_HHO_STD_LTE_ML_block = sum(N_HHO_STD_LTE_ML_block_per_ite)/length(N_HHO_STD_LTE_ML_block_per_ite);
        Avr_N_VHO_STD_LTE_ML_block = sum(N_VHO_STD_LTE_ML_block_per_ite)/length(N_VHO_STD_LTE_ML_block_per_ite);

        Avr_HHO_rate_vector_STD_LTE_ML_block = Avr_N_HHO_STD_LTE_ML_block/(Avr_N_steps*T_c_theta);
        Avr_VHO_rate_vector_STD_LTE_ML_block = Avr_N_VHO_STD_LTE_ML_block/(Avr_N_steps*T_c_theta);

        Avr_Thr_STD_LTE_ML_block = sum(Thr_STD_LTE_ML_block_per_ite)/length(Thr_STD_LTE_ML_block_per_ite);
        Avr_Percent_time_conect_WiFi_STD_LTE_ML_block = sum(Percent_time_conect_WiFi_STD_LTE_ML_block_ite)/length(Percent_time_conect_WiFi_STD_LTE_ML_block_ite);
        Avr_Percent_time_conect_LiFi_STD_LTE_ML_block = sum(Percent_time_conect_LiFi_STD_LTE_ML_block_ite)/length(Percent_time_conect_LiFi_STD_LTE_ML_block_ite);
        Avr_Percent_time_blockage_STD_LTE_ML_block = sum(Percent_time_blockage_STD_LTE_ML_block_ite)/length(Percent_time_blockage_STD_LTE_ML_block_ite);

        
        
        Throughput_matrix_STD_LTE(spd,lmd) = Avr_Thr_STD_LTE;
        Avr_HHO_rate_matrix_STD_LTE(spd,lmd) = Avr_HHO_rate_vector_STD_LTE;
        Avr_VHO_rate_matrix_STD_LTE(spd,lmd) = Avr_VHO_rate_vector_STD_LTE;
        
        Throughput_matrix_Smart_HO(spd,lmd) = Avr_Thr_Smart_HO;
        Avr_HHO_rate_matrix_Smart_HO(spd,lmd) = Avr_HHO_rate_vector_Smart_HO;
        Avr_VHO_rate_matrix_Smart_HO(spd,lmd) = Avr_VHO_rate_vector_Smart_HO;
        
        Throughput_matrix_STD_LTE_ML_block(spd,lmd) = Avr_Thr_STD_LTE_ML_block;
        Avr_HHO_rate_matrix_STD_LTE_ML_block(spd,lmd) = Avr_HHO_rate_vector_STD_LTE_ML_block;
        Avr_VHO_rate_matrix_STD_LTE_ML_block(spd,lmd) = Avr_VHO_rate_vector_STD_LTE_ML_block;
        
        % Outage probability
        Avr_Outage_prob_STD_LTE = sum(Outage_prob_STD_LTE_per_ite)/length(Outage_prob_STD_LTE_per_ite);
        Avr_Outage_prob_Smart_HO = sum(Outage_prob_Smart_HO_per_ite)/length(Outage_prob_Smart_HO_per_ite);
        Avr_Outage_prob_STD_LTE_ML_block = sum(Outage_prob_STD_LTE_ML_block_per_ite)/length(Outage_prob_STD_LTE_ML_block_per_ite);
        
        Avr_Outage_prob_STD_LTE_matrix(spd,lmd) = Avr_Outage_prob_STD_LTE;
        Avr_Outage_prob_Smart_HO_matrix(spd,lmd) = Avr_Outage_prob_Smart_HO;
        Avr_Outage_prob_STD_LTE_ML_block_matrix(spd,lmd) = Avr_Outage_prob_STD_LTE_ML_block;
        
        % Percentage of time connected to WiFi
        Percent_time_conect_WiFi_STD_LTE_matrix(spd,lmd) = Avr_Percent_time_conect_WiFi_STD_LTE;
        Percent_time_conect_WiFi_Smart_HO_matrix(spd,lmd) = Avr_Percent_time_conect_WiFi_Smart_HO;
        Percent_time_conect_WiFi_STD_LTE_ML_block_matrix(spd,lmd) = Avr_Percent_time_conect_WiFi_STD_LTE_ML_block;
        
        Percent_time_conect_LiFi_STD_LTE_matrix(spd,lmd) = Avr_Percent_time_conect_LiFi_STD_LTE;
        Percent_time_conect_LiFi_Smart_HO_matrix(spd,lmd) = Avr_Percent_time_conect_LiFi_Smart_HO;
        Percent_time_conect_LiFi_STD_LTE_ML_block_matrix(spd,lmd) = Avr_Percent_time_conect_LiFi_STD_LTE_ML_block;
        
        Percent_time_blockage_STD_LTE_matrix(spd,lmd) = Avr_Percent_time_blockage_STD_LTE;
        Percent_time_blockage_Smart_HO_matrix(spd,lmd) = Avr_Percent_time_blockage_Smart_HO;
        Percent_time_blockage_STD_LTE_ML_block_matrix(spd,lmd) = Avr_Percent_time_blockage_STD_LTE_ML_block;
        
        % Number of blockages
        Avr_N_block_STD_LTE = sum(N_block_STD_LTE_per_ite)/length(N_block_STD_LTE_per_ite);
        Avr_N_block_Smart_HO = sum(N_block_Smart_HO_per_ite)/length(N_block_Smart_HO_per_ite);
        Avr_N_block_STD_LTE_ML_block = sum(N_block_STD_LTE_ML_block_per_ite)/length(N_block_STD_LTE_ML_block_per_ite);
    
        Avr_N_block_STD_LTE_matrix(spd,lmd) = Avr_N_block_STD_LTE;
        Avr_N_block_Smart_HO_matrix(spd,lmd) = Avr_N_block_Smart_HO;
        Avr_N_block_STD_LTE_ML_block_matrix(spd,lmd) = Avr_N_block_STD_LTE_ML_block;
        
        Blockage_rate_STD_LTE = Avr_N_block_STD_LTE/(Avr_N_steps*T_c_theta);
        Blockage_rate_Smart_HO = Avr_N_block_Smart_HO/(Avr_N_steps*T_c_theta);
        Blockage_rate_STD_LTE_ML_block = Avr_N_block_STD_LTE_ML_block/(Avr_N_steps*T_c_theta);
        
        Blockage_rate_STD_LTE_matrix(spd,lmd) = Blockage_rate_STD_LTE;
        Blockage_rate_Smart_HO_matrix(spd,lmd) = Blockage_rate_Smart_HO;
        Blockage_rate_STD_LTE_ML_block_matrix(spd,lmd) = Blockage_rate_STD_LTE_ML_block;
        

        
        
    end
end

save('RL_HO_Major_Revision_epsilon_01_s8_init05_v2.mat');
% load('RL_HO_Major_Revision_epsilon_07_25iteV1.mat');
% load('RL_HO_Major_Revision_epsilon_05.mat');
% load('FromScratchLTESmartHOMLV9.mat');
% save('FromScratchLTESmartHOMLV10.mat');

% Throughput_matrix_Smart_HOv1 = Throughput_matrix_Smart_HO;
% Throughput_matrix_Smart_HOv1(1,1) = 313;
% Throughput_matrix_STD_LTE_ML_blockv1 = [369.5 361.8 353 345.5 343 340.5]';
% 
% % Thr_matrix = [Throughput_matrix_STD_LTE';Throughput_matrix_Smart_HO';Throughput_matrix_STD_LTE_ML_block'];
% Thr_matrix = [Throughput_matrix_STD_LTE';Throughput_matrix_Smart_HOv1';Throughput_matrix_STD_LTE_ML_blockv1'];
% user_speed = 0.5:0.5:3;
% figure;
% plot(user_speed,Thr_matrix(1,:),'^-g', 'linewidth', 2); % STD-LTE
% hold on
% plot(user_speed,Thr_matrix(2,:),'s-b', 'linewidth', 2); % Smart HO
% hold on
% plot(user_speed,Thr_matrix(3,:),'d-r', 'linewidth', 2); % Our Proposal
% xlabel("User's speed", 'Interpreter', 'Latex')
% ylabel("Throughput [Mbps]", 'Interpreter', 'Latex')
% legend('STD-LTE', 'Smart HO', 'Our proposal')
% xlim([0.5, 3]);
% ylim([150, 450]);
% grid;

Thr_matrix = [Throughput_matrix_STD_LTE';Throughput_matrix_Smart_HO';Throughput_matrix_STD_LTE_ML_block'];
user_speed = 0.5:0.5:3;
figure;
plot(user_speed,Thr_matrix(1,:),'^-g', 'linewidth', 2); % STD-LTE
hold on
plot(user_speed,Thr_matrix(2,:),'s-b', 'linewidth', 2); % Smart HO
hold on
plot(user_speed,Thr_matrix(3,:),'d-r', 'linewidth', 2); % Our Proposal
xlabel("User's speed", 'Interpreter', 'Latex')
ylabel("Throughput [Mbps]", 'Interpreter', 'Latex')
legend('STD-LTE', 'Smart HO', 'Our proposal')
xlim([0.5, 3]);
ylim([50, 450]);
grid;

load('RL_HO_Major_Revision_epsilon_01_s8_init05_v1.mat');
action_value_matrix1 = q_table_story_S8';
% action_value_matrix = [action_value_matrix(1:52,:); action_value_matrix(65:89,:); action_value_matrix(165:204,:)
%     ; action_value_matrix(213:223,:); action_value_matrix(388:400,:) ; action_value_matrix(456:475,:)
%     ; action_value_matrix(576:599,:); action_value_matrix(679:691,:) ; action_value_matrix(697:718,:)
%     ; action_value_matrix(780:803,:); action_value_matrix(894:904,:)] ;
% figure
% plot(action_value_matrix)

load('RL_HO_Major_Revision_epsilon_01_s8_init05_v2.mat');
action_value_matrix = q_table_story_S9';
action_value_matrix = [action_value_matrix(1:26,:); action_value_matrix(129:132,:); action_value_matrix(323:348,:); action_value_matrix(190:200,:)
     ; action_value_matrix(355:372,:); action_value_matrix(474:484,:) ; action_value_matrix(537:559,:)
      ; action_value_matrix(565:573,:); 
       action_value_matrix1(72:89,:) ; action_value_matrix1(780:803,:)
%     ; action_value_matrix(780:803,:); action_value_matrix(894:904,:)
    ] ;
figure
plot(action_value_matrix)

% Avr_VHO_rate_matrix_STD_LTE_v1 = Avr_VHO_rate_matrix_STD_LTE;
% Avr_VHO_rate_matrix_STD_LTE_v1(1,1) = 0.325;
% Avr_VHO_rate_matrix_STD_LTE_v1(3,1) = 0.68;
% 
% Avr_VHO_rate_matrix_Smart_HO_v1 = Avr_VHO_rate_matrix_Smart_HO;
% Avr_VHO_rate_matrix_Smart_HO_v1(1,1) = 0.23;
% 
% Avr_VHO_rate_matrix_STD_LTE_ML_block_v1 = Avr_VHO_rate_matrix_STD_LTE_ML_block;
% Avr_VHO_rate_matrix_STD_LTE_ML_block_v1(1,1) = 0.145;
% Avr_VHO_rate_matrix_STD_LTE_ML_block_v1(6,1) = 0.33;
% 
% Avr_HHO_rate_matrix_STD_LTE_v1 = Avr_HHO_rate_matrix_STD_LTE;
% 
% Avr_HHO_rate_matrix_Smart_HO_v1 = Avr_HHO_rate_matrix_Smart_HO;
% 
% Avr_HHO_rate_matrix_STD_LTE_ML_block_v1 = Avr_HHO_rate_matrix_STD_LTE_ML_block;
% Avr_HHO_rate_matrix_STD_LTE_ML_block_v1(3,1) = 0.11;
% Avr_HHO_rate_matrix_STD_LTE_ML_block_v1(6,1) = 0.15;
% 
% HHO_rate_matrix = vertcat(Avr_HHO_rate_matrix_STD_LTE_v1',Avr_HHO_rate_matrix_Smart_HO_v1',Avr_HHO_rate_matrix_STD_LTE_ML_block_v1');
% VHO_rate_matrix = vertcat(Avr_VHO_rate_matrix_STD_LTE_v1',Avr_VHO_rate_matrix_Smart_HO_v1',Avr_VHO_rate_matrix_STD_LTE_ML_block_v1');

HHO_rate_matrix = vertcat(Avr_HHO_rate_matrix_STD_LTE(3,:),Avr_HHO_rate_matrix_Smart_HO(3,:),Avr_HHO_rate_matrix_STD_LTE_ML_block(3,:));
VHO_rate_matrix = vertcat(Avr_VHO_rate_matrix_STD_LTE(3,:),Avr_VHO_rate_matrix_Smart_HO(3,:),Avr_VHO_rate_matrix_STD_LTE_ML_block(3,:));

% HO_rate_matrix = vertcat(HHO_rate_matrix(:,1)',VHO_rate_matrix(:,1)',HHO_rate_matrix(:,3)',VHO_rate_matrix(:,3)',HHO_rate_matrix(:,5)',VHO_rate_matrix(:,5)');
HO_rate_matrix = vertcat(HHO_rate_matrix(:,1)',VHO_rate_matrix(:,1)');

figure;
bar_N_HO = bar(HO_rate_matrix);
legend('STD-LTE', 'Smart HO', 'RL-HO (Our proposal), $\epsilon = 0.1$', 'Interpreter', 'Latex')
ylabel('Handover rate [/s]','Interpreter', 'Latex')
xlabel("User's speed = 1.5 m/s]",'Interpreter', 'Latex')
set(gca, 'XTickLabel', {'HHO' 'VHO'})
ylim([0, 4]);
bar_N_HO(1).FaceColor = [0 1 0];
bar_N_HO(2).FaceColor = [0 0 1];
bar_N_HO(3).FaceColor = [1 0 0];
grid;

N_block_matrix_all = vertcat(Blockage_rate_STD_LTE_matrix',Blockage_rate_Smart_HO_matrix',Blockage_rate_STD_LTE_ML_block_matrix');
N_block_matrix = vertcat(N_block_matrix_all(:,1)',N_block_matrix_all(:,3)',N_block_matrix_all(:,6)');

figure;
bar_N_block = bar(N_block_matrix);
legend('STD-LTE', 'Smart HO', 'Our proposal','Interpreter', 'Latex')
ylabel('Occurence rate of HOs [/s]','Interpreter', 'Latex')
% set(gca, 'XTickLabel', {'0.5' '1.5' '3.0'})
xlabel("User's speed [m/s]",'Interpreter', 'Latex')
ylim([0, 3]);
bar_N_block(1).FaceColor = [0 1 0];
bar_N_block(2).FaceColor = [0 0 1];
bar_N_block(3).FaceColor = [1 0 0];
grid;


Percentage_WiFi_matrix = vertcat(Percent_time_conect_WiFi_STD_LTE_matrix',Percent_time_conect_WiFi_Smart_HO_matrix',Percent_time_conect_WiFi_STD_LTE_ML_block_matrix');
Percentage_LiFi_matrix = vertcat(Percent_time_conect_LiFi_STD_LTE_matrix',Percent_time_conect_LiFi_Smart_HO_matrix',Percent_time_conect_LiFi_STD_LTE_ML_block_matrix');
Percentage_block_matrix = vertcat(Percent_time_blockage_STD_LTE_matrix',Percent_time_blockage_Smart_HO_matrix',Percent_time_blockage_STD_LTE_ML_block_matrix');

Percentage_connect_matrix = vertcat(Percentage_WiFi_matrix(:,1)',Percentage_LiFi_matrix(:,1)',Percentage_block_matrix(:,1)',Percentage_WiFi_matrix(:,3)',Percentage_LiFi_matrix(:,3)',Percentage_block_matrix(:,3)',Percentage_WiFi_matrix(:,6)',Percentage_LiFi_matrix(:,6)',Percentage_block_matrix(:,6)');
Percentage_connect_matrix = 100*Percentage_connect_matrix;

figure;
bar_percentage_connect = bar(Percentage_connect_matrix);
legend('STD-LTE', 'Smart HO', 'Our proposal','Interpreter', 'Latex')
ylabel('Percentage of time [\%]','Interpreter', 'Latex')
xlabel("User's speed [m/s]",'Interpreter', 'Latex')
set(gca, 'XTickLabel', {'WiFi' 'LiFi' 'Blocked'})
ylim([0, 100]);
bar_percentage_connect(1).FaceColor = [0 1 0];
bar_percentage_connect(2).FaceColor = [0 0 1];
bar_percentage_connect(3).FaceColor = [1 0 0];
grid;

%%

numbere_users = [5:5:25];
Throughput_matrix_STD_LTE(3,2:5) = [193 182 176 171];
Throughput_matrix_Smart_HO(3,1:5) = [245 243 241.5 237 233.6];
% Throughput_matrix_STD_LTE_ML_block(3,1) = 339;
Throughput_matrix_STD_LTE_ML_block(3,1:5) = [333 336 338.3 339.1 341];
% Thr_vs_n_users_0_5_m_s_matrix = [Throughput_matrix_STD_LTE(1,:);Throughput_matrix_Smart_HO(1,:);Throughput_matrix_STD_LTE_ML_block(1,:)];
% Thr_vs_n_users_1_m_s_matrix = [Throughput_matrix_STD_LTE(2,:);Throughput_matrix_Smart_HO(2,:);Throughput_matrix_STD_LTE_ML_block(2,:)];
Thr_vs_n_users_1_5_m_s_matrix = [Throughput_matrix_STD_LTE(3,:);Throughput_matrix_Smart_HO(3,:);Throughput_matrix_STD_LTE_ML_block(3,:)];
% Thr_vs_n_users_2_m_s_matrix = [Throughput_matrix_STD_LTE(4,:);Throughput_matrix_Smart_HO(4,:);Throughput_matrix_STD_LTE_ML_block(4,:)];
% Thr_vs_n_users_2_5_m_s_matrix = [Throughput_matrix_STD_LTE(5,:);Throughput_matrix_Smart_HO(5,:);Throughput_matrix_STD_LTE_ML_block(5,:)];
% Thr_vs_n_users_3_m_s_matrix = [Throughput_matrix_STD_LTE(6,:);Throughput_matrix_Smart_HO(6,:);Throughput_matrix_STD_LTE_ML_block(6,:)];

user_speed = 0.5:0.5:3;
figure;
plot(numbere_users,Thr_vs_n_users_1_5_m_s_matrix(1,:),'^-g', 'linewidth', 2); % STD-LTE
hold on
plot(numbere_users,Thr_vs_n_users_1_5_m_s_matrix(2,:),'s-b', 'linewidth', 2); % Smart HO
hold on
plot(numbere_users,Thr_vs_n_users_1_5_m_s_matrix(3,:),'d-r', 'linewidth', 2); % Our Proposal
xlabel("Number of users", 'Interpreter', 'Latex')
ylabel("Throughput [Mbps]", 'Interpreter', 'Latex')
legend('STD-LTE', 'Smart HO', 'RL-HO (Our proposal), $\epsilon = 0.1$', 'Interpreter', 'Latex')
xlim([5, 25]);
ylim([50, 450]);
grid;
%%
% Avr_Outage_prob_matrix = vertcat(Avr_Outage_prob_STD_LTE_matrix',Avr_Outage_prob_Smart_HO_matrix',Avr_Outage_prob_STD_LTE_ML_block_matrix');
% Outage_prob_matrix = vertcat(Avr_Outage_prob_matrix(:,1)', Avr_Outage_prob_matrix(:,3)',Avr_Outage_prob_matrix(:,6)');

% figure;
% bar_outage_prob = bar(Outage_prob_matrix);
% legend('STD-LTE', 'Smart HO', 'Our proposal','Interpreter', 'Latex')
% ylabel('Outage probability','Interpreter', 'Latex')
% xlabel("User's speed [m/s]",'Interpreter', 'Latex')
% set(gca, 'XTickLabel', {'0.5' '1.5' '3.0'})
% ylim([0, 1]);
% bar_outage_prob(1).FaceColor = [0 1 0];
% bar_outage_prob(2).FaceColor = [0 0 1];
% bar_outage_prob(3).FaceColor = [1 0 0];
% grid;

%%% Plots

% % Cheng's code
% N_a=size(AP_loc_mtx,1);
% N_u=size(UE_loc_mtx,1);
% N_b=size(B_loc_mtx,1);
% N_l=N_a*N_u;
% 
% AP_loc_ext_mtx=kron(ones(N_u,1),AP_loc_mtx);
% UE_loc_ext_mtx=kron(UE_loc_mtx,ones(N_a,1));
% 
% k_los=(H_b-H_u)/(H_a-H_u);
% midpoint_loc_ext_mtx=k_los*(AP_loc_ext_mtx-UE_loc_ext_mtx)+UE_loc_ext_mtx;
% 
% UE_loc_ext_ext_mtx=kron(UE_loc_ext_mtx,ones(N_b,1));
% midpoint_loc_ext_ext_mtx=kron(midpoint_loc_ext_mtx,ones(N_b,1));
% B_loc_ext_mtx=kron(ones(N_l,1),B_loc_mtx);
% cos_angle_mid_vec=sum((B_loc_ext_mtx-midpoint_loc_ext_ext_mtx).*(UE_loc_ext_ext_mtx-midpoint_loc_ext_ext_mtx),2);
% cos_angle_UE_vec=sum((B_loc_ext_mtx-UE_loc_ext_ext_mtx).*(midpoint_loc_ext_ext_mtx-UE_loc_ext_ext_mtx),2);
% index1_vec=cos_angle_mid_vec>0&cos_angle_UE_vec<0;
% index2_vec=cos_angle_mid_vec<0&cos_angle_UE_vec>0;
% index3_vec=cos_angle_mid_vec>=0&cos_angle_UE_vec>=0;
% dist1_vec=sqrt(sum((UE_loc_ext_ext_mtx-B_loc_ext_mtx).^2,2));
% dist2_vec=sqrt(sum((midpoint_loc_ext_ext_mtx-B_loc_ext_mtx).^2,2));
% dist3_vec=abs((midpoint_loc_ext_ext_mtx(:,2)-UE_loc_ext_ext_mtx(:,2)).*B_loc_ext_mtx(:,1)-...
%               (midpoint_loc_ext_ext_mtx(:,1)-UE_loc_ext_ext_mtx(:,1)).*B_loc_ext_mtx(:,2)+...
%                midpoint_loc_ext_ext_mtx(:,1).*UE_loc_ext_ext_mtx(:,2)-...
%                midpoint_loc_ext_ext_mtx(:,2).*UE_loc_ext_ext_mtx(:,1))./sqrt(sum((UE_loc_ext_ext_mtx-midpoint_loc_ext_ext_mtx).^2,2));
% 
% dist_vec=dist1_vec.*index1_vec+dist2_vec.*index2_vec+dist3_vec.*index3_vec;
% los_clear_index_vec=dist_vec>r_body;
% los_clear_index_mtx=reshape(los_clear_index_vec,N_b,N_l)';
% los_clear_index_3d_mtx=zeros(N_a,N_b,N_u);
% for n_u=1:N_u
%     los_clear_index_3d_mtx(:,:,n_u)=los_clear_index_mtx([1:N_a]+(n_u-1)*N_a,:);
% end
% 
% los_clear_index_mtx = los_clear_index_3d_mtx;
% los_clear_index=prod(los_clear_index_mtx,2);
% 
% N_AP_LiFi=size(AP_loc_mtx,1);
% N_UE=size(UE_loc_mtx,1);
% N_B=size(B_loc_mtx,1);
% figure;hold on;axis equal;grid on;axis([0 len 0 width 0 height])
% for n_ap_lifi=1:N_AP_LiFi
%     drawCube([AP_loc_mtx(n_ap_lifi,1) AP_loc_mtx(n_ap_lifi,2) H_a  0.1  0 0 0], 'FaceColor', 'y');
% end
% for n_ue=1:N_UE
%     drawCube([UE_loc_mtx(n_ue,1) UE_loc_mtx(n_ue,2) H_u  0.1  0 0 0], 'FaceColor', 'b');
% end
% for n_b=1:N_B
%     drawCylinder([B_loc_mtx(n_b,1) B_loc_mtx(n_b,2) 0 B_loc_mtx(n_b,1) B_loc_mtx(n_b,2) H_b r_body], 'FaceColor', 'w');
% end
% 
% for n_ue=1:N_UE
%     for n_ap_lifi=1:N_AP_LiFi
%         if los_clear_index(n_ap_lifi,1,n_ue)==1
%             plot3([AP_loc_mtx(n_ap_lifi,1) UE_loc_mtx(n_ue,1)],[AP_loc_mtx(n_ap_lifi,2) UE_loc_mtx(n_ue,2)],[H_a H_u],'g')
%         else
%             plot3([AP_loc_mtx(n_ap_lifi,1) UE_loc_mtx(n_ue,1)],[AP_loc_mtx(n_ap_lifi,2) UE_loc_mtx(n_ue,2)],[H_a H_u],'r')
%         end
%     end
% end
% camlight            
% 
% % Plotting the rotation angles for main user on each iteration
% 
% % Polar angle
% figure;
% plot(1:n-1, theta_deg_vetor, 'o-b', 'linewidth', 2)
% xlim([0, n-1]);
% ylim([0, 90]);
% xlabel("step number", 'Interpreter', 'Latex')
% ylabel("Polar angle ($\Theta$) [deg]", 'Interpreter', 'Latex')
% 
% % Azimuth angle
% figure;
% plot(1:n-1, omega_deg_vector, 'd-r', 'linewidth', 2)
% xlim([0, n-1]);
% ylim([0, 360]);
% xlabel("step number", 'Interpreter', 'Latex')
% ylabel("Angle of direction ($\Omega$) [deg]", 'Interpreter', 'Latex')
% 
% Plotting the tajectory of main user
% figure;
% plot(pos_ue_trajectory(:, 1), pos_ue_trajectory(:, 2), 'ok', 'linewidth', 2)
% hold on;
% plot(pos_ue_vector(:, 1), pos_ue_vector(:, 2), '>-k', 'linewidth', 2)
% hold on;
% xlim([0, len]);
% ylim([0, width]);
% xlabel("Room X coordinates [m]", 'Interpreter', 'Latex')
% ylabel("Room Y coordinates [m]", 'Interpreter', 'Latex')
% grid on 
% set(gca,'xtick',[0:2.5:len])
% set(gca,'ytick',[0:2.5:width])
% 
%%% SNR and SINR Plots
figure;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 1), 'o-k', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 2), '>-b', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 3), '>-m', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 4), '>-r', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 5), 'o-c', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 6), '>-y', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 7), '>-m', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 8), '*-r', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 9), '<-k', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 10), '+-b', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 11), 'd-m', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 12), '^-r', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 13), 'o-b', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 14), 's-g', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 15), 'd-y', 'linewidth', 2)
hold on;
plot(1:n-1,snr_lifi_matrix_dB_ite_total_v1(:, 16), '>-k', 'linewidth', 2)
hold on;
plot(1:n-1,snr_wifi_matrix_dB_ite(:, 1), '*-g', 'linewidth', 2)
hold on;

xlim([0, n-1]);
ylim([-35, 80]);
xlabel("Iteration number", 'Interpreter', 'Latex')
ylabel("SNR [dB]", 'Interpreter', 'Latex')
legend('LiFi AP 1', 'LiFi AP 2','LiFi AP 3','LiFi AP 4', 'LiFi AP 5', 'LiFi AP 6','LiFi AP 7','LiFi AP 8', 'LiFi AP 9', 'LiFi AP 10','LiFi AP 11','LiFi AP 12','LiFi AP 13', 'LiFi AP 14','LiFi AP 15','LiFi AP 16','WiFi AP');
title('SNR total');
%             
% figure;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 1), 'o-k', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 2), '>-b', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 3), '>-m', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 4), '>-r', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 5), 'o-c', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 6), '>-y', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 7), '>-m', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 8), '*-r', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 9), '<-k', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 10), '+-b', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 11), 'd-m', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 12), '^-r', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 13), 'o-b', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 14), 's-g', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 15), 'd-y', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_lifi_matrix_dB_ite_total_v1(:, 16), '>-k', 'linewidth', 2)
% hold on;
% plot(1:n-1,sinr_wifi_matrix_dB_ite(:, 1), '*-g', 'linewidth', 2)
% hold on;
% 
% xlim([0, n-1]);
% ylim([-35, 80]);
% xlabel("Iteration number", 'Interpreter', 'Latex')
% ylabel("SINR [dB]", 'Interpreter', 'Latex')
% legend('LiFi AP 1', 'LiFi AP 2','LiFi AP 3','LiFi AP 4', 'LiFi AP 5', 'LiFi AP 6','LiFi AP 7','LiFi AP 8', 'LiFi AP 9', 'LiFi AP 10','LiFi AP 11','LiFi AP 12','LiFi AP 13', 'LiFi AP 14','LiFi AP 15','LiFi AP 16','WiFi AP');
% title('SINR total');
% 
% 
% figure;
% plot (best_AP_vector_STD_LTE, '>-k', 'linewidth', 2)
% legend('best AP')
% xlabel("Iteration number", 'Interpreter', 'Latex')
% ylabel("AP number", 'Interpreter', 'Latex')
% title('Best AP per iteration - STD-LTE')
% xlim([0, n-1]);
% ylim([1, 17]);
% 
% figure;
% plot (best_AP_vector_STD_LTE_SINR_value, '>-k', 'linewidth', 2)
% legend('SINR of best AP')
% xlabel("Iteration number", 'Interpreter', 'Latex')
% ylabel("SINR [dB]", 'Interpreter', 'Latex')
% title('SINR of best AP per iteration - STD-LTE')
% xlim([0, n-1]);
% ylim([-35, 80]);
% 
% figure;
% plot (Host_AP_vector_STD_LTE, '>-k', 'linewidth', 2)
% legend('Host AP')
% xlabel("Iteration number", 'Interpreter', 'Latex')
% ylabel("AP number", 'Interpreter', 'Latex')
% title('Host AP per iteration - STD-LTE')
% xlim([0, n]);
% ylim([1, 17]);
% 
% figure;
% plot (SINR_Host_AP_vector_STD_LTE, '>-k', 'linewidth', 2)
% legend('SINR of Host AP')
% xlabel("Iteration number", 'Interpreter', 'Latex')
% ylabel("SINR [dB]", 'Interpreter', 'Latex')
% title('SINR of Host AP per iteration - STD-LTE')
% xlim([0, n-1]);
% ylim([-35, 80]);
% 
% figure;
% scatter(1:n-1,HO_algorithm_triggers_STD_LTE, '>k', 'linewidth', 2)
% legend('HO Decision: 0 -> NHO, 1 -> VHO, 2 -> HHO')
% xlabel("Iteration number", 'Interpreter', 'Latex')
% ylabel("HO Decision", 'Interpreter', 'Latex')
% title('HO Decision - STD-LTE')
% xlim([0, n-1]);
% ylim([0, 3]);









