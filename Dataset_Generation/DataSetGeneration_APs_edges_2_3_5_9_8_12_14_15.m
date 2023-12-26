%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%   IMPORTANT: All the parameters here, must be the same as the ones in
%%%%   the main script for simulations RL-HO.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Init

clear all;
close all;
rng(2,'twister'); % Fix the random number generator

%%% Scenario parameters

% Room
len = 10; % x
width = 10; % y
height = 3;

%%% Scenario parameters

% Room
len = 10; % x
width = 10; % y
height = 3;
inter_lifi_ap_dist = 2.5;

% LiFi APs

pos_lifi_ap_aux = combvec(1.25:inter_lifi_ap_dist:len, 1.25:inter_lifi_ap_dist:width); % Create all combinations of vectors. [x,y] of APs
pos_lifi_ap = [pos_lifi_ap_aux(2,:).' pos_lifi_ap_aux(1,:).' ones(size(pos_lifi_ap_aux,2),1)*height]; % [x,y,z] of APs.

% WiFi APs

pos_wifi_ap = [width len height]; % AP in a corner
% pos_wifi_ap = [width/2 len/2 height]; % AP in the ceiling 
% pos_wifi_ap = [width/2 len/2 3]; % AP in the ground (center of the room)
% pos_wifi_ap = [20 20 height];

% Channel parameters
t_trigger = 0.260; % Time to trigger HO algorithm 
T_c_theta = 130e-3; % Coherence time of the polar angle. 134 ms -> walking. 373 ms -> sitting

% Trajectories

distances = [0.5 1 1.5]; % distance from start point to end point
speeds = [0.5 1 1.5 2 2.5 3];
trajectory_types = [1 2 3]; % 1 -> to wall, 2 -> to center of cell, 3 -> to cross the edge of cell
AP_numbers = [2 3 5 9 8 12 14 15]; 

for k = 1: length(speeds)
    
    for kk = 1: length(distances) 
        
        %%% AP2

        % Points
        wall_end_points_ap2 = [zeros(1,3); [(inter_lifi_ap_dist/4)*[4:2:8]]]';
        edge_points_ap2 = [(inter_lifi_ap_dist/4)*[2 4 4 4 2];[(inter_lifi_ap_dist/4)*[4 4 6 8 8]]]';
        adjacent_aps_ap2 = [pos_lifi_ap(1,:); pos_lifi_ap(5,:); pos_lifi_ap(6,:); pos_lifi_ap(7,:); pos_lifi_ap(3,:)]; % same order as edge points
        cross_points_ap2 =  [(inter_lifi_ap_dist/4)*[3 4 4 3];[(inter_lifi_ap_dist/4)*[4 5 7 8]]]';
        adjacent_cross_points_ap2 = [(inter_lifi_ap_dist/4)*[4 5 5 4];[(inter_lifi_ap_dist/4)*[3 4 8 9]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap2 = [45 0 315]'; % angles for trajectory
        angles_edge_points_ap2 = [90 135 180 225 270]';
        angles_cross_points_ap2 = [135 135 225 225]';

        % Trajectories
        startPoint_to_wall_ap2 = [wall_end_points_ap2(:,1) + cosd(angles_start_points_to_wall_ap2)*distances(1,kk) wall_end_points_ap2(:,2) + sind(angles_start_points_to_wall_ap2)*distances(1,kk)];
        startPoint_to_cell_center_ap2 = [edge_points_ap2(:,1) + cosd(angles_edge_points_ap2)*distances(1,kk) edge_points_ap2(:,2) + sind(angles_edge_points_ap2)*distances(1,kk)];
        startPoint_to_cross_edge_ap2 = [cross_points_ap2(:,1) + cosd(angles_cross_points_ap2)*distances(1,kk) cross_points_ap2(:,2) + sind(angles_cross_points_ap2)*distances(1,kk)];

        diff_wall_ap2 = wall_end_points_ap2 - startPoint_to_wall_ap2;
        diff_cell_center_ap2 = edge_points_ap2 - startPoint_to_cell_center_ap2;
        diff_cell_edge_ap2 = cross_points_ap2 - startPoint_to_cross_edge_ap2;
        
        if k == 1
            figure;
            quiver(startPoint_to_wall_ap2(:,1),startPoint_to_wall_ap2(:,2),diff_wall_ap2(:,1),diff_wall_ap2(:,2),0,'b', 'linewidth', 2);
            hold on;
            quiver(startPoint_to_cell_center_ap2(:,1),startPoint_to_cell_center_ap2(:,2),diff_cell_center_ap2(:,1),diff_cell_center_ap2(:,2),0,'r', 'linewidth', 2);
            hold on;
            quiver(startPoint_to_cross_edge_ap2(:,1),startPoint_to_cross_edge_ap2(:,2),diff_cell_edge_ap2(:,1),diff_cell_edge_ap2(:,2),0,'m', 'linewidth', 2);
            xlim([0, 10]);
            ylim([0,10]);
            grid;
            title('Trajectories', 'Interpreter', 'Latex')
            legend('to the wall', 'to the center of a cell', 'to the edge of a cell', 'Interpreter', 'Latex')
        end
        
        %%% AP3

        % Points
        wall_end_points_ap3 = [zeros(1,3); [(inter_lifi_ap_dist/4)*[8:2:12]]]';
        edge_points_ap3 = [(inter_lifi_ap_dist/4)*[2 4 4 4 2];[(inter_lifi_ap_dist/4)*[8 8 10 12 12]]]';
        adjacent_aps_ap3 = [pos_lifi_ap(2,:); pos_lifi_ap(6,:); pos_lifi_ap(7,:); pos_lifi_ap(8,:); pos_lifi_ap(4,:)]; % same order as edge points
        cross_points_ap3 =  [(inter_lifi_ap_dist/4)*[3 4 4 3];[(inter_lifi_ap_dist/4)*[8 9 11 12]]]';
        adjacent_cross_points_ap3 = [(inter_lifi_ap_dist/4)*[4 5 5 4];[(inter_lifi_ap_dist/4)*[7 8 12 13]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap3 = [45 0 315]'; % angles for trajectory
        angles_edge_points_ap3 = [90 135 180 225 270]';
        angles_cross_points_ap3 = [135 135 225 225]';

        % Trajectories
        startPoint_to_wall_ap3 = [wall_end_points_ap3(:,1) + cosd(angles_start_points_to_wall_ap3)*distances(1,kk) wall_end_points_ap3(:,2) + sind(angles_start_points_to_wall_ap3)*distances(1,kk)];
        startPoint_to_cell_center_ap3 = [edge_points_ap3(:,1) + cosd(angles_edge_points_ap3)*distances(1,kk) edge_points_ap3(:,2) + sind(angles_edge_points_ap3)*distances(1,kk)];
        startPoint_to_cross_edge_ap3 = [cross_points_ap3(:,1) + cosd(angles_cross_points_ap3)*distances(1,kk) cross_points_ap3(:,2) + sind(angles_cross_points_ap3)*distances(1,kk)];

        %%%% AP5

        % Points
        wall_end_points_ap5 = [(inter_lifi_ap_dist/4)*[4:2:8]; [0*(inter_lifi_ap_dist/4)*ones(1,3)]]';
        edge_points_ap5 = [(inter_lifi_ap_dist/4)*[4 4 6 8 8];[(inter_lifi_ap_dist/4)*[2 4 4 4 2]]]';
        adjacent_aps_ap5 = [pos_lifi_ap(1,:); pos_lifi_ap(2,:); pos_lifi_ap(6,:); pos_lifi_ap(10,:); pos_lifi_ap(9,:)]; % same order as edge points
        cross_points_ap5 =  [(inter_lifi_ap_dist/4)*[4 5 7 8 ];[(inter_lifi_ap_dist/4)*[3 4 4 3]]]';
        adjacent_cross_points_ap5 = [(inter_lifi_ap_dist/4)*[3 4 8 9];[(inter_lifi_ap_dist/4)*[4 5 5 4]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap5 = [45 90 135]'; % angles for trajectory
        angles_edge_points_ap5 = [0 315 270 225 180]';
        angles_cross_points_ap5 = [315 315 225 225]';

        % Trajectories
        startPoint_to_wall_ap5 = [wall_end_points_ap5(:,1) + cosd(angles_start_points_to_wall_ap5)*distances(1,kk) wall_end_points_ap5(:,2) + sind(angles_start_points_to_wall_ap5)*distances(1,kk)];
        startPoint_to_cell_center_ap5 = [edge_points_ap5(:,1) + cosd(angles_edge_points_ap5)*distances(1,kk) edge_points_ap5(:,2) + sind(angles_edge_points_ap5)*distances(1,kk)];
        startPoint_to_cross_edge_ap5 = [cross_points_ap5(:,1) + cosd(angles_cross_points_ap5)*distances(1,kk) cross_points_ap5(:,2) + sind(angles_cross_points_ap5)*distances(1,kk)];
        
        %%% AP9

        % Points 
        wall_end_points_ap9 = [(inter_lifi_ap_dist/4)*[8:2:12]; [0*(inter_lifi_ap_dist/4)*ones(1,3)]]';
        edge_points_ap9 = [(inter_lifi_ap_dist/4)*[8 8 10 12 12];[(inter_lifi_ap_dist/4)*[2 4 4 4 2]]]';
        adjacent_aps_ap9 = [pos_lifi_ap(5,:); pos_lifi_ap(6,:); pos_lifi_ap(10,:); pos_lifi_ap(14,:); pos_lifi_ap(13,:)]; % same order as edge points
        cross_points_ap9 =  [(inter_lifi_ap_dist/4)*[8 9 11 12 ];[(inter_lifi_ap_dist/4)*[3 4 4 3]]]';
        adjacent_cross_points_ap9 = [(inter_lifi_ap_dist/4)*[7 8 12 13];[(inter_lifi_ap_dist/4)*[4 5 5 4]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap9 = [45 90 135]'; % angles for trajectory
        angles_edge_points_ap9 = [0 315 270 225 180]';
        angles_cross_points_ap9 = [315 315 225 225]';

        % Trajectories
        startPoint_to_wall_ap9 = [wall_end_points_ap9(:,1) + cosd(angles_start_points_to_wall_ap9)*distances(1,kk) wall_end_points_ap9(:,2) + sind(angles_start_points_to_wall_ap9)*distances(1,kk)];
        startPoint_to_cell_center_ap9 = [edge_points_ap9(:,1) + cosd(angles_edge_points_ap9)*distances(1,kk) edge_points_ap9(:,2) + sind(angles_edge_points_ap9)*distances(1,kk)];
        startPoint_to_cross_edge_ap9 = [cross_points_ap9(:,1) + cosd(angles_cross_points_ap9)*distances(1,kk) cross_points_ap9(:,2) + sind(angles_cross_points_ap9)*distances(1,kk)];
        
        %%% AP8

        % Points 
        wall_end_points_ap8 = [(inter_lifi_ap_dist/4)*[4:2:8]; [16*(inter_lifi_ap_dist/4)*ones(1,3)]]';
        edge_points_ap8 = [(inter_lifi_ap_dist/4)*[4 4 6 8 8];[(inter_lifi_ap_dist/4)*[14 12 12 12 14]]]';
        adjacent_aps_ap8 = [pos_lifi_ap(4,:); pos_lifi_ap(3,:); pos_lifi_ap(7,:); pos_lifi_ap(11,:); pos_lifi_ap(12,:)]; % same order as edge points
        cross_points_ap8 =  [(inter_lifi_ap_dist/4)*[4 5 7 8 ];[(inter_lifi_ap_dist/4)*[13 12 12 13]]]';
        adjacent_cross_points_ap8 = [(inter_lifi_ap_dist/4)*[3 4 8 9];[(inter_lifi_ap_dist/4)*[12 11 11 12]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap8 = [315 270 225]'; % angles for trajectory
        angles_edge_points_ap8 = [0 45 90 135 180]';
        angles_cross_points_ap8 = [45 45 135 135]';

        % Trajectories
        startPoint_to_wall_ap8 = [wall_end_points_ap8(:,1) + cosd(angles_start_points_to_wall_ap8)*distances(1,kk) wall_end_points_ap8(:,2) + sind(angles_start_points_to_wall_ap8)*distances(1,kk)];
        startPoint_to_cell_center_ap8 = [edge_points_ap8(:,1) + cosd(angles_edge_points_ap8)*distances(1,kk) edge_points_ap8(:,2) + sind(angles_edge_points_ap8)*distances(1,kk)];
        startPoint_to_cross_edge_ap8 = [cross_points_ap8(:,1) + cosd(angles_cross_points_ap8)*distances(1,kk) cross_points_ap8(:,2) + sind(angles_cross_points_ap8)*distances(1,kk)];

        
        %%% AP12

        % Points 
        wall_end_points_ap12 = [(inter_lifi_ap_dist/4)*[8:2:12]; [16*(inter_lifi_ap_dist/4)*ones(1,3)]]';
        edge_points_ap12 = [(inter_lifi_ap_dist/4)*[8 8 10 12 12];[(inter_lifi_ap_dist/4)*[14 12 12 12 14]]]';
        adjacent_aps_ap12 = [pos_lifi_ap(8,:); pos_lifi_ap(7,:); pos_lifi_ap(11,:); pos_lifi_ap(15,:); pos_lifi_ap(16,:)]; % same order as edge points
        cross_points_ap12 =  [(inter_lifi_ap_dist/4)*[8 9 11 12 ];[(inter_lifi_ap_dist/4)*[13 12 12 13]]]';
        adjacent_cross_points_ap12 = [(inter_lifi_ap_dist/4)*[7 8 12 13];[(inter_lifi_ap_dist/4)*[12 11 11 12]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap12 = [315 270 225]'; % angles for trajectory
        angles_edge_points_ap12 = [0 45 90 135 180]';
        angles_cross_points_ap12 = [45 45 135 135]';

        % Trajectories
        startPoint_to_wall_ap12 = [wall_end_points_ap12(:,1) + cosd(angles_start_points_to_wall_ap12)*distances(1,kk) wall_end_points_ap12(:,2) + sind(angles_start_points_to_wall_ap12)*distances(1,kk)];
        startPoint_to_cell_center_ap12 = [edge_points_ap12(:,1) + cosd(angles_edge_points_ap12)*distances(1,kk) edge_points_ap12(:,2) + sind(angles_edge_points_ap12)*distances(1,kk)];
        startPoint_to_cross_edge_ap12 = [cross_points_ap12(:,1) + cosd(angles_cross_points_ap12)*distances(1,kk) cross_points_ap12(:,2) + sind(angles_cross_points_ap12)*distances(1,kk)];
        
        %%% AP14

        % Points
        wall_end_points_ap14 = [10*ones(1,3); [(inter_lifi_ap_dist/4)*[4:2:8]]]';
        edge_points_ap14 = [(inter_lifi_ap_dist/4)*[14 12 12 12 14];[(inter_lifi_ap_dist/4)*[4 4 6 8 8]]]';
        adjacent_aps_ap14 = [pos_lifi_ap(13,:); pos_lifi_ap(9,:); pos_lifi_ap(10,:); pos_lifi_ap(11,:); pos_lifi_ap(15,:)]; % same order as edge points
        cross_points_ap14 =  [(inter_lifi_ap_dist/4)*[13 12 12 13];[(inter_lifi_ap_dist/4)*[4 5 7 8]]]';
        adjacent_cross_points_ap14 = [(inter_lifi_ap_dist/4)*[12 11 11 12];[(inter_lifi_ap_dist/4)*[3 4 8 9]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap14 = [135 180 225]'; % angles for trajectory
        angles_edge_points_ap14 = [90 45 0 315 270]';
        angles_cross_points_ap14 = [45 45 315 315]';

        % Trajectories
        startPoint_to_wall_ap14 = [wall_end_points_ap14(:,1) + cosd(angles_start_points_to_wall_ap14)*distances(1,kk) wall_end_points_ap14(:,2) + sind(angles_start_points_to_wall_ap14)*distances(1,kk)];
        startPoint_to_cell_center_ap14 = [edge_points_ap14(:,1) + cosd(angles_edge_points_ap14)*distances(1,kk) edge_points_ap14(:,2) + sind(angles_edge_points_ap14)*distances(1,kk)];
        startPoint_to_cross_edge_ap14 = [cross_points_ap14(:,1) + cosd(angles_cross_points_ap14)*distances(1,kk) cross_points_ap14(:,2) + sind(angles_cross_points_ap14)*distances(1,kk)];
        
        %%% AP15

        % Points
        wall_end_points_ap15 = [10*ones(1,3); [(inter_lifi_ap_dist/4)*[8:2:12]]]';
        edge_points_ap15 = [(inter_lifi_ap_dist/4)*[14 12 12 12 14];[(inter_lifi_ap_dist/4)*[8 8 10 12 12]]]';
        adjacent_aps_ap15 = [pos_lifi_ap(14,:); pos_lifi_ap(10,:); pos_lifi_ap(11,:); pos_lifi_ap(12,:); pos_lifi_ap(16,:)]; % same order as edge points
        cross_points_ap15 =  [(inter_lifi_ap_dist/4)*[13 12 12 13];[(inter_lifi_ap_dist/4)*[8 9 11 12]]]';
        adjacent_cross_points_ap15 = [(inter_lifi_ap_dist/4)*[12 11 11 12];[(inter_lifi_ap_dist/4)*[7 8 12 13]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap15 = [135 180 225]'; % angles for trajectory
        angles_edge_points_ap15 = [90 45 0 315 270]';
        angles_cross_points_ap15 = [45 45 315 315]';

        % Trajectories
        startPoint_to_wall_ap15 = [wall_end_points_ap15(:,1) + cosd(angles_start_points_to_wall_ap15)*distances(1,kk) wall_end_points_ap15(:,2) + sind(angles_start_points_to_wall_ap15)*distances(1,kk)];
        startPoint_to_cell_center_ap15 = [edge_points_ap15(:,1) + cosd(angles_edge_points_ap15)*distances(1,kk) edge_points_ap15(:,2) + sind(angles_edge_points_ap15)*distances(1,kk)];
        startPoint_to_cross_edge_ap15 = [cross_points_ap15(:,1) + cosd(angles_cross_points_ap15)*distances(1,kk) cross_points_ap15(:,2) + sind(angles_cross_points_ap15)*distances(1,kk)];
        
        %%% Peparing data structures

        ap_to_wall_trajectory_dictionary(:,:,1) = [startPoint_to_wall_ap2 wall_end_points_ap2]; 
        ap_to_wall_trajectory_dictionary(:,:,2) = [startPoint_to_wall_ap3 wall_end_points_ap3]; 
        ap_to_wall_trajectory_dictionary(:,:,3) = [startPoint_to_wall_ap5 wall_end_points_ap5]; 
        ap_to_wall_trajectory_dictionary(:,:,4) = [startPoint_to_wall_ap9 wall_end_points_ap9]; 
        ap_to_wall_trajectory_dictionary(:,:,5) = [startPoint_to_wall_ap8 wall_end_points_ap8]; 
        ap_to_wall_trajectory_dictionary(:,:,6) = [startPoint_to_wall_ap12 wall_end_points_ap12]; 
        ap_to_wall_trajectory_dictionary(:,:,7) = [startPoint_to_wall_ap14 wall_end_points_ap14]; 
        ap_to_wall_trajectory_dictionary(:,:,8) = [startPoint_to_wall_ap15 wall_end_points_ap15]; 
        
        ap_to_cell_center_trajectory_dictionary(:,:,1) = [startPoint_to_cell_center_ap2 edge_points_ap2]; 
        ap_to_cell_center_trajectory_dictionary(:,:,2) = [startPoint_to_cell_center_ap3 edge_points_ap3]; 
        ap_to_cell_center_trajectory_dictionary(:,:,3) = [startPoint_to_cell_center_ap5 edge_points_ap5]; 
        ap_to_cell_center_trajectory_dictionary(:,:,4) = [startPoint_to_cell_center_ap9 edge_points_ap9]; 
        ap_to_cell_center_trajectory_dictionary(:,:,5) = [startPoint_to_cell_center_ap8 edge_points_ap8];
        ap_to_cell_center_trajectory_dictionary(:,:,6) = [startPoint_to_cell_center_ap12 edge_points_ap12]; 
        ap_to_cell_center_trajectory_dictionary(:,:,7) = [startPoint_to_cell_center_ap14 edge_points_ap14]; 
        ap_to_cell_center_trajectory_dictionary(:,:,8) = [startPoint_to_cell_center_ap15 edge_points_ap15]; 
        
        ap_to_cross_edge_trajectory_dictionary(:,:,1) = [startPoint_to_cross_edge_ap2 adjacent_cross_points_ap2]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,2) = [startPoint_to_cross_edge_ap3 adjacent_cross_points_ap3]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,3) = [startPoint_to_cross_edge_ap5 adjacent_cross_points_ap5]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,4) = [startPoint_to_cross_edge_ap9 adjacent_cross_points_ap9]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,5) = [startPoint_to_cross_edge_ap8 adjacent_cross_points_ap8]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,6) = [startPoint_to_cross_edge_ap12 adjacent_cross_points_ap12]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,7) = [startPoint_to_cross_edge_ap14 adjacent_cross_points_ap14]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,8) = [startPoint_to_cross_edge_ap15 adjacent_cross_points_ap15]; 
        
        trajetroyType_cell{1} = ap_to_wall_trajectory_dictionary;
        trajetroyType_cell{2} = ap_to_cell_center_trajectory_dictionary;
        trajetroyType_cell{3} = ap_to_cross_edge_trajectory_dictionary;

        
        for traj_type = 1: length(trajectory_types)
            
            trajectory_type = trajectory_types(1,traj_type); % To the wall
            current_trajectory_type_dictionary = cell2mat(trajetroyType_cell(traj_type));
            
            for ap_num = 1 : length(AP_numbers)
                
                cell_number = AP_numbers(1,ap_num);
                current_ap_trajectories = current_trajectory_type_dictionary(:,:,ap_num);
                
                for p = 1: length(current_ap_trajectories(:,1))
                    pos_ue_k_1 = current_ap_trajectories(p,1:2);
                    pos_ue_k = current_ap_trajectories(p,3:4);

                    pos_ue_n_1 = pos_ue_k_1;

                    %%% Compute transition length D_k = |Pk - Pk-1|  
                    dist_vect = [pos_ue_k_1;pos_ue_k];
                    D_k = pdist(dist_vect,'euclidean');

                    %%% Compute angle of direction omega_d = tan-1(yk - yk-1/xk -xk-1)
                    delta_x = pos_ue_k(1,1) - pos_ue_k_1(1,1);
                    delta_y = pos_ue_k(1,2) - pos_ue_k_1(1,2);
                    angle = atand(abs(delta_y/delta_x));

                    %%% Initialize parameters of current excursion from Pk-1 to Pk
                    N_measurement = 0;

                    while N_measurement <= t_trigger/T_c_theta

                        if delta_x >= 0
                            if delta_y >= 0
                                x_n = pos_ue_n_1(1,1) + speeds(1,k)* T_c_theta * cosd(angle); % case 1
                                y_n = pos_ue_n_1(1,2) + speeds(1,k)* T_c_theta * sind(angle);
                                omega_d_deg = angle;
                            else
                                x_n = pos_ue_n_1(1,1) + speeds(1,k)* T_c_theta * cosd(360 - angle); % case 4
                                y_n = pos_ue_n_1(1,2) + speeds(1,k)* T_c_theta * sind(360 - angle);
                                omega_d_deg = 360 - angle;
                            end
                        else
                            if delta_y >= 0
                                x_n = pos_ue_n_1(1,1) + speeds(1,k)* T_c_theta * cosd(180 - angle); % case 2
                                y_n = pos_ue_n_1(1,2) + speeds(1,k)* T_c_theta * sind(180 - angle);
                                omega_d_deg = 180 - angle;
                            else
                                x_n = pos_ue_n_1(1,1) + speeds(1,k)* T_c_theta * cosd(180 + angle); % case 3
                                y_n = pos_ue_n_1(1,2) + speeds(1,k)* T_c_theta * sind(180 + angle);
                                omega_d_deg = 180 + angle;
                            end
                        end

                        %%% Sava data to csv file

                        [snr_lifi_db_main_user,snr_wifi_db_main_user,pos_UEs] = get_lifi_wifi_snr(x_n,y_n,omega_d_deg);
                        input_all_data = [AP_numbers(1,ap_num),speeds(1,k),pos_UEs(1,:),snr_lifi_db_main_user,snr_wifi_db_main_user,trajectory_type];
                        dlmwrite('all_data.csv',input_all_data,'delimiter',',','-append');

                        if N_measurement == 0
                           snr_lifi_t0 = snr_lifi_db_main_user;
                           snr_wifi_t0 = snr_wifi_db_main_user;
                        end

                        if N_measurement == 2
                           snr_lifi_t = snr_lifi_db_main_user - snr_lifi_t0;
                           snr_wifi_t = snr_wifi_db_main_user - snr_wifi_t0;
                           
                           input_dataset = [AP_numbers(1,ap_num),speeds(1,k),pos_UEs(1,:),snr_lifi_db_main_user, snr_lifi_t,snr_wifi_t,trajectory_type];
                           dlmwrite('dataset.csv',input_dataset,'delimiter',',','-append');
                           
                        end

                        %%% Preparing the next iteration whithin the while loop
                        pos_ue_n = [x_n y_n];  % position
                        pos_ue_n_1 = pos_ue_n;
                        N_measurement = N_measurement + 1;
                    end
                end

            end
   

        end
   
    end
    
end












