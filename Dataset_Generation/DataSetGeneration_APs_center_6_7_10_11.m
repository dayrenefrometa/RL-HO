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
trajectory_types = [2 3]; % 1 -> to wall, 2 -> to center of cell, 3 -> to cross the edge of cell
AP_numbers = [6 7 10 11]; 

for k = 1: length(speeds)
    
    for kk = 1: length(distances) 
        
        %%% AP6

        % Points
        edge_points_ap6 = [(inter_lifi_ap_dist/4)*[4 6 8 8 8 6 4 4];[(inter_lifi_ap_dist/4)*[4 4 4 6 8 8 8 6]]]';
        adjacent_aps_ap6 = [pos_lifi_ap(1,:); pos_lifi_ap(5,:); pos_lifi_ap(9,:); pos_lifi_ap(10,:); pos_lifi_ap(11,:); pos_lifi_ap(7,:); pos_lifi_ap(3,:); pos_lifi_ap(2,:)]; % same order as edge points
        cross_points_ap6 =  [(inter_lifi_ap_dist/4)*[5 7 8 8 7 5 4 4];[(inter_lifi_ap_dist/4)*[4 4 5 7 8 8 7 5]]]';
        adjacent_cross_points_ap6 = [(inter_lifi_ap_dist/4)*[4 8 9 9 8 4 3 3];[(inter_lifi_ap_dist/4)*[3 3 4 8 9 9 8 4]]]';

        % Angles to define trajectories
        angles_edge_points_ap6 = [45 90 135 180 225 270 315 0]'; % angles for trajectory
        angles_cross_points_ap6 = [45 135 135 225 225 315 315 45]';

        % Trajectories
        startPoint_to_cell_center_ap6 = [edge_points_ap6(:,1) + cosd(angles_edge_points_ap6)*distances(1,kk) edge_points_ap6(:,2) + sind(angles_edge_points_ap6)*distances(1,kk)];
        startPoint_to_cross_edge_ap6 = [cross_points_ap6(:,1) + cosd(angles_cross_points_ap6)*distances(1,kk) cross_points_ap6(:,2) + sind(angles_cross_points_ap6)*distances(1,kk)];
        
        diff_cell_center = edge_points_ap6 - startPoint_to_cell_center_ap6;
        diff_cell_edge = cross_points_ap6 - startPoint_to_cross_edge_ap6;
            
        if k == 1
            figure;
            quiver(startPoint_to_cell_center_ap6(:,1),startPoint_to_cell_center_ap6(:,2),diff_cell_center(:,1),diff_cell_center(:,2),0,'r', 'linewidth', 2);
            hold on;
            quiver(startPoint_to_cross_edge_ap6(:,1),startPoint_to_cross_edge_ap6(:,2),diff_cell_edge(:,1),diff_cell_edge(:,2),0,'m', 'linewidth', 2);
            xlim([0, 10]);
            ylim([0,10]);
            grid;
            title('Trajectories', 'Interpreter', 'Latex')
            legend('to the center of a cell', 'to the edge of a cell', 'Interpreter', 'Latex')
        end

        %%% AP7

        % Points
        edge_points_ap7 = [(inter_lifi_ap_dist/4)*[4 6 8 8 8 6 4 4];[(inter_lifi_ap_dist/4)*[8 8 8 10 12 12 12 10]]]';
        adjacent_aps_ap7 = [pos_lifi_ap(2,:); pos_lifi_ap(5,:); pos_lifi_ap(10,:); pos_lifi_ap(11,:); pos_lifi_ap(12,:); pos_lifi_ap(8,:); pos_lifi_ap(4,:); pos_lifi_ap(3,:)]; % same order as edge points
        cross_points_ap7 =  [(inter_lifi_ap_dist/4)*[5 7 8 8 7 5 4 4];[(inter_lifi_ap_dist/4)*[8 8 9 11 12 12 11 9]]]';
        adjacent_cross_points_ap7 = [(inter_lifi_ap_dist/4)*[4 8 9 9 8 4 3 3];[(inter_lifi_ap_dist/4)*[7 7 8 12 13 13 12 8]]]';

        % Angles to define trajectories
        angles_edge_points_ap7 = [45 90 135 180 225 270 315 0]'; % angles for trajectory
        angles_cross_points_ap7 = [45 135 135 225 225 315 315 45]';

        % Trajectories
        startPoint_to_cell_center_ap7 = [edge_points_ap7(:,1) + cosd(angles_edge_points_ap7)*distances(1,kk) edge_points_ap7(:,2) + sind(angles_edge_points_ap7)*distances(1,kk)];
        startPoint_to_cross_edge_ap7 = [cross_points_ap7(:,1) + cosd(angles_cross_points_ap7)*distances(1,kk) cross_points_ap7(:,2) + sind(angles_cross_points_ap7)*distances(1,kk)];

        %%% AP10

        % Points
        edge_points_ap10 = [(inter_lifi_ap_dist/4)*[8 10 12 12 12 10 8 8];[(inter_lifi_ap_dist/4)*[4 4 4 6 8 8 8 6]]]';
        adjacent_aps_ap10 = [pos_lifi_ap(5,:); pos_lifi_ap(9,:); pos_lifi_ap(13,:); pos_lifi_ap(14,:); pos_lifi_ap(15,:); pos_lifi_ap(11,:); pos_lifi_ap(7,:); pos_lifi_ap(6,:)]; % same order as edge points
        cross_points_ap10 =  [(inter_lifi_ap_dist/4)*[9 11 12 12 11 9 8 8];[(inter_lifi_ap_dist/4)*[4 4 5 7 8 8 7 5]]]';
        adjacent_cross_points_ap10 = [(inter_lifi_ap_dist/4)*[8 12 13 13 12 8 7 7];[(inter_lifi_ap_dist/4)*[3 3 4 8 9 9 8 4]]]';

        % Angles to define trajectories
        angles_edge_points_ap10 = [45 90 135 180 225 270 315 0]'; % angles for trajectory
        angles_cross_points_ap10 = [45 135 135 225 225 315 315 45]';

        % Trajectories
        startPoint_to_cell_center_ap10 = [edge_points_ap10(:,1) + cosd(angles_edge_points_ap10)*distances(1,kk) edge_points_ap10(:,2) + sind(angles_edge_points_ap10)*distances(1,kk)];
        startPoint_to_cross_edge_ap10 = [cross_points_ap10(:,1) + cosd(angles_cross_points_ap10)*distances(1,kk) cross_points_ap10(:,2) + sind(angles_cross_points_ap10)*distances(1,kk)];
        
        %%% AP11

        % Points
        edge_points_ap11 = [(inter_lifi_ap_dist/4)*[8 10 12 12 12 10 8 8];[(inter_lifi_ap_dist/4)*[8 8 8 10 12 12 12 10]]]';
        adjacent_aps_ap11 = [pos_lifi_ap(5,:); pos_lifi_ap(9,:); pos_lifi_ap(13,:); pos_lifi_ap(14,:); pos_lifi_ap(15,:); pos_lifi_ap(11,:); pos_lifi_ap(7,:); pos_lifi_ap(6,:)]; % same order as edge points
        cross_points_ap11 =  [(inter_lifi_ap_dist/4)*[9 11 12 12 11 9 8 8];[(inter_lifi_ap_dist/4)*[8 8 9 11 12 12 11 9]]]';
        adjacent_cross_points_ap11 = [(inter_lifi_ap_dist/4)*[8 12 13 13 12 8 7 7];[(inter_lifi_ap_dist/4)*[7 7 8 12 13 13 12 8]]]';

        % Angles to define trajectories
        angles_edge_points_ap11 = [45 90 135 180 225 270 315 0]'; % angles for trajectory
        angles_cross_points_ap11 = [45 135 135 225 225 315 315 45]';

        % Trajectories
        startPoint_to_cell_center_ap11 = [edge_points_ap11(:,1) + cosd(angles_edge_points_ap11)*distances(1,kk) edge_points_ap11(:,2) + sind(angles_edge_points_ap11)*distances(1,kk)];
        startPoint_to_cross_edge_ap11 = [cross_points_ap11(:,1) + cosd(angles_cross_points_ap11)*distances(1,kk) cross_points_ap11(:,2) + sind(angles_cross_points_ap11)*distances(1,kk)];
        
        %%% Peparing data structures
       
        ap_to_cell_center_trajectory_dictionary(:,:,1) = [startPoint_to_cell_center_ap6 edge_points_ap6]; 
        ap_to_cell_center_trajectory_dictionary(:,:,2) = [startPoint_to_cell_center_ap7 edge_points_ap7]; 
        ap_to_cell_center_trajectory_dictionary(:,:,3) = [startPoint_to_cell_center_ap10 edge_points_ap10]; 
        ap_to_cell_center_trajectory_dictionary(:,:,4) = [startPoint_to_cell_center_ap11 edge_points_ap11]; 
        
        ap_to_cross_edge_trajectory_dictionary(:,:,1) = [startPoint_to_cross_edge_ap6 adjacent_cross_points_ap6]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,2) = [startPoint_to_cross_edge_ap7 adjacent_cross_points_ap7]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,3) = [startPoint_to_cross_edge_ap10 adjacent_cross_points_ap10]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,4) = [startPoint_to_cross_edge_ap11 adjacent_cross_points_ap11]; 

        trajetroyType_cell{1} = ap_to_cell_center_trajectory_dictionary;
        trajetroyType_cell{2} = ap_to_cross_edge_trajectory_dictionary;

        
        for traj_type = 1: length(trajectory_types)
            
            trajectory_type = trajectory_types(1,traj_type); 
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

