clear all;
close all;

rng(2,'twister'); % Fix the random number generator

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

inter_lifi_ap_dist = 2.5;

N_users = 5; % It includes the main user that we are tracking, plus N_users - 1 users with random location and rotation angles

pos_lifi_ap_aux = combvec(1.25:inter_lifi_ap_dist:len, 1.25:inter_lifi_ap_dist:width); % Create all combinations of vectors. [x,y] of APs
% pos_lifi_ap = [pos_lifi_ap_aux(2,:).' pos_lifi_ap_aux(1,:).']; % [x,y] of APs.
pos_lifi_ap = [pos_lifi_ap_aux(2,:).' pos_lifi_ap_aux(1,:).' ones(size(pos_lifi_ap_aux,2),1)*height]; % [x,y] of APs.

N_lifi_ap = size(pos_lifi_ap_aux,2); % number of LiFi APs

% WiFi

% pos_wifi_ap = [width/2 len/2 height]; % AP in the ceiling
pos_wifi_ap = [width len height]; % AP in a corner
% pos_wifi_ap = [width/2 len/2 3]; % AP in the ground
% pos_wifi_ap = [20 20 height];
N_wifi_ap = size(pos_wifi_ap,2); % number of WiFi APs

% Trajectories

distances = [0.5 1 1.5]; % distance from start point to end point
speeds = [0.5 1 1.5 2 2.5 3];
t_trigger = 0.260; % Time to trigger HO algorithm
T_c_theta = 130e-3; % Coherence time of the polar angle. 134 ms -> walking. 373 ms -> sitting

%         trajetroyType_cell{2} = [startPoint_to_cell_center_ap1 wall_end_points_ap1];


trajectory_types = [1 2 3]; % 1 -> to wall, 2 -> to center of cell, 3 -> to cross the edge of cell
% trajectory_types = [1]; % 1 -> to wall, 2 -> to center of cell, 3 -> to cross the edge of cell
AP_numbers = [1 4 13 16]; 

for k = 1: length(speeds)
    
    for kk = 1: length(distances) 
        
        %%% AP1

        % Points
        wall_end_points_ap1 = [[zeros(1,3) (inter_lifi_ap_dist/4)*[2:2:4]]; [(inter_lifi_ap_dist/4)*[0:2:4] zeros(1,2)]]';
        edge_points_ap1 = [(inter_lifi_ap_dist/4)*[2:2:4 4];[(inter_lifi_ap_dist/4)*[4 4 2]]]';
        adjacent_aps_ap1 = [pos_lifi_ap(2,1:2); pos_lifi_ap(6,1:2); pos_lifi_ap(5,1:2)]; % same order as edge points
        cross_points_ap1 = [(inter_lifi_ap_dist/4)*[3 4] ;[(inter_lifi_ap_dist/4)*[4 3]]]';
        adjacent_cross_points_ap1 = [(inter_lifi_ap_dist/4)*[4 5] ;[(inter_lifi_ap_dist/4)*[5 4]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap1 = [45 0 315 90 135]'; % angles for trajectory
        angles_edge_points_ap1 = [270 225 180]';
        angles_cross_points_ap1 = [225 225]';

        % Trajectories
        startPoint_to_wall_ap1 = [wall_end_points_ap1(:,1) + cosd(angles_start_points_to_wall_ap1)*distances(1,kk) wall_end_points_ap1(:,2) + sind(angles_start_points_to_wall_ap1)*distances(1,kk)];
        startPoint_to_cell_center_ap1 = [edge_points_ap1(:,1) + cosd(angles_edge_points_ap1)*distances(1,kk) edge_points_ap1(:,2) + sind(angles_edge_points_ap1)*distances(1,kk)];
        startPoint_to_cross_edge_ap1 = [cross_points_ap1(:,1) + cosd(angles_cross_points_ap1)*distances(1,kk) cross_points_ap1(:,2) + sind(angles_cross_points_ap1)*distances(1,kk)];

        diff_wall_ap1 = wall_end_points_ap1 - startPoint_to_wall_ap1;
        diff_cell_center_ap1 = edge_points_ap1 - startPoint_to_cell_center_ap1;
        diff_cell_edge_ap1 = cross_points_ap1 - startPoint_to_cross_edge_ap1;
        if k == 1
            figure;
            quiver(startPoint_to_wall_ap1(:,1),startPoint_to_wall_ap1(:,2),diff_wall_ap1(:,1),diff_wall_ap1(:,2),0,'b', 'linewidth', 2);
            hold on;
            quiver(startPoint_to_cell_center_ap1(:,1),startPoint_to_cell_center_ap1(:,2),diff_cell_center_ap1(:,1),diff_cell_center_ap1(:,2),0,'r', 'linewidth', 2);
            hold on;
            quiver(startPoint_to_cross_edge_ap1(:,1),startPoint_to_cross_edge_ap1(:,2),diff_cell_edge_ap1(:,1),diff_cell_edge_ap1(:,2),0,'m', 'linewidth', 2);
            xlim([0, 10]);
            ylim([0,10]);
            grid;
            title('Trajectories', 'Interpreter', 'Latex')
            legend('to the wall', 'to the center of a cell', 'to the edge of a cell', 'Interpreter', 'Latex')
        end

        %%% AP4

        % Points 
        wall_end_points_ap4 = [[zeros(1,3) (inter_lifi_ap_dist/4)*[2:2:4]]; [(inter_lifi_ap_dist/4)*[12:2:16] 16*(inter_lifi_ap_dist/4)*ones(1,2)]]';
        edge_points_ap4 = [(inter_lifi_ap_dist/4)*[2:2:4 4];[(inter_lifi_ap_dist/4)*[12 12 14]]]';
        adjacent_aps_ap4 = [pos_lifi_ap(3,:); pos_lifi_ap(7,:); pos_lifi_ap(8,:)];
        cross_points_ap4 = [(inter_lifi_ap_dist/4)*[3 4] ;[(inter_lifi_ap_dist/4)*[12 13]]]';
        adjacent_cross_points_ap4 = [(inter_lifi_ap_dist/4)*[4 5] ;[(inter_lifi_ap_dist/4)*[11 12]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap4 = [45 0 315 270 225]'; % angles for trajectory
        angles_edge_points_ap4 = [90 135 180]';
        angles_cross_points_ap4 = [135 135]';

        % Trajectories
        startPoint_to_wall_ap4 = [wall_end_points_ap4(:,1) + cosd(angles_start_points_to_wall_ap4)*distances(1,kk) wall_end_points_ap4(:,2) + sind(angles_start_points_to_wall_ap4)*distances(1,kk)];
        startPoint_to_cell_center_ap4 = [edge_points_ap4(:,1) + cosd(angles_edge_points_ap4)*distances(1,kk) edge_points_ap4(:,2) + sind(angles_edge_points_ap4)*distances(1,kk)];
        startPoint_to_cross_edge_ap4 = [cross_points_ap4(:,1) + cosd(angles_cross_points_ap4)*distances(1,kk) cross_points_ap4(:,2) + sind(angles_cross_points_ap4)*distances(1,kk)];
        
        %%% AP13

        % Points
        wall_end_points_ap13 = [(inter_lifi_ap_dist/4)*[12:2:16 16 16]; [zeros(1,3) (inter_lifi_ap_dist/4)*[2:2:4]]]';
        edge_points_ap13 = [(inter_lifi_ap_dist/4)*[12 12 14];[(inter_lifi_ap_dist/4)*[2 4 4]]]';
        adjacent_aps_ap13 = [pos_lifi_ap(9,:); pos_lifi_ap(10,:); pos_lifi_ap(14,:)]; % same order as edge points
        cross_points_ap13 = [(inter_lifi_ap_dist/4)*[12 13] ;[(inter_lifi_ap_dist/4)*[3 4]]]';
        adjacent_cross_points_ap13 = [(inter_lifi_ap_dist/4)*[11 12] ;[(inter_lifi_ap_dist/4)*[4 5]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap13 = [45 90 135 180 225]'; % angles for trajectory
        angles_edge_points_ap13 = [0 315 270]';
        angles_cross_points_ap13 = [315 315]';

        % Trajectories
        startPoint_to_wall_ap13 = [wall_end_points_ap13(:,1) + cosd(angles_start_points_to_wall_ap13)*distances(1,kk) wall_end_points_ap13(:,2) + sind(angles_start_points_to_wall_ap13)*distances(1,kk)];
        startPoint_to_cell_center_ap13 = [edge_points_ap13(:,1) + cosd(angles_edge_points_ap13)*distances(1,kk) edge_points_ap13(:,2) + sind(angles_edge_points_ap13)*distances(1,kk)];
        startPoint_to_cross_edge_ap13 = [cross_points_ap13(:,1) + cosd(angles_cross_points_ap13)*distances(1,kk) cross_points_ap13(:,2) + sind(angles_cross_points_ap13)*distances(1,kk)];
 
        %%% AP16

        % Points 
        wall_end_points_ap16 = [(inter_lifi_ap_dist/4)*[12:2:16 16 16]; [16*(inter_lifi_ap_dist/4)*ones(1,3) (inter_lifi_ap_dist/4)*[14 12]]]';
        edge_points_ap16 = [(inter_lifi_ap_dist/4)*[12 12 14];[(inter_lifi_ap_dist/4)*[14 12 12]]]';
        adjacent_aps_ap16 = [pos_lifi_ap(12,:); pos_lifi_ap(11,:); pos_lifi_ap(15,:)]; % same order as edge points
        cross_points_ap16 = [(inter_lifi_ap_dist/4)*[12 13] ;[(inter_lifi_ap_dist/4)*[13 12]]]';
        adjacent_cross_points_ap16 = [(inter_lifi_ap_dist/4)*[11 12] ;[(inter_lifi_ap_dist/4)*[12 11]]]';

        % Angles to define trajectories
        angles_start_points_to_wall_ap16 = [315 270 225 180 135]'; % angles for trajectory
        angles_edge_points_ap16 = [0 45 90]';
        angles_cross_points_ap16 = [45 45]';

        % Trajectories
        startPoint_to_wall_ap16 = [wall_end_points_ap16(:,1) + cosd(angles_start_points_to_wall_ap16)*distances(1,kk) wall_end_points_ap16(:,2) + sind(angles_start_points_to_wall_ap16)*distances(1,kk)];
        startPoint_to_cell_center_ap16 = [edge_points_ap16(:,1) + cosd(angles_edge_points_ap16)*distances(1,kk) edge_points_ap16(:,2) + sind(angles_edge_points_ap16)*distances(1,kk)];
        startPoint_to_cross_edge_ap16 = [cross_points_ap16(:,1) + cosd(angles_cross_points_ap16)*distances(1,kk) cross_points_ap16(:,2) + sind(angles_cross_points_ap16)*distances(1,kk)];

        %%% Peparing data structures

        ap_to_wall_trajectory_dictionary(:,:,1) = [startPoint_to_wall_ap1 wall_end_points_ap1]; 
        ap_to_wall_trajectory_dictionary(:,:,2) = [startPoint_to_wall_ap4 wall_end_points_ap4]; 
        ap_to_wall_trajectory_dictionary(:,:,3) = [startPoint_to_wall_ap13 wall_end_points_ap13]; 
        ap_to_wall_trajectory_dictionary(:,:,4) = [startPoint_to_wall_ap16 wall_end_points_ap16]; 
        
        ap_to_cell_center_trajectory_dictionary(:,:,1) = [startPoint_to_cell_center_ap1 edge_points_ap1]; 
        ap_to_cell_center_trajectory_dictionary(:,:,2) = [startPoint_to_cell_center_ap4 edge_points_ap4]; 
        ap_to_cell_center_trajectory_dictionary(:,:,3) = [startPoint_to_cell_center_ap13 edge_points_ap13]; 
        ap_to_cell_center_trajectory_dictionary(:,:,4) = [startPoint_to_cell_center_ap16 edge_points_ap16]; 
        
        ap_to_cross_edge_trajectory_dictionary(:,:,1) = [startPoint_to_cross_edge_ap1 adjacent_cross_points_ap1]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,2) = [startPoint_to_cross_edge_ap4 adjacent_cross_points_ap4]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,3) = [startPoint_to_cross_edge_ap13 adjacent_cross_points_ap13]; 
        ap_to_cross_edge_trajectory_dictionary(:,:,4) = [startPoint_to_cross_edge_ap16 adjacent_cross_points_ap16]; 

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

%                         input = [cell_number, speeds(1,k),distances(1,kk),trajectory_type,p,pos_UEs(1,:),snr_lifi_db_main_user,snr_wifi_db_main_user];
%                         dlmwrite('all_data.csv',input,'delimiter',',','-append');
%                         
%                         input_v1 = [speeds(1,k),distances(1,kk),trajectory_type,cell_number,p,pos_UEs(1,:),snr_lifi_db_main_user,snr_wifi_db_main_user];
%                         dlmwrite('all_data_v1.csv',input_v1,'delimiter',',','-append');
%                         
%                         input_v2 = [speeds(1,k),snr_lifi_db_main_user,trajectory_type];
%                         dlmwrite('all_data_v2.csv',input_v2,'delimiter',',','-append');
                        
%                         input_v3 = [speeds(1,k),pos_UEs(1,:),snr_lifi_db_main_user,trajectory_type];
%                         dlmwrite('all_data_v3.csv',input_v3,'delimiter',',','-append');

%                         input_v4 = [speeds(1,k),AP_numbers(1,ap_num),snr_lifi_db_main_user,trajectory_type];
%                         dlmwrite('all_data_v4.csv',input_v4,'delimiter',',','-append');

%                         input_v5 = [speeds(1,k),AP_numbers(1,ap_num),snr_lifi_db_main_user,trajectory_type];
%                         dlmwrite('all_data_v5.csv',input_v5,'delimiter',',','-append');

%                         input_v6 = [speeds(1,k),AP_numbers(1,ap_num),pos_UEs(1,:),snr_lifi_db_main_user,trajectory_type];
%                         dlmwrite('all_data_v6.csv',input_v6,'delimiter',',','-append');

%                         input_v7 = [AP_numbers(1,ap_num),speeds(1,k),pos_UEs(1,:),snr_lifi_db_main_user,snr_wifi_db_main_user,trajectory_type];
%                         dlmwrite('all_data_v7.csv',input_v7,'delimiter',',','-append');
                        
%                         input_v9 = [AP_numbers(1,ap_num),speeds(1,k),pos_UEs(1,:),sinr_lifi_db_main_user,snr_wifi_db_main_user,trajectory_type];
%                         dlmwrite('all_data_v9.csv',input_v9,'delimiter',',','-append');

                        input_v10 = [AP_numbers(1,ap_num),speeds(1,k),pos_UEs(1,:),snr_lifi_db_main_user,snr_wifi_db_main_user,trajectory_type];
                        dlmwrite('all_data_v10.csv',input_v10,'delimiter',',','-append');

                        if N_measurement == 0
                           snr_lifi_t0 = snr_lifi_db_main_user;
                           snr_wifi_t0 = snr_wifi_db_main_user;
                        end

                        if N_measurement == 2
                           snr_lifi_t = snr_lifi_db_main_user - snr_lifi_t0;
                           snr_wifi_t = snr_wifi_db_main_user - snr_wifi_t0;
                           
%                            input = [cell_number, speeds(1,k),distances(1,kk),trajectory_type,p,pos_UEs(1,:),snr_lifi_t,snr_wifi_t];
%                            dlmwrite('dataset.csv',input,'delimiter',',','-append');

%                            input_v1 = [speeds(1,k),distances(1,kk),trajectory_type,cell_number, p,pos_UEs(1,:),snr_lifi_t,snr_wifi_t];
%                            dlmwrite('dataset_v1.csv',input_v1,'delimiter',',','-append');

%                            input_v2 = [speeds(1,k),snr_lifi_t,trajectory_type];
%                            dlmwrite('dataset_v2.csv',input_v2,'delimiter',',','-append');

%                            input_v3 = [speeds(1,k),pos_UEs(1,:),snr_lifi_t,trajectory_type];
%                            dlmwrite('dataset_v3.csv',input_v3,'delimiter',',','-append');

%                            input_v4 = [speeds(1,k),AP_numbers(1,ap_num),snr_lifi_t,trajectory_type];
%                            dlmwrite('dataset_v4.csv',input_v4,'delimiter',',','-append');

%                            input_v6 = [speeds(1,k),AP_numbers(1,ap_num),pos_UEs(1,:),snr_lifi_t,trajectory_type];
%                            dlmwrite('dataset_v6.csv',input_v6,'delimiter',',','-append');
                           
%                            input_v7 = [AP_numbers(1,ap_num),speeds(1,k),pos_UEs(1,:),snr_lifi_t,snr_wifi_t,trajectory_type];
%                            dlmwrite('dataset_v7.csv',input_v7,'delimiter',',','-append');
                           
%                            input_v9 = [AP_numbers(1,ap_num),speeds(1,k),pos_UEs(1,:),snr_lifi_db_main_user, snr_lifi_t,snr_wifi_t,trajectory_type];
%                            dlmwrite('dataset_v9.csv',input_v9,'delimiter',',','-append');

                           input_v10 = [AP_numbers(1,ap_num),speeds(1,k),pos_UEs(1,:),snr_lifi_db_main_user, snr_lifi_t,snr_wifi_t,trajectory_type];
                           dlmwrite('dataset_v10.csv',input_v10,'delimiter',',','-append');
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

