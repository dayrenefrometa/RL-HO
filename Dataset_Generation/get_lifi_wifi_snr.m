function [snr_lifi_db_main_user, snr_wifi_dB_main_user,pos_UEs] = get_lifi_wifi_snr(x_n, y_n,omega_d_deg)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%   IMPORTANT: All the simulation parameters must be the same in RL-HO.m and the scripts to generate the dataset.
%%%%   REFERENCES:
%%%%   For LiFi Channel:
%%%%   [1]: "Modeling the Random Orientation of Mobile Devices: Measurement, Analysis and LiFi Use Case" https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8540452.
%%%%   For WiFi Channel:
%%%%   [2] "Mobility-aware load balancing for hybrid LiFi and WiFi networks" https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8863828
%%%%   [3] "Next Generation Wireless LANs 802.11n and 802.11ac" https://www.cambridge.org/core/books/next-generation-wireless-lans/1C3DF09331104E23D48599AE1D6373D4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

% LiFi
N_users = 5; % It includes the main user that we are tracking, plus N_users - 1 users with random location and rotation angles
inter_lifi_ap_dist = 2.5;
mu_theta_deg = 29.67; % mean of the Gaussian RP [degrees]
sigma_theta_deg = 7.78; % variance onf the Gaussian RP [degrees]

pos_lifi_ap_aux = combvec(1.25:inter_lifi_ap_dist:len, 1.25:inter_lifi_ap_dist:width); % Create all combinations of vectors. [x,y] of APs
pos_lifi_ap = [pos_lifi_ap_aux(2,:).' pos_lifi_ap_aux(1,:).' ones(size(pos_lifi_ap_aux,2),1)*height]; % [x,y,z] of APs.
N_lifi_ap = size(pos_lifi_ap_aux,2); % number of LiFi APs

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
                        11 12 15 0 0 0 0 0];

% WiFi

pos_wifi_ap = [width len height]; % AP in a corner
% pos_wifi_ap = [width/2 len/2 height]; % AP in the ceiling 
% pos_wifi_ap = [width/2 len/2 3]; % AP in the ground (center of the room)
% pos_wifi_ap = [20 20 height];
N_wifi_ap = size(pos_wifi_ap,1); % number of WiFi APs


%%% LiFi channel parameters (as in [1])

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
% Aapd = 1*10^-4; % 1cm2
Aapd_PD1 = 1*10^-4; % 1cm2
% Aapd_PD2 = 0; % 1cm2
Aapd_PD2 = 1*10^-4; % 1cm2
% Aapd = 0.5*10^-4; % 1cm2

% WiFi channel parameters (as in [2,3])

% d_BP = 10; % breakpoint distance in meters. According to [3], Table 3.4 Path loss model parameters (Model D, IEEE 802.11n, typical office enviroment)
d_BP = 5; % Channel model C (small office)
% freq = 2400000000; % operation frequency
freq = 5000000000; % operation frequency

%%% LiFi SNR calculation parameters

% Transmitters
Popt = 3.5; % [Watt] Transmitted optical power
B_LiFi = 20*10^6; %[MHz] Bandwidth of the LiFi APs
% l_conv_elect_opt = 3; %  electric to optical power conversion ratio. An increase of l results in a decrease of the probability of the LiFi signals being outside the LED linear working region. In general, l = 3 can guarantee that less than 0.3% of the signals are clipped

% Receivers
Rpd = 0.53; % [A/W] detector responsivity
k_conv_eff_opt_elect = 1; % optical to electric power conversion effciency
N_LiFi = 1*10^-21; % [A^2/Hz] power spectral density of noise in LiFi (includes shot noise and thermal noise)

%%% WiFi SNR calculation parameters

P_WiFi_AP_dBm = 0; % TX power at WiFi AP in dBm
P_WiFi_AP_Watt = (10^(P_WiFi_AP_dBm/10))/1000;

N_WiFi_dBm = -174; % PSD of noise at the RX in dBm/Hz
N_WiFi_AP_Watt = (10^(N_WiFi_dBm/10))/1000;
B_WiFi = 40*10^6; % Bandwidth at the WiFI AP in Hz

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
omega_rnd_ue_deg = rand(N_users-1,1)*360; 
omega_ue_deg = [omega_d_deg; omega_rnd_ue_deg]; % the azimuth angle for all UEs. The one for main user is on the first element of this vector
omega_UEs = omega_ue_deg*pi/180;

% Polar angle
pd = makedist('Normal','mu',mu_theta_deg,'sigma',sigma_theta_deg);
pd_trunc = truncate(pd,0,90);
theta_n_deg_UEs = random(pd_trunc,N_users,1); % the rotation angle for all UEs. The one for main user is on the first element of this vector
theta_n_UEs = theta_n_deg_UEs*pi/180;
theta_n_UEs_PD2 = pi/2 - theta_n_UEs;

% For debugging fix it to 0 degrees. All UEs looking to the ceiling
%     theta_n_deg_UEs = 90*ones(N_users,1);
%     theta_n_deg_UEs = zeros(N_users,1);
%     theta_n_UEs = theta_n_deg_UEs*pi/180;
%     theta_n_UEs_PD2 = pi/2 - theta_n_UEs;

%%% Blocking elements
cyl_radius = 0.15; % meters
cyl_height = 1.75; % m standing up. Sitting up = 1 m
d_p_ue = 0.3; % distance between UE and person holding it

% users holding the UEs
blocking_elements(1:N_users,:) = [pos_UEs(:,1)-d_p_ue*cos(omega_UEs) pos_UEs(:,2)-d_p_ue*sin(omega_UEs)]; % location (x,y) of the people holding the UEs

% people without UEs (just blocking elements)
lambda_b = 0.1; %unit/m^2
blocking_elements_b = get_poisson_positions(lambda_b, len, width); % this is the location of blocking elements
N_b = size(blocking_elements_b,1); % number of people without device

blocking_elements(N_users+1:N_users+N_b,:) = blocking_elements_b;

AP_loc_mtx=pos_lifi_ap(:,1:2);
UE_loc_mtx=pos_UEs(:,1:2);
B_loc_mtx=blocking_elements; % blocking elements
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

% Calculating the cos of incidence angles

cos_incidence_angle = zeros(N_users,N_lifi_ap);
cos_incidence_angle_PD2 = zeros(N_users,N_lifi_ap);
for usr = 1:N_users
    % cosine of the incidence angle according to [1]
    cos_incidence_angle(usr,:) = (distance_users_aps_x(usr,:)*sin(theta_n_UEs(usr))*cos(omega_UEs(usr)) + distance_users_aps_y(usr,:)*sin(theta_n_UEs(usr))*sin(omega_UEs(usr)) + distance_users_aps_z(usr,:)*cos(theta_n_UEs(usr)))./distance_users_aps(usr,:);
    cos_incidence_angle_PD2(usr,:) = (distance_users_aps_x(usr,:)*sin(theta_n_UEs_PD2(usr))*cos(omega_UEs(usr)) + distance_users_aps_y(usr,:)*sin(theta_n_UEs_PD2(usr))*sin(omega_UEs(usr)) + distance_users_aps_z(usr,:)*cos(theta_n_UEs_PD2(usr)))./distance_users_aps(usr,:);
end

% cos_incidence_angle_deg = cos_incidence_angle*180/pi;        
% cos_incidence_angle_PD2_deg = cos_incidence_angle_PD2*180/pi;

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

snr_lifi_db_main_user = snr_lifi_matrix_dB_total_v1(1,:);
snr_wifi_dB_main_user = snr_wifi_matrix_dB(1,1);

end