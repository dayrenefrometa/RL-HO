function [los_clear_index_3d_mtx]=cylinder_blocker_status_fun_v2(r_body,H_a,H_u,H_b,B_loc_mtx,AP_loc_mtx,UE_loc_mtx)
% function to determine whether the LoS path are blocked by blockers with cylinder shape
% input: 
% - r_body          % radius of cylinder [1x1]
% - H_a             % hieght of AP [1x1]
% - H_u             % hieght of UE [1x1]
% - H_b             % hieght of blocker [1x1]
% - B_loc_mtx       % box 2D locations [N_cylinder x2]
% - AP_loc_mtx      % AP 3D locations [N_AP x2]
% - UE_loc_mtx      % UE 3D locations [N_UE x2]
% output:
% - los_clear_index_3d_mtx  % matrix shows the blockage status [N_AP x N_cylinder x N_UE] (0 means blocked, 1 means no blockage)


N_a=size(AP_loc_mtx,1);
N_u=size(UE_loc_mtx,1);
N_b=size(B_loc_mtx,1);
N_l=N_a*N_u;

AP_loc_ext_mtx=kron(ones(N_u,1),AP_loc_mtx);
UE_loc_ext_mtx=kron(UE_loc_mtx,ones(N_a,1));

k_los=(H_b-H_u)/(H_a-H_u);
midpoint_loc_ext_mtx=k_los*(AP_loc_ext_mtx-UE_loc_ext_mtx)+UE_loc_ext_mtx;

UE_loc_ext_ext_mtx=kron(UE_loc_ext_mtx,ones(N_b,1));
midpoint_loc_ext_ext_mtx=kron(midpoint_loc_ext_mtx,ones(N_b,1));
B_loc_ext_mtx=kron(ones(N_l,1),B_loc_mtx);
cos_angle_mid_vec=sum((B_loc_ext_mtx-midpoint_loc_ext_ext_mtx).*(UE_loc_ext_ext_mtx-midpoint_loc_ext_ext_mtx),2);
cos_angle_UE_vec=sum((B_loc_ext_mtx-UE_loc_ext_ext_mtx).*(midpoint_loc_ext_ext_mtx-UE_loc_ext_ext_mtx),2);
index1_vec=cos_angle_mid_vec>0&cos_angle_UE_vec<0;
index2_vec=cos_angle_mid_vec<0&cos_angle_UE_vec>0;
index3_vec=cos_angle_mid_vec>=0&cos_angle_UE_vec>=0;
dist1_vec=sqrt(sum((UE_loc_ext_ext_mtx-B_loc_ext_mtx).^2,2));
dist2_vec=sqrt(sum((midpoint_loc_ext_ext_mtx-B_loc_ext_mtx).^2,2));
dist3_vec=abs((midpoint_loc_ext_ext_mtx(:,2)-UE_loc_ext_ext_mtx(:,2)).*B_loc_ext_mtx(:,1)-...
              (midpoint_loc_ext_ext_mtx(:,1)-UE_loc_ext_ext_mtx(:,1)).*B_loc_ext_mtx(:,2)+...
               midpoint_loc_ext_ext_mtx(:,1).*UE_loc_ext_ext_mtx(:,2)-...
               midpoint_loc_ext_ext_mtx(:,2).*UE_loc_ext_ext_mtx(:,1))./sqrt(sum((UE_loc_ext_ext_mtx-midpoint_loc_ext_ext_mtx).^2,2));

dist_vec=dist1_vec.*index1_vec+dist2_vec.*index2_vec+dist3_vec.*index3_vec;
los_clear_index_vec=dist_vec>r_body;
los_clear_index_mtx=reshape(los_clear_index_vec,N_b,N_l)';
los_clear_index_3d_mtx=zeros(N_a,N_b,N_u);
for n_u=1:N_u
    los_clear_index_3d_mtx(:,:,n_u)=los_clear_index_mtx([1:N_a]+(n_u-1)*N_a,:);
end


% figure;hold on
% plot(AP_loc_ext_mtx(:,1),AP_loc_ext_mtx(:,2),'y*')
% plot(midpoint_loc_ext_mtx(:,1),midpoint_loc_ext_mtx(:,2),'r*')
% plot(UE_loc_ext_mtx(:,1),UE_loc_ext_mtx(:,2),'g*')
% plot([midpoint_loc_ext_mtx(:,1) UE_loc_ext_mtx(:,1)]',[midpoint_loc_ext_mtx(:,2) UE_loc_ext_mtx(:,2)]','k-')
% plot(B_loc_mtx(:,1),B_loc_mtx(:,2),'bo')
% axis equal






