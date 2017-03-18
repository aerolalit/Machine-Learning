%% The raw 200 samples of particular digit out of 2000 data samples is loaded in a matrix called digits
clear
clc
addpath(genpath('.'));
load mfeat-pix.txt -ascii
size(mfeat_pix);
dig = 3; % choose any one of the digit from 0-9
N = 200 ;% number of samples
raw_digits = mfeat_pix(200*(dig)+1: 200 * (dig+1), :); % matrix of sample digit which are not yet centralized
%% Here we calculate the mean of the digit samples
mean = sum( raw_digits(:,:)) / N;
%% All the points in points cloud are centralized
digits = raw_digits - repmat(mean, 200, 1);
%% covariance and finding PCs
C = 1/N * digits' * digits; % we have row vectors in valiable [digits] instead od column vectors
[U,S,V] = svd(C);
sumof_sigma_sqr = sum(diag(S.^2)); % sum of the squre of sigmas
%%
projection = ( (U(:,1:240))' *  digits')' ; % projecction of centralized points on PCs
recons_100_pcnt = projection * U(:,1:240)';
C1 = 1/N * recons_100_pcnt' * recons_100_pcnt;  %covariance matrix
[U1, S1, V1] = svd(C1);
%% percentage reconstruction
recons_percent = 50  % in percentage
%%
sum = 0;
k  = 1 ; %initialization
for i = 240:-1:1
    sum = sum + S1(i,i)^2;
    if(sum/sumof_sigma_sqr > (100 - recons_percent) * 1/100)
        k = i;
        break;
    end
end
%%
k % no of PCs used
projection = ( (U(:,1:k))' *  digits')' ;
result_digits = projection * U(:,1:k)' +repmat(mean, 200, 1) ;
%%
figure(1);
num = 5
for i = 1:num
    for j = 1:num
        pic = result_digits(1 * (i-1)+j ,:);  
        picmatreverse = zeros(15,16);
        % the filling of (:) is done columnwise!
        picmatreverse(:)= - pic;
        picmat = zeros(15,16);
        for k = 1:15
            picmat(:,k)=picmatreverse(:,16-k);
        end
        subplot(num,num,(i-1)* num + j);
        pcolor(picmat');
        axis off;
       colormap(gray(10));
    end
end

