%% The raw 100 samples of particular digit out of 2000 data samples is loaded in a matrix called digits
addpath(genpath('.'));
load mfeat-pix.txt -ascii
size(mfeat_pix);
Train_vectors = zeros(1,240); % matlab needs to know dimension, for stacking all the digits in the train matrix
Test_vectors = zeros(1,240);
for dig = 0:9     % for each digit from 0-9 stack first 100 vectors for trainging and other 100 for testing
    raw_digits = mfeat_pix(200*(dig)+1: 100*(dig+dig+1), :); 
    Train_vectors = [Train_vectors;raw_digits];
    raw_digits = mfeat_pix(200*(dig)+101: 200*(dig+1),:);
    Test_vectors = [Test_vectors;raw_digits];
end
%% EXTRATCT ZERO ROW
Train_vectors = Train_vectors(2:1001,:); % delete the first row because it is just a zero vector
Test_vectors = Test_vectors(2:1001,:);
%%
N = 1000 ;          % number of samples
mean = sum( Train_vectors(:,:)) / N;    % mean of the samples    
% Points centralization
digits = Train_vectors - repmat(mean, 1000, 1);

% covariance and finding PCs
C = 1/N * digits' * digits; % we have row vectors in valiable [digits] instead of column vectors
[U,S,V] = svd(C);
for m= 5:5:240
    %m = 5; % extracted features
    projection = ( (U(:,1:m))' *  digits')' ;
    PHI = projection; % regarding (phi1, phi2, ..., phim), as column vectors
    %% Z matrix
    Z = zeros(1000,10);
    for i = 0:9
        for j = 1 : 100
            Z(100*i+j, i+1) = 1;
        end
    end
    W_opt = pinv(PHI)*Z; % psudo inverse: or type directly (PHI'*PHI)^(-1)*PHI'
    W_opt = W_opt';
    %% checking the result for all vector   
    Z_train = W_opt * (U(:,1:m))'* Train_vectors';% Z_approximate
    Z_test = W_opt *(U(:,1:m))'* Test_vectors'; % Z test, decision
    %% checking the # of misclassifications
    n_train = 0; % number of correct class of training data set
    n_test = 0;
    for i = 0:9
        z_opt = zeros(10,1);
        z_opt(i+1)= 1; % setting z_opt (i+1)th element to one for each class
        for j = 1:100
            z_train = Z_train(:,100*i+j); % extract each column of Z_apr matrix
            z_test = Z_test(:, 100*i+j); % each row of Z_test matrix
            % then: the idea is to set to 1 the highest number of vector, other values to 0
            z_train = abs(z_train); % taking abs value of each element
            z_test = abs(z_test);
            max_z_train = max(z_train);
            max_z_test = max(z_test);
            for k = 1:10
                if z_train(k) ~= max_z_train % not equal
                    z_train(k) = 0;
                else
                    z_train(k)= 1;
                end
                if z_test(k)~=max_z_test
                    z_test(k)= 0;
                else
                    z_test(k) = 1;
                end
            end
            
            % comparing z_apr with z_opt
            if z_opt - z_train == zeros(10,1)
                n_train = n_train+1;
            end
            if z_opt - z_test == zeros(10,1)
                n_test = n_test + 1;
            end
        end
    end
    n_mis_train = 1000 - n_train; % # of miscl of training data set
    n_mis_test = 1000 - n_test;
    scatter(m, n_mis_train,'b'); %plot(m, n_mis_train, '-o');
    scatter(m, n_mis_test,'r');
    hold on
    
end
legend('testing set of images','training set of images')
xlabel('no of feature vectors considered (m)')
ylabel('Total number of missclassifications')

