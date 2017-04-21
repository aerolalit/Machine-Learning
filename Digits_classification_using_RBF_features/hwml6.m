%Authors: Lalit Singh
%l.singh@jacobs-university.de
%Lisence: If you are using any part of code or algorithm  presented in this code (for your homework)
%please do not forget to include the source as reference to maintain academic integrity.
clc
addpath(genpath('.'));
load mfeat-pix.txt -ascii
size(mfeat_pix);

%% SPLITTING DATA SET
Train_vectors =mfeat_pix(1: 100, :); % matlab needs to know dimension, 
Test_vectors = mfeat_pix(101: 200, :);  % for stacking all the digits in the train matrix
% fprintf('Splitting data sets to Testing and Training sets......\n');
for dig = 1:9     % choose any one of the digit from 0-9
    raw_digits = mfeat_pix(200*(dig)+1: 100*(2*dig+1), :); %taking first 100 train vectors of each digit
    Train_vectors = [Train_vectors;raw_digits];
    raw_digits = mfeat_pix(200*(dig)+101: 200*(dig+1),:);
    Test_vectors = [Test_vectors;raw_digits];
end
%% Centering Datas
% fprintf('Centering datas.\n');
N = 1000 ;% number of samples
mean_train = sum( Train_vectors(:,:)) / N;
%mean_test = sum( Test_vectors(:,:)) / N;
% Points centralization
digits = Train_vectors - repmat(mean_train, 1000, 1);
test_digits = Test_vectors - repmat(mean_train,1000,1);   % note that we are centralizing the the data around single origin i.e origin of training set of data
%% Initializing cross-validation and Tunable Parameters
factor = 5 % division factor for dividing the data set into trainging and validation set
%syms alpha beta K;
steps  = 20;
%Tuning_parameters = [alpha; beta; K];
K =560; % number of clusters for k means clustering
K_ul = 800; K_ll = 200; % upper and lower limits for K
beta = 2750; % 500 - 4000
alpha = 0.0500; % 0.08 to 0.16 or 0.12
alpha_ll = 0.05; alpha_ul = 1.16;
Tuning_range = [alpha_ll : (alpha_ul-alpha_ll)/steps : alpha_ul ; 500: (3000-500)/steps : 3000 ; K_ll: (K_ul-K_ll)/steps : K_ul];
%%
for tp = 1:3
    average_error = 1000; % initialization with arbitary max num ----average number of miss classification
    best_param = Tuning_range (tp,1);
        
    for tr = 1: steps+1
        if tp ==1       alpha = Tuning_range(tp,tr);
            fprintf('\n current alpha = %d\n', alpha);
        elseif tp ==2   beta = Tuning_range(tp, tr);
            fprintf('\n current beta = %d\n', beta);
        elseif tp ==3   K = Tuning_range(tp,tr) ;
            fprintf('\n current K = %d\n', K);
        else error('Error 101');
        end
      %%  Starting cross validation
                      n_validation = 1000/factor;
        n_test = 1000 - n_validation; % size of the train set
        if( floor(1000/factor) ~= 1000/factor ) 
            error('invalide factor for division'); 
        end
        %% Cross Validation
        total_missC_train = 0;
        total_missC_validation = 0;
        for i =1:factor
            %% Divide available training data into Train_Sets and Validation_sets
            validation_set = zeros(n_validation, 240) ;
            train_set  = zeros(n_test, 240) ;
            num = 1000/(10*factor) ;% = 20
            for j = 0:9
                train_set (j * (100-num) +1 : j*(100-num) +(i-1) *num , : )  = digits(j * 100 +1 : j *100 + (i-1) *num, : );
                validation_set(j * num +1 : (j+1) * num, : ) =  digits(j *100 +(i-1)*num + 1 : j*100+i*num, : );
                train_set (j * (100-num) +(i-1) * num +1 : (100-num) * (j+1), : )  = digits(j * 100 + i*num +1 : (j+1) *100, : );
            end
                %% K-MEANS CLUSTERRING
            [idx, centroid] = kmeans(train_set, K)  ; %vectors of K centroid points
            %%
            % see eqn no (39)
        %     fprintf('Computing feature vectors.\n')
            %f(x, c, beta) = exp(-norm(c-x)^2/beta) ; % to find RBF features

            X = zeros(K, n_test); % feature vector calculated using equation RBF feature extraction equation
            for i = 1:n_test
                for j= 1:K
                    X(j,i) =  exp(-norm(centroid(j,:) - train_set(i,:))^2/beta);   % RBF feature extraction using the equation given in question paper
                end
            end

            %% Y matrix
            Y = zeros(n_test,10);
            for i = 0:9
                for j = 1 : n_test/10
                    Y(n_test/10*i+j, i+1) = 1;
                end
            end
            %% Equation (39) from lecture notes
            W_opt = ( inv(X*X' + alpha ^ 2 * eye(K)) * X * Y  )';
            %% construction of feature vector
             phi_train = X; % we already have computed the feature vector for 
             phi_validation = ones(K,n_validation);
            for i =1: n_validation
                for j= 1:K
                    phi_validation(j,i) =  exp(-norm(centroid(j,:) - validation_set(i,:))^2/beta);  
                    % RBF feature extraction using the equation given in question paper
                end
            end
            %% making prediction of the digits
        %     fprintf('Final Step : Making predictions !!!\n')
            z_train = W_opt * phi_train;
            z_validation = W_opt * phi_validation;
            miss_count_train = 0;
            miss_count_validation = 0;
            for i =1:n_test
                [val, index] =  max(Y(i,:));
                [val1, index1] = max(z_train(:, i));

                if index1 ~= index
                    miss_count_train = miss_count_train +1;
                end

            end

            for i = 1:n_validation
                [val, index] = max(z_validation(:, i));
                if index ~= floor((i-1)/(n_validation/10)) +1
                    miss_count_validation = miss_count_validation +1;
                end
            end
        %     fprintf('Task completed !!!\n\n')
            fprintf('\nTraining miss   = %d\n',miss_count_train);
            fprintf(  'Validation miss = %d \n',miss_count_validation);
            total_missC_train = total_missC_train +  miss_count_train;
            total_missC_validation = total_missC_validation + miss_count_validation;

        end
        fprintf('\nAverage training misclass count = %d\n', total_missC_train /factor);
        fprintf('Average validation misclass count = %d\n', total_missC_validation /factor);
        %%
        if average_error >  total_missC_validation /factor
            best_param = Tuning_range(tp,tr);
            average_error = total_missC_validation/factor;
        end
        if tp ==1 & tr==1 figure(1);
        elseif tp ==2 & tr ==1 figure(2);
        elseif tp ==3 & tr ==1 figure(3);
        else
        end
        
        if tp ==1  scatter(alpha, total_missC_validation/(factor *n_validation *100),'b'); %plot(m, n_mis_train, '-o');
            scatter(alpha, total_missC_train/(factor *n_test *100)  ,'r');
            hold on
        elseif tp ==2
            scatter(beta, total_missC_validation/(factor *n_validation *100),'b'); %plot(m, n_mis_train, '-o');
            scatter(beta, total_missC_train/(factor *n_test *100)  ,'r');
            hold on
        elseif tp ==3
            scatter(K, total_missC_validation/(factor *n_validation *100),'b'); %plot(m, n_mis_train, '-o');
            scatter(K, total_missC_train/(factor *n_test *100)  ,'r');
            hold on
        end
    end
    
    if tp ==1       alpha = best_param;
        fprintf('\nBest alpha found::::>  %d\n', alpha);
        fprintf('\nAverage error  ::::>    %d\n', average_error);
        xlabel('alpha');
        ylabel('percentage of missscalssifications');
        legend('% of training missclass','% of validation misscalssfication')
        title('plot showing the average number of missclassification');
        grid on;
       hold off;
    elseif tp ==2   beta = best_param;
        fprintf('Best beta found:::::>  %d\n', beta);
        fprintf('\nAverage error  ::::>    %d\n', average_error);
         xlabel('beta');
        ylabel('percentage of missscalssifications');
        legend('% of training missclass','% of validation misscalssfication');
        title('plot showing the average number of missclassification');
        grid on;
       hold off;
    elseif tp ==3   K = best_param ;
         xlabel('K');
        ylabel('percentage of missscalssifications');
        legend('% of training missclass','% of validation misscalssfication');
        title('plot showing the average number of missclassification');
        grid on;
       hold off;
        fprintf('Best K found:::::::> %d\n', K);
        fprintf('\nAverage error  ::::>    %d\n', average_error);
    else error('Error 101');
    end
    
end
%%
alpha
beta 
K
%% Now before packaging Supplying the algoriths to the customer let's use all i.e 1000 training set to train the model
[idx, centroid] = kmeans(digits, K) ;
X = zeros(K, 1000); % feature vector calculated using equation RBF feature extraction equation
for i = 1:1000
    for j= 1:K
        X(j,i) =  exp(-norm(centroid(j,:) - digits(i,:))^2/beta);   % RBF feature extraction using the equation given in question paper
    end
end

%% Y matrix
Y = zeros(1000,10);
for i = 0:9
    for j = 1 : 100
        Y(100*i+j, i+1) = 1;
    end
end
 %% Equation (39) from lecture notes
W_opt = ( inv(X*X' + alpha ^ 2 * eye(K)) * X * Y  )';
%This is the model that we supply to costumer
%% We take the customer's data that we hide in secret place and check the performance of our model
 phi_test = ones(K,1000);
for i =1: 1000
    for j= 1:K
        phi_test(j,i) =  exp(-norm(centroid(j,:) - test_digits(i,:))^2/beta);  
        % RBF feature extraction using the equation given in question paper
    end
end
z_test = W_opt * phi_test;
miss_count_test =0;
for i = 1:1000
    [val, index] = max(z_test(:, i));
    if index ~= floor((i-1)/(1000/10)) +1
        miss_count_test = miss_count_test +1;
    end
end
miss_count_test
fprintf('Average  percentage of misscalassification on testing set =    %d percent\n', miss_count_test/10);
