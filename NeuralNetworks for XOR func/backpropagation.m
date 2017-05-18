clc
clear
L = [2 3 1] % number of nodes in each layer starting from layer 1 to K
    % only the number of nodes in layer 2 of L is cangable for this implementaion

w1 = rand( L(2),L(1) )   %   dim: 3*2 //    w1(i,j) = $ w_ij^1 $ where i = 1..L^1  & j = 1..L^0 //  weight for link from layer 0 to layer 1
w2 = rand( L(3),L(2) )   %   dim: 1*3 //    w2(i,j) = $ w_ij^2 $ where i = 1..L^2  & j = 1..L^1 // weight for links from layer 1 to layer 2

w1 = [rand( L(2), 1 )  w1 ]  % dim: 3*3 // padding the weights for the link from bias of layer m-1 to layer: m
w2 = [rand( L(3), 1 )  w2 ]

input  = [ 0 0; 0 1; 1 0; 1 1] % all possible input patterns
output = [0 ; 1; 1; 0]
lamda = 0.1853
%%
figure % opens new figure window
for epoch = 1 : 2000
    w1_dot     = zeros( [  size(w1') ] ) ;
    w1_dot_sum = zeros( [  size(w1') ] ) ;
    w2_dot     = zeros( [ size(w2') ] ) ;
    w2_dot_sum = zeros( [ size(w2') ] ) ;
   miss = 0;
%    fprintf('1\n');
    error = zeros(1,length(output));
    for ix = 1: length(input)  ;
        x0 = [ 1; input(ix, :)' ]  ;% activations from layer: 0
        a1 = [1; w1 * x0 ]   ;         % potential for layer: 1     
        x1 = ( ( 1/ ( 1 + exp(-a1)) ) )';       %      " "        layer: 1
        x2 = w2 * x1    ;    %      " "         layer: 1
%          fprintf('2\n');
        y_hat = x2   ;  % the output of the network
        if round(y_hat) ~= output(ix,:)
            miss = miss + 1;
        end
        d2 = 2 *   ( y_hat - output(ix) ) ;
        
        error(1,ix) = (y_hat - output(ix));
        fprintf('Error :: %f\n',error(1,ix));
        
        
        d_sig = x1 .* (1-x1) ;
        d1 = d_sig .* ( w2' * d2 ) ; % delta^1
%          fprintf('3\n');
        w2_dot = x1 * d2'  ;
        w2_dot_sum = w2_dot_sum + w2_dot ;

        w1_dot = x0 * d1(2:length(d1),:)' ;% remove d1_1 from d1
        w1_dot_sum = w1_dot_sum + w1_dot ;
    end
    plot(epoch, error(1,1), '-.r*',epoch, error(1,2), '-.c*', epoch, error(1,3), ':bs',epoch, error(1,4), '-.g*');
    hold on;
    fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~\nEpoch : %d    miss:  %d \n\n',epoch, miss);
    w2_dot_mean = ( w2_dot_sum ./ length(input) )' ;
    w1_dot_mean = ( w1_dot_sum ./ length(input) )' ;

    w1 =  w1 - lamda * w1_dot_mean;
    w2 =  w2 - lamda * w2_dot_mean;

end
legend('Error for input: (0,0)','Error for input: (0,1)','Error for input: (1,0)','Error for input: (1,1)');
xlabel('Number of Epoch');
ylabel('Error');
title('[ N(\theta)-y ] vs number of Epoch')

