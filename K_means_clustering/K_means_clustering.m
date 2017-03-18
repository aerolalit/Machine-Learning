%Author: Lalit Singh
%Jacobs University
%l.singh@jacobs-university.de
addpath(genpath('.'));
load mfeat-pix.txt -ascii
size(mfeat_pix);
dig1 = 5;
dig2 = 2;
n1 = 200 ;% number of samples
n2 = 200;
digits1 = mfeat_pix(200*(dig1)+1: 200 * (dig1+1), :);
digits2 = mfeat_pix(200*(dig2)+1: 200 * (dig2+1), :);
digits = [digits1; digits2];
size(digits);
K =4; % number of K
n = n1 +n2;
centroid = zeros(K, 240)  ; %vectors of K centroid points
for i = 1:K
    centroid(i,:) = digits(i, :); % picks up first K digits as centroid
end                      % centroids should be choosed randomly however each digits are unique so randomization is not mandatory
%%
%centroid
cluster = zeros(n,1); % keeps track of which digit in digits lies in which of the K clusters
%% Initialization
for i = 1:n
    index = 1;
    min_dist = norm(centroid(1,:) - digits(i,:)); % calculates the euclidian distance between centroid and the particular digits in set of digits
    for j = 2:K
        temp = norm(centroid(j,:) - digits(i, :));
        if temp < min_dist
            min_dist = temp;
            index =j;
        end
        
    end
    cluster(i,1) = index;
end
%% Loop
loopcount = 1; % number of times that the while loop runs to converge
convergence = false;
while(convergence == false)
    loopcount = loopcount +1;
    diff = zeros(K,240); % make matrix of zeros because prof. pathak likes zeros and he is legend
    count = zeros(K,1); % counts the number of elements in particular clustering
    for i = 1:n
        diff(cluster(i,1), :) = diff(cluster(i,1),:) + digits(i,:) - centroid(cluster(i,1),:);
        count(cluster(i,1),1) = count(cluster(i,1),1) +1;
    end
    
    convergence = true;
    for i = 1:K
        prev = centroid(i,:);
        centroid(i,:) = centroid(i,:) + diff(i,:) ./ count(K,1);
        change = prev - centroid(i,:);
        for j = 1:240
            if change(1,j) > 0.00003 %threshold
                convergence = false;
                break;
            end
        end
        
    end
    if convergence == true break; end
    %%% Reclustering
    for i = 1:n
        index = 1;
        min_dist = norm(centroid(1,:) - digits(i,:)); % calculates the euclidian distance between centroid and the particular disits in set of digits
        for j = 2:K
            temp = norm(centroid(j,:) - digits(i, :));
            if temp < min_dist
                min_dist = temp;
                index =j;
            end
        end
        cluster(i,1) = index;
    end
    %%%
end

for i = 1:K
    count = 1;
    for j = 1:n
        if i == cluster(j,1)
            count = count +1;
            pic = digits(j,:);  
            picmatreverse = zeros(15,16);
            % the filling of (:) is done columnwise!
            picmatreverse(:)= - pic;
            picmat = zeros(15,16);
            for k = 1:15
                picmat(:,k)=picmatreverse(:,16-k);
            end
            subplot(10,10,(i-1) * 10 + count);
            pcolor(picmat');
            axis off;
            colormap(gray(10));
        end
        if count ==10 break; end;
    end
    
end

%centroid

