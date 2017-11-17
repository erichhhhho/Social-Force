% MAIN10 evaluation of SVMs

% addpath('libsvm-mat-3.0-1/');
% 
% % Tables:
% %   Obsv(dataset,time,person,px,py,vx,vy,dest,speed,group,flag)
% %   Obst(dataset,px,py)
% %   Dest(dataset,px,py)
% D = load('dataset.mat','D'); D = D.D;
% [Obsv, Obst, Dest] = data2table(D);
% 
% 
% % Global config
% Ninterval = 4;            % Sample every 4 steps = 1.2 seconds
% Npast = [0 1 2 4 8 inf];  % Try 0, 1, 2, 4, 8, and all past
% Nfold = 3;                % 3-fold cross validation
% Datasets = unique(Dest(:,1))';
% 
% % Try different length
% Rd = cell(length(Npast),1);
% Rg = cell(length(Npast),1);
% for i = 1:length(Npast)
%     fprintf('Npast: %d\n', Npast(i));
% 
%     % Prepare training data
%     %   Obs(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
%     %   Sims(simulation,dataset,person,start,duration)
%     [Obs, Sims] = obsv2sim(Obsv,...
%         'Interval',Ninterval,...    % Sampling rate
%         'Duration',0,...            % Future not needed for prediction
%         'Past',Npast(i));           % Past steps to use
%     
%     % Destination classifier
%     fprintf('Destination classifier\n');
%     tic;
%     [Cd, X] = destTrain(Obs,Sims,'Nfold',Nfold);
%     Rd{i} = [Npast(i)*ones(size(X,1),1) X];
%     toc;
%     
%     % Group classifier
%     fprintf('Group classifier\n');
%     tic;
%     rg = cell(length(Datasets),1);
%     for j = Datasets
%         fprintf('  Dataset %d\n',j);
%         [Cg, r] = grpTrain2(Obs,Sims(Sims(:,2)==j,:),'Nfold',Nfold);
%         rg{j} = [j*ones(size(r,1),1) r];
%     end
%     rg = cell2mat(rg);
%     Rg{i} = [Npast(i)*ones(size(rg,1),1) rg];
%     toc;
% end
% Rd = cell2mat(Rd); % (steps,dataset,fold,truth,prediction)
% Rg = cell2mat(Rg); % (steps,dataset,fold,truth,prediction)
% 
% save main10result.mat;
if ~exist('D','var'), load main10result.mat; end

% Analyze the result
Md = zeros(length(Datasets),length(Npast));
Mg = zeros(length(Datasets),length(Npast));
Sd = Md;
Sg = Mg;
for j = Datasets
    for i = 1:length(Npast)
        rd = Rd(Rd(:,1)==Npast(i)&Rd(:,2)==j,3:5);
        rg = Rg(Rg(:,1)==Npast(i)&Rg(:,2)==j,3:5);
        Ad = arrayfun(@(k) nnz(rd(rd(:,1)==k,2)==rd(rd(:,1)==k,3))/nnz(rd(:,1)==k),1:Nfold);
        Ag = arrayfun(@(k) nnz(rg(rg(:,1)==k,2)==1&rg(rg(:,1)==k,3)==1)/nnz(rg(rg(:,1)==k,2)==1),1:Nfold);
        Md(j,i) = mean(Ad);
        Mg(j,i) = mean(Ag);
        Sd(j,i) = std(Ad);
        Sg(j,i) = std(Ag);
    end
end

labels = char({D.label}');              % Dataset names
Ndest = cellfun(@(c) c.C.nr_class ,Cd); % Number of destination class

% Display
fprintf('Destination classification accuracy\n');
% disp(Npast);
% disp(Md);
% disp(Sd);
fprintf('Dataset    & N ');
for j = 1:size(Md,2), fprintf('& %4d ',Npast(j)); end
fprintf('\\\\\n');
for i = 1:size(Md,1)
    fprintf('%s & %d ',labels(i,:),Ndest(i));
    for j = 1:size(Md,2)
        fprintf('& %2.1f ',100*Md(i,j));
    end
    fprintf('\\\\\n');
end
% figure;
% plot(Md');
% title('Destination classification accuracy');
% ylim([0 1]);
% set(gca,'XTickLabel',Npast);
% legend(labels,'Location','SouthEast');


% Display group classification
fprintf('Group classification accuracy\n');
fprintf('Dataset    ');
for j = 1:size(Mg,2), fprintf('& %4d ',Npast(j)); end
fprintf('\\\\\n');
for i = 1:size(Mg,1)
    fprintf('%s ',labels(i,:));
    for j = 1:size(Mg,2)
        fprintf('& %2.1f ',100*Mg(i,j));
    end
    fprintf('\\\\\n');
end
% disp(Npast);
% disp(Mg);
% disp(Sg);
% figure;
% plot(Mg');
% title('Group classification accuracy');
% ylim([0 1]);
% set(gca,'XTickLabel',Npast);
% legend(labels,'Location','SouthEast');