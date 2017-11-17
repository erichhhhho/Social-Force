%% Main script for single-pedestrian simulation

addpath('libsvm-mat-3.0-1/');

% Load and convert dataset
D = importData();
% D = load('dataset.mat','D');
% D = D.D;

%   Obsv(dataset,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Obst(dataset,px,py)
%   Dest(dataset,px,py)
[Obsv, Obst, Dest] = data2table(D);

% Split dataset into first/second half
Dind = false(size(Obsv,1),1);
for did = 1:length(D)
    Dind(Obsv(:,1)==did & Obsv(:,2)<median(Obsv(Obsv(:,1)==did,2)))=true;
end

%% Global config
Ninterval = 4;  % Start every 4 frames (1.6 seconds)
Nduration = 12; % 12 step prediction (4.8 seconds)
Npast = 5;      % Use at most 5 history records to predict hidden
TestSet = [3 4 5];


for fold = 2 % Start 2-fold validation
    Dind = ~Dind;
    %% Train SVM classifiers
    fprintf('\n=== Training SVMs ===\n');
    fprintf('  Ninterval: %d\n', Ninterval);
    fprintf('  Npast: %d\n', Npast);
    fprintf('Group classifier\n');
    % Group classifier
    [Obs, Sims] = obsv2sim(Obsv(Dind,:),...
        'Interval',Ninterval,...    % Sampling rate
        'Duration',0,...            % Future not needed
        'Past',Npast);              % Past steps to use
    Cg = grpTrain2(Obs,Sims);
    % Destination classifier
    fprintf('Destination classifier\n');
    Cd = destTrain(Obs,Sims);

    %% Set up simulation config
    params = {...
        [],...
        [1.034086 1.068818 0.822700 0.940671 1.005634 0.842588],...
        [0.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420],...
    };

    % (label,method,parameter,Cd,Cg)
    config = {...
        'LIN    ','lin', 1,[],[];...
        'LTA    ','ewap',2,[],[];...
        ...'LTA+D  ','ewap',2,Cd,[];...
        'ATTR   ','attr',3,[],[];...
        ...'ATTR+D ','attr',3,Cd,[];...
        ...'ATTR+G ','attr',3,[],Cg;...
        ...'ATTR+DG','attr',3,Cd,Cg;...
    };

    %% Evaluate path prediction error
    for ts = TestSet
        fprintf('\n=== TESTSET: %s, FOLD %d ===\n',D(ts).label,fold);
        % Evaluate path prediction error
        fprintf('  Ninterval: %d\n', Ninterval);
        fprintf('  Nduration: %d\n', Nduration);

        % Create simulation tables:
        %   Obs(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
        %   Sims(simulation,dataset,person,start,duration)
        [Obs, Sims] = obsv2sim(Obsv(Obsv(:,1)==ts&~Dind,:),... % Test dataset
                                'Interval',Ninterval,...    % Sampling rate
                                'Duration',Nduration,...    % Simulation duration
                                'Past',Npast);              % Must be the same to C
        % Evaluate error for each methods
        Err = cell(size(config,1),1);
        Res = cell(size(config,1),1);
        Grp = cell(size(config,1),1);

        for i = 1:size(config,1)
            fprintf('  %s ',config{i,1}); tic;
            [Err{i}, Res{i}, Grp{i}] = simError(Obs,Sims,config{i,2},...
                'Obst',Obst,'Dest',Dest,'Params',params{config{i,3}},...
                'DestClassifier',config{i,4},'GroupClassifier',config{i,5},...
                'Past',Npast);
            toc;
        end
        save(['main11_' D(ts).label '_f' num2str(fold) '.mat']);
        
%         %% Render video result
%         aviobj = avifile(['behavior_' D(ts).label '_f' num2str(fold) '.avi'],'fps',2.5);
%         for simid = 1:size(Sims,1)
%             simVisualize(Sims(simid,:),Obs,Res,config(:,1)',D(Sims(simid,2)).video,D(Sims(simid,2)).H);
%             aviobj = addframe(aviobj,gca);
%         end
%         aviobj = close(aviobj);

        %% Report
        fprintf('\nResults\n');

        % Header
        fprintf('||Method  ');
        for i = 1:size(config,1)
            fprintf('||%s',config{i,1});
        end
        fprintf('||\n');

        % Average error
        E = zeros(1,size(config,1));
        fprintf('||Error(m)');
        for i = 1:size(config,1)
            E(i) = mean(Err{i});
            fprintf('||% f',E(i));
        end
        fprintf('||\n');

        % Improvement error
        fprintf('||Improve ');
        for i = 1:length(Err)
            fprintf('||% f',(E(1)-E(i))/E(1));
        end
        fprintf('||\n');

        % Destination accuracy
        fprintf('||Dest Acc');
        for i = 1:length(Res) % for each method
            E = false(size(Res{i},1),1);
            for j = 1:size(Res{i},1) % for each record
                % find the ground truth by matching (1,2,3) columns
                % and see if its 8th column is the same or not
                x = Obs(Obs(:,1)==Res{i}(j,1),:);
                E(j) = x(x(:,2)==Res{i}(j,2)&...
                         x(:,3)==Res{i}(j,3),8)==Res{i}(j,8);
            end
            fprintf('||% f',nnz(E)/size(Res{i},1));
        end
        fprintf('||\n');

        % Group accuracy
        fprintf('||Group Ac');
        for i = 1:length(Grp)
            fprintf('||% f',nnz(Grp{i}(:,4)==Grp{i}(:,5))/size(Grp{i},1));
        end
        fprintf('||\n');
    end

    Dind = ~Dind;
end % End 2-fold validation


% save main11result.mat