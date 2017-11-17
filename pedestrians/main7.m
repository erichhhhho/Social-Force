%% Main script for single-pedestrian simulation

% Load and convert dataset
%[D,T,Obj] = importData();
%save dataset;
D = load('dataset.mat','D');
D = D.D;

% Create tables:
%   Obsv(dataset,time,person,px,py,vx,vy,dest,speed,group,flag)
%   Obst(dataset,px,py)
%   Dest(dataset,px,py)
[Obsv, Obst, Dest] = data2table(D);
save;

load;
addpath('libsvm-mat-3.0-1/');

%% Global config
Ninterval = 4;
Npast = 1;
TestSet = 3;

fprintf('=== SVM Config ===\n');
fprintf('  Ninterval: %d\n', Ninterval);
fprintf('  Npast: %d\n', Npast);

%% Train SVM classifier

fprintf('\n=== Training SVMs ===\n');
fprintf('Group classifier\n');
% Group classifier
[Obs, Sims] = obsv2sim(Obsv,'Interval',Ninterval,...    % Sampling rate
                            'Duration',0,...            % Future not needed
                            'Past',Npast);              % Past steps to use
Cg = grpTrain2(Obs,Sims);
fprintf('Destination classifier\n');
% Destination classifier
Cd = destTrain(Obs,Sims);

%% Learn parameters

for nduration = 12
    Ninterval = 12;
    Nduration = nduration;
    fprintf('=== Simulation Config ===\n');
    fprintf('  Ninterval: %d\n', Ninterval);
    fprintf('  Nduration: %d\n', Nduration);
    
    % let's use precomputed values...
    params = {...
        [1.034086 1.068818 0.822700 0.940671 1.005634 0.842588],...
        [0.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 0.557420],...
         ...[0.694989 0.654856 0.353771 0.993633 1.866139 0.844283],... %10.824449
         ...[0.800902 0.624982 1.332215 1.035327 0.239304 0.029079 2.248789 0.140373 0.584072]... %17.376034
    };
    fprintf('  Params:\n');
    fprintf('    EWAP:');
    for i = 1:length(params{1}), fprintf(' %f',params{1}(i)); end
    fprintf('\n');
    fprintf('    ATTR:');
    for i = 1:length(params{2}), fprintf(' %f',params{2}(i)); end
    fprintf('\n');

    %% Evaluate path prediction error

    fprintf('\n=== Evaluating simulation ===\n');

    % Create simulation tables:
    %   Obs(simulation,time,person,px,py,vx,vy,dest,speed,group,flag)
    %   Sims(simulation,dataset,person,start,duration)
    time = unique(Obsv(Obsv(:,1)==TestSet,2));
    [Obs, Sims] = obsv2sim(Obsv(Obsv(:,1)==TestSet&Obsv(:,2)<=median(time),:),... % Test dataset
                            'Interval',Ninterval,...    % Sampling rate
                            'Duration',Nduration,...    % Simulation duration
                            'Past',Npast);              % Must be the same to C
    % Evaluate error for each methods
    methods = {'LIN      ','EWAP     ','EWAP+DST ','ATTR     ',...
               'ATTR+DST ','ATTR+GRP ','ATTR+D&G '};
    Err = cell(1,length(methods));
    Res = cell(1,length(methods));
    Grp = cell(1,length(methods));
    
%     % DEBUG
%     [Err{6}, Res{6}, Grp{6}] = simError(Obs,Sims,'attr',...
%         'Params',params{2},'Obst',Obst,'Dest',Dest,'GroupClassifier',Cg);
%     keyboard;

    fprintf('%s',methods{1}); tic;
    [Err{1}, Res{1}, Grp{1}] = simError(Obs,Sims,'lin');
    toc; fprintf('%s',methods{2}); tic;
    [Err{2}, Res{2}, Grp{2}] = simError(Obs,Sims,'ewap',...
        'Params',params{1},'Obst',Obst,'Dest',Dest);
    toc; fprintf('%s',methods{3}); tic;
    [Err{3}, Res{3}, Grp{3}] = simError(Obs,Sims,'ewap',...
        'Params',params{1},'Obst',Obst,'Dest',Dest,'DestClassifier',Cd);
    toc; fprintf('%s',methods{4}); tic;
    [Err{4}, Res{4}, Grp{4}] = simError(Obs,Sims,'attr',...
        'Params',params{2},'Obst',Obst,'Dest',Dest);
    toc; fprintf('%s',methods{5}); tic;
    [Err{5}, Res{5}, Grp{5}] = simError(Obs,Sims,'attr',...
        'Params',params{2},'Obst',Obst,'Dest',Dest,'DestClassifier',Cd);
    toc; fprintf('%s',methods{6}); tic;
    [Err{6}, Res{6}, Grp{6}] = simError(Obs,Sims,'attr',...
        'Params',params{2},'Obst',Obst,'Dest',Dest,'GroupClassifier',Cg);
    toc; fprintf('%s',methods{7}); tic;
    [Err{7}, Res{7}, Grp{7}] = simError(Obs,Sims,'attr',...
        'Params',params{2},'Obst',Obst,'Dest',Dest,...
        'DestClassifier',Cd,'GroupClassifier',Cg);
    toc; fprintf('\n');

    %% Report
    fprintf('\n=== Results ===\n');

    % Header
    fprintf('||Method  ');
    for i = 1:length(methods)
        fprintf('||%s',methods{i});
    end
    fprintf('||\n');

    % Average error
    E = zeros(1,length(Err));
    fprintf('||Error(m)');
    for i = 1:length(Err)
        E(i) = mean(Err{i});
        fprintf('||% f',E(i));
    end
    fprintf('||\n');

    % Average error
    fprintf('||Improve ');
    for i = 1:length(Err)
        fprintf('||% f',(E(1)-E(i))/E(1));
    end
    fprintf('||\n');

    % Destination accuracy
    fprintf('||Dest Acc');
    for i = 1:length(Res)
        E = false(size(Res{i},1),1);
        for j = 1:size(Res{i},1)
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

save main7result.mat