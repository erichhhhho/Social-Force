%% main4.m
% TESTING SCRIPT FOR PARAMETER LEARNING OF ATTRACTION TERM

% Load and preprocess dataset
% if ~exist('ewap_dataset.mat','file')
    seq = ewapLoad('ewap_dataset');
    save ewap_dataset.mat seq;
% end
load ewap_dataset.mat;
[D,Obj] = seq2ewap(seq);

%% Cross validation
nfolds = 3;
% initParams = [0.130081 2.087902 2.327072 2.0732 1.461249 0.7304224];
% initParams = [ 0.010883 2.646596 0.069719 2.496652 2.531520 ...
%                0.145634 0.122078 0.176857];
% initParams = [0.130081 2.087902 2.327072 2.0732 1.461249...
%               0.059400 1.127248 0.215000];
% initParams = [0.621705 5.933826 1.762825 3.830825...
%               0.377449 0.058362 3.758142 0.205888];
initParams = [1.000000 1.000000 1.000000 1.000000...
              1.000000 0.058362 1.000000 0.205888 1.000000];
lb = [0 0 0 0 0 0 0 0 0];
ub = [inf inf inf inf inf inf inf inf inf];
regularization = [];

Dind = unique(D(:,[1 3]),'rows');   % Unique pairs of (dataset,person_id)

if exist('matlabpool','file')==2, parpool open 3; end
params = zeros(nfolds,size(initParams,2));
E = zeros(nfolds,2); % Array to store Avg. Error
for expId = 1:nfolds
    tic;
    % Prepare index of testing samples
    Train = Dind(mod(1:size(Dind,1),nfolds)==expId-1,:);
    Test  = Dind(mod(1:size(Dind,1),nfolds)~=expId-1,:);
    
    % Training = !Testing index
    % Find lambda_3, lambda_5
    params(expId,[6 8]) = ...
        fminsearch(@(x) sum(attractionError(D,Obj,...
            [initParams(1:5) x(1) initParams(7) x(2) initParams(9)],...
            'Index',Train)),...
            initParams([6 8]),...lb([6 8]),ub([6 8]),...
            optimset('Display','iter',...
                     'Diagnostics','on',...
                     'Jacobian','off',...
                     'MaxIter',30)...
                    );
    % Find other params
    params(expId,[1:5 7 9]) =...
        ga(@(x) mean(myError3(D,Obj,...
                    [x(1:5) params(expId,6) x(6) params(expId,8) x(7)],...
                    'Index',Train)),...
            7,[],[],[],[],lb([1:5 7 9]),ub([1:5 7 9]),[],...
            gaoptimset('Display','iter',...
                'CreationFcn',@gacreationlinearfeasible,...
                'PopInitRange',[(1-0.3)*initParams([1:5 7 9]);...
                                (1+0.3)*initParams([1:5 7 9])],...
                'PopulationSize',10,...
                'StallGenLimit',10,...
                'Generations',30,...
                'UseParallel','always')...
            );
%     params(expId,[1:5 7]) =...
%         lsqnonlin(@(x) myError3(D,Obj,...
%                 [x(1:5) params(expId,6) x(6) params(expId,8)],...
%                 'Index',Train),...
%             initParams([1:5 7]),lb([1:5 7]),ub([1:5 7]),...
%             optimset('Display','iter',...
%                 'GradObj','off',...
%                 'MaxIter',30)...
%             );
    % Test
    E(expId,1) = mean(myError3(D,Obj,params(expId,:),'Index',Test));
    % Error in linear model
    E(expId,2) = mean(linError(D,'Index',Test));
    % Print
    fprintf('\nParams\n');
    fprintf('||  sigma_d||  sigma_w||     beta|| lambda_1|| lambda_2|| lambda_3|| lambda_4|| lambda_5|| lambda_6||\n');
    for j=1:size(params,2)
        fprintf('||% 8f',params(expId,j));
    end
    fprintf('||\n');
    fprintf('\nError\n');
    fprintf('||LTA Avg. (m)||LIN Avg. (m)||Improvement (%%)||\n');
    fprintf('||% f',E(expId,1));
    fprintf('||% f',E(expId,2));
    fprintf('||% f',(E(expId,2)-E(expId,1))/E(expId,2)*100);
    fprintf('||\n');
    toc;
end
if exist('matlabpool','file')==2, parpool close; end
save ewap_mod_results

%% Display
fprintf('\nParams\n');
fprintf('||Fold||  sigma_d||  sigma_w||     beta|| lambda_1|| lambda_2|| lambda_3|| lambda_4|| lambda_5|| lambda_6||\n');

for i = 1:size(params,1)
    fprintf('||%4d',i);
    for j=1:size(params,2)
        fprintf('||% 8f',params(i,j));
    end
    fprintf('||\n');
end
fprintf('||Init');for j=1:size(params,2), fprintf('||% 8f',initParams(j));end; fprintf('||\n');
fprintf('||  LB');for j=1:size(params,2), fprintf('||% 8f',lb(j));end; fprintf('||\n');
fprintf('||  UB');for j=1:size(params,2), fprintf('||% 8f',ub(j));end; fprintf('||\n');
for i = 1:size(regularization,1)
    fprintf('||Reg.');for j=1:size(params,2), fprintf('||% 8f',regularization(i,j));end; fprintf('||\n');
end
fprintf('\nAvg Error\n');
fprintf('||Fold||LTA   (m)||LIN   (m)||Improvement (%%)||\n');
for i = 1:size(params,1)
    fprintf('||%4d',i);
    fprintf('||% f',E(i,1));
    fprintf('||% f',E(i,2));
    fprintf('||% f',(E(i,2)-E(i,1))/E(i,2)*100);
    fprintf('||\n');
end
fprintf('||Avg.||% f||% f||% f||\n',...
    mean(E(:,1)),mean(E(:,2)),mean((E(:,2)-E(:,1))./E(:,2))*100);
