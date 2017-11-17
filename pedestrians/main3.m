%% main3.m
% EWAP modification
% TESTING SCRIPT TO EVALUATE GRADIENT DESCENT FOR PARAMETER LEARNING

% Load and preprocess dataset
if ~exist('ewap_dataset.mat','file')
    seq = ewapLoad('ewap_dataset');
    save ewap_dataset.mat seq;
end
load ewap_dataset.mat;
[D,Obj] = seq2ewap(seq);

%% Cross validation
nfolds = 3;
initParams = [0.130065 2.088003 2.085519 2.073509 1.461433 0.729837];
% initParams = [0 0 0 0 0 1];
% initParams = [ 0.010883 2.646596 0.069719 2.496652 2.531520 0.353709];
% initParams = [0.117811 3.680120 1.867231 2.496652 2.531520 0.426200];
% initParams = [0.016899 0.986243 0.085706 1.164646 0.723846 0.423630];
% initParams = [0.143719 2.051546 2.719708 1.666738 1.312681 0.588800];
% initParams = [0.202575 0.764088 4.390793 1.127075 4.082286 0.340538];
lb = [0 0 0 0 0 0];
ub = [inf inf inf inf inf 1.0];
regularization = [];

Dind = unique(D(:,[1 3]),'rows');   % Unique pairs of (dataset,person_id)

if exist('matlabpool','file')==2, parpool open 3; end
params = zeros(nfolds,size(initParams,2));
E = zeros(nfolds,1); % Array to store error
for expId = 1:nfolds
    tic;
    % Prepare index of training/testing samples
    Train = Dind(mod(1:size(Dind,1),nfolds)==expId-1,:);
    Test  = Dind(mod(1:size(Dind,1),nfolds)~=expId-1,:);
    
    % Training = !Testing index
    params(expId,:) =...
        ga(@(x) mean(ewapError3(D,Obj,x,'Index',Train)),...
            6,[],[],[],[],lb,ub,[],...
            gaoptimset( 'Display','iter',...
                'CreationFcn',@gacreationlinearfeasible,...
                'PopInitRange',[(1-0.5)*initParams;(1+0.5)*initParams],...
                'PopulationSize',10,...
                'StallGenLimit',10,...
                'Generations',30,...
                'UseParallel','always')...
            );
%     params(expId,:) =...
%         lsqnonlin(@(x) (ewapError3(D,Obj,x,'Index',Train)),...
%             initParams,lb,ub,...
%             optimset('Display','iter',...
%                 'GradObj','off',...
%                 'MaxIter',30)...
%             );
    % Test in N-step prediction
    E(expId,1) = mean(ewapError3(D,Obj,params(expId,:),'Index',Test));
    % Error in N-step prediction of linear model
    E(expId,2) = mean(linError(D,'Index',Test));
    % Print
    fprintf('\nParams\n');
    fprintf('||  sigma_d||  sigma_w||     beta|| lambda_1|| lambda_2||    alpha||\n');
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
fprintf('||Fold||  sigma_d||  sigma_w||     beta|| lambda_1|| lambda_2||    alpha||\n');

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

fprintf('\nAvg. Error\n');
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
