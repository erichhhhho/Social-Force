%% main2.m
% EWAP implementation

% Load and preprocess dataset
if ~exist('ewap_dataset.mat','file')
    seq = ewapLoad('ewap_dataset');
    save ewap_dataset.mat seq;
end
load ewap_dataset.mat;
[D,Others] = seq2ewap(seq);

%% Cross validation
nfolds = 5;
% initParams = [0.130081 2.087902 2.327072 2.0732 1.461249 0.7304224];
initParams = [ 0.010883 2.646596 0.069719 2.496652 2.531520 0.353709];

if exist('matlabpool','file')==2, parpool open 2; end
params = zeros(nfolds,size(initParams,2));
E = zeros(nfolds,2); % Array to store RMSE
tic;
for expId = 1:nfolds
    % Prepare index of testing samples
    ind = mod((1:size(D,1))+expId-1,nfolds) == 0;
%     ind = ceil(nfolds*(1:size(D,1))/(size(D,1))) == expId;
    
    % Training = !Testing index
%     params(expId,:) =...
%         ga(@(q) sum(sum(ewapError(D(~ind,:),Others(~ind),q).^2,1),2),...
%             6,...
%             [],[],[],[],...
%             zeros(6,1),...                    % Lower bound
%             [inf inf inf inf inf 1.0],...   % Upper bound
%             [],...
%             gaoptimset( 'Display','iter',...
%             'CreationFcn',@gacreationlinearfeasible,...
%             'PopInitRange',[(1-0.2)*initParams;(1+0.2)*initParams],...
%             'PopulationSize',10,...
%             'StallGenLimit',10,...
%             'TimeLimit',60*60*2,...             % 2 hours
%             'Generations',40,...
%             'UseParallel','always')...
%             );
    params(expId,:) =...
        lsqnonlin(@(x) ewapError2(D(~ind,:),Others(~ind),x),...
            initParams,...
            [0 0 0 2 2 0],...                    % Lower bound
            [inf inf inf inf inf 1.0],...   % Upper bound
            optimset('Display','iter',...
            'Diagnostics','on',...
            'MaxIter',30,...
            'Jacobian','off')...
            );
    % Test
    E(expId,1) = sqrt(sum(...
                sum(ewapError(D(ind,:),Others(ind),params(expId,:)).^2,2)...
                )/nnz(ind));
    % Error in linear model
    E(expId,2) = sqrt(sum(...
                sum((D(ind,[8 9])-D(ind,[4 5])-0.4*D(ind,[6 7])).^2,2)...
                )/nnz(ind));
    % Print
    fprintf('\nParams\n');
    fprintf('||  sigma_d||  sigma_w||     beta|| lambda_1|| lambda_2||    alpha||\n');
    for j=1:size(params,2)
        fprintf('||% 8f',params(expId,j));
    end
    fprintf('||\n');
    fprintf('\nError\n');
    fprintf('||LTA RMSE (m)||LIN RMSE (m)||Improvement (%%)||\n');
    fprintf('||% f',E(expId,1));
    fprintf('||% f',E(expId,2));
    fprintf('||% f',(E(expId,2)-E(expId,1))/E(expId,2)*100);
    fprintf('||\n');
    
    save ewap_mod_results
end
toc;
clear ind expId;
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

fprintf('\nRMS Error\n');
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
