%% Main script for parameter learning

% Set up
addpath('libsvm-mat-3.0-1/');
% if ~exist('dataset.mat','file')
    [D,T,Obj] = importData();
%     save dataset;
% end
% load dataset;

methods = {'LINE','ORIG','ATTR'};
params = {...
    [],...
    [1 1 1 1 1 0.7],...
    [1 1 1 1 1 1 1 1],...
...[1.085866 0.997198 0.864808 0.812508 1.218730 0.852900],...
...[0.753824 0.694030 1.320429 0.796903 0.317338 0.071199 2.727593 0.776883],...
...[1.034086 1.068818 0.822700 0.940671 1.005634 0.842588],...12.570702
...[0.780582 0.619282 1.303413 1.241795 0.223758 0.063400 2.145226 1.519417],...16.408654
... %%% Model changed above %%%
...[0.971591 1.143258 0.857114 0.908489 0.911492 0.894517],...15.765998 <- sanity check
...[1.155051 0.920003 1.038165 0.956342 0.187045 0.026875 1.425865 0.232656 1.519417],...18.690060 <- sanity check
...[0.106600 1.736115 2.903942 3.885594 3.658966 0.848382],...%8.421595%
...[0.794061 0.493533 2.310295 1.061027 0.760730 0.022921 3.852594 0.178507 1.631713],...%19.441777% (+SVM:20.805826)
...[0.169284 1.721694 2.796398 3.353471 3.509591 0.853483],...%9.244544%
...[0.803139 0.461904 2.348088 1.050391 0.752578 0.022120 3.257493 0.180731 2.049471],...%20.438035% (+SVM:21.682020)
...[0.1724 1.6979 2.6764 3.3017 3.4169 0.9086],...%14.667996%
...[0.8249 0.4544 2.0320 1.3348 0.6790 0.0224 3.2771 0.1701 2.0011],...%19.524906% (20.660792% with SVM)
...%     [0.711504 2.285561 2.724817 2.234390 2.189269 0.700000]...
...%     [0.939106 1.031759 2.064315 1.694289 1.165338 0.057395 4.156209 0.211012 1.364862]...
...%[0.813039 0.762212 0.751723 1.562003 1.173357 0.851584],...%10.114519%
...%[1.035776 0.902440 1.396630 0.636877 0.654448 0.100625 2.056375 0.201875 1.670276]...%18.904044
...[1 1 1 1 1 0.7],...
...[1 1 1 1 1 0.1 1 0.2 1],...
...%[0.784724 0.832599 0.125332 1.777328 1.343519 0.844591],...%9.676653%
...%[0.789494 0.874236 1.384411 0.637523 0.496495 0.100625 1.287123 0.201875 1.010875]...% 16.654239%
...%[0.225063 4.107420 4.051493 4.027708 0.798678 0.764522],...% 9.065255%
...%[1.046042 0.967508 1.659289 1.345678 1.341796 0.057574 6.281420 0.221592 2.627939]... % 18.863414%
...%[0.211504 3.285561 3.724817 4.234390 2.189269 0.839546]...
...%[0.939106 0.631759 3.064315 1.694289 1.165338 0.057395 4.156209 0.211012 1.364862]...
};
lb = {[],[0 0 0 0 0 0],[0 0 0 0 0 0 0 0]};
ub = {[],[inf inf inf inf inf 1.0],[inf inf inf inf inf inf inf inf]};

Ninterval = 4;

for itr = 1:10
    fprintf('== Itr %d ==\n\n', itr);
    % Decide training set pick from 1,2
    Train = unique(T(T(:,1)==1|T(:,1)==2,[1 3]),'rows');
    Train = Train(mod(1:size(Train,1),Ninterval)'==mod(itr,Ninterval),:); % Subsample
    % Train = [];
    Test = unique(T(T(:,1)==3,[1 3]),'rows');
    % Train = Test;

    %% Solve for the optimal parameters

    if exist('matlabpool','file')==2, matlabpool open; end
    if ~isempty(Train)
        % ORG
        fprintf('Learning original model\n'); tic;
        % params{2} = ...
        %     ga(@(x) mean(ewapError3(T,Obj,x,'Index',Train)),...
        %         6,[],[],[],[],lb{2},ub{2},[],...
        %         gaoptimset( 'Display','iter',...
        %             'CreationFcn',@gacreationlinearfeasible,...
        %             'PopInitRange',[(1-0.4)*params{2};(1+0.4)*params{2}],...
        %             'PopulationSize',10,...
        %             'StallGenLimit',10,...
        %             'Generations',30,...
        %             'UseParallel','always',...
        %             'HybridFcn',{@fmincon,optimset('display','iter',...
        %                                            'MaxIter',10,...,...
        %                                            'GradObj','off')})...
        %         );
%         params{2} = lsqnonlin(@(x) ewapError3(T,Obj,x,'Index',Train),...
%                         params{2},lb{2},ub{2},...
%                         optimset('Display','iter',...
%                                 'GradObj','off',...
%                                 'MaxIter',30)...
%                         );
        params{2} = fmincon(@(x) mean(ewapError3(T,Obj,x,'Index',Train)),...
                        params{2},[],[],[],[],lb{2},ub{2},[],...
                        optimset('Display','iter',...
                                'GradObj','off',...
                                'Algorithm','active-set',...
                                'MaxIter',30)...
                        );        
%         params{2} = fminsearch(@(x) mean(ewapError3(T,Obj,x,'Index',Train)),...
%                         params{2},...
%                         optimset('Display','iter',...
%                                 'GradObj','off',...
%                                 'Algorithm','active-set',...
%                                 'MaxIter',30)...
%                         );
        disp(params{2}); toc;
        % ATR
        fprintf('Learning attraction model\n'); tic;
        % params{3}([1:5 7 9]) =...
        %     ga(@(x) mean(myError3(T,Obj,...
        %                 [x(1:5) params{3}(6) x(6) params{3}(8) x(7)],...
        %                 'Index',Train)),...
        %         7,[],[],[],[],lb{3}([1:5 7 9]),ub{3}([1:5 7 9]),[],...
        %         gaoptimset('Display','iter',...
        %             'CreationFcn',@gacreationlinearfeasible,...
        %             'PopInitRange',[(1-0.4)*params{3}([1:5 7 9]);...
        %                             (1+0.4)*params{3}([1:5 7 9])],...
        %             'PopulationSize',10,...
        %             'StallGenLimit',10,...
        %             'Generations',30,...
        %             'UseParallel','always',...
        %             'HybridFcn',{@fmincon,optimset('display','iter',...
        %                                            'MaxIter',10,...
        %                                            'GradObj','off')})...
        %         );
%         params{3} =...
%             lsqnonlin(@(x) myError3(T,Obj,x,...
%                         'Index',Train),...
%                 params{3},lb{3},ub{3},...
%                 optimset('Display','iter',...
%                          'GradObj','off',...
%                          'MaxIter',30)...
%                 );
        params{3} =...
            fmincon(@(x) mean(myError3(T,Obj,x,'Index',Train)),...
                params{3},...
                [],[],[],[],lb{3},ub{3},[],...
                optimset('Display','iter',...
                    'GradObj','off',...
                    'MaxIter',30)...
                );
%         params{3} =...
%             fminsearch(@(x) mean(myError3(T,Obj,x,'Index',Train)),...
%                 params{3},...
%                 optimset('Display','iter',...
%                     'GradObj','off',...
%                     'Algorithm','active-set',...
%                     'MaxIter',30)...
%                 );
        disp(params{3}); toc;
    end

    %% Compute error in prediction
    tic;
    E = zeros(1,length(methods)); % Array to store error
    E(1) = mean(linError(T,'Index',Test));
    E(2) = mean(ewapError3(T,Obj,params{2},'Index',Test));
    E(3) = mean(myError3(T,Obj,params{3},'Index',Test));
    toc;
    if exist('matlabpool','file')==2, matlabpool close; end

    %% Display results

    fprintf('\nParams\n');
    for i = 1:length(methods)
        fprintf(' %s:',methods{i});
        for j = 1:length(params{i})
            fprintf('% f',params{i}(j));
        end
        fprintf('\n');
    end

    fprintf('\n||Methods        ||');
    for i = 1:length(methods),fprintf('     %s||',methods{i}); end
    fprintf('\n');
    fprintf('||Avg Error   (m)||');
    for i = 1:length(methods),fprintf('% f||',E(i)); end
    fprintf('\n');
    fprintf('||Improvement (%%)||');
    for i = 1:length(methods),fprintf('% 2.6f||',(E(1)-E(i))/E(1)); end
    fprintf('\n\n');
end

save main6result.mat