function [ seq ] = ucyLoad()
%UCYLOAD load ucy dataset into matlab
%
% T(dataset,time,person,px,py)
% Obj(dataset,px,py)


if ~exist('mmreader','class'), error('mmreader not found'); end

prefix = 'ucy_crowd/data_';
datasets = {'zara01','zara02','students03'};

for did = 1:length(datasets)
    % Read video
    seq.(datasets{did}).video = ...
        mmreader([prefix datasets{did} '/video.avi']);
    
    % Read annotation file
    fid = fopen([prefix datasets{did} '/annotation.vsp'],'r');
    nTr = fscanf(fid,'%d - the number of splines\n',1); % header
    Tr = cell(nTr,1);
    for i = 1:nTr
        nCP = fscanf(fid,'%d - Num of control points\n',1);
        Tr{i} = fscanf(fid,'%f %f %d %f - (2D point, m_id)\n',[4 nCP])';
    end
    fclose(fid);
    
    % Homography
    seq.(datasets{did}).H = load([prefix datasets{did} '/H.txt']);
    
    % Objects
    seq.(datasets{did}).static =...
        load([prefix datasets{did} '/static.txt']);
    
    % Destinations
    seq.(datasets{did}).destinations =...
        load([prefix datasets{did} '/destinations.txt']);
    
    % Decide duration
    t_s = min(cellfun(@(x) min(x(:,3)),Tr));
    t_e = max(cellfun(@(x) max(x(:,3)),Tr));
    frames = (t_s:(seq.(datasets{did}).video.FrameRate/2.5):t_e)';
    
    % Interpolation
    D = cell(size(Tr));
    for i = 1:length(D)
        duration = frames(min(Tr{i}(:,3))<=frames&...
                          max(Tr{i}(:,3))>=frames);
        D{i} = [did*ones(size(duration))...       % dataset id
                (duration+1)...                   % timestamp
                i*ones(size(duration))...         % person id
                interp1(Tr{i}(:,3),Tr{i}(:,1),duration)... % py
                interp1(Tr{i}(:,3),Tr{i}(:,2),duration)... % px
                interp1(Tr{i}(:,3),Tr{i}(:,4),duration)... % gaze angle
                ];
        if any(isnan(D{i}(:,4:5))), keyboard; end
    end
    
    % Remove buggy annotation
%     ind = ...
%         cellfun(@(x) all(x(:,4)>-0.5*seq.(datasets{did}).video.Width) &...
%                      all(x(:,5)>-0.5*seq.(datasets{did}).video.Height)&...
%                      all(x(:,4)< 0.5*seq.(datasets{did}).video.Width) &...
%                      all(x(:,5)< 0.5*seq.(datasets{did}).video.Height),...
%                      D);
%     D = D(ind);
    
    % Reshape
    D = cat(1,D{:});
    
    % Shift image coordinates
    D(:,4) = D(:,4) + seq.(datasets{did}).video.Width/2;
    D(:,5) = -D(:,5) + seq.(datasets{did}).video.Height/2;
    if strcmp(datasets{did},'zara02')
        D(:,5) = D(:,5) + 50; % Move annotation from head to foot
    end
    % Homography transform
    P = [D(:,4:5) ones(size(D,1),1)] * seq.(datasets{did}).H';
    D(:,4:5) = [P(:,1)./P(:,3) P(:,2)./P(:,3)];
    
    % Store
    seq.(datasets{did}).obsmat = D;
end

end