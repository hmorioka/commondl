% Common dictionary learning demo
%
% Version 1.0, July 1 2015
% Author: Hiroshi Morioka
% License: Apache License, Version 2.0
%

clear variables
dbstop if error
set_path

tic;
disp('-------------------- Main Start: demo --------------------');


% Parameters --------------------------------------------
% -------------------------------------------------------

% For artificial data -------------------------

dataParm.imageSize = 4;
dataParm.Nbasis = 32;
dataParm.noiseStd = 0.05;
dataParm.basisSparsity = 0.2;
dataParm.basisNactive = 4;

dataParm.blk(1)     = struct('subject',1 ,'run',1 ,'Ndata',1e5, 'shift',[0,0] ,'blursize',5,'blurstd',0.5);
dataParm.blk(end+1) = struct('subject',1 ,'run',2 ,'Ndata',1e5, 'shift',[0,0] ,'blursize',5,'blurstd',0.5);
dataParm.blk(end+1) = struct('subject',2 ,'run',1 ,'Ndata',1e5, 'shift',[0.3,0.3] ,'blursize',5,'blurstd',0.5);
dataParm.blk(end+1) = struct('subject',2 ,'run',2 ,'Ndata',1e5, 'shift',[0.3,0.3] ,'blursize',5,'blurstd',0.5);
dataParm.blk(end+1) = struct('subject',3 ,'run',1 ,'Ndata',1e5, 'shift',[-0.3,0.3] ,'blursize',5,'blurstd',0.5);
dataParm.blk(end+1) = struct('subject',3 ,'run',2 ,'Ndata',1e5, 'shift',[-0.3,0.3] ,'blursize',5,'blurstd',0.5);
dataParm.blk(end+1) = struct('subject',4 ,'run',1 ,'Ndata',1e5, 'shift',[0.3,-0.3] ,'blursize',5,'blurstd',0.5);
dataParm.blk(end+1) = struct('subject',4 ,'run',2 ,'Ndata',1e5, 'shift',[0.3,-0.3] ,'blursize',5,'blurstd',0.5);
dataParm.blk(end+1) = struct('subject',5 ,'run',1 ,'Ndata',1e5, 'shift',[-0.3,-0.3] ,'blursize',5,'blurstd',0.5);
dataParm.blk(end+1) = struct('subject',5 ,'run',2 ,'Ndata',1e5, 'shift',[-0.3,-0.3] ,'blursize',5,'blurstd',0.5);


% Parameter for DL ----------------------------

% % SPAMS
% parm.method = 'SPAMS';
% parm.K = 32;
% parm.L = 4;
% parm.iter = 1e5;
% parm.batchsize = 512;
% parm.numThreads = 8;
% parm.randSeed = 0;

% Common dictionary learning
parm.method = 'commonDL';
parm.K = 32;
parm.L = 4;
parm.lambda = 1e-7;
parm.iter = 1e5;
parm.batchsize = 512;
parm.initialLearningRate = 5e0; % 
parm.lambda_init = 1e-3;
parm.ratePolicy = '1/10';
parm.iter_init = 1e5;
parm.batchsize_init = 512;
parm.numThreads = 8;
parm.randSeed = 0;


% Make artificial data ----------------------------------
% -------------------------------------------------------
rng(parm.randSeed);

[data, basis, B, source] = generate_artificial_data(dataParm);
% Note: L2-normalization of each data point is necessary for DL
data.x = mexNormalize(data.x);

% Plot spatial bases
showmatgrid(reshape(basis,[dataParm.imageSize,dataParm.imageSize,size(basis,2)]),struct('title','basis (truth)'))
% Plot blurring matrix
showmatgrid(reshape([B.mat],[size(B(1).mat,1),size(B(1).mat,2),length(B)]),struct('title','blurring matrix'))


% Dicitonary learning ----------------------------------
% ------------------------------------------------------
switch parm.method
    % Proposed -----------------------------------------
    % --------------------------------------------------
    case 'commonDL'

        % Common dictionary learning -------------------
        [D,Z,dicParm] = commonDL(data, parm);
        
        % Subject-run-specific sparse coding -----------
        srScParm = struct('L',parm.L);
        if isfield(parm,'numThreads'), srScParm.numThreads = parm.numThreads; end
        % ssScParm.useRunName = [];
        A = runwiseSC(data, D, Z, srScParm); 

        
    % SPAMS --------------------------------------------
    % --------------------------------------------------
    case 'SPAMS'  

        % Dicstionary learning -------------------------
        dlParm = struct('K',parm.K,'lambda',parm.L,'iter',parm.iter,'batchsize',parm.batchsize,'mode',3);
        if isfield(parm,'Dinit'), dlParm.D = parm.Dinit; end
        if isfield(parm,'numThreads'), dlParm.numThreads = parm.numThreads; end
        D = mexTrainDL(data.x(:,randperm(size(data.x,2))), dlParm);

        % Sparse coding --------------------------------
        scParm = struct('L',parm.L);
        if isfield(parm,'numThreads'), scParm.numThreads = parm.numThreads; end
        A = mexOMP(data.x, D, scParm);

end

% find atoms nearest to ground truth of bases -------
% ---------------------------------------------------
distBasisD = pdist2(basis',D','euclidean');
[mindist,nearestIdx] = min(distBasisD,[],2);
Dnearest = D(:,nearestIdx);
distBasisD = pdist2(basis',Dnearest','euclidean');

fprintf('mean error: %f\n', mean(mindist));


% Plot dictonary atoms nearest to bases
showmatgrid(reshape(Dnearest,[dataParm.imageSize,dataParm.imageSize,size(Dnearest,2)]),struct('title','basis (estimated)'))

if strcmp(parm.method,'commonDL')
    % Plot estimated Z
    showmatgrid(reshape([Z.mat],[size(Z(1).mat,1),size(Z(1).mat,2),length(Z)]),struct('title','Z'))   
end


