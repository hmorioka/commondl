function [D,Z,dictParm] = commonDL(data, parm)
% commonDL - main algorithm of the common dictionary learning.
%
% [D,Z,outParm] = commonDL(data, parm)
%
% Perform common dictionary learning from given multi-subject-run signals. 
% Output an estimate of the common dictionary D and subject-run-specific
% transforms Z. 
%
% --- Input ------------------------------------------------------
% data [struct]
%   .x              : multichannel signals temporaly concatenated across
%                     subjects and runs [Nchannel x Ntime]
%                     (each column must be l2-noramlized)
%   .s              : subject label of each time point [1 x Ntime]
%   .r              : run label of each time point [1 x Ntime]
% parm [struct]
%   (model parameters)
%   .K              : dictionary size
%   .L              : sparsness constraint
%   .lambda         : regularization constant (||Z-I||_F)
%   (SGD paramerers)
%   .iter           : number of iterations of SGD optimization (default: 1e5)
%   .batchsize      : mini-batch size of SGD optimization (default: 512)
%   .initialLearningRate : initial learning rate (default: 5)
%   .iterobj        : duration for calculating intermediate objective function 
%                     (default: .iter*0.05)
%   .ratePolicy     : learning rate plicy ['1/10' | 'constant'] (default: '1/10')
%   (parameters for initializing D and Z)
%   .lambda_init    : regularization constant of ridge-regression for initializing Z
%   .Dinit          : (optional) initial value of D. If this is not provided, 
%                     SPAMS toolbox is used for the initialization of D.
%   .Dinitinit      : (for init. by SPAMS) (optional) Initial value of Dinit 
%                     for initialization by SPAMS. If this is not provided, 
%                     the SPAMS toolbox initializes the dictionary with random 
%                     elements from the data.x.
%   .iter_init      : (for init. by SPAMS) number of iterations for initializing D
%   .batchsize_init : (for init. by SPAMS) mini-batch size for initializing D (default: 512)
%   (optional parameters)
%   .numThreads     : (for SPAMS toolbox) number of threads for exploiting 
%                     multi-core / multi-cpus. 
%   .randSeed       : random seed
%   .doplot         : plot objective function or not [true | false] (default: false)
%
% --- Output -----------------------------------------------------
% D                 : common dictionary [Nchannel x K]
% Z [structure array; 1 x (Nsubject x Nrun)]
%   .mat            : subject-run-specific transform [Nchannel x Nchannel]
%   .s              : subject name
%   .r              : run name
% dictParm [struct]
%   .stats          : structure array of statistics during SGD
%   .Dinit          : Initial value of D
%   .Zinit          : Initial value of Z
%
% Note: This function depends on SPAMS toolbox, and requires that to be added to matlab path.
%       (J. Mairal et al., http://spams-devel.gforge.inria.fr/)
%     : Each column of data.x must be L2-normalized.
%
% Source: Morioka, H., Kanemura, A., Hirayama, J., Shikauchi, M., 
%   Ogawa, T., Ikeda, S., Kawanabe, M., Ishii, S. Learning a common 
%   dictionary for subject-transfer decoding with resting calibration. 
%   NeuroImage, vol.111, pages167-178, 2015.
%
% Version 1.0, July 1 2015
% Author: Hiroshi Morioka
% License: Apache License, Version 2.0
%

% (model parameters)
if isfield(parm,'K'), K = parm.K; else K = size(data,1)*2; end
if isfield(parm,'L'), L = parm.L; else L = round(size(data,1)/3); end
if isfield(parm,'lambda'), lambda = parm.lambda; else lambda = 1e-7; end
% (for SGD)
if isfield(parm,'iter'), Niteration = parm.iter; else Niteration = 1e5; end
if isfield(parm,'batchsize'), Nbatch = parm.batchsize; else Nbatch = 512; end
if isfield(parm,'initialLearningRate'), initialLearningRate = parm.initialLearningRate; else initialLearningRate = 5e0; end
if isfield(parm,'iterobj'), Niterobj = parm.iterobj; else Niterobj = round(Niteration*0.05); end
if isfield(parm,'ratePolicy'), ratePolicy = parm.ratePolicy; else ratePolicy = '1/10'; end
% (for initialization)
if isfield(parm,'Dinit'), Dinit = parm.Dinit; else Dinit = []; end
if isfield(parm,'Dinitinit'), Dinitinit = parm.Dinitinit; else Dinitinit = []; end
if isfield(parm,'lambda_init'), lambda_init = parm.lambda_init; else lambda_init = 1e-3; end 
if isfield(parm,'iter_init'), iter_init = parm.iter_init; else iter_init = 1e4; end
if isfield(parm,'batchsize_init'), batchsize_init = parm.batchsize_init; else batchsize_init = 512; end
% (option)
if isfield(parm,'numThreads'), numThreads = parm.numThreads; else numThreads = []; end
if isfield(parm,'randSeed'), randSeed = parm.randSeed; else randSeed = []; end
if isfield(parm,'doplot'), doplot = parm.doplot; else doplot = false; end


% --------------------------------------------------------
% Execute main procedure
% --------------------------------------------------------

algoStartTime = toc;
if ~isempty(randSeed), rng(randSeed); end

subjList = unique(data.s);
Nsubj = length(subjList);
Nch = size(data.x,1);
Nallsession4Dupdate = 0;
for sn = 1:Nsubj
    rList = unique(data.r(data.s==subjList(sn)));
    Nallsession4Dupdate = Nallsession4Dupdate + length(rList);
end

if max(abs(1-sqrt(sum(data.x.^2,1)))) > 1e-9
    error('data.x must be l2-normalized for each column.');
end

% Scheduling for gradient decent -------------------------
% --------------------------------------------------------
fprintf('Initializing pickup-table... : '); starttime = toc;

Nmaxrun = length(unique(data.r));
NMaxUsedInOneRun = ceil(Niteration/Nsubj/Nmaxrun)*Nbatch;
for sn = 1:Nsubj
    fprintf('%d.',subjList(sn));
    runList = unique(data.r(data.s==subjList(sn)));
    for rn = 1:length(runList)
        snrnDataIdxs = find(data.s==subjList(sn) & data.r==runList(rn));
        Nsample = length(snrnDataIdxs);
        pickupTable{sn}.table{rn} = schedSampling(Nsample, NMaxUsedInOneRun, 'Random');
        pickupTable{sn}.lastLoad(rn) = 0;
        pickupTable{sn}.runName(rn) = runList(rn);
        pickupTable{sn}.lastLoadRun = 0;
        pickupTable{sn}.dataIdx{rn} = snrnDataIdxs;
    end
end
fprintf('done!; time=%d[s]\n',floor(toc-starttime));


% Initialize the dictionary by SPAMS ---------------------
% --------------------------------------------------------
if isempty(Dinit)
    fprintf('Initializing D... : '); starttime = toc;
    
    dlParm = struct('K',K,'lambda',L,'iter',iter_init,'batchsize',batchsize_init,'mode',3);
    if ~isempty(numThreads), dlParm.numThreads = numThreads; end
    if ~isempty(Dinitinit), dlParm.D = Dinitinit; end
    Dinit = mexTrainDL(data.x(:,randperm(size(data.x,2))), dlParm);
    
    fprintf('done!; time=%d[s]\n',floor(toc-starttime));
end
D = Dinit;


% Sparse coding of input data (OMP) ----------------------
% --------------------------------------------------------
fprintf('Initializing A... : '); starttime = toc;

scParm = struct('L',L);
if ~isempty(numThreads), scParm.numThreads = numThreads; end
A = mexOMP(data.x,D,scParm);
A = full(A);

fprintf('done!; time=%d[s]\n',floor(toc-starttime));


% Reconstruction error -----------------------------------
% --------------------------------------------------------
e = data.x - D*A;
e = sum(e(:).^2);
statsInit = struct('obj',e,'recon',e,'reg',0);


% Initialize Z by ridge-regression -----------------------
% --------------------------------------------------------
fprintf('Initializing Z... : '); starttime = toc;

cnt = 1;
for sn = 1:Nsubj
    fprintf('%d.',subjList(sn));
    runList = unique(data.r(data.s==subjList(sn)));
    for rn = 1:length(runList)
        snrnIdx = (data.s==subjList(sn) & data.r==runList(rn));
        Xijt = data.x(:,snrnIdx);
        DA = D*A(:,snrnIdx);
        leftTerm = DA*DA' + lambda_init*size(Xijt,2)*eye(Nch);
        rightTerm = DA*Xijt';
        Ztf = leftTerm\rightTerm;
        Z(cnt) = struct('mat',Ztf','s',subjList(sn),'r',runList(rn)); 
        cnt = cnt + 1;
    end
end
fprintf('done!; time=%d[s]\n',floor(toc-starttime));


% Normalize Z to be close to I ----------------------------
% ---------------------------------------------------------
fprintf('Normalizing Z... : '); starttime = toc;

zNormCoeffs = zeros(1,length(Z));
for i = 1:length(Z)
    Zij = Z(i).mat;
    if max(abs(Zij(:))) ~= 0
        y = eye(Nch); y = y(:);
        x = Zij; x = x(:);
        a = (x'*x)^-1*x'*y;
        Zij = Zij.*a;
        zNormCoeffs(i) = a;
    end
    Z(i).mat = Zij;
end
Zinit = Z;
fprintf('done!; time=%d[s]\n',floor(toc-starttime));


% Calculate objective function ---------------------------
% --------------------------------------------------------
srScParm = struct('L',L);
if ~isempty(numThreads), srScParm.numThreads = numThreads; end
[~, e, r] = runwiseSC(data, D, Z, srScParm); 
r = lambda*r; % reguralization term
f = e + r; % objective function

clear A

% --------------------------------------------------------------------
% Gradient descent ---------------------------------------------------
% --------------------------------------------------------------------
iterStartTime = toc;
stats(1) = struct('iter',0,'time',0,'obj',f,'recon',e,'reg',r);

for iter = 1:Niteration
    
    fprintf('iter: %5.0d, %5.0f[s]',iter,toc-iterStartTime);
    
    % Draw a subject -----------------------------------------
    % --------------------------------------------------------
    subjIdx = mod(iter,Nsubj);
    if subjIdx==0, subjIdx = Nsubj; end
    subjName = subjList(subjIdx);

    
    % Draw a run from the subject ----------------------------
    % --------------------------------------------------------
    runIdx = pickupTable{subjIdx}.lastLoadRun + 1;
    if runIdx > Nmaxrun, runIdx = 1; end
    pickupTable{subjIdx}.lastLoadRun = runIdx;
    

    % Draw data from the (subject, run) ----------------------
    % --------------------------------------------------------
    if runIdx <= length(pickupTable{subjIdx}.runName)
        runName = pickupTable{subjIdx}.runName(runIdx);
        snrnTable = pickupTable{subjIdx}.table{runIdx};
        snrnLastLoadIdx = pickupTable{subjIdx}.lastLoad(runIdx);
        snrnDataIdxs = pickupTable{subjIdx}.dataIdx{runIdx};
        snrnDrawIdx = snrnTable(snrnLastLoadIdx+1 : snrnLastLoadIdx+Nbatch);
        Xijt = data.x(:,snrnDataIdxs(snrnDrawIdx));
        zidx = [Z.s]==subjName & [Z.r]==runName;
        Zij = Z(zidx).mat;
        pickupTable{subjIdx}.lastLoad(runIdx) = snrnLastLoadIdx + Nbatch;
    else
        runName = -1;
        Xijt = [];
    end
    fprintf(', sub%2.0d, run%2.0d',subjName,runName);
    
    
    % Choose the learning rate -------------------------------
    % --------------------------------------------------------
    if subjIdx==1 && runIdx==1 % update the rate only when all subject were updated
        switch ratePolicy
            case '1/10'
                t0 = Niteration/10;
                if (iter <= t0), learning_rate = initialLearningRate;
                else learning_rate = initialLearningRate*t0/iter; 
                end
            case 'constant'
                learning_rate = initialLearningRate;
        end
    end

    
    % Calculate dLu ------------------------------------------
    % --------------------------------------------------------
    if ~isempty(Xijt)
        % sprse coding ------------------
        ZijD = mexCalcXY(Zij,D); % transformed dictionary
        l2norm = sqrt(sum(ZijD.^2,1)); % L2-norm of ZijD
        ZijDnorm = bsxfun(@rdivide,ZijD,l2norm); % normalized-ZijD (L2=1 for each atom) 
        Aijtw = mexOMP(Xijt,ZijDnorm,scParm); % OMP with normalized-ZijD
        Aijt = bsxfun(@rdivide,Aijtw,l2norm'); % Re-weight Atw by the norm of ZijD

        X_ZDA = Xijt - ZijD*Aijt; 
        DA = D*Aijt;
        % dlu_dD ------------------------
        dlu_dD = -Zij'*X_ZDA*Aijt'; 
        % dlu_dZ ------------------------
        dlu_dZ = -mexCalcXYt(X_ZDA,DA) + lambda*(Zij-eye(Nch))*Nbatch; 
    else
        dlu_dD = [];
        dlu_dZ = [];
    end
    dD = dlu_dD / Nbatch; 
    dZ = dlu_dZ / Nbatch; 
    
 
    % Update parameters --------------------------------------
    % --------------------------------------------------------
    if ~isempty(dD) 
        D = D - (learning_rate/Nallsession4Dupdate)*dD; % modify learning_rate for selection bias between D and Z
        D = mexNormalize(D); % normalization (orthogonal projection)
    end
    if ~isempty(dZ) 
        Z(zidx).mat = Z(zidx).mat - learning_rate*dZ;
    end
    
    
    % Calculate objective function ---------------------------
    % --------------------------------------------------------
    if mod(iter,Niterobj) == 0
        fprintf('\nCalculating reconstruction error... : '); 
        [~, e, r] = runwiseSC(data, D, Z, srScParm); 
        r = lambda*r;
        f = e+r;
        stats(end+1) = struct('iter',iter,'time',toc-iterStartTime,'obj',f,'recon',e,'reg',r);
    end
    fprintf(', obj% .5e\n',f);    
    
end

fprintf('\nOptimization finish: %f\n',toc-algoStartTime);

% output parameters -----------------------
dictParm.stats = stats;
dictParm.statsSpams = statsInit;
dictParm.Dinit = Dinit;
dictParm.Zinit = Zinit;

% plot objective funtion ------------------
if doplot
    fontsize = 14;
    figure; hold on;
    plot([stats.iter],[stats.obj],'Color',[0, 114, 178]./255,'LineWidth',3)
    set(gca,'FontSize',fontsize);
    xlabel('Iteration','FontSize',fontsize);
    ylabel('Objective function','FontSize',fontsize);
end

end


