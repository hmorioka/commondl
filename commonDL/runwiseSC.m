function [A, e, r] = runwiseSC(data, D, Z, parm)
% runwiseSC - run-wise sparse coding
%
% [A, e, r] = runwiseSC(data, D, Z, parm)
%
% Perform run-wise sparse coding from given multi-subject-run signals, 
% common dictionary D, and subject-run-specific transforms Z. 
% This function also calculates reconstruction error and reguralization term.
%
% --- Input ------------------------------------------------------
% data [struct]
%   .x      : multichannel data temporaly concatenated across
%             subjects and runs [Nchannel x Ntime]
%             (each column must be l2-noramlized)
%   .s      : subject label of each time point [1 x Ntime]
%   .r      : run label of each time point [1 x Ntime]
% D         : common dictionary [Nchannel x K]
% Z [structure array; 1 x (Nsubject x Nrun)]
%   .mat    : subject-run-specific transform [Nchannel x Nchannel]
%   .s      : subject label
%   .r      : run label
% parm [struct]
%   .L      : sparsness constraint
%   .useRunName : (optional) specify a run to be used
%
% --- Output -----------------------------------------------------
% A         : sparse code [K x Ntime]
% e         : reconstruction error
% r         : reguralization term
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

if isfield(parm,'L'), L = parm.L; else error('parm.L is necessary'); end
if isfield(parm,'numThreads'), numThreads = parm.numThreads; else numThreads = []; end
if isfield(parm,'useRunName'), useRunName = parm.useRunName; else useRunName = []; end

% Main procedure ------------------------------------------
% ---------------------------------------------------------
K = size(D,2);
[Nch, Nt] = size(data.x);

if max(abs(1-sqrt(sum(data.x.^2,1)))) > 1e-9
    error('data.x must be l2-normalized for each column.');
end

% parameters for SPAMS
scParm = struct('L',L);
if ~isempty(numThreads), scParm.numThreads = numThreads; end

A = zeros(K, Nt); e = 0; r = 0;
fprintf('Calculating A... : '); starttime = toc;
dataSubjList = unique(data.s);
for sn = 1:length(dataSubjList)
    fprintf('%d.',dataSubjList(sn));
    subjName = dataSubjList(sn);
    dataRunList = unique(data.r(data.s==subjName));
    for rn = 1:length(dataRunList)
        runName = dataRunList(rn);
        snrnIdx = (data.s==subjName & data.r==runName);
        Xijt = data.x(:,snrnIdx);

        % extract Z of (sn,rn) ------------------------------
        if ~isempty(useRunName)
            loadRunName = useRunName;
        else
            loadRunName = runName;
        end
        zIdx = find([Z.s]==subjName & [Z.r]==loadRunName);
        
        if ~isempty(zIdx)
            Zij = Z(zIdx).mat;
        else
            error('Information of subj:%d run:%d is not in Z',subjName,loadRunName)
        end
        
        % sprse coding ------------------
        ZijD = mexCalcXY(Zij,D); % transformed dictionary
        l2norm = sqrt(sum(ZijD.^2,1)); % L2-norm of ZijD
        ZijDnorm = bsxfun(@rdivide,ZijD,l2norm); % normalized-ZijD (L2=1 for each atom) 
        Aijtw = mexOMP(Xijt,ZijDnorm,scParm); % OMP with normalized-ZijD
        Aijt = bsxfun(@rdivide,Aijtw,l2norm'); % Re-weight Atw by the norm of ZijD
        A(:,snrnIdx) = full(Aijt);
        
        % Reconstruction error -------------------------------
        Eijt = Xijt - ZijD*Aijt;
        e = e + sum(Eijt(:).^2); 
        
        % Regularizatin term ---------------------------------
        Zij_I = Zij - eye(Nch);
        r = r + size(Xijt,2)*sum(Zij_I(:).^2); % summation across data points
    end
end
fprintf('done!; time=%d[s]\n',floor(toc-starttime));

end

