function schedule = schedSampling(Nsample,Niteration,mode)
% schedSampling - pre-assignment of sample numbers to be picked up for 
%                 each iteration.
%
% schedule = schedSampling(Nsample,Niteration,mode)
%
% --- Input ------------------------------------------------------
% Nsample     : number of samples available for SGD
% Niteration  : number of iterations
% mode        : 'random'- randomize for each epoch.
%             : otherwise - no-randomize
%
% --- Output -----------------------------------------------------
% schedule    : sample number to be picked up for each iteration [1 x Niteration]
%
% Version 1.0, July 1 2015
% Author: Hiroshi Morioka
% License: Apache License, Version 2.0
%

if nargin < 3, mode = 'random'; end  

if Nsample <= 0, return; end

Nepoch = ceil(Niteration/Nsample);

schedule = [];
for epoch = 1:Nepoch 
    switch mode
        case 'random'
            schedule_epoch = randperm(Nsample)';
        otherwise
            schedule_epoch = [1:Nsample]';
    end
    schedule = [schedule;
                schedule_epoch];
end
schedule = schedule(1:Niteration);


