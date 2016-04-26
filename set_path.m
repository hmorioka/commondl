
% SPAMS --------------------
dir_spams = './spams-matlab';
addpath(fullfile(dir_spams, 'test_release'));
addpath(fullfile(dir_spams, 'src_release'));
addpath(fullfile(dir_spams, 'build'));
setenv('MKL_NUM_THREADS','1')
setenv('MKL_SERIAL','YES')
setenv('MKL_DYNAMIC','NO')

% commonDictLearn ----------
addpath(fullfile(pwd, 'commonDL'));

