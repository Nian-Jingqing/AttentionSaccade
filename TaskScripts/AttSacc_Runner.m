clear all
close all
clc

%studypath = '/Work/kia/Experiments/AttSacc';
studypath = 'C:\Experiments\Nick\AttSacc';
cd(studypath)
fs        = filesep;
addpath([studypath fs 'Toolbox' fs]);
addpath([studypath fs 'Toolbox' fs 'CircStat' fs]);

randstate = sum(100*clock);%
%comment these out in newer matlab versions
rand('twister',randstate);
randn('state',randstate);

argindlg = inputdlg({'Identifier (S##)' 'Gender (M/F)' 'Age (y)' 'Handedness (L/R)'},'',1); 
if isempty(argindlg)
    error('Experiment cancelled.');
end
participant = [];
participant.identifier = upper(argindlg{1});
participant.gender     = argindlg{2};
participant.age        = argindlg{3};
participant.handedness = argindlg{4};
participant.randstate  = randstate;

isubject = sscanf(participant.identifier,'S%d');
if ~isempty(isubject) && isubject > 0
    participant.colmapping   = mod(isubject-1,2)+1;
else
    error('Invalid identifier.');
end
participant.filename = sprintf('AttSacc_%s_%s',participant.identifier,datestr(now,'yyyymmdd-HHMM'));

% set to true for practice session!
ispractice = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
if ispractice
    practice                    = AttSacc_BuildExperiment(3,true);
    [response, practice, video] = AttSacc(participant,practice,false,false,true);
    clc
end
sequence = AttSacc_BuildExperiment(16,false);
%%
participant.filename = sprintf('AttSacc_%s_%s',participant.identifier,datestr(now,'yyyymmdd-HHMM'));
iseyelinked      = 1;
sendtriggers     = 1;
[response, sequence, video, eyetracking] = AttSacc(participant,sequence,iseyelinked,sendtriggers);
save(['./Data/',participant.filename,'_experiment.mat'],'participant','sequence','response','video','eyetracking');