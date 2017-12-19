clear all;
close all;
clc;
%%
datafolder = '/home/sammirc/Experiments/Nick/AttentionSaccade/behaviour'; %workstation data directory
% datafolder = '/Users/user/Desktop/Experiments/Nick/AttentionSaccade/behaviour'; % laptop data directory
cd(datafolder);

sublist = [1,2,3,4,5,6,7,8,9,10,11,12,13];

parts   = ['a','b'];
nsub = length(sublist);
%%
for isub = 1:length(sublist)
    fprintf('\nWorking on S%02d/%02d.',isub,length(sublist));
    if ismember(sublist(isub), [1,2])
        fname = sprintf('%s/AttSacc_S%02d.mat', datafolder, sublist(isub));
        data = load(fname, 'participant', 'sequence', 'response');
        iblocks = 1:length(data.response);
        resp     = cat(1,data.response(iblocks).resp);         % 0 = no response, 1 = , 2 = 
        time     = cat(1,data.response(iblocks).time);         % NaN on saccade trials, otherwise RT in seconds 
        corr     = cat(1,data.response(iblocks).corr);         % 0 = incorrect, 1 = correct (all saccade trials coded as 0 as no resp!)

        cueloc   = cat(1,data.sequence(iblocks).cuelocation);  % 1:8 - cued location
        targloc  = cat(1,data.sequence(iblocks).targlocation); % 1:8 - location of the target
        validity = cat(1,data.sequence(iblocks).validity);     % 1 = valid, 0 = invalid
        targtilt = cat(1,data.sequence(iblocks).targtilt);     % 1 = , 2 =  
        task     = cat(1,data.sequence(iblocks).task);         % 1 = attention , 2 = saccade
        delay    = cat(1,data.sequence(iblocks).delay);        %
        cuecol   = cat(1,data.sequence(iblocks).cuecolor);     %

        colmap   = data.participant.colmapping;                % colour mapping
        hands    = data.participant.handedness;                % handedness of participant
        gender   = data.participant.gender;                    % gender of participant
        buttass  = data.participant.buttonassignment;          % button assignment for horizontal vs. vertical
        contrast = data.participant.contrast;                  % gabor contrast for participant 

        subject    = zeros(length(resp),1);
        subject(:) = sublist(isub);
        session    = zeros(length(resp),1);
        session(:) = NaN;
        trial     = 1:length(resp); trial = trial';

        dat   = cat(2, subject ,  session ,  trial ,  task ,  cuecol ,  cueloc ,  validity ,  targloc ,  targtilt ,  delay ,  resp ,  time ,  corr);
        names = {     'subject', 'session', 'trial', 'task', 'cuecol', 'cueloc', 'validity', 'targloc', 'targtilt', 'delay', 'resp', 'time', 'corr'};
        write = dataset(subject, session, trial, task, cuecol, cueloc, validity, targloc, targtilt, delay, resp, time, corr, 'VarNames', names);
        
        filename = sprintf('%s/csv/AttSacc_S%02d.csv',datafolder, sublist(isub));
        export(write, 'File', filename, 'delimiter', ',');
        clear data dat write
    else
        for ipart = 1:length(parts)
            fname     = sprintf('%s/AttSacc_S%02d%s.mat', datafolder, sublist(isub), parts(ipart));
            data      = load(fname, 'participant', 'sequence', 'response');
            iblocks   = 1:length(data.response);

            resp     = cat(1,data.response(iblocks).resp);         % 0 = no response, 1 = , 2 = 
            time     = cat(1,data.response(iblocks).time);         % NaN on saccade trials, otherwise RT in seconds 
            corr     = cat(1,data.response(iblocks).corr);         % 0 = incorrect, 1 = correct (all saccade trials coded as 0 as no resp!)

            cueloc   = cat(1,data.sequence(iblocks).cuelocation);  % 1:8 - cued location
            targloc  = cat(1,data.sequence(iblocks).targlocation); % 1:8 - location of the target
            validity = cat(1,data.sequence(iblocks).validity);     % 1 = valid, 0 = invalid
            targtilt = cat(1,data.sequence(iblocks).targtilt);     % 1 = , 2 =  
            task     = cat(1,data.sequence(iblocks).task);         % 1 = attention , 2 = saccade
            delay    = cat(1,data.sequence(iblocks).delay);        %
            cuecol   = cat(1,data.sequence(iblocks).cuecolor);     %

            colmap   = data.participant.colmapping;                % colour mapping
            hands    = data.participant.handedness;                % handedness of participant
            gender   = data.participant.gender;                    % gender of participant
            buttass  = data.participant.buttonassignment;          % button assignment for horizontal vs. vertical
            contrast = data.participant.contrast;                  % gabor contrast for participant 

            subject    = zeros(length(resp),1);
            subject(:) = sublist(isub);
            session    = zeros(length(resp),1);
            session(:) = ipart;
            trial      = 1:length(resp); trial = trial';

            dat   = cat(2, subject ,  session ,  trial,   task ,  cuecol ,  cueloc ,  validity ,  targloc ,  targtilt ,  delay ,  resp ,  time ,  corr);
            names = {     'subject', 'session', 'trial', 'task', 'cuecol', 'cueloc', 'validity', 'targloc', 'targtilt', 'delay', 'resp', 'time', 'corr'};
            write = dataset(subject, session, trial, task, cuecol, cueloc, validity, targloc, targtilt, delay, resp, time, corr, 'VarNames', names);

            filename = sprintf('%s/csv/AttSacc_S%02d%s.csv',datafolder, sublist(isub), parts(ipart));
            export(write, 'File', filename, 'delimiter', ',');
            clear data dat write
        
        end
    end
end