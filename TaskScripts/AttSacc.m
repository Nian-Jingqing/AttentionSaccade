function [response,sequence,video,eyetracking] = AttSacc(participant,sequence,iseyelinked,sendtriggers,practice)

if nargin < 5,
    practice = false;
end
if nargin < 4
    sendtriggers = false;
end
if nargin < 3
    iseyelinked = false;
end
if nargin < 2
    error('Missing input argument(s).');
end

recalibrate = true;
mainlab     = 1;
booth4      = false;
apple       = 0;
fast        = false;

gamma = 1.0;
if apple,
    resxy = [1280 800];
    reshz = 60;%Main Lab near
elseif mainlab   
    resxy = [1920 1080]; %new Booth 1/ Booth 2
    reshz = 100;
    %reshz   = 144;%new Booth 1/ Booth 2
    screenWidthCM = 60; %Booth1/2
    vDistCM       = 90; %Booth1/2    
elseif booth4
    resxy = [1920 1080];
    reshz = 60;%Main Lab near
    screenWidthCM = 51; %Booth1/2
    vDistCM       = 70; %Booth1/2    
else
    resxy = [1024 768];
    gamma = 2.2;
end;

lumibg  = 0.5;
%Deg to Pixels
if apple, fixed_ppd = true;
else      fixed_ppd = false; end
if ~fixed_ppd
    ppd = pi * resxy(1) / atan(screenWidthCM/vDistCM/2) / 360 ; %average pixels per degree across screen
    ppd = round(ppd/4)*4;
else
    ppd = 40;
end
  
tasknames = {'Attention' 'Saccade'};
timeouts  = [2 1];
%-------------------
%Stimulus Dimensions
patchsize = round(0.75*ppd/2)*2;
patchenv  = patchsize/5; % patch spatial envelope s.d. (pix)
patchlum  = lumibg; % patch background luminance
gaborper  = 1/(4/ppd); % Gabor spatial period (pix)
gaborangs = [92 88]; % Gabor orientation (rad)
gaborphi  = 0; % Gabor unit phase
gaborcon  = 0.200;
alpha     = CreateCircularAperture(patchsize);

placecolor = [1 1 1]*0.70;
fixcolor   = [1 1 1]*0.70; 
%---------------
%[0.30,0.59,0.11]: rel. luminance of R, G, and B
lumicol  = 0.6; %cue color brightness - no higher than 0.4!!
cuecolor = {[1.0,0.5,0.0]*lumicol/0.595,... %orange
            [1.0,0.0,1.0]*lumicol/0.410};   %pink
        
cuelocLUT = [1 2 3 4 5 6 7 8
             5 6 7 8 1 2 3 4];
         
%Cue Dimensions
cuesiz    = round(0.20*ppd);
cueecc    = round(0.25*ppd);

%-----------------------------%
% Possible Stimulus locations %
%-----------------------------%
stimecc       = 6.00*ppd;
stimlocs      = 22.5:45:360;
nlocs         = length(stimlocs);
             
%-------------------
%Presentation Timing
cueonsetdelay  = 0.500;
cuepresent     = 0.200;
delay          = [0.9 1.3];
targpresent    = 0.200;
itirange       = [0.8 1 2];

%-------------------
%Response Parameters
response = [];
KbName('UnifyKeyNames');
keyquit   = KbName('ESCAPE');
keypause  = KbName('p'); % pause key (keep pressed until next trial)
keystart  = KbName('r'); % start key
keyreport = KbName({'c','m'});
mouseresp = [01];%left mouse button

%-------------------------------------
%EEG trigger setup, initially set to 0
if sendtriggers,
    IOPortfolder = 'C:\MATLAB\IOPort';
    addpath(IOPortfolder);
    [portobject, portaddress] = OpenIOPort;

    triggerlength = 0.005; %send trigger for 5 ms
    holdvalue     = 0;
end
trigglen   = 0.005; %set trigger for 10 ms
stimtrigg  = 128;%129: sample 1, 130: sample 2, ... 138: sample 10
masktrigg  = 132;
cuetrigg   = 096;
probetrigg = 160;%161: Left, 162: Right
keytrigg   = 192;%193: diagonal, 194: cardinal, 195: no response(timeout)
pausetrigg = 100;%if pause button is hit
triggoffs  = 32;
    
video = [];
try
    if apple,
        video.i = min(Screen('Screens'));
        Screen('Resolution',video.i,resxy(1),resxy(2));%reset
        Screen('Preference', 'SkipSyncTests', 1);
    else
        video.i = max(Screen('Screens'));
        Screen('Resolution',video.i,resxy(1),resxy(2),reshz);
    end;
    HideCursor;%reset
    FlushEvents;
    ListenChar(2);%reset
    PsychImaging('PrepareConfiguration');
    PsychImaging('AddTask','General','UseFastOffscreenWindows');
    PsychImaging('AddTask','General','NormalizedHighresColorRange');
    PsychImaging('AddTask','FinalFormatting','DisplayColorCorrection','SimpleGamma');
    video.h = PsychImaging('OpenWindow',video.i,0);
    
    [video.x,video.y] = Screen('WindowSize',video.h);
    video.ifi = Screen('GetFlipInterval',video.h,100,50e-6,10);
    video.ppd = ppd; video.lumibg = lumibg;
    Screen(video.h,'BlendFunction',GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    LoadIdentityClut(video.h);
    PsychColorCorrection('SetColorClampingRange',video.h,0,1);
    PsychColorCorrection('SetEncodingGamma',video.h,1/gamma);
    Priority(MaxPriority(video.h));
    Screen('FillRect',video.h,lumibg);
    Screen('Flip',video.h);
    Screen('ColorRange',video.h,1);
    Screen('TextFont',video.h,'Arial');
    Screen('TextSize',video.h,round(0.50*ppd));
    WaitSecs(0.100);
    
    %--------------------%
    %Create Fix Dot etc. %
    %--------------------%
    cuetex     = Screen('MakeTexture',video.h,cat(3,ones(cuesiz),CreateCircularAperture(cuesiz)),[],[],2);
    fixrec     = CenterRectOnPoint(Screen('Rect',cuetex),video.x/2,video.y/2);
    
    %-----------------------%
    % prepare placeholders  %
    %-----------------------%
    %placetex = Screen('MakeTexture',video.h,cat(3,ones(patchsize),alpha),[],[],2);
    placetex = Screen('MakeTexture',video.h,cat(3,ones(patchsize),CreateCircle(patchsize,round(0.05*ppd))),[],[],2);
    
    for iloc = 1:nlocs
        x = stimlocs(iloc);
        xy = round([cosd(x) sind(x)]*stimecc);
        placerec(:,iloc) = CenterRectOnPoint(Screen('Rect',placetex),video.x/2+xy(1),video.y/2+xy(2));
        xy = round([cosd(x) sind(x)]*cueecc);
        cuerec(:,iloc)   = CenterRectOnPoint(Screen('Rect',cuetex),  video.x/2+xy(1),video.y/2+xy(2));
    end
    
    %-----------------------%
    % prepare Gabor patches %
    %-----------------------%
    for iang = 1:length(gaborangs)
        gabor = CreateGabor(patchsize,patchenv,1/gaborper,gaborangs(iang),gaborphi,gaborcon);
        gabor = gabor+patchlum;
        gabor = min(1,max(0,gabor));
        gabortex(iang,1) = Screen('MakeTexture',video.h,cat(3,gabor,alpha),[],[],2);
    end
    
    %-----------------
    %Eye Tracker setup
    eyetracking = [];
    if iseyelinked
        eyetracking = struct;
        if EyelinkInit() ~= 1
            return
        end
        el = EyelinkInitDefaults(video.h);
        Eyelink('Command','binocular_enabled = YES');
%        Eyelink('Command','active_eye = LEFT');
        Eyelink('Command','pupil_size_diameter = YES');
        Eyelink('Command','file_sample_data = GAZE,AREA');
        Eyelink('Command','file_event_data = GAZE,AREA,VELOCITY');
        Eyelink('Command','file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK');
%        Eyelink('Command','file_event_filter = LEFT,FIXATION,SACCADE,BLINK');
        Eyelink('Command','link_sample_data = GAZE,AREA');
        Eyelink('Command','link_event_data = GAZE,AREA,VELOCITY');
        Eyelink('Command','link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK');
%        Eyelink('Command','link_event_filter = LEFT,FIXATION,SACCADE,BLINK');
        ETfilename = sprintf('AS_%s',participant.identifier);
        Eyelink('OpenFile',ETfilename);
        EyelinkDoTrackerSetup(el);
        Screen('FillRect',video.h,lumibg);
        Screen('Flip',video.h);
        %Eyelink('StartRecording');
    end
    
    %-----
    %Begin 
    t = Screen('Flip',video.h);
    aborted = false;
    nblocks = size(sequence,1);
    
    for iblock = 1:nblocks
        %recalibrate eyetracker every block??
        %if iseyelinked && recalibrate && iblock>1
        if iseyelinked && recalibrate && mod(iblock,2) && iblock>1
            EyelinkDoTrackerSetup(el);
            %Eyelink('StartRecording');
            Screen('FillRect',video.h,lumibg);
            Screen('Flip',video.h);
        end

        currtask = sequence(iblock,1).task(1);
        timeout  = timeouts(currtask);
        %Draw Block Start Screen
        label    = {};
        label{1} = sprintf('Press any key to start block %d/%d.',iblock,nblocks);
        label{2} = sprintf('Current task: %s.',tasknames{currtask});
        
        for i = 1:length(label)
            rec = CenterRectOnPoint(Screen('TextBounds',video.h,label{i}),video.x/2,round(1*ppd)+(i-1)*round(0.80*ppd));
            Screen('DrawText',video.h,label{i},rec(1),rec(2),1);
        end
        Screen('DrawingFinished',video.h);
        Screen('Flip',video.h,t+(round(0.100/video.ifi)-0.5)*video.ifi);
        KbStrokeWait;
        
        ntrials = size(sequence(iblock).cuelocation,1);
        
        response(iblock).resp = nan(ntrials,1);
        response(iblock).corr = nan(ntrials,1);
        response(iblock).time = nan(ntrials,1);
        sequence(iblock).delay = round((rand(ntrials,1)*diff(delay)+min(delay))/video.ifi)*video.ifi;
        sequence(iblock).iti   = randexp(itirange(1),itirange(2),itirange(3),[ntrials 1]);
        
        if iseyelinked
           eyetracking(iblock,1).trial = struct; 
        end
        %------------
        for itrial = 1:ntrials
            if CheckKeyPress(keyquit)
                aborted = true;
                break %out of trial loop (at end of current trial)
            end
            if CheckKeyPress(keypause)
                label = 'Please wait for instructions';
                rct = CenterRectOnPoint(Screen('TextBounds',video.h,label),video.x/2,round(1*ppd));
                Screen('DrawText',video.h,label,rct(1),rct(2),1);
                Screen('DrawTexture',video.h,fixtex,[],fixrec,[],[],[],fixcolor);
                Screen('DrawingFinished',video.h);
                Screen('Flip',video.h);
                WaitKeyPress(keystart,inf);
                
                label    = {};
                label{1} = 'Click the mouse to continue';
                for i = 1:length(label)
                    rec = CenterRectOnPoint(Screen('TextBounds',video.h,label{i}),video.x/2,round(1*ppd)+(i-1)*round(0.8*ppd));
                    Screen('DrawText',video.h,label{i},rec(1),rec(2),1);
                end
                Screen('DrawingFinished',video.h);
                Screen('Flip',video.h);
                KbStrokeWait;
                t = Screen('Flip',video.h);
                Screen('DrawTexture',video.h,fixtex,[],fixrec,[],[],[],fixcolor);
                Screen('DrawingFinished',video.h);
                t = Screen('Flip',video.h,t+(round(0.500/video.ifi)-0.5)*video.ifi);
            end
            
            %---------------------------
            %short break after half a block
            if floor(mod(itrial,ntrials/2)) == 0 && itrial ~=ntrials && ~practice
                t = Screen('Flip',video.h);
                label    = {};
                label{1} = 'Click the mouse when you are ready.';
                for i = 1:length(label)
                    rec = CenterRectOnPoint(Screen('TextBounds',video.h,label{i}),video.x/2,round(1*ppd)+(i-1)*round(0.8*ppd));
                    Screen('DrawText',video.h,label{i},rec(1),rec(2),1);
                end
                Screen('DrawingFinished',video.h);
                Screen('Flip',video.h,t+(round(1.000/video.ifi)-0.5)*video.ifi);
                KbStrokeWait;
                t = Screen('Flip',video.h);
                Screen('DrawTexture',video.h,cuetex,[],fixrec,[],[],[],fixcolor);
                Screen('DrawingFinished',video.h);
                t = Screen('Flip',video.h,t+(round(0.250/video.ifi)-0.5)*video.ifi);
            end
            
            %---------------------
            %set ITI, memory delay
            iti = sequence(iblock).iti(itrial);
            
            sequence(iblock).cuecolor(itrial,1) = ceil(rand*2);
            color   = cuecolor{sequence(iblock,1).cuecolor(itrial)};
            uncolor = cuecolor{3-sequence(iblock,1).cuecolor(itrial)};
            if iseyelinked
                Eyelink('StartRecording');
                Eyelink('Message',sprintf('B%02d_T%02d_ITI',iblock,itrial));
            end
            
            %-----%
            % ITI %
            %-----%
            Screen('DrawTexture',video.h,cuetex,[],fixrec,[],[],[],fixcolor);
            Screen('DrawingFinished',video.h);
            t = Screen('Flip',video.h,t+(round(iti/video.ifi)-0.5)*video.ifi);
            if iseyelinked,
                Eyelink('Message',sprintf('B%02d_T%02d_BEG',iblock,itrial));
            end
            if sendtriggers
                io64( portobject, portaddress, itrial );
                WaitSecs(triggerlength);
                io64( portobject, portaddress, holdvalue );
            end
            
            %------------------------%
            %  flash cue for 200 ms  %
            %------------------------%
            cueloc = sequence(iblock).cuelocation(itrial);
            Screen('DrawTexture', video.h,cuetex,[],fixrec,[],[],[],color);
            Screen('DrawTextures',video.h,cuetex,[],cuerec,[],[],[],fixcolor);
            Screen('DrawTexture', video.h,cuetex,[],cuerec(:,cueloc),[],[],[],  color);
            Screen('DrawTexture', video.h,cuetex,[],cuerec(:,cuelocLUT(2,cueloc)),[],[],[],uncolor);
            Screen('DrawTextures',video.h,placetex,[],placerec,[],[],[],placecolor);
            Screen('DrawingFinished',video.h);
            t = Screen('Flip',video.h,t+(round((cueonsetdelay)/video.ifi)-0.5)*video.ifi);
            if iseyelinked,
                Eyelink('Message',sprintf('B%02d_T%02d_CUE',iblock,itrial));
            end
            if sendtriggers
                io64( portobject, portaddress, cueloc + nlocs*(currtask-1) + 120 );
                WaitSecs(triggerlength);
                io64( portobject, portaddress, holdvalue );
            end

            Screen('DrawTexture',video.h,cuetex,[],fixrec,[],[],[],fixcolor);
            Screen('DrawTextures',video.h,placetex,[],placerec,[],[],[],placecolor);
            Screen('DrawingFinished',video.h);
            t = Screen('Flip',video.h,t+(round((cuepresent)/video.ifi)-0.5)*video.ifi);%stimulus offset
            
            %-------------------------%
            % present stimulus stream %
            %-------------------------%
            currdelay = sequence(iblock).delay(itrial);
            targloc   = sequence(iblock).targlocation(itrial);
            targtilt  = sequence(iblock).targtilt(itrial);
            
            Screen('DrawTexture',video.h,cuetex,[],fixrec,[],[],[],fixcolor);
            Screen('DrawTextures',video.h,placetex,[],placerec,[],[],[],placecolor);
            Screen('DrawTexture',video.h,gabortex(targtilt),[],placerec(:,targloc));
            Screen('DrawingFinished',video.h);
            t = Screen('Flip',video.h,t+(round((currdelay)/video.ifi)-0.5)*video.ifi);%stimulus onset
            if iseyelinked,
                Eyelink('Message',sprintf('B%02d_T%02d_ARR',iblock,itrial));
            end
            
            if sendtriggers
                io64( portobject, portaddress, targloc + nlocs*(currtask-1) + 136 );
                WaitSecs(triggerlength);
                io64( portobject, portaddress, holdvalue );
            end
            Screen('DrawTextures',video.h,placetex,[],placerec,[],[],[],placecolor);
            Screen('DrawTexture',video.h,cuetex,[],fixrec,[],[],[],fixcolor);
            Screen('DrawingFinished',video.h);
            t = Screen('Flip',video.h,t+(round(targpresent/video.ifi)-0.5)*video.ifi);%stimulus offset
            
            %---------------%
            % response loop %
            %---------------%
            responded = false;
            tons      = GetSecs;
            while ~responded
                tnow = GetSecs;
                %end response polling after 'timeout'
                if tnow-tons > timeout
                    response(iblock).resp(itrial,1) = 0;
                    response(iblock).corr(itrial,1) = false;
                    response(iblock).time(itrial,1) = NaN;
                    response(iblock).tout(itrial,1) = true;
                    break
                end
                %check for response
                if apple
                    [key,tkey] = CheckKeyPress(keyreport);
                else
                    [key,tkey] = CheckKeyPress(keyreport);
                end
                %if a key has been pressed, get accuracy and RT and end
                %response polling
                if key > 0
                    timestamps(itrial,6,1:2)        = tkey;
                    response(iblock).resp(itrial,1) = key;
                    response(iblock).corr(itrial,1) = response(iblock).resp(itrial,1) == sequence(iblock).targtilt(itrial);
                    response(iblock).time(itrial,1) = tkey-tons;
                    response(iblock).tout(itrial,1) = false;
                    responded = true;
                    if sendtriggers
                        io64( portobject, portaddress, key + 2*(currtask-1) + 152 );
                        WaitSecs(triggerlength);
                        io64( portobject, portaddress, holdvalue );
                    end
                    if iseyelinked
                        Eyelink('Message',sprintf('B%02d_T%02d_RESP',iblock,itrial));
                    end
                end
            end

            %present feedback
            Screen('DrawTexture',video.h,cuetex,[],fixrec,[],[],[],fixcolor);
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h,t+(round(0.016/video.ifi)-0.5)*video.ifi);
            if iseyelinked,
                Eyelink('Message',sprintf('B%02d_T%02d_END',iblock,itrial));
                Eyelink('StopRecording');
            end
        end %trial loop
        
        if aborted
            break %out of block loop
        end
        %-------------------------------
        %End of Block
        Screen('DrawingFinished',video.h);
        t = Screen('Flip',video.h);
        if ~ practice,            
            label    = {};
            if iblock < nblocks
                label{1} = sprintf(['Well done! Take a break.']);
            else
                label{1} = sprintf('You''re done! Thanks for participating.');
            end
            for i = 1:length(label)
                rec = CenterRectOnPoint(Screen('TextBounds',video.h,label{i}),video.x/2,round(1*ppd)+(i-1)*round(0.8*ppd));
                Screen('DrawText',video.h,label{i},rec(1),rec(2),1);
            end
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h,t+(round(0.500/video.ifi)-0.5)*video.ifi);
        else
            label    = {};
            label{1} = sprintf('You are now done with round %d of %d practice blocks.',iblock,nblocks);
            for i = 1:length(label)
                rec = CenterRectOnPoint(Screen('TextBounds',video.h,label{i}),video.x/2,round(1*ppd)+(i-1)*round(0.8*ppd));
                Screen('DrawText',video.h,label{i},rec(1),rec(2),1);
            end
            Screen('DrawingFinished',video.h);
            Screen('Flip',video.h,t+(round(0.500/video.ifi)-0.5)*video.ifi);
        end;
        KbStrokeWait;
        t = Screen('Flip',video.h);
        if ~practice, save(sprintf('./Data/%s_B%02d_experiment.mat',participant.filename,iblock),'participant','sequence','response','video','eyetracking'); end
        
    end %block loop
    
    if sendtriggers
        try
            CloseIOPort;
        catch
        end
    end
    
    if iseyelinked
        try
            Eyelink('StopRecording');
        catch
        end
        Eyelink('CloseFile');
        ntimes = 1;
        while ntimes <= 10
            status = Eyelink('ReceiveFile');
            if status > 0
                break
            end
            ntimes = ntimes+1;
        end
        if status <= 0
            warning('EyeLink data has not been saved properly.');
        else
            %rename file with participant id!!!
            copyfile([ETfilename '.edf'],['./Data/' participant.filename '.edf']);
        end
    end
    
    if ~practice
        label    = {};
        label{1} = 'Press any key to exit.';
        for i = 1:length(label)
            rec = CenterRectOnPoint(Screen('TextBounds',video.h,label{i}),video.x/2,round(1*ppd)+(i-1)*round(0.8*ppd));
            Screen('DrawText',video.h,label{i},rec(1),rec(2),1);
        end
        Screen('DrawingFinished',video.h);    
        Screen('Flip',video.h,t+(round(0.250/video.ifi)-0.5)*video.ifi);
        KbStrokeWait;
    end
    
    Priority(0);
    Screen('CloseAll');
    FlushEvents;
    ListenChar(0);
    ShowCursor;
    
catch
    if sendtriggers
        try
        CloseIOPort;
        catch
        end
    end
    
    Priority(0);
    if isfield(video,'h')
        Screen('CloseAll');
    end
    FlushEvents;
    ListenChar(0);
    ShowCursor;
    
    rethrow(lasterror);
    
end

end