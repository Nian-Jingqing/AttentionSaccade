function [sequence] = AttSacc_BuildExperiment(nblocks,practice)
if nargin < 2
    practice = false;
end
if nargin < 1
    nblocks = 16;
end
if mod(nblocks,2) & ~practice
    error('Number of blocks must be even!')
end
%%
nlocs    = 8;
validity = 0.80;
validity = round(validity/(1-validity)); %transform validity to a ratio
design   = [1:nlocs]';
design   = cat(1,[design zeros(nlocs,1)],repmat([design ones(nlocs,1)],validity,1));
       
if practice
    trialsperblock = 40;
else
    trialsperblock = 120;
end

nreps   = ceil(trialsperblock/size(design,1));
seq     = repmat(design,nreps,1);
ntrials = size(seq,1);

sequence  = struct;
task      = [ones(1,nblocks/2) ones(1,nblocks/2)*2];
task      = task(randperm(nblocks));

targettilt = [ones(ntrials/2,1);ones(ntrials/2,1)*2];

for block = 1:nblocks
    seq    = seq(randperm(ntrials),:);
    sequence(block,1).cuelocation  = seq(:,1);
    sequence(block,1).validity     = seq(:,2);
    sequence(block,1).targlocation = seq(:,1);
    sequence(block,1).targtilt     = targettilt(randperm(ntrials));
    for itrial = find(~seq(:,2)')
        possloc = setdiff(1:nlocs,sequence(block,1).targlocation(itrial));
        possloc = possloc(randperm(nlocs-1));
        sequence(block,1).targlocation(itrial) = possloc(1);
    end
    sequence(block,1).task  = ones(ntrials,1)*task(block);
end

end