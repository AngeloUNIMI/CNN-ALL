function [] = step_A_process_ALL_IDB(dirOrig, dirTest, ext, fidLogs, savefile)

%dbname
dbnameOld = 'ALL-IDB';
dbnameNew = 'ALL_IDB';

%part
dbPartOld = 'ALL_IDB2';
dbPartNew = 'ALL_IDB2';

%dirIn e Out
dirIn = [dirOrig dbnameOld '\' dbPartOld '\img\'];
dirOut = [dirTest dbnameNew '\' dbPartNew '\'];
mkdir_pers(dirOut, savefile);

%loop
files = dir([dirIn '*.' ext]);

%init
%class = -1 .* ones(length(files));

for i = 1 : numel(files)
    
    filename = files(i).name;
    
    %display
    fprintf_pers(fidLogs, [filename '\n']);
    
    [C, ~] = strsplit(filename, 'Im');
   
    filenameNew = C{2};
    
    copyfile([dirIn files(i).name], [dirOut filenameNew]);
    
    %class
    [C, ~] = strsplit(filename, {'_', '.'});
    
    %structure
    problem(i).filename = filenameNew;
    problem(i).class = str2double(C{2});
    
end %for i


%save
save([dirOut 'classes.mat'], 'problem');