fprintf(1,'Executing %s at %s:\n',mfilename(),datestr(now));
ver,
try,
addpath('/usr/local/MATLAB/R2014a/toolbox/spm12');

        %% Generated by nipype.interfaces.spm
        if isempty(which('spm')),
             throw(MException('SPMCheck:NotFound', 'SPM not in matlab path'));
        end
        [name, version] = spm('ver');
        fprintf('SPM version: %s Release: %s\n',name, version);
        fprintf('SPM path: %s\n', which('spm'));
        spm('Defaults','fMRI');

        if strcmp(name, 'SPM8') || strcmp(name(1:5), 'SPM12'),
           spm_jobman('initcfg');
           spm_get_defaults('cmdline', 1);
        end

        jobs{1}.spm.spatial.normalise.estwrite.subj.vol = {...
'/mnt/HDD/Data/nipype_test/output/freesurfer/converted_subject.nii,1';...
};
jobs{1}.spm.spatial.normalise.estwrite.subj.resample = {...
'/mnt/HDD/Data/nipype_test/output/freesurfer/converted_subject.nii,1';...
};
jobs{1}.spm.spatial.normalise.estwrite.eoptions.tpm = {...
'/mnt/HDD/Data/nipype_test/output/freesurfer/converted_template.nii,1';...
};
jobs{1}.spm.spatial.normalise.estwrite.woptions.prefix = 'w';

        spm_jobman('run', jobs);

        
,catch ME,
fprintf(2,'MATLAB code threw an exception:\n');
fprintf(2,'%s\n',ME.message);
if length(ME.stack) ~= 0, fprintf(2,'File:%s\nName:%s\nLine:%d\n',ME.stack.file,ME.stack.name,ME.stack.line);, end;
end;