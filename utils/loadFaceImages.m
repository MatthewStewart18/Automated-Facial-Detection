function [images, labels] = loadFaceImages(filename,augmented,sampling)

% this is a flag that allows you to activate/deactivate the data augmentation
% set this to 1 for it to be activated
% Data augmentation will increase the size of the dataset by created variations 
%(mirroring, flipping, displacements) of each given image. This aims to produce more
% training images and, therefore, improve performance
if nargin<2
    sampling =1;
    augmented=1;
end

if nargin<3
    sampling =1;
end


fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);


line1=fgetl(fp);
line2=fgetl(fp);

numberOfImages = fscanf(fp,'%d',1);

images=[];
labels =[];
for im=1:sampling:numberOfImages
    
    label = fscanf(fp,'%d',1);
    
    labels= [labels; label];
    
    imfile = fscanf(fp,'%s',1);
    I=imread(imfile);
    if size(I,3)>1
        I=rgb2gray(I);
    end
    vector = reshape(I,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    
    images= [images; vector];
    
    if augmented == 1
        
        if label==1
            Itemp =fliplr(I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(I,1)
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(I,-1);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(I,[0 1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(I,[0 -1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(fliplr(I),1)
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(fliplr(I),-1);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(fliplr(I),[0 1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =circshift(fliplr(I),[0 -1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
        else
            Itemp =fliplr(I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =flipud(I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            
            Itemp =flipud(fliplr(I));
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];      
        end

    end
    
end

fclose(fp);

end
