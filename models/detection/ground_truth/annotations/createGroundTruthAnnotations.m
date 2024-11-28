function createGroundTruthAnnotations(imagePath, outputPath)
    img = imread(imagePath);
   
    bounding_boxes = [];
    
    figure('Name', 'Bounding boxes');
    imshow(img);
    hold on;
    
    while true
        % Let user draw rectangle
        title(['Lets goooo baby draw a rectangle (type done in terminal' ...
            ' when finished or press Enter to select another face)']);
        rect = drawrectangle();
        
        % Store coordinates
        if ~isempty(rect)
            pos = rect.Position;
            bounding_boxes = [bounding_boxes; pos];
            
            fprintf('Added bounding box: x1=%d, y1=%d, x2=%d, y2=%d\n', ...
                round(pos(1)), round(pos(2)), round(pos(3)), round(pos(4)));
        end
        
        % allow another face entry
        key = input('Press Enter to add another face, or type "done" to finish: ', 's');
        if strcmp(key, 'done')
            break;
        end
    end
    
    % Save the coords
    fid = fopen(outputPath, 'w');
    fprintf(fid, '# Format: <x1> <y1> <x2> <y2>\n');
    for i = 1:size(bounding_boxes, 1)
        fprintf(fid, '%d %d %d %d\n', round(bounding_boxes(i,:)));
    end
    fclose(fid);
    
    close all;
end
