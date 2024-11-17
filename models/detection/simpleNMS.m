function Objects = simpleNMS(Objects, th)

% Initialize a list to keep track of the boxes we want to keep
keep = true(size(Objects, 1), 1);

for i = 1:size(Objects, 1)
    if ~keep(i)
        continue; % Skip if already marked for removal
    end
    
    % Current bounding box
    box1 = Objects(i, :);
    area1 = box1(3) * box1(4); % Width * Height
    
    % Compare with all subsequent bounding boxes
    for j = i+1:size(Objects, 1)
        if ~keep(j)
            continue; % Skip if already marked for removal
        end
        
        % Second bounding box
        box2 = Objects(j, :);
        area2 = box2(3) * box2(4);

        % Calculate the intersection area
        interArea = rectint([box1(1:2), box1(3:4)], [box2(1:2), box2(3:4)]);
        
        % Calculate the intersection over the area of the smaller box
        minArea = min(area1, area2);
        overlapRatio = interArea / minArea;

        % Suppress the box with lower confidence if overlap is above threshold
        if overlapRatio > th
            if box1(5) > box2(5)
                keep(j) = false; % Remove box2
            else
                keep(i) = false; % Remove box1
                break; % Stop comparing box1 as it's being removed
            end
        end
    end
end

Objects = Objects(keep, :);
end