function [accuracy, precision, recall, cumtp, cumfp, fn] = evaluateDetections(detections, groundTruth, iouThreshold)
    [~, sortIdx] = sort(detections(:,5), 'descend');
    detections = detections(sortIdx,:);
    
    numDetections = size(detections, 1);
    numGroundTruths = size(groundTruth, 1);
    
    tp = zeros(numDetections, 1);
    fp = zeros(numDetections, 1);
    gtMatched = false(numGroundTruths, 1);
    
    for i = 1:numDetections
        maxIoU = 0;
        maxIdx = 0;
        
        for j = 1:numGroundTruths
            if ~gtMatched(j)
                iou = bboxOverlapRatio(detections(i,1:4), groundTruth(j,1:4));
                if iou > maxIoU
                    maxIoU = iou;
                    maxIdx = j;
                end
            end
        end
        
        if maxIoU >= iouThreshold
            tp(i) = 1;
            gtMatched(maxIdx) = true;
        else
            fp(i) = 1;
        end
    end
    
    fn = sum(~gtMatched);
    cumtp = cumsum(tp);
    cumfp = cumsum(fp);
    precision = cumtp ./ (cumtp + cumfp);
    recall = cumtp / numGroundTruths;
    accuracy = cumtp(end) / (cumtp(end) + cumfp(end) + fn);
end
