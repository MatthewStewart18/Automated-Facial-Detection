function [Edges, Iver, Ihor] = edgeExtraction(Iin, B1, B2)
Iin = double(Iin);
Iver = filter2(double(B1), Iin);
Ihor = filter2(double(B2), Iin);
Edges = uint8(sqrt((Iver).^2 + (Ihor).^2));
end