imgname = 'circle_small.png';
I = imread(imgname);
r = I(:,:,1); g = I(:,:,2); b = I(:,:,3);
ridx = r > g;
gidx = g > r;

r(ridx) = 255; g(ridx) = 0; b(ridx) = 0;
g(gidx) = 255; r(gidx) = 0; b(gidx) = 0;

I = cat(3, r, g, b);
imwrite(I, imgname);