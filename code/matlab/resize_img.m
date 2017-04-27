I = imread('test.png');

for scale = 1:-0.1:0.1
    Iout = imresize(I, scale, 'nearest'); 
    imwrite(Iout, ['test' num2str(scale) '.png'] );
end