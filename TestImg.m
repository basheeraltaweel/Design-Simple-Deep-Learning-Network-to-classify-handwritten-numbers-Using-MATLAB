close all
Image=imread('pic13.jpg');
I=rgb2gray(Image);
I=imresize(I,[28,28]);
I=255-I;
figure
imshow(I)
result_of_image_read= classify(net,I);
title(result_of_image_read)
