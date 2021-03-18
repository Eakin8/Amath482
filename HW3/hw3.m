%% Load video
clear all; clc
load('cam1_1.mat')
load('cam2_1.mat')
load('cam3_1.mat')
implay(vidFrames1_1)
implay(vidFrames2_1)
implay(vidFrames3_1)

numFrames11 = size(vidFrames1_1,4);
% [height11 width11 rgb11 numFrames11] = size(vidFrames1_1);
numFrames21 = size(vidFrames2_1,4);
numFrames31 = size(vidFrames3_1,4);
%% Cam 1

xa = zeros(1,numFrames11);
ya = zeros(1,numFrames11);

for j = 1:numFrames11
    X = vidFrames1_1(:,:,:,j);
    X = rgb2gray(X);
    %X11(:,:,j) = im2double(X);
    X = im2double(X);
   
    X(:,1:300) = 0;
    X(:,400:end) = 0;
    X(1:200,:) = 0;
    
    %imshow(X); drawnow
    [M,I] = max(X(:));
    [y,x] = ind2sub(size(X),I);
    xa(j) = x;
    ya(j) = y;
    %xa = [xa,x];
    %ya = [ya,y];
end

%% Cam 2

xb = zeros(1,numFrames21);
yb = zeros(1,numFrames21);

for j = 1:numFrames21
    X = vidFrames2_1(:,:,:,j);
    X = rgb2gray(X);
    %X21(:,:,j) = im2double(X);
    X = im2double(X);
  
    X(:,1:250) = 0;
    X(:,350:end) = 0;
    X(1:80,:) = 0;
    X(400:end,:) = 0;
    
    %imshow(X); drawnow
    [M,I] = max(X(:));
    [y,x] = ind2sub(size(X),I);
    xb(j) = x;
    yb(j) = y;
end

%% Cam 3

xc = zeros(1,numFrames31);
yc = zeros(1,numFrames31);

for j = 1:numFrames31
    X = vidFrames3_1(:,:,:,j);
    X = rgb2gray(X);
    %X31(:,:,j) = im2double(X);
    X = im2double(X);
    
    X(:,1:250) = 0;
    X(:,480:end) = 0;
    X(1:240,:) = 0;
    X(330:end,:) = 0;
    
    %imshow(X); drawnow
    [M,I] = max(X(:));
    [y,x] = ind2sub(size(X),I);
    xc(j) = x;
    yc(j) = y;
end

%% Alignment
yamin = min(ya);
yamin_loci = find(ya == yamin);
linedxa = xa(yamin_loci(1):end);
linedya = ya(yamin_loci(1):end);

linelength = length(linedya);
ybmin = min(yb);
ybmin_loci = find(yb == ybmin);
newlength = ybmin_loci(1)+linelength-1;
linedxb = xb(ybmin_loci(1):newlength);
linedyb = yb(ybmin_loci(1):newlength);

xcmin = min(xc);
xcmin_loci = find(xc == xcmin);
newlength2 = xcmin_loci(1)+linelength-1;
linedxc = xc(xcmin_loci(1):newlength2);
linedyc = yc(xcmin_loci(1):newlength2);

%plot(1:linelength,linedya,1:linelength,linedyb,1:linelength,linedxc);

%% SVD
vec = [linedxa;linedya;linedxb;linedyb;linedxc;linedyc];
[m,n] = size(vec);
mn = mean(vec,2);
vec = vec-repmat(mn,1,n);

CX = (1/(n-1))*vec*vec';
A = vec/sqrt(n-1);
[U,S,V] = svd(A, 'econ');
Y = U'*vec;

%% Position Graph

figure(1)
subplot(3,2,1)
plot(vec(1,:))
%plot(1:numFrames11,vec(1,:))
%plot(xa);
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in X')
title('Case 1 Cam 1')

subplot(3,2,2)
plot(vec(2,:))
%plot(1:numFrames11,vec(2,:))
%plot(ya);
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in Y')
title('Case 1 Cam 1')

subplot(3,2,3)
plot(vec(3,:))
%plot(1:numFrames21,vec(3,:))
%plot(xb);
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in X')
title('Case 1 Cam 2')

subplot(3,2,4)
plot(vec(4,:))
ylim([-200,200])
%plot(1:numFrames21,vec(4,:))
%plot(yb);
xlabel('Time Frame')
ylabel('Position in Y')
title('Case 1 Cam 2')

subplot(3,2,5)
plot(vec(5,:))
%plot(1:numFrames31,vec(5,:))
%plot(xc);
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in X')
title('Case 1 Cam 3')

subplot(3,2,6)
plot(vec(6,:))
%plot(1:numFrames31,vec(6,:))
%plot(yc);
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in Y')
title('Case 1 Cam 3')

%% Energy Captured
sigma = diag(S);

energy = zeros(1,length(sigma));
for j = 1:length(sigma)
    energy(j) = sigma(j)^2/sum(sigma.^2);
end
figure(2)
subplot(3,1,1)
plot(energy,'ko','Linewidth',2)
ylabel('Energy')
ylim([0,1])
title('Energy')
subplot(3,1,2)
plot(sigma,'ko','Linewidth',2)
ylabel('\sigma')
title('Singular Value')
subplot(3,1,3)
plot(cumsum(energy),'ko','Linewidth',2)
ylabel('Cumu Energy')
ylim([0,1])
title('Cumulative Energy')

%% PCA
figure(3)
plot(Y(1,:)), hold on
plot(Y(2,:))
title('Case 1 - Ideal Case')
legend('PC1','PC2')




    