%% Load video
clear all; clc
load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')
%implay(vidFrames1_2)
%implay(vidFrames2_2)
%implay(vidFrames3_2)

numFrames13 = size(vidFrames1_3,4);
numFrames23 = size(vidFrames2_3,4);
numFrames33 = size(vidFrames3_3,4);

% Cam 1

xa = zeros(1,numFrames13);
ya = zeros(1,numFrames13);

for j = 1:numFrames13
    X = vidFrames1_3(:,:,:,j);
    X = rgb2gray(X);
    X = im2double(X);
   
    X(:,1:280) = 0;
    X(:,380:end) = 0;
    X(1:220,:) = 0;
    X(400:end,:) = 0;
    
    %imshow(X); drawnow
    [M,I] = max(X(:));
    [y,x] = ind2sub(size(X),I);
    xa(j) = x;
    ya(j) = y;
end

% Cam 2

xb = zeros(1,numFrames23);
yb = zeros(1,numFrames23);

for j = 1:numFrames23
    X = vidFrames2_3(:,:,:,j);
    X = rgb2gray(X);
    X = im2double(X);
  
    X(:,1:200) = 0;
    X(:,380:end) = 0;
    X(1:150,:) = 0;
    X(400:end,:) = 0;
    
    %imshow(X); drawnow
    [M,I] = max(X(:));
    [y,x] = ind2sub(size(X),I);
    xb(j) = x;
    yb(j) = y;
end

% Cam 3

xc = zeros(1,numFrames33);
yc = zeros(1,numFrames33);

for j = 1:numFrames33
    X = vidFrames3_3(:,:,:,j);
    X = rgb2gray(X);
    X = im2double(X);
    
    X(:,1:200) = 0;
    X(:,450:end) = 0;
    X(1:180,:) = 0;
    X(300:end,:) = 0;
    
    %imshow(X); drawnow
    [M,I] = max(X(:));
    [y,x] = ind2sub(size(X),I);
    xc(j) = x;
    yc(j) = y;
end

% position check
figure(1)
subplot(3,2,1)
plot(xa)
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in X')
title('Case 1 Cam 1')

subplot(3,2,2)
plot(ya)
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in Y')
title('Case 1 Cam 1')

subplot(3,2,3)
plot(xb)
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in X')
title('Case 1 Cam 2')

subplot(3,2,4)
plot(yb)
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in Y')
title('Case 1 Cam 2')

subplot(3,2,5)
plot(xc)
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in X')
title('Case 1 Cam 3')

subplot(3,2,6)
plot(yc)
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in Y')
title('Case 1 Cam 3')






% Alignment
yamin = 271;
%yamin = min(ya);
yamin_loci = find(ya == yamin);
linedxa = xa(yamin_loci(1):end);
linedya = ya(yamin_loci(1):end);

linelength = length(linedya);
ybmin = 207;
ybmin_loci = find(yb == ybmin);
newlength = ybmin_loci(1)+linelength-1;
linedxb = xb(ybmin_loci(1):newlength);
linedyb = yb(ybmin_loci(1):newlength);

xcmin = 296;
xcmin_loci = find(xc == xcmin);
newlength2 = xcmin_loci(1)+linelength-1;
linedxc = xc(xcmin_loci(1):newlength2);
linedyc = yc(xcmin_loci(1):newlength2);

plot(1:linelength,linedya,1:linelength,linedyb,1:linelength,linedxc);

% SVD
vec = [linedxa;linedya;linedxb;linedyb;linedxc;linedyc];
[m,n] = size(vec);
mn = mean(vec,2);
vec = vec-repmat(mn,1,n);

CX = (1/(n-1))*vec*vec';
A = vec/sqrt(n-1);
[U,S,V] = svd(A, 'econ');
Y = U'*vec;

% Position Graph

figure(1)
subplot(3,2,1)
plot(vec(1,:))
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in X')
title('Case 3 Cam 1')

subplot(3,2,2)
plot(vec(2,:))
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in Y')
title('Case 3 Cam 1')

subplot(3,2,3)
plot(vec(3,:))
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in X')
title('Case 3 Cam 2')

subplot(3,2,4)
plot(vec(4,:))
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in Y')
title('Case 3 Cam 2')

subplot(3,2,5)
plot(vec(5,:))
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in X')
title('Case 3 Cam 3')

subplot(3,2,6)
plot(vec(6,:))
ylim([-200,200])
xlabel('Time Frame')
ylabel('Position in Y')
title('Case 3 Cam 3')

% Energy Captured
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

% PCA
figure(3)
plot(Y(1,:)), hold on
plot(Y(2,:)),hold on
plot(Y(3,:))
title('Case 3 - Horizontal Displacement')
legend('PC1','PC2','PC3')




    