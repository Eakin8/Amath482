clear all; clc
%% Load
v = VideoReader('ski_drop_low.mp4');

numFrame = v.NumberOfFrames;
dura = v.Duration;
height = v.Height;
width = v.Width;

m_frames = [];
for j = 1:numFrame
    frames = rgb2gray(read(v,j));
    frames = reshape(frames,height*width,1);
    m_frames = [m_frames frames];
    %imshow(frames); drawnow
end;
m_frames = im2double(m_frames);

%% Energy
X1 = m_frames(:,1:end-1);
X2 = m_frames(:,2:end);
[U,Sigma,V] = svd(X1,'econ');

figure(1)
sig = diag(Sigma);
energy = zeros(1,length(sig));
for j = 1:length(sig)
    energy(j) = sig(j)^2/sum(sig.^2);
end
subplot(2,1,1)
plot(sig, 'ko', 'Linewidth',2)
xlim([0,100])
ylabel('\sigma')
title('Singular Value')
subplot(2,1,2)
plot(cumsum(energy),'ko', 'Linewidth',2)
xlim([0,100])
ylabel('Energy')
title('Cumulative Energy')

%% DMD
r = 1;
U_r = U(:,1:r);
Sigma_r = Sigma(1:r,1:r);
V_r = V(:,1:r);
S = U_r' * X2 * V_r /Sigma_r;
[eV,D] = eig(S);
mu = diag(D);

dt = dura/(numFrame - 1);
omega = log(mu)/dt;

Phi = X2*V_r/Sigma_r*eV;

%% Reconstruction 
y0 = Phi\X1(:,1); 

t = 0:dt:dura;
modes = zeros(r,length(t));
for iter = 1:length(t)
    modes(:,iter) = y0.*exp(omega*t(iter));
end
dmd = Phi*modes;

%% Sparse
sparse = m_frames - abs(dmd);
R = sparse .* (sparse < 0);
fore = sparse - R;
fore_gray = sparse + 0.3;

figure(2)
for j = 1:numFrame
    subplot(2,2,1)
    ori = reshape(m_frames(:,j),height,width);
    imshow(ori);
    title('Original Video');
   
    subplot(2,2,2)
    bg = reshape(dmd(:,j),height,width);
    imshow(bg);
    title('Background');
    
    subplot(2,2,3)
    fg = reshape(fore(:,j),height,width);
    imshow(fg);
    title('Foreground with R subtraction');
    
    subplot(2,2,4)
    fg = reshape(fore_gray(:,j),height,width);
    imshow(fg);
    title('Foreground without R subtraction');
    
    drawnow
end
