% Clean workspace
clear all; close all; clc
 
load subdata.mat % imports data as the 262144*49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y = x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

% compute transfered noisy signal using a loop
sum_ave = zeros(n,n,n);

for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    M = max(abs(Un),[],'all');
    Utn = fftn(Un);
    sum_ave = sum_ave + Utn;
    close all, isosurface(X,Y,Z,abs(Un)/M,0.7)
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
    pause(0.01)
end

% averaging the spectrum and determine the frequency signature
ave = abs(fftshift(sum_ave))/49;
M = max(abs(ave),[],'all');
close all, 
[f,v] = isosurface(Kx,Ky,Kz,abs(ave)/M,0.7);
axis([-20 20 -20 20 -20 20]), grid on, drawnow

cfreq = mean(v);
kx = cfreq(1);
ky = cfreq(2);
kz = cfreq(3);

%% 
% filter the data around the central frequency 
tau = 0.2;

filter_x = exp(-tau*(Kx-kx).^2);
filter_y = exp(-tau*(Ky-ky).^2);
filter_z = exp(-tau*(Kz-kz).^2);
filter_center = filter_x.*filter_y.*filter_z;

loci = zeros(49,3);
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j), n,n,n);
    Utn = fftn(Un);
    Unft = filter_center.*fftshift(Utn);
    Unft = ifftshift(Unft);
    Unf = ifftn(Unft);
    M = max(abs(Unf),[],'all');
    [f,v] = isosurface(X,Y,Z,abs(Unf)/M,0.7);
    loci(j,:) = mean(v);
end

% plot the path of submarine
plot3(loci(:,1),loci(:,2),loci(:,3));
legend ('Path of submarine','Location','best');
xlabel('x');
ylabel('y');
zlabel('z');
title("Track the submarine", 'Fontsize', 14)

% table of x and y lociinates of P-8 Poseidon subtracking aircraft position
T = table(loci(49,1),loci(49,2));
