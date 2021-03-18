%% Load music clip for GNR
figure(1)
[y, Fs] = audioread('GNR_T.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Sweet Child O Mine');
p8 = audioplayer(y,Fs); playblocking(p8);

L = tr_gnr; n = length(y);
t2 = linspace(0,L,n+1);
t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);


%%% Use Gabor Transform
% Creating a spectrogram
a = 1000;
tau = 0:0.1:L;

Sgt_spec = zeros(length(y),length(tau));
for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2); % window function
    Sg = g.*y';
    Sgt = fft(Sg);
    Sgt_spec(:,j) = fftshift(abs(Sgt));
end

figure(2)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[0 2000],'Fontsize',16)
colormap(hot);
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title("Spectrogram for GNR in the Trimmed 2 seconds", 'Fontsize', 14)

%% Load music clip for Floyd
figure(3)
[y, Fs] = audioread('Floyd_T2.m4a');
tr_floyd = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Comfortably Numb');
p8 = audioplayer(y,Fs); playblocking(p8);

L = tr_floyd; n = length(y);
t2 = linspace(0,L,n+1);
t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

%%% Use Gabor Transform
% Creating a spectrogram 
a = 100;
tau = 0:0.1:L;

Sgt_spec = zeros(length(y),length(tau));
for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2); % window function
    Sg = g.*y';
    Sgt = fft(Sg);
    Sgt_spec(:,j) = fftshift(abs(Sgt));
end

figure(4)
dim_agree = pcolor(tau,ks,Sgt_spec(1:length(ks),:));
shading interp
set(gca,'ylim',[0 300],'Fontsize',16)
colormap(hot);
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title("Spectrogram for Floyd", 'Fontsize', 14)

%% Filtering the Bass

close all, clear

[y, Fs] = audioread('Floyd.m4a');
tr_floyd = length(y)/Fs; % record time in seconds


L = tr_floyd; n = length(y);
t2 = linspace(0,L,n+1);
t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

% Gaussian filter
a = 150;
tau = 0:0.5:L;
time = zeros(length(ks),length(tau));
for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2);
    Sg = g.*y';
    Sgt = fft(Sg);
    Sgt_spec(:,j) = fftshift(abs(Sgt));
    Sgt_spec_time = Sgt_spec(:,j);
    positive_ks = abs(ks);
    amp = positive_ks < 250;
    amp_vec = Sgt_spec_time(amp);
    [M,I] = max(amp_vec);
    amp_loci = find(Sgt_spec_time == M);
    max_freq = ks(amp_loci);
    
    new_tau = 0.3;
    filter_1 = exp(-new_tau*(ks-max_freq(1)).^2);
    filter_2 = exp(-new_tau*(ks-max_freq(2)).^2);
    temp = Sgt_spec(1:length(ks));
    Sgt_spec(1:length(ks),j) = Sgt_spec(1:length(ks),j).'.*filter_1 + Sgt_spec(1:length(ks),j).'.*filter_2;
    
    % transfer to music audio
    Sgt = fftshift(Sgt);
    sgft = Sgt(1:length(ks)).*filter_1 + Sgt(1:length(ks)).*filter_2;
    Sgf = ifft(sgft);
    time = Sgf.' + time;
   
end
plot((1:length(ks))/Fs,time);
xlabel('Time [sec]'); ylabel('Amplitude');
title("Audio of bass in Floyd");


figure(5)
pcolor(tau,ks,Sgt_spec(1:length(ks),:))
shading interp
set(gca, 'ylim', [0 300], 'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title("Spectrogram for Floyd Bass", 'Fontsize', 14);

%% Filtering the Guitar

close all, clear

[y, Fs] = audioread('Floyd_T1.m4a');
tr_floyd = length(y)/Fs; % record time in seconds


L = tr_floyd; n = length(y);
t2 = linspace(0,L,n+1);
t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

% Gaussian filter
a = 150;
tau = 0:0.5:L;
time = zeros(length(ks),length(tau));
for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2);
    Sg = g.*y';
    Sgt = fft(Sg);
    Sgt_spec(:,j) = fftshift(abs(Sgt));
    Sgt_spec_time = Sgt_spec(:,j);
    positive_ks = abs(ks);
    amp = positive_ks > 250;
    amp_vec = Sgt_spec_time(amp);
    [M,I] = max(amp_vec);
    amp_loci = find(Sgt_spec_time == M);
    max_freq = ks(amp_loci);
    
    new_tau = 0.3;
    filter_1 = exp(-new_tau*(ks-max_freq(1)).^2);
    filter_2 = exp(-new_tau*(ks-max_freq(2)).^2);
    temp = Sgt_spec(1:length(ks));
    Sgt_spec(1:length(ks),j) = Sgt_spec(1:length(ks),j).'.*filter_1 + Sgt_spec(1:length(ks),j).'.*filter_2;
    
    % transfer to music audio
    Sgt = fftshift(Sgt);
    sgft = Sgt(1:length(ks)).*filter_1 + Sgt(1:length(ks)).*filter_2;
    Sgf = ifft(sgft);
    time = Sgf.' + time;
   
end
plot((1:length(ks))/Fs,time);
xlabel('Time [sec]'); ylabel('Amplitude');
title("Audio of Guitar in Floyd");


figure(6)
pcolor(tau,ks,Sgt_spec(1:length(ks),:))
shading interp
set(gca, 'ylim', [200 1000], 'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (k)')
title("Spectrogram for Floyd Guitar", 'Fontsize', 14);




