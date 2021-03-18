close all, clc, clear

% Load MNIST data 
[images, labels] = mnist_parse('train-images-idx3-ubyte','train-labels-idx1-ubyte');
[images_t, labels_t] = mnist_parse('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte');

% Perform analysis of data set
% SVD
vec = im2double(reshape(images,28*28,60000));
[m,n] = size(vec);
mn = mean(vec,2);
vec = vec - repmat(mn,1,n);
[U,S,V] = svd(vec,'econ');

vec_t = im2double(reshape(images_t,28*28,10000));
vec_t = vec_t - repmat(mn,1,10000);
[Ut,St,Vt] = svd(vec_t,'econ');

% Plot singular value spectrum
plot(diag(S),'ko','Linewidth',2)
set(gca,'Fontsize',14,'Xlim',[0,150])
ylabel('\sigma')
title('Singular Value Spectrum')

% Find the low-rank approximation
rank_5 = U(:,1:5)*S(1:5,1:5)*V(:,1:5)';
rank_15 = U(:,1:15)*S(1:15,1:15)*V(:,1:15)';
rank_20 = U(:,1:20)*S(1:20,1:20)*V(:,1:20)';
rank_25 = U(:,1:25)*S(1:25,1:25)*V(:,1:25)';

subplot(2,2,1)
plot_5 = reshape(rank_5(:,6),28,28);
imshow(plot_5)
title('Rank 5')
subplot(2,2,2)
plot_15 = reshape(rank_15(:,6),28,28);
imshow(plot_15)
title('Rank 15')
subplot(2,2,3)
plot_20 = reshape(rank_20(:,6),28,28);
imshow(plot_20)
title('Rank 20')
subplot(2,2,4)
plot_25 = reshape(rank_25(:,6),28,28);
imshow(plot_25)
title('Rank 25')

% Project three selected V-modes onto 3D
v_mode = U(:,[2,3,5])'*vec;

figure(2)
for j = 0:9
    loca = v_mode(:,find(labels == j));
    plot3(loca(1,:),loca(2,:),loca(3,:),'o'); hold on
end
%plot3(v_mode(1,:),v_mode(2,:),v_mode(3,:)); hold on
xlabel('2nd V-mode')
ylabel('3rd V-mode')
zlabel('5th V-mode')
legend('0','1','2','3','4','5','6','7','8','9')

%% Build a LDA for 2 digits
dig1 = 3;
dig2 = 5;

feature = 25;
num = U(:,1:feature)'*vec;
num_t = U(:,1:feature)'*vec_t;

digit0 = num(:,find(labels == 0));
digit1 = num(:,find(labels == 1));
digit2 = num(:,find(labels == 2));
digit3 = num(:,find(labels == 3));
digit4 = num(:,find(labels == 4));
digit5 = num(:,find(labels == 5));
digit6 = num(:,find(labels == 6));
digit7 = num(:,find(labels == 7));
digit8 = num(:,find(labels == 8));
digit9 = num(:,find(labels == 9));

digit_t0 = num_t(:,find(labels_t == 0));
digit_t1 = num_t(:,find(labels_t == 1));
digit_t2 = num_t(:,find(labels_t == 2));
digit_t3 = num_t(:,find(labels_t == 3));
digit_t4 = num_t(:,find(labels_t == 4));
digit_t5 = num_t(:,find(labels_t == 5));
digit_t6 = num_t(:,find(labels_t == 6));
digit_t7 = num_t(:,find(labels_t == 7));
digit_t8 = num_t(:,find(labels_t == 8));
digit_t9 = num_t(:,find(labels_t == 9));

%put into a cell array to store our data
cell = {digit0,digit1,digit2,digit3,digit4,digit5,digit6,digit7,digit8,digit9};
cell_t = {digit_t0,digit_t1,digit_t2,digit_t3,digit_t4,digit_t5,digit_t6, ...
    digit_t7,digit_t8,digit_t9};

train_dig1 = cell{dig1+1};
train_dig2 = cell{dig2+1};

test_dig1 = cell_t{dig1+1};
test_dig2 = cell_t{dig2+1};

size1 = size(train_dig1,2);
size2 = size(train_dig2,2);
%size3 = size(test_dig1,2);
%size4 = size(test_dig2,2);

% LDA calculation 
%scatter matrix
m1 = mean(train_dig1,2);
m2 = mean(train_dig2,2);

Sw = 0; % within class
for k = 1:size1
    Sw = Sw + (train_dig1(:,k)-m1)*(train_dig1(:,k)-m1)';
end

for k = 1:size2
    Sw = Sw + (train_dig2(:,k)-m2)*(train_dig2(:,k)-m2)';
end

Sb = (m1-m2)*(m1-m2)'; % between class

[V2,D] = eig(Sb,Sw);
[lambda,ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

%project onto w
vdig1 = w'*train_dig1;
vdig2 = w'*train_dig2;
vdig_t1 = w'*test_dig1;
vdig_t2 = w'*test_dig2;

% find thershold for classifier in train and test
if mean(vdig1) > mean(vdig2)
    w = -w;
    vdig1 = -vdig1;
    vdig2 = -vdig2;
    vdig_t1 = -vdig_t1;
    vdig_t2 = -vdig_t2;
end

% set thershold value
sort1 = sort(vdig1);
sort2 = sort(vdig2);
sort3 = sort(vdig_t1);
sort4 = sort(vdig_t2);

t1 = length(sort1);
t2 = 1;
while sort1(t1) > sort(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort1(t1) + sort2(t2))/2;

% plot histogram for two digit value in test
% figure(3)
% subplot(1,2,1)
% histogram(sort3,30); hold on, plot([threshold threshold],[0,800],'r')
% set(gca, 'Xlim',[-4 4],'Ylim',[0 800],'Fontsize',14)
% title('Selected digit 7')
% subplot(1,2,2)
% histogram(sort4,30); hold on, plot([threshold threshold], [0,800],'r')
% set(gca, 'Xlim',[-4 4],'Ylim',[0 800],'Fontsize',14)
% title('Selected digit 9')

% calculate the accuracy for test and train
%train
ResVec1 = (vdig1 > threshold);
err1 = ResVec1(:,find(ResVec1 == 1));
errNum1 = size(err1);
testNum1 = size(vdig1);
sucRate1 = 1-(errNum1/testNum1);

ResVec2 = (vdig2 < threshold);
err2 = ResVec2(:,find(ResVec2 == 1));
errNum2 = size(err2);
testNum2 = size(vdig2);
sucRate2 = 1-(errNum2/testNum2);

ResVec3 = (vdig_t1 > threshold);
err3 = ResVec3(:,find(ResVec3 == 1));
errNum3 = size(err3);
testNum3 = size(vdig_t1);
sucRate3 = 1-(errNum3/testNum3);
%test
ResVec4 = (vdig_t2 < threshold);
err4 = ResVec4(:,find(ResVec4 == 1));
errNum4 = size(err4);
testNum4 = size(vdig_t2);
sucRate4 = 1-(errNum4/testNum4);

% SVM classifier  

normal1 = train_dig1./max(train_dig1(:));
normal2 = train_dig2./max(train_dig2(:));
normal_t = num_t/max(num_t(:));
dig_lab1 = dig1.*ones(length(sort1),1);
dig_lab2 = dig2.*ones(length(sort2),1);
xtrain = [normal1 normal2];
xtrain = xtrain';
label = [dig_lab1; dig_lab2];
Mdl = fitcecoc(xtrain,label);
accu = resubLoss(Mdl);
normal_t = normal_t';
test_labels = predict(Mdl,normal_t);
CVMdl = crossval(Mdl);
genError = kfoldLoss(CVMdl);

% classification tree 

tree = fitctree(xtrain,label);
accu2 = predict(tree,normal_t);
