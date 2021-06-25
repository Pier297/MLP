clear;
rng(1);

% Load the data
inputs = load('inputs.mat').MLCUP20TR{:, :};
outputs = load('outputs.mat').MLCUP20TR{:, :};

% Shuffle the data
perm = randperm(size(inputs, 1));
inputs = inputs(perm, :)';
outputs = outputs(perm, :)';

net = feedforwardnet([50, 50],'trainbfg');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'purelin';

% net.trainParam.lr = 0.00003; %learning rate for gradient descent alg.
% net.trainParam.mc = 0.5; %momentum constant
net.trainParam.epochs = 4000; %maximum number of epochs

tr_indices = 1:800; %indices used for training
tv_indices = 801:1200; %indices used for validation
ts_indices = 1201:1524; % indices used for *test*
net.divideFcn = 'divideind';
net.divideParam.trainInd = tr_indices;
% Test: Used for final assessment only
net.divideParam.testInd = ts_indices;
% Validation: Used for early stopping
net.divideParam.valInd = tv_indices;

[net, tr] = train(net,inputs, outputs);

% Plot loss function
figure
plotperform(tr)

% Scatter training set
figure
test_outputs = net(inputs(:, 1:800))';
real_outputs = outputs(:, 1:800)';
scatter(test_outputs(:, 1), test_outputs(:, 2), 2, 'MarkerEdgeColor',[1 0 0])
hold on
scatter(real_outputs(:, 1), real_outputs(:, 2), 2, 'MarkerEdgeColor',[0 1 0])
title('Scatter Training data')
legend('Model','Test','Location','Best');

% Scatter test set
figure
test_outputs = net(inputs(:, 1201:end))';
real_outputs = outputs(:, 1201:end)';
scatter(test_outputs(:, 1), test_outputs(:, 2), 2, 'MarkerEdgeColor',[1 0 0])
hold on
title('Scatter test data')
scatter(real_outputs(:, 1), real_outputs(:, 2), 2, 'MarkerEdgeColor',[0 1 0])
legend('Model','Test','Location','Best');