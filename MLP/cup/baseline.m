clear;
rng(1);

% Load the data
inputs = load('inputs.mat').MLCUP20TR{:, :};
outputs = load('outputs.mat').MLCUP20TR{:, :};

% Shuffle the data
perm = randperm(size(inputs, 1));
inputs = inputs(perm, :)';
outputs = outputs(perm, :)';

net = feedforwardnet([16, 16],'trainbfg');
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

[net, tr] = train(net,inputs, outputs, 'useParallel','yes');

train_outputs = net(inputs(:, 1:1200))';
train_targets = outputs(:, 1:1200)';

test_outputs = net(inputs(:, 1201:end))';
test_targets = outputs(:, 1201:end)';

fprintf('Train MEE = %d\n', MEE(train_outputs, train_targets))
fprintf('Test MEE = %d\n', MEE(test_outputs, test_targets))

% Plot loss function
%figure
%plotperform(tr)

% Scatter training set
figure

scatter(train_outputs(:, 1), train_outputs(:, 2), 2, 'MarkerEdgeColor',[1 0 0])
hold on
scatter(train_targets(:, 1), train_targets(:, 2), 2, 'MarkerEdgeColor',[0 1 0])
title('Scatter Training (and val.) data')
legend('Model','Test','Location','Best');

% Scatter test set
figure

scatter(test_outputs(:, 1), test_outputs(:, 2), 2, 'MarkerEdgeColor',[1 0 0])
hold on
title('Scatter test data')
scatter(test_targets(:, 1), test_targets(:, 2), 2, 'MarkerEdgeColor',[0 1 0])
legend('Model','Test','Location','Best');

function e = MEE(output, target)
    y1_output = output(:, 1);
    y2_output = output(:, 2);
    y1_target = target(:, 1);
    y2_target = target(:, 2);
    e = mean(sqrt((y1_target - y1_output).^2 + (y2_target - y2_output).^2));
end