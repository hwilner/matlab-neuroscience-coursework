clc
clear all
close all
%% Initialize decision parameters
p0w1 = 1;
p0w2 = 1;
alpha = 10^-4;
beta = 10^-3;

%% Get letter distributions
w1_vector = extract_letters ('English.txt');
w2_vector = extract_letters ('French.txt');

Pw1 = hist(w1_vector, 1:26)/sum(hist(w1_vector, 1:26));
Pw2 = hist(w2_vector, 1:26)/sum(hist(w2_vector, 1:26));

%% Sequential decision of test data
% get test data
%X = extract_letters ('test.txt');
X = extract_letters ('dutch.txt');
% calculate decision thresholds 
A = log10((1-alpha)/beta);
B = log10(alpha/(1-beta));


% SPRT process 
trials = 1;
aOrB = log10(p0w1);

while ((((B<aOrB )&&(A>aOrB))||(trials>length(X))))
   pFr = Pw1(X(trials));
   pEng = Pw2(X(trials));
   LR = log10(pFr/pEng);
   aOrB = aOrB + LR;
   LR_Vec(trials) = aOrB;
   trials = trials + 1;
end
%%
% plot likelihood ratio as function of time
figure
plot(LR_Vec);
ylabel('LR'); xlabel('trials');
hold on
plot(ones(size(LR_Vec))*A,'--r');
plot(ones(size(LR_Vec))*B,'--m');
axis([0, length(LR_Vec), -(max(abs(A),abs(B))+1), max(abs(A),abs(B))+1]);

%% DKL
%we used: log10(a/b) = loga-logb 
DKL1 = sum(Pw1.*(log10(Pw1)-log10(Pw2)));
DKL2 = sum(Pw2.*(log10(Pw1)-log10(Pw2)));

sampels1 = round((A-log10(p0w1/p0w2))/DKL1)
sampels2 = round((B-log10(p0w1/p0w2))/DKL2);

