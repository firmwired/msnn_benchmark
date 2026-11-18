% Plain Default MATLAB Plotting Script
clear; clc; close all;

if isfile('benchmark_results.mat')
    load('benchmark_results.mat');
elseif isfile('Matlab/benchmark_results.mat')
    load('Matlab/benchmark_results.mat');
else
    error('benchmark_results.mat not found.');
end

folds         = double(folds(:)');
noisy_acc     = double(noisy_acc(:)');
ideal_acc     = double(ideal_acc(:)');
lstm_acc      = double(lstm_acc(:)');
noisy_l1      = double(noisy_l1(:)');
noisy_l2      = double(noisy_l2(:)');
input_spikes  = double(input_spikes(:)');
all_noisy_acc = double(all_noisy_acc);
all_ideal_acc = double(all_ideal_acc);

% Reset to standard MATLAB defaults
reset(0);

% Hardware calculations
T_steps = 30; N_in = 288; N_h = 128; N_out = 3;
synops_l1_fold    = N_in * T_steps .* (input_spikes / 100) * N_h;
synops_l2_fold    = N_h  * T_steps .* (noisy_l1 / 100)     * N_out;
synops_total_fold = synops_l1_fold + synops_l2_fold;

e_xbar_fold  = synops_total_fold * 1.0e-3;
e_adc_fold   = (N_h + N_out) * T_steps * 8.0e-3 * ones(size(folds));
e_lif_fold   = (N_h + N_out) * T_steps * 2.5e-3 * ones(size(folds));
e_leak_fold  = 0.225 * ones(size(folds));
e_total_fold = e_xbar_fold + e_adc_fold + e_lif_fold + e_leak_fold;

mean_synops = mean(synops_total_fold);
mean_e_xbar = mean(e_xbar_fold);
mean_e_adc  = mean(e_adc_fold);
mean_e_lif  = mean(e_lif_fold);
mean_e_leak = mean(e_leak_fold);
mean_e_tot  = mean(e_total_fold);


%% Figure 1: Accuracy Across Folds
figure(1);
subplot(1, 2, 1);
plot(folds, ideal_acc, '-s', 'LineWidth', 1.5); hold on;
plot(folds, noisy_acc, '-o', 'LineWidth', 1.5);
plot(folds, lstm_acc,  '-^', 'LineWidth', 1.5);
yline(33.33, '--k');
hold off;
grid on;
xlabel('Fold');
ylabel('Test Accuracy (%)');
title('(a) Test Accuracy Across Folds');
xticks(1:9);
ylim([25, 70]);
legend('MSNN Ideal', 'MSNN Noisy', 'Digital LSTM', 'Random Chance (33.3%)', 'Location', 'northwest');

subplot(1, 2, 2);
acc_drop = ideal_acc - noisy_acc;
bar(folds, acc_drop); hold on;
yline(mean(acc_drop), '--r', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Fold');
ylabel('Accuracy Drop (%)');
title('(b) Accuracy Drop (Ideal minus Noisy)');
xticks(1:9);
legend('Accuracy Drop', 'Mean Drop', 'Location', 'northwest');

exportgraphics(gcf, 'fig1_accuracy_resilience.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'fig1_accuracy_resilience.png', 'Resolution', 300);


%% Figure 2: Multi-Seed Results
mean_noisy = mean(all_noisy_acc, 1);
std_noisy  = std(all_noisy_acc, 0, 1);
mean_ideal = mean(all_ideal_acc, 1);
std_ideal  = std(all_ideal_acc, 0, 1);

figure(2);
subplot(2, 2, 1);
plot(folds, all_noisy_acc(1, :), '-o', 'LineWidth', 1.2); hold on;
plot(folds, all_noisy_acc(2, :), '-s', 'LineWidth', 1.2);
plot(folds, all_noisy_acc(3, :), '-^', 'LineWidth', 1.2);
plot(folds, all_noisy_acc(4, :), '-d', 'LineWidth', 1.2);
hold off;
grid on;
xlabel('Fold');
ylabel('Accuracy (%)');
title('(a) Noisy MSNN (4 Seeds)');
xticks(1:9);
ylim([40, 70]);
legend('Seed 123', 'Seed 42', 'Seed 7', 'Seed 3', 'Location', 'northwest');

subplot(2, 2, 2);
plot(folds, all_ideal_acc(1, :), '-o', 'LineWidth', 1.2); hold on;
plot(folds, all_ideal_acc(2, :), '-s', 'LineWidth', 1.2);
plot(folds, all_ideal_acc(3, :), '-^', 'LineWidth', 1.2);
plot(folds, all_ideal_acc(4, :), '-d', 'LineWidth', 1.2);
hold off;
grid on;
xlabel('Fold');
ylabel('Accuracy (%)');
title('(b) Ideal MSNN (4 Seeds)');
xticks(1:9);
ylim([40, 70]);
legend('Seed 123', 'Seed 42', 'Seed 7', 'Seed 3', 'Location', 'northwest');

subplot(2, 2, 3);
errorbar(folds, mean_ideal, std_ideal, '-s', 'LineWidth', 1.5); hold on;
errorbar(folds, mean_noisy, std_noisy, '-o', 'LineWidth', 1.5);
plot(folds, lstm_acc, '-^', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Fold');
ylabel('Accuracy (%)');
title('(c) Mean and Std Dev Across Seeds');
xticks(1:9);
ylim([35, 70]);
legend('Ideal MSNN (Mean +/- Std)', 'Noisy MSNN (Mean +/- Std)', 'Digital LSTM', 'Location', 'northwest');

subplot(2, 2, 4);
bar(folds, [std_noisy(:), std_ideal(:)]);
grid on;
xlabel('Fold');
ylabel('Std Dev (%)');
title('(d) Seed Standard Deviation per Fold');
xticks(1:9);
legend('Noisy Std Dev', 'Ideal Std Dev', 'Location', 'northeast');

exportgraphics(gcf, 'fig2_multiseed_stability.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'fig2_multiseed_stability.png', 'Resolution', 300);


%% Figure 3: Sparsity and Synaptic Operations
figure(3);
subplot(1, 2, 1);
plot(folds, input_spikes, '-o', 'LineWidth', 1.5); hold on;
plot(folds, noisy_l1,     '-s', 'LineWidth', 1.5);
plot(folds, noisy_l2,     '-^', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Fold');
ylabel('Spike Density (%)');
title('(a) Layer Spike Density Across Folds');
xticks(1:9);
ylim([0, 10]);
legend('Input Spikes', 'Layer 1 (Hidden)', 'Layer 2 (Output)', 'Location', 'northeast');

subplot(1, 2, 2);
bar(folds, synops_total_fold / 1e3); hold on;
yline(mean_synops / 1e3, '--r', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Fold');
ylabel('SynOps (Thousands)');
title('(b) Synaptic Operations per Window');
xticks(1:9);
legend('SynOps per Fold', 'Mean SynOps', 'Location', 'northeast');

exportgraphics(gcf, 'fig3_neuromorphic_sparsity.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'fig3_neuromorphic_sparsity.png', 'Resolution', 300);


%% Figure 4: Energy and Area Breakdown
lstm_macs   = 4190208;
e_lstm_int8 = lstm_macs * 0.2e-3;
e_lstm_fp32 = lstm_macs * 3.1e-3;
area_cells  = 2980;
area_macro  = 16554;
area_cmos   = 3420000;

figure(4);
subplot(2, 2, 1);
sub_vals = [mean_e_xbar, mean_e_adc, mean_e_lif, mean_e_leak];
bar(1:4, sub_vals);
grid on;
xticks(1:4);
xticklabels({'Crossbar', 'ADC', 'LIF', 'Leakage'});
ylabel('Energy (nJ)');
title('(a) MSNN Subsystem Energy Breakdown');

subplot(2, 2, 2);
e_stack = [e_xbar_fold(:), e_adc_fold(:), e_lif_fold(:), e_leak_fold(:)];
bar(folds, e_stack, 'stacked');
grid on;
xlabel('Fold');
ylabel('Energy (nJ)');
title('(b) Total Energy per Fold');
xticks(1:9);
ylim([0, 150]);
legend('Crossbar', 'ADC', 'LIF', 'Leakage', 'Location', 'northeast');

subplot(2, 2, 3);
bar(1:3, [mean_e_tot, e_lstm_int8, e_lstm_fp32]);
set(gca, 'YScale', 'log');
grid on;
xticks(1:3);
xticklabels({'MSNN', 'LSTM (INT8)', 'LSTM (FP32)'});
ylabel('Energy (nJ) [Log Scale]');
title('(c) Energy Comparison');

subplot(2, 2, 4);
bar(1:3, [area_cells, area_macro, area_cmos]);
set(gca, 'YScale', 'log');
grid on;
xticks(1:3);
xticklabels({'1T1R Cells', 'Full Macro', 'Digital LSTM'});
ylabel('Area (um^2) [Log Scale]');
title('(d) Area Comparison');

exportgraphics(gcf, 'fig4_energy_area_breakdown.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'fig4_energy_area_breakdown.png', 'Resolution', 300);


%% Figure 5: Summary Dashboard
figure(5);
subplot(2, 2, 1);
plot(folds, ideal_acc, '-s', 'LineWidth', 1.5); hold on;
plot(folds, noisy_acc, '-o', 'LineWidth', 1.5);
plot(folds, lstm_acc,  '-^', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Fold');
ylabel('Accuracy (%)');
title('(a) Directional Accuracy');
xticks(1:9);
legend('MSNN Ideal', 'MSNN Noisy', 'Digital LSTM', 'Location', 'northwest');

subplot(2, 2, 2);
plot(folds, input_spikes, '-o', 'LineWidth', 1.5); hold on;
plot(folds, noisy_l1,     '-s', 'LineWidth', 1.5);
plot(folds, noisy_l2,     '-^', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Fold');
ylabel('Spike Density (%)');
title('(b) Spike Density');
xticks(1:9);
legend('Input', 'Layer 1', 'Layer 2', 'Location', 'northeast');

subplot(2, 2, 3);
bar(1:4, sub_vals);
grid on;
xticks(1:4);
xticklabels({'Crossbar', 'ADC', 'LIF', 'Leakage'});
ylabel('Energy (nJ)');
title('(c) Energy Breakdown');

subplot(2, 2, 4);
mult_vals = [e_lstm_int8/mean_e_tot, lstm_macs/mean_synops, (147971*32)/74496, e_lstm_fp32/mean_e_tot, area_cmos/area_macro];
barh(1:5, mult_vals);
set(gca, 'XScale', 'log');
grid on;
yticks(1:5);
yticklabels({'Energy (INT8)', 'Operations', 'Memory', 'Energy (FP32)', 'Area'});
xlabel('Efficiency Factor (Log Scale)');
title('(d) Advantage vs Digital LSTM');

exportgraphics(gcf, 'fig5_master_dashboard.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'fig5_master_dashboard.png', 'Resolution', 300);
