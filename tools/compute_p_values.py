import torch
from scipy import stats

print('for COCOA val')
acc_1, iou_1 = torch.load('experiments/COCOA/p_values_results/default_no_rgb_val.pkl')
# acc_2, iou_2 = torch.load('experiments/COCOA/p_values_results/boundary_no_rgb_val.pkl')
# acc_2, iou_2 = torch.load('experiments/COCOA/p_values_results/boundary_no_rgb_mumford_shah_val.pkl')
# acc_2, iou_2 = torch.load('experiments/COCOA/p_values_results/std_no_rgb_mumford_shah_val.pkl')
acc_2, iou_2 = torch.load('experiments/COCOA/p_values_results/std_no_rgb_cross_entropy_gaussian_val.pkl')

acc_p_value = stats.ttest_ind(acc_1, acc_2)
iou_p_value = stats.ttest_ind(iou_1, iou_2)
print(acc_p_value, iou_p_value)

print('for COCOA test')
acc_1, iou_1 = torch.load('experiments/COCOA/p_values_results/default_no_rgb_test.pkl')
# acc_2, iou_2 = torch.load('experiments/COCOA/p_values_results/boundary_no_rgb_test.pkl')
# acc_2, iou_2 = torch.load('experiments/COCOA/p_values_results/boundary_no_rgb_mumford_shah_test.pkl')
# acc_2, iou_2 = torch.load('experiments/COCOA/p_values_results/std_no_rgb_mumford_shah_test.pkl')
acc_2, iou_2 = torch.load('experiments/COCOA/p_values_results/std_no_rgb_cross_entropy_gaussian_test.pkl')

acc_p_value = stats.ttest_ind(acc_1, acc_2)
iou_p_value = stats.ttest_ind(iou_1, iou_2)
print(acc_p_value, iou_p_value)


# print('for KINS test')
# iou_1, acc_1, inv_iou_1 = torch.load('experiments/KINS/p_values_results/default_no_rgb_test.pkl')
# # iou_2, acc_2 = torch.load('experiments/KINS/p_values_results/std_no_rgb_gaussian_test.pkl')
# # iou_2, acc_2 = torch.load('experiments/KINS/p_values_results/boundary_no_rgb_gaussian_test.pkl')
# iou_2, acc_2, inv_iou_2 = torch.load('experiments/KINS/p_values_results/std_no_rgb_gaussian_test.pkl')

# acc_p_value = stats.ttest_ind(acc_1, acc_2)
# iou_p_value = stats.ttest_ind(iou_1, iou_2)
# inv_iou_p_value = stats.ttest_ind(inv_iou_1, inv_iou_2)
# print(acc_p_value, iou_p_value, inv_iou_p_value)