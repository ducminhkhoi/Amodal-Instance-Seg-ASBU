import matplotlib.pyplot as plt
import numpy as np

test_set = 'val'

con0 = np.load(f'experiments/COCOA/stats/stat_default_no_rgb_{test_set}.npy')
con1 = np.load(f'experiments/COCOA/stats/stat_std_no_rgb_mumford_shah_{test_set}.npy')
con2 = np.load(f'experiments/COCOA/stats/stat_boundary_no_rgb_{test_set}.npy')
con12 = np.load(f'experiments/COCOA/stats/stat_boundary_no_rgb_mumford_shah_{test_set}.npy')

occ_rate = 1 - con0[:, 2]/con0[:, 3]

# sorted_indices = occ_rate.argsort()
# sorted_occ_rate = np.sort(occ_rate)
# sorted_con0_iou = con0_iou[sorted_indices]

labels = ['con0', 'con1', 'con2', 'con12']
num_bins = 11

for k, iou in enumerate([con0, con1, con2, con12]):
    con_iou = iou[:, 0] / iou[:, 1]

    avg_iou = [1.]
    range_values = np.linspace(0, 1, num_bins)
    for i in range(len(range_values)-1):
        if i == len(range_values) - 2:
            avg_iou.append(con_iou[(range_values[i] <= occ_rate) & (occ_rate <= range_values[i+1])].mean())
        else:
            avg_iou.append(con_iou[(range_values[i] <= occ_rate) & (occ_rate < range_values[i+1])].mean())

    plt.plot(range_values, avg_iou, label=labels[k])

plt.legend()
plt.savefig('experiments/COCOA/stats/occ_rate.jpg')
