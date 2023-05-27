import tensorflow_privacy as tfp
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp_sample_without_replacement, get_privacy_spent
import numpy as np
import sys

# In essence this is a (eps, delta)- privacy analysis by using Rényi Differential Privacy (RDP) 
#   by using the conversion from RDP to (eps, delta)-DP

# !! Smaller eps means larger privacy !!

# (Google: How to apply DP): In deep learning, epsilon choice is usually relaxed to ε ≤ 10.

# (One of the publications and Tensorflow): Delta should be less or equal to 1/training_dataset_length.

# (Google: How to apply DP): The following result allows for converting from (α, ε)-RDP to (ε′, δ) DP: for any α, ε and ε′ > ε (α, ε)-
#   RDP implies (ε, δ)-DP where δ = exp(−(α − 1)(ε′ − ε)) (Mironov, 2017; Abadi et al., 2016). Since this
#   result holds for all orders α, to obtain the best guarantees, the Moments Accountant needs to optimize over
#   continuous 1 < α < 32. Mironov (2017) however showed that the using only a restricted set of discrete α
#   values is sufficient to preserve the tightness of privacy analysis.
max_order = 32
orders = range(2, max_order + 1)

n = 15130 # n = 15130 for battery rul # total data set size # n = 60000 for MNIST model

q = 32 / n # batch_size / total data set size | 32 / n for battery rul

z = 0.49 # noise multiplier

delta = 1 / n 

## For single run tests
# rdp = compute_rdp_sample_without_replacement(q, z, n, orders)
# eps, _, _ = get_privacy_spent(rdp=rdp, orders=orders, target_delta=delta)
# print(eps)
# exit()

# Getting the noise multiplier values for each client setting
#  in order to be in the privacy budget (eps < 10) in csv format.
min_client_count = 2
max_client_count = 8
round_count = 5

noise_multiplier_step=0.001
print("client_setting,round_setting,noise_multiplier")

z_set_helper = np.float64(0.001) 
for client_setting in range(max_client_count, min_client_count - 1, -1):

    z_set = z_set_helper
    setting_data_size = n/client_setting
    q = 32 / setting_data_size
    delta = 1 / n
    for round_setting in range(1, round_count+1):
        while True:
            rdp = compute_rdp_sample_without_replacement(q, z_set, setting_data_size, orders)
            eps, _, _ = get_privacy_spent(rdp=rdp, orders=orders, target_delta=delta)
            if eps*float(round_setting) < 10.0:
                if client_setting == 2:
                    z_set_helper = z_set
                    
                z_set -= 0.001
                print(str(client_setting) + "," + str(round_setting) + "," + str(np.around(z_set, 3)))
                break
            else:
                z_set+=0.001