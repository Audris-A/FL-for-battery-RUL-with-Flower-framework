import flwr as fl
from customFedAvg import CustomFedAvg
from DPFedAvgAdaptive import DPFedAvgAdaptive
from DPFedAvgFixed import DPFedAvgFixed
import sys

round_count = int(sys.argv[1])

num_sampled_clients = int(sys.argv[2])
noise_multiplier = float(sys.argv[3])

# We should concentrate on LDP because - False - 
#  the EV drivers probably would want to send private weights'
server_side_noising = True

## The most important DP results below are from server side DP noising 
# which were done after client side noising that is visible in the masters thesis.
# eps = 9.923967393585414 for clip_norm=0.024898126098117875, noise_multiplier=0.568 -> bad (unusable) precision, about 8x worse loss
# eps = 15129824133.326149 for clip_norm=14.142135623730953, noise_multiplier=0.001 -> good precision (usable) but not private, almost the same loss.
# This means that even with server side noising we cannot achieve reasonable precision with a privacy budget less then 10
# without changing the used model hyperparameters. But it is possible to use another DP or aggregation method to get better results. 
strategy = DPFedAvgFixed(CustomFedAvg(), num_sampled_clients=num_sampled_clients, server_side_noising=server_side_noising, clip_norm=14.142135623730953, noise_multiplier=0.001)

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=round_count), strategy=strategy)


