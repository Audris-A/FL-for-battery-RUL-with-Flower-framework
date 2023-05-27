import flwr as fl
from customFedAvg import CustomFedAvg
from DPFedAvgAdaptive import DPFedAvgAdaptive
import sys

round_count = int(sys.argv[1])

num_sampled_clients = int(sys.argv[2])

noise_multiplier = float(sys.argv[3])

# We should concentrate on LDP because
#  the EV drivers probably would want to send private weights
server_side_noising = False

# Use DPFedAvgAdaptive in order to skip setting clip norm explicitly
strategy = DPFedAvgAdaptive(CustomFedAvg(), num_sampled_clients=num_sampled_clients, noise_multiplier=noise_multiplier, server_side_noising=server_side_noising)
 
fl.server.start_server(config=fl.server.ServerConfig(num_rounds=round_count), strategy=strategy)


