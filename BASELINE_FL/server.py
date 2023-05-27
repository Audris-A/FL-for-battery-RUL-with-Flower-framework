import flwr as fl
from customFedAvg import CustomFedAvg
import sys

round_count = int(sys.argv[1])

strategy = CustomFedAvg()

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=round_count), strategy=strategy)


