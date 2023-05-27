import numpy as np
import matplotlib.pyplot as plt
from datetime import time
import sys

clients_in_setting = int(sys.argv[1])
rounds_in_setting = int(sys.argv[2])
setting_run = int(sys.argv[3])

# Using only one client for usage analysis
log_file_1="test_logs/" + str(clients_in_setting) + "_client_setup_logs/" + str(rounds_in_setting) + "_round_setup_logs/BASELINE_FL/client_0_usage_log_run_" + str(setting_run) + ".txt"
log_file_2="test_logs/" + str(clients_in_setting) + "_client_setup_logs/" + str(rounds_in_setting) + "_round_setup_logs/DP_HE_FL/client_0_usage_log_run_" + str(setting_run) + ".txt"
log_file_3="test_logs/" + str(clients_in_setting) + "_client_setup_logs/" + str(rounds_in_setting) + "_round_setup_logs/HOMOMORPHIC_ENCRYPTION/client_0_usage_log_run_" + str(setting_run) + ".txt"
log_file_4="test_logs/" + str(clients_in_setting) + "_client_setup_logs/" + str(rounds_in_setting) + "_round_setup_logs/DP_FL/client_0_usage_log_run_" + str(setting_run) + ".txt"

## Base setting
#
# Always use this setting to compare base to other methods
#base_model_file="base_model_test_results/base_model_test_2_run_usage_log.txt"
base_model_file="base_model_test_long_run_usage_log.txt"

skip_lines=2

RAM_LIMIT = 40 # gb

metric_fig_y = {"Time" : "Time", 
                "%CPU" : "CPU, cores", 
                "%MEM" : "RAM, GB"
                }

def process_a_log_file(path_to_file):
    data_dict = []
    with open(path_to_file, "r") as log_file_d:
        lines_seen=0
        for line in log_file_d:
            if skip_lines < lines_seen:

                if line[0] != "#" and len(line) > 2:
                    split_line = " ".join(line.split())
                    split_line = split_line.split(" ")
                    
                    # Add time
                    data_dict[0]["values"].append(time.fromisoformat(split_line[0]))

                    # Add %CPU
                    data_dict[1]["values"].append(float(split_line[7])/100)

                    # Add %MEM
                    data_dict[2]["values"].append(RAM_LIMIT * (float(split_line[13])/100))

            elif skip_lines == lines_seen:
                split_line = " ".join(line[2:].split())
                #print(split_line)
                split_line = split_line.split(" ")

                for key in split_line:
                    if key in ["Time", "%CPU", "%MEM"]: 
                        print(metric_fig_y[key])
                        data_dict.append(
                            {
                                "name": metric_fig_y[key],
                                "file_addition": key,
                                "values": []
                            }
                        )

                lines_seen += 1
            elif skip_lines > lines_seen:
                lines_seen += 1
    
    return data_dict

data_dict_1 = process_a_log_file(log_file_1)
data_dict_2 = process_a_log_file(log_file_2)
data_dict_3 = process_a_log_file(log_file_3)
data_dict_4 = process_a_log_file(log_file_4)

data_dict_5 = process_a_log_file(base_model_file)


for data_dict_obj_it in range(1, len(data_dict_1)):
    fig, axs = plt.subplots(1, 1, figsize=(9, 5), sharey=True)
    metric_name = data_dict_1[data_dict_obj_it]["name"]
    file_addition = data_dict_1[data_dict_obj_it]["file_addition"]
    print(metric_name)
    # plot first client
    first_sample_len = len(data_dict_1[data_dict_obj_it]["values"])
    print(first_sample_len)
    axs.plot([x for x in range(0,first_sample_len)], data_dict_1[data_dict_obj_it]["values"])
    
    # second
    second_sample_len = len(data_dict_2[data_dict_obj_it]["values"])
    print(second_sample_len)
    axs.plot([x for x in range(0,second_sample_len)], data_dict_2[data_dict_obj_it]["values"])

    # third
    third_sample_len = len(data_dict_3[data_dict_obj_it]["values"])
    print(third_sample_len)
    axs.plot([x for x in range(0,third_sample_len)], data_dict_3[data_dict_obj_it]["values"])

    # fourth
    fourth_sample_len = len(data_dict_4[data_dict_obj_it]["values"])
    print(fourth_sample_len)
    axs.plot([x for x in range(0,fourth_sample_len)], data_dict_4[data_dict_obj_it]["values"])

    # five
    five_sample_len = len(data_dict_5[data_dict_obj_it]["values"])
    print(five_sample_len)
    axs.plot([x for x in range(0,five_sample_len)], data_dict_5[data_dict_obj_it]["values"])
    print(data_dict_5[data_dict_obj_it]["name"])
    axs.set_ylabel(data_dict_5[data_dict_obj_it]["name"])
    axs.set_xlabel("time, s")

    fig.suptitle(metric_name + " - " + str(clients_in_setting) + " client, " + str(rounds_in_setting) + " round setup")

    plt.legend(['BASELINE_FL', 'DP_HE_FL', 'HOMOMORPHIC_ENCRYPTION', 'DP_FL', 'BASE'])

    # For debugging
    #fig.savefig("log_processed_data/" + file_addition + "usage.png")

    fig.savefig("log_processed_data/" + metric_name +"_results" + "_clients" + str(clients_in_setting) + "_rounds" + str(rounds_in_setting) + "_run" + str(setting_run) + ".png")

    print("Processed " + metric_name + " !!")