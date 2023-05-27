import numpy as np
import matplotlib.pyplot as plt
from datetime import time
import sys

## Base setting
#
# Always use this setting to compare base to other methods
#base_model_file="base_model_test_results/base_model_test_2_run_usage_log.txt"
base_model_file="base_model_test_long_run_usage_log.txt"

processed_log_path="processed_one_logs.txt"
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
                        #print(metric_fig_y[key])
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

def get_latex_table_format_data(data):
    base_str = ""
    dp_fl_str = ""
    he_fl_str = ""
    dp_he_fl_str = ""
    for col in data:
        base_str += str(int(col[0])) + " & "
        dp_fl_str += str(int(col[1])) + " & "
        he_fl_str += str(int(col[2])) + " & "
        dp_he_fl_str += str(int(col[3])) + " & "
    
    return base_str[:-2], dp_fl_str[:-2], he_fl_str[:-2], dp_he_fl_str[:-2]

# data_dict_1 = process_a_log_file(log_file_1)
# data_dict_2 = process_a_log_file(log_file_2)
# data_dict_3 = process_a_log_file(log_file_3)
# data_dict_4 = process_a_log_file(log_file_4)

cpus = []
rams = []
for client in range(2, 9):
    baseline_fl_cpu_sum = 0
    dp_fl_cpu_sum = 0
    he_fl_cpu_sum = 0
    dp_he_fl_cpu_sum = 0

    baseline_fl_ram_sum = 0
    dp_fl_ram_sum = 0
    he_fl_ram_sum = 0
    dp_he_fl_ram_sum = 0

    cpus.append([])
    rams.append([])
    for d_client in range(0, 8):
        for t_round in range(1, 6):
            for t_run in range(1, 4):
                x= 1
                log_file_1="test_logs/" + str(client) + "_client_setup_logs/" + str(t_round) + "_round_setup_logs/BASELINE_FL/client_" + str(d_client) + "_usage_log_run_" + str(t_run) + ".txt"
                log_file_2="test_logs/" + str(client) + "_client_setup_logs/" + str(t_round) + "_round_setup_logs/DP_HE_FL/client_" + str(d_client) + "_usage_log_run_" + str(t_run) + ".txt"
                log_file_3="test_logs/" + str(client) + "_client_setup_logs/" + str(t_round) + "_round_setup_logs/HOMOMORPHIC_ENCRYPTION/client_" + str(d_client) + "_usage_log_run_" + str(t_run) + ".txt"
                log_file_4="test_logs/" + str(client) + "_client_setup_logs/" + str(t_round) + "_round_setup_logs/DP_FL/client_" + str(d_client) + "_usage_log_run_" + str(t_run) + ".txt"

                data_dict_1 = process_a_log_file(log_file_1)
                data_dict_2 = process_a_log_file(log_file_2)
                data_dict_3 = process_a_log_file(log_file_3)
                data_dict_4 = process_a_log_file(log_file_4)

                baseline_fl_cpu_sum += sum(data_dict_1[1]["values"])
                dp_fl_cpu_sum += sum(data_dict_4[1]["values"])
                he_fl_cpu_sum += sum(data_dict_3[1]["values"])
                dp_he_fl_cpu_sum += sum(data_dict_2[1]["values"])

                baseline_fl_ram_sum += sum(data_dict_1[2]["values"])
                dp_fl_ram_sum += sum(data_dict_4[2]["values"])
                he_fl_ram_sum += sum(data_dict_3[2]["values"])
                dp_he_fl_ram_sum += sum(data_dict_2[2]["values"])
        #print(d_client)
        if (d_client + 1) == client:
            break

    cpus[len(cpus)-1].append(baseline_fl_cpu_sum)
    cpus[len(cpus)-1].append(dp_fl_cpu_sum)
    cpus[len(cpus)-1].append(he_fl_cpu_sum)
    cpus[len(cpus)-1].append(dp_he_fl_cpu_sum)

    print("baseline_fl_cpu_sum =", baseline_fl_cpu_sum)
    print("dp_fl_cpu_sum =", dp_fl_cpu_sum)
    print("he_fl_cpu_sum =", he_fl_cpu_sum)
    print("dp_he_fl_cpu_sum =", dp_he_fl_cpu_sum)

    rams[len(rams)-1].append(baseline_fl_ram_sum)
    rams[len(rams)-1].append(dp_fl_ram_sum)
    rams[len(rams)-1].append(he_fl_ram_sum)
    rams[len(rams)-1].append(dp_he_fl_ram_sum)

    print("baseline_fl_ram_sum =", baseline_fl_ram_sum)
    print("dp_fl_ram_sum =", dp_fl_ram_sum)
    print("he_fl_ram_sum =", he_fl_ram_sum)
    print("dp_he_fl_ram_sum =", dp_he_fl_ram_sum)
    print("=======================")


base_cpu_str, dp_cpu_str, he_cpu_str, dp_he_cpu_str = get_latex_table_format_data(cpus)

print("CPU:")
print(base_cpu_str)
print(dp_cpu_str)
print(he_cpu_str)
print(dp_he_cpu_str)
print("===============")

base_ram_str, dp_ram_str, he_ram_str, dp_he_ram_str = get_latex_table_format_data(rams)

print("RAM:")
print(base_ram_str)
print(dp_ram_str)
print(he_ram_str)
print(dp_he_ram_str)
