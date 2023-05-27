import numpy as np
import matplotlib.pyplot as plt
import ast
import sys


clients_in_setting = int(sys.argv[1])
rounds_in_setting = int(sys.argv[2])
setting_run = int(sys.argv[3])

# FL settings
log_file_1="test_logs/" + str(clients_in_setting) + "_client_setup_logs/" + str(rounds_in_setting) + "_round_setup_logs/BASELINE_FL/server_output_run_" + str(setting_run) + ".txt"
log_file_2="test_logs/" + str(clients_in_setting) + "_client_setup_logs/" + str(rounds_in_setting) + "_round_setup_logs/DP_HE_FL/server_output_run_" + str(setting_run) + ".txt"
log_file_3="test_logs/" + str(clients_in_setting) + "_client_setup_logs/" + str(rounds_in_setting) + "_round_setup_logs/HOMOMORPHIC_ENCRYPTION/server_output_run_" + str(setting_run) + ".txt"
log_file_4="test_logs/" + str(clients_in_setting) + "_client_setup_logs/" + str(rounds_in_setting) + "_round_setup_logs/DP_FL/server_output_run_" + str(setting_run) + ".txt"

## Base setting
#
# Always use this setting to compare base to other methods
base_model_file="base_model_test_results/base_model_test_1_run_output_log.txt"

# TODO: speed this up by reading from the end
def get_metrics(path_to_file):
    with open(path_to_file, "r") as log_file_d:
        process_next_line = False

        metric_info = []
        loss_info = None
        FL_time_info = None

        mse = []
        mae = []
        mape = []
        rmse = []

        for line in log_file_d:

            if "aggregate_evaluate eval_metrics" in line:
                process_next_line = True
            elif process_next_line:
                metric_info.append(ast.literal_eval(line.split("|")[2][1:]))
                process_next_line = False
            
            if "FL finished in" in line:
                FL_time_info = float(line.split("|")[2].split(" ")[4][:-1])
                print(FL_time_info)

            if "losses_distributed" in line:
                # TODO: maybe think of a way without column length usage
                loss_info = ast.literal_eval(line.split("|")[2][29:])

        ## For aggr values:
        for round_metrics in metric_info:
            mse_avg = 0
            mae_avg = 0
            mape_avg = 0
            rmse_avg = 0

            for client_metric in round_metrics:
                mse_avg += client_metric[1]["mse"]
                mae_avg += client_metric[1]["mae"]
                mape_avg += client_metric[1]["mape"]
                rmse_avg += client_metric[1]["rmse"]

            mse_avg /= len(round_metrics)
            mae_avg /= len(round_metrics)
            mape_avg /= len(round_metrics)
            rmse_avg /= len(round_metrics)

            mse.append(mse_avg)
            mae.append(mae_avg)
            mape.append(mape_avg)
            rmse.append(rmse_avg)
        
        losses_by_rounds = []
        for client_loss in loss_info:
            losses_by_rounds.append(client_loss[1])
        
        return {
                "loss": losses_by_rounds,
                "mse" : mse,
                "mae" : mae,
                "mape" : mape,
                "rmse" : rmse,
                "FL time" : FL_time_info
            }

def get_base_metrics(path_to_file):
    with open(path_to_file, "r") as log_file_d:
        metric_dict = None
        
        for line in log_file_d:

            if "Max rmse" in line:
                metric_dict = ast.literal_eval(metric_dict)
                break

            metric_dict = line

        return {
                "loss": metric_dict["loss"],
                "mse" : metric_dict["mse"],
                "mae" : metric_dict["mae"],
                "mape" : metric_dict["mape"],
                "rmse" : metric_dict["rmse"],
                "FL time" : 0
            }


# We plot the average client metric value for each round
def plot_specific_metric(setting_results, metric_name, rounds_in_setting, clients_in_setting, graph_offset=0.15, offset_helper=0): # 3 0.15
    fig, axs = plt.subplots(1, 1, figsize=(6, 6), sharey=True)
    axs.set_title(metric_name + " - " + str(rounds_in_setting) + " round setting with " + str(clients_in_setting) + " clients in run " + str(setting_run))

    x = np.arange(rounds_in_setting if metric_name != "FL time" else 1)
    width = 0.15   # the width of the bars
    multiplier = 0
    for key in setting_results:
        if (metric_name == "FL time" and key != "base_model") or (metric_name != "FL time"):
            offset = width * multiplier-offset_helper
            rects = axs.bar(x + offset, setting_results[key][metric_name], width, label=key)
            multiplier += 1

            plt.xticks(np.arange(min(x), max(x)+1, 1.0))
            
            axs.set_ylabel(metric_name)
            axs.set_xlabel("round")

    axs.legend()
    
    axs.set_xticks(x + width + graph_offset, [round_it for round_it in range(0, rounds_in_setting if metric_name != "FL time" else 1)])

    # for debugging
    #fig.savefig("log_processed_data/" + metric_name + "output.png")

    fig.savefig("log_processed_data/" + str(clients_in_setting) + "_clients/" + metric_name + "_clients" + str(clients_in_setting) + "_rounds" + str(rounds_in_setting) + "_run" + str(setting_run) + ".png")

result_dict = {
    "base_fl" : None,
    "dp_he_fl" : None,
    "he_fl" : None,
    "dp_fl" : None
}

result_dict["base_model"] = get_base_metrics(base_model_file)
result_dict["base_fl"] = get_metrics(log_file_1)
result_dict["dp_he_fl"] = get_metrics(log_file_2)
result_dict["he_fl"] = get_metrics(log_file_3)
result_dict["dp_fl"] = get_metrics(log_file_4)


plot_specific_metric(result_dict, "loss", rounds_in_setting, clients_in_setting)
plot_specific_metric(result_dict, "mse", rounds_in_setting, clients_in_setting)
plot_specific_metric(result_dict, "mae", rounds_in_setting, clients_in_setting)
plot_specific_metric(result_dict, "mape", rounds_in_setting, clients_in_setting)
plot_specific_metric(result_dict, "rmse", rounds_in_setting, clients_in_setting)
plot_specific_metric(result_dict, "FL time", rounds_in_setting, clients_in_setting, 0.06, 0.011)
