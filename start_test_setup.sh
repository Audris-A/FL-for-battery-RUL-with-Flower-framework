base_fl_path=BASELINE_FL/
dp_fl_path=DP_FL/
he_fl_path=HOMOMORPHIC_ENCRYPTION/
dp_he_fl_path=DP_HE_FL/
test_setup_type=$1

echo "Start time"
date

CNN_base_epoch_count=60 #3000

echo $test_setup_type

test_path=""

case $test_setup_type in
  "base_fl")
    test_path=$base_fl_path
    ;;
  "dp_fl")
    test_path=$dp_fl_path
    ;;
  "he_fl")
    test_path=$he_fl_path
    ;;
  "dp_he_fl")
    test_path=$dp_he_fl_path
    ;;
  *)  
    echo "Unrecognized setup type"
    exit 
    ;;  
esac

echo $test_path

cd $test_path

noise_multiplier=0.000
max_run_count=3
max_client_count=8
max_round_count=5
for run_it in $(seq 1 $max_run_count)
do
    echo "Run " $run_it
    for client_count in $(seq 2 $max_client_count)
    do
        echo "    client " $client_count
        
        client_data_path=../data_preprocessing/test_setup_data/${client_count}_client_setup/
        
        for round_count in $(seq 1 $max_round_count)
        do
            let "client_epoch_count=$CNN_base_epoch_count / $round_count"

            echo "        round " $round_count ", client=" $client_count ", run=" $run_it

            while IFS=, read -r field1 field2 field3
            do  
                # fixes the expression error in the if statement
                typeset -i nm_cl=$field1
                typeset -i nm_ro=$field2

                # Select the appropriate noise multiplier
                if [ $nm_cl -eq $client_count ] && [ $nm_ro -eq $round_count ] 
                then
                    noise_multiplier=$field3
                    break
                fi
            done < ../noise_multiplier_analysis.txt

            ## Start server here
            #
            python server.py $round_count $client_count $noise_multiplier > ../test_logs/${client_count}_client_setup_logs/${round_count}_round_setup_logs/${test_path}server_output_run_${run_it}.txt 2>&1 & server_pid=$!

            pidstat -h -r -u -p $server_pid 1 > ../test_logs/${client_count}_client_setup_logs/${round_count}_round_setup_logs/${test_path}server_usage_log_run_${run_it}.txt 2>&1 &

            echo "        server pid = " $server_pid
            ## Start clients here
            #
            CLIENT_PIDS=()
            let "test_setup_client_count=$client_count - 1"
            for client in $(seq 0 $test_setup_client_count)
            do
                python client.py ${client_data_path}x_data/client_${client}_x.npy \
                                            ${client_data_path}y_data/client_${client}_y.npy \
                                            $client_epoch_count \
                                            > ../test_logs/$(($test_setup_client_count + 1))_client_setup_logs/${round_count}_round_setup_logs/${test_path}client_${client}_output_run_${run_it}.txt 2>&1 & client_pid=$! \
                
                echo "            " ${client_data_path}x_data/client_${client}_x.npy
                echo "            " ${client_data_path}y_data/client_${client}_y.npy
                echo "            " epochs = $client_epoch_count
                echo "              Client " $client "pid = " $client_pid
                
                CLIENT_PIDS+=($client_pid)
                
                pidstat -h -r -u -p $client_pid 1 > ../test_logs/$(($test_setup_client_count + 1))_client_setup_logs/${round_count}_round_setup_logs/${test_path}client_${client}_usage_log_run_${run_it}.txt 2>&1 &
            done

            echo "        Waiting for server with pid" $server_pid "..."
            wait $server_pid

            for client_pid in ${!CLIENT_PIDS[@]}
            do
              echo "        Waiting for client with pid " ${CLIENT_PIDS[$client_pid]} "..."
              wait ${CLIENT_PIDS[$client_pid]}
            done

            ps ax | grep python
        done
    done
done

echo "End time"
date

printf "\n\n TESTS ARE DONE! :)\n\n"
