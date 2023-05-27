for run_it in $(seq 1 $1)
do
    # replace long with ${run_it}
    python base_model.py $run_it > base_model_test_${run_it}_run_output_log.txt 2>&1 & base_pid=$!
    pidstat -h -r -u -p $base_pid 1 > base_model_test_${run_it}_run_usage_log.txt

    wait $base_pid
    echo Run ${run_it} is DONE!
done