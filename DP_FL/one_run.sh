python server.py 1 2 0.01 > ../runs/server_output_dp_fl_test_hyper.txt 2>&1 &
python client.py ../data_preprocessing/test_setup_data/2_client_setup/x_data/client_1_x.npy \
                 ../data_preprocessing/test_setup_data/2_client_setup/y_data/client_1_y.npy \
                 30 \
                 > ../runs/client1_output_dp_fl_test_hyper.txt 2>&1 & \

python client.py ../data_preprocessing/test_setup_data/2_client_setup/x_data/client_1_x.npy \
                 ../data_preprocessing/test_setup_data/2_client_setup/y_data/client_1_y.npy \
                 30 \
                 > ../runs/client2_output_dp_fl_test_hyper.txt 2>&1 & \
