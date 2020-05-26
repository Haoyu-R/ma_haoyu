# Framwork for driving scenarios identification in measurement data (Master Thesis)

This framework aims at identify highway scenarios in real measurement data, which is based on a combination of the state machine, convolutional neural network (CNN), and recurrent neural network (RNN). 
It offers a standard procedure, from data cleaning to data visualization and dataset building, culminating in neural network modeling and parameter tuning. 
Furthermore, the trained neural network can be activated upon detecting one of the driving scenarios that has been conducted recently. 
The framwork is designed/tailored for recorded CAN-Bus data collected by the test ﬂeet.
The files in this repo also includes some other helper functions: e.g. 
comparing diﬀerent structures of the neural network, converting GPS signal to KML, analyzing the predictive power of each signal for scenarios.


## Code overview

1. The files for data preprocessing
  * process_main.py
  * NN_process_main.py
  * NN_preprocess_utils.py
  * NN_\*_dataset.py (Dataset construction for different configurations)
  * ...
  
2. The files for neural network training
  * NN_\*_training.py (* can be replaced by both, cut_in, lane_change, which correspondes to different training configuration during debug)
  * ...
  
3. The files for Neural network after processing
  * NN_\*_check_one_sample.py 
  * NN_\*_feature_selection.py
  * ...
  
4. The files for data visualization
  * visualization_main.py
  * visualization_utils.py
  * plot_*.py
  * ...
  
5. Other helper functions
  * extract_GPS_2_kml.py
  * split_csv_in_chunks.py
  * choose_section_from_csv.py
  * ...

