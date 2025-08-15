## Process the Data

Download the raw data (timeline_*.txt) and processed data (*_dx_1st_seq_OS.csv) from cBioPortal (MSK-CHORD (MSK, Nature 2024)).  
Then use SQLiteStudio to load all the data and run 'process_raw_data_in_sqlite.sql'.  
You should get a view called '*_survival_lstm_dataset'
