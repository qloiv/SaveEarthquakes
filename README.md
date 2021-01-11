# SaveEarthquakes
GPD implementation

The network model in network.py is used for training, testing and predicting. 
It is used in model.py, where the training and testing routines are implemented.
The pytorch dataloader classes with data preparation are included in the load.py file.
predict.py chooses a stream from a random event and station and predicts a label function, slided over the waveform stream.

simpleLoad.py and simpleModel.py were used for creating the first model.
