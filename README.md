# SaveEarthquakes
GPD implementation

The network model in network.py is used for training, testing and predicting. 
It is used in model.py, where the training and testing routines are implemented.
The pytorch dataloader classes with data preparation are included in the load.py file.
predict.py chooses a stream from a random event and station and predicts a label function, slided over the waveform stream.
It uses a learned network, generated after running model.py.

A similar implementation using pytorch lightning is included in all Lit*.py files.
In LitNetwork.py the network, training/test/validation-steps, the optimizer and data loading routines are defined. The pytorch dataloaders are however still included in the old load.py file.
LitModel.py consists of the lightning training routine, LitTest.py has an additional test routine (same as in the end of model.py), that loads a user defined model from checkpoints, and LitPredict.py is basically the same prediction routine as in predict.py, except it loads the save lightning checkpoint model as in LitTest.py

Thesis.pdf is my current thesis draft.
