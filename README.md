# Overzealous Oracle

This project is an exploration into using Keras and Tensorflow to analyse financial data and attempt to make useful predictions. I understand that this may be a fundamentally impossible goal, but nevertheless it's a fun project to explore regression techniques! I've applied it to both traditional stock markets (ASX) and cryptocurrency markets.

Explores:
* Data collection and preparation
* Feature extraction
* Short, medium, and/or long term predictions
* Convolutional layers
* WaveNet (originally by Google)
* LSTM and GRU recurrent neural networks
* Regularization, inc. dropout
* Early stopping, reverting
* Loss functions, metrics, fitness
* Optimizers, learning rates
* Keras Tuner hyperparameter search
* Custom hyperparameter comparisons

This repo is primarily a personal project so it's as clean or user friendly as it would otherwise be. 

The [configuration file](/scripts/Config_CC.py) gives an idea of the broad capabilities of the system.  

See the [example output](/ExampleOutput.pdf) for a rough idea of what is printed for a single test.

### Installing
I use venv.
```
python3 -m venv ./venv
.\venv\Scripts\activate.bat
python3 -m pip install -r requirements.txt
```

Run [Main_CC.py](/scripts/Main_CC.py) in Jupyter. Using VSCode.  
I obtained crypto market historical data from Kraken. See the function ReadKrakenCsv in [Crypto_GetData.py](/scripts/Crypto_GetData.py).


Dean Reading
