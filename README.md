# Q-Trader analysis

An implementation of Q-learning starting upon [Edward Lu's repository](https://github.com/edwardhdlu/q-trader).

The main purpose of this repository is not to directly build a trading bot model, but split out some parts of the provess and being able to systematize the analysis and comparison of multiple models, datasets and hyperparameters.

The final outcome of the training can be used to perform actual trading processes, in any case use it at your own responsibility, as the outcome of this model is by no mean a trading advice whatsoever.

## Installation

The Trading bot requires Python 3 and relies on Keras + Tensorflow as a Deep Learning backend. It also it requires to have installed the C++ TA-Lib in case that pip doesn't install it already.

In order to install it, clone that repository and create a virtual environment:

```
$ git clone (repo url)
$ cd q-trader
$ virtualenv -p python3 ./venv/
```

Then activate the virtual environment and install the required packages:

```
$ source venv/bin/activate
...
(venv) $ pip install -r requirements.txt
```

## Getting (or generating) a dataset

Afterwards it's required to get a dataset with the following columns:

* time: an integer counter, whether it's a timestamp or an incremental value.
* open: First price of the trade during the period.
* close: price of the last trade during the period.
* high: maximum trade price during the period.
* low: minimum trade price during the period.
* volume: overall volume of the trades performed during that period

### Generate random dataset

If you can't find one, you can generate a simple one by either downloading the sample datasets from the original repository, or by generating a random walk through the python console:

```
(venv) $ python random-walk.py
```

It will generate the file ./data/random-walk-1000.csv, which includes 1000 rows of random walk close data into an OHCLV dataset.

### Getting a dataset from an external service

You can also download a training and test csv files from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) into `data/`. Please take into account that after downloadeind the dataset


## Running the code

### Model training

Taking as a sample the random walk dataset, it's possible to train the model:

```
(venv) $ python train_dataframe_ta.py --stock=random-walk-1000 --window-size=50 --episodes=10 --model=baseline --batch-size=5
```

The allowed arguments for that release are:

* stock: Dataset to be used.
* episodes: How many training cycles will be run over the training dataset.
* model: List of columns that will be taken into account to train the neural network model.
* window-size: How many elements in the training dataset are used to learn its effects.
* batch-size: How many actions are taken after every episode in order to _replay_ the actions and actually train the model.


### Testing / Evaluating the model

The cross-validation of the model requires to have a CSV file having the same name as the stock but with the "-cv" suffix.  For a quick test, you can simply clone the same random-walk-1000 value to another file:

```
(venv) $ cp data/random-walk-1000.csv data/random-walk-1000-cv.csv
```

Now you can run the cross-validation over the generated model:

```
(venv) $ python  cv_dataframe_ta.py --stock=random-walk-1000 --window-size=50 --episodes=1 --model=baseline --batch-size=5
```

In this case, the argument `episodes` tells which is the version of the model (at which episode was trained) that will be used for the cross-validation. This allows to check the performance in each step.

## Output data

The training stage stores each episode outcome in a different file, into the output folder. It generates a subfolder named after the name of the stock and their arguments. For instance the previous command:

```
(venv) $ python train_dataframe_ta.py --stock=random-walk-1000 --window-size=50 --episodes=10 --model=baseline --batch-size=5
```

Will generate the file `output/models/baseline/model-w50-ep-1.model`, and it will also

It also stores the outcome of each step and episode during the training at `output/transitions/baseline/transitions-random-walk-1000-50-e001.csv`, which may be used mainly for debugging purposes.

## References

[Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) - Q-learning overview and Agent skeleton code