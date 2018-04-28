"""Utilitfy functions for train and test of qtrader bot."""
import logging
import numpy as np
import pandas as pd
import talib as tl

logging.basicConfig(filename='./qatrader.log')


def format_price(n):
    """Print formatted price."""
    return "{:+.2f} Eur".format(abs(n))


def get_cv_data(key, window_size=None, limit=None):
    """Return the crossvalidation dataset."""
    return get_train_data(
        key + "-cv",
        window_size=window_size,
        limit=limit
    )


def get_train_data(key, window_size=None, limit=None):
    """Return the vector containing stock data from a fixed file."""
    with open("data/" + key + ".csv", "r") as infile:
        lines = pd.read_csv(infile)


    if limit is not None:
        lines = lines.tail(limit).reindex()

    if 'high' in lines.columns and 'low' in lines.columns:
        lines['sar'] = tl.SAR(
            lines['high'].values,
            lines['low'].values,
            acceleration=0.025,
            maximum=0.05
        )

    lines['rsi14'] = tl.RSI(
        lines['close'].values,
        timeperiod=14
    )

    lines['rsi70'] = tl.RSI(
        lines['close'].values,
        timeperiod=70
    )

    lines['sma3'] = tl.SMA(
        lines['close'].values,
        timeperiod=3
    )

    lines['sma12'] = tl.SMA(
        lines['close'].values,
        timeperiod=12
    )

    lines['sma36'] = tl.SMA(
        lines['close'].values,
        timeperiod=36
    )

    lines['sma48'] = tl.SMA(
        lines['close'].values,
        timeperiod=48
    )

    lines['sma72'] = tl.SMA(
        lines['close'].values,
        timeperiod=72
    )

    lines['sma144'] = tl.SMA(
        lines['close'].values,
        timeperiod=144
    )

    lines['ema12'] = tl.EMA(
        lines['close'].values,
        timeperiod=12
    )

    lines['ema48'] = tl.EMA(
        lines['close'].values,
        timeperiod=48
    )

    lines['ema72'] = tl.EMA(
        lines['close'].values,
        timeperiod=72
    )

    lines['ema144'] = tl.EMA(
        lines['close'].values,
        timeperiod=144
    )

    # lines['sma288'] = tl.SMA(
    #     lines['close'].values,
    #     timeperiod=288
    # )

    lines['bb_lower'], \
        lines['bb_middle'], \
        lines['bb_upper'] = tl.BBANDS(
            lines['close'].values,
            timeperiod=24,
            nbdevup=2.0,
            nbdevdn=2.0,
            matype=0
    )

    lines['macd_val'], \
        lines['macd_signal'], \
        lines['macd_hist'] = tl.MACD(
            lines['close'].values,
            fastperiod=12,
            slowperiod=48,
            signalperiod=6
    )

    lines[[
        'sar',
        'rsi14',
        'rsi70',
        'sma3',
        'sma12',
        'sma36',
        'sma48',
        'sma72',
        'sma144',
        'ema12',
        'ema48',
        'ema72',
        'ema144',
        'bb_lower',
        'bb_middle',
        'bb_upper',
        'macd_val',
        'macd_signal',
        'macd_hist'
    ]] = lines[[
        'sar',
        'rsi14',
        'rsi70',
        'sma3',
        'sma12',
        'sma36',
        'sma48',
        'sma72',
        'sma144',
        'ema12',
        'ema48',
        'ema72',
        'ema144',
        'bb_lower',
        'bb_middle',
        'bb_upper',
        'macd_val',
        'macd_signal',
        'macd_hist'
    ]].fillna(0)

    lines['next_close'] = lines['close'].shift(-1).fillna(0)
    lines['prev_close'] = lines['close'].shift(1).fillna(0)

    lines['returns'] = (
        (lines['close'] - lines['prev_close']) /
        lines['prev_close']
    )
    lines['next_returns'] = (
        (lines['next_close'] - lines['close']) /
        lines['close']
    )

    lines['returns'].fillna(0, inplace=True)
    lines['next_returns'].fillna(0, inplace=True)

    # returns either -1 for sell or 1 for buy
    lines['ideal_decision'] = (
        lines['next_returns'] /
        (lines['next_returns'].map(np.abs))
    ).map(lambda x: 1 if x > 0 else 2)

    if window_size is not None:
        lines['done'] = False
        lines.loc[len(lines) - window_size - 2, 'done'] = True

        lines = lines.tail(
            len(lines) - window_size
        ).head(
            len(lines) - window_size - 2
        )

    return lines


def get_state(block):
    """Return an n-period state representation ending at time t.

    It is based on the sigmoid function:

    1 / (1 + e^-x)

    """
    block = np.negative(block)
    block = 1 + np.exp(np.array(block, dtype=np.float32))
    return 1 / (1 + block)


def train_state_ta(w, **kwargs):
    """Train the agent based on the input and output."""
    train_count = w.index.values[0]

    # print(kwargs['state'])

    #: At the beginning of the period we take a decision on how
    #: to act. At the end of the state we get the reward
    rnd_string, next_state_action, actions_scores = kwargs['agent'].act(kwargs['state'])

    #: Switching from the state at the beginning of the period
    #: (prev_state) to the state at the end of it (curr_state).
    prev_action = int(kwargs['curr_action'])
    curr_action = int(kwargs['curr_action']) \
        if next_state_action in \
        [
            kwargs['agent'].ACTION_SIT,
            kwargs['agent'].ACTION_WITHDRAW1,
            kwargs['agent'].ACTION_REFUND1
    ] else next_state_action

    #: If we buy BTC -> we get the earnings from the BTC reward,
    #: otherwise we'll get the EUR reward.
    kwargs['balance_value'] =  \
        kwargs['balance_fiat'] + \
        kwargs['balance_coin']

    stepwise_lossfunc_args = {
        'prev_action': prev_action,
        'curr_action': curr_action,
        'balance_value': kwargs['balance_value'],
        'balance_fiat': kwargs['balance_fiat'],
        'balance_coin': kwargs['balance_coin'],
        'next_returns': w.tail(1)['next_returns'].values[0]
    }

    # rollwise_lossfunc_args = {
    #     'prev_state': prev_state,
    #     'curr_state': curr_state,
    #     'balance_value': kwargs['balance_value'],
    #     'balance_fiat': kwargs['balance_fiat'],
    #     'balance_coin': kwargs['balance_coin'],
    #     'next_returns': w.tail(1)['next_returns'].values[0]
    # }

    # rollwise_ema_lossfunc_args = {
    #     'prev_state': prev_state,
    #     'curr_state': curr_state,
    #     'balance_value': kwargs['balance_value'],
    #     'balance_fiat': kwargs['balance_fiat'],
    #     'balance_coin': kwargs['balance_coin'],
    #     'next_returns': w.tail(1)['next_returns'].values[0]
    # }

    change, \
        kwargs['balance_value'], \
        kwargs['balance_fiat'], \
        kwargs['balance_coin'] = loss_func(**stepwise_lossfunc_args)

    kwargs['balance_value_history'].append(
        kwargs['balance_value']
    )

    kwargs['balance_value_history'] = kwargs[
        'balance_value_history'
    ][-len(w):]

    print(
        "#{:03d}/{:05d}: "
        "Price: {:.2f} -> {:.2f} | "
        "[{}{}]: {} -> {} | "
        "chg: {:+.4f}%, rwd: {:.3f} | "
        "balance = {:+.2f}% (fiat {:+.2f}% + coin {:+.2f}%)".format(
            kwargs['epoch'],
            train_count,
            w.tail(1)['close'].values[0],
            w.tail(1)['next_close'].values[0],
            rnd_string,
            next_state_action,
            prev_action,
            curr_action,
            w.tail(1)['next_returns'].values[0] * 100,
            max(change, 0) * 10000,
            kwargs['balance_value'] * 100,
            kwargs['balance_fiat'] * 100,
            kwargs['balance_coin'] * 100
        )
    )

    model_cols = len(kwargs['features']) - len(kwargs['exclude_variables'])

    state_data = np.empty(
        len(w) * model_cols,
        dtype=np.float32
    )

    for col in range(model_cols):
        curr_col = kwargs['features'][col]
        if curr_col not in kwargs['exclude_variables']:
            state_data[col::model_cols] = w.loc[
                :,
                kwargs['features'][col]
            ].values

    next_state = np.array([
        get_state(
            state_data
        )
    ])

    return {
        'step': train_count,
        'action': next_state_action,
        'prev_action': prev_action,
        'curr_action': curr_action,
        'state': kwargs['state'],
        'next_state': next_state,
        'balance_fiat': kwargs['balance_fiat'],
        'balance_coin': kwargs['balance_coin'],
        'balance_value': kwargs['balance_value'],
        'change': change,
        'epsilon': kwargs['agent'].epsilon,
        'gamma': kwargs['agent'].gamma
    }


def loss_func(**lossfunc_args):
    """Generate the loss function based on the last step."""
    prev_balance_fiat = lossfunc_args['balance_fiat']
    prev_balance_coin = lossfunc_args['balance_coin']

    lossfunc_args['balance_fiat'] = prev_balance_fiat \
        if lossfunc_args['prev_action'] == lossfunc_args['curr_action'] \
        else prev_balance_coin

    lossfunc_args['balance_coin'] = prev_balance_coin \
        if lossfunc_args['prev_action'] == lossfunc_args['curr_action'] \
        else prev_balance_fiat

    balance_fiat = lossfunc_args['balance_fiat'] * \
        (1 - lossfunc_args['next_returns'])
    balance_coin = lossfunc_args['balance_coin'] * \
        (1 + lossfunc_args['next_returns'])
    balance_value = balance_fiat + balance_coin

    prev_balance = lossfunc_args['balance_value']

    change = (balance_value - prev_balance) / prev_balance

    return (
        change,
        balance_value,
        balance_fiat,
        balance_coin
    )

def xrange(a, b, c=1):
    """Shim for xrange in py3."""
    return range(a, b, c)
