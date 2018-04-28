import sys
import os

from agent import Agent
from functions_dataframe import *
import models_features as mf

if len(sys.argv) != 4:
    print(sys.argv)
    print("Usage: python train.py [stock] [window] [episodes]")
    exit()

args = {
    'stock_name': '',
    'window_size': 50,
    'batch_size': 32,
    'episode_count': '',
    'selected_model': 'baseline'
}

exclude_variables = ['next_close', 'next_returns', 'done']

args['stock_name'], \
    args['window_size'], \
    args['episode_count'] = \
    sys.argv[1], \
    int(sys.argv[2]), \
    int(sys.argv[3])

agent = Agent(args['window_size'])
data = get_train_data(args['stock_name'])

# data = data.head(1000)

data['next_close'] = data['close'].shift(-1)
data['next_close_diff'] = (data['next_close'] - data['close']) / data['close']

data['prev_close'] = data['close'].shift(1)
data['returns_eur'] = (
    (data['close'] - data['prev_close']) /
    data['prev_close']
) + 1
data['returns_btc'] = (
    (1 / data['close'] - 1 / data['prev_close']) /
    (1 / data['prev_close'])
) + 1
data['close_diff'] = (data['close'] - data['prev_close']) / data['prev_close']
data['close_diff'].fillna(0, inplace=True)
data['diff'] = data['close'] - data['prev_close']

data['ideal_decision'] = data['next_close_diff'] / \
    (data['next_close_diff'].map(np.abs))
data['ideal_decision'] = data['ideal_decision'].fillna(0)

# print(data)

rows = len(data) - 1
batch_size = 32

model_dir = "output/models/{}".format(args['selected_model'])
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

model_cols = len(mf.models_features[args['selected_model']]['features']) - \
    len(exclude_variables)

input_features = [
    column for column in
    mf.models_features[args['selected_model']]['features']
    if column not in exclude_variables
]

for e in range(args['episode_count'] + 1):
    file_path = "{}/model_ep{}".format(model_dir, str(e))
    print(
        "******************************"
        "Episode {:06d}/{:06d}"
        "******************************".format(
            e,
            args['episode_count']
        )
    )

    if os.path.isfile(file_path):
        agent.load(file_path)
        print("Model already trained, loading it")
        continue

    roll_step = 0

    w = data.loc[
        (
            args['window_size'] + 1 + roll_step
        ):(
            2 * args['window_size'] + roll_step
        ),
        input_features
    ]

    print(
        len(w),
        len(w) * model_cols
    )

    state_data = np.empty(
        len(w) * model_cols,
        dtype=np.float32
    )

    for col in range(model_cols):
        curr_col = mf.models_features[args['selected_model']]['features'][col]
        if curr_col not in exclude_variables:
            state_data[col::model_cols] = w.loc[
                :,
                curr_col
            ].values

    state = np.array([
        get_state(state_data)
    ])

    state_transitions = pd.DataFrame({
        'step': [],
        'action': [],
        'prev_action': [],
        'curr_action': [],
        'state': [],
        'next_state': [],
        'balance_fiat': [],
        'balance_coin': [],
        'balance_value': [],
        'change': [],
        'epsilon': [],
        'gamma': []
    })

    curr_action = 2

    transition_args = {
        'epoch': e,
        'batch_size': args['batch_size'],
        'selected_model': args['selected_model'],
        'features': mf.models_features[args['selected_model']]['features'],
        'exclude_variables': exclude_variables,
        'agent': agent,
        'state': state,
        'curr_action': curr_action,
        'inactive_balance': 0.0,
        'balance_fiat': 1.0,
        'balance_coin': 0.0,
        'balance_value': 1.0,
        'slippage': 0.0,
        'inactive_balance_history': [0.0] * args['window_size'],
        'active_balance_history': [1.0] * args['window_size'],
        'balance_value_history': [1.0] * args['window_size']
    }

    for roll_step in range(
        args['window_size'],
        len(data) - args['window_size'] - 1
    ):

        w = data.loc[
            (roll_step):(args['window_size'] + roll_step - 1),
            mf.models_features[args['selected_model']]['features']
        ]

        train_results = train_state_ta(
            w,
            **transition_args
        )

        state_transitions = state_transitions.append(
            train_results,
            ignore_index=True
        )

        transition_args['curr_action'] = train_results['curr_action']
        transition_args['prev_action'] = train_results['prev_action']
        transition_args['balance_fiat'] = train_results['balance_fiat']
        transition_args['balance_coin'] = train_results['balance_coin']
        transition_args['balance_value'] = train_results['balance_value']
        transition_args['state'] = train_results['next_state']

        transition_args['balance_value_history'].append(
            train_results['balance_value']
        )

        transition_args['agent'].memory.append(
            (
                np.array([
                    train_results['state'][0]
                ]),
                train_results['action'],
                max(0, train_results['change']),
                np.array([
                    train_results['next_state'][0]
                ]),
                w.tail(1)['done'].values[0]
            )
        )

        if len(transition_args['agent'].memory) > args['batch_size']:
            transition_args['agent'].experiment_replay(args['batch_size'])

        # print(transition_args)

    transitions_file = transitions_path.format(
        args['selected_model'],
        stock_name,
        window_size,
        e
    )
    transitions_dir = os.path.dirname(transitions_file)
    if not os.path.isdir(transitions_dir):
        os.makedirs(transitions_dir)

    state_transitions[[
        col for col in state_transitions.columns
        if col not in ['state', 'next_state']
    ]].to_csv(
        transitions_file,
        index=False
    )

    savefile = model_path.format(
        args['selected_model'],
        args['stock_name'],
        args['window_size'],
        str(e)
    )
    savedir = os.path.dirname(os.path.abspath(savefile))
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    agent.model.save(savefile)
