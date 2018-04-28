import os
import sys
import logging

import argparse

if __name__ == '__main__':
    from keras.models import load_model

    from agent import AgentTa
    from functions_dataframe import *
    import models_features as mf

    #: The model_path includes the model name, the dataset used,
    #: window  size and episode being saved
    model_path = "./output/models/{}/{}/model_w{}-ep{}.model"
    transitions_path = "./output/transitions/{}/transitions-{}-{}-e{:03d}-cv.csv"

    stock_name = "xxbteur-kraken-filtered"
    episode_count = 200
    window_size = 50
    selected_model = "ta_sma_rsi"
    batch_size = 32

    parser = argparse.ArgumentParser(
        description='Q-trader sample with pandas dataframe'
    )

    parser.add_argument(
        '--stock',
        action="store",
        dest="stock_name",
        default=stock_name
    )

    parser.add_argument(
        '--window-size',
        action="store",
        type=int,
        dest="window_size",
        default=window_size
    )

    parser.add_argument(
        '--batch-size',
        action="store",
        type=int,
        dest="batch_size",
        default=batch_size
    )

    parser.add_argument(
        '--episodes',
        action="store",
        dest="episode_count",
        type=int,
        default=episode_count
    )

    parser.add_argument(
        '--model',
        action="store",
        dest="selected_model",
        default=selected_model
    )

    args = parser.parse_args()

    exclude_variables = ['next_close', 'next_returns', 'done']

    model_cols = len(mf.models_features[args.selected_model]['features']) - \
        len(exclude_variables)

    input_features = [
        column for column in
        mf.models_features[args.selected_model]['features']
        if column not in exclude_variables
    ]

    data = get_cv_data(
        args.stock_name,
        window_size=args.window_size
    )

    agent = AgentTa(
        args.window_size * model_cols,
        is_eval=False,
        model_path=model_path.format(
            args.selected_model,
            args.stock_name,
            args.window_size,
            args.episode_count
        )
    )
    agent.epsilon = 0.0
    agent.epsilon_min = 0.0

    roll_step = 0

    w = data.loc[
        (
            args.window_size + 1 + roll_step
        ):(
            2 * args.window_size + roll_step
        ),
        input_features
    ]

    state_data = np.empty(
        len(w) * model_cols,
        dtype=np.float32
    )

    for col in range(model_cols):
        curr_col = mf.models_features[args.selected_model]['features'][col]
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
        'epoch': 1,
        'batch_size': args.batch_size,
        'selected_model': args.selected_model,
        'features': mf.models_features[args.selected_model]['features'],
        'exclude_variables': exclude_variables,
        'agent': agent,
        'state': state,
        'curr_action': curr_action,
        'inactive_balance': 0.0,
        'balance_fiat': 1.0,
        'balance_coin': 0.0,
        'balance_value': 1.0,
        'slippage': 0.0,
        'inactive_balance_history': [0.0] * args.window_size,
        'active_balance_history': [1.0] * args.window_size,
        'balance_value_history': [1.0] * args.window_size
    }

    print(
        "Starting loop from {} to {}".format(
            args.window_size,
            len(data) - args.window_size - 1
        )
    )

    for roll_step in range(args.window_size, len(data) - args.window_size - 1):

        w = data.loc[
            (roll_step):(args.window_size + roll_step - 1),
            mf.models_features[args.selected_model]['features']
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
        # print(train_results['next_state'])

        transition_args['balance_value_history'].append(
            train_results['balance_value']
        )

        # transition_args['agent'].memory.append(
        #     (
        #         np.array([
        #             train_results['state']
        #         ]),
        #         train_results['action'],
        #         max(0, train_results['change']),
        #         np.array([
        #             train_results['next_state']
        #         ]),
        #         w.tail(1)['done'].values[0]
        #     )
        # )

    transitions_file = transitions_path.format(
        args.selected_model,
        stock_name,
        window_size,
        args.episode_count
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
