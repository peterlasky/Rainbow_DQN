import os

from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

def plot_multiple_results(log_dir: str, 
                          names: List[str],
                          data_col: str = 'eval_avg'):
    assert data_col in ['best_score','eval_avg','trailing_avg','loss']

    # search log dir for each sub directory.  In each sub directory, look for the csv file
    # and load it into a pd datframe. 
    # append the dataframes to a list of dataframes
    dfs = []
    for name in names:
        csv_path = os.path.join(log_dir, name, name + '.csv')
        print(f'Looking for {csv_path}')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs.append(df)
        else:
            print(f'No csv file found for {csv_path}')
    if len(dfs) == 0:
        print(f'No csv files found in {log_dir}')
        return
    # use first dataframe to get the steps column
    main_df = dfs[0]['steps']

    # build combined dataframe
    for i, df in enumerate(dfs):
        # take each eval_avg column from each df, rename it eval_eval_{i} and add it to the main df, matching the steps column
        # first rename the eval_avg column to eval_avg_{i}
        df.rename(columns={data_col: f'{names[i]}'}, inplace=True)
        # then merge the dataframes on the steps column
        main_df = pd.merge(main_df, df[['steps', f'{names[i]}']], on='steps', how='outer')

    main_df['steps'] = main_df['steps'] / 1_000.0
    plt.figure(figsize=(10, 5))
    plt.title(data_col)
    for col in main_df.columns[1:]:
        plt.plot(main_df['steps'], main_df[col], label=col)
    plt.xlabel('thousand steps')
    plt.legend()
    plt.show()
    print(main_df)


    
