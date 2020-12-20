import pandas as pd

def split_data(df: pd.DataFrame, data_config: dict) -> list:
    """
    Returns splitted data
    """
    if data_config['data_split']['type'] == 'date_split':
        train_df = df.loc[
            (df[data_config['data_split']['date_colname']] >= data_config['data_split']['train']['start']) & (df[data_config['data_split']['date_colname']] <= data_config['data_split']['train']['end'])
            ]
        valid_df = df.loc[
            (df[data_config['data_split']['date_colname']] >= data_config['data_split']['valid']['start']) & (df[data_config['data_split']['date_colname']] <= data_config['data_split']['valid']['end'])
            ]
        test_df = df.loc[
            (df[data_config['data_split']['date_colname']] >= data_config['data_split']['test']['start']) & (df[data_config['data_split']['date_colname']] <= data_config['data_split']['test']['end'])
            ]

    return train_df, valid_df, test_df