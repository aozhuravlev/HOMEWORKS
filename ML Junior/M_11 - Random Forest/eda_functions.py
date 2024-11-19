import re


def str_to_float(input_df, txt):
    df = input_df.copy()
    cols_to_float = re.findall(r"\b[a-zA-Z_]+\b", txt)

    for col in cols_to_float:
        df[col] = df[col].apply(
            lambda x: float(x.split()[0]) if isinstance(x, str) else None
        )

    return df


def fill_with_mode(input_df):
    df = input_df.copy()
    for i in df.columns:
        df[i] = df[i].fillna(df[i].mode()[0])

    return df
