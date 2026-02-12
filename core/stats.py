def correlatdef correlation(df, col1, col2):
    return df[col1].corr(df[col2])

def summary_stats(df, col):
    series = df[col].dropna()
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std()
    }￼Enter
