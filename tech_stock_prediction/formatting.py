DECIMALS = 5
CSV_FLOAT_FORMAT = f"%.{DECIMALS}f"


def set_pandas_display_options(pd):
    pd.set_option("display.float_format", lambda value: f"{value:.{DECIMALS}f}")


def save_csv(data, path):
    data.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT)


def format_decimal(value):
    return f"{value:.{DECIMALS}f}"


def format_percent(value):
    return f"{value:.{DECIMALS}%}"
