import re


def car_age(row):
    return 2024 - row["year"]


def km_per_year(row):
    return row["km_driven"] / (row["car_age"] + 1)


def owner_age_ratio(row):
    return row["car_age"] / (row["owner"] + 1)


def torque(row):
    nm_match = re.search(r"([\d.]+)\s*nm", row.torque.lower())
    kgm_match = re.search(r"([\d.]+)\s*kgm", row.torque.lower())
    generic_match = re.search(r"([\d.]+)(?=@|/|\()", row.torque)
    if nm_match:
        return float(nm_match.group(1))
    elif kgm_match:
        return float(kgm_match.group(1)) * 9.80665
    elif generic_match:
        if float(generic_match.group(1)) > 40:
            return float(generic_match.group(1))
        else:
            return float(generic_match.group(1)) * 9.80665
    else:
        return float(row.torque.split()[0])


def power_to_engine_ratio(row):
    return row["max_power"] / row["engine"]


def mileage_per_power(row):
    return row["mileage"] / (row["max_power"] + 1)


def owner_age_ratio(row):
    return row["car_age"] / (row["owner"] + 1)


def performance_score(row):
    return row["max_power"] * row["engine"]


def power_per_seat(row):
    return row["max_power"] / (row["seats"] + 1)


def engineer_features(input_df):
    df = input_df.copy()
    function_list = [
        name
        for name in globals()
        if callable(globals()[name]) and name != "engineer_features"
    ]
    for function_name in function_list:
        df[function_name] = df.apply(globals()[function_name], axis=1)

    print(function_list)

    return df
