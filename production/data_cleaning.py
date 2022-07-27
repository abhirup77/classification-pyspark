"""Processors for the data cleaning step of the worklow.
The processors in this step apply the various cleaning steps identified
during EDA to create the training datasets.
"""

import os.path as op
import yaml
import pandas as pd
from pyspark.sql import types as DT, functions as F, Window

from ta_lib.pyspark.processors import register_processor
from ta_lib.pyspark import dp


HERE = op.dirname(op.abspath(__file__))

with open(op.join(HERE, "conf", "data_catalog", "remote.yml"), "r") as fp:
    data_config = yaml.load(fp)


@register_processor("data-cleaning", "call_data")
def clean_call_table(context, params):
    """
    Clean the call_data dataset
    """

    input_dataset = (
        data_config["raw"]["base_path"] + data_config["raw"]["call_data_path"]
    )
    # Todo: context.data_catalog['data']['raw']['base_path'] + context.data_catalog['data']['raw']['carrier_data_path']
    output_dataset = (
        data_config["clean"]["base_path"] + data_config["clean"]["call_data_path"]
    )
    fs = data_config["clean"]["filesystem"]

    spark = context.CreateSparkSession

    df_call_data = dp.read_data(
        spark=context.spark,
        paths=[input_dataset],
        fs=data_config["raw"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )
    reference_date = pd.to_datetime(data_config["reference_date"])

    ## Filter carrier data dataframe
    df_call_data = df_call_data.withColumn(
        "call_date",
        F.to_date(
            F.unix_timestamp(F.col("call_date"), "ddMMMyyyy:HH:mm:ss").cast("timestamp")
        ),
    ).filter(F.col("call_date") <= reference_date)

    # Save the dataset
    dp.save_data(df_call_data, path=fs + ":" + output_dataset)

    return df_call_data


@register_processor("data-cleaning", "customer_activity_data")
def clean_customer_activity_table(context, params):
    """
    Clean the fuel_prices customer_activity_data
    """

    input_dataset = (
        data_config["raw"]["base_path"] + data_config["raw"]["last_activity_data_path"]
    )
    output_dataset = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["last_activity_data_path"]
    )
    fs = data_config["clean"]["filesystem"]

    spark = context.CreateSparkSession

    df_last_activity_data = dp.read_data(
        spark=context.spark,
        paths=[input_dataset],
        fs=data_config["raw"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )
    reference_date = pd.to_datetime(data_config["reference_date"])
    df_last_activity_data = df_last_activity_data.dropDuplicates(
        ["customer_id", "actvty_date", "actvty_type", "actvty_prod"]
    )
    last_activity_date_cols = ["load_date", "actvty_date"]
    for col in last_activity_date_cols:
        df_last_activity_data = df_last_activity_data.withColumn(
            col,
            F.to_date(
                F.unix_timestamp(F.col(col), "ddMMMyyyy:HH:mm:ss").cast("timestamp")
            ),
        ).filter(F.col("actvty_date") <= reference_date)

    # Save the dataset
    dp.save_data(df_last_activity_data, path=fs + ":" + output_dataset)

    return df_last_activity_data


@register_processor("data-cleaning", "booking_data")
def clean_booking_table(context, params):
    """
    Clean the booking  dataset
    """

    input_dataset = (
        data_config["raw"]["base_path"] + data_config["raw"]["booking_data_path"]
    )  # Todo: context.data_catalog['data']['raw']['base_path'] + context.data_catalog['data']['raw']['fuel_prices_data_path']
    output_dataset = (
        data_config["clean"]["base_path"] + data_config["clean"]["booking_data_path"]
    )
    fs = data_config["clean"]["filesystem"]

    df_booking_data = dp.read_data(
        spark=context.spark,
        paths=[input_dataset],
        fs=data_config["raw"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )
    reference_date = pd.to_datetime(data_config["reference_date"])

    pred_period_start = reference_date + pd.Timedelta(days=1)
    pred_period_end = reference_date + pd.Timedelta(
        days=data_config["num_days_prediction"]
    )

    df_booking_data = (
        df_booking_data.withColumn(
            "booking_create_date",
            F.to_date(
                F.unix_timestamp(
                    F.col("booking_create_date"), "ddMMMyyyy:HH:mm:ss"
                ).cast("timestamp")
            ),
        )
        .filter(
            (F.col("booking_create_date") >= pred_period_start)
            & (F.col("booking_create_date") <= pred_period_end)
        )
        .select("customer_id")
        .dropDuplicates()
        .withColumn("target_var", F.lit(1))
    )

    # Save the dataset
    dp.save_data(df_booking_data, path=fs + ":" + output_dataset)

    return df_booking_data


@register_processor("data-cleaning", "consumer_data")
def clean_consumer_table(context, params):
    """
    Clean the consumer  dataset
    """

    input_dataset = (
        data_config["raw"]["base_path"] + data_config["raw"]["consumer_data_path"]
    )  # Todo: context.data_catalog['data']['raw']['base_path'] + context.data_catalog['data']['raw']['fuel_prices_data_path']
    output_dataset = (
        data_config["clean"]["base_path"] + data_config["clean"]["consumer_data_path"]
    )
    fs = data_config["clean"]["filesystem"]

    df_consumer_data = dp.read_data(
        spark=context.spark,
        paths=[input_dataset],
        fs=data_config["raw"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )
    reference_date = pd.to_datetime(data_config["reference_date"])

    pred_period_start = reference_date + pd.Timedelta(days=1)
    pred_period_end = reference_date + pd.Timedelta(
        days=data_config["num_days_prediction"]
    )

    drop_lst = ["click_pct", "open_pct", "max_event_date", "booked_flag"]
    df_consumer_data = (
        df_consumer_data.drop(*drop_lst)
        .withColumn(
            "cel_first_cruise_date",
            F.to_date(
                F.unix_timestamp(
                    F.col("cel_first_cruise_date"), "ddMMMyyyy:HH:mm:ss"
                ).cast("timestamp")
            ),
        )
        .filter(F.col("cel_first_cruise_date") <= reference_date)
    )

    # Save the dataset
    dp.save_data(df_consumer_data, path=fs + ":" + output_dataset)

    return df_consumer_data


@register_processor("data-cleaning", "web_data")
def clean_web_table(context, params):
    """
    Clean the web  dataset
    """

    input_dataset = (
        data_config["raw"]["base_path"] + data_config["raw"]["web_data_path"]
    )  # Todo: context.data_catalog['data']['raw']['base_path'] + context.data_catalog['data']['raw']['fuel_prices_data_path']
    output_dataset = (
        data_config["clean"]["base_path"] + data_config["clean"]["web_data_path"]
    )
    fs = data_config["clean"]["filesystem"]

    df_web_data = dp.read_data(
        spark=context.spark,
        paths=[input_dataset],
        fs=data_config["raw"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )
    reference_date = pd.to_datetime(data_config["reference_date"])

    pred_period_start = reference_date + pd.Timedelta(days=1)
    pred_period_end = reference_date + pd.Timedelta(
        days=data_config["num_days_prediction"]
    )

    df_web_data = (
        df_web_data.dropDuplicates(
            ["customer_id", "visit_date", "device_type_name", "visit_type"]
        )
        .withColumn(
            "visit_date",
            F.to_date(
                F.unix_timestamp(F.col("visit_date"), "ddMMMyyyy:HH:mm:ss").cast(
                    "timestamp"
                )
            ),
        )
        .filter(F.col("visit_date") <= reference_date)
    )

    # Save the dataset
    dp.save_data(df_web_data, path=fs + ":" + output_dataset)

    return df_web_data
