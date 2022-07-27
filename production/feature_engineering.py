"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import os
import yaml
from ta_lib.pyspark import dp
import pandas as pd
import os.path as op
import logging

from pyspark.sql import types as DT, functions as F, Window

logger = logging.getLogger(__name__)

from ta_lib.pyspark.processors import register_processor

HERE = op.dirname(op.abspath(__file__))
## Todo: Reference the below from cli.py if possible?
with open(op.join(HERE, "conf", "data_catalog", "remote.yml"), "r") as fp:
    data_config = yaml.load(fp)


@register_processor("feature-engineering", "transform-data")
def transform_data(context, params):
    """
    Create features
    Create target columns by finding  common consumers across all feature tables. Binary column is built based on occurence of a booking for a consumer in the pred period
    """

    input_dataset_customer = (
        data_config["clean"]["base_path"] + data_config["clean"]["consumer_data_path"]
    )
    input_dataset_calls = (
        data_config["clean"]["base_path"] + data_config["clean"]["call_data_path"]
    )
    input_dataset_activity = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["last_activity_data_path"]
    )
    input_dataset_booking = (
        data_config["clean"]["base_path"] + data_config["clean"]["booking_data_path"]
    )
    input_dataset_web = (
        data_config["clean"]["base_path"] + data_config["clean"]["web_data_path"]
    )
    output_dataset = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["final_vacation_data_path"]
    )
    fs = data_config["clean"]["filesystem"]

    spark = context.CreateSparkSession

    df_call_data = dp.read_data(
        spark=context.spark,
        paths=[input_dataset_calls],
        fs=data_config["clean"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )

    df_last_activity_data = dp.read_data(
        spark=context.spark,
        paths=[input_dataset_activity],
        fs=data_config["raw"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )

    df_booking_data = dp.read_data(
        spark=context.spark,
        paths=[input_dataset_booking],
        fs=data_config["raw"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )

    df_consumer_data = dp.read_data(
        spark=context.spark,
        paths=[input_dataset_customer],
        fs=data_config["raw"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )

    df_web_data = dp.read_data(
        spark=context.spark,
        paths=[input_dataset_web],
        fs=data_config["raw"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )
    reference_date = pd.to_datetime(data_config["reference_date"])
    fs = data_config["clean"]["filesystem"]

    df_common_consumer = (
        df_consumer_data.select("customer_id")
        .join(df_call_data.select("customer_id"), on="customer_id", how="inner")
        .join(df_web_data.select("customer_id"), on="customer_id", how="inner")
        .join(
            df_last_activity_data.select("customer_id"), on="customer_id", how="inner"
        )
        .dropDuplicates()
    )

    df_common_consumer_booking = (
        df_common_consumer.join(df_booking_data, on="customer_id", how="left")
        .dropDuplicates()
        .fillna(0, subset=["target_var"])
    )

    # number of days since the consumer made the last call
    df_call_data = (
        df_call_data.withColumn(
            "last_call_date_diff",
            F.datediff(
                F.to_date(F.lit(reference_date.strftime("%Y-%m-%d"))),
                F.col("call_date"),
            ),
        )
        .groupby("customer_id")
        .agg(F.min("last_call_date_diff").alias("last_call_date_diff"))
        .where(F.col("customer_id").isNotNull())
    )

    # number of days since the consumer was last active
    df_last_activity_data = (
        df_last_activity_data.withColumn(
            "last_act_date_diff",
            F.datediff(
                F.to_date(F.lit(reference_date.strftime("%Y-%m-%d"))),
                F.col("actvty_date"),
            ),
        )
        .groupby("customer_id")
        .agg(F.min("last_act_date_diff").alias("last_act_date_diff"))
    )

    # number of days since the consumer had last web activity, total time spent in seconds, total page view count
    df_web_data = df_web_data.withColumn(
        "last_web_date_diff",
        F.datediff(
            F.to_date(F.lit(reference_date.strftime("%Y-%m-%d"))), F.col("visit_date")
        ),
    )

    df_web_data = df_web_data.groupby("customer_id").agg(
        F.min("last_web_date_diff").alias("last_web_date_diff"),
        F.sum("sec_time_spent_on_nbr").alias("total_sec_spent"),
        F.sum("page_view_count").alias("total_page_view_count"),
    )

    # Filtering the columns from the primary dataframe where all other feature tables would be merged
    df_consumer_data = (
        df_consumer_data.select(
            "customer_id", "age", "gender_code", "state_code", "rci_qualify_cruise_qty"
        )
        .join(df_call_data, on="customer_id", how="left")
        .join(df_last_activity_data, on="customer_id", how="left")
        .join(df_web_data, on="customer_id", how="left")
        .join(df_common_consumer_booking, on="customer_id", how="inner")
    )

    # Impute missing values
    imputer = dp.Imputer(cols=[])
    imputer.fit(df_consumer_data)
    imputed_data = imputer.transform(df_consumer_data)

    # Save the dataset
    dp.save_data(imputed_data, path=fs + ":" + output_dataset)

    return imputed_data


@register_processor("feature-engineering", "train_test_split")
def create_train_test_split(context, params):
    """Transform dataset to create training datasets."""

    input_dataset = (
        data_config["clean"]["base_path"]
        + data_config["clean"]["final_vacation_data_path"]
    )
    output_train_dataset = (
        data_config["processed"]["base_path"] + data_config["processed"]["train"]
    )

    output_test_dataset = (
        data_config["processed"]["base_path"] + data_config["processed"]["test"]
    )

    fs = data_config["clean"]["filesystem"]

    spark = context.CreateSparkSession

    # load datasets
    final_df = dp.read_data(
        spark=context.spark,
        paths=[input_dataset],
        fs=data_config["clean"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )

    train_df, test_df = dp.test_train_split(
        spark,
        data=final_df,
        target_col="target_var",
        train_prop=0.7,
        random_seed=0,
        stratify=True,
        target_type="categorical",
    )
    # Save the dataset
    dp.save_data(train_df, path=fs + ":" + output_train_dataset)
    dp.save_data(test_df, path=fs + ":" + output_test_dataset)
