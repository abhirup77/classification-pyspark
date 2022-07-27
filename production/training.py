"""Processors for the model training step of the worklow."""
import logging
import os.path as op
import os
import yaml
from pyspark.ml.classification import LogisticRegression
from ta_lib.pyspark import dp, features, model_gen
from ta_lib.pyspark.utils import save_model


from ta_lib.pyspark.constants import DEFAULT_ARTIFACTS_PATH

logger = logging.getLogger(__name__)

from ta_lib.pyspark.processors import register_processor

HERE = op.dirname(op.abspath(__file__))
## Todo: Reference the below from cli.py if possible?
with open(op.join(HERE, "conf", "data_catalog", "remote.yml"), "r") as fp:
    data_config = yaml.load(fp)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model."""
    train_dataset = (
        data_config["processed"]["base_path"] + data_config["processed"]["train"]
    )

    fs = data_config["clean"]["filesystem"]

    spark = context.CreateSparkSession

    # load datasets
    train_df = dp.read_data(
        spark=context.spark,
        paths=[train_dataset],
        fs=data_config["clean"][
            "filesystem"
        ],  # Todo: context.data_catalog['data']['raw']['filesystem']
    )

    # Outlier treatment
    outlier = dp.Outlier_Treatment(
        cols=["age", "last_call_date_diff", "last_act_date_diff"],
        drop=True,
        cap=False,
        method="iqr",
        iqr_multiplier=1.5,
    )
    outlier.fit(train_df)
    train_df = outlier.transform(train_df)

    # Encoding categorical features
    encoder = features.Encoder(
        cols=["gender_code", "state_code"],
        rules={
            "state_code": {"method": "target", "target_col": "target_var"},
            "gender_code": {"method": "onehot"},
        },
    )
    encoder.fit(train_df)
    train_df = encoder.transform(train_df)

    id_cols = "customer_id"
    num_cols = dp.list_numerical_columns(train_df)
    cat_cols = dp.list_categorical_columns(train_df)
    date_cols = dp.list_datelike_columns(train_df)
    bool_cols = dp.list_boolean_columns(train_df)

    target_col = "target_var"
    non_relevant_cat_cols = []
    non_relevant_num_cols = [x for x in num_cols if "index" in x]
    feature_cols = train_df.columns
    feature_cols = [
        x
        for x in feature_cols
        if x
        not in cat_cols
        + date_cols
        + bool_cols
        + [id_cols]
        + non_relevant_num_cols
        + non_relevant_cat_cols
        + [target_col]
    ]

    # Generating the Features Vector
    train_df = dp.generate_features_vector(
        spark, train_df, feature_cols, output_col="features"
    )

    # renaming target col as y
    train_df = train_df.withColumnRenamed(target_col, "y")

    ## estimator declaration
    m = LogisticRegression(featuresCol="features", labelCol="y", predictionCol="yhat")
    model = m.fit(train_df)

    # save fitted training pipeline
    save_model(
        model,
        data_config["processed"]["filesystem"]
        + ":"
        + data_config["processed"]["base_path"]
        + "/models_vacation",
    )
