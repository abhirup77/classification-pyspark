"""Module for Model Evaluation and Interpretation."""

from collections import defaultdict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import table

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from pyspark.sql import types as DT
from pyspark_dist_explore import hist
from ta_lib.pyspark.dp import identify_col_data_type
from ta_lib.pyspark.handy_spark_cd import BinaryClassificationMetrics

sns.set()

_VALID_REGRESSION_METRICS_ = {
    "Explained Variance": "exp_var",
    "RMSE": "rmse",
    "MAE": "mae",
    "MSE": "mse",
    "MAPE": "mape",
    "WMAPE": "wmape",
    "R.Sq": "r2",
}


def get_regression_metrics(spark, data, y_col, y_pred_cols, sig=2):
    """Generate the regression summary and metrics for a model trained.

    Parameters
    ----------
        spark: SparkSession
        data: pyspark.sql.DataFrame
        y_pred:  list(str)
                list of col_names for predictions
                if list of 1 represents the results for 1 model
                len(list)>0 implies multiple model results are presented here
        y:  str
                colname of the actuals in the data
        sig: int
                significance in terms of decimals for metrics as well as threshold

    Returns
    -------
        metrics: pyspark.sql.DataFrame
                metric included are explained_variance, mae, mape, mse, Rsq, wmape
                wmape uses custome function, rest use the Regression Evaluator
    """
    df = defaultdict(list)
    metrics = _VALID_REGRESSION_METRICS_
    for metric in metrics.keys():
        df["metric"].append(metric)
        for y_pred_col in y_pred_cols:
            if metrics[metric] == "wmape":
                val = wmape(spark, data, y_col, y_pred_col)
            elif metrics[metric] == "mape":
                val = mape(spark, data, y_col, y_pred_col)
            elif metrics[metric] == "exp_var":
                val = exp_var(spark, data, y_col, y_pred_col)
            else:
                e = RegressionEvaluator(labelCol=y_col, predictionCol=y_pred_col)
                val = e.evaluate(data, {e.metricName: metrics[metric]})
            df[y_pred_col].append(round(val, sig))
    df = dict(df)
    df = pd.DataFrame(df)
    df = spark.createDataFrame(df)
    return df


def wmape(spark, data, y_col, y_hat_col):
    """Calculate the WMAPE in a data.

    Parameters
    ----------
        spark:
        data - spark.DataFrame
        y_col - str
        y_hat_col - str

    Returns
    -------
        wmape - numeric
    """
    wmape = (
        data.groupBy()
        .agg((F.sum(F.abs(F.col(y_hat_col) - F.col(y_col))) / F.sum(F.col(y_col))))
        .collect()[0][0]
    )
    return wmape


def mape(spark, data, y_col, y_hat_col):
    """Calculate the MAPE in a data.

    Parameters
    ----------
            spark: SparkSession
            data:  pyspark.sql.DataFrame
            y_col: str
            y_hat_col: str

    Returns
    -------
            mape: numeric
    """
    mape = (
        data.groupBy()
        .agg((F.mean(F.abs(F.col(y_hat_col) - F.col(y_col)) / F.col(y_col))))
        .collect()[0][0]
    )
    return mape


def exp_var(spark, data, y_col, y_hat_col):
    """Calculate the Explained Variance in a data.

    Parameters
    ----------
            spark: SparkSession
            data - pyspark.sql.DataFrame
            y_col - str
            y_hat_col - str

    Returns
    -------
            exp_var - numeric
            explainedVariance = 1 - [variance(y - yhat) / variance(y)]
    """
    exp_var = (
        data.groupby()
        .agg(
            F.pow(F.stddev(F.col(y_col) - F.col(y_hat_col)), 2)
            / F.pow(F.stddev(F.col(y_col)), 2)
        )
        .collect()[0][0]
    )
    return exp_var


def get_regression_plots(
    spark, data, y_col, y_pred_col, threshold=0.5, feature_cols=[], feature_plots=False
):
    """Generate the regression summary and metrics for a model trained.

    Parameters
    ----------
            spark: SparkSession
            data: pyspark.sql.DataFrame
            y_col: str
                Column name of the actuals in the data
            y_pred_col: str
                Column name of the predictions column in the data
            threshold: numeric in range [0,1]
                Threshold for identifying overpredictions/underpredictions
            feature_cols: list (str)
                Columns we consider as features in the data
                Useful for feature level comparison of model performance
            feature_plots: bool, default is False
                If True, then plot the relevant features

    Returns
    -------
            residual_distribution_plot - frequency dist
            actual v predicted - scatter
            residual vs predicted - scatter
            If feature_plots is True:
                Plot relevant feature plots
    """

    data = data.withColumn("residual", F.col(y_pred_col) - F.col(y_col))
    data = data.withColumn(
        "forecast_flag",
        F.when((F.col(y_pred_col) > (1 + threshold) * F.col(y_col)), "Above threshold")
        .when((F.col(y_pred_col) < (1 - threshold) * F.col(y_col)), "Below threshold")
        .otherwise("Within threshold"),
    )
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(10, 20)

    # Residual Distribution Plot
    hist(axes[0], [data.select("residual")], bins=20)
    axes[0].set_title("01. Residual Histogram")

    # Actual vs Predicted Scatter
    for k, v in {
        "Above threshold": "royalblue",
        "Below threshold": "darkorange",
    }.items():
        plot_df = data.filter((F.col("forecast_flag") == k))
        x = plot_df.select(y_col).collect()
        y = plot_df.select(y_pred_col).collect()
        axes[1].scatter(x=x, y=y, label=k, color=v, s=10)
    axes[1].set_xlabel(y_col)
    axes[1].set_ylabel(y_pred_col)
    axes[1].legend()
    axes[1].set_title(f"02. Actual vs Predicted with threshold={threshold}")

    # Residual vs Predicted Scatter
    for k, v in {
        "Above threshold": "royalblue",
        "Below threshold": "darkorange",
    }.items():
        plot_df = data.filter((F.col("forecast_flag") == k))
        x = plot_df.select(y_pred_col).collect()
        y = plot_df.select("residual").collect()
        axes[2].scatter(x=x, y=y, label=k, color=v, s=10)
    axes[2].set_xlabel(y_pred_col)
    axes[2].set_ylabel("Residual")
    axes[2].legend()
    axes[2].set_title(f"03. Residual vs Predicted with threshold={threshold}")

    plt.show()

    if feature_plots:
        # Plotting interaction plots
        fig, axs = plt.subplots(
            nrows=len(feature_cols), ncols=1, figsize=(18, 6 * len(feature_cols))
        )
        for idx, feature in enumerate(feature_cols):
            axs[idx] = plot_interaction(spark, data, feature, "residual", ax=axs[idx])
            # plot_interaction(
            #     spark,
            #     data,
            #     feature,
            #     "residual",
            #     ax=axs[idx])
            axs[idx].set_title(f"Interaction of {feature} with Residual")
        # fig.suptitle('Feature v Residual Plots', fontsize=10)
        plt.show()

    return


# FIX ME
# Interactive y,yhat plot
# How can we get this in pyspark

# -----------------------------------------------------------------------
# Classification - Individual Model (WIP)
# -----------------------------------------------------------------------
_BINARY_CLASSIFICATION_METRICS_ = {
    "Accuracy": "accuracy",
    "F1 Score": "f1",
    "TPR": "tpr",
    "FPR": "fpr",
    "Precision": "precision",
    "Recall": "recall",
    "AuROC": "auROC",
    "AuPR": "auPR",
}


def get_binary_classification_metrics(
    spark, data, y_col, y_pred_cols, probability_cols, threshold=0.50, sig=2
):
    """Get the regression summary and metrics for a model trained.

    Parameters
    ----------
            spark: SparkSession
            data: pyspark.sql.DataFrame
            y_pred_cols: list(str)
                    list of col_names for predictions
                    if list of 1 represents the results for 1 model
                    len(list)>0 implies multiple model results are presented here
            y_col: str
                    colname of actuals in the data

            probability_cols: list(str)
                    list of col_names for probabilities of predictions
                    if list of 1 represents the results for 1 model
                    len(list)>0 implies multiple model results are presented here
                    (have to orderd similar to ref the y_pred_cols)
            threshold: double
                    threshold to consider for calculation of metrics
                    0.5 y default
            sig: int
                    significance in terms of decimals for metrics as well as threshold

    Returns
    -------
            metrics: pyspark.sql.DataFrame
    """
    relevant_cols = [y_col] + y_pred_cols + probability_cols
    data = data.select(*relevant_cols)
    df = defaultdict(list)
    metrics = _BINARY_CLASSIFICATION_METRICS_
    for metric in metrics.keys():
        df["metric"].append(metric)

    for idx, y_pred_col in enumerate(y_pred_cols):
        bcm = BinaryClassificationMetrics(
            data, scoreCol=probability_cols[idx], labelCol=y_col
        )
        conf_matrix = bcm.confusionMatrix(threshold=threshold).toArray().tolist()
        tp = conf_matrix[1][1]
        fp = conf_matrix[0][1]
        tn = conf_matrix[0][0]
        fn = conf_matrix[1][0]

        accuracy = (tp + tn) / (tp + fp + tn + fn)  # noqa
        tpr = tp / (tp + fn)  # noqa
        fpr = fp / (fp + tn)  # noqa
        precision = tp / (tp + fp)  # noqa
        recall = tp / (tp + fn)  # noqa
        f1 = 2 * precision * recall / (precision + recall)  # noqa

        auROC = bcm.areaUnderROC  # noqa
        auPR = bcm.areaUnderPR  # noqa
        for metric in df["metric"]:
            val = round(eval(metrics[metric]), sig)  # noqa
            df[y_pred_col].append(val)
    df = pd.DataFrame(df)
    df = spark.createDataFrame(df)
    return df


def get_binary_classification_plots(
    spark,
    data,
    y_col,
    y_pred_col,
    probability_col,
    feature_cols=[],
    threshold=0.5,
    feature_plots=False,
):
    """Get the binary classification summary and metrics for a model trained.

    Parameters
    ----------
            spark: SparkSession
            data: pyspark.sql.DataFrame
            y_pred_col: str
                    column of prediction label in the data
            y_col: str
                    colname of actuals in the data
            probability_cols: str
                    colname of probability column in the data
            feature_cols: list(str)
                    columns we consider as features in the data
                    useful for feature level comparison of the performance
                    they have to be present as columns in the data
            threshold: double
                    threshold to consider for calculation of metrics
                    0.5 y default
            sig: int
                    significance in terms of decimals for metrics as well as threshold

    Returns
    -------
            Plots the following Plots
            Default - Confusion Matrix, ROC Curve, PR Curve
            if feature plots are True.
                    plot relevant feature plots
    """
    bcm = BinaryClassificationMetrics(data, scoreCol=probability_col, labelCol=y_col)
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    bcm.plot_roc_curve(ax=axs[0])
    bcm.plot_pr_curve(ax=axs[1])

    # 	cm = bcm.print_confusion_matrix(threshold=threshold)
    # 	if 'Predicted' in cm.columns[0]:
    # 		xlabel='Predicted'
    # 		ylabel='Actual'
    # 	else:
    # 		xlabel='Actual'
    # 		ylabel='Predicted'

    # 	cm.columns=[x[1] for x in cm.columns]
    # 	cm.index=[x[1] for x in cm.index]

    # 	sns.heatmap(cm, annot=True, ax = axs[2])
    # 	axs[2].set_title('Confusion Matrix')
    # 	axs[2].set_xlabel(xlabel)
    # 	axs[2].set_ylabel(ylabel)
    # fig.suptitle('Evaluation Plots', fontsize=10)

    plt.show()

    fig, ax = plt.subplots(figsize=(10, 1))
    conf_matrix = bcm.print_confusion_matrix(threshold=threshold)
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)
    ax.set_title("Confusion Matrix", fontsize=14)
    # 	table(ax, conf_matrix)
    tab = table(
        ax, conf_matrix, loc="upper center", colWidths=[0.17] * len(conf_matrix.columns)
    )
    tab.auto_set_font_size(False)  # Activate set fontsize manually
    tab.set_fontsize(12)
    tab.scale(1.4, 1.4)
    plt.show()

    if feature_plots:
        # Plotting Interaction Plots with Confusion Cell
        data = generate_confusion_cell_col(
            spark,
            data,
            y_col,
            probability_col,
            confusion_cell_col="confusion_matrix_cell",
            threshold=threshold,
        )
        fig, axs = plt.subplots(
            nrows=len(feature_cols), ncols=1, figsize=(18, 6 * len(feature_cols))
        )
        for idx, feature in enumerate(feature_cols):
            axs[idx] = plot_interaction(
                spark, data, feature, "confusion_matrix_cell", ax=axs[idx]
            )
        # fig.suptitle('Feature v Confusion Plots', fontsize=10)
        plt.show()


def generate_confusion_cell_col(
    spark,
    data,
    y_col,
    probability_col,
    confusion_cell_col="confusion_matrix_cell",
    threshold=0.5,
):
    """Column to Generate the column to define the cell of confusion matrix the row belongs to.

    Paramters:
    ----------
            spark: SparkSession
            data: pyspark.sql.DataFrame
            y_col: str
            probability_col: str
            threshold: numeric
                    probability threshold for calculating the prediction labels.

    Returns
    -------
            df - pyspark.sql.DataFrame
    """

    def _get_label(probabilities):
        if probabilities[1] > threshold:
            return 1.0
        else:
            return 0.0

    def _get_conf_cell(pred_label, actual_label):
        if pred_label == 1:
            if actual_label == 1:
                return "TP"
            else:
                return "FP"
        else:
            if actual_label == 0:
                return "TN"
            else:
                return "FN"

    _get_label = F.udf(_get_label, DT.DoubleType())
    _get_conf_cell = F.udf(_get_conf_cell, DT.StringType())

    data = data.withColumn("pred_label", _get_label(F.col(probability_col)))
    data = data.withColumn(
        confusion_cell_col, _get_conf_cell(F.col("pred_label"), F.col(y_col))
    )

    return data


def plot_interaction(spark, data, col1, col2, ax):
    """Plot the interaction b/w 2 columns in a dataframe.

    Plots -
            continuous vs continuous - scatter plot
            continuous vs categorical - distribution plot
            categorical vs categorical - stacked bar plot

    Parameters
    ----------
            spark: SparkSession
            data - pyspark.sql.DataFrame
            col1 - str
            col2 - str
            ax - matplotlib axis

    Returns
    -------
            ax - matplotlib axis
    """
    col1_type = identify_col_data_type(data, col1)
    col2_type = identify_col_data_type(data, col2)

    if (col1_type == "date_like") | (col2_type == "date_like"):
        raise (
            NotImplementedError("Datelike column interactions not applicable as of now")
        )

    if (col1_type != "numerical") & (col2_type != "numerical"):
        ax = stacked_bar_plot(spark, data, col1, col2, ax)
    elif col1_type != "numerical":
        ax = distribution_plot(spark, data, cat_col=col1, cont_col=col2, ax=ax)
    else:
        ax = scatter_plot(spark, data, col1=col2, col2=col1, ax=ax)

    return ax


def scatter_plot(spark, data, col1, col2, ax):
    """Generate a scatter plot for interaction b.w 2 continuous columns.

    Parameters
    ----------
            spark: SparkSession
            data: pyspark.sql.DataFrame
            col1: str
            col2: str
            ax: matplotlib axis

    Returns
    -------
            ax: matplotlib axis
    """
    x = data.select(col1).collect()
    y = data.select(col2).collect()
    ax.scatter(x=x, y=y, marker="o")  # , ax=ax)
    ax.set_title(f"Scatter Plot of {col2} vs {col1}")
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    return ax


def distribution_plot(spark, data, cat_col, cont_col, ax):
    """Generate a  distribution plot for interaction b.w a categorical and continuous variable.

    Parameters
    ----------
            spark: SparkSession
            data: pyspark.sql.DataFrame
            cat_col: str
                    categorical column name
            cont_col: str
                    numeric column name
            ax: matplotlib axis
    Returns
    -------
            ax: matplotlib axis
    """
    df = data.select(cat_col, cont_col).toPandas()
    df.groupby(cat_col)[cont_col].plot(kind="density", legend=True, ax=ax)
    ax.set_title("Density Plot(" + cont_col + ")")
    return ax


def stacked_bar_plot(spark, data, col1, col2, ax):
    """Generate a scatter plot for interaction b.w a categorical and continuous variable.

    Parameters
    ----------
            spark: SparkSession
            data: pyspark.sql.DataFrame
            col1: str
            col2: str
            ax: matplotlib axis

    Returns
    -------
            ax: matplotlib axis
    """
    df = data.groupBy(col1, col2).agg(F.count(F.col(col1)).alias("count")).toPandas()
    df = (
        (df.groupby([col1, col2])["count"].sum() / df.groupby(col1)["count"].sum())
        * 100
    ).reset_index()
    df = df.pivot(index=col1, columns=col2, values="count")
    df.plot(kind="bar", ax=ax)
    ax.set_title("Stacked Bar Plot")
    ax.set_xlabel(col1)
    return ax
