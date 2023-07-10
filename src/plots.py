# %%
import plotly.express as px
from src.dataset import Dataset
import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# %%
dataset = Dataset("/workspaces/thesis/wiki-RfA.txt")
dataframe = dataset.read_file()
df_preprocessed = dataset.preprocess_dataframe(df=dataframe)

# %%

results = pd.read_csv("/workspaces/thesis/grid_search_results.csv")
metrics = ["precision", "recall"]
results["recall"] = (
    results["recall"]
    .str.replace(r"(\d+)(?!,)\b", r"\1,", regex=True)
    .str.replace("0,", "0")
    .apply(lambda x: ast.literal_eval(x))
)
results["precision"] = (
    results["precision"]
    .str.replace(r"(\d+)(?!,)\b", r"\1,", regex=True)
    .str.replace("0,", "0")
    .apply(lambda x: ast.literal_eval(x))
)
for metric in metrics:
    new_col_list = [f"{metric}_negative", f"{metric}_nonexistent", f"{metric}_positive"]
    for n, col in enumerate(new_col_list):
        results[col] = results[metric].apply(lambda x: x[n])
    results = results.drop(metric, axis=1)
# %%
# best models
idx = results.groupby("method")["roc"].transform(max) == results["roc"]
best = results[idx]
best.rename(columns={"acc": "Accuracy", "roc": "ROC"}, inplace=True)
best = best.T
best.columns = best.iloc[0]
best = best[1:]
best.columns = ["SGCN", "LGBM"]
best["SGCN"] = best["SGCN"].astype(float)
best["LGBM"] = best["LGBM"].astype(float)
# %%
plt.style.use("fast")


xd = best.loc[best.index.isin(["Accuracy", "ROC"]), :]
xd.columns.name = "Value"
xd.index.name = "Metric"
xd.index = ["Accuracy", "AUC"]
xd.plot.barh(title="Accuracy and AUC of best models").legend(
    loc="center left", bbox_to_anchor=(1, 0.5), reverse=True
)
plt.savefig("/workspaces/thesis/src/plots/acc_roc1.png", dpi=300, bbox_inches="tight")
# %%
fig = px.bar(xd, barmode="group", orientation="h")
fig.update_layout(
    title=dict(text="Accuracy and AUC of best models", y=0.99),
    font=dict(size=40),
    xaxis_title="Value",
    yaxis_title="Metric",
    legend=dict(title="Model", traceorder="reversed"),
    template="simple_white",
)
fig.write_image("/workspaces/thesis/src/plots/acc_roc.png", width=1500, height=1000)
fig.show()

# %%
xd = best.loc[
    best.index.isin(
        ["precision_negative", "precision_nonexistent", "precision_positive"]
    ),
    :,
]
xd.index = [x[len("precision_") :].capitalize() for x in xd.index]  # noqa

xd.columns.name = "Value"
xd.index.name = "Class type"
xd.plot.barh(title="Precision of best models").legend(
    loc="center left", bbox_to_anchor=(1, 0.5), reverse=True
)
plt.savefig("/workspaces/thesis/src/plots/precision1.png", dpi=300, bbox_inches="tight")

# %%
fig_prec = px.bar(
    xd,
    barmode="group",
    orientation="h",
)
fig_prec.update_layout(
    title=dict(text="Precision of best models", y=0.99),
    font=dict(size=40),
    xaxis_title="Value",
    yaxis_title="Class name",
    legend=dict(title="Model", traceorder="reversed"),
    template="simple_white",
)
fig_prec.write_image(
    "/workspaces/thesis/src/plots/precision.png", width=1500, height=1000
)
fig_prec.show()

# %%
xd = best.loc[
    best.index.isin(["recall_negative", "recall_nonexistent", "recall_positive"]), :
]
xd.index = [x[len("recall_") :].capitalize() for x in xd.index]  # noqa

xd.columns.name = "Value"
xd.index.name = "Class type"
xd.plot.barh(title="Recall of best models").legend(
    loc="center left", bbox_to_anchor=(1, 0.5), reverse=True
)
plt.savefig("/workspaces/thesis/src/plots/recall1.png", dpi=300, bbox_inches="tight")

# %%
fig_rec = px.bar(
    xd,
    barmode="group",
    orientation="h",
)
fig_rec.update_layout(
    title=dict(text="Recall of best models", y=0.99),
    font=dict(size=40),
    xaxis_title="Value",
    yaxis_title="Class name",
    legend=dict(title="Model", traceorder="reversed"),
    template="simple_white",
)
fig_rec.write_image("/workspaces/thesis/src/plots/recall.png", width=1500, height=1000)
fig_rec.show()
# %%

plt.figure(figsize=(6, 6))
graph = nx.from_pandas_edgelist(
    df_preprocessed, "SOURCE", "TARGET", edge_attr="VOTE", create_using=nx.DiGraph
)
sub = nx.subgraph(graph, list(graph.nodes())[:50])
sub_edges = list(sub.edges(data=True))
colors = ["r" if sub_edges[i][2]["VOTE"] == -1 else "g" for i in range(len(sub_edges))]
nx.draw_networkx(sub, arrows=True, edge_color=colors, with_labels=False)
plt.savefig("/workspaces/thesis/src/plots/network.png", dpi=300, bbox_inches="tight")


# %%
def perm_importance(model, X_train, X_test, y_test):
    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, X_test, y_test)
    feat_importances = pd.Series(r["importances_mean"], index=X_train.columns)
    feat_importances = feat_importances.reset_index()
    feat_importances.columns = ["Predictor", "Score"]
    feat_importances.sort_values(by="Score", inplace=True, ascending=False)
    feat_importances["Score"] = np.round(feat_importances["Score"], 3)
    feat_importances["Predictor"] = (
        feat_importances["Predictor"]
        .replace(
            {
                "_2": "_target",
                "_1": "_source",
                "SOURCE": "source_name",
                "TARGET": "target_name",
            },
            regex=True,
        )
        .replace({"adj_matrix_target": "adj_matrix_2"}, regex=True)
    )
    feat_importances.to_latex(
        index=False,
        caption="Permutation feature importance from the model",
        label="Feature importance",
    )


def roc_curve_class(y_train, y_test, proba, model_name):
    from sklearn.metrics import RocCurveDisplay
    from itertools import cycle
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelBinarizer

    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    plt.style.use("fast")
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    target_names = ["negative", "nonexistent", "positive"]
    for class_id, color in zip(range(3), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            proba[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
        )
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves for the {model_name} model")
    plt.legend()
    plt.savefig(f"roc_{model_name}.png", dpi=300, bbox_inches="tight")
