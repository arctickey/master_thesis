# %%
import plotly.express as px
from src.dataset import Dataset
import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt

# %%
dataset = Dataset("/workspaces/thesis/wiki-RfA.txt")
dataframe = dataset.read_file()
df_preprocessed = dataset.preprocess_dataframe(df=dataframe)
votes_coutns = df_preprocessed["VOTE"].value_counts().reset_index()
votes_coutns["VOTE"] = votes_coutns["VOTE"].astype(pd.Categorical)
fig_votes = px.bar(
    df_preprocessed["VOTE"].value_counts().reset_index(),
    x="VOTE",
    y="count",
    title="Vote distribution",
)
fig_votes.update_layout(showlegend=False, xaxis_type="category")
# %%

results = pd.read_csv("/workspaces/thesis/src/grid_search_results.csv")
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
fig = px.bar(best.loc[best.index.isin(["Accuracy", "ROC"]), :], barmode="group", orientation="h")
fig.update_layout(
    title="Accuracy and ROC of best chosen models",
    font=dict(size=25),
    xaxis_title="Value",
    yaxis_title="Method",
    legend=dict(title="Model", traceorder="reversed"),
    template="simple_white",
)
fig.write_image("/workspaces/thesis/src/plots/acc_roc.png", width=1000, height=600)
fig.show()

# %%
xd = best.loc[
    best.index.isin(["precision_negative", "precision_nonexistent", "precision_positive"]), :
]
xd.index = [x.capitalize() for x in xd.index]
fig_prec = px.bar(
    xd,
    barmode="group",
    orientation="h",
)
fig_prec.update_layout(
    title="Precision of best chosen models",
    font=dict(size=25),
    xaxis_title="Value",
    yaxis_title="Method",
    legend=dict(title="Model", traceorder="reversed"),
    template="simple_white",
)
fig_prec.write_image("/workspaces/thesis/src/plots/precision.png", width=1000, height=600)
fig_prec.show()

# %%
xd = best.loc[best.index.isin(["recall_negative", "recall_nonexistent", "recall_positive"]), :]
xd.index = [x.capitalize() for x in xd.index]
fig_rec = px.bar(
    xd,
    barmode="group",
    orientation="h",
)
fig_rec.update_layout(
    title="Recall of best chosen models",
    font=dict(size=25),
    xaxis_title="Value",
    yaxis_title="Method",
    legend=dict(title="Model", traceorder="reversed"),
    template="simple_white",
)
fig_rec.write_image("/workspaces/thesis/src/plots/recall.png", width=1000, height=600)
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
plt.savefig("/workspaces/thesis/src/plots/network.png", dpi=300)
# %%
