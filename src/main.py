# %%
from src.dataset import Dataset
from src.LGBM_approach import TreeApproach
from src.SGCN_approach import GNNAproach
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch_geometric.utils.negative_sampling import negative_sampling
import pandas as pd
import lightgbm as lgb
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Dataset("/workspaces/thesis/wiki-RfA.txt")
dataframe = dataset.read_file()
df_preprocessed = dataset.preprocess_dataframe(df=dataframe)
torch_dataset = dataset.to_torch(dataset.graph)
results = []

gnn_grid = {"num_layers": [2, 3, 4], "num_neurons": [64, 128, 256, 512]}

help_gnn = GNNAproach(dataset=torch_dataset, num_neurons=64, num_layers=2)
test_dataset = help_gnn._build_test_dataset()
test_indices = torch.tensor(test_dataset.loc[:, ["SOURCE", "TARGET"]].values)
y_test = test_dataset.pop("VOTE")

for num_layers in gnn_grid["num_layers"]:
    for num_neurons in gnn_grid["num_neurons"]:
        gnn = GNNAproach(dataset=torch_dataset, num_neurons=num_neurons, num_layers=num_layers)
        gnn.train_neg_edge_index = help_gnn.train_neg_edge_index.to(device)
        gnn.train_pos_edge_index = help_gnn.train_pos_edge_index.to(device)
        gnn.test_pos_edge_index = help_gnn.test_pos_edge_index.to(device)
        gnn.test_neg_edge_index = help_gnn.test_neg_edge_index.to(device)

        gnn.spectral_features = gnn.model.create_spectral_features(
            gnn.train_neg_edge_index, gnn.train_pos_edge_index
        )
        gnn.train()
        gnn_preds = gnn.predict(test_indices)
        gnn_proba = gnn.predict_proba(test_indices)
        gnn_acc = accuracy_score(y_test, gnn_preds)
        gnn_prec = precision_score(y_test, gnn_preds, average=None)
        gnn_rec = recall_score(y_test, gnn_preds, average=None)
        gnn_roc = roc_auc_score(
            y_test,
            gnn_proba,
            multi_class="ovo",
        )
        result_object = {
            "method": "gnn",
            "num_layers": num_layers,
            "num_neurons": num_neurons,
            "acc": gnn_acc,
            "precision": gnn_prec,
            "recall": gnn_rec,
            "roc": gnn_roc,
        }
        results.append(result_object)
        print(result_object)


negative_samples = negative_sampling(torch_dataset.edge_index)
negative_samples = pd.DataFrame(
    [list(x.numpy()) for x in negative_samples.T], columns=["SOURCE", "TARGET"]
)
negative_samples["VOTE"] = 0
negative_samples["DAT"] = pd.NaT

test_dataset["VOTE"] = y_test
df_tree = pd.concat(
    [df_preprocessed, negative_samples, test_dataset.loc[test_dataset["VOTE"] == 0]]
)
y_test = test_dataset.pop("VOTE")
tree = TreeApproach(df=df_tree, graph=dataset.graph)
df_tree = tree.create_modellng_df()

outer_join = test_dataset.merge(df_tree, on=["SOURCE", "TARGET"], how="outer", indicator=True)
X_train = outer_join[(outer_join._merge == "right_only")].drop("_merge", axis=1)
y_train = X_train.pop("VOTE")
X_test = test_dataset.merge(df_tree, on=["SOURCE", "TARGET"], how="inner")
X_test.drop_duplicates(subset=["SOURCE", "TARGET"], inplace=True)
y_test = X_test.pop("VOTE")
tree_grid = {"n_estimators": [400, 600, 800, 1000], "max_depth": [4, 8, 10, 15]}
from sklearn.metrics import f1_score


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return "f1", f1_score(y_true, y_hat), True


for n_estimators in tree_grid["n_estimators"]:
    for depth in tree_grid["max_depth"]:
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=depth,
            class_weight="balanced",
            eval_metric="auc",
            early_stopping_rounds=200,
        )
        tree_preds = tree.train_predict(model, X_train, y_train, X_test, y_test)
        tree_proba = model.predict_proba(X_test)
        tree_acc = accuracy_score(y_test, tree_preds)
        tree_prec = precision_score(y_test, tree_preds, average=None)
        tree_rec = recall_score(y_test, tree_preds, average=None)
        tree_roc = roc_auc_score(y_test, tree_proba, multi_class="ovo")
        result_object = {
            "method": "tree",
            "n_estimators": n_estimators,
            "max_depth": depth,
            "acc": tree_acc,
            "precision": tree_prec,
            "recall": tree_rec,
            "roc": tree_roc,
        }
        results.append(result_object)
        print(result_object)
result_df = pd.DataFrame(results)
result_df.to_csv("grid_search_results.csv", index=False)


model.feature_importances_
