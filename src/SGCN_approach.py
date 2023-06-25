import torch
from torch_geometric.nn import SignedGCN
from torch_geometric.data import Data
import pandas as pd
from torch_geometric.utils.negative_sampling import negative_sampling
import numpy as np


class GNNAproach:
    def __init__(self, dataset: Data, num_neurons: int, num_layers: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.num_epochs = 501
        self.model = SignedGCN(num_neurons, num_neurons, num_layers=num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        self.train_pos_edge_index = None
        self.train_neg_edge_index = None
        self.test_pos_edge_index = None
        self.test_neg_edge_index = None
        self.spectral_features = None

    def build_model(self):
        pos_edge_indices, neg_edge_indices = (
            self.dataset.edge_index[:, self.dataset.edge_weight > 0],
            self.dataset.edge_index[:, self.dataset.edge_weight < 0],
        )

        self.train_pos_edge_index, test_pos_edge_index = self.model.split_edges(
            pos_edge_indices, test_ratio=0.1
        )
        self.train_neg_edge_index, test_neg_edge_index = self.model.split_edges(
            neg_edge_indices, test_ratio=0.1
        )
        x = self.model.create_spectral_features(
            self.train_pos_edge_index, self.train_neg_edge_index
        )
        return x, test_pos_edge_index, test_neg_edge_index

    def _build_test_dataset(self):
        (
            self.spectral_features,
            self.test_pos_edge_index,
            self.test_neg_edge_index,
        ) = self.build_model()
        pos_items = list(self.test_pos_edge_index.T)
        pos_items = pd.DataFrame([list(x.numpy()) for x in pos_items], columns=["SOURCE", "TARGET"])
        pos_items["VOTE"] = 1
        neg_items = list(self.test_neg_edge_index.T)
        neg_items = pd.DataFrame([list(x.numpy()) for x in neg_items], columns=["SOURCE", "TARGET"])
        neg_items["VOTE"] = -1
        neu_items = negative_sampling(self.dataset.edge_index, num_neg_samples=neg_items.shape[0])
        neu_items = pd.DataFrame(
            [list(x.numpy()) for x in neu_items.T], columns=["SOURCE", "TARGET"]
        )
        neu_items["VOTE"] = 0
        test_items = pd.concat([pos_items, neu_items, neg_items])
        return test_items

    def _train(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model(self.spectral_features, self.train_pos_edge_index, self.train_neg_edge_index)
        loss = self.model.loss(z, self.train_pos_edge_index, self.train_neg_edge_index)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            z = self.model(
                self.spectral_features, self.train_pos_edge_index, self.train_neg_edge_index
            )
        return self.model.test(z, self.test_pos_edge_index, self.test_neg_edge_index)

    def train(self):
        losses = []
        early_stopper = EarlyStopper(patience=200)
        for round in range(self.num_epochs):
            loss = self._train()
            losses.append(loss)
            test_auc, _ = self.test()
            if early_stopper.early_stop(test_auc):
                print(f"Breaking after {round} rounds")
                break
        return losses

    def predict(self, test_indices: torch.tensor) -> list[int]:
        self.model.eval()
        with torch.no_grad():
            embedings = self.model(
                self.spectral_features, self.train_pos_edge_index, self.train_neg_edge_index
            )
            outputs = list(
                self.model.discriminate(embedings, torch.tensor(test_indices).T).max(dim=1)[1]
            )
        outputs = [x.item() for x in outputs]
        map_outputs = {0: 1, 1: -1, 2: 0}
        outputs = list(map(map_outputs.get, outputs))
        return outputs

    def predict_proba(self, test_indices: torch.tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            embedings = self.model(
                self.spectral_features, self.train_pos_edge_index, self.train_neg_edge_index
            )
            outputs = np.exp(self.model.discriminate(embedings, test_indices.T).detach().numpy())[
                :, [1, 2, 0]
            ]
        return outputs


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
