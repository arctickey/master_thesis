import pandas as pd
from collections import defaultdict
from torch_geometric.data import Data
import torch
import networkx as nx


class Dataset:
    def __init__(self, path: str):
        self.path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_index = None
        self.graph = None

    def read_file(self) -> pd.DataFrame:
        with open(self.path, "r", encoding="utf-8") as file:
            dataset_text = file.read()
            records = dataset_text.strip().split("\n\n")
            KEYS = ["SRC", "TGT", "VOT", "RES", "YEA", "DAT", "TXT"]
            input_dict = defaultdict(list)
            for record in records:
                lines = record.strip().split("\n")
                record_dict = {}
                for line in lines:
                    key, value = line.split(":", 1)
                    record_dict[key.strip()] = value.strip()
                for key in KEYS:
                    input_dict[key].append(record_dict.get(key))
            df = pd.DataFrame(input_dict)
            return df

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={"SRC": "SOURCE", "TGT": "TARGET", "VOT": "VOTE"})
        df.drop(columns=["RES", "YEA", "TXT"], inplace=True)
        df["DAT"] = pd.to_datetime(
            df["DAT"].str.split(", ").str[1], format="%d %B %Y", errors="coerce"
        )
        df["VOTE"] = df["VOTE"].astype(int)
        df = df.loc[df["VOTE"] != 0]
        df.drop_duplicates(subset=["SOURCE", "TARGET"], inplace=True)
        unique_users = set(df.loc[:, "SOURCE"].unique()) | set(df.loc[:, "TARGET"].unique())
        mapping = {node: index for index, node in enumerate(unique_users)}
        df["SOURCE"] = df["SOURCE"].map(mapping).values
        df["TARGET"] = df["TARGET"].map(mapping).values
        df.reset_index(inplace=True, drop=True)
        self.graph = nx.from_pandas_edgelist(
            df, "SOURCE", "TARGET", edge_attr="VOTE", create_using=nx.DiGraph
        )
        self.edge_index = torch.tensor(list(self.graph.edges()), dtype=torch.long).to(self.device)
        return df

    def to_torch(self, graph: nx.DiGraph) -> Data:
        edge_index_to_create_graph = torch.tensor(list(graph.edges()), dtype=torch.long).to(
            self.device
        )
        x = torch.tensor(list(graph.nodes()), dtype=torch.float).to(self.device)
        weights = list(graph.edges(data=True))
        weights = [x[2]["VOTE"] for x in weights]
        weights = torch.tensor(weights, dtype=torch.long).to(self.device)

        dataset = Data(
            x=x,
            edge_index=edge_index_to_create_graph.t().contiguous(),
            edge_weight=weights,
        )
        return dataset

    def prepare_for_modelling(self) -> tuple[pd.DataFrame | Data, list[int, int]]:
        dataframe = self.read_file()
        df_preprocessed = self.preprocess_dataframe(df=dataframe)
        return df_preprocessed
