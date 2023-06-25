# %%
import pandas as pd
import networkx as nx
from typing import Callable
import lightgbm as lgb


class TreeApproach:
    def __init__(self, df: pd.DataFrame, graph: nx.DiGraph) -> None:
        self.metrics = [
            nx.resource_allocation_index,
            nx.jaccard_coefficient,
            nx.preferential_attachment,
        ]
        self.df = df
        self.graph = graph

    def get_power_of_adj_matrix(self, df: pd.DataFrame, g: nx.DiGraph, power: int):
        adj = nx.adjacency_matrix(g)
        if power == 2:
            a_power = adj @ adj.T
            a_power.setdiag(0)
        elif power == 3:
            a_power = (adj @ adj.T) @ adj.T
        chosen = a_power[df["SOURCE"].values, df["TARGET"].values]
        df[f"adj_matrix_{power}"] = chosen.reshape(-1, 1)
        return df

    def compute_metric(
        self, graph: nx.Graph, metrics: list[Callable], df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute metrics given within `metrics` list for each pair of nodes within `df.
        Args:
            graph (nx.Graph): Graph based on edges from `df`
            metrics (list[Callable]): Metrics to be computed
            df (pd.DataFrame): DataFrame with all edges between nodes

        Returns:
            pd.DataFrame: DataFrame with calculated metrics between nodes
        """

        def _compute_metric(graph: nx.Graph, metric: Callable, df: pd.DataFrame) -> pd.DataFrame:
            node_pairs = list(zip(df.SOURCE, df.TARGET))
            generator = metric(graph, node_pairs)
            score_pairs = [score for _, _, score in generator]
            df[metric.__name__] = score_pairs
            return df

        for metric in metrics:
            df = _compute_metric(graph=graph.to_undirected(), metric=metric, df=df)
        return df

    def compute_clustering_and_degree(self, graph: nx.DiGraph, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute clustering coef for each node as well as in and out degree. These features
        are added as columns to existing dataframe with edges.

        Args:
            graph (nx.Graph): Graph based on edges from `df`
            df (pd.DataFrame): DataFrame with all edges between nodes

        Returns:
            pd.DataFrame: DataFrame with calculated metrics between nodes
        """

        clustering_dict = nx.algorithms.cluster.clustering(graph)
        clustering = pd.DataFrame(clustering_dict.items(), columns=["node", "value"])

        degree_in_dict = {node: graph.in_degree(node) for node in graph.nodes()}
        degree_in = pd.DataFrame(degree_in_dict.items(), columns=["node", "value"])

        degree_out_dict = {node: graph.out_degree(node) for node in graph.nodes()}
        degree_out = pd.DataFrame(degree_out_dict.items(), columns=["node", "value"])

        page_rank_dict = nx.pagerank(graph)
        page_rank = pd.DataFrame(page_rank_dict.items(), columns=["node", "value"])

        dict_to_merge = {
            "clustering": clustering,
            "degree_in": degree_in,
            "degree_out": degree_out,
            "page_rank": page_rank,
        }
        for name, dataframe in dict_to_merge.items():
            df = (
                df.merge(dataframe, how="left", left_on="SOURCE", right_on="node")
                .drop(columns=["node"])
                .rename(columns={"value": f"{name}_1"})
            )
            df = (
                df.merge(dataframe, how="left", left_on="TARGET", right_on="node")
                .drop(columns=["node"])
                .rename(columns={"value": f"{name}_2"})
            )
        return df

    def calculate_voting_power(self, df: pd.DataFrame) -> pd.DataFrame:
        node_power = df["SOURCE"].value_counts().reset_index()
        node_power.columns = ["username", "vote_power"]
        df = df.merge(node_power, how="left", left_on="SOURCE", right_on="username").drop(
            columns=["username"]
        )
        return df

    def calculate_number_of_common_neighbors(self, graph: nx.DiGraph, df: pd.DataFrame):
        commons = {}
        g = graph.to_undirected()
        for source, target in g.edges():
            commons[(source, target)] = sum(1 for _ in nx.common_neighbors(g, source, target))
        commons_pd = pd.DataFrame.from_dict(commons, orient="index").reset_index()
        commons_pd.columns = ["nodes", "common_neighbors"]
        commons_pd["SOURCE"], commons_pd["TARGET"] = zip(*commons_pd["nodes"])
        commons_pd.drop("nodes", axis=1, inplace=True)
        df = df.merge(commons_pd, how="left", on=["SOURCE", "TARGET"])
        return df

    def create_modellng_df(self) -> pd.DataFrame:
        self.df = self.get_power_of_adj_matrix(df=self.df, g=self.graph, power=2)
        self.df = self.get_power_of_adj_matrix(df=self.df, g=self.graph, power=3)
        self.df = self.compute_metric(df=self.df, metrics=self.metrics, graph=self.graph)
        self.df = self.compute_clustering_and_degree(graph=self.graph, df=self.df)
        self.df = self.calculate_voting_power(self.df)
        self.df = self.df.drop(columns=["DAT"])
        self.df = self.calculate_number_of_common_neighbors(df=self.df, graph=self.graph)
        for col in ["SOURCE", "TARGET"]:
            self.df[col] = self.df[col].astype(int)
        return self.df

    def train_predict(
        self,
        model: lgb.LGBMClassifier,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> list[int]:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        preds = list(model.predict(X_test))
        return preds


# %%
