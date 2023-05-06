import flwr as fl
import sys
import numpy as np
from datamaker import func


class CustomAggregationStrategy(fl.server.strategy.FedAvg):

    def __init__(self, num_clusters: int = 2, top_n: int = 3):
        super().__init__(fraction_fit=1.0, fraction_evaluate=0.5, min_fit_clients=5,
                         min_evaluate_clients=5, min_available_clients=5)
        self.num_clusters = num_clusters
        self.top_n = top_n
        self.lastResult = None

    def cluster_clients(self, results):
        # Divide clients into `self.num_clusters` clusters based on their hamming percentage
        # results.sort(key=lambda x: x[1][0])
        cluster_size = 100 // self.num_clusters
        clusters = [[] for _ in range(self.num_clusters + 1)]
        hamp = []
        for i in range(len(results)):
            hamp.append(results[i][1].metrics['ham'])
        print(hamp, len(clusters))
        minimum = min(hamp)
        maximum = max(hamp)
        diff = maximum - minimum
        for i in range(len(results)):
            cur = results[i][1].metrics['ham'] - minimum
            cur = (cur / diff) * 100
            print(cur, int(cur // cluster_size))
            clusters[int(cur // cluster_size)].append(results[i])
            # clusters[int(results[i][1].metrics['ham'] // cluster_size)].append(results[i])
        # clusters = [results[i:i + cluster_size] for i in range(0, len(results), cluster_size)]
        return clusters

    def select_top_models(self, cluster):
        # Select the top `self.top_n` models from each cluster
        cluster.sort(key=lambda x: x[1].metrics['accuracy'])
        return cluster[:min(self.top_n, len(cluster))]

    def aggregate_fit(self,
                      rnd: int,
                      results,
                      failures,
                      ):

        aggregated_weights = []

        # Average the selected models' weights
        # for result in results:
        #     # aggregated_weights.append(i)
        #     if result[1].metrics['newAccuracy'] >= result[1].metrics['oldAccuracy']:
        #         aggregated_weights.append(result)
        #
        #     # cluster_weights = [m[1][2] for m in top_models]
        #     # avg_weights = self.average_weights(cluster_weights)
        #     # aggregated_weights.append(avg_weights)
        #
        # if len(aggregated_weights):
        #     aggregated_weights = super().aggregate_fit(rnd, aggregated_weights, failures)
        #     self.lastResult = aggregated_weights
        # else:
        #     if self.lastResult:
        #         aggregated_weights = self.lastResult
        #     else:
        #         aggregated_weights = super().aggregate_fit(rnd, results, failures)

        # Save the aggregated weights
        aggregated_weights = super().aggregate_fit(rnd, results, failures)  # uncomment this
        if aggregated_weights:
            print(f"Round {rnd} - Saving aggregated weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        func()
        return aggregated_weights


# Create strategy and run server
strategy = CustomAggregationStrategy(int(sys.argv[2]), int(sys.argv[3]))
# strategy = fl.server.strategy.FedAvg()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address='localhost:' + str(sys.argv[1]),
    config=fl.server.ServerConfig(num_rounds=int(sys.argv[4])),
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy
)
