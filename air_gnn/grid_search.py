import json

from model import GNN_Model, load_graph

hidden_nodes = [16, 32]
learning_rates = [0.001, 0.005]
weight_decays = [0, 1e-5]


performances = {}

for hidden_node in hidden_nodes:
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            name = f"H{hidden_node}_LR{learning_rate}_WD{weight_decay}"
            print(f"Training {name}")
            train_graph = load_graph("appartements/graphs/train/nx_Graph.pickle")
            test_graph = load_graph("appartements/graphs/test/nx_Graph.pickle")

            train_labels = json.load(open("appartements/graphs/train/labels.json"))
            test_labels = json.load(open("appartements/graphs/test/labels.json"))

            model = GNN_Model(
                train_graph=train_graph,
                train_labels=train_labels,
                test_graph=test_graph,
                test_labels=test_labels,
                hidden_message=hidden_node,
                hidden_update=hidden_node,
                hidden_readout=hidden_node,
                optim_learning_rate=learning_rate,
                optim_weight_decay=weight_decay,
                model_name=name,
            )

            model.train(epochs=500)
            model.save(f"models/{name}.pickle")
            performances[name] = model.evaluate()
            print("DONE!", performances[name])

json.dump(performances, open("logs/performances.json", "w"))
