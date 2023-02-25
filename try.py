import os
import pickle



train_data = []

for filename in os.listdir("..\\graph_data"):
    path = os.path.join("..\\graph_data", filename)
    with open(path, 'rb') as pickle_file:
        graph = pickle.load(pickle_file)
        train_data.append(graph) 
    break
print(train_data[0])