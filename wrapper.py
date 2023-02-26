import numpy as np

class GraphWrapper:
    def __init__(self, graph):
        self.graph = graph

    def __hash__(self) -> int:
        g = self.graph
        nodeTypes = sorted(g.vs["type"])
        n2 =  "".join([str(nodeType) for nodeType in nodeTypes + [0]] + [str(neighbour)  for nodeType in nodeTypes for neighbour in sorted([g.vs[nodeIndex]["type"] for nodeIndex in g.neighbors(g.vs.find(type = nodeType), 'in')])+[0]])
        return int(n2)
        # n_c = []
        # c_c = {}
        # adj = self.graph.get_adjacency()
        # for i_l,l in enumerate(adj[:]):
        #     if len(c_c) == 0:
        #             c_c[self.graph.vs["type"][i_l]] = i_l
        #     if i_l != adj.shape[0]-1 and l[i_l+1] == 0:
        #         c_c[self.graph.vs["type"][i_l+1]] = i_l+1
        #     else:
        #         n_c.append(c_c)
        #         c_c = {}

        # e_c = []
        # for n in n_c:
        #     x = dict(sorted(n.items()))
        #     e_c.append(x)

        # res = []


        # res = [adj[s_e] for e in e_c for s_e in e.values()]
        # res_labels = [s_e for e in e_c for s_e in e.keys()]
        # r_n = np.array(res)
        # res2 = [r_n[:,s_e] for e in e_c for s_e in e.values()]
        # flat_list = [item for sublist in res2 for item in sublist]
        # flat_list_str = ''.join(map(str, flat_list))
        # flat_list_int = int(flat_list_str,2)
        # flat_list_str = str(flat_list_int)
        # res_labels_str = ''.join(map(str,res_labels))
        # hash_list = flat_list_str + res_labels_str
        # hash_list_str = ''.join(map(str,hash_list))
        # return int(hash_list_str)


