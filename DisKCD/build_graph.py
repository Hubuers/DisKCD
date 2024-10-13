import dgl


def build_graph(type, node):
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []

    if type == 'kk_directed':
        with open(r'../data/ASSIST/graph/KK_directed.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)

        return g
    elif type == 'kuk_directed':
        with open('../data/ASSIST/graph/KUK_directed.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'ukuk_directed':
        with open(r'../data/ASSIST/graph/UKUK_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                src, dst = int(line[0]) - 100, int(line[1]) - 100
                edge_list.append((src, dst))

        g.add_edges(*zip(*edge_list))
        return g
    elif type == 'ukk_directed':
        with open(r'../data/ASSIST/graph/UKK_directed.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'kk_undirected':
        with open('../data/ASSIST/graph/KK_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'kuk_undirected':
        with open('../data/ASSIST/graph/KUK_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'ukuk_undirected':
        with open(r'../data/ASSIST/graph/UKUK_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                src, dst = int(line[0]) - 100, int(line[1]) - 100
                edge_list.append((src, dst))

        g.add_edges(*zip(*edge_list))
        return g


    elif type == 'ukk_undirected':
        with open(r'../data/ASSIST/graph/UKK_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'k_from_e':
        with open('../data/ASSIST/graph/k_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_k':
        with open('../data/ASSIST/graph/e_from_k.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'u_from_e':
        with open('../data/ASSIST/graph/u_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_u':
        with open('../data/ASSIST/graph/e_from_u.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
