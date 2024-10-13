import argparse
from DisKCD.build_graph import build_graph

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--exer_n', type=int, default=17746)
        self.add_argument('--knowledge_n', type=int, default=100)
        self.add_argument('--unknowledge_n', type=int, default=23)
        self.add_argument('--student_n', type=int, default=2493)
        self.add_argument('--gpu', type=int, default=0)
        self.add_argument('--epoch_n', type=int, default=15)
        self.add_argument('--test', action='store_true')
        self.add_argument('--lr', type=float, default=0.001)
        self.add_argument('--batch_size', type=int, default=128)
        self.add_argument('--value_range', type=int, default=1)
        self.add_argument('--a_range', type=int, default=1)
        self.add_argument('--latent_dim', type=int, default=5)
def construct_local_map(args):
    local_map = {
        'kk_directed_g': build_graph('kk_directed', args.knowledge_n),
        'kuk_directed_g': build_graph('kuk_directed', args.knowledge_n+args.unknowledge_n),
        'ukuk_directed_g': build_graph('ukuk_directed', args.unknowledge_n),
        'ukk_directed_g': build_graph('ukk_directed', args.knowledge_n+args.unknowledge_n),
        'kk_undirected_g': build_graph('kk_undirected', args.knowledge_n),
        'kuk_undirected_g': build_graph('kuk_undirected', args.knowledge_n+args.unknowledge_n),
        'ukuk_undirected_g': build_graph('ukuk_undirected', args.unknowledge_n),
        'ukk_undirected_g': build_graph('ukk_undirected', args.knowledge_n+args.unknowledge_n),
        'k_from_e': build_graph('k_from_e', args.knowledge_n + args.exer_n),
        'e_from_k': build_graph('e_from_k', args.knowledge_n + args.exer_n),
        'u_from_e': build_graph('u_from_e', args.student_n + args.exer_n),
        'e_from_u': build_graph('e_from_u', args.student_n + args.exer_n),
    }
    return local_map

