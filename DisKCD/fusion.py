import torch
import torch.nn as nn
import torch.nn.functional as F
from .GraphLayer import GraphLayer

class Fusion(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n + args.unknowledge_n
        self.k_n = args.knowledge_n
        self.uk_n = args.unknowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim

        # graph structure
        self.kk_directed_g = local_map['kk_directed_g'].to(self.device)
        self.kuk_directed_g = local_map['kuk_directed_g'].to(self.device)
        self.ukuk_directed_g = local_map['ukuk_directed_g'].to(self.device)
        self.ukk_directed_g = local_map['ukk_directed_g'].to(self.device)
        self.kk_undirected_g = local_map['kk_undirected_g'].to(self.device)
        self.kuk_undirected_g = local_map['kuk_undirected_g'].to(self.device)
        self.ukuk_undirected_g = local_map['ukuk_undirected_g'].to(self.device)
        self.ukk_undirected_g = local_map['ukk_undirected_g'].to(self.device)
        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)
        self.u_from_e = local_map['u_from_e'].to(self.device)
        self.e_from_u = local_map['e_from_u'].to(self.device)

        super(Fusion, self).__init__()

        self.kk_directed_gat = GraphLayer(self.kk_directed_g, self.knowledge_dim, self.knowledge_dim)
        self.kuk_directed_gat = GraphLayer(self.kuk_directed_g, self.knowledge_dim, self.knowledge_dim)
        self.ukuk_directed_gat = GraphLayer(self.ukuk_directed_g, self.knowledge_dim, self.knowledge_dim)
        self.ukk_directed_gat = GraphLayer(self.ukk_directed_g, self.knowledge_dim, self.knowledge_dim)

        self.kk_undirected_gat = GraphLayer(self.kk_undirected_g, self.knowledge_dim, self.knowledge_dim)
        self.kuk_undirected_gat = GraphLayer(self.kuk_undirected_g, self.knowledge_dim, self.knowledge_dim)
        self.ukuk_undirected_gat = GraphLayer(self.ukuk_undirected_g, self.knowledge_dim, self.knowledge_dim)
        self.ukk_undirected_gat = GraphLayer(self.ukk_undirected_g, self.knowledge_dim, self.knowledge_dim)


        self.k_from_e = GraphLayer(self.k_from_e, self.knowledge_dim, self.knowledge_dim)  # src: e
        self.e_from_k = GraphLayer(self.e_from_k, self.knowledge_dim, self.knowledge_dim)  # src: k

        self.u_from_e = GraphLayer(self.u_from_e, self.knowledge_dim, self.knowledge_dim)  # src: e
        self.e_from_u = GraphLayer(self.e_from_u, self.knowledge_dim, self.knowledge_dim)  # src: u

        self.k_attn_fc1 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.k_attn_fc2 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.k_attn_fc3 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.k_attn_fc4 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.k_attn_fc5 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)

        self.uk_attn_fc1 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.uk_attn_fc2 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.uk_attn_fc3 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.uk_attn_fc4 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.uk_attn_fc5 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)

        self.e_attn_fc1 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.e_attn_fc2 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)
        self.e_attn_fc3 = nn.Linear(2 * self.knowledge_dim, 1, bias=True)

    def forward(self, kn_emb, exer_emb, all_stu_emb,ukn_emb):
        kk_directed = self.kk_directed_gat(kn_emb)
        ukuk_directed = self.ukuk_directed_gat(ukn_emb)

        k_uk_directed_graph=torch.cat((kn_emb,ukn_emb),dim=0)
        kuk_directed_graph = self.kuk_directed_gat(k_uk_directed_graph)
        ukk_directed_graph = self.ukk_directed_gat(k_uk_directed_graph)



        kk_undirected = self.kk_undirected_gat(kn_emb)
        ukuk_undirected = self.ukuk_undirected_gat(ukn_emb)

        k_uk_undirected_graph = torch.cat((kn_emb, ukn_emb), dim=0)
        kuk_undirected_graph = self.kuk_undirected_gat(k_uk_undirected_graph)
        ukk_undirected_graph = self.ukk_undirected_gat(k_uk_undirected_graph)


        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        k_from_e_graph = self.k_from_e(e_k_graph)
        e_from_k_graph = self.e_from_k(e_k_graph)


        e_u_graph = torch.cat((exer_emb, all_stu_emb), dim=0)
        u_from_e_graph = self.u_from_e(e_u_graph)
        e_from_u_graph = self.e_from_u(e_u_graph)

        # update knowledge
        A = kn_emb
        B = k_from_e_graph[self.exer_n:]
        C = kk_undirected
        D = kk_directed
        E = kuk_undirected_graph[0:self.k_n]
        G = kuk_directed_graph[0:self.k_n]

        concat_c_1 = torch.cat([A, B], dim=1)
        concat_c_2 = torch.cat([A, C], dim=1)
        concat_c_3 = torch.cat([A, D], dim=1)
        concat_c_4 = torch.cat([A, E], dim=1)
        concat_c_5 = torch.cat([A, G], dim=1)

        score1 = self.k_attn_fc1(concat_c_1)
        score2 = self.k_attn_fc2(concat_c_2)
        score3 = self.k_attn_fc3(concat_c_3)
        score4 = self.k_attn_fc4(concat_c_4)
        score5 = self.k_attn_fc5(concat_c_5)
        input_score=torch.cat([torch.cat([torch.cat([torch.cat([score1,score2],dim=1),score3],dim=1),score4],dim=1),score5],dim=1)
        score = F.softmax(input_score,dim=1)

        kn_emb = A + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C + score[:, 2].unsqueeze(1) * D +score[:,3].unsqueeze(1) * E +score[:,4].unsqueeze(1) * G

        # updata unknowledge
        A = ukn_emb
        B = ukuk_undirected
        C = ukuk_directed
        D = ukk_undirected_graph[self.k_n:]
        E = ukk_directed_graph[self.k_n:]

        concat_c_1 = torch.cat([A, B], dim=1)
        concat_c_2 = torch.cat([A, C], dim=1)
        concat_c_3 = torch.cat([A, D], dim=1)
        concat_c_4 = torch.cat([A, E], dim=1)


        score1 = self.uk_attn_fc2(concat_c_1)
        score2 = self.uk_attn_fc3(concat_c_2)
        score3 = self.uk_attn_fc4(concat_c_3)
        score4 = self.uk_attn_fc5(concat_c_4)

        score = F.softmax(torch.cat([torch.cat([torch.cat([score1,score2],dim=1),score3],dim=1),score4],dim=1),dim=1)
        ukn_emb = A + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C + score[:, 2].unsqueeze(1) * D + score[:,3].unsqueeze(
            1) * E

        # updated exercises
        A = exer_emb
        B = e_from_u_graph[0: self.exer_n]
        C = e_from_k_graph[0: self.exer_n]
        concat_e_1 = torch.cat([A, B], dim=1)
        concat_e_2 = torch.cat([A, C], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        score2 = self.e_attn_fc2(concat_e_2)
        score = F.softmax(torch.cat([score1, score2],dim=1),dim=1)
        exer_emb = exer_emb + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C

        # updated students
        all_stu_emb = all_stu_emb + u_from_e_graph[self.exer_n:]

        return kn_emb, exer_emb, all_stu_emb, ukn_emb
