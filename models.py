from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from torch.nn import init
from features_load import *
from layer import *


class CMIIModelMM(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """

        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0

                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]

                    rhs, rhs_s, rhs_t, rhs_i, rhs_st = self.get_rhs(c_begin, chunk_size)
                    q, q_s, q_t, q_i, q_st = self.get_queries(these_queries)
                    scores_str = q @ rhs
                    scores_s = q_s @ rhs_s

                    scores_t = q_t @ rhs_t
                    scores_i = q_i @ rhs_i
                    scores_st = q_st @ rhs_st

                    scores = self.beta_s * scores_s + self.beta_m * scores_str + self.beta_i * scores_i + self.beta_t * scores_t + self.beta_st * scores_st

                    targets = self.score(these_queries)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6

                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class MGClipComplExV3(CMIIModelMM):
    def __init__(
            self, args,
            sizes: Tuple[int, int, int],
            init_size
    ):
        super(MGClipComplExV3, self).__init__()
        self.sizes = sizes
        self.args = args
        self.rank = args.rank

        self.temperature = 0.5
        self.memory_bank = None

        self.struct_relation = pickle.load(
            open(args.root + '/features/structure/{}/GAT_relation.pkl'.format(args.dataset), 'rb')).data.cuda()
        self.img_relation = pickle.load(
            open(args.root + '/features/structure/{}/GAT_relation_img.pkl'.format(args.dataset), 'rb')).data.cuda()
        self.text_relation = pickle.load(
            open(args.root + '/features/structure/{}/GAT_relation_text.pkl'.format(args.dataset), 'rb')).data.cuda()

        self.struct_relation = F.normalize(torch.Tensor(self.struct_relation), p=2, dim=1)
        self.img_relation = F.normalize(torch.Tensor(self.img_relation), p=2, dim=1)
        self.text_relation = F.normalize(torch.Tensor(self.text_relation), p=2, dim=1)

        image_encoder_path, text_encoder_path = clip_model_reload(args)
        self.image_encoder = torch.load(image_encoder_path).partial_model
        self.text_encoder = torch.load(text_encoder_path).partial_model

        img_features = self.image_encoder(args.img.cuda())
        entity_features = self.text_encoder(args.entity_features.cuda(), args.entity_eot_indices.cuda())
        relation_feature = self.text_encoder(args.relation_features.cuda(), args.relation_eot_indices.cuda())
        text_feature = self.text_encoder(args.text_features.cuda(), args.text_eot_indices.cuda())

        self. w = F.normalize(torch.Tensor(img_features), p=2, dim=1)
        self.entity = F.normalize(torch.Tensor(entity_features), p=2, dim=1)
        self.relation = F.normalize(torch.Tensor(relation_feature), p=2, dim=1)
        self.text = F.normalize(torch.Tensor(text_feature), p=2, dim=1)

        self.struct_entity = args.struct_entity
        self.struct_relation = args.struct_relation

        self.beta_s = nn.Parameter(torch.tensor(args.beta_s), requires_grad=args.trainable_beta)
        self.beta_i = nn.Parameter(torch.tensor(args.beta_i), requires_grad=args.trainable_beta)
        self.beta_t = nn.Parameter(torch.tensor(args.beta_t), requires_grad=args.trainable_beta)
        self.beta_m = nn.Parameter(torch.tensor(args.beta_m), requires_grad=args.trainable_beta)
        self.beta_st = nn.Parameter(torch.tensor(args.beta_st), requires_grad=args.trainable_beta)

        self.attention = Attention2(input_dim=768 * args.image_views, output_dim=2 * self.rank, hidden_dim=128).cuda()

        self.fusion = MFB(input_size=768, output_size=200, factor_num=200)
        # self.fusion_rel = MFB1(input_size=200, output_size=2*self.rank, factor_num=200)
        self.fusion_rel = MFB1(input_size=200, output_size=200, factor_num=200)

        self.postmat_relation = PostmatHead('pooling_linear', self.args.clip_encoder, self.rank)
        self.postmat_relation_st = PostmatHead('pooling_linear', 'GAT', self.rank)
        self.postmat_relation_i = PostmatHead('pooling_linear', 'GAT', self.rank)
        self.postmat_relation_t = PostmatHead('pooling_linear', 'GAT', self.rank)
        # self.postmat_entity = PostmatHead('partial tuning', self.args.clip_encoder, self.rank)

        self.postmat_image = PostmatHead('pooling_linear', self.args.clip_encoder, self.rank)
        self.postmat_text = PostmatHead('pooling_linear', self.args.clip_encoder, self.rank)
        self.postmat_struct = PostmatHead('pooling_linear', 'GAT', rank=self.rank)
        self.postmat_multi = PostmatHead('pooling_linear', 'GAT', self.rank)
        self.postmat_multi_rel = PostmatHead('pooling_linear', 'GAT', self.rank)

        self.postmat_relation = PostmatHead('linear', self.args.clip_encoder, self.rank)
        self.postmat_relation_st = PostmatHead('linear', 'GAT', self.rank)
        self.postmat_relation_i = PostmatHead('linear', 'GAT', self.rank)
        self.postmat_relation_t = PostmatHead('linear', 'GAT', self.rank)

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 2 * self.rank, sparse=True),
            nn.Embedding(sizes[1], 2 * self.rank, sparse=True),  # relation
        ]).cuda()

        self.entity_init = self.postmat_entity.forward(self.entity)
        self.relation_init = self.postmat_relation.forward(self.relation)
        self.relation_init_st = self.postmat_relation_st.forward(self.struct_relation)
        self.relation_init_i = self.postmat_relation_i.forward(self.img_relation)
        self.relation_init_t = self.postmat_relation_t.forward(self.text_relation)

        self.embeddings[0].weight.data = self.entity_init
        init.xavier_uniform_(self.embeddings[0].weight.data, init_size)
        self.embeddings[1].weight.data = (self.relation_init.cuda() * self.relation_init_st.cuda() *
        self.relation_init_t.cuda() * self.relation_init_i.cuda())

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        if self.args.image_views > 1:
            img_embeddings, image = self.attention.forward(self.image.view(-1, self.args.image_views, 768))
        else:
            image = self.image
            img_embeddings = self.postmat_image.forward(image)

        multi = self.fusion.forward(self.entity, image, self.text, self.struct_entity)
        ent_embeddings = self.postmat_multi(multi)
        struct_embedding = self.postmat_struct(self.struct_entity)
        text_embeddings = self.postmat_text.forward(self.text)
        multi_rel = self.fusion_rel.forward(self.relation, self.img_relation, self.text_relation, self.struct_relation)
        multi_rel_embeddings = self.postmat_multi_rel.forward(multi_rel)

        lhs = ent_embeddings[(x[:, 0])]
        rel = (multi_rel_embeddings[(x[:, 1])] + self.embeddings[1](x[:, 1])) / 2.0
        rhs = ent_embeddings[(x[:, 2])]

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        score_str = torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

        rel_s = self.embeddings[1](x[:, 1])

        lhs_s = self.embeddings[0](x[:, 0])
        rhs_s = self.embeddings[0](x[:, 2])

        lhs_s = lhs_s[:, :self.rank], lhs_s[:, self.rank:]
        rel_s = rel_s[:, :self.rank], rel_s[:, self.rank:]
        rhs_s = rhs_s[:, :self.rank], rhs_s[:, self.rank:]

        s_score = torch.sum(
            (lhs_s[0] * rel_s[0] - lhs_s[1] * rel_s[1]) * rhs_s[0] +
            (lhs_s[0] * rel_s[1] + lhs_s[1] * rel_s[0]) * rhs_s[1],
            1, keepdim=True
        )

        rel_i = (self.postmat_relation_i(self.img_relation)[(x[:, 1])] + self.embeddings[1](x[:, 1])) / 2.0
        rel_t = (self.postmat_relation_t(self.text_relation)[(x[:, 1])] + self.embeddings[1](x[:, 1])) / 2.0
        rel_st = (self.postmat_relation_st(self.struct_relation)[(x[:, 1])] + self.embeddings[1](x[:, 1])) / 2.0
        lhs_img = img_embeddings[(x[:, 0])]
        rhs_img = img_embeddings[(x[:, 2])]
        lhs_text = text_embeddings[(x[:, 0])]
        rhs_text = text_embeddings[(x[:, 2])]
        lhs_st = struct_embedding[(x[:, 0])]
        rhs_st = struct_embedding[(x[:, 2])]

        lhs_img = lhs_img[:, :self.rank], lhs_img[:, self.rank:]
        rel_i = rel_i[:, :self.rank], rel_i[:, self.rank:]
        rhs_img = rhs_img[:, :self.rank], rhs_img[:, self.rank:]
        lhs_text = lhs_text[:, :self.rank], lhs_text[:, self.rank:]
        rel_t = rel_t[:, :self.rank], rel_t[:, self.rank:]
        rhs_text = rhs_text[:, :self.rank], rhs_text[:, self.rank:]
        lhs_st = lhs_st[:, :self.rank], lhs_st[:, self.rank:]
        rel_st = rel_st[:, :self.rank], rel_st[:, self.rank:]
        rhs_st = rhs_st[:, :self.rank], rhs_st[:, self.rank:]

        i_score = torch.sum(
            (lhs_img[0] * rel_i[0] - lhs_img[1] * rel_i[1]) * rhs_img[0] +
            (lhs_img[0] * rel_i[1] + lhs_img[1] * rel_i[0]) * rhs_img[1],
            1, keepdim=True
        )

        t_score = torch.sum(
            (lhs_text[0] * rel_t[0] - lhs_text[1] * rel_t[1]) * rhs_text[0] +
            (lhs_text[0] * rel_t[1] + lhs_text[1] * rel_t[0]) * rhs_text[1],
            1, keepdim=True
        )

        st_score = torch.sum(
            (lhs_st[0] * rel_st[0] - lhs_st[1] * rel_st[1]) * rhs_st[0] +
            (lhs_st[0] * rel_st[1] + lhs_st[1] * rel_st[0]) * rhs_st[1],
            1, keepdim=True
        )

        return self.beta_s * s_score + self.beta_m * score_str + self.beta_i * i_score + self.beta_t * t_score + self.beta_st * st_score

    def forward(self, x):
        if self.args.image_views > 1:
            img_embeddings, image = self.attention.forward(self.image.view(-1, self.args.image_views, 768))
        else:
            image = self.image
            img_embeddings = self.postmat_image.forward(image)

        multi = self.fusion.forward(self.entity, image, self.text, self.struct_entity)
        ent_embeddings = self.postmat_multi(multi)
        struct_embedding = self.postmat_struct(self.struct_entity)
        text_embeddings = self.postmat_text.forward(self.text)

        multi_rel = self.fusion_rel.forward(self.relation, self.img_relation, self.text_relation, self.struct_relation)
        multi_rel_embeddings = self.postmat_multi_rel.forward(multi_rel)

        lhs = ent_embeddings[(x[:, 0])]
        rel = (multi_rel_embeddings[(x[:, 1])] + self.embeddings[1](x[:, 1])) / 2.0
        # rel = self.embeddings[1](x[:, 1])
        rhs = ent_embeddings[(x[:, 2])]

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = ent_embeddings
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]

        rel_s = self.embeddings[1](x[:, 1])
        lhs_s = self.embeddings[0](x[:, 0])
        rhs_s = self.embeddings[0](x[:, 2])

        lhs_s = lhs_s[:, :self.rank], lhs_s[:, self.rank:]
        rel_s = rel_s[:, :self.rank], rel_s[:, self.rank:]
        rhs_s = rhs_s[:, :self.rank], rhs_s[:, self.rank:]

        to_score_s = self.embeddings[0].weight
        to_score_s = to_score_s[:, :self.rank], to_score_s[:, self.rank:]

        rel_i = (self.postmat_relation_i(self.img_relation)[(x[:, 1])] + self.embeddings[1](x[:, 1])) / 2.0
        rel_t = (self.postmat_relation_t(self.text_relation)[(x[:, 1])] + self.embeddings[1](x[:, 1])) / 2.0
        rel_st = (self.postmat_relation_st(self.struct_relation)[(x[:, 1])] + self.embeddings[1](x[:, 1])) / 2.0

        lhs_img = img_embeddings[(x[:, 0])]
        rhs_img = img_embeddings[(x[:, 2])]
        lhs_text = text_embeddings[(x[:, 0])]
        rhs_text = text_embeddings[(x[:, 2])]
        lhs_st = struct_embedding[(x[:, 0])]
        rhs_st = struct_embedding[(x[:, 2])]

        lhs_img = lhs_img[:, :self.rank], lhs_img[:, self.rank:]
        rel_i = rel_i[:, :self.rank], rel_i[:, self.rank:]

        rhs_img = rhs_img[:, :self.rank], rhs_img[:, self.rank:]
        lhs_text = lhs_text[:, :self.rank], lhs_text[:, self.rank:]
        rel_t = rel_t[:, :self.rank], rel_t[:, self.rank:]
        rhs_text = rhs_text[:, :self.rank], rhs_text[:, self.rank:]
        lhs_st = lhs_st[:, :self.rank], lhs_st[:, self.rank:]
        rel_st = rel_st[:, :self.rank], rel_st[:, self.rank:]
        rhs_st = rhs_st[:, :self.rank], rhs_st[:, self.rank:]

        to_score_img = img_embeddings
        to_score_img = to_score_img[:, :self.rank], to_score_img[:, self.rank:]

        to_score_text = text_embeddings
        to_score_text = to_score_text[:, :self.rank], to_score_text[:, self.rank:]

        to_score_st = struct_embedding
        to_score_st = to_score_st[:, :self.rank], to_score_st[:, self.rank:]

        s = (torch.sqrt(lhs_s[0] ** 2 + lhs_s[1] ** 2), torch.sqrt(rel_s[0] ** 2 + rel_s[1] ** 2), torch.sqrt(rhs_s[0] ** 2 + rhs_s[1] ** 2))
        t = (torch.sqrt(lhs_text[0] ** 2 + lhs_text[1] ** 2), torch.sqrt(rel_t[0] ** 2 + rel_t[1] ** 2), torch.sqrt(rhs_text[0] ** 2 + rhs_text[1] ** 2))
        st = (torch.sqrt(lhs_st[0] ** 2 + lhs_st[1] ** 2), torch.sqrt(rel_st[0] ** 2 + rel_st[1] ** 2), torch.sqrt(rhs_st[0] ** 2 + rhs_st[1] ** 2))
        i = (torch.sqrt(lhs_img[0] ** 2 + lhs_img[1] ** 2), torch.sqrt(rel_i[0] ** 2 + rel_i[1] ** 2), torch.sqrt(rhs_img[0] ** 2 + rhs_img[1] ** 2))
        m = (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2), torch.sqrt(rel[0] ** 2 + rel[1] ** 2), torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))

        s_score = (lhs_s[0] * rel_s[0] - lhs_s[1] * rel_s[1]) @ to_score_s[0].transpose(0, 1) + \
                  (lhs_s[0] * rel_s[1] + lhs_s[1] * rel_s[0]) @ to_score_s[1].transpose(0, 1)

        m_score = (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) + \
                  (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)

        i_score = (lhs_img[0] * rel_i[0] - lhs_img[1] * rel_i[1]) @ to_score_img[0].transpose(0, 1) + \
                  (lhs_img[0] * rel_i[1] + lhs_img[1] * rel_i[0]) @ to_score_img[1].transpose(0, 1)

        t_score = (lhs_text[0] * rel_t[0] - lhs_text[1] * rel_t[1]) @ to_score_text[0].transpose(0, 1) + \
                  (lhs_text[0] * rel_t[1] + lhs_text[1] * rel_t[0]) @ to_score_text[1].transpose(0, 1)

        st_score = (lhs_st[0] * rel_st[0] - lhs_st[1] * rel_st[1]) @ to_score_st[0].transpose(0, 1) + \
                   (lhs_st[0] * rel_st[1] + lhs_st[1] * rel_st[0]) @ to_score_st[1].transpose(0, 1)

        return self.beta_s * s_score + self.beta_m * m_score + self.beta_i * i_score + self.beta_t * t_score + self.beta_st * st_score, [m, s, t, st, i]

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        if self.args.image_views > 1:
            img_embeddings, image = self.attention.forward(self.image.view(-1, self.args.image_views, 768))
        else:
            image = self.image
            img_embeddings = self.postmat_image.forward(image)

        multi = self.fusion.forward(self.entity, image, self.text, self.struct_entity)
        ent_embeddings = self.postmat_multi(multi)
        struct_embedding = self.postmat_struct(self.struct_entity)
        text_embeddings = self.postmat_text.forward(self.text)
        ent_embeddings_s = self.embeddings[0].weight

        return (ent_embeddings[chunk_begin:chunk_begin + chunk_size].transpose(0, 1),
                ent_embeddings_s[chunk_begin:chunk_begin + chunk_size].transpose(0, 1),
                text_embeddings[chunk_begin:chunk_begin + chunk_size].transpose(0, 1),
                img_embeddings[chunk_begin:chunk_begin + chunk_size].transpose(0, 1),
                struct_embedding[chunk_begin:chunk_begin + chunk_size].transpose(0, 1))

    def get_queries(self, queries: torch.Tensor):
        if self.args.image_views > 1:
            img_embeddings, image = self.attention.forward(self.image.view(-1, self.args.image_views, 768))
        else:
            image = self.image
            img_embeddings = self.postmat_image.forward(image)

        multi = self.fusion.forward(self.entity, image, self.text, self.struct_entity)
        ent_embeddings = self.postmat_multi(multi)
        struct_embedding = self.postmat_struct(self.struct_entity)
        text_embeddings = self.postmat_text.forward(self.text)

        multi_rel = self.fusion_rel.forward(self.relation, self.img_relation, self.text_relation, self.struct_relation)
        multi_rel_embeddings = self.postmat_multi_rel.forward(multi_rel)  # main here

        rel_i = (self.postmat_relation_i(self.img_relation)[(queries[:, 1])] + self.embeddings[1](queries[:, 1])) / 2.0
        rel_t = (self.postmat_relation_t(self.text_relation)[(queries[:, 1])] + self.embeddings[1](queries[:, 1])) / 2.0
        rel_st = (self.postmat_relation_st(self.struct_relation)[(queries[:, 1])] + self.embeddings[1](queries[:, 1])) / 2.0

        lhs = ent_embeddings[(queries[:, 0])]
        rel = (multi_rel_embeddings[(queries[:, 1])] + self.embeddings[1](queries[:, 1])) / 2.0
        rel_s = self.embeddings[1](queries[:, 1])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        lhs_s = self.embeddings[0](queries[:, 0])
        lhs_s = lhs_s[:, :self.rank], lhs_s[:, self.rank:]
        rel_s = rel_s[:, :self.rank], rel_s[:, self.rank:]

        lhs_img = img_embeddings[queries[:, 0]]
        lhs_text = text_embeddings[queries[:, 0]]
        lhs_st = struct_embedding[queries[:, 0]]

        lhs_img = lhs_img[:, :self.rank], lhs_img[:, self.rank:]
        lhs_text = lhs_text[:, :self.rank], lhs_text[:, self.rank:]
        lhs_st = lhs_st[:, :self.rank], lhs_st[:, self.rank:]

        rel_i = rel_i[:, :self.rank], rel_i[:, self.rank:]
        rel_t = rel_t[:, :self.rank], rel_t[:, self.rank:]
        rel_st = rel_st[:, :self.rank], rel_st[:, self.rank:]

        return (torch.cat([lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]], 1),
                torch.cat([lhs_s[0] * rel_s[0] - lhs_s[1] * rel_s[1], lhs_s[0] * rel_s[1] + lhs_s[1] * rel_s[0]], 1),
                torch.cat([lhs_text[0] * rel_t[0] - lhs_text[1] * rel_t[1], lhs_text[0] * rel_t[1] + lhs_text[1] * rel_t[0]],
                          1),
                torch.cat([lhs_img[0] * rel_i[0] - lhs_img[1] * rel_i[1], lhs_img[0] * rel_i[1] + lhs_img[1] * rel_i[0]], 1),
                torch.cat([lhs_st[0] * rel_st[0] - lhs_st[1] * rel_st[1], lhs_st[0] * rel_st[1] + lhs_st[1] * rel_st[0]], 1))