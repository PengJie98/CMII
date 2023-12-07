import json
from pathlib import Path
import numpy as np
import os
import random
import pickle
import math


def get_img_vec_array(proportion, img_vec_path='fb15k_vit.pickle', eutput_file='img_vec_id_fb15k_{}_vit.pickle', dim=1000):
    img_vec = pickle.load(open(img_vec_path, 'rb'))
    img_vec = {k.split('/')[-2]: v for k, v in img_vec.items()}
    f = open('./data/FB15K/ent_id'.format(proportion), 'r')
    Lines = f.readlines()

    id2ent={}
    img_vec_array=[]
    for l in Lines:
        ent, id = l.strip().split()
        id2ent[id]=ent
        if ent.replace('/', '.')[1:] in img_vec.keys():
            print(id,ent)
            # id2vec[id]=img_vec[ent.replace('/', '.')[1:]]
            img_vec_array.append(img_vec[ent.replace('/','.')[1:]])
        else:
            img_vec_array.append([0 for i in range(dim)])
    img_vec_by_id=np.array(img_vec_array)
    out=open(eutput_file.format(proportion),'wb')
    pickle.dump(img_vec_by_id,out)
    out.close()


def get_img_vec_array_forget(proportion, remember_proportion, rank_file='ent_clip_rank.json', eutput_file='rel_MPR_PD_clip_{}_mrp{}.pickle'):
    # with open(rank_file,'r') as f:
    #     Ranks=f.readlines()
    #     rel_rank={}
    #     for r in Ranks:
    #         try:
    #             rel,mrp=r.strip().split('\t')
    #         except Exception as e:
    #             print(e)
    #             print(r)
    #             continue
    #         rel_rank[rel[10:]]=float(mrp[12:])
    with open(Path(basepath) / rank_file, 'r') as f:
        rel_rank = {}
        data = json.load(f)
        for ele in data:
            rel_rank[ele['relation id']] = ele['rank']

    with open(Path(basepath) / 'rel_id','r') as f:
        Lines = f.readlines()

    rel_id_pd = []
    for l in Lines:
        rel, id = l.strip().split()
        try:
            if rel_rank[int(id)] < remember_proportion/100.0:
                rel_id_pd.append([1])
            else:
                rel_id_pd.append([0])
        except Exception as e:
            print(e)
            rel_id_pd.append([0])
            continue

    rel_id_pd=np.array(rel_id_pd)

    with open(Path(basepath) / eutput_file.format(proportion, remember_proportion),'wb') as out:
        pickle.dump(rel_id_pd,out)


def get_img_vec_sig_alpha(proportion, basepath, rank_file='ent_clip_rank.json', eutput_file='rel_MPR_SIG_clip_{}.pickle'):
    with open(Path(basepath) / rank_file, 'r') as f:
        rel_rank = {}
        data = json.load(f)
        for ele in data:
            rel_rank[ele['relation id']] = ele['rank']

    # with open(Path(basepath) / rank_file,'r') as f:
    #     Ranks=f.readlines()
    #     rel_rank={}
    #     for r in Ranks:
    #         try:
    #             rel, mrp=r.strip().split('\t')[:2]
    #             print(rel)
    #         except Exception as e:
    #             print(e)
    #             continue
    #         rel_rank[rel[13:]]=float(mrp[6:])

    with open(Path(basepath) /'rel_id','r') as f:
        Lines=f.readlines()

    rel_sig_alpha=[]
    for l in Lines:
        rel, id =l.strip().split()
        try:
            rel_sig_alpha.append([1/(1+math.exp(rel_rank[int(id)]))])
        except Exception as e:
            rel_sig_alpha.append([1 / (1 + math.exp(1))])
            continue

    rel_id_pd = np.array(rel_sig_alpha)

    with open(Path(basepath) / eutput_file.format(proportion),'wb') as out:
        pickle.dump(rel_id_pd,out)


def sample(proportion, data_path):
    # with open(data_path+'/train'+'.pickle') as f:
    #     Ls=f.readlines()
    #     L = [random.randint(0, len(Ls)-1) for _ in range(round(len(Ls)*proportion))]
    #     Lf=[Ls[l] for l in L]

    triples = pickle.load(open(Path(data_path) / ('train' + '.pickle'), 'rb'))
    L = [random.randint(0, len(triples) - 1) for _ in range(round(len(triples) * proportion))]
    Lf = triples[L]

    if not os.path.exists(data_path+'_{}/'.format(round(proportion*100))):
        os.mkdir(data_path+'_{}/'.format(round(proportion*100)))
    Ent = set()

    out=open(data_path+'_{}/train'.format(round(100*proportion))+'.pickle', 'wb')
    pickle.dump(Lf, out)
    out.close()

    # with open(data_path+'_{}/train'.format(round(100*proportion))+'.pickle','w') as f:
    for l in Lf:
        h,r,t=l
        Ent.add(h)
        Ent.add(r)
        Ent.add(t)

    triples_valid = pickle.load(open(Path(data_path) / ('valid' + '.pickle'), 'rb'))
    valid_index = []
    for i in range(len(triples_valid)):
        h, r, t = triples_valid[i]
        if h in Ent and r in Ent and t in Ent:
            valid_index.append(i)

    out=open(data_path+'_{}/valid'.format(round(100*proportion))+'.pickle', 'wb')
    pickle.dump(triples_valid[valid_index], out)
    out.close()

    triples_test = pickle.load(open(Path(data_path) / ('test' + '.pickle'), 'rb'))
    test_index = []
    for i in range(len(triples_test)):
        h, r, t = triples_test[i]
        if h in Ent and r in Ent and t in Ent:
            test_index.append(i)

    out = open(data_path + '_{}/test'.format(round(100 * proportion)) + '.pickle', 'wb')
    pickle.dump(triples_test[test_index], out)
    out.close()

    # with open(data_path+'_{}/valid'.format(round(100*proportion))+'.pickle','w') as f:
    #     for l in triples_valid:
    #         h,r,t=l.strip().split()
    #         if h in Ent and r in Ent and t in Ent:
    #             f.write(l)
    #             f.flush()
    #         else:
    #             print(l.strip()+' pass')
    #
    # with open(data_path+'/test'+'.pickle','r') as f:
    #     Ls = f.readlines()
    #
    # with open(data_path+'_{}/test'.format(round(proportion*100))+'.pickle','w') as f:
    #     for l in Ls:
    #         h, r, t = l.strip().split()
    #         if h in Ent and r in Ent and t in Ent:
    #             f.write(l)
    #             f.flush()
    #         else:
    #             print(l.strip()+' pass')


if __name__ == '__main__':
    basepath, outputfile = '../data/WN18/', '../data/WN18/'

    sample(0.5, data_path=basepath)
    # get_img_vec_array(50)
    # get_img_vec_sig_alpha(proportion=50, basepath=basepath, rank_file='ent_clip_text_rank.json', eutput_file='rel_MPR_SIG_clip_text_{}.pickle')
    # get_img_vec_sig_alpha(proportion=50, basepath=basepath)
    # get_img_vec_array_forget(50, remember_proportion=80, rank_file='ent_clip_text_rank.json', eutput_file='rel_MPR_PD_clip_text_{}_mrp{}.pickle')
    get_img_vec_array_forget(50, remember_proportion=80)

    # sample(0.3)
    # get_img_vec_array(30)
