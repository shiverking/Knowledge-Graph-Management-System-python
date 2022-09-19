import os
from helper import *
from data_loader import *
from model import *
from d2l import torch as d2l
import pandas as pd
from termcolor import colored

import numpy as np
from py2neo import Graph, Node, Relationship, Graph, NodeMatcher, RelationshipMatcher
import pandas as pd

graph = Graph('http://192.168.100.114:7474', auth=('neo4j', 'Sicdp2021cjw'))

# 查找实体类型对应实例，返回list
def get_all_entities_of_ent_typ(graph, ent_typ):
    matcher = NodeMatcher(graph)
    ent_list = list(matcher.match(ent_typ))
    ent_list = [ent['name'] for ent in ent_list]
    return ent_list

# 查找所有的实体类型
def get_all_entity_types(graph):
    return list(graph.schema.node_labels)

# 查找所有的关系类型
def get_all_relationship_types(graph):
    return list(graph.schema.relationship_types)

# 三元组插入neo4j
def triples2neo4j(graph, triples, one2many=False, many2one=False):
    for triple in triples:
        # 取出头实体、尾实体、关系
        head, head_typ, rel, tail, tail_typ = triple
        head_node = Node(head_typ, name=head)
        # tail, tail_typ = ent_2
        tail_node = Node(tail_typ, name=tail)
        # head类型list
        head_list = get_all_entities_of_ent_typ(graph, head_typ)
        # tail类型list
        tail_list = get_all_entities_of_ent_typ(graph, tail_typ)
        # 头实体和尾实体都存在 """暂且不判断正误"""
        if head in head_list and tail in tail_list:
            graph.merge(head_node, head_typ, "name")
            graph.merge(tail_node, tail_typ, "name")
            if list(RelationshipMatcher(graph).match((head_node, tail_node), r_type = rel)):
                print(f'三元组 ({head} ,{head_typ}, {rel}, {tail}, {tail_typ}) 已存在于图谱中，插入失败！')
            else:
                graph.create(Relationship(head_node, rel, tail_node))
                print(f'三元组 ({head} ,{head_typ}, {rel}, {tail}, {tail_typ}) 插入成功！')
        # 头实体已存在 """暂且不判断正误"""
        elif head in head_list and tail not in tail_list:
            graph.merge(head_node, head_typ, "name")
            if list(RelationshipMatcher(graph).match((head_node, None), r_type = rel)):
                if one2many == False:
                    print(f'头实体 {head} 已存在关系 {rel} 对应的三元组 ({head} ,{head_typ}, {rel}, {tail}, {tail_typ})，插入失败！')
                    continue
            graph.create(tail_node)
            graph.create(Relationship(head_node, rel, tail_node))
            print(f'三元组 ({head} ,{head_typ}, {rel}, {tail}, {tail_typ}) 插入成功！')
        # 尾实体已存在 """暂且不判断正误"""
        elif head not in head_list and tail in tail_list:
            graph.merge(tail_node, tail_typ, "name")
            if list(RelationshipMatcher(graph).match((None, tail_node), r_type = rel)):
                if many2one == False:
                    print(f'尾实体 {tail} 已存在关系 {rel} 对应的三元组 ({head} ,{head_typ}, {rel}, {tail}, {tail_typ})，插入失败！')   
                    continue             
            graph.create(head_node)
            graph.create(Relationship(head_node, rel, tail_node))
            print(f'三元组 ({head} ,{head_typ}, {rel}, {tail}, {tail_typ}) 插入成功！')
        # 头实体、尾实体均不存在
        else:                    
            graph.create(head_node)
            graph.create(tail_node)
            graph.create(Relationship(head_node, rel, tail_node))
            print(f'三元组 ({head} ,{head_typ}, {rel}, {tail}, {tail_typ}) 插入成功！')

#获取所有尾实体
def get_all_tail_and_tail_typ():
    train_set = pd.read_excel(f'/home/cjw/KGtest/train.xlsx').values.tolist()
    obj_set = []
    for line in train_set:
        sub, sub_typ, rel, obj , obj_typ= line[0], line[1], line[4], line[2], line[3]
        obj_set.append((obj, obj_typ))
    return obj_set

# 抽取训练集数据对尾实体随机替换
def create_triples_for_link_prediction(train_set_30):
    obj_set = []
    triple_list = []
    res = []
    for line in train_set_30:
        sub, sub_typ, rel, obj , obj_typ= line[0], line[1], line[4], line[2], line[3]
        triple_list.append([sub, sub_typ, rel, obj , obj_typ])
        obj_set.append((obj, obj_typ))
        obj_set = list(set(obj_set))
    for triple in triple_list:
        triple[3], triple[4] = random.choice(obj_set)
        res.append((triple[0], triple[1], triple[2] ,triple[3], triple[4]))
    return res

# 三元组检测
def triple_classification(model, triples):
    model.load_model("/home/cjw/KGtest/ConvE/models/testrun_15_08_2022_15:38:29")
    res = []
    for triple in triples:
        sum = 0
        list = model.link_prediction(triple[0], triple[2])[:]
        if list:
            tuple_list = [tuple[0] for tuple in list]
            if triple[3] in tuple_list:
                for tuple in list:
                    sum = sum + tuple[2]
                score = sum / len(list)
                print(f"{triple} 检测通过，置信度{score}！")
                res.append(triple)
            elif triple[3] not in tuple_list:
                print(f"{triple} 检测不通过")
        else:
            print(f"{triple} 检测不通过")
    return res

class Main(object):

    def load_data(self):

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in pd.read_excel(f'/home/cjw/KGtest/{split}.xlsx').values.tolist():
                sub, rel, obj = line[0], line[4], line[2]
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})
        

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            for line in pd.read_excel(f'/home/cjw/KGtest/{split}.xlsx').values.tolist():
                sub, rel, obj = line[0], line[4], line[2]
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]

                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)
        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})


        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train'	:   get_data_loader(TrainDataset, 'train', 	self.p.batch_size),
            'valid_head'	:   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
            'valid_tail'	:   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
            'test_head'	:   get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
            'test_tail'	:   get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
        }

    def __init__(self, params):
        self.p = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        # self.logger.info(vars(self.p))
        # pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            # self.device = d2l.try_all_gpus()
            # print(self.device)
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model = self.add_model()
        self.optimizer = self.add_optimizer(self.model.parameters())

    def add_model(self):
        model = ConvE(self.p)
        model.to(self.device)
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
        return model

    def add_optimizer(self, parameters):
        if self.p.opt == 'adam' : return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
        else                    : return torch.optim.SGD(parameters,  lr=self.p.lr, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        if split == 'train':
            triple, label = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [ _.to(self.device) for _ in batch]
            # print(batch)
            # print(triple)
            # print(label)
            # print(triple[:, 0], triple[:, 1], triple[:, 2], label)
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        print(f"saving model to {save_path}...")
        state = {
            'state_dict'	: self.model.state_dict(),
            'best_val'	: self.best_val,
            'best_test'	: self.best_test,
            'best_epoch'	: self.best_epoch,
            'optimizer'	: self.optimizer.state_dict(),
            'args'		: vars(self.p)
        }
        torch.save(state, save_path)
        print(f"model has been saved...")

    def load_model(self, load_path):
        state				= torch.load(load_path)
        state_dict			= state['state_dict']
        self.best_val_mrr 		= state['best_val']['mrr']
        self.best_test_mrr 		= state['best_test']['mrr']
        self.best_test 			= state['best_test']
        self.best_val 			= state['best_val']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch):
        left_results   = self.predict(split=split, mode='tail_batch')
        right_results  = self.predict(split=split, mode='head_batch')

        results       = get_combined_results(left_results, right_results)
        self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(
            epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
        return results

    def link_prediction_all(self, b_size:int = 128, threshold:float = 0.5):
        self.model.eval()
        links = []
        for e,eid in list(self.ent2id.items()):
            for r,rid in list(self.rel2id.items())[:int(len(self.rel2id) / 2)]:
                links.append((eid, rid))
        self.logger.info(f"link prediction info: have {len(links)} links, batch_size {b_size}, threshold: {threshold}")
        predictions = []
        # all_links = links.copy()
        all_len = len(links)
        while len(links) > 0:
            b_links, links = links[:b_size], links[b_size:]
            b_tensor = torch.Tensor(b_links).long().to(self.device)
            with torch.no_grad():
                # split into head tensor and tail tensor
                h_tensor, r_tensor = b_tensor[:,0],b_tensor[:,1]
                preds = self.model.forward(h_tensor, r_tensor)
                preds = preds.data.tolist()
                for (hid, rid), pred in zip(b_links, preds):
                    h_name, r_name = self.id2ent[hid], self.id2rel[rid]
                    # sort predictions
                    pred = [(self.id2ent[i], i, v) for i, v in sorted(enumerate(pred), key=lambda x: x[1])]
                    pred.reverse()
                    # filtered predictions via scores
                    pred = [item for item in pred if item[2] >= threshold]
                    ents = [name for name, id, score in pred]
                    tails = []
                    # filtered predictions via training set
                    if (hid, rid) in self.sr2o_all.keys():
                        tails = [self.id2ent[t] for t in self.sr2o_all[(hid, rid)]]
                        for tail, i, v in pred:
                            if tail not in tails:
                                predictions.append((h_name, r_name, tail, v))

                        self.logger.info(f"h: {h_name} r: {r_name} prediction over threshold {threshold} {pred}")
                        self.logger.info(f"expected tail is {tails}, hits at {[ents.index(tail) + 1 for tail in tails if tail in ents]}")

                    else:
                        for tail, i, v in pred:
                            predictions.append((h_name, r_name, tail, v))

            with open(f"/home/cjw/kg-reeval/ConvE/results/{self.p.name}", encoding='utf-8', mode='a') as file:
                for h,r,t,v in predictions:
                    file.write(f"{h}\t{r}\t{t}\t{v}\n")
            predictions = []
            self.logger.info(f"predicts {(all_len - len(links))} residue {len(links)}...")

        self.logger.info(f"link prediction finished.")

    def link_prediction(self, h_name, r_name, n:int = 100, threshold:float = 0.5):
        self.model.eval()
        h, r = self.ent2id[h_name], self.rel2id[r_name]
        h_tensor = torch.IntTensor([h]).to(self.device)
        r_tensor = torch.IntTensor([r]).to(self.device)
        with torch.no_grad():
            pred = self.model.forward(h_tensor, r_tensor)
            pred = pred.data.tolist()[0]
            pred = [(self.id2ent[i], i,v) for i,v in sorted(enumerate(pred), key=lambda x:x[1])]
            pred = [item for item in pred if item[2] >= threshold]
            pred = pred[-n:]
            pred.reverse()
            ents = [name for name,id,score in pred]
            try:
                tails = [self.id2ent[t] for t in self.sr2o_all[(h,r)]]
                # print(f"{h_name} {r_name} top {n} prediction over threshold {threshold} {pred}")
                # print(f"expected tail is {tails}, hits at {[ents.index(tail) + 1 for tail in tails if tail in ents]}")
            except KeyError:
               pass
        return pred

    def link_prediction_for_generation_triples(self, h_name, r_name, n:int = 100, threshold:float = 0.5):
        self.model.eval()
        h, r = self.ent2id[h_name], self.rel2id[r_name]
        h_tensor = torch.IntTensor([h]).to(self.device)
        r_tensor = torch.IntTensor([r]).to(self.device)
        with torch.no_grad():
            pred = self.model.forward(h_tensor, r_tensor)
            pred = pred.data.tolist()[0]
            pred = [(self.id2ent[i], i,v) for i,v in sorted(enumerate(pred), key=lambda x:x[1])]
            pred = [item for item in pred if item[2] >= threshold]
            pred = pred[-n:]
            pred.reverse()
            ents = [name for name,id,score in pred]
            try:
                tails = [self.id2ent[t] for t in self.sr2o_all[(h,r)]]
                print(f"{h_name} {r_name} top {n} prediction over threshold {threshold} {pred}")
                print(f"expected tail is {tails}, hits at {[ents.index(tail) + 1 for tail in tails if tail in ents]}")
            except KeyError:
               print('当前头 实体-关系 组合 {h_name} , {r_name} 不存在，无法预测！')
        return pred

    def predict(self, split='valid', mode='tail_batch'):
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label	= self.read_batch(batch, split)
                pred			= self.model.forward(sub, rel)

                b_range			= torch.arange(pred.size()[0], device=self.device)
                target_pred		= pred[b_range, obj]
                pred 			= torch.where(label.byte(), torch.zeros_like(pred), pred)	# Filtering
                pred[b_range, obj] 	= target_pred

                pred = pred.cpu().numpy()
                obj  = obj.cpu().numpy()

                for i in range(pred.shape[0]):
                    scores	= pred[i]
                    target	= obj[i]
                    tar_scr	= scores[target]
                    scores  = np.delete(scores, target)

                    if self.p.eval_type == 'top':
                        scores = np.insert(scores, 0, tar_scr)

                    elif self.p.eval_type == 'bottom':
                        scores = np.concatenate([scores, [tar_scr]], 0)

                    elif self.p.eval_type == 'random':
                        rand = np.random.randint(scores.shape[0])
                        scores = np.insert(scores, rand, tar_scr)

                    else:
                        raise NotImplementedError

                    sorted_indices = np.argsort(-scores, kind='stable')

                    if 	self.p.eval_type == 'top':
                        _filter = np.where(sorted_indices == 0)[0][0]
                    elif 	self.p.eval_type == 'bottom':
                        _filter = np.where(sorted_indices == scores.shape[0] -1)[0][0]
                    elif 	self.p.eval_type == 'random':
                        _filter = np.where(sorted_indices == rand)[0][0]
                    else:
                        raise NotImplementedError

                    results['count']	= 1 + results.get('count', 0.0)
                    results['mr']		= (_filter + 1) + results.get('mr', 0.0)
                    results['mrr']		= (1.0 / (_filter + 1)) + results.get('mrr', 0.0)

                    for k in range(10):
                        if _filter <= k:
                            results['hits@{}'.format( k +1)] = 1 + results.get('hits@{}'.format( k +1), 0.0)

                if step % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

        return results

    def run_epoch(self, epoch, val_mrr = 0):
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()

            sub, rel, obj, label = self.read_batch(batch, 'train')

            pred	= self.model.forward(sub, rel)
            loss	= self.model.module.loss(pred, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if step % 100 == 0:
                self.logger.info \
                    ('[E:{} | {}]: Train Loss:{:.4},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def fit(self):
        self.best_val_mrr, self.best_val, self.best_test, self.best_test_mrr, self.best_epoch = 0., {}, {}, 0., 0.
        val_mrr = 0
        save_path = os.path.join('./models', self.p.name)

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')
            test_results = self.evaluate('test', -1)
            pprint(test_results)

        kill_cnt = 0
        for epoch in range(self.p.max_epochs):
            train_loss = self.run_epoch(epoch, val_mrr)

            val_results = self.evaluate('valid', epoch)

            if val_results['mrr'] > self.best_val_mrr:
                test_results		= self.evaluate('test', epoch)
                self.best_val		= val_results
                self.best_val_mrr	= val_results['mrr']
                self.best_test		= test_results
                self.best_test_mrr	= self.best_test['mrr']
                self.best_epoch		= epoch
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt > 20:
                    model.save_model(f"./models/{args.name}")
                    exit(0)

            self.logger.info \
                ('[Epoch {}]:  Training Loss: {:.5},  Valid MRR: {:.5},  Test MRR: {:.5}\n\n\n'.format(epoch, train_loss, self.best_val_mrr, self.best_test_mrr))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-name",            default='testrun',					help="Set filename for saving or restoring models")

    parser.add_argument('-batch',           dest="batch_size",      default=128,    type=int,       help='Batch size')

    parser.add_argument("-gpu",		type=str,               default='0',			help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0, For Multiple GPU = 0,1")
    parser.add_argument("-l2",		type=float,             default=0.0,			help="L2 Regularization for Optimizer")
    parser.add_argument("-lr",		type=float,             default=0.001,			help="For Standard/Adaptive: Starting Learning Rate, CLR : Minimum LR")
    parser.add_argument('-loss',            dest="loss",		default='bce',			help='GPU to use')

    parser.add_argument("-lbl_smooth",      dest='lbl_smooth',	type=float,    default=0.1,	help="Label Smoothing enable or disable, reqd True")
    parser.add_argument('-no_rev',         	dest="reverse",         action='store_false',           help='Use uniform weighting')
    parser.add_argument('-nosave',       	dest="save",       	action='store_false',           help='Negative adversarial sampling')

    parser.add_argument("-epoch",		dest='max_epochs', 	type=int,         default=300,  help="Number of epochs")
    parser.add_argument("-num_workers",	type=int,               default=5,                      help="For CPU:0, For GPU Serial:1, For GPU PS and COLLECTIVE_ALL_REDUCE: 1+")
    parser.add_argument("-embed_dim",	type=int,              	default=None,                   help="For CPU:0, For GPU Serial:1, For GPU PS and COLLECTIVE_ALL_REDUCE: 1+")

    parser.add_argument('-opt',             dest="opt",             default='adam',                 help='GPU to use')
    parser.add_argument('-restore',         dest="restore",         action='store_true',            help='Restore from the previously saved model')
    parser.add_argument('-seed',            dest="seed",            default=42,   type=int,       	help='Seed for randomization')
    parser.add_argument('-bias',            dest="bias",            action='store_true',            help='Restore from the previously saved model')

    parser.add_argument('-eval_type',       dest="eval_type",       default='random',           	help='Evaluation Protocol to use. Options: top/bottom/random')

    parser.add_argument('-form',		type=str,               default='alternate',            help='Input concat form')
    parser.add_argument('-k_w',	  	dest="k_w", 		default=10,   	type=int, 	help='k_w')
    parser.add_argument('-k_h',	  	dest="k_h", 		default=20,   	type=int, 	help='k_h')

    parser.add_argument('-inp_drop',  	dest="inp_drop", 	default=0.2,  	type=float,	help='Dropout for full connected layer')
    parser.add_argument('-hid_drop',  	dest="hid_drop", 	default=0.3,  	type=float,	help='Dropout for full connected layer')
    parser.add_argument('-feat_drop', 	dest="feat_drop", 	default=0.3,  	type=float,	help='Dropout for full connected layer')

    parser.add_argument('-num_filt',  	dest="num_filt", 	default=32,   	type=int, 	help='Number of filters in convolution')
    parser.add_argument('-ker_sz',    	dest="ker_sz", 		default=3,   	type=int, 	help='ker_sz')

    parser.add_argument('-logdir',          dest="log_dir",         default='./log/',               help='Log directory')
    parser.add_argument('-config',          dest="config_dir",      default='./config/',           help='Config directory')

    args = parser.parse_args()

    if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 60%的训练集数据插入图谱
    print(colored('将训练集60%的数据插入图谱........', attrs=['bold']))
    train_set = pd.read_excel(f'/home/cjw/KGtest/train.xlsx')
    first_insert, others = np.split(train_set.sample(frac=1), [int(.6*len(train_set))])
    del first_insert['relation']
    del others['relation']
    first_insert = first_insert.values.tolist()
    others = others.values.tolist()
    first_insert_res = []
    for insert in first_insert:
        first_insert_res.append((insert[0], insert[1], insert[4], insert[2], insert[3]))
    triples2neo4j(graph, first_insert_res)
    input()

    # 对测试集和验证集的数据进行三元组检测
    print(colored('对测试集/验证集进行三元组质检........', attrs=['bold']))
    test_and_valid_triples = []
    model = Main(args)
    for split in ['test', 'valid']:
        for line in pd.read_excel(f'/home/cjw/KGtest/{split}.xlsx').values.tolist():
            test_and_valid_triples.append((line[0],line[1], line[4], line[2], line[3]))
    test_and_valid_res = triple_classification(model, test_and_valid_triples)
    input()

    # 40%的训练集数据，对尾实体进行随即替换  
    print(colored('40%的训练集数据，对尾实体进行随即替换........', attrs=['bold']))
    generation_triples = create_triples_for_link_prediction(others)
    print(colored('替换生成的三元组如下........', attrs=['bold']))
    pprint(generation_triples)
    input()
    print(colored('对替换生成的三元组进行三元组质检........', attrs=['bold']))
    generation_triples_res = triple_classification(model, generation_triples)
    input()

    # 对样本三元组进行链接预测
    print(colored('对替换生成的三元组进行链接预测........', attrs=['bold']))
    generation_triples_after_pre = []
    obj_set = get_all_tail_and_tail_typ()
    for generation_triple in generation_triples:
        link_prediction_for_generation_res= model.link_prediction_for_generation_triples(generation_triple[0], generation_triple[2])
        if link_prediction_for_generation_res:
            for obj in obj_set:
                if obj[0] == link_prediction_for_generation_res[0][0]:
                    obj_typ = obj[1]
            generation_triples_after_pre.append((generation_triple[0], generation_triple[1], generation_triple[2], link_prediction_for_generation_res[0][0], obj_typ))
    input()

    print(colored('将链接预测中概率最大的尾实体替换之前的尾实体，构成新的三元组........', attrs=['bold']))
    pprint(generation_triples_after_pre)
    input()

    # 链接预测结果进行三元组检测
    print(colored('对链接预测生成的三元组进行三元组质检........', attrs=['bold']))
    generation_link_pre = triple_classification(model, generation_triples_after_pre)
    input()

    # 通过质检的三元组插入neo4j
    print(colored('将通过质检的三元组插入图谱........', attrs=['bold']))
    pprint(test_and_valid_res)
    triples2neo4j(graph, test_and_valid_res)
    pprint(generation_link_pre)
    triples2neo4j(graph, generation_link_pre)    

    # result = model.predict()
    # print(result)