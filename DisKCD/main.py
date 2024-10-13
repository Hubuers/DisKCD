import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
from DisKCD.data_loader import TrainDataLoader, ValTestDataLoader
from DisKCD.model import Net
from DisKCD.utils import CommonArgParser, construct_local_map

def train(args, local_map,j):
    batch_size=args.batch_size
    data_loader = TrainDataLoader(batch_size)
    device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
    net = Net(args, local_map)
    net = net.to(device)
    net.train()
    print (net)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    print('training model...')

    loss_function = nn.NLLLoss()
    for epoch in range(args.epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output+1e-10), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 100))
                running_loss = 0.0

        save_snapshot(net, '../model/time'+str(j+1)+'_model_epoch_RCD' + str(epoch + 1))
        net.eval()
        rmse, auc = predict(args, net, epoch,j)
        net.train()

def predict(args, net, epoch,j):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    data_loader = ValTestDataLoader('test')
    print('predicting model...')
    data_loader.reset()
    net.eval()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)

        for i in range(len(labels)):
            pred_label = 1 if output[i] > 0.5 else 0
            if labels[i] != pred_label:
                pass
            else:
                correct_count += 1

        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()


    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    auc = roc_auc_score(label_all, pred_all)

    pred_labels=(pred_all>0.5).astype(int)
    precision=precision_score(label_all,pred_labels,zero_division=1)
    recall=recall_score(label_all,pred_labels,zero_division=1)
    f1=f1_score(label_all,pred_labels,zero_division=1)

    print('i= %d, epoch= %d, accuracy= %f, rmse= %f, auc= %f, precision= %f, recall= %f, f1 score= %f' % (j+1, epoch+1, accuracy, rmse, auc, precision, recall, f1))
    with open('../result/model_val_RCD.txt', 'a', encoding='utf8') as f:
        f.write('%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n' % (j+1, epoch+1, accuracy, rmse, auc, precision, recall, f1))

    return rmse, auc

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
    times = 1
    for j in range(times):
        args = CommonArgParser().parse_args()
        train(args, construct_local_map(args),j)

