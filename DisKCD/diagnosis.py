import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from DisKCD.data_loader import ValTestDataLoader
from DisKCD.model import Net
from DisKCD.utils import CommonArgParser, construct_local_map


student_n=2493
exer_n=17746

def diagnosis(time,epoch,args,local_map):
    device = torch.device(('cuda:%d' % (0)) if torch.cuda.is_available() else 'cpu')
    data_loader = ValTestDataLoader('diagnosis')
    net=Net(args, local_map)
    print('diagnosis model...')
    data_loader.reset()
    load_snapshot(net,'../model/time'+str(time)+'_model_epoch_RCD'+str(epoch))
    net=net.to(device)
    net.eval()
    knowledge_embs_all = []
    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        knowledge_embs_all.append(input_knowledge_embs)

        output = net(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    knowledge_embs_all = torch.cat(knowledge_embs_all, dim=0)
    pred_all = np.array(pred_all)
    label_all = np.array(label_all)

    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))

    pred_labels = (pred_all > 0.5).astype(int)
    auc = roc_auc_score(label_all, pred_labels)
    precision = precision_score(label_all, pred_labels, zero_division=1)
    recall = recall_score(label_all, pred_labels, zero_division=1)
    f1 = f1_score(label_all, pred_labels, zero_division=1)

    # diagnosis model
    stu_state=[]
    with open('../result/student_stat_RCD.txt', 'w', encoding='utf8') as output_file:
        for stu_id in range(student_n):
            # get knowledge status of student with stu_id (index)
            status = net.get_stu_know(torch.LongTensor([stu_id])).tolist()[0]
            status_str = '\t'.join(map(str, status))
            output_file.write(status_str + '\n')
            stu_state.append(status)
    stu_state=torch.tensor(stu_state)
    with open('../result/exercise_stat_RCD.txt', 'w', encoding='utf8') as output_file:
        for exer_id in range(exer_n):
            # get knowledge status of student with stu_id (index)
            exer_status = net.get_exer(torch.LongTensor([exer_id])).tolist()[0]
            exer_str = '\t'.join(map(str, exer_status))
            output_file.write(exer_str + '\n')

    doa=degree_of_agreement(knowledge_embs_all,stu_state.numpy(),data_loader.load_all_data())

    print('i= %d, epoch= %d, accuracy= %f, rmse= %f, doa=%f, auc= %f, precision= %f, recall= %f, f1 score= %f' % (
    time, epoch, accuracy, rmse, doa, auc,  precision, recall, f1))
    with open('../result/model_diagnosis_RCD.txt', 'a', encoding='utf8') as f:
        f.write('%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (time, epoch, accuracy, rmse, auc, precision, recall, f1,doa))



def load_snapshot(model, filename):
    f=open(filename,'rb')
    model.load_state_dict(torch.load(f,map_location=lambda s,loc:s))
    f.close()

def degree_of_agreement(q_matrix, proficiency, dataset):
    problem_number, knowledge_number = q_matrix.shape
    student_number = proficiency.shape[0]

    q_matrix = torch.tensor(q_matrix, device='cuda')
    proficiency = torch.tensor(proficiency, device='cuda')

    r_matrix = torch.full((student_number, problem_number), -1, device='cuda')

    for lines in dataset:
        student_id_batch, question_batch, _, y_batch = lines
        for student_id, question, y in zip(student_id_batch, question_batch, y_batch):
            r_matrix[student_id][question] = y

    doaList = []
    for k in range(knowledge_number):
        numerator = 0.0
        denominator = 0.0

        delta_matrix = (proficiency[:, k].reshape(-1, 1) > proficiency[:, k].reshape(1, -1)).float()

        question_hask = torch.where(q_matrix[:, k] != 0)[0].tolist()

        for j in question_hask:
            row_vec = (r_matrix[:, j].reshape(1, -1) != -1).float()
            column_vec = (r_matrix[:, j].reshape(-1, 1) != -1).float()
            mask = row_vec * column_vec

            delta_response_logs = (r_matrix[:, j].reshape(-1, 1) > r_matrix[:, j].reshape(1, -1)).float()
            i_matrix = (r_matrix[:, j].reshape(-1, 1) != r_matrix[:, j].reshape(1, -1)).float()

            numerator += torch.sum(delta_matrix * torch.logical_and(mask, delta_response_logs))
            denominator += torch.sum(delta_matrix * torch.logical_and(mask, i_matrix))

        if denominator == 0:
            doaList.append(0)
        else:
            doaList.append(numerator / denominator)

    non_zero_elements = torch.tensor(doaList, device='cuda')[torch.tensor(doaList, device='cuda') != 0]
    if non_zero_elements.numel() > 0:
        mean_value = torch.mean(non_zero_elements).item()
    else:
        mean_value = 0

    return mean_value




if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    time=1
    epoch=10
    diagnosis(time,epoch, args, construct_local_map(args))
