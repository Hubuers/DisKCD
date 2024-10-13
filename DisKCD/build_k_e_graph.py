import json

def build_ke_local_map():
    data_file = '../data/ASSIST/train_set.json'
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n,_ = list(map(eval, i_f.readline().split(',')))

    temp_list = []
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    k_from_e = ''
    e_from_k = ''
    print (len(data))
    for line in data:
        exer_id = line['exer_id'] - 1
        '''k=line['knowledge_code']
        if (str(exer_id) + '\t' + str(k - 1 + exer_n)) not in temp_list or (
                str(k - 1 + exer_n) + '\t' + str(exer_id)) not in temp_list:
            k_from_e += str(exer_id) + '\t' + str(k - 1 + exer_n) + '\n'
            e_from_k += str(k - 1 + exer_n) + '\t' + str(exer_id) + '\n'
            temp_list.append((str(exer_id) + '\t' + str(k - 1 + exer_n)))
            temp_list.append((str(k - 1 + exer_n) + '\t' + str(exer_id)))'''
        k = line['knowledge_code']
        for k_item in k:
            k_str = str(k_item - 1 + exer_n)
            if (str(exer_id) + '\t' + k_str) not in temp_list and (k_str + '\t' + str(exer_id)) not in temp_list:
                k_from_e += str(exer_id) + '\t' + k_str + '\n'
                e_from_k += k_str + '\t' + str(exer_id) + '\n'
                temp_list.append(str(exer_id) + '\t' + k_str)
                temp_list.append(k_str + '\t' + str(exer_id))

    with open('../data/ASSIST/graph/k_from_e.txt', 'w') as f:
        f.write(k_from_e)
    with open('../data/ASSIST/graph/e_from_k.txt', 'w') as f:
        f.write(e_from_k)



if __name__ == '__main__':
    build_ke_local_map()