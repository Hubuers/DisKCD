import csv
import random
import json

def switch_data():
    java_data = []
    with open('../data/jad/data_original/tested_exer_log.csv') as i_f:
        reader = csv.reader(i_f)
        for item in reader:
            java_data.append(item)


        learn_list = []
        for sublist in java_data:
            learner_id, exer_id, score, skill_code = sublist

            learner_id = int(learner_id)

            log_entry = {
                'exer_id': int(exer_id),
                'score': float(score),
                'knowledge_code': int(skill_code)
            }

            found = False
            for learner_info in learn_list:
                if learner_info['user_id'] == learner_id:
                    learner_info['logs'].append(log_entry)
                    learner_info['log_num'] += 1
                    found = True
                    break

            if not found:
                new_learner_info = {
                    'user_id': learner_id,
                    'log_num': 1,
                    'logs': [log_entry]
                }
                learn_list.append(new_learner_info)

        with open('../data/jad/log_data.json', 'w') as file:
            json.dump(learn_list, file, indent=4)



def divide_data():
    with open('../data/jad/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)

    train_set,test_set = [], []
    for stu in stus:
        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        logs_num=stu['log_num']
        train_size = int(logs_num * 0.8)
        test_size = logs_num - train_size

        logs = []
        for log in stu['logs']:
            logs.append(log)
        random.shuffle(logs)
        stu_train['logs'] = logs[:train_size]
        stu_test['logs'] = logs[-test_size:]
        stu_train['log_num'] = len(stu_train['logs'])
        stu_test['log_num'] = len(stu_test['logs'])
        test_set.append(stu_test)

        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'knowledge_code': log['knowledge_code']})
    random.shuffle(train_set)

    with open('../data/jad/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('../data/jad/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    switch_data()
    divide_data()