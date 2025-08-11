import random

from datasets import load_dataset

question_type = '填空题'
if question_type=='填空题':
    question_cots = 10
else:
    question_cots = 200

def load_task_dataset():
    dataset_name = f"/tmp/pycharm_project_581/EleQA-master/{question_type}"
    dataset = load_dataset(dataset_name)
    new_dataset = dict(train=[], test=[])

    def process_split(split_name):
        for example in dataset[split_name]:
            # Extract choices and answer key from the example
            # print(example)

            # Append to the new dataset
            new_dataset[split_name].append(dict(question=example["question"], answer=example["answer"]))

    process_split('train')
    process_split('test')
    dataset = new_dataset
    random.seed(30)
    random.shuffle(dataset['train'])
    random.shuffle(dataset['test'])
    questions_train = dataset['train'][:2000]
    questions_eval = dataset['train'][2000:2050]
    questions_test = dataset['test'][0:79]
    return questions_train, questions_eval, questions_test
