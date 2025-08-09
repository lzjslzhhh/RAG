from typing import Optional

from datasets import load_dataset
import random
import re


# Function to load the dataset
# Need return three sub datasets for training,evaluating and testing.
# para:None
def load_task_dataset():
    dataset_name = "/tmp/pycharm_project_581/EleQA-master/ele_qa"
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


# reformat the dataset before passing to the prompt agent
# para: question_list
def reformat_data(question_list):
    return question_list


# function to extract the answer from the response by LLM
def extract_answer(message):
    pattern = r"<answer>\s*([A-Za-z])\s*\..*?</answer>"
    answer = re.search(pattern, message)
    if answer == None:
        pattern = r"<answer>\s*([A-Za-z])\s*</answer>"
        answer = re.search(pattern, message)
    if answer == None:
        pattern = r"<([A-Za-z])>"
        answer = re.search(pattern, message)
    if answer:
        answer = answer.group(1)
    return answer


def extract_judgment(text: str) -> Optional[str]:
    """
    从文本中提取判断题答案（正确/错误）
    支持格式：
    - "<answer>正确</answer>"
    - "结论：错误"
    - "[判断结果] 正确"
    """
    patterns = [
        r"<answer>\s*(正确|错误)\s*</answer>",  # XML标签
        r"结论[:：]\s*(正确|错误)",  # 中文冒号
        r"$$判断结果$$\s*(正确|错误)",  # 方括号格式
        r"最终答案\s*[:：]?\s*(正确|错误)",  # 宽松匹配
        r"(正确|错误)(?=[。！？\s]|$)"  # 句尾匹配
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


# function to check the answer correct or not
def check_anwser(model_answer, correct_answer):
    if model_answer == correct_answer:
        return True
    return False

if __name__ == '__main__':
    questions_train, questions_eval, questions_test = load_task_dataset()
    print(questions_train[:5])
    print(questions_eval[:5])
    print(questions_test[:5])