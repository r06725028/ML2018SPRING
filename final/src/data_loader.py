from glob import glob
import random


def jaccard_similarity(sentence_a, sentence_b):
    set_a = set(sentence_a)
    set_b = set(sentence_b)
    return len(set_a & set_b) / len(set_a | set_b)


class DataLoader(object):
    def __init__(self, training_dir, testing_path):
        all_data = []
        self.training_data = []
        self.validation_data = []
        for filename in glob(f'{training_dir}/*'):
            with open(filename, 'r') as fp:
                all_data.append([line.strip() for line in fp])
        for data in all_data:
            data_size = len(data)
            split_point = int(data_size * 0.9)
            self.training_data.append(data[:split_point])
            self.validation_data.append(data[split_point:])

    def gen_positive(self, part):
        questions = []
        answers = []
        for data in getattr(self, f'{part}_data'):
            for sentences in zip(data, data[1:], data[2:], data[3:]):
                for question, answer in zip(['\t'.join(sentences[:idx]) for idx in range(1, 4)], sentences[1:]):
                    if len(question) <= 71:
                        questions.append(question)
                        answers.append(answer)
        return questions, answers, [1 for _ in range(len(answers))]

    def gen_negative(self, part, seed=1126):
        questions = []
        answers = []
        random.seed(seed)
        for data in getattr(self, f'{part}_data'):
            for sentences in zip(data, data[1:], data[2:], data[3:]):
                for question, answer in zip(['\t'.join(sentences[:idx]) for idx in range(1, 4)], sentences[1:]):
                    if len(question) <= 71:
                        for sentences in getattr(self, f'{part}_data'):
                            neg_answer = random.choice(sentences)
                            while jaccard_similarity(neg_answer, answer) > 0.5:
                                neg_answer = random.choice(sentences)
                            questions.append(question)
                            answers.append(neg_answer)
        return questions, answers, [0 for _ in range(len(answers))]
