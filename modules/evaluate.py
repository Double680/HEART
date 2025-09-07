import os
import json
import math
import re
import string
import numpy as np
from scipy.optimize import linear_sum_assignment


def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

def _white_space_fix(text: str) -> str:
    return " ".join(text.split())

EXCLUDE = set(string.punctuation)

def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text

def _lower(text: str) -> str:
    return text.lower()

def _tokenize(text: str):
    return re.split(" |-", text)
   
def _normalize_answer(text: str) -> str:
    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized

def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False

def _normalize_number(text):
    if _is_number(text):
        return str(float(text))
    else:
        return text

def _answer_to_bags(answer):
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags

def _align_bags(predicted, gold):
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores

def _compute_f1(predicted_bag, gold_bag) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1

def _match_numbers_if_present(gold_bag, predicted_bag) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False

def get_span_selection_metrics(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def evaluate(args):
    start = args.start
    end = args.end
    samples = args.samples
    save_root = args.save_root_setting

    if args.dev:
        exact, f1 = 0.0, 0.0
        for i in range(end-start):
            with open(os.path.join(save_root, f'{i}.json'), 'r') as file:
                result = json.loads(file.read())
            pred, gold = result['prediction'], str(result['gold'])
            exact_acc, f1_acc = get_span_selection_metrics(pred, gold)
            exact += exact_acc
            f1 += f1_acc
            result["exact_match"] = exact_acc
            result["f1_score"] = f1_acc
            with open(os.path.join(save_root, f'{i}.json'), 'w') as file:
                file.write(json.dumps(result, indent=2))
        exact = exact / (end-start)
        f1 = f1 / (end-start)
        print(f'Exact Match: {exact*100:.2f}, F1: {f1*100:.2f}')

        if args.retrieve_type != "none":
            text_pre, text_rec, text_ndcg, table_pre, table_rec, table_ndcg = 0, 0, 0, 0, 0, 0
            for i in range(end-start):
                with open(os.path.join(save_root, f'{i}.json'), 'r') as file:
                    result = json.loads(file.read())
                text_gth = list(dict.fromkeys(samples[i]['qa']['text_evidence']).keys())
                text_pred = list(dict.fromkeys(result['retrieved_text_ids']).keys())
                text_join = set(text_gth).intersection(set(text_pred))
                try:
                    text_pre += len(list(text_join)) / len(text_pred)
                except ZeroDivisionError:
                    text_pre += 1
                try:
                    text_rec += len(list(text_join)) / len(text_gth)
                except ZeroDivisionError:
                    text_rec += 1

                text_dcg, text_idcg = 0, 0
                for j, item in enumerate(text_pred):
                    if item in text_gth:
                        text_dcg += 1 / math.log2(j+2)
                for j in range(len(text_gth)):
                    if j == len(text_pred):
                        break
                    text_idcg += 1 / math.log2(j+2)
                try:
                    text_ndcg += text_dcg / text_idcg
                except ZeroDivisionError:
                    text_ndcg += 1

                # if args.tabheader:
                #     table_gth = list(dict.fromkeys(samples[i]['qa']['table_evidence']).keys())
                #     table_pred = result['retrieved_table_ids']
                #     cleaned_col_indices, cleaned_row_indices = [], []
                #     for item in table_pred[0]:
                #         for site in item[2]:
                #             cleaned_col_indices.append((item[0], item[1], site))
                #     for item in table_pred[1]:
                #         for site in item[2]:
                #             cleaned_row_indices.append((item[0], item[1], site))
                #     hit = 0
                #     for item in table_gth:
                #         id, row, col = item.split('-')
                #         id, row, col = int(id), int(row), int(col)
                #         if (id, 'row', row) in cleaned_row_indices and (id, 'col', col) in cleaned_col_indices:
                #             hit += 1
                #     try:
                #         table_rec += hit / len(table_gth)
                #     except ZeroDivisionError:
                #         table_rec += 1
                # else:
                if not args.tabheader:
                    table_gth = list(dict.fromkeys(samples[i]['qa']['table_evidence']).keys())
                    table_pred = list(dict.fromkeys(result['retrieved_table_ids']).keys())
                    table_gth_dict = {key: j for j, key in enumerate(samples[i]['table_description'])}
                    table_gth_norm = [table_gth_dict[key] for key in table_gth]

                    table_join = set(table_gth_norm).intersection(set(table_pred))
                    try:
                        table_pre += len(list(table_join)) / len(table_pred)
                    except ZeroDivisionError:
                        table_pre += 1
                    try:
                        table_rec += len(list(table_join)) / len(table_gth_norm)
                    except ZeroDivisionError:
                        table_rec += 1

                    table_dcg, table_idcg = 0, 0
                    for j, item in enumerate(table_pred):
                        if item in table_gth_norm:
                            table_dcg += 1 / math.log2(j+2)
                    for j in range(len(table_gth_norm)):
                        if j == len(table_pred):
                            break
                        table_idcg += 1 / math.log2(j+2)
                    try:
                        table_ndcg += table_dcg / table_idcg
                    except ZeroDivisionError:
                        table_ndcg += 1
                    
            text_pre = text_pre / (end-start)
            text_rec = text_rec / (end-start)
            text_ndcg = text_ndcg / (end-start)
            print(f'Retrieved Texts Presicion: {text_pre*100:.2f}, Recall: {text_rec*100:.2f}, NDCG: {text_ndcg*100:.2f}')

            # if args.tabheader:
            #     table_rec = table_rec / (end-start)
            #     print(f'Retrieved Tables Recall: {table_rec*100:.2f}')
            # else:
            if not args.tabheader:
                table_pre = table_pre / (end-start)
                table_rec = table_rec / (end-start)
                table_ndcg = table_ndcg / (end-start)
                print(f'Retrieved Tables Presicion: {table_pre*100:.2f}, Recall: {table_rec*100:.2f}, NDCG: {table_ndcg*100:.2f}')

    else:
        results = []
        for i in range(end-start):
            with open(os.path.join(save_root, f'{i}.json'), 'r') as file:
                result = json.loads(file.read())
            results.append({
                "uid": result["uid"], "predicted_ans": result["prediction"], "predicted_program": []
            })
        with open(os.path.join(save_root, f'test_predictions.json'), 'w') as file:
            file.write(json.dumps(results, indent=2))
        print('Predictions saved!')