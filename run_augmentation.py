from ast import literal_eval
import pandas as pd
import numpy as np
import collections
import torch

def extract_data(text_list,spans_list):
    toxic_words_all = []
    idx2toxic_words = []
    idx2toxic_spans = []
    for it, (text, spans) in enumerate(zip(text_list,spans_list)):
        if len(spans) == 0:
            idx2toxic_words.append([])
            idx2toxic_spans.append([])
            continue
        span_list = []
        prev = spans[0]
        span = [prev]
        for cur in spans[1:]:
            if cur == (prev + 1):
                span.append(cur)
            else:
                span_list.append(span)
                span = [cur]
            prev = cur
        span_list.append(span)
        toxic_words = []
        for span in span_list:
            word = "".join([text[i] for i in span])
            toxic_words.append(word)
        idx2toxic_words.append(toxic_words)
        toxic_words_all.extend(toxic_words)
        idx2toxic_spans.append(span_list)
    return toxic_words_all, idx2toxic_words, idx2toxic_spans


def get_prob(idx2word):
    idx2prob = []
    for word in idx2word:
        prob = toxic_words_count[word]/len(toxic_words_all)
        idx2prob.append(prob**0.75)
    sum_all = sum(idx2prob)
    for i in range(len(idx2prob)):
        idx2prob[i] = idx2prob[i]/sum_all
    return idx2prob


def exchange_toxic_spans(text_list, spans_list, idx2toxic_words, idx2toxic_spans, sample_times=5):
    data_aug = []
    for it, (text, spans, toxic_words, span_list) in enumerate(zip(text_list[:], spans_list[:], idx2toxic_words[:], idx2toxic_spans[:])):
        if spans == [] or toxic_words == []:
            continue
        spans = spans[:]
        for _ in range(sample_times):
            mask_word = np.random.choice(toxic_words)
            candidate_word = np.random.choice(a=idx2word,p=idx2prob)
            mask_idx = toxic_words.index(mask_word)
            mask_span = span_list[mask_idx]
            if mask_word != candidate_word and abs(len(mask_word)-len(candidate_word))<6:
                break
        if mask_word != candidate_word and abs(len(mask_word)-len(candidate_word))<6:
            text_aug = text.replace(mask_word,candidate_word)
            start = spans.index(mask_span[0])
            end = spans.index(mask_span[-1])+1
            if len(mask_word)==len(candidate_word):
                span_aug = spans.copy()
            elif len(mask_word)<len(candidate_word):
                dif = len(candidate_word)-len(mask_word)
                candidate_span = mask_span[:]
                for i in range(dif):
                    candidate_span.append(mask_span[-1]+i+1)
                for idx in range(len(spans)):
                    if spans[idx] > mask_span[-1]:
                        spans[idx] += dif
                spans_aug = spans[:start] + candidate_span + spans[end:]
            elif len(mask_word)>len(candidate_word):
                dif = len(mask_word)-len(candidate_word)
                candidate_span = mask_span[:len(candidate_word)]
                for idx in range(len(spans)):
                    if spans[idx] > mask_span[-1]:
                        spans[idx] -= dif
                spans_aug = spans[:start] + candidate_span + spans[end:]
            new_data = {"text_aug":text_aug,
                        "spans_aug":spans}
            data_aug.append(new_data)
    return data_aug


def remove_toxic_spans(text_list, spans_list, idx2toxic_words, idx2toxic_spans):
    data_aug = []
    for _, (text, spans, toxic_words, span_list) in enumerate(zip(text_list[:], spans_list[:], idx2toxic_words[:], idx2toxic_spans[:])):
        if spans == [] or toxic_words == []:
            continue
        for word in toxic_words:
            text = text.replace(word,"")
        new_data = {"text_aug":text,
                    "spans_aug":[]}
        data_aug.append(new_data)
        
    return data_aug


def data_augmentation(text_list, spans_list, idx2toxic_words, idx2toxic_spans, sample_times=5, iteration=1):
    data_aug_1 = []
    for _ in range(iteration):
        data_aug_1.extend(exchange_toxic_spans(text_list, spans_list, idx2toxic_words, idx2toxic_spans, sample_times=sample_times))
    data_aug_2 = remove_toxic_spans(text_list, spans_list, idx2toxic_words, idx2toxic_spans)
    return data_aug_1 + data_aug_2


if __name__ == '__main__':
    
    translationTable = str.maketrans("éàèùaêóïüÉ","eaeuaeoiuE")
    tsd = pd.read_csv("toxic_spans/data/tsd_train.csv") 
    tsd.text = tsd.text.apply(lambda x:x.translate(translationTable))
    tsd.spans = tsd.spans.apply(literal_eval)
    text_list = tsd.text.to_list()
    spans_list = tsd.spans.to_list()

    toxic_words_all, idx2toxic_words, idx2toxic_spans = extract_data(text_list,spans_list)
    toxic_words_count = collections.Counter(toxic_words_all)
    all_count = sum(toxic_words_count.values())
    idx2word = [i[0] for i in toxic_words_count.most_common(len(toxic_words_count))]
    word2idx = {word:idx for idx,word in enumerate(idx2word)}

    idx2prob = get_prob(idx2word)
    data_aug = data_augmentation(text_list, spans_list, idx2toxic_words, idx2toxic_spans, sample_times=5, iteration=1)
    file_name = "augmented_data_all.pt"
    torch.save(data_aug,file_name)
    print("Augmentation success! Saved to {}".format(file_name))
