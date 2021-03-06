# -*- coding: utf-8 -*-
"""chinese comments processor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_Zd0watR45vQ57z1PjBBixlzjpKDQfUA
"""

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
import tensorflow.python.keras.utils as ku
import pandas as pd
import numpy as np
import string, os
import warnings
# from keras.callbacks import ModelCheckpoint
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import Embedding, LSTM, Dense, Dropout
# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping
# from keras.models import Sequential
# import keras.utils as ku
# import pandas as pd
# import numpy as np
# import string, os
# import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# from google.colab import drive
# drive.mount('/content/drive/')

leon_comments=pd.read_excel('good_comments.xlsx')
leon_comments

all_comments=list(leon_comments.content.values)

len(all_comments)

# corpus按照['COMMENT1','COMMENT2']的格式，未取符号
corpus = [x for x in all_comments]
corpus
# 将所有corpus的评论都合成为一起，形式：['COMMENT1+COMMENT2'],未去符号
comments_alltegether=''.join(corpus)
print(comments_alltegether)

import re
text='里昂只有一颗盆栽$$#，不善言辞，爱喝牛奶。他不像，却真正是一个杀手。玛蒂达的到来，是包袱，也给里昂带来了生机。不过这种设定，注定是悲剧收场。里昂死后，玛蒂达将他盆栽的种子落地生根，里昂终于不再每日拿着手枪在椅子上不安地入睡，他落地了。娜塔莉波特曼太灵了，玛蒂达是如此特别。'
text2='被Portman和Oldman的表演惊到了。。。。Portman唱麦姐的《LIKEAVIRGIN》时喷了。。。'

def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    # print(chinese)
    return chinese
 
def find_unchinese(file):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    unchinese = re.sub(pattern,"",file)
    # print(unchinese)

import re
def char_preprocess(text):
  pattern = re.compile(r'[\u4e00-\u9fa5]|[，。]|[A-Za-z0-9]')
  str=''.join(pattern.findall(text))
  return str

    
# 只留下中文，其他全删
chinese_text=char_preprocess(text2)

comments_chinese=char_preprocess(comments_alltegether)

import jieba
# 通过结巴进行分词
kk=jieba.cut(comments_chinese,cut_all=False)
# 未去重的words
words_raw_list=','.join(kk).split(',')

words_list = []
for i in words_raw_list:
    if i not in words_list:
        words_list.append(i)

import random 
random.shuffle(words_list)
df_list=pd.DataFrame(words_list)
print(words_list)

len_max=0
for i in words_list:
  if(len(i)>len_max):
    len_max=len(i)
len_max

# 词到数字的字典
dict_words_to_digit={}
for i, element in enumerate(words_list):
  dict_words_to_digit[element]=i
# dict_words_update=d_order=sorted(dict_words.items(),key=lambda dict_words:dict_words[1],reverse=False)  # 按字典集合中，每一个元组的第二个元素排列。
# dict_words_update
# dict_words=dict(dict_words_update)
# dict_words 就是已经去重后按顺序的字典
dict_words_to_digit
# 数字到词的字典
dict_digit_to_words={value:key for key, value in dict_words_to_digit.items()}
dict_digit_to_words

total_words=len(words_list)
print(total_words)

text_try='玛蒂达的到来，是包袱，也给里昂带来了生机。'
def texts_to_sequences(texts):
  ch_text=find_chinese(texts)
  text_code=[]
  next_i=0
  for i in range(len(ch_text)):
    if i!= next_i:
      continue
    

    for j in range(i+6,i,-1):
      # print('when i:',i,'。j=',j)
      if ch_text[i:j] in dict_words_to_digit.keys():
        # print('word:',dict_words_to_digit[ch_text[i:j]],ch_text[i:j])
        text_code.append(dict_words_to_digit[ch_text[i:j]])
        next_i=j
        break
  # print(text_code)
  return text_code


def sequences_to_text(seq):
  all_text=''
  for data in seq:
    text=dict_digit_to_words[data]
    all_text+=text
    
  # print(all_text)
  return all_text
text_code=texts_to_sequences(text_try)
sequences_to_text(text_code)

def get_sequence_of_tokens(corpus):
  input_sequences=[]
  total_words=len(words_list)
  for line in corpus:
    ch_line=find_chinese(line)
    token_list=texts_to_sequences(ch_line)
    for i in range(1, len(token_list)):
      n_gram_sequence = token_list[:i + 1]
      input_sequences.append(n_gram_sequence)
  return input_sequences,total_words

input_sequences,total_words=get_sequence_of_tokens(corpus)
print(input_sequences)
print(len(input_sequences))
print(total_words)

from keras.utils import np_utils
# pad sequences
def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = np_utils.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(input_sequences)

print(predictors.shape)
print(label.shape)
print(max_sequence_len)

def create_model(max_sequence_len, total_words):
    model = Sequential()

    # Add Input Embedding Layer
    model.add(Embedding(total_words, 80, input_length=max_sequence_len - 1))

    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.3))
    # model.add(LSTM(50))
    # model.add(Dropout(0.3))


    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


model = create_model(max_sequence_len, total_words)
model.summary()
checkpoint = ModelCheckpoint('model_for_chinese_comments_processor', monitor='loss', verbose=2, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

model.fit(predictors, label, epochs=100, callbacks=callbacks_list)

def generate_text(seed_text, next_words, model, max_seq_len):
    for _ in range(next_words):
        token_list = texts_to_sequences(seed_text)
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')

        predicted = model.predict_classes(token_list, verbose=2)

        output_word = ''

        for word, index in dict_words_to_digit.items():
            if index == predicted:
                output_word = word
                break

        seed_text = seed_text  + output_word

    return seed_text.title()

# print(generate_text("有了牵挂", 100, model, max_sequence_len))
# print()
# print(generate_text("他们都是受伤", 200, model, max_sequence_len))
# print()
# print(generate_text('全模式', 300, model, max_sequence_len))

print(generate_text("很小的时候", 20, model, max_sequence_len))

model.save('model')