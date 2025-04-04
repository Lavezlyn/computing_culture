# filter the keyword dict and merge into one json

import json
import os

input_dir = "analyzed_texts"
output_path = "merged_keywords.json"

def check_valid_keyword(keyword):
    if "keywords" not in keyword:
        return False
    return True

def check_valid_word(word):
    if "word" not in word:
        return False
    if "type" not in word:
        return False
    if "importance" not in word:
        return False
    return True

def check_valid_word_list(word_list):
    for word in word_list:
        if not check_valid_word(word):
            return False
    return True

# read all the json files in the input_dir

merged_keywords = []

for file in os.listdir(input_dir):
    with open(os.path.join(input_dir, file), 'r', encoding='utf-8') as f:
        data = json.load(f)
        for keyword in data:
            if check_valid_keyword(keyword):
                word_list = keyword['keywords']
                if check_valid_word_list(word_list):
                    merged_keywords.append(keyword)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(merged_keywords, f, ensure_ascii=False, indent=4)

print("merged_keywords: ", len(merged_keywords))
print("output_path: ", output_path)



