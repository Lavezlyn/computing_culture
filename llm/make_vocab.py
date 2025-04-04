import json

with open("analysis_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

overall_ranking = data["overall_frequencies_rankings"]


frequencies_rankings_in_each_type = data["frequencies_rankings_in_each_type"]

type_list = list(frequencies_rankings_in_each_type.keys())

word_list = set()

for type_ in type_list:
    for word, _ in frequencies_rankings_in_each_type[type_]:
        word_list.add(word)

with open("vocab.txt", "w", encoding="utf-8") as f:
    for word in word_list:
        f.write(word + "\n")

with open("type_list.json", "w", encoding="utf-8") as f:
    json.dump(type_list, f, ensure_ascii=False, indent=4)
