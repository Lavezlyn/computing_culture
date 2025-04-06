import json
import argparse
from typing import List, Dict, Any

def extract_keywords(input_file: str, keywords: List[str], output_file: str) -> None:
    """
    Extract specified keywords from input JSON file and save to output file.
    
    Args:
        input_file (str): Path to input JSON file
        keywords (List[str]): List of keywords to extract
        output_file (str): Path to output JSON file
    """
    try:
        # Read input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Initialize output data structure
        output_data = []

        source = list(data.keys())[0]
        data = data[source]
        
        # Process each entry in the input data
        for chapter, entry in data.items():
            for keyword in entry["keywords"]:
                if keyword["keyword"] in keywords:
                        # Create new entry with required fields
                    new_entry = {
                        "source": source,
                        "chapter": chapter,
                        "keyword": keyword["keyword"],
                        "score": keyword["score"],
                        "pos": keyword["pos"],
                        "embedding": keyword["embedding"],
                        "similarities": keyword["similarities"]
                        }
                    output_data.append(new_entry)
        
        # Write output to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
            
        print(f"Successfully extracted keywords and saved to {output_file}, total {len(output_data)} keywords")
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")

def main():
    input_file = "/aifs4su/yaodong/sitong/projects/nlp/chapter_keyword/n019.json_chapter_keywords.json"
    output_file = "/aifs4su/yaodong/sitong/projects/nlp/mengzi_keywords.json"
    #keywords = ["堯舜", "桀為", "伊尹", "明君", "人臣","亂臣" "萬民", "耕耨", "私利", "稱道"] #hanfeizi
    #keywords = ["堯則", "爾舜", "伊尹", "邦君", "亂臣", "臣臣", "臣不臣", "惠而不費", "子罕言利", "聞道", "道者", "弘道"] #lunyu
    #keywords = ["故堯", "唯堯", "為堯", "雖桀", "明君者", "君尊", "君道", "治國者", "人君者", "亂世之君臣", "君臣", "臣道", "庸民", "私利", "裕利", "市利", "為道"] #shangjunshu
    keywords = ["堯舜之道者", "堯舜", "舜禹", "耕耨", "伊尹相湯", "伊尹", "耒耜", "桀紂", "惟君", "得行道焉","佚道", "甲利兵"]
    
    extract_keywords(input_file, keywords, output_file)

if __name__ == "__main__":
    main() 