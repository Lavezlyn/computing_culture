# Computing Culture
Codes for course project of Computing Culture, 2025 spring, PKU.

## Project Overview
This project focuses on analyzing and extracting keywords from Chinese classical texts using modern NLP techniques. It provides tools for keyword extraction, chapter analysis, and network visualization of keyword relationships.

## Repository Structure
```
.
├── llm/                    # Large Language Model related code (if needed)
├── keyword/               # Keyword extraction results (not uploaded)
├── chapter_keyword/       # Chapter-level keyword analysis (not uploaded)
├── chinese/              # Chinese datasets (not uploaded)
├── keyword_extract.py    # Main keyword extraction script
├── chapter_keyword_extract.py  # Chapter-level keyword extraction
├── keyword_network.py    # Network visualization of keywords
└── font.py              # Font utilities for visualization
```

## Key Features
- Keyword extraction from Chinese classical texts using BERT-based models
- Chapter-level keyword analysis
- Network visualization of keyword relationships
- Support for both modern and classical Chinese text processing
- Customizable stop words and processing parameters

## Dependencies
- Python 3.11+
- keybert
- transformers
- jieba
- pkuseg
- torch
- numpy
- scikit-learn

## Usage
1. **Basic Keyword Extraction**
```python
python keyword_extract.py
```

2. **Chapter-level Keyword Analysis**
```python
python chapter_keyword_extract.py
```

3. **Keyword Network Visualization**
```python
python keyword_network.py
```

## Contributing
Please feel free to submit issues and enhancement requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
