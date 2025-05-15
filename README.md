# Research Paper Analysis and Ranking System

A sophisticated tool for analyzing, comparing, and ranking academic research papers. This application helps researchers, students, and academic professionals evaluate papers based on multiple criteria, providing objective insights into their relative quality and impact.

## 📋 Features

- **Dual Paper Analysis**: Upload and compare two research papers side-by-side
- **Automatic Section Extraction**: Identifies abstract, introduction, methodology, results, and conclusion
- **Publication Year Detection**: Automatically detects when papers were published
- **Model & Dataset Identification**: Recognizes ML/AI models and datasets mentioned in papers
- **Keyword Extraction**: Identifies key topics and themes using KeyBERT
- **Citation Analysis**: Estimates citation impact using multiple data sources
- **Novelty Assessment**: Calculates uniqueness compared to other papers
- **Semantic Similarity**: Measures content overlap between papers
- **Enhanced Ranking Algorithm**: Combines multiple factors for comprehensive paper scoring

## 🔍 Ranking Methodology

Papers are ranked using our Enhanced Paper Ranking Model that evaluates five key metrics:

1. **Publication Year** - More recent papers receive higher scores
2. **Reported Accuracy** - Papers demonstrating higher performance metrics are prioritized 
3. **Citation Count** - Papers with more citations indicate higher impact
4. **Novelty Score** - Papers with unique content receive higher ratings
5. **Content Overlap** - Lower overlap with other papers is preferred

The model uses dynamic weighting, adjusting the importance of these factors based on context.

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/research-paper-analyzer.git
cd research-paper-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The key dependencies include:
- streamlit
- pymupdf
- pandas
- numpy
- keybert
- sentence-transformers
- scikit-learn
- scholarly
- requests

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## 📊 How to Use

1. **Upload Papers**: Use the file uploaders to select two PDF research papers
2. **Analyze**: Click the "Analyze and Compare Papers" button
3. **Review Results**: Examine the detailed analysis including:
   - Basic paper information (title, year, models, datasets)
   - Key metrics (accuracy, citations)
   - Comparison metrics (similarity, novelty, overlap)
   - Paper rankings with explanations

## 📝 Citation Analysis

The system attempts to find real citation data using the following approach:

1. **Primary Source**: Searches scholarly databases for papers matching title and publication year
2. **Verification**: Compares multiple matching papers to find the best match
3. **Fallback**: Provides an estimate based on publication year if no reliable data is found

## 🧪 Technical Details

### Key Components

- **Text Extraction**: Uses PyMuPDF for accurate PDF parsing
- **Section Identification**: Employs regex patterns to locate paper sections
- **Keyword Extraction**: Leverages KeyBERT for identifying important terms
- **Semantic Analysis**: Utilizes SentenceTransformers with ScieBERT embeddings
- **Citation Retrieval**: Uses multiple APIs including CrossRef and Semantic Scholar

### Limitations

- PDF extraction quality depends on the document formatting
- Citation counts are estimates when papers cannot be found in databases
- Model and dataset detection is limited to known names in the pattern database

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgements

- The KeyBERT library for keyword extraction
- SentenceTransformers and the ScieBERT model
- Scholarly, CrossRef and Semantic Scholar for citation data
- PyMuPDF for PDF text extraction
