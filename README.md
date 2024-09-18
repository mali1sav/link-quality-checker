# Backlink Quality and Relevancy Analyser

This tool is designed for link building managers to efficiently analyze and evaluate backlinks for quality and relevance, with a focus on the industrial and electronic components sector.

## Features

- Bulk analysis of backlinks across different markets
- Comprehensive evaluation combining technical SEO metrics with content relevancy analysis
- Multilingual capability for analyzing backlinks from various language markets
- Industry-specific focus tailored to the industrial and electronic components sector
- Balanced scoring considering both link strength and content relevance
- Actionable insights with human-readable explanations
- Flexible verdict adjustments allowing integration of human expertise

## How It Works

1. **Data Input**: Upload a CSV or Excel file containing backlink data (Market, Source Domain, Target URL).

2. **Data Processing**:
   - Backlink Profile Analysis: Fetches comprehensive backlink data (Spam Score, Referring Domains, Total Backlinks).
   - SERP Data Retrieval: Gets top 20 organic results for the source domain.
   - Content Relevancy Analysis: Generates domain summaries, translates non-English content, and computes content similarity.
   - Overall Score Calculation: Combines backlink profile metrics with content similarity score.
   - Verdict Generation: Provides human-readable explanations of backlink quality and relevance.

3. **Results Presentation**: Displays analysis results in an interactive interface with key metrics and explanations.

4. **Data Export**: Option to download complete analysis as an Excel file.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/mali1sav/link-quality-checker.git
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your .env file with the necessary API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DATAFORSEO_API_KEY=your_dataforseo_api_key
   ```

## Usage

Run the Streamlit app:
```
streamlit run main.py
```

Then, follow the on-screen instructions to upload your backlink data file and analyze the results.

## Benefits

- Efficient bulk analysis of numerous backlinks
- Comprehensive evaluation of backlink quality and relevance
- Multilingual analysis capabilities
- Industry-specific focus for more accurate results
- Balanced scoring system
- Actionable insights to guide link building strategies
- Flexibility to incorporate human expertise

This tool empowers link building managers to make data-driven decisions, focusing efforts on high-quality, relevant backlinks that align with your industry and target markets.