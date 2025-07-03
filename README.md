# AI Research Assistant

An automated research tool that performs comprehensive web research using Google search operators and OpenAI's GPT-4 for analysis.

## Features

- Decomposes complex queries into subcomponents
- Uses advanced Google search operators for targeted searching
- Crawls and analyzes web content
- Generates comprehensive research reports
- Fallback mechanism for search failures

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Install required packages:
```bash
pip install openai
pip install googlesearch-python
pip install crawl4ai
pip install beautifulsoup4
pip install requests
```

## Configuration

1. Create a `config.json` file in the project root directory
2. Add your OpenAI API key and configure settings:

```json
{
    "api_key": "your-openai-api-key-here",
    "max_loops": 3,
    "search_results_per_query": 3
}
```

### Configuration Parameters:
- `api_key`: Your OpenAI API key
- `max_loops`: Maximum number of search iterations per subcomponent (default: 3)
- `search_results_per_query`: Number of search results to process per query (default: 3)

## Usage

Run the script:
```bash
python research_assistant.py
```

The script will:
1. Load configuration from `config.json`
2. Process the research query
3. Generate a comprehensive report saved as `research_report_YYYYMMDD_HHMMSS.md`

## Output

The tool generates markdown-formatted research reports containing:
- Executive summary
- Detailed findings for each research component
- Source citations
- Conclusion with key insights

## Requirements

- Python 3.7+
- OpenAI API key
- Internet connection for web searching and crawling
