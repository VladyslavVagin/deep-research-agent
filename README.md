# Deep Research Agent - Agent Schema

## Agent Architecture

### 1. ResearchManager
**Purpose**: Orchestrates the entire research workflow
**Input**: Query string
**Output**: Research report and email notification
**Workflow**:
1. Plan searches → 2. Perform searches → 3. Write report → 4. Send email

### 2. PlannerAgent
**Purpose**: Creates search strategy for a given query
**Input**: Research query
**Output**: WebSearchPlan (5 search items)
**Schema**:
```python
class WebSearchItem:
    reason: str      # Reasoning for search importance
    query: str       # Search term to use

class WebSearchPlan:
    searches: list[WebSearchItem]  # List of 5 searches to perform
```

### 3. SearchAgent
**Purpose**: Performs web searches and summarizes results
**Input**: Search term + reason
**Output**: Concise summary (2-3 paragraphs, <300 words)
**Tools**: WebSearchTool
**Model**: gpt-4o-mini

### 4. WriterAgent
**Purpose**: Synthesizes search results into final report
**Input**: Original query + search results
**Output**: ReportData
**Schema**:
```python
class ReportData:
    short_summary: str           # 2-3 sentence summary
    markdown_report: str         # Final detailed report
    follow_up_questions: list[str]  # Suggested further research topics
```

### 5. EmailAgent
**Purpose**: Sends formatted email with research report
**Input**: Markdown report
**Output**: Email sent via SendGrid
**Tools**: send_email function
**Model**: gpt-4o-mini

## Agent Flow
```
Query → PlannerAgent → [SearchAgent × 5] → WriterAgent → EmailAgent
``` 