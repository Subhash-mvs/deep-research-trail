import asyncio
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import re
from openai import OpenAI
from googlesearch import search
from crawl4ai import AsyncWebCrawler
import time
import urllib.parse
import random
import requests    
from bs4 import BeautifulSoup     

@dataclass
class WebsiteReport:
    """Holds scraped website data and relevance analysis"""
    url: str
    content: str
    relevance_score: float
    summary: str
    relevant_info: str

@dataclass
class ResearchReport:
    """Final research report structure"""
    query: str
    subcomponents: List[str]
    findings: Dict[str, List[WebsiteReport]]
    final_report: str
    timestamp: str
    sources: List[str]

class CustomResearcher:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the researcher with config file"""
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.client = OpenAI(api_key=config['api_key'])
        self.max_loops = config['max_loops']
        self.search_results_per_query = config['search_results_per_query']
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/127.0",
        ]
        
    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define OpenAI function calling tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_search_queries",
                    "description": "Generate Google search queries using various operators for comprehensive research",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": """List of search queries using Google operators:
                                - site: (search within specific site)
                                - intitle: (words in title)
                                - inurl: (words in URL)
                                - intext: (words in page content)
                                - filetype: (specific file types like pdf, doc)
                                - related: (sites similar to URL)
                                - cache: (Google's cached version)
                                - link: (pages linking to URL)
                                - "exact phrase" (exact match)
                                - -word (exclude word)
                                - word1 OR word2 (either term)
                                - word1 AND word2 (both terms)
                                - AROUND(X) (words within X words of each other)
                                - define: (definitions)
                                - weather: (weather info)
                                - stocks: (stock info)
                                - map: (map results)
                                - movie: (movie info)
                                - before:YYYY-MM-DD (before date)
                                - after:YYYY-MM-DD (after date)
                                - @socialmedia (search social media)
                                - #hashtag (search hashtags)
                                - $price (price search)
                                - number..number (number range)
                                Combine operators intelligently based on research needs."""
                            },
                            "operator_rationale": {
                                "type": "object",
                                "description": "Explanation for why specific operators were chosen",
                                "properties": {
                                    "site_operators": {"type": "string"},
                                    "time_operators": {"type": "string"},
                                    "content_operators": {"type": "string"},
                                    "file_operators": {"type": "string"}
                                }
                            },
                            "knowledge_gaps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Identified knowledge gaps that need more research"
                            }
                        },
                        "required": ["queries"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_website_relevance",
                    "description": "Analyze if website content is relevant to the research query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relevance_score": {
                                "type": "number",
                                "description": "Relevance score from 0 to 1"
                            },
                            "summary": {
                                "type": "string",
                                "description": "Brief summary of the website content"
                            },
                            "relevant_info": {
                                "type": "string",
                                "description": "Extracted relevant information"
                            }
                        },
                        "required": ["relevance_score", "summary", "relevant_info"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "decompose_query",
                    "description": "Break down complex queries into subcomponents",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subcomponents": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of query subcomponents"
                            }
                        },
                        "required": ["subcomponents"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_final_report",
                    "description": "Create the final research report",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "report": {
                                "type": "string",
                                "description": "Final markdown formatted report"
                            },
                            "has_sufficient_info": {
                                "type": "boolean",
                                "description": "Whether enough information was gathered"
                            },
                            "missing_info": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of missing information if any"
                            }
                        },
                        "required": ["report", "has_sufficient_info"]
                    }
                }
            }
        ]
    
    async def _crawl_website(self, url: str) -> Optional[str]:
        """Crawl a website and return its content"""
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                return result.markdown if result else None
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            return None
    
    def _google_search(self, query: str) -> List[str]:
        """Search via googlesearch; on quota/HTTP errors fall back to requests scrape."""
        # 1️⃣ primary attempt with googlesearch
        try:
            return [
                url
                for url in search(
                    query,
                    num_results=self.search_results_per_query,
                )
            ]
        except Exception as e:
            print(f"[googlesearch] {e} – falling back to requests scrape")

        # 2️⃣ fallback
        return self._scrape_google_serp(query)
    
    def _scrape_google_serp(self, query: str) -> List[str]:
        """Very lightweight HTML scrape of Google SERP (first N organic links)."""
        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={encoded}&num={self.search_results_per_query}"

        headers = {"User-Agent": random.choice(self.user_agents)}
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[fallback] HTTP error: {e}")
            return []

        soup = BeautifulSoup(r.text, "html.parser")

        # Google's markup: each result lives in <div class="g"> → first <a>
        links = []
        for a in soup.select("div.g a"):
            href = a.get("href")
            if href and href.startswith("http"):
                links.append(href)
            if len(links) >= self.search_results_per_query:
                break
        return links

    
    def _call_openai_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """Call OpenAI API with function calling"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        return response
    
    async def _analyze_website_content(self, url: str, content: str, query: str) -> WebsiteReport:
        """Analyze website content for relevance"""
        messages = [
            {"role": "system", "content": "You are a research analyst. Analyze the website content for relevance to the research query."},
            {"role": "user", "content": f"Research query: {query}\n\nWebsite URL: {url}\n\nContent:\n{content[:10000]}"}
        ]
        
        response = self._call_openai_with_tools(messages, self._define_tools())
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "analyze_website_relevance":
                args = json.loads(tool_call.function.arguments)
                return WebsiteReport(
                    url=url,
                    content=content,
                    relevance_score=args["relevance_score"],
                    summary=args["summary"],
                    relevant_info=args["relevant_info"]
                )
        
        # Default if no tool call
        return WebsiteReport(url=url, content=content, relevance_score=0, summary="", relevant_info="")
    
    async def _research_subcomponent(self, subquery: str, loop_count: int = 0) -> Dict[str, Any]:
        """Research a single subcomponent with loop handling"""
        all_reports = []
        knowledge_gaps = []
        search_history = []  # Track previous searches to avoid repetition
        
        while loop_count < self.max_loops:
            # Generate search queries with context about previous searches
            messages = [
                {"role": "system", "content": """You are an expert research assistant skilled in using Google search operators.
                Based on the research topic and any knowledge gaps, create diverse search queries using appropriate operators but they should have reseach topic in it.
                Consider:
                - Use site: for specific domains (news sites, academic sites, social media)
                - Use filetype: for PDFs, docs when looking for reports/papers
                - Use intitle: or inurl: for specific types of pages
                - Use date operators (before:/after:) for time-sensitive information
                - Use quotes for exact phrases
                - Use OR for alternative terms
                - Use -term to exclude irrelevant results
                - Use AROUND(X) for related concepts
                - Use @platform for social media searches
                - Combine multiple operators for precision"""},
                {"role": "user", "content": f"""Generate search queries for: {subquery} and strictly make sure search queries dont devaite from {subquery}
                Knowledge gaps from previous searches: {knowledge_gaps}
                Previous searches conducted: {search_history}
                Create only 2 queries using different operators to get comprehensive results."""}
            ]
            
            response = self._call_openai_with_tools(messages, self._define_tools())
            
            if not response.choices[0].message.tool_calls:
                break
                
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name != "generate_search_queries":
                break
                
            args = json.loads(tool_call.function.arguments)
            search_queries = args.get("queries", [])
            operator_rationale = args.get("operator_rationale", {})
            
            print(f"\nLoop {loop_count + 1} - Generated queries:")
            for q in search_queries:
                print(f"  - {q}")
            if operator_rationale:
                print(f"  Rationale: {operator_rationale}")
            
            # Add to search history
            search_history.extend(search_queries)
            
            # Perform searches and crawl
            for search_query in search_queries:
                urls = self._google_search(search_query)
                time.sleep(.600)
                
                for url in urls[:3]:  # Limit to top 3 results per query
                    content = await self._crawl_website(url)
                    if content:
                        report = await self._analyze_website_content(url, content, subquery)
                        if report.relevance_score > 0.3:
                            all_reports.append(report)
                            print(f"{url}..The info is relevant")
            
            # Check if we have sufficient information
            messages = [
                {"role": "system", "content": "You are a research analyst. Determine if we have sufficient information."},
                {"role": "user", "content": f"Query: {subquery}\n\nCollected information:\n" + 
                 "\n".join([f"- {r.relevant_info}" for r in all_reports])}
            ]
            
            response = self._call_openai_with_tools(messages, self._define_tools())
            
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function.name == "create_final_report":
                    args = json.loads(tool_call.function.arguments)
                    if args.get("has_sufficient_info", False):
                        break
                    knowledge_gaps = args.get("missing_info", [])
            
            loop_count += 1
            
            # Add delay to avoid rate limiting
            await asyncio.sleep(1)
        
        return {
            "subquery": subquery,
            "reports": all_reports,
            "loop_count": loop_count,
            "search_queries_used": search_history
        }
    
    async def research(self, query: str) -> ResearchReport:
        """Main research method with guaranteed complete reports"""
        print(f"Starting research for: {query}")
        
        # Check if query is complex and needs decomposition
        messages = [
            {"role": "system", "content": "You are a research planner. Decompose complex queries into subcomponents."},
            {"role": "user", "content": f"Query: {query}\nIs this complex? If yes, break it into subcomponents. The subcomponents should be detailed and not overlaping. The subcomponents shouldnt forget its roots and it should always have connection to the main topic"}
        ]
        
        response = self._call_openai_with_tools(messages, self._define_tools())
        
        subcomponents = [query]  # Default to single component
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "decompose_query":
                args = json.loads(tool_call.function.arguments)
                subcomponents = args.get("subcomponents", [query])
        
        # Research each subcomponent and generate individual reports
        all_findings = {}
        subcomponent_reports = {}
        
        print(f"Query decomposed into {len(subcomponents)} subcomponents:")
        for idx, subcomponent in enumerate(subcomponents, 1):
            print(f"\n{idx}. {subcomponent}")
        
        for subcomponent in subcomponents:
            print(f"\n{'='*60}")
            print(f"Researching subcomponent: {subcomponent}")
            print(f"{'='*60}")
            
            # Get research results
            result = await self._research_subcomponent(subcomponent)
            all_findings[subcomponent] = result["reports"]
            
            # Generate report for this subcomponent
            if result["reports"]:
                print(f"\nGenerating report for subcomponent with {len(result['reports'])} relevant sources...")
                
                # Direct approach without tool calling for subcomponent reports
                subcomponent_report = self._generate_subcomponent_report(subcomponent, result["reports"])
                subcomponent_reports[subcomponent] = subcomponent_report
                print(f"✓ Report section generated for: {subcomponent}")
            else:
                print(f"⚠ No relevant findings for: {subcomponent}")
                subcomponent_reports[subcomponent] = f"No relevant information found for this topic.\n"
        
        # Combine all subcomponent reports into final report
        print(f"\n{'='*60}")
        print("Combining all sections into final report...")
        print(f"{'='*60}")
        
        # Generate executive summary directly
        total_sources = sum(len(v) for v in all_findings.values())
        executive_summary = self._generate_executive_summary(query, subcomponents, total_sources, all_findings)
        
        # Generate conclusion directly
        conclusion = self._generate_conclusion(query, all_findings, subcomponent_reports)
        
        # Combine all parts
        final_report_parts = [
            "# Research Report\n\n",
            executive_summary,
            "\n## Detailed Findings\n"
        ]
        
        for subcomponent, report in subcomponent_reports.items():
            final_report_parts.append(f"\n### {subcomponent}\n")
            final_report_parts.append(report)
            final_report_parts.append("\n")
        
        final_report_parts.append(conclusion)
        
        # Combine everything
        final_report = "".join(final_report_parts)
        
        # Collect all sources
        all_sources = []
        for reports in all_findings.values():
            all_sources.extend([r.url for r in reports])
        
        research_report = ResearchReport(
            query=query,
            subcomponents=subcomponents,
            findings=all_findings,
            final_report=final_report,
            timestamp=datetime.now().isoformat(),
            sources=list(set(all_sources))
        )
        
        # Save report
        self._save_report(research_report)
        
        print(f"\n{'='*60}")
        print(f"✓ Research complete! Total sources: {len(research_report.sources)}")
        print(f"✓ Report saved as: research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        print(f"{'='*60}")
        
        return research_report

    def _generate_subcomponent_report(self, subcomponent: str, reports: List[WebsiteReport]) -> str:
        """Generate report for a subcomponent using direct OpenAI call"""
        try:
            findings_text = "\n\n".join([
                f"Source: {r.url}\n"
                f"Summary: {r.summary}\n"
                f"Key Information: {r.relevant_info}"
                for r in reports[:10]  # Limit to top 10 to avoid token limits
            ])
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research analyst. Create a detailed, well-structured report section."},
                    {"role": "user", "content": f"""Create a comprehensive report section for the topic: {subcomponent}

    Based on these findings:
    {findings_text}

    Requirements:
    1. Start with a brief overview
    2. List key findings with bullet points
    3. Identify trends and patterns
    4. Provide insights and analysis
    5. Use clear headings and formatting
    6. Cite sources where appropriate

    Write in a professional but accessible style."""}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating subcomponent report: {e}")
            # Fallback report generation
            return f"""## Analysis of {subcomponent}

    Based on {len(reports)} sources analyzed:

    ### Key Findings:
    """ + "\n".join([f"- {r.relevant_info[:200]}... (Source: {r.url})" for r in reports[:5]])

    def _generate_executive_summary(self, query: str, subcomponents: List[str], total_sources: int, findings: Dict) -> str:
        """Generate executive summary using direct OpenAI call"""
        try:
            # Prepare summary of findings
            findings_summary = []
            for subcomp, reports in findings.items():
                if reports:
                    findings_summary.append(f"- {subcomp}: {len(reports)} relevant sources found")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research report writer. Create clear, concise executive summaries."},
                    {"role": "user", "content": f"""Create an executive summary for this research report.

    Original Query: {query}

    Research Approach:
    - Query was analyzed and broken into {len(subcomponents)} key areas
    - Total of {total_sources} sources were analyzed
    - Subcomponents researched:
    {chr(10).join(f'  • {sc}' for sc in subcomponents)}

    Findings Overview:
    {chr(10).join(findings_summary)}

    Write a 2-3 paragraph executive summary that:
    1. Restates the research objective
    2. Explains the methodology briefly
    3. Highlights the most significant findings
    4. Sets up the detailed sections that follow

    Keep it concise but informative."""}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return f"## Executive Summary\n\n{response.choices[0].message.content}\n"
            
        except Exception as e:
            print(f"Error generating executive summary: {e}")
            # Fallback summary
            return f"""## Executive Summary

    This research report examines "{query}" through a comprehensive analysis of {total_sources} sources. The research was structured around {len(subcomponents)} key areas to ensure thorough coverage of the topic. Each area was investigated through targeted searches and careful analysis of relevant sources.

    The following sections provide detailed findings for each research component, offering insights, trends, and key discoveries from the analyzed sources.
    """

    def _generate_conclusion(self, query: str, findings: Dict, subcomponent_reports: Dict) -> str:
        """Generate conclusion using direct OpenAI call"""
        try:
            # Prepare key findings summary
            key_findings = []
            for subcomp, reports in findings.items():
                if reports and len(reports) > 0:
                    key_findings.append(f"{subcomp}: {len(reports)} sources analyzed, revealing significant developments")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert research analyst. Create insightful, synthesizing conclusions."},
                    {"role": "user", "content": f"""Create a conclusion for this research report on: {query}

    Key areas researched:
    {chr(10).join(f'- {k}' for k in findings.keys())}

    Total sources analyzed: {sum(len(v) for v in findings.values())}

    Write a conclusion that:
    1. Synthesizes the key findings across all areas
    2. Identifies major themes and patterns
    3. Highlights the most significant insights
    4. Discusses implications for the future
    5. Suggests areas for further research if applicable

    Make it comprehensive but concise (3-4 paragraphs)."""}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            return f"\n## Conclusion\n\n{response.choices[0].message.content}\n"
            
        except Exception as e:
            print(f"Error generating conclusion: {e}")
            # Fallback conclusion
            return f"""\n## Conclusion

    This comprehensive research into "{query}" has revealed significant insights across multiple dimensions. The analysis of {sum(len(v) for v in findings.values())} sources provides a robust foundation for understanding the current state and future trajectory of this topic.

    The findings demonstrate clear patterns of innovation and development, with implications for both immediate applications and long-term strategic planning. As this field continues to evolve, ongoing monitoring and research will be essential to stay current with emerging trends and opportunities.
    """
        
    
    def _save_report(self, report: ResearchReport):
        """Save report as markdown file"""
        filename = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report.final_report)
            f.write("\n\n## Sources\n")
            for i, source in enumerate(report.sources, 1):
                f.write(f"{i}. {source}\n")
            f.write(f"\n\n*Report generated on: {report.timestamp}*")
        
        print(f"Report saved as: {filename}")

# Example usage
async def main():
    # Initialize researcher with config file
    researcher = CustomResearcher(config_path="config.json")
    
    # Example queries demonstrating different operator needs
    queries = [
        "why did snow white 2025 movie fail"
    ]
    
    for query in queries:
        try:
            report = await researcher.research(query)
            print(f"\nCompleted research for: {query}")
            print(f"Subcomponents: {report.subcomponents}")
            print(f"Sources found: {len(report.sources)}")
            print("-" * 50)
        except Exception as e:
            print(f"Error researching '{query}': {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
