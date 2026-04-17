# The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution

- Decision: Accept (Poster)
- Scores: 4, 6, 2

## Abstract
Real-world language agents must handle complex, multi-step workflows across diverse applications. For instance, an agent may manage emails by coordinating with calendars and file systems, or monitor a production database like BigQuery to detect anomalies and generate reports following a standard operating manual. However, existing language agent benchmarks often focus on narrow domains or simplified tasks that lack the diversity, realism, and long-horizon complexity required to evaluate agents' real-world performance. To address this gap, we introduce the Tool Decathlon (dubbed as Toolathlon), a benchmark for language agents offering diverse applications and tools, realistic environment setup, and reliable execution-based evaluation. Toolathlon spans 32 software applications and 604 tools, ranging from everyday platforms such as Google Calendar and Notion to professional applications like WooCommerce, Kubernetes, and BigQuery. Most of the tools are based on a high-quality set of Model Context Protocol (MCP) servers that we may have revised or implemented ourselves. Unlike prior works, which primarily ensure functional realism but offer limited environment state diversity, we provide realistic initial environment states from real software, such as Canvas courses with dozens of students or real-world financial spreadsheets. The Toolathlon benchmark includes 108 manually sourced or crafted tasks in total, requiring interacting with multiple applications over around 20 turns on average to complete. Each task is strictly verifiable through dedicated evaluation scripts. Comprehensive evaluation of state-of-the-art models highlights their significant shortcomings in performing real-world, long-horizon tasks: the best-performing model, Claude-4.5-Sonnet, achieves only a 38.6% success rate with 20.2 tool calling turns on average, while the top open-weights model DeepSeek-V3.2-Exp reaches 20.1%. We expect Toolathlon to drive the development of more capable language agents for real-world, long-horizon task execution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces TOOLATHLON, a benchmark designed to evaluate LLMs on complex, real-world tasks using MCP tools. It extends existing MCP benchmarks by incorporating 108 real-world tasks and 604 tools across 32 different software applications. The authors benchmark current LLMs on this dataset and find that state-of-the-art models still perform poorly: the best model, Claude-4-Sonnet, achieves only a 29.9% success rate. These results highlight that long-horizon, multi-application task execution remains a significant challenge for current AI agents.

### Strengths
1. The benchmark contains a substantially larger number of tasks than prior MCP benchmarks and evaluates models across realistic settings, including real-world state initialization and fuzzy instructions.

2. Rigorous task validation and design make the benchmark reliable and valuable for MCP-agent research.

3. The authors benchmark a wide range of state-of-the-art LLMs and provide detailed analysis, including case studies of failure modes and in-depth discussion of experimental results.

### Weaknesses
The current evaluation setup for LLM agents with tools is relatively simple. The authors specify all tools for a task and provide the full tool list to the agent, which can significantly inflate the input context. More sophisticated tool-selection strategies could be explored—for example, using retrieval methods to surface relevant tools dynamically rather than supplying the entire tool inventory upfront.

### Questions
see above

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Tool Decathlon (TOOLATHLON), a new benchmark designed to evaluate tool-using language agents on tasks that are diverse, realistic, and long-horizon. The authors argue that existing benchmarks are often limited in scope, focusing on narrow domains, single applications, or tasks that lack realistic complexity and environment states. To address this, TOOLATHLON spans 32 software applications (from everyday tools like Google Calendar to professional ones like Kubernetes and BigQuery) and 604 tools, mostly sourced from real-world Model Context Protocol (MCP) servers.

### Strengths
- A strong benchmark, heavy engineering behind 
- will be good use for the community

### Weaknesses
- Not really a weakness but the paper uses the average number of turns as a proxy for task difficulty. While reasonable, this is an outcome-based metric that can be influenced by the agent's (in)efficiency. A more intrinsic, task-defined complexity metric (e.g., based on the number of required applications, minimum number of steps in a ground-truth trajectory) could provide a slightly more objective measure of difficulty when analyzing performance across Easy/Medium/Hard tasks.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces TOOLATHLON (Tool Decathlon) — a benchmark designed to evaluate the performance of language agents on complex, long-horizon, and cross-application tasks. The benchmark covers 108 tasks, 32 applications and 604 tools, ranging from common productivity software (e.g., Notion, Google Sheets, WooCommerce) to domain-specific systems (e.g., Kubernetes, Snowflake). Each of the 108 tasks requires multi-step reasoning, realistic initial environment states, and verifiable execution via deterministic scripts.

### Strengths
+ TOOLATHLON spans a 32 real-world applications demonstrating tool and environment diversity.

+ Execution-Based Evaluation: TOOLATHLON uses deterministic evaluation scripts that compare environment states, ensuring objectivity and reproducibility.

### Weaknesses
Benchmark Labeling and Definitions: The categorization of existing benchmarks in Table 1 as “Real Tools” or “Not-Real Tools” appears inconsistent and potentially misleading. For example, τ-Bench is flagged as not-real despite subsets (“airline,” “retail”) interacting with actual databases. Similarly, BFCL is marked not-real, although it supports real execution in the “Execute” category,  with "Crowd Sourced" being community-contributed tools. LiveMCPBench is labeled not-real despite its claim to run live MCP tools, while TOOLATHLON’s own use of simulated components (e.g., local Poste.io for Gmail) blurs the same boundary. 

Novelty Ambiguity: While TOOLATHLON integrates many existing elements (real MCP servers, realistic states, cross-app tasks), the conceptual novelty beyond combining these components is modest. The work would benefit from a clearer articulation of methodological advances versus engineering scale-up.

### Questions
Are the locally containerized applications (e.g., Poste.io for Gmail) treated as “real tools,” and if so, how do they differ philosophically from “simulated” counterparts in other benchmarks?

### Soundness
3

### Presentation
3

### Contribution
1
