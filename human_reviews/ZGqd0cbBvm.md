# InsightBench: Evaluating Business Analytics Agents Through Multi-Step Insight Generation

- Decision: Accept (Poster)
- Scores: 8, 3, 6, 6

## Abstract
Data analytics is essential for extracting valuable insights from data that can assist organizations in making effective decisions. We introduce InsightBench, a benchmark dataset with three key features. First, it consists of 100 datasets representing diverse business use cases such as finance and incident management, each accompanied by a carefully curated set of insights planted in the datasets. Second, unlike existing benchmarks focusing on answering single queries, InsightBench evaluates agents based on their ability to perform end-to-end data analytics, including formulating questions, interpreting answers, and generating a summary of insights and actionable steps. Third, we conducted comprehensive quality assurance to ensure that each dataset in the benchmark had clear goals and included relevant and meaningful questions and analysis. Furthermore, we implement a two-way evaluation mechanism using LLaMA-3 as an effective, open-source evaluator to assess agents’ ability to extract insights. We also propose AgentPoirot, our baseline data analysis agent capable of performing end-to-end data analytics. Our evaluation on InsightBench shows that AgentPoirot outperforms existing approaches (such as Pandas Agent) that focus on resolving single queries. We also compare the performance of open- and closed-source LLMs and various evaluation strategies. Overall, this benchmark serves as a testbed to motivate further development in comprehensive automated data analytics and can be accessed here: https://github.com/ServiceNow/insight-bench.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a new benchmark dataset called InsightBench, which evaluates end-to-end analytics tasks' performance in real-world enterprise troubleshooting. The benchmark includes 100 datasets extracted from the ServiceNow platform. The current benchmarks mainly focus on single round-trip query-answer evaluations. This proposed benchmark aims to evaluate an agent's holistic problem-solving skills.

In addition to proposing a new benchmark dataset, the authors also developed a new baseline agent called AgentPoirot, which demonstrated superior performance compared to existing popular agents such as Pandas Agent.

### Strengths
1) This paper is a leap forward in laying a foundation for building more robust enterprise-grade business analytics agents. I agree that the current benchmarks primarily just focus on simple code completion or single-query tasks. Real-world problems almost always involve multi-step reasoning and insight development.

2) The benchmark building process is very well-documented and detailed, which was a huge plus from me. It was also clever to use the real-world data schema from a popular troubleshooting ticketing platform, ServiceNow, which ensures the applicability of this approach.

3) Using LLaMA-3-Eval as an alternative to GPT-4 which is a closed API. For the sake of science, using LLaMa is a huge boost in transparency.

4) Plus, AgentPoirot showed impressive performance against Pandas Agent in end-to-end analysis. This result shows that properly designed benchmarks can drive the development of superior agents.

5) The quality assurance process using both expert reviewers and extensive validation was impressive.

### Weaknesses
1) Whether synthetic data represents the real-world context is questionable. The synthetic data generation part is less well-documented than other parts of the paper. The authors should explain better how their synthetic data generation process works and how they ensured the representativeness of the synthetic data.

2) The authors mentioned the four types of analytics up front (descriptive, diagnostic, predictive, and prescriptive). However, this notion of four types of analytics was not as prevalent in the later part, where the results were presented. Consider looping back to these four types of analytics when presenting the results and performance.

3) I understand Pandas Agent is probably the most popular agent aiming to achieve similar goals. However, I would like to see the comparison against some other agents, if there are any.

4) This paper focuses too much on showing the strengths of the benchmark and the specialized agent, Agent Poirot. Providing the boundary conditions and limits where the agent fails would be helpful for future studies in this line.

### Questions
1) Do you have any evidence that the distribution of your data (for example, insight types--incident, user, asset, finance, goal management) resembles real-world business scenarios?

2) Agent Poirot works best with a temperature of 0.2. This value looks interesting to me. Do you know why the agent works best at such a low temperature?

3) It was encouraging to see that LLaMa-3-Eval produced consistent outcomes compared to G-Eval, which is known to be well aligned with human judgments. Was there any way to evaluate whether LLaMa-3-Eval output is also directly aligned with human judgments?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
InsightBench is a new way to test how well AI language models can handle complex business analysis tasks. Unlike older tests that only look at simple questions, this one checks if AI can do a whole analysis from start to finish - asking questions, finding important information, and suggesting what to do next.

The creators made 100 different datasets about things like finance and managing problems. Some of the data is made up, and some is real, to make it feel like actual business situations. They also came up with their own AI agent called AgentPoirot, which did better than other AI tools at figuring out different kinds of insights from the data.

To make sure anyone can use and improve on their work, the team used an open-source AI called LLaMA-3 to judge how well the AI agents do. They hope this will help push forward the development of AI that can do complete data analysis for businesses without needing humans to guide it every step of the way.

### Strengths
1. Diverse coverage: The dataset covers diverse business domains like finance and incident management, so can be widely useful, although one can argue that the dataset is not focused on a particular domain.
1. Quality: The dataset insights are evaluated by experts, hence there is an assurance of quality. The two way evaluation (summary + insights) will also help users of the platform be more confident in the results.

### Weaknesses
1. Synthetic data: This is a long standing tussle for evaluation -- how realistic the data can get. InsightBench datasets depend on synthetic data generation which can make the data quality lower as the data regresses from real world scenarios. Particularly, it is unclear how well the data represents the unhappy paths / scenarios often found in real world. One suggestion might be to further split up the datasets into Happy path and Challenge datasets, with challenge datasets using less synthetic data, and being more realistic of real world scenarios.
1. Dependency on structure: InsightBench focuses on structured, tabular data, but increasingly other data formats like unstructured text, images, or complex graphs are becoming relevant for business analytics.
1. Scale: It is unclear how the dataset can be kept up to date with evolving business tasks, how quality will be maintained, how new domains will be added etc.

### Questions
1. Synthetic data
   * How do you ensure that InsightBench datasets capture the complexity of real-world enterprise data?
   * Could you elaborate on the methods used to simulate unpredictable or "noisy" data patterns?
   * Have you incorporated real-world anomalies or edge cases into your data generation process?
   * Have you validated the synthetic data against any real-world datasets?
1. Dependency on structure
   * Do you have plans to incorporate unstructured data (like text logs or emails) that is common in businesses?
1. Scale
   * Do you have a process in place for regularly updating the dataset, or have you considered partnering with industry experts to ensure the dataset remains relevant to current business practices?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces InsightBench, a benchmark dataset designed to evaluate the performance of data analytics agents in generating multi-step insights from business data. The authors propose AgentPoirot, a new agent model that outperforms existing tools such as the Pandas Agent. The benchmark dataset includes 100 datasets across various business themes, each with planted insights for the agents to identify. A two-way evaluation mechanism using LLaMA-3 is implemented to assess agent performance.

### Strengths
· Originality: The introduction of InsightBench is a significant innovation in the evaluation of data analytics agents. Unlike previous benchmarks, InsightBench focuses on multi-step analytics, providing a comprehensive framework for assessing agents' abilities to perform holistic analysis rather than isolated tasks.

· Clarity: The paper is clearly written, with a logical progression from motivation to methodology and results. Figures and tables effectively support the text, particularly in explaining how InsightBench differs from existing single-step benchmarks.

· Significance: InsightBench is a meaningful contribution that could substantially impact the development of data analytics agents. By offering a platform for evaluating agents' end-to-end performance, InsightBench encourages the creation of more versatile and context-aware analytics models, filling a notable gap in the data analytics research community.

### Weaknesses
· Limitations in Real-World Validation: While InsightBench provides synthetic data and insights, the effectiveness of AgentPoirot and other evaluated agents in handling real-world, noisy datasets remains untested. To demonstrate its practical value, further evaluation on real-world data (beyond synthetic datasets) would enhance the relevance of InsightBench.

· Interpretability of LLaMA-3-Eval Scores: The LLaMA-3-Eval scoring method could benefit from additional explanation regarding its interpretability and stability, particularly in cases where scores may fluctuate with subtle prompt changes.

### Questions
1. How does InsightBench handle noisy or incomplete data? While the synthetic datasets are well-curated, it would be valuable to know how robust InsightBench and AgentPoirot are to missing data or noise, as these are common in real-world applications.

2. What is the scalability of InsightBench for larger datasets? Since InsightBench datasets are capped at 500 entries, how would the benchmark scale to larger, more complex datasets? Would AgentPoirot's performance degrade with larger data volumes?

3. Are there plans to incorporate real-world data into InsightBench? A dataset that mixes synthetic and real-world data could potentially improve the benchmark’s applicability to practical scenarios.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces InsightBench, a benchmark designed to evaluate the effectiveness of large language model-based (LLM-based) agents in performing comprehensive data analytics tasks. Unlike previous benchmarks, InsightBench focuses on end-to-end data analytics, assessing capabilities in question formulation, insight extraction, and summary generation across diverse business scenarios such as finance and incident management. InsightBench includes 100 datasets derived from the ServiceNow platform, enriched with specific insights and questions to evaluate an agent’s ability to recommend tasks and extract actionable insights. A baseline agent, AgentPoirot, is introduced and evaluated against other existing tools, with results showing it outperforms simpler query-based agents. The evaluation also incorporates LLaMA-3-Eval, an open-source alternative to GPT-4, offering a cost-effective and stable approach to assessing agent performance in text-based insight matching.

### Strengths
- InsightBench represents a significant step forward from single-query benchmarks, testing agents' holistic capabilities in an end-to-end analytics process more reflective of real-world data analysis.
- The paper showcases careful design in creating datasets with embedded trends, anomalies, and comprehensive goals that simulate actual enterprise challenges, making the benchmark realistic and versatile.
- By introducing LLaMA-3-Eval, the authors provide a stable and open-source alternative to GPT-4-based evaluations, reducing costs and ensuring reproducibility, a valuable contribution to the LLM community.
- AgentPoirot effectively demonstrates the viability of performing multi-step data analytics, including descriptive, diagnostic, predictive, and prescriptive tasks, positioning it as a useful baseline for future improvements in analytics agents.

### Weaknesses
- LLaMA-3-Eval offers advantages, but its consistency with GPT-4 evaluations could benefit from more rigorous comparison across various agent tasks beyond simple insight extraction.
- Despite the variety of themes, the benchmark may not fully capture the complexity of certain domains, potentially limiting generalizability to more complex or highly regulated fields like healthcare or law.
- Synthetic data generated from ServiceNow templates may lack the nuances of actual enterprise data, which could affect the evaluation's applicability to real-world use cases.

### Questions
- It would be valuable to test InsightBench on actual enterprise data and synthetic data to better understand its applicability to real-world tasks.
- Given the evolving nature of LLMs, a dynamic benchmark that updates or adjusts insight difficulty could be beneficial for testing agent adaptability over time.
- Further breakdown of evaluation scores (e.g., insight precision vs. general text similarity) could improve the interpretability and reliability of agent assessments.

### Soundness
3

### Presentation
3

### Contribution
3
