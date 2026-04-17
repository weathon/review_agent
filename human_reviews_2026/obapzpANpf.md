# DatasetResearch: Benchmarking Agent Systems for Demand-Driven Dataset Discovery

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
The rapid advancement of large language models has fundamentally shifted the bottleneck in AI development from computational power to data availability—with countless valuable datasets remaining hidden across specialized repositories, research appendices, and domain platforms. As reasoning capabilities and deep research methodologies continue to evolve, a critical question emerges: can AI agents transcend conventional search to systematically discover any dataset that meets specific user requirements, enabling truly autonomous demand-driven data curation? We introduce DatasetResearch, the first comprehensive benchmark evaluating AI agents' ability to discover and synthesize datasets from 208 real-world demands across knowledge-intensive and reasoning-intensive tasks. Our tri-dimensional evaluation framework reveals a stark reality: even advanced deep research systems achieve only 22% score on our challenging DatasetResearch-pro subset, exposing the vast gap between current capabilities and perfect dataset discovery. Our analysis uncovers a fundamental dichotomy—search agents excel at knowledge tasks through retrieval breadth, while synthesis agents dominate reasoning challenges via structured generation—yet both catastrophically fail on ''corner cases'' outside existing distributions. These findings establish the first rigorous baseline for dataset discovery agents and illuminate the path toward AI systems capable of finding any dataset in the digital universe. Our benchmark and comprehensive analysis provide the foundation for the next generation of self-improving AI systems. The code and dataset will be open-sourced soon.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces DATASETRESEARCH, a benchmark for evaluating agent systems on demand-driven dataset discovery and synthesis. It curates 208 real-world dataset demands (from Hugging Face and Papers with Code) and pairs each with a reference dataset and reference metadata (MetaTriplets). Agents are assessed along three axes: (i) metadata alignment (o3-judged semantic similarity), (ii) few-shot performance (1/3/5-shot), and (iii) fine-tuning performance on LLaMA-3.1-8B, with scores normalized by the reference-data upper bound. Baselines span search agents, synthesis agents (o3-generated 500-sample datasets), and deep research agents. Results show a clear split: search excels on knowledge-based tasks, synthesis on reasoning-based tasks, while all methods struggle on the harder DatasetResearch-pro subset (best ≈0.22), highlighting substantial headroom for hybrid and more generalizable approaches.

### Strengths
Turns “find or build the dataset that matches a natural-language demand” into a measurable benchmark with paired reference data and metadata, covering the full path from requirements to downstream utility.

Combines intrinsic (metadata similarity) and extrinsic (few-shot and fine-tuned task performance) measures, with normalization that enables comparison across heterogeneous NLP tasks.

Systematically contrasts search, synthesis, and deep research paradigms, revealing a knowledge vs. reasoning specialization and consistent failure on corner cases—useful guidance for designing future hybrid agents.

### Weaknesses
- The same model family (o3) is used to generate reference metadata/demands, parse discovered data, and score alignment—inviting self-consistency bias rather than genuine agreement, and masking contamination through stylistic echoing.
- Overreliance on closed-source systems (o3 for synthesis/judging; GPT-4o/Deep Research variants for search) undermines reproducibility, accessibility, and cost realism. Results may reflect vendor-specific capabilities rather than agent design quality.
- No systematic comparisons with open retrieval stacks (e.g., BM25 + dense retrievers/ColBERT), open reasoning LMs (e.g., Llama-3.x-70B, Mistral-Large-Instruct, Qwen2.x/3-Instruct), or open toolformer/agent frameworks.
- Despite broad claims, coverage is text-only across six NLP tasks; no CV/audio/tabular/time-series/multimodal demands; limited external validity.
- The benchmark is closer to a controlled template-matching exercise that reproduces a known target than to open-world dataset discovery. In practice, the “task” is largely to find (or approximate) someone else’s already-curated dataset given a stylized natural-language demand. But in real settings, you usually don’t have such ready-made, perfectly matched datasets—you have to prospect, acquire, clean, align schemas, and handle licensing/privacy—so the setup falls well short of real-world data discovery.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper points out that relevant data forms a crucial bottleneck to advance AI models. The authors seek to answer if “conventional data search” methods can be replaced with “AI-agent search.” To answer this question, they propose a new benchmark aimed at evaluating AI agents’ ability to discover and synthetically generate datasets. Sourcing data from Hugging Face and papers with code, the authors built a benchmark that automatically generates data demands split between knowledge-intensive and reasoning-intensive tasks. They use this benchmark to evaluate several leading models and conclude that current models do not perform well. The authors further show that “search agents” do better at knowledge tasks while “synthesis agents” outperform on reasoning tasks. Both agent types perform poorly on “corner cases.”

### Strengths
[**significance**] The authors correctly identify data and data discovery as an important challenge to improving AI models. Efforts aimed at creating benchmarks designed to isolate capabilities useful for automating such challenges is an important endeavor.

### Weaknesses
[**clarity**]
- The paper (rightfully) emphasizes the importance of data to further advance AI models. Unfortunately, the problem specification is overly vague, entangling different challenges and data use cases. For example, the abstract mentions “countless valuable datasets [...] and domain platforms]” (l13-14], but does not specify if these are hidden due to access constraints or limitations attributable to search algorithms. 
- The word “synthesis” is used several times in the introduction without defining its meaning.
- The experimental setup misses many important details


[**quality**]
- The related work section is severely lacking. For example, it appears to completely ignore decades of information retrieval research. The literature on synthetic data generation similarly lacks any discussion of core concepts like diversity, complexity, and quality. Also absent is the extensive literature on “retrieval augmented generation” (RAG) based systems.
- The core contribution of this work is a new benchmark designed to simulate real-world data discovery. However, the methodology used to create this benchmark appears to have various questionable aspects (see questions). 
- The paper seemingly uses OpenAI’s o3 model for every step of the pipeline. This reviewer fears that any takeaways or analysis is therefore overly biased and does not necessarily generalize.
- Reported metrics lack confidence intervals.
- In Section 5.2, the authors write “we identify that [...] instruction-following capabilities” (l431-448). As OAI o3 is used to generate data, this can simply reflect existing data knowledge of o3, rather than any relation to retrieved or “discovered” data. This is an important confounding factor not accounted for in the empirical evaluation.

[**significance**] The authors claim that their work provides “the foundation for the next generation of self-improving AI systems” (l29-30), which is a lofty claim that does not appear to be supported by empirical or theoretical evidence.

### Questions
Q1. Confusing notation: “Given a natural language [...] the specified demand D” ( l146-148), in this notation, what is the subscript “d” in S_d? Do the authors mean a “set” of datasets? This continues in l150-153, where now “r_i” is used without introduction.


Q2. In Section 3.2, Step 1, the authors write that “gated” datasets are used to mitigate data leakage. What evaluation was performed to check against data leakage? 


Q3. In Section 3.2, Step 2 and 3, a number of filtering steps are performed to narrow down the dataset candidates. If the goal of this challenge is to measure models on “realistic” conditions, these steps appear to strongly bias the remaining set towards an unrepresentative sample.


Q4. In Section 3.2, Step 6-7: What were the rejection criteria to decide if a dataset was “unsuitable for fine-tuning” (l236), and what were the criteria used to check if the generated meta data and demand descriptions are faithful to the underlying data and ecologically valid? (l235-241)


Q5. In Section 3.2, l250-260, the authors propose a binary classification of knowledge-based vs. reasoning-based tasks. Yet, to this reviewer, it appears that *many* queries require a combination of these two. Could you please provide the systematic rubrics used to annotate these tasks? Were any consistency checks performed, e.g., cross-annotator agreement?


Q6. In 3.3 it is claimed that using OpenAI’s o3 model for scoring both reference and discovered metadata mitigates potential scoring biases (l296-l297). This claim lacks evidence, e.g., are the reference and discovered metadata distributions similar? Is scoring consistent across different types of underlying data? Is scoring robust across multiple samples and/or prompts?


Q7. The authors report a “Normalized Score” (l321), which finetunes a model on a reference dataset and uses this as the “theoretical maximum performance achievable” (l316-317). First, this assumes that a reference dataset contains both a train and test subset. Second, this reviewer sees no reason why combining one or multiple other datasets could not lead to a better performance. For example, training a model on a challenging math dataset and evaluating it on a simpler reference dataset fits this scenario. As such, what is the difference of dividing the S_{\text{eval}} by an arbitrary fixed number, given that scores are now “on a scale from 0 to 1, or higher” (l322)? 


Q8. A “synthesis agent” uses OAI o3 to generate 500 data samples (l357): How? What criteria are used to evaluate these samples?


Q9. How is finetuning done?


Q10. Key experimental setup details are missing to explain how the “deep research” systems of various providers were evaluated. The text mentions manual actions: what were these?

Suggestions:
- typo: “evaluable” (l204)

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores a novel agent application that utilizes LLMs to discover or synthesis datasets that meets specific user requirements. Its major contribution is the introduction of a dataset discovery benchmark covering 208 types of demands covering knowledge-intensive and reasoning-intensive tasks. The paper hopes the benchmark and analysis can benefit the progress of self-improving AI systems.

### Strengths
If well-justified, dataset discovery would be an interesting direction for LLM agents to explore.

### Weaknesses
1. The paper still needs in-depth justification on the motivation of dataset discovery demands. It is always intriguing to utilize LLM-based agents for exploring different applications. However, it is still lacking examples of practical use cases for human users to utilize dataset discovery agents.

2. The paper claims the dataset discovery agent shows interesting demands related to knowledge-intensive tasks or reasoning-intensive tasks. However, both tasks have mature strategies regarding data exploration/building. For instance, RAG and search agent related techniques are widely used in knowledge-intensive tasks; and reasoning tasks involves data synthesis (e.g., WizardMath) and RL-related long CoT & test-time scaling strategies. It is unclear how the proposed data discovery agent differs from these widely used existing methods on resolving related tasks.

3. The benchmark results in Table 2 seems incomplete. It misses multiple open-source models like QWen, DeepSeek, etc, and other popular models like Gemini and Claude series. I would also be interested to see performances of GPT-4.1 and GPT-5 models.

### Questions
None.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduce a benchmark designed to evaluate AI agents’ ability to autonomously discover or synthesize datasets given natural-language task requirements. 

The benchmark consists of 208 real-world NLP dataset, with reference datasets and metadata for objective comparison. Models are assessed via metadata alignment, few-shot performance, and fine-tuned downstream results. Experiments show a clear split: search-based agents excel at knowledge-oriented tasks, while synthesis-based models achieve superior performance on reasoning tasks. The work provides the first systematic evaluation pipeline for demand-driven data discovery.

### Strengths
1. First comprehensive framework targeting automated data discovery—a growing but under-studied problem.

2. Uses gated datasets + reference metadata, preventing leakage and reflecting real research workflows.

3. Combines metadata scoring, few-shot results, and fine-tuning—much richer than single-metric evaluation.

### Weaknesses
1. Used OpenAI o3 to generate both reference/discovered metadata and judges metadata similarity. This creates a closed loop that may favor o3’s rather than true task fit. 


2. When starting from gated datasets, it prevents agents from downloading the ground-truth data. This structurally disadvantages search agents (vs. synthesis) and conflates ``access policy'' with ``discovery ability.'' 

3. Data scope is narrow: NLP-only and text-only.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
