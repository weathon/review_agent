# Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents}

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. 

To address this gap, we present PROBE (Proactive Resolution of Bottlenecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and show promising future directions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a benchmark for evaluating proactivity of LLM agents. The authors define proactivity as a three-stage process: 1) searching over datastore; 2) identifying the most critical bottleneck; 3) executing an appropriate resolution.
They construct a synthetic data generation pipeline, which consists of a  synthetic world model to provide persona profile, and a user datastore to simulate the working environment context. They created 1,000 diverse test samples, and test LLMs like GPT-5, Claude Opus-4.1 and agent frameworks like ReACT, Reflexion, ReWoo, showing that current models performs poorly on the benchmark.

### Strengths
1. The paper addresses an important and under-explored problem. Proactivity is an important aspect intelligent agents, yet current research mainly focuses on reactive paradigms. Constructing a benchmark for this direction is valuable.
2. The proactive behavior is decomposed into  three distinct stages: search, identification, and execution. This provides a clear and interpretable lens for analyses and could inspire more principled model designs.

### Weaknesses
1. The paper shares similar concepts with Lu et al. (2024), which also investigates proactive behavior and synthetic benchmarks. It would be benefit for the paper to clearer explain how it extends or complements that prior work.
2. Unlike Lu et al. (2024), which also proposed models and data pipelines to improve proactivity, this work remains only an evaluation benchmark, without algorithmic insights or new modeling strategies.
3. The benchmark scenarios are restricted to synthetic, office-style contexts. It is unclear whether these setups can generalize to other forms of proactive tasks such as embodied or multimodal environments.
4. The environment information is under-specified. The environment (or user datastore) is central to enabling proactivity, because agents must perceive the current state and generate appropriate proposals. More details and explanation are needed to show how much genuine proactivity can be tested under this setup.
5. The submission appears somewhat rough:
    * The title on the ICLR submission page includes an extra closing brace “}”;
    * Several lines contain garbled symbols (e.g., lines 92, 296, 364, 560, 566, 1064, etc.);
    * The references list the same Lu et al. paper multiple times as “2024a/2024b.” 
The authors can carefully polish and refine the paper to improve its overall clarity and presentation quality.

### Questions
Please refer to the Weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose PROBE, a benchmark designed to evaluate the capabilities of proactive LLM agents in searching, identifying issues, and executing actions to solve them. They employ a world model to generate the entire dataset, which contains 1,000 entries, and have three annotators assess the difficulty of the generated samples. Among these, only 26 samples were successfully completed by humans. The authors further compare the performance of several state-of-the-art models, highlighting the challenging nature of the dataset. In addition, they conduct an error analysis to identify and categorize the key issues faced by proactive agents.

### Strengths
1. The authors present a clear and systematic formulation of the data generation pipeline, which helps readers understand the overall design and underlying methodology of the proposed benchmark.
2. The paper provides a detailed error analysis, enabling readers to identify the key bottlenecks and challenges in proactive problem-solving for LLM-based agents.

### Weaknesses
1. Lack of practical validation for the benchmark design. There is no clear evidence demonstrating that the constructed benchmark is well designed from practical or realistic considerations. It is particularly concerning that even human experts failed to complete most of the tasks, while some state-of-the-art models managed to perform better — a result that appears counterintuitive and warrants further explanation.
2. Unreliable evaluation using LLM-as-a-Judge. The evaluation process lacks robustness, as the authors only assessed 50 GPT-4.1 prediction–output pairs, with no detailed description of the evaluation criteria, sampling process, or inter-rater consistency. This raises doubts about the reliability and reproducibility of the reported results, especially when considering the human experts failed in Table 2.
3. Unfair comparison in the agentic framework. The proposed agentic framework demonstrates very poor performance, which raises concerns about the fairness of the comparisons made. As mentioned by the authors (line 387), the current setup may not ensure an equitable evaluation across different models. Additional experiments or ablation studies are needed to provide a more balanced and fair comparison.

### Questions
1. Could the authors provide more details about the human evaluation process? Specifically, how is task difficulty validated when human annotators achieve near-zero performance in the Bottleneck Identification component?
2. Is the benchmark entirely generated by language models? If so, which model(s) were used, and what was the rationale behind their selection?
3. How do the authors ensure that the evaluation on PROBE accurately reflects real-world agent proactiveness? Does the simulation include any real-world scenarios or instances, or is it purely synthetic? Clarifying this would help assess the benchmark’s external validity.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a benchmark, PROBE, to test LLMs abilities at proactively identifying issues. PROBE is synthetically generated using personas extracted from real LinkedIn users and an LLM generated bottleneck from which problematic and distractor documents are generated to be identified. PROBE tests LLMs abilities at searching for problematic documents, identifying bottlenecks, and choosing the right solution from a set of actions to address the bottleneck.

### Strengths
$Originality$: This paper introduces a new benchmark testing LLMs ability at proactiveness in a new setting: identifying bottlenecks, problematic documents, and resolution actions given a world model (user)

$Quality$: The benchmark has been tested carefully including a user study that revealed the difficulty of the tasks and an LLM-as-a-judge evaluator that is well aligned with annotators

$Clarity$: The benchmark details were easy to understand and the results are clear

$Significance$: Proactive LLMs is an increasingly important field that requires more benchmarks

### Weaknesses
- The user study was conducted on about 3% of the entire benchmark and performance was very low which could suggest the benchmark is difficult or the synthetic generation pipeline has issues
- The quality of the benchmark is based on iterating over prompts and running an adversarial agent to locate artifacts in 5 samples until no artifacts are detected.

### Questions
- Why does the Bottleneck Identification performance plateau around 42-43% across various models?
- How does GPT-4.1-mini achieve the same bottleneck identification performance as GPT-4.1 with significantly worse search ability?
- The agent frameworks are very low compared to the comparable base model GPT-4.1-mini (which the caption states is similar to GPT-5 mini performance). This is surprising to me, especially since Reflexion performs about the same as ReAct. Where are the baseline / base model prompts for conducting the action located and where do they differ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces PROBE, a benchmark and accompanying pipeline designed to evaluate the proactivity of LLMs through a structured workflow: searching unsolved issues, identifying bottlenecks and executing appropriate resolutions. The benchmark has been applied across diverse models and agentic frameworks, with results revealing significant challenges in achieving effective proactive assistance.

### Strengths
1. This paper introduces a focus on long-term proactivity, which meaningfully distinguishes it from prior work on proactive tasks that typically address short-horizon or immediate user intents.
2. The paper presents a well-structured and systematic pipeline for constructing the PROBE benchmark—a notable contribution that advances the methodology for synthetic data generation in agent evaluation.

### Weaknesses
1. Insufficient Benchmark Description: The paper does not adequately present the PROBE benchmark. It only reports aggregate statistics—such as the number of tokens, actions, and documents without providing concrete details about the tools used, the scenarios covered, or other essential contextual information.
2. Unclear Realism Gap: The distinction between synthetic data and real-world scenarios remains ambiguous. The claim that the benchmark is “realistic” appears largely subjective. In contrast, works like *Proactive Agent* using real-world data in their evaluation.

### Questions
1. What specific scenarios does your benchmark include? This information is essential for evaluating the benchmark’s scope and should be clearly presented in the main paper.
2. What distinguishes your benchmark from previous benchmarks? 
3. When and how will your agent perform proactive actions? How could users or annotators acknowledge their need for proactive problem solving in your settings?
4. Appendix Section B.3.1 shows the prompt for search, identification, and task-selection pipeline. The current design requires the model to output all three actions in a single response. However, this should be a multi-step process. How could LLMs leverage the context of searching results?
5. How does persona meaningfully influence LLM behavior or performance? From the current description, personas are used primarily to structure interpersonal relationships and contextualize bottlenecks. Yet, since all relevant evidence (e.g., emails, documents, calendar entries) is explicitly provided, it remains unclear how “person attribution” affects bottleneck identification.

### Soundness
2

### Presentation
2

### Contribution
2
