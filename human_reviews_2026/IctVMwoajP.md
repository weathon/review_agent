# WebAggregator: Scaling Complex Logical Information Aggregation for Web Agents Foundation Models

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2, 4

## Abstract
Deep research web agents must not only retrieve information from diverse sources such as web environments, files, and multimodal inputs, but more importantly, they need to rigorously analyze and aggregate knowledge in order to generate high-quality, insightful research.
However, existing open-source deep research agent systems predominantly focus on enhancing *information seeking* capabilities of web agents to *locate* specific information, while overlooking the essential need for *information aggregation*, which would limit their ability to generate coherent insights or support in-depth research.

In this paper, we propose a paradigm for scalably constructing verifiable training datasets for web agents, by framing data construction as an agentic task grounded in real web pages while placing additional focus on developing fine-grained rules that enable complex information aggregation.
Our approach synthesizes tasks by first collecting information through *proactive online web exploring* on the real web environment,
followed by *Complex Aggregation Logic Injection* to compose the verifiable question-answer pairs from aggregated knowledge snippets, covering over 12 logical operations.
The resulting dataset contains about 10K samples across 50K websites, covering more than 11 domains.
Based on an open-source agent framework, SmolAgents, we collect supervised fine-tuning trajectories to develop a series of foundation models, named WebAggregator.
WebAggregator-8B matches the performance of GPT-4.1, while the 32B variant surpasses GPT-4.1 by more than 10\% on GAIA-text and closely approaches the performance of Claude-3.7-sonnet.
Moreover, given the limited availability of benchmarks that evaluate web agents’ information aggregation abilities, we construct a human-annotated evaluation split of WebAggregatorQA as a challenging test set. On this benchmark, Claude-3.7-sonnet only achieves 28%, and GPT-4.1 scores 25.8%, and even after retrieving all of the references, they still struggle on WebAggregatorQA, highlighting the need to strengthen the information aggregation capabilities of web agent foundations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper, WebAggregator: Scaling Complex Logical Information Aggregation for Web Agent Foundation Models, presents a new paradigm for developing web agents capable of information aggregation rather than mere retrieval. The authors introduce WebAggregatorQA, a large-scale dataset automatically constructed through “Proactive Online Web Exploring” and “Complex Aggregation Logic Injection,” covering 10K samples from 50K websites across 12 domains. The dataset captures 12 categories of aggregation logic (e.g., mathematical, statistical, temporal reasoning) grounded in real web interactions. On top of this dataset, the authors train WebAggregator models (8B, 32B) based on the SmolAgents framework, achieving performance on par with or exceeding GPT-4.1 on GAIA-text and new benchmarks. The results reveal that even strong commercial models like GPT-4.1 and Claude 3.7 Sonnet underperform on the human-verified WebAggregatorQA test set, emphasizing the challenge of complex web information aggregation

### Strengths
The work tackles a critical bottleneck in web agent research—the lack of robust aggregation capabilities beyond information retrieval. The proposed data construction pipeline is scalable, verifiable, and largely autonomous, enabling reproducible and diverse task generation grounded in real-world web environments. The taxonomy of 12 logical operations provides fine-grained control over task complexity, bridging low-level data fusion with high-level reasoning. Empirical evaluations are rigorous and comprehensive, spanning multiple datasets (GAIA, WebWalkerQA, XBench) and baselines, demonstrating significant gains over prior open-source agents. The paper also shows strong generalization and efficiency, with smaller models (8B) achieving competitive results using only a fraction of the data. Overall, this is a technically solid and methodologically well-executed contribution toward building web agent foundation models.

### Weaknesses
Despite its innovation, the paper exhibits several limitations. The model diversity is insufficient—only Qwen-based models are explored, excluding more varied architectures or scales (e.g., Llama, Mistral, Gemini, Claude Opus). This limits generality and may bias the conclusions toward one training paradigm. The evaluation primarily relies on synthetic QA formulations, and the absence of end-to-end deployment testing under real dynamic web conditions weakens ecological validity. Furthermore, while the dataset captures aggregation logic, it does not explicitly disentangle reasoning from retrieval—leaving unclear whether improvements stem from aggregation training or memorization of structured tasks. The paper also lacks deeper error analysis or ablation on aggregation subtypes (e.g., temporal vs. statistical), and training details (e.g., efficiency, scaling laws) could be elaborated. Lastly, though claims of surpassing GPT-4.1 are impressive, comparisons against newer closed models (e.g., GPT-5, Claude 3.7 Opus, Gemini 2.5 Pro) would strengthen the claims of frontier performance.

### Questions
1. How does the model performance vary across different aggregation logic types (e.g., temporal vs. statistical reasoning), and what are the main failure modes?

2. Could the authors expand the benchmark by including more model families and scales, especially closed-source models like Claude 3 Opus, Gemini 2.5 Pro, or GPT-5, to better assess generality?

3. How does WebAggregatorQA handle dynamic and multimodal web content (e.g., changing JavaScript-rendered pages, tables, or charts), and can it adapt over time?

4. Have the authors considered integrating mechanistic interpretability or trace-level reasoning evaluation to better understand how the model aggregates information?

5. Can this data-construction pipeline be extended to multimodal or multi-agent settings, enabling aggregation of text, vision, and structured data jointly?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed a disciplined method to curate diverse web research tasks, which incorporates proactive online exploration and complex aggregation logic injection, and collected 6K trajectories to fine-tune LLMs for web-research tasks.
8B LLM fine-tuned with the collected trajectories achieved performance on par with GPT-4.1, and 32B LLM fine-tuned with the collected trajectories achieved performance higher than GPT-4.1 on GAIA benchmark.

### Strengths
* This paper introduced disciplined method to collect dataset for training web-based research agents at scale.
* Authors also provided WebAggregatorQA, which is challenging evaluation tasks aimed at evaluating research agents.
* Authors demonstrated the quality of the training dataset by achieving performance superior to previous baselines in GAIA benchmark.
* Comparison with comprehensive baselines and detailed analysis.

### Weaknesses
[W1] While the authors emphasize "information aggregation" over the "information seeking" in the abstract, empirical results demonstrating current research agents' inability to aggregate information is not provided. It would be better if the paper provides 1) failure modes of baselines (e.g., circumstances that baselines successfully seek web pages related to answer but failed to aggregate them in to a correct answer),  and 2) whether the agent trained with WebAggregatorQA dataset mitigate this issue. 

[W2] In section 2.2.2, explanation about aggregation logic injection is quite hard to clearly understand. I could understand the detailed process by looking at the prompts in the Appendix. It would be better if there is a figure showing the process of QA pair is being synthesized from the exploration trajectory.

[W3] (minor) In the current draft, abstract comprises 2 paragraphs and is quite lengthy. It would be better to merge them into a single paragraph and shorten the contents.

### Questions
[Q1]. How can we ensure that there is no QA samples that overlap with evaluation task in GAIA benchmark? 

[Q2]. In Appendix B.3, isn't the presented prompt for aggregation logic injection?  

[Q3]. Do the authors have any plan to opensource the collected training dataset and the model checkpoint?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces WebAggregator, an agent-based framework and dataset designed to improve web agents’ information aggregation capabilities. The authors propose an automated pipeline that constructs question–answer pairs by performing “Proactive Online Web Exploring” and “Complex Aggregation Logic Injection” on real web environments, resulting in a new dataset called WebAggregatorQA (about 10K samples from 50K websites). Using this dataset, the authors fine-tune open-source LLMs (Qwen2.5/3-7B/32B) to develop WebAggregator models, which reportedly outperform GPT-4.1 and Claude-3.7-sonnet on GAIA-text and the newly created benchmark. The paper argues that this dataset and framework strengthen information aggregation, a currently underexplored ability in web agents.

### Strengths
1. The paper is clearly written and well-structured, with detailed methodology and visualized pipelines.
2. The dataset construction process is systematic, and the authors demonstrate strong engineering execution.

### Weaknesses
1. The contribution seems more system-integration–oriented than research-driven. Although the implementation is impressive, the framework does not introduce a new learning objective, reasoning algorithm, or theoretical insight. The claimed innovations (such as “Proactive Exploring” or “Aggregation Logic Injection”) could be described more precisely to highlight what differentiates them conceptually from previous pipelines.
2.The prompt design within the agent likely has a substantial influence on the reported results, yet this factor is not thoroughly analyzed. Since prompts often dominate model behavior, the absence of prompt ablations makes it difficult to determine whether the improvements stem from the proposed framework or from more effective prompting strategies.
3. The data generation pipeline appears largely self-contained, with both question synthesis and answer derivation relying on the same model-driven process. This raises concerns about self-reinforcement bias, where models generate and validate their own outputs without external verification or human grounding.
4. The description of the “Complex Aggregation Logic Injection” step is vague and lacks operational details. It seems to rely on ad hoc prompt mappings rather than a formalized or reproducible algorithmic process.
5. Since the web environment is inherently dynamic, the “Proactive Online Web Exploring” process may yield non-deterministic outputs across runs. The paper does not describe any environment control or caching strategy, which undermines reproducibility.
6. Because the model is trained and evaluated on data derived from the same generation procedure, there is a risk of target leakage or self-consistency bias. Improvements may reflect distributional alignment rather than genuine reasoning gains.

### Questions
weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the purported critical limitation that existing web agents neglect complex information aggregation, focusing excessively on simple retrieval. The authors introduce an automated, scalable methodology to synthesize WebAggregatorQA, a training resource grounded in real web environments using Proactive Online Web Exploring and Complex Aggregation Logic Injection (12+ logical operations). Models fine-tuned on this data, WebAggregator, demonstrate improved performance, matching or surpassing models like GPT-4.1 on both the GAIA-text subset and the newly constructed, specialized WebAggregatorQA test set.

### Strengths
The primary strength lies in identifying and explicitly targeting the under-explored necessity of information aggregation in deep research agents. The paper delivers a robust, scalable data construction paradigm that mandates complex aggregation logic and utilizes heterogeneous, dynamic web sources, including files. Furthermore, the resulting WebAggregatorQA test set is shown to be highly challenging; even state-of-the-art zero-shot agents like GPT-4.1 (25.8%) and Claude-3.7-sonnet (28.3%) struggle significantly, effectively setting a difficult new benchmark and validating the importance of this aggregation focus.

### Weaknesses
The discussion on solutions for failure cases was not convincing to me. While it is beneficial to evaluate the limitations of the model's capabilities, the fundamental problem with GUI Agent evaluation is the lack of a unified paradigm, which results in highly noisy outcomes. For instance, could we not use a GUI Agent specialized only in interacting with the browser to acquire specific information for environment interaction and data collection, and then train a separate model dedicated to planning and information aggregation/reasoning? Given such possibilities, the training paradigm and experiments presented in this paper might become less persuasive. This is not a flaw of this specific paper, but rather reflects the difficulty in defining truly valuable problems within the agent field itself. What I was hoping to see was a truly detailed discussion on failure cases, including how to solve these problems under various paradigms, but this is lacking. Consequently, the insights provided by a portion of the paper (the training section) are not substantial enough, and in my opinion, this paper is therefore not suitable for inclusion in ICLR.

### Questions
1. What is the true motivation for conducting the model training experiments? What do you intend to verify? Have you considered the possibility of other training paradigms?
2. What are your next steps or ideas for addressing the failure cases?
3. WebAggregator uses Pass@1 for evaluation. What are your thoughts on finer-grained evaluation methods? What would be the benefits? What problems would they introduce? Why did you initially choose not to include a finer-grained evaluation?
4. WebAggregator explicitly excluded tools like Screenshot and Scroll from its training trajectory sampling. What was the rationale behind this decision?

### Soundness
2

### Presentation
3

### Contribution
2
