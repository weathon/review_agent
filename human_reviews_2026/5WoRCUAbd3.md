# Advancing and Benchmarking Personalized Tool Invocation for LLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Tool invocation is a crucial mechanism for extending the capabilities of Large Language Models (LLMs) and has recently garnered significant attention. It enables LLMs to solve complex problems through tool calls while accessing up-to-date world knowledge. However, existing work primarily focuses on the fundamental ability of LLMs to invoke tools for problem-solving, without considering personalized constraints in tool invocation. In this work, we introduce the concept of Personalized Tool Invocation and define two key tasks: Tool Personalization and Parameter Personalization. Tool Personalization addresses user preferences when selecting among functionally similar tools, while Parameter Personalization considers cases where a user query lacks certain tool parameters, requiring the model to infer them from the user profile. To tackle these challenges, we propose PTool, a data synthesis framework designed for personalized tool invocation. Additionally, we construct PTBench, the first benchmark to evaluate personalized tool invocation. We then fine-tune various open-source models, demonstrating the effectiveness of our framework and providing valuable insights. Our model, training data, and the benchmark will be publicly released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the concept of personalized tool invocation, highlighting the importance of personalization in large language models (LLMs) when performing tool-based reasoning and decision-making. The authors propose PTool, a data synthesis framework that constructs user profiles and simulates behavior to generate high-quality personalized tool-invocation data. Using this framework, they further develop PTBench, a benchmark designed to evaluate personalized tool invocation abilities. They fine-tune open-source models on their synthesized dataset and report improvements in tool preference accuracy and parameter inference without sacrificing general capabilities.

### Strengths
1. Good direction. Personalization is indeed an important yet underexplored aspect of LLM-based agents. 
2. Framework design. The proposed PTool pipeline (tool generation, profile construction, and query-solution synthesis) is clearly structured and systematically implemented.

### Weaknesses
1.	Overclaim of novelty. The paper claims to be the first benchmark for evaluating personalized tool invocation. However, related works such as PEToolLLM (2025.2) [1] have already explored personalized tool use. Moreover, some prevalent benchmarks also include personalization aspects, such as ACEBench [3].
2.	Lack of related work discussion and comparison. The paper omits a thorough comparison with existing benchmarks on personalized tool use [1–3]. For a benchmark paper, it is crucial to clearly articulate how this work differs from prior studies in terms of data structure, personalization mechanisms, and dataset construction methods.
3.	Limited generalization analysis. Since the paper involves model fine-tuning, it is necessary to evaluate generalization across other personalized tool use datasets [1–3]. Otherwise, it cannot be determined whether the proposed model works only for the specific personalization forms seen in PTBench.
4. Disorganized metrics. The authors introduce a large number of fine-grained metrics, but it is unclear whether all are necessary. Some metrics appear overlapping—for instance, Param Value and T-value. Besides, the authors should clarify which metrics are related to personalization. Too many metrics without proper categorization and justification may reduce the precision and interpretability of the evaluation.
5. Dataset release. As this paper proposes a benchmark, it is expected that both the training and test datasets be publicly released to facilitate better understanding of the work.

[1] Q. Xu et al, PEToolLLM: Towards Personalized Tool Learning in Large Language Models

[2] Z. Cheng et al, ToolSpectrum : Towards Personalized Tool Utilization for Large Language Models

[3] C. Chen et al, ACEBench: Who Wins the Match Point in Tool Usage?

### Questions
In Figure 6, are the models trained solely on PTBench, or do they also incorporate other general-purpose datasets? In prior works such as ToolGen [1], fine-tuning on tool-related data often results in a noticeable degradation in general-domain performance. However, this paper reports almost no such decline. Could the authors clarify what specific characteristics of the proposed dataset or training methodology help prevent this degradation? Moreover, benchmarks that exhibit significant degradation in [1], such as ARC, are also expected to be reported for a more comprehensive comparison.

[1] R. Wang et al, ToolGen: Unified Tool Retrieval and Calling via Generation.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces personalized tool invocation for LLMs with two sub-tasks: tool personalization and parameter personalization. It proposes PTool, a three-stage data-synthesis pipeline and builds PTBench. Fine-tuning an OSS model improves platform preference selection and profile-dependent parameter filling; ablations show user history aids tool preference while basic attributes aid parameter completion.

### Strengths
Personalized tool use is a pivotal challenge for practical LLM deployment. This paper frames the problem via tool personalization and parameter personalization, proposes a data-generation pipeline accordingly, and substantiates its design with targeted ablation studies.

### Weaknesses
1. The first benchmark to evaluate personalized tool invocation claim likely overstates. Contemporary works [1] [2] already target personalized tool use.
2. The scale and diversity of the dataset may be insufficient. With 80 users and 3 platforms per scenario, it’s unclear whether user profile coverage is broad enough and whether platform differences are strongly identifiable across scenarios.
3. The benchmark hides profiles at query time but expects profile-based completion. Add adversarial pairs that contrast parameters guessable by commonsense defaults vs. those only recoverable from profiles, and report discrimination performance to ensure genuine profile use.

[1] Evaluating Personalized Tool-Augmented LLMs from the Perspectives of Personalization and Proactivity.

[2]  ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models.

### Questions
How does this work compare against other personalized tool benchmarks.

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
2

### Summary
This paper introduces the concept of Personalized Tool Invocation, addressing two key challenges: (1) Tool Personalization – selecting among functionally similar tools based on user preferences, and (2) Parameter Personalization – inferring missing tool parameters from the user profile. The authors propose an automated data synthesis framework called PTool, consisting of tool generation, user profile construction, and behavior simulation, and build PTBench, the first benchmark with 1,199 annotated samples for this task. Experiments show that fine-tuning open-source models on the synthesized data significantly enhances personalized tool invocation capabilities without harming general performance

### Strengths
1. This paper proposes a paradigm for personalized tool invocation, which is an important research topic.

2. A complete pipeline for data synthesis with rule-based + LLM verification and human-in-the-loop curation yields a usable benchmark.

3. Experiments with both API and OSS models show the effectiveness of the proposed PTool method.

### Weaknesses
1. All data are synthetic (seeded by GPT-4-turbo) with limited manual verification scale (1,199 test items), leaving open how results transfer to real-wolrd scenarios.

2. While many baselines are reported, there is no head-to-head against other personalization methods or retrieval-augmented profile resolvers, and no BFCL evaluation.

3. The code and experimental data are not open-sourced.

### Questions
1. The unseen-scenario study (116 samples) is promising but small. Could you scale to more domains and report per-domain confidence intervals?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the novel task of Personalized Tool Invocation, which encompasses two core sub-tasks: Tool Personalization and Parameter Personalization. The authors develop an automated data synthesis framework, PTool, and construct the first benchmark, PTBench, to evaluate Large Language Models' capabilities in personalized tool invocation. Through fine-tuning open-source models on a large-scale synthesized dataset, experiments demonstrate that their approach significantly enhances model performance on personalized tool invocation tasks without compromising general capabilities.

### Strengths
1.	Clearly distinguishing the two key challenges of tool preference and parameter inference, thereby filling a gap in the existing tool learning literature regarding personalization.
2.	Proposes a systematic data synthesis framework (PTool) that covers tool generation, user profile construction, behavior simulation, and query-solution generation. The pipeline is clear and extensible.
3.	Constructs the first benchmark for personalized tool invocation (PTBench), comprising high-quality, human-annotated data and a multi-dimensional evaluation, providing a crucial foundation for future research.

### Weaknesses
1.While PTBench contains 1,199 test samples, the training data is primarily synthetic and does not incorporate real user behavior data, which might affect the model's generalization in real-world scenarios.
2.	Evaluation metrics rely heavily on automated measures, lacking subjective assessment of the model's reasoning process or user satisfaction, making it difficult to fully reflect the "human-like" quality of personalized invocations.
3.	The description of the user profile construction process is somewhat brief, particularly regarding how "psychological traits" and "behavioral tendencies" are concretely transformed into operational features, which may hinder reproducibility for readers.

### Questions
1.	When constructing user profiles, how did you balance the influence of "basic features" versus "behavioral history"? Is there quantitative analysis identifying which features are most critical for tool preference and parameter inference?
2.	Have you considered extending the PTool framework to more diverse or cross-domain scenarios (e.g., healthcare, finance)? What new challenges might personalized tool invocation face in these domains?

### Soundness
3

### Presentation
3

### Contribution
3
