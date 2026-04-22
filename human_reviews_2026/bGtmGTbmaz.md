# Eigen-Agent: Adaptive Multi-Agent Scientific Reasoning with Monitor-Based RAG

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Large language models (LLMs) have recently shown strong progress on scientific reasoning, yet two major bottlenecks remain. First, explicit retrieval fragments reasoning, imposing a hidden tool tax of extra tokens and steps. Second, multi-agent pipelines often dilute strong solutions by averaging across all candidates. We address these challenges with a unified framework that combines implicit retrieval and structured collaboration. At its foundation, a Monitor-based retrieval module operates at the token level, integrating external knowledge with minimal disruption to reasoning. On top of this substrate, Hierarchical Solution Refinement (HSR) iteratively designates each candidate as an anchor to be repaired by its peers, while Quality-Aware Iterative Reasoning (QAIR) adapts refinement to solution quality. On Humanity’s Last Exam (HLE) Bio/Chem Gold, our framework achieves 48.3% accuracy—the highest reported to date, surpassing the strongest agent baseline by 13.4 points and leading frontier LLMs by up to 18.1 points, while simultaneously reducing token usage by 53.5% and agent steps by 43.7%. Results on SuperGPQA and TRQA confirm robustness across domains. Error analysis shows that reasoning failures and knowledge gaps co-occur in over 85% of cases, while diversity analysis reveals a clear dichotomy: retrieval tasks benefit from solution variety, whereas reasoning tasks favor consensus. Together, these findings demonstrate how implicit augmentation and structured refinement overcome the inefficiencies of explicit tool use and uniform aggregation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents EIGEN-1, a multi-agent framework for scientific reasoning that integrates three modules:
(1) Monitor-based RAG, an implicit retrieval mechanism that adaptively augments reasoning without explicit tool calls;
(2) Hierarchical Solution Refinement (HSR), which coordinates agents through an anchor–reference structure for iterative correction; and
(3) Quality-Aware Iterative Reasoning (QAIR), a quality-gated refinement loop.
The framework is evaluated on HLE Bio/Chem, SuperGPQA, and TRQA datasets, showing clear gains over strong LLM baselines.
The paper is clearly written, well-structured, and the empirical results are strong.

### Strengths
1.Comprehensive System Design:
The integration of retrieval, multi-agent collaboration, and quality-aware refinement forms a coherent and practically useful system.The proposed orchestration shows engineering sophistication and appears well implemented.

2.Strong Empirical Results:
Across multiple benchmarks, EIGEN-1 demonstrates consistent improvements in accuracy and efficiency.The ablation analyses (e.g., removing Monitor or QAIR) are detailed and convincing.

3.Deployment Practicality:
Since the framework operates over frozen LLMs without training, it can be directly applied to existing models, which increases its practical relevance.

### Weaknesses
Domain-specific RAG corpus and evaluation validity

The paper’s most significant issue lies in how the retrieval corpus was constructed and evaluated.The authors build a biology/chemistry-specific RAG database from domain papers (Appendix A.1) and test primarily on HLE Bio/Chem and SuperGPQA Bio datasets.
However, the HLE benchmark itself explicitly states that:
“Each question has a known solution that is unambiguous and easily verifiable, but cannot be quickly answered via internet retrieval.”

This creates a conceptual inconsistency: the proposed system demonstrates large gains precisely on a dataset that was designed to minimize the benefits of retrieval. Even if the database does not contain direct answers, its domain overlap likely gives the model privileged access to related context or background information. Thus, the reported improvements may reflect content alignment rather than genuine reasoning enhancement.

I strongly recommend the authors to: Report results using a generic, mixed-domain corpus (e.g., Wikipedia or full arXiv); Or evaluate cross-domain performance, e.g., using the Bio/Chem corpus for non-scientific HLE subsets. These analyses would clarify whether EIGEN-1’s gains stem from its mechanism or its domain priors.

### Questions
Refer to the weaknesses.

### Soundness
3

### Presentation
3

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
This paper presents EIGEN-1, a novel framework for scientific reasoning that unifies implicit retrieval with structured multi-agent refinement. The authors aim to address two major challenges in retrieval-augmented reasoning systems: the fragmentation of reasoning caused by explicit retrieval—referred to as the “tool tax”—and the dilution of strong candidate solutions through democratic multi-agent aggregation.

On the Humanity’s Last Exam (HLE) Bio/Chem Gold benchmark, EIGEN-1 achieves 48.3% accuracy, outperforming the strongest baseline model, SciMaster, by 13.4 points while reducing token usage by 53.5% and reasoning steps by 43.7%. Further evaluations on SuperGPQA and TRQA confirm its robustness. The paper also provides insightful analyses, including a quantitative study of the tool tax and the diversity–consensus trade-off in retrieval versus reasoning tasks.

### Strengths
1. The integration of implicit retrieval and hierarchical refinement is novel, bridging retrieval efficiency and collaborative reasoning in a unified system.

2. The modular design (Monitor-Querier-Injector-HSR-QAIR) is well-justified and empirically validated through controlled ablations and analyses.

3. The paper is clearly organized with intuitive visuals and detailed appendices (e.g., Algorithm 1).

4. The quantitative treatment of tool tax offers a new analytical lens for reasoning system efficiency, which can influence future research directions.

5. Strong benchmarks, token-efficiency metrics, and error-type breakdowns demonstrate robustness and depth of evaluation.

### Weaknesses
1. Lack of prompt sensitivity analysis.
Although the paper provides detailed templates for each component (Monitor, Querier, Injector, HSR, QAIR), it does not evaluate how sensitive the framework’s performance is to prompt wording, structure, or length. Since EIGEN-1 heavily relies on prompt-driven modular coordination, empirical evidence on prompt robustness would strengthen the paper’s claims of generality.

2. Limited exploration of base-model dependence.
All experiments are conducted on DeepSeek-V3.1 as the underlying model. The authors claim that Monitor-based RAG is “model-agnostic,” but they do not test the framework on other LLMs such as GPT-4.1 or Claude Opus. It remains unclear how performance, retrieval triggering frequency, or refinement behavior change when the base model’s reasoning or linguistic style differs.

3. Potential evaluator bias.
The QAIR and automatic judging components rely on LLM-based evaluators (e.g., o3-mini). Without cross-model validation, the risk of evaluator–solver coupling remains unquantified.

4. Domain focus.
The benchmarks are concentrated in biology and chemistry. Additional experiments in other reasoning domains (e.g., physics, math) would better establish the framework’s general applicability.

### Questions
1. How sensitive is EIGEN-1 to prompt variations? Have you observed significant degradation when the prompts for the Monitor, Querier, or QAIR modules are paraphrased or shortened?

2. Since the method is described as model-agnostic, could you report or estimate its performance when using a different base model (e.g., GPT-4, Claude 3, or Gemini 2.5)? Does the Monitor module’s uncertainty detection generalize across architectures?

3. QAIR and the o3-mini judge play central roles in assessment. How robust are the results if these evaluators are replaced by another LLM or a human scoring subset?

4. Do you expect similar gains in domains that require abstract reasoning but less factual grounding (e.g., logic puzzles, programming tasks)?

### Soundness
3

### Presentation
4

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
This paper presents EIGEN-1, a new multi-agent system for scientific reasoning. It tries to fix two big problems: 1) RAG calls breaking the flow of thought (the "tool tax") and 2) multi-agent groups getting bogged down by bad ideas. The main ideas are an "implicit" RAG that injects knowledge when the model seems confused and a "hierarchical" refinement process where solutions are repaired by peers instead of just averaged. The system gets SOTA results on HLE Bio/Chem Gold, hitting 48.3% and beating strong models by a lot , all while using way fewer tokens.

### Strengths
The paper offers clever fixes for real problems in agent reasoning. The "Monitor-based RAG" is a great idea; it avoids stopping the reasoning process just to look something up, which clearly saves a lot of compute (53.5% token reduction ). The hierarchical refinement (HSR/QAIR) is also a strong point. Making agents repair each other's work instead of just voting makes a lot of sense and seems to work well based on the ablations. The SOTA results on a tough benchmark like HLE are impressive. I also liked the extra analyses, like the component breakdowns and the look at diversity vs. consensus, which back up the main claims.

### Weaknesses
My main concern is that this framework is very complex. There are a lot of moving parts (Monitor, Querier, Injector, HSR, QAIR, etc. ) and it's not obvious how sensitive it is to tuning. The ablation justifies each part, but it's hard to tell how they interact. For instance, does HSR only work well because the Monitor-RAG gives it better solutions to start with, or are those two gains separate? Also, the error analysis is interesting, but the appendix case studies mostly show HSR/QAIR failing to fix the problems. It would be good to see a more direct link between the new components and the specific errors they claim to fix.

### Questions
Your system has a lot of parts. Can you say more about how the Monitor-RAG and the HSR/QAIR parts work together? Does the implicit RAG just make the initial solutions better for HSR, or are the improvements from these two modules mostly separate?

You show that reasoning errors are the biggest problem (92.78% of failures ). The Monitor-RAG seems to help with knowledge errors, but how does your system fix pure logical mistakes?

### Soundness
3

### Presentation
3

### Contribution
2
