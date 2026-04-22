# LLMSynthor: Macro-Aligned Micro-Records Synthesis with Large Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Macro-aligned micro-records are essential for simulations in social science and urban studies. For instance, epidemic models of urban disease spread are only credible when micro-level records reproduce realistic individual mobility and contact patterns, while macro-level aggregates match real-world statistics such as case counts or travel flows. Still, large-scale collection of such fine-grained data is impractical, leaving researchers with only macro-statistics (e.g., travel surveys or case counts). Large Language Models (LLMs), leveraging rich real-world priors learned from vast corpora, excel at generating realistic micro-records, but standard record-by-record sampling is inefficient and fails to enforce alignment with target macro-statistics. Given this, we propose LLMSynthor, a framework capable of synthesizing realistic micro-records that are statistically aligned with target macro-statistics. LLMSynthor transforms a pre-trained LLM into a macro-aware simulator that incrementally builds a synthetic dataset through an iterative process. At each iteration, a batch of micro-records is generated to reduce the discrepancy between synthetic and target macro-statistics. By treating the LLM as a nonparametric copula for inferring joint dependencies over variable combinations, the iterative process ensures the synthetic data are macro-statistically aligned with the target marginals and joints. To address sampling inefficiency, we introduce LLM Proposal Sampling, where the LLM, guided by discrepancies, generates a plan of proposals, each defining specific values or ranges for all variables and specifying the number of records to generate. This enables the framework to minimize discrepancies efficiently while preserving the realism grounded in the LLM’s priors. Evaluations on synthetic and real-world datasets (mobility, e-commerce, population) encompassing diverse formats and settings show that LLMSynthor achieves high record realism, statistical fidelity, and practical utility, positioning it broadly applicable across economics, social science, urban studies, and beyond.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work aims to generate realistic individual-level records that follow pre-specified population-level statistics. Individual-level records are much harder to collect than population-level statistics, but are useful to expert practitioners in domains like public health or urban planning. The ability to simulate realistic data at the micro-level while following macro-level statistics would help bridge this gap.

The authors propose to tackle this problem by using an LLM to guide the generation of individual-level records while steering the generation based on the discrepancy with high-level statistics. To improve generation efficiency, they implement “generation plans” where the LLM doesn't generate records directly but instead creates joint distributions that can be sampled from efficiently.

The problem is interesting and well motivated, but the paper suffers from several flaws in its method, experimental validation and presentation.

### Strengths
- The problem of micro-level generation following macro-level statistics is interesting
- Efforts to improve and evaluate the practical downstream utility in the method (producing generation plans for faster records generation) and experimental results (prediction of derived variables using synthesised data) are appreciated.

### Weaknesses
Major:

- W1: the method relies heavily on the LLM for variable dependency inference and micro-level generation plans synthesis. This leads to several concerns in terms of realism, bias and auditability, detailed below.
    - a. the realism of the generated records hinges on the LLM’s ability, but LLMs are notoriously prone to hallucinations. Furthermore, matching the macro-level statistics in aggregate does not guarantee data realism at the low-level. The realism of the generated low-level records should be thoroughly demonstrated experimentally, which is not the case currently (cf W4).
    - b. LLM generation might introduce bias at both at the variable dependency inference stage (ie the model might group spuriously correlated variables) and micro-level generation stage levels (ie the actual joint distribution between selected variables). This is partly noted by the authors in the discussion (l.459-460), but should be acknowledged and discussed beyond the misalignment with target statistics.
    - c. Auditability —ie providing practitioners information to understand how the micro-records were generated— is essential to mitigate the realism and bias pitfalls previously mentioned. This is not possible here since the micro-records are generated directly by the LLM (via the generation plans), which operates as a black-box.
- W2: In Section 3.2, the authors state that “If an inferred dependency lacks corresponding data, practitioners are encouraged to collect additional statistics, if unavailable, the LLM can approximate the joint distribution using existing macro-statistics” (l.215). If I understand correctly this means that if the LLM proposes a variable subset that is not exactly available in the macro-statistics S_target, then the method doesn’t work unless the practitioners collect additional statistics. This is a major limitation for the practical applicability of the method. Meanwhile, the claim that LLM can approximate the joint distribution itself (l.215-216) is unsubstantiated.
- W3: The discrepancy-guided iterative synthesis scheme (Section 3.3) means that generated low-level records are not i.i.d, since records from time t+1 depend on previously generated records at times ≤t (via the discrepancy signals \Delta^t). I would expect that i.i.d data is a common assumption in low-level records —as noted by the authors l.464.
- W4: A thorough evaluation of the realism of generated records is missing from the experimental results. Currently this is only briefly mentioned in Appendix C.3 and limited to illustrative examples. As mentioned in W1, the realism of the generated records is by no means guaranteed by the method and should therefore be extensively assessed experimentally. One way to evaluate this could be to use a dataset where ground-truth micro-level records are available, run the method based only on the macro-level statistics, and measure the similarity between ground-truth and generated low-level records distributions at various granularity levels (not just at the macro-level).
- W5: In Section 4.2, the comparison to low-level generation baselines is unfair since only LLMSynthor has access to the macro-level statistics. A fairer comparison with these baselines would be in terms of the realism of the generated low-level records (cf W4). Fairer baselines for macro-level statistics alignment include direct prompting of the LLM (a form of ablation experiment on the proposed method, eg without discrepancy-guided iterative sampling), and simulator-based methods (eg [1]).
- W6: Figure 1 and 2 are illegible. I had a lot of trouble following them even after getting a good grasp of the method from the text. I would recommend updating these figures to make them less cluttered and improving clarity.
- W7: the related work section is missing simulator-based methods, such as [1], which are highly relevant to this problem. 

Minor:
- The two definitions in Section 3 are informal. They could be made sharper by separating the core definition to discussion points around it.
- Typos: problem in sentence l.200; missing period l.409

[1]: Holt et al, “G-sim: Generative simulations with large language models and gradient-free calibration.” (ICML’2025)

### Questions
- l.216: “The aggregation operator is then updated..” - what does this update look like in practice?
- What if variables in one of the c_k corresponds to a subset of a joint statistics available in the macro-records (eg macro records contain joint statistics over variables (x,y,z), but one c_k=(y,z))? Can we marginalise over the non-included variables?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper presents LLMSYNTHOR, a framework for generating micro-level synthetic records that align with given macro-level statistics. It repurposes a pre-trained LLM as a macro-aware simulator that iteratively refines data generation through discrepancy feedback. The approach combines (i) LLM-based inference of variable dependencies and (ii) proposal-level sampling to improve efficiency and alignment. The paper evaluates the framework on three domains—mobility, e-commerce, and population synthesis—covering both structured and unstructured data. The authors claim the framework improves both realism and efficiency compared to standard LLM sampling, and demonstrate qualitative controllability through proposal-level generation instead of micro-record generation.

### Strengths
Synthesizing datasets using LLM is a good topic.

### Weaknesses
- The paper lacks a description of important details, such as performance metric computation and the proposal for the micro-records sampling step. In Table 2, it is unclear what the Jsd column means. For other specified metrics, it is unclear how they are computed and why those metrics make sense in this application.

- Figure 11 appears to suggest that LLM-based generation still scales poorly with dataset size, potentially far slower than purely generative baselines like GReaT, which trains once and can then sample nearly 10x times faster.

- The paper claims that LLMSYNTHOR ensures realistic micro-records, yet there is no quantitative or human validation of realism.

- All compared baselines (TVAE, CTGAN, CopulaGAN, GReaT, TabSyn, CP, HMM, NVI) are trained on full micro-record datasets, whereas LLMSYNTHOR only accesses macro statistics. Evaluation is non-comparable: the baselines operate with privileged access to individual-level data.

- Synthesizing micro-records that align with target macro-statistics is conceptually similar to synthesizing samples conditioned on target population characteristics. The paper may benefit from comparing with related work in this direction [1, 2, 3], which also explores LLM-based methods for generating population-level or behaviorally diverse synthetic data.

References: 
- [1] Bui, Ngoc, et al. "Mixture-of-personas language models for population simulation." arXiv preprint arXiv:2504.05019 (2025).
- [2] Choi, Hyeong Kyu, and Yixuan Li. "Picle: Eliciting diverse behaviors from large language models with persona in-context learning." arXiv preprint arXiv:2405.02501 (2024).
- [3] Yu, Yue, et al. "Large language model as attributed training data generator: A tale of diversity and bias." Advances in neural information processing systems 36 (2023): 55734-55784.

### Questions
- Are there any macro-only or aggregate-aware baselines that could serve as fairer comparisons?
- How many LLM invocations are made per dataset? What is the number of micro-records per proposal?
- How does LLMSYNTHOR validate the authenticity of generated records? Are there rule-based or domain constraints to reject implausible samples (e.g., “infant working full-time”)?
- Was any human evaluation or domain-expert assessment performed to verify the realism of the synthetic records or the proposals?
Once an LLM outputs a proposal (e.g., “50 records: origin=Brooklyn, time ∈ [6,9]”), how are the individual attribute values sampled? Uniformly? Using empirical distributions?
- How interpretable and realistic are these proposals in practice? Do domain experts consider them meaningful?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a simulator that uses a pre-trained LLM for synthesizing tabular data that match given macro-statistics. They treat the LLM as a non-parametric copula to infer variable dependencies and decide which joints to control and use discrepancy-guided iterative synthesis with LLM Proposal Sampling to reduce the macro-level gaps between current synthetic aggregates and targets. Experiments on mobility, e-commerce (tabular), and population synthesis show good macro alignment and downstream utility; the paper also reports strong performance versus tabular baselines.

### Strengths
- The paper is very well written with illustrative figures.
- Synthetic data generation with LLMs is an important and timely problem.
- Method design feels practical. The loop is simple and likely easy to implement.
- Some limitations of LLMs are clearly discussed.

### Weaknesses
- Evaluation alignment vs. leakage. In section 4.2, it is not fully clear which macro-stats are used as inputs to LLMSynthor. Are they the same as evaluation metrics? If they coincide, the method could look stronger than baselines because it is explicitly optimizing those statistics, whereas baselines also model micro-level realism beyond the chosen stats. 
- Baselines breadth & recency. The population-synthesis baselines (CP/HMM/NVI) appear to be outdated relative to modern diffusion/tabular-LLM generators. Similarly, some tabular synthesis, TabSyn and GReaT are included, but they seems old as well. How's your method compared to recent LLM-based tabular generator,s such as - Yang et al Doubling Your Data in Minutes: Ultra-fast Tabular Data Generation via LLM-Induced Dependency Graphs. or Long et al. LLM-TabLogic: Preserving Inter-Column Logical Relationships in Synthetic Tabular Data via Prompt-Guided Latent Diffusion.

### Questions
- (Line ~140) Variable types. You define micro-records over variables (discrete or continuous). Why is the taxonomy restricted to these two? Can your method be extended to synthesizing other dataset formats with texts, for example, since you're using LLMs?
-  Some citations are needed for baselines in population synthesis experiments. 
- Why synthetic e-commerce? Could you replicate section 4.2 on a public retail/marketplace dataset (e.g., an Amazon open dataset) to demonstrate performance with real macro targets and natural noise? What prevents doing this today?
- Can you please report the quantitative evaluation of the realism of micro-records for the experiment in section 4.2?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a new framework for synthesizing realistic micro-records (individual-level data) to align with the given macro-statistics (aggregate-level constraints). The method iteratively generates batches of synthetic micro-records, where each iteration measures the discrepancy between current synthetic macro-statistics and target aggregates. They empirically validate their method in three different applications (urban mobility, e-commerce transactions, population synthesis).

### Strengths
- This paper addresses a practical challenge in data synthesis: generating individual-level records when only aggregate statistics are available.
- The experimental setup on three different applications demonstrate broad applicability and outperforms the considered baselines.

### Weaknesses
- How do you prevent "over-correction", where repeated discrepancy-guided sampling causes loss of diversity?
- Method limitations. In real world applications, macro-statistics might be noisy or incomplete, can the method incorporate uncertainty over the target aggregates?
- Limited empirical evaluation. How sensitive is the framework to LLM quality?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
