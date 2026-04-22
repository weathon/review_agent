# MetaFlow: A Meta Approach of Training LLMs into Generalizable Workflow Generators

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Large language models (LLMs) excel across a wide range of tasks, yet their instance-specific solutions often lack the structural consistency needed for reliable deployment. Workflows that encode recurring algorithmic patterns at the task level provide a principled framework, offering robustness across instance variations, interpretable traces for debugging, and reusability across problem instances. However, manually designing such workflows requires significant expertise and effort, limiting their broader application. While automatic workflow generation could address this bottleneck, existing methods either produce instance-specific solutions without learning task-level patterns, or cannot generalize beyond their training configurations. We present MetaFlow, which casts workflow generation as a meta-learning problem: given a task and an operator set, the model learns to compose solution strategies. MetaFlow trains in two stages—supervised fine-tuning on synthetic workflow data, followed by reinforcement learning with verifiable rewards (RLVR) that uses execution feedback across problem instances in the task to improve end-to-end success. The resulting model produces effective workflows for trained tasks and exhibits strong generalization to untrained tasks and novel operator sets. Across benchmarks in question answering, code generation, and mathematical reasoning, MetaFlow achieves performance comparable to state-of-the-art baselines on in-domain tasks with single inference, while demonstrating remarkable zero-shot generalization capabilities on out-of-domain tasks and operator sets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes MetaFlow, a meta-learning training method that enables LLMs to generate task-level workflows for unseen tasks and operators. MetaFlow includes two-stage training, where the first stage uses high-quality expert data (Qwen-Max) for supervised learning fine-tuning, and the second stage uses RLVR, with the former solving the cold-start problem of the latter. In the RLVR stage, the model updates its policy (SFT-finetuned Qwen3-8B) through the GRPO algorithm, using the average test score on the sampled data as the reward. Throught experiments on DROP, MBPP, GSM8K, MATH, MetaFlow demonstrates competitive performance on in-domain tasks.

### Strengths
- **Clear motivation**: Task-level approaches (ADAS, AFlow) require expensive re-optimization for new domains, while instance-level methods (ScoreFlow, FlowReasoner) generate per-query workflows without learning reusable patterns. MetaFlow addresses both via meta-learning for zero-shot generalization.
- **Reasonable writing structure**: The paper has clear structure with well-defined sections. The methodology (SFT + RLVR training, code-based workflow representation) is explained clearly.

### Weaknesses
- **Overclaimed novelty and contributions**: The paper uses standard SFT + GRPO, thus the only contribution is applying these to workflow generation. Just showing that if you train an LLM on selected tasks, it can generalize to similar tasks. This is expected behavior, not a novel insight.
- **Insufficient and unconvincing experiments**: The four questions in Section 5 are poorly addressed. (1) Table 1 formatting doesn't reflect best scores; (2) Missing ablation studies; (3) Claimed "OOD generalization" uses train/test splits from the same datasets (GSM8K, DROP, MBPP, HumanEval), not truly unseen domains; (4) Operator generalization shown only through case studies, lacking systematic statistical analysis.
- **Poor writing details**: Multiple presentation issues affect clarity. (1) Figure 1 has an uninformative caption and appears to be a low-quality raster image instead of a pdf or svg; (2) Code listings in the appendix lack syntax highlighting; (3) Critical details missing: learning rates, batch sizes, hardware, ...; (4) No code availability mentioned, harming reproducibility.

### Questions
- **Meta-learning justification**: MetaFlow trains on multiple tasks and tests on similar ones without gradient-based adaptation. How does this differ from standard multi-task learning? What specific meta-learning principle distinguishes MetaFlow?
- **Performance gap with ScoreFlow**: Table 1 shows MetaFlow (78.8) underperforms ScoreFlow (82.5). Why doesn't MetaFlow achieve at least comparable performance?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes MetaFlow, reformulating workflow generation as a meta-learning problem. Given a task description C and operator set Ops, the model learns to directly synthesize workflows W through single inference, enabling zero-shot generation. MetaFlow employs two-stage training: (1) supervised fine-tuning on 1,300 synthetic workflows using Qwen3-8B; (2) reinforcement learning with verifiable rewards (RLVR) using the GRPO algorithm, training for 137 steps across 4 tasks to optimize expected rewards over the task-operator distribution. Experiments show MetaFlow achieves an average score of 78.8 on in-domain tasks (compared to ScoreFlow's 82.5), and demonstrates zero-shot generalization to the Programmer operator and MATH task.

### Strengths
1. **Clear Problem Formulation**: The paper identifies two fundamental limitations of existing approaches—task-level methods require expensive re-optimization for each new task, while instance-level methods cannot reuse task-level patterns—and proposes meta-learning as an alternative path. The motivation is clearly articulated and the problem setting is well-justified.

2. **Rigorous Formalization**: Section 3 formulates workflow generation as a bi-level optimization problem, explicitly defining the task distribution, fast adaptation mechanism, and meta-optimization objective, establishing clear connections to classic meta-learning literature (MAML, Franceschi et al. 2018).

3. **Informative Qualitative Cases**: The appendix's before-and-after training comparison demonstrates workflow design improvements (e.g., replacing information compression with structured extraction, parallel candidate generation), which, though qualitative, helps understand training effects. Appendix A showcases learning to use the VectorSearch operator on HotpotQA.

### Weaknesses
**1. Insufficient Evidence for Core Claims**

The paper's core value proposition is "zero-shot generalization to unseen tasks and operators," but critical baselines and details are missing:

- **Missing Base Model Baselines**: No performance reported for Qwen3-8B zero-shot/few-shot workflow generation, making it impossible to verify whether the 78.8 score stems from meta-learning training or the base LLM's in-context learning ability. Given complete natural language operator descriptions, the base LLM might achieve similar results.

- **MATH is Not Truly OOD**: Even if SFT excludes MATH samples, MATH cannot be considered truly out-of-distribution. The planner uses Qwen3-8B, which has seen extensive mathematical data during pretraining and possesses strong mathematical capabilities matching MATH dataset difficulty.

- **Novel Operator Generalization Not Proven from Training**: The paper claims MetaFlow can use the unseen Programmer operator during training (Section 5.2.2). While Appendix A.0.1 provides qualitative before-and-after cases, a single example cannot rule out performance differences due to sampling randomness. Quantitative performance experiments are still missing: no reporting of accuracy differences between "trained MetaFlow vs untrained Qwen3-8B" when using the Programmer operator. Given complete natural language operator descriptions, Qwen3-8B itself may already possess the ability to use new operators. Quantitative baselines of the base model across all test tasks are needed to prove this generalization capability comes from meta-learning training rather than the model's inherent instruction-following ability.

**2. Cannot Separate SFT vs RLVR Contributions**

The paper claims the "meta-learning framework" is the core contribution, but lacks a critical ablation: no performance reported for "SFT only (without RLVR)". This is a fundamental attribution problem—unable to quantify RLVR's incremental contribution over SFT, or verify whether zero-shot generalization comes from cross-task meta-optimization or simply supervised learning on 1,300 diverse samples.

**3. Performance Consistently Inferior to ScoreFlow Without Explanation**

MetaFlow (78.8) underperforms ScoreFlow (82.5) across all tasks: DROP (-3.4), MBPP (-7.2), GSM8K (-0.8), MATH (-3.4). Theoretically, meta-learning should achieve stronger generalization through cross-task learning, at least matching on in-domain tasks. The paper claims "comparable" performance to SOTA, but Table 1 shows consistent underperformance. Possible explanations include: insufficient Qwen3-8B capacity, SFT data quality issues, inadequate RLVR training (only 137 steps), or single-task methods' deep optimization advantages—none are discussed. The largest MBPP gap (-7.2) particularly needs explanation, along with whether the performance-generalization tradeoff is acceptable.

**4. Incomplete and Inconsistent Experimental Details**

- **RL Step Count Inconsistency**: Text states GRPO runs for 137 steps, but figures only show up to 100 iterations, with no explanation.

- **Opaque SFT Data**: 1,300 samples generated by Qwen-Max, covering "four tasks" and "a single operator set," but does not specify which four tasks, the Qwen-Max prompts, or how sample quality was verified (human review?). If SFT includes MATH, zero-shot claims fail; if quality is poor, this may explain underperformance vs ScoreFlow.

- **Missing Cost/Latency Reports**: No specific token consumption and runtime provided for training (SFT data creation, RLVR iterations) and inference (candidate generation, validation selection, execution), or comparison with single-task search methods. GRPO requires executing k candidate workflows on N instances per step to compute rewards, but k, N, and average operator calls per workflow-instance pair are unreported.

- **best-of-20 Strategy Consistency**: Paper claims "single inference" avoids search, but evaluation generates 20 candidates and selects the best on 50 validation instances, which is essentially search (though lighter than AFlow's 20-round iterations).


**5. Figure Quality Issues**

The paper's schematic diagrams (e.g., Figure 1, Figure 5) suffer from readability problems: text in figures is too small and blurry, making it difficult to discern key labels and process descriptions. In contrast, statistical plots (e.g., training curves in Figures 2-4) use excessively large fonts that occupy too much space. Additionally, the figure organization is illogical: Figure 2 (SFT loss curve) and Figure 3 (RL running reward) are actually subfigures of Figure 4 (Training curves) and should not be presented as separate peer-level figures, leading to confusing numbering and wasted space. The figure design should be revised to ensure schematic text is clear and legible, statistical plot fonts are appropriately sized, and figure hierarchy is properly organized to improve overall presentation quality.

### Questions
1. **Missing Baselines and Controls**:
   - Please provide zero-shot/few-shot workflow generation baselines for the untrained base model (Qwen3-8B) across all evaluation tasks, to verify whether the 78.8 improvement stems from meta-learning or the base LLM's instruction-following ability.
   - Please provide performance for "SFT only (without RLVR)" and sensitivity analysis across different RL training steps, to quantify RLVR's incremental contribution. Text states GRPO trains for 137 steps while figures show only to 100 iterations—please explain this inconsistency and provide convergence justification.
   - For unseen operators (e.g., Programmer), please report quantitative comparisons of "trained Planner vs untrained Qwen3-8B" to support the core claim of "zero-shot novel operator integration." The qualitative case in the appendix is insufficient to replace systematic comparison.

2. **SFT Data and Transparency**:
   - Clarify the specific composition of the "four tasks" in the SFT stage, whether they align with the RL stage's GSM8K/DROP/MBPP/HumanEval, and their respective sample proportions; specify whether MATH-related samples are included.
   - Provide the prompt template/procedure for Qwen-Max sample generation, and the quality verification method for "1,300 high-quality samples" (human review? selection criteria?). If possible, please release a subset of the SFT data for verification.

3. **Evaluation Protocol and Terminology**:
   - The paper claims "single inference," but evaluation uses best-of-20 (selecting the best candidate on 50 validation instances). Please clarify the relationship between this procedure and "single inference."
   - For fair comparison, please specify whether all methods uniformly use the same candidate selection and validation protocol; and report specific inference costs (candidate count, validation set size, execution call volume).

4. **Performance Gap Analysis**:
   - Table 1 shows MetaFlow underperforms ScoreFlow across all tasks (especially the large MBPP gap). Please provide possible causes and analysis (e.g., planner scale, SFT data distribution, RLVR training budget limitations), and discuss the performance-generalization tradeoff.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes MetaFlow, a meta-learning framework for generalizable workflow generation in LLM-based agents. Instead of generating task-specific or instance-level workflows, MetaFlow learns a meta-policy that maps task and operator descriptions to reusable workflows through a two-stage training pipeline (SFT + RL with verifiable rewards). The paper aims to improve generalization across tasks and operators, pushing toward scalable automation of agentic workflow generation.

### Strengths
1. The paper tackles an important problem: generalizing workflow generation beyond specific tasks or instances, which is timely and relevant for the automation of LLM-based agents.
2. The meta-learning formulation is conceptually elegant and represents a clear shift from instance-level reasoning (as in FlowReasoner) to task-level generalization.
3. The qualitative analysis and case studies (e.g., learning unseen operators like VectorSearch) provide some interpretability and insight into the learned workflows.

### Weaknesses
1. Experimental results are weak: MetaFlow performs worse than several baselines (including AFlow and ScoreFlow) and fails to reach SOTA. Moreover, the omission of FlowReasoner, a direct conceptual baseline, makes the evaluation incomplete.

2. Figures are poorly designed — Figure 1 and Figure 5 use raster graphics, and Figure 4’s oversized fonts distort layout — which undermines the professionalism expected of a top-tier paper.

3. Successfully using a new operator does not convincingly demonstrate true out-of-distribution generalization. The authors need to give more evidence instead of only use the datasets used by aflow and scoreflow.

### Questions
The paper’s main conceptual gap lies in its unclear definition of “meta-learning” and “OOD generalization.” The method essentially performs conditional text generation rather than true meta-level adaptation, and the evidence provided (e.g., new operator usage) does not convincingly demonstrate structural generalization. So the key question is the evidence, authors need more evidence to fill their claim.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents MetaFlow, a meta-learning framework that reformulates workflow generation from task-specific optimization to learning generalizable workflow construction strategies across diverse task-operator combinations. Through a two-stage training paradigm combining supervised fine-tuning and reinforcement learning with verifiable rewards (RLVR), MetaFlow learns to synthesize workflows via a single inference without requiring re-optimization for new domains. While the approach offers a novel perspective on automatic workflow generation, the experimental results demonstrate limited improvements over existing methods, and the presentation and experimental analysis require substantial strengthening.

### Strengths
The paper introduces a novel meta-learning perspective to workflow generation that shifts from instance-level or task-specific optimization to learning cross-domain generalization capabilities, which is conceptually interesting and worth encouraging despite suboptimal performance.

### Weaknesses
W1. The experimental results are underwhelming, with MetaFlow achieving an average score of 78.8 compared to ScoreFlow's 82.5, showing almost no state-of-the-art performance on any benchmark and raising questions about the practical value of the proposed approach.

W2. The presentation quality needs significant improvement: Figure 1 merely lists text without visual details (consider FlowReasoner's illustration style); Figure 5's operator visualization appears unnecessary; and the overall figure quality lacks the polish expected for a top-tier venue.

W3. The experimental section is inadequately developed, dedicating excessive space to configuration details while providing minimal analysis, ablation studies, or cost comparisons, which undermines confidence in the thoroughness of the empirical validation.

W4. Several claims are overstated or insufficiently supported: the "OOD generalization" in Section 5.2.2 is questionable since MATH shares characteristics with GSM8K (both mathematical reasoning), and the novel operator integration (Decompose, Programmer) represents a common capability in prior automatic workflow literature rather than a unique contribution.

### Questions
Could you provide more detailed ablation studies showing the individual contributions of SFT vs. RLVR stages, and include a comprehensive cost analysis comparing the computational overhead of your single-inference approach against iterative methods like ScoreFlow and AFlow across different scales?

### Soundness
3

### Presentation
3

### Contribution
3
