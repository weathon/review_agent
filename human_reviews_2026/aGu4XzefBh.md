# D-CORE: Incentivizing Task Decomposition in Large Reasoning Models for Complex Tool Use

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 8, 4

## Abstract
Effective tool use and reasoning are essential capabilities for large reasoning models (LRMs) to address complex real-world problems. Through empirical analysis, we identify a prevalent "Lazy Reasoning" phenomenon, where LRMs frequently engage in repetitive and meaningless reflective reasoning. This occurs primarily due to their inadequate ability to decompose tasks when reasoning in complex tool use scenarios. To address this, we propose a two-stage training framework D-CORE ($\underline{\textbf{D}}$ecomposing tasks and $\underline{\textbf{Co}}$mposing $\underline{\textbf{Re}}$asoning processes) that first incentivize the LRM’s task decomposition reasoning capability via self-distillation, followed by diversity-aware reinforcement learning (RL) to restore LRM's reflective reasoning capability. D-CORE achieves robust tool-use improvements across diverse benchmarks and model scales. Experiments on BFCLv3 demonstrate superiority of our method: D-CORE-8B reaches 77.7\% accuracy, surpassing the best-performing 8B model by 5.7\%. Meanwhile, D-CORE-14B establishes a new state-of-the-art at 79.3\%, outperforming 70B models despite being 5× smaller.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
he paper identifies a critical failure mode called “Lazy Reasoning” in Large Reasoning Models (LRMs): when faced with complex tool-use or multi-turn reasoning tasks, LRMs tend to loop through meaningless reflective thoughts instead of decomposing tasks into subtasks.

To address this, the authors propose D-CORE, a two-stage training framework:

Self-Distillation Stage – teaches the LRM to perform task decomposition and subtask reasoning by generating and reusing its own reasoning trajectories.

Diversity-Aware RL (DA-GRPO) – reintroduces reflective diversity by adding entropy-based terms into the advantage function to encourage exploration without reverting to unproductive reasoning.

### Strengths
This paper presents a well-motivated and empirically grounded study on addressing the “Lazy Reasoning” phenomenon in large reasoning models (LRMs), offering a novel and systematic framework called D-CORE that combines self-distillation with diversity-aware reinforcement learning. The problem is clearly identified through quantitative and qualitative evidence, showing how excessive reflection hinders multi-turn tool use reasoning. The proposed two-stage method is conceptually coherent—self-distillation promotes task decomposition, while the entropy-shaped advantage in DA-GRPO effectively restores reflective diversity. The paper demonstrates strong empirical results across BFCLv3 and τ-Bench, with significant improvements over both open-source and proprietary baselines, and provides detailed behavioral analysis that connects reasoning structure to performance. The overall experimental design, ablation studies, and large-scale evaluations reflect thoughtful execution and a meaningful step forward in enhancing compositional reasoning for tool-use scenarios.

### Weaknesses
The entropy-based advantage formulation is only intuitively justified, without a formal connection to reasoning diversity or convergence guarantees. Several implementation details—such as the construction of the 40K self-distillation dataset, the specific prompt templates, and hyperparameter settings—are deferred to the appendix or omitted entirely, making the training pipeline hard to replicate. The evaluation scope is relatively narrow, relying only on BFCLv3 and τ-Bench, both focused on structured tool use, leaving questions about cross-domain generalization. Additionally, qualitative comparisons of reasoning trajectories before and after D-CORE are missing, which weakens claims about interpretability. Minor but pervasive grammatical inconsistencies and figure referencing issues also reduce readability and polish. Overall, while the framework is promising, the paper would benefit from deeper theoretical analysis, richer qualitative evidence, and clearer experimental documentation to meet the standards of a top-tier ICLR submission.  Please provide clarifications on the questions asked above.

### Questions
Please address the questions in the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces D-CORE (Decomposing tasks and Composing Reasoning processes), a two-stage training framework designed to enhance large reasoning models (LRMs) in complex multi-turn tool-use scenarios. The authors first identify a phenomenon termed Lazy Reasoning, where LRMs engage in lengthy but unproductive reflections instead of performing effective task decomposition. To address this issue, D-CORE first employs self-distillation to teach the model to decompose complex tasks into subtasks and compose coherent reasoning trajectories, and then applies diversity-aware reinforcement learning (DA-GRPO) to restore reflective and exploratory reasoning behaviors. Experimental results demonstrate clear performance improvements, and the paper includes comprehensive ablation studies to validate the contribution of each proposed component.

### Strengths
- The problem studied in this paper is interesting. The Lazy Reasoning phenomenon is indeed commonly observed when using large language models, making the motivation of this work clear and compelling.

- The paper proposes a novel and well-structured framework. The framework itself, along with the specific techniques employed in each component (e.g., self-distillation, the generation of self-distillation instances, and diversity-aware reinforcement learning), is thoughtfully designed and empirically validated through comprehensive ablation studies. These techniques might influence future research in this area.

### Weaknesses
- The overall writing could be improved. Some parts of the paper are difficult to follow at first reading. For example, the description related to Figure 3(c) is somewhat confusing, as the detailed definitions and procedures are only provided in the appendix. It would improve readability if part of this information were moved to the main text to make the presentation more self-contained.

- In Algorithm 1, some referenced functions are not rigorously defined (e.g., Verify($\hat{Y}, Y*$)). While their meanings can be intuitively understood, a more formal and precise specification would improve clarity. In addition, $\hat{Y}$ and $Y*$ actually represent different entities: $\hat{Y}$ denotes a reasoning trajectory, whereas $Y*$ corresponds only to the ground-truth tool-use information. The current notation may lead readers to mistakenly assume they are of the same type. It would be clearer to use distinct notation (for example, a $\mathcal{\hat{Y}}$ for the trajectory) to better reflect this distinction.

- The potential iterative application of the proposed framework is not discussed in the paper. Since the combination of self-distillation and reinforcement learning could be executed repeatedly, it would be valuable to analyze whether iterative cycles of D-CORE would lead to further improvements or possible instability. In addition, the paper does not discuss a simple yet relevant baseline: explicitly prompting the model to reduce Lazy Reasoning during reasoning. Including such an analysis would help clarify how much of the improvement comes from architectural innovation versus prompt-level guidance.

### Questions
- I noticed that the algorithm may return an empty set in some cases (e.g., line 2 of Algorithm 1). Does this mean that if the LLM performs an incorrect decomposition, the corresponding sample is discarded, and a new one must be generated until a valid trajectory is obtained? Also, is the success rate of this decomposition process the one reported in Table 7? If so, how does this success rate change after SFT/RL? Does the distilled model show a consistent improvement in decomposition success rate compared to the initial one?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces D-CORE, a two stage training framework designed to enhance the complex tool use capabilities of Large Reasoning Models (LRMs). The authors identify a recurring failure mode termed “Lazy Reasoning”, in which models engage in repetitive, low value reflective reasoning rather than decomposing complex tasks effectively.

D-CORE addresses this by first employing self-distillation to teach structured task decomposition, followed by diversity-aware reinforcement learning (DA-GRPO) to restore reflective reasoning diversity. The method achieves strong empirical results, notably 79.3% accuracy on BFCLv3 and state-of-the-art performance on τ-bench, with models up to five times smaller outperforming larger baselines. The work presents an organized empirical study, clear problem framing, and impactful results on benchmarks of practical importance.

The paper is well motivated, clearly presented, and empirically strong, offering valuable insights into the limitations of reasoning in large models and proposing a structured remedy with convincing results. However, the core methodological innovation feels incremental, relying heavily on ground truth assisted data generation and established reinforcement learning heuristics. The contribution is thus more of a careful systemization and validation of known ideas rather than a breakthrough in reasoning architectures.

In summary, this is a solid and impactful piece of empirical work with clear value for the community, but its conceptual depth and theoretical originality fall slightly short of the standard expected for acceptance. With additional evidence of generalization and stronger justification of its mechanisms, this work could become a meaningful contribution in subsequent revisions.

### Strengths
Clear Problem Definition and Motivation:
The paper compellingly identifies and analyzes the “Lazy Reasoning” phenomenon in LRMs, providing behavioral and quantitative evidence of how excessive reflection impedes performance in multi-turn tool-use scenarios. This focus on reasoning efficiency rather than sheer model size is a valuable direction for scaling reasoning systems.

Strong Empirical Validation:
Results are robust and consistent across multiple model sizes and benchmarks. The reported +30% improvement in multi-turn performance and significant gains over ToolRL and Qwen baselines demonstrate that D-CORE achieves real, measurable advances. The use of ablations to isolate the effects of self-distillation and DA-GRPO is systematic and convincing.

Methodical and Transparent Experimental Design:
The paper offers well-structured experiments and diagnostic analyses. The manual decomposition experiment (Fig. 4b) substantiates the hypothesis that explicit decomposition mitigates lazy reasoning. The inclusion of detailed algorithms, prompt templates, and multi-model evaluations enhances credibility and reproducibility.

Clarity and Presentation:
The writing is clear and logically structured, supported by intuitive figures (especially Figures 1 and 5) that convey the framework and reasoning behavior differences. The presentation of results and ablations is thorough without being overwhelming.

Practical Relevance:
The efficiency gains, where a 14B model matches or surpasses 70B baselines, underscore the significance of the contribution for real world applications in reasoning centric AI systems and agentic tool use.

### Weaknesses
Limited Novelty of Core Mechanisms:
The core technical ideas, self distillation using ground truth guided decomposition and entropy-based reinforcement learning, are adaptations of existing concepts rather than fundamental innovations. Stage 1 primarily represents a structured data-generation and fine tuning pipeline, while Stage 2 employs a standard entropy based exploration adjustment. The methodology feels more engineering-driven than theoretically novel.

Dependence on Ground Truth for Decomposition:
A key limitation lies in the reliance on access to ground truth trajectories during the self distillation phase. As shown in Table 7, decomposition success rates drop from 93% to 73% without this supervision. This dependence reduces the framework’s applicability to domains where labeled trajectories are unavailable.

Fragile Reinforcement Stage:
The DA-GRPO component assumes that higher entropy equates to productive reflection, but this connection is empirically delicate. The paper itself notes that overly large entropy coefficients reintroduce unproductive “lazy reasoning,” suggesting that the framework may require careful, task specific tuning to remain stable.

Incomplete Theoretical Justification:
The reasoning behind the entropy advantage link is intuitive but not analytically supported. The framework would benefit from deeper theoretical grounding or stronger empirical validation showing that entropy directly correlates with meaningful reflection rather than randomness.

Limited Generalization Analysis:
Although results on BFCLv3 and τ-bench are strong, it remains unclear how D-CORE performs in out-of-distribution or real world settings where task structures and tools differ significantly. Broader validation or zero shot experiments would strengthen claims of generality.

### Questions
Ground Truth Dependence: How does D-CORE perform in data-sparse or unsupervised settings where ground-truth outputs are unavailable for decomposition and verification?

Entropy–Reflection Relationship: Can the authors empirically demonstrate that the entropy-based advantage term promotes useful reflection rather than random exploration?

Unified Learning Framework: Could a single training objective jointly encourage decomposition and reflection, eliminating the need for a sequential “break-and-fix” two-stage setup?

Generalization: Have the authors tested D-CORE on unseen tool-use domains or tasks beyond BFCLv3 and τ-bench to assess adaptability?

Failure Handling: When decomposition fails (as indicated by the 26–30% failure rate without ground truth), are such cases discarded, or do they contribute to training in some corrective form?

### Soundness
3

### Presentation
3

### Contribution
2
