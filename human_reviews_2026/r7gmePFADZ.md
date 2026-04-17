# SmartDS-Solver: Agentic AI for Vertical Domain Problem Solving in Data Science

- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Automating complex, multi-step vertical domain tasks—such as Data Science (DS) workflows—presents significant challenges for large language model (LLM) agents. Existing AutoDS approaches often rely on prompt-sensitive, fragmented multi-turn interactions and costly full re-generation upon execution failure, leading to unstable workflow coherence and high token consumption. We introduce SmartDS-Solver, a reasoning-centric agentic system designed to enhance the stability, robustness, and cost efficiency of these workflows. Our core approach integrates rigorous workflow planning into a domain-specialized Reasoning LLM, which is trained using structured methodological distillation and a two-stage Group Relative Policy Optimization (GRPO) procedure. Crucially, SmartDS-Solver employs a lightweight agentic layer featuring the novel State-Aware Refinement and Temperature Exploration (SARTE) algorithm. SARTE dynamically adjusts the LLM’s decoding strategy based on deterministic execution feedback, enabling minimally invasive patching rather than costly full re-planning. We performed a comprehensive evaluation across 32 datasets covering 11 MLE-Bench tasks, 18 AutoML-Agent benchmarks, and 3 real-world tasks, showing consistent gains while reducing inference and modification token usage. In the MLE-Bench benchmark, our 32B model attains an 81.8% win rate over the AIDE+o1-preview baseline, and on the 18 AutoML-Agent tasks, the win rate reaches 94%. Notably, even a 7B model produces fully executable solutions on all evaluated tasks, demonstrating the scalability and robustness of our method. SmartDS-Solver reduces token usage by approximately 78% on the 11 MLE-Bench tasks. The SARTE meta-control mechanism significantly boosts decoding performance—raising average accuracy by 3.9%, lowering error rates by 12%, and delivering an overall 75% significant improvement on MLE-Bench tasks (p = 0.0173).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to improve the capabilities of automated data science for LLM agents. Specifically, this paper proposes SmartDS-Solver, which consists of (1) domain-specific finetuning; two-stage GRPO finetuning; (3) Hierarchical agent framework. Extensive experiments on 11 MLE-Bench tasks demonstrate the effectiveness of the proposed SmartDS-Solver.

### Strengths
- The investigated research problem is interesting and of significance.

### Weaknesses
- The writing is poor. It is hard for me to follow. I suggest the authors carefully revising the paper to meet the basic bar of academic writing. Also, Introduction is important. The current manuscript is lack of this part. The Background and Motivation section of this paper is plain and ambiguous. 

- How to compose the data used for finetuning? What does the data look like? What loss function is utilized for finetuning. For the RLFT, what is the interactive environment? I cannot figure out the techniques in this paper.

- Lack of comparison of SOTA data science agents, such as [1].

[1] MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement, NeurIPS 2025.

### Questions
I think this paper is poor in writing quality. The revision for this paper should undergo a new round of review.

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
3

### Summary
This paper introduces a hierarchical multi-agent framework (called SmartDS-Solver) , which efficiently automates data science workflows. SmartDS-Solver proposes a specialized reasoning LLM and a task-decoupled agent architecture to tackle with three key challenges: fragile task coherence, excessive reliance on prompt-based interactions, and a tendency toward task silos. Experiments on 11 MLE-Bench tasks show an 81.8% win rate over baselines while reducing overhead.

### Strengths
• Developing LLM-driven agents for automating end-to-end data science pipelines is interesting and can augment human analysts.

### Weaknesses
• The proposed method is complicated, which involves a multi-stage pipeline with multiple complex components. This paper lacks sufficient implementation details to reimplement the SmartDS-Solver. 

• The proposed SmartDS-Solver architecture is incremental, which combines multiple existing techniques (e.g., meta-learning, data augmentation, SFT, RL with GRPO).

• As the SmartDS-Solver architecture consists of multiple complex components, it would be better to analyse the computational cost compared to baselines. 

• To thoroughly evaluate the performance of the proposed method, more advanced automated data science systems should be included as comparison baselines, such as “Data Interpreter: An LLM Agent for Data Science”.

### Questions
• The proposed method is complicated, which involves a multi-stage pipeline with multiple complex components. This paper lacks sufficient implementation details to reimplement the SmartDS-Solver. 
• As the SmartDS-Solver architecture consists of multiple complex components. It would be better to analyse the computational cost compared to baselines. 
• To thoroughly evaluate the performance of the proposed method, more advanced automated data science systems should be included as comparison baselines, such as “Data Interpreter: An LLM Agent for Data Science”.

### Soundness
2

### Presentation
2

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
This paper presents SmartDS-Solver, a hierarchical multi-agent framework for automating data science workflows. The system combines a domain-specific reasoning LLM (trained via structured distillation and GRPO fine-tuning) with a meta-learning agent (SARTE) that dynamically adjusts decoding parameters. Evaluated on MLE-Bench tasks, the system achieves an 81.8% win rate over AIDE+o1-preview while reducing computational costs.

### Strengths
1. The paper addresses real limitations in current AutoML agents - high costs, fragile multi-agent interactions, and excessive reliance on expensive models.
2. Comprehensive training methodology. The three-stage training pipeline (SFT → GRPO1 → GRPO2) with carefully designed reward functions is well-documented and appears reproducible.
3. SARTE's dynamic temperature adjustment based on execution feedback is creative and shows meaningful performance gains (+3.9% accuracy, -12% error rate).
4. Testing across 11 MLE-Bench tasks, 3 real-world tasks, and 18 AutoML-Agent benchmark tasks demonstrates broad applicability.
5. Significant reduction in token consumption compared to AIDE+o1-preview (e.g., ~90% reduction in inference tokens) while maintaining competitive performance.
6. 81.8% win rate with Qwen-32B-GRPO2 and 100% executable code generation on real-world tasks.

### Weaknesses
1. Limited baseline comparisons: The paper primarily compares against AIDE+o1-preview and AutoML-Agent. Missing comparisons with other recent systems like Agent-K, SELA (only shown in AutoML-Agent table), or AutoGluon on the primary benchmark would strengthen claims.
Incomplete reproducibility details:

2. Hardware requirements not fully specified (only "2 NVIDIA H100 GPUs" mentioned)
Training time and convergence details missing
Hyperparameter selection methodology for SARTE not clearly explained
How were the 11 MLE-Bench tasks selected from the 75 available?

3. Sample size is relatively small (11-14 tasks for main comparisons). Statistical significance testing only appears in appendix (Table A11). Some results show marginal significance (p=0.0995 for Qwen-7B).

4. All experiments are in data science domain - claims about "vertical domain" applicability need validation. Temperature sensitivity analysis (Table 3) shows high variance across tasks - unclear how SARTE would perform in completely new domains. The 7B model shows notably lower performance, questioning scalability to resource-constrained settings.

5. No ablation on individual GRPO stages (SFT+GRPO1 vs SFT+GRPO2). Limited analysis of reward function components ($\alpha, \beta, \gama$ weights). Code Agent's "minimally invasive patching" not empirically validated separately

Presentation issues:

1. Figure 1 is dense and difficult to parse. Some notation inconsistencies (e.g., "RLM" vs "reasoning LLM"). The distinction between GRPO1 and GRPO2 training objectives could be clearer

2. The composite reward function (Equation 2) has fixed weights - no justification or sensitivity analysis provided. SARTE's boundary-aware step-size control has multiple hyperparameters (line 16-18 in Algorithm 1) with unclear tuning process. The "semantic similarity" threshold for early stopping not specified.

Experimental design:

1. Different models trained to different stages (72B/70B only SFT) makes fair comparison difficult. AIDE configuration uses 20 steps uniformly - no exploration of whether fewer steps would be sufficient. Real-world tasks (Section 4, Table 1) show one failure for AIDE but unclear if this is representative.

Data concerns:

Training data construction relies heavily on DeepSeek R1 for augmentation - potential bias inheritance
Quality filtering uses Gemma3-27B scoring - criteria not validated
Code4ML and cell2doc datasets are relatively old (2023-2024)

Specific Technical Issues

1. SARTE algorithm: The control factor computation (Algorithm 1, lines 4-11) uses different formulas for success/failure/no-code cases, but the rationale for these specific functional forms is not provided. Why piecewise nonlinear for success but linear penalty for failure?
2. Reward function design: Equation 3 uses "Aggregate" function that's not defined until Appendix (Table A3). The weighting scheme between feature/algorithm/metric dimensions is not justified.

### Questions
1. How does SmartDS-Solver perform on tasks outside data science? Even a preliminary experiment in one other domain would strengthen generalization claims.
2. Can authors provide ablation studies isolating the contribution of each training stage (SFT, GRPO1, GRPO2)?
3. What is the sensitivity of performance to the reward function weights (α=0.5, β=0.25, γ=0.25)?
4. How were the specific functional forms in SARTE's control factor (Algorithm 1, Equations on lines 4-11) derived?
5. Can authors clarify the task selection process for the 11 MLE-Bench tasks used in primary evaluation?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a solid and well-executed system for automated data science with impressive empirical results (81.8% win rate, 93% token reduction) and exceptional implementation details rarely seen in current LLM research. However, the writing quality is surprisingly poor for ICLR. The paper structure is severely imbalanced: Introduction (1 page) is too brief, Methodology (5 pages) is overly detailed, and Experiments (2 pages) lacks depth, with most critical results buried in the appendix. The presentation reads more like a technical report than an academic paper, with confusing organization and crude figures. If the authors can substantially restructure the paper—expanding the introduction and experimental analysis while condensing the methodology—I would be willing to raise my score. The core contributions are valuable, but they are currently obscured by poor presentation.

### Strengths
1. Strong empirical results with solid methodology.
The paper achieves an 81.8% win rate against AIDE+o1-preview on MLE-Bench while reducing token consumption by 93%, demonstrating an excellent cost-performance trade-off. Extensive experiments across 32 tasks and 20 configurations validate the method's effectiveness and robustness.
2. Exceptional implementation details and reproducibility.
The 27-page appendix provides complete prompt templates, pseudocode, data construction pipelines, and hyperparameter settings, which is rare in current LLM research. The authors demonstrate strong engineering commitment, facilitating reproduction and future improvements.
3. Methodological innovation in training data construction.
The proposed three-component framework (Full Workflow + Decision Logic + Adjustment Trail) encodes agentic reasoning capabilities into training samples, going beyond existing work that only provides code+comments. Quality control uses a dual-layer mechanism (format checking + semantic alignment + Gemma3 scoring) to ensure data reliability.
4. Clever and practical SARTE mechanism design.
The approach models hyperparameter tuning as an online learning problem with O(1) space complexity (depending only on previous-step feedback) without requiring model retraining. The boundary-aware update strategy incorporates physical intuition, and Table 3 demonstrates that optimal temperatures vary dramatically across tasks/models (range up to 0.58), validating the necessity of dynamic adjustment.

### Weaknesses
1. Severely imbalanced paper structure, failing to meet academic standards.
The Introduction spans only 1 page with insufficient background and motivation—readers cannot understand why existing methods are inadequate. The methodology occupies 5 pages (54% of content) while experiments and conclusions take only 2 pages, lacking in-depth analysis and insights. The overall presentation reads like a technical report rather than an academic paper.
2. Confusing organization in the methodology section, poor readability.
Section 3 mixes training (3.2), inference (3.3), and code execution (3.4) into a single section when these should be separate chapters for clarity. Key innovations (e.g., the SARTE algorithm) are buried in implementation details, making it difficult for readers to quickly grasp the core contributions.
3. Insufficient experimental content in the main text; most experiments relegated to appendix with minimal analysis.
The main text contains only Figure 3 and Tables 1-2, while critical detailed results (Table A5 with full configuration comparisons, Table A8 with token consumption breakdown, Table A10 with temperature analysis) are all in the appendix. The main text completely lacks error analysis, failure case discussions, or deep investigation into why the method works—it merely stops at "proving the method works."
4. Poor figure quality with crude presentation.
Figure 3(b) appears as an unfinished draft with box plots missing legend explanations. Figure 1 is overly complex with too much text, making the system architecture hard to grasp quickly.

### Questions
The paper's detail is enough, the most important problem is the writing is terrible.

### Soundness
3

### Presentation
1

### Contribution
3
