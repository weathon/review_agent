# Transparent and Robust RAG: Adaptive-Reward Reinforcement Learning for Decision Traceability

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6

## Abstract
Retrieval-Augmented Generation (RAG) delivers substantial value in knowledge-intensive applications. Many recent works use reinforcement learning (RL) to effectively elicit strong reasoning in *RAG generators*. However, two key challenges remain unresolved: **(1) Transparency**: 
most prior methods do not explicitly indicate which references 
are actually used during the reasoning that leads to the final answer, limiting interpretability and visibility; **(2) Stability**: 
the KL divergence estimator used in existing RL-based approaches may cause gradient spikes,
which can lead to unstable training. To address these challenges, we propose **A**daptive-**R**ewarded **E**vidence **N**avigation **A**gent (**ARENA**), a transparent and robust RAG generator framework trained via RL with designed rewards. Based on our proposed 
structured protocol, KL divergence stabilization, and adaptive reward calculation modules, **ARENA** enables RAG generator to identify key evidence, perform structured reasoning, and generate answers with interpretable decision traces. 
Applied to Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct, abundant experiments with various baselines demonstrate that our model achieves highly transparent outputs with 10–30% accuracy improvements across three multi-hop QA datasets, which is comparable with advanced closed-source LLMs (e.g., OpenAI-o1, DeepSeek-R1). Further analyses show ARENA has strong generalization to unseen datasets and tasks. Our models and codes will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ARENA, a framework aimed at enhancing the generator component of retrieval-augmented generation (RAG). The authors identify two main shortcomings in existing RAG methods: (1) inadequate emphasis on improving the generator’s reasoning ability, and (2) limited transparency in its decision-making process. ARENA incorporates the following key components: (1) structured generation for more interpretable reasoning, (2) adaptive reward functions tailored to multi-hop question answering, and (3) KL-based stabilization techniques to ensure more stable RL. Experiments show that ARENA has strong generalization to unseen datasets and tasks.

### Strengths
1. The paper targets an important practical issue: traceability and interpretability in RAG systems.

2. Incorporating explicit evidence citation and structured output could benefit downstream auditing and reasoning interpretability.

### Weaknesses
1. **Heuristic and under-analyzed reward design.**
The adaptive reward shaping is highly ad hoc: the use of a fixed +10 bonus arbitrarily distorts the reward distribution. There is no sensitivity or ablation study to justify the weighting choices, making it unclear whether the reported gains are robust or merely tuned artifacts. Also, it seems that the authors combine several different rewards together in RL, which is incremental in nature.


2. **Conceptual ambiguity around “transparency” and “robustness.”**
These terms are used repeatedly but without precise definition or operationalization. It remains unclear whether the improvements stem from genuinely better reasoning and retrieval grounding, or simply from stronger format adherence due to the structured output template.

3. **Limited technical novelty.**
The paper lacks genuine methodological innovation. All major components—structured output generation, adaptive rewards, and KL stabilization—are incremental variations of well-established ideas from GRPO, DeepSeek-R1, or R1-Searcher. Their combination feels more like an engineering integration rather than a principled advance in RAG optimization.

4. **Narrow and unconvincing experimental scope.**
The evaluation is restricted to simple multi-hop QA. It is also better to use some other datasets that requires knowledge and reasoning, such as browsecomp, simpleQA, GPQA etc., for better evaluation.

### Questions
See above.

Missing reference: [1] Tang, Yunhao, and Rémi Munos. "On a few pitfalls in KL divergence gradient estimation for RL." arXiv preprint arXiv:2506.09477 (2025).

This paper also examines the use of K2 and K3 estimators for KL regularization in reinforcement learning. If this submission focuses on the general benefits of using K2 as the regularization term, it would strengthen the work to include additional benchmarks, such as MATH or Code reasoning, to demonstrate broader applicability beyond the current evaluation scope.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ARENA, an RL-trained RAG framework that aims to improve (i) transparency via a structured output protocol with explicit citations and (ii) stability via a stabilized KL term by using the k2 estimator. An adaptive reward combines format, answer correctness, citation relevance, and a large “all-correct” bonus. Experiments on HotpotQA, 2Wiki, and MuSiQue report gains over several baselines, with additional generalization and ablation studies. The work contrasts k2 vs. the commonly used k3 estimator for the KL regularizer in GRPO, arguing for lower variance and better stability for k2, with a brief analysis around gradients of D_KL.

### Strengths
1. This paper presents a clear, reproducible, and structured protocol that is easy to use and auto-grade.
2. This paper has broad empirical coverage (multiple datasets, retrieval settings, and model sizes) and some ablations.
3. This paper has some practical engineering contributions: reward design and training configuration are described in enough detail to be reimplemented.

### Weaknesses
1. The paper lists "transparency" and "stability" as two major challenges, and the corresponding solutions (structured protocols/reward and KL estimator replacement) are almost entirely independent of each other. It looks like two different ideas are stacked into the same paper.

2. Table 4: This table only compares different "training combinations" for ARENA, which has limited significance for intrinsic comparison. It lacks a comparison with strong baselines under the same training combination settings. Without this, it is unclear whether the improvements come from data mixing or the method itself.

3. The k2 estimator used in the paper is maybe similar to approaches in existing works (e.g., "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization"). It seems more like adopting an off-the-shelf alternative rather than introducing a novel theoretical method. The paper does not clearly articulate the fundamental differences or incremental contributions compared to these prior works.

4. k2 appears more like a switchable implementation or hyperparameter choice rather than a core methodological innovation. The paper lacks comprehensive comparisons and statistical significance tests for robustness.

5. Figure 5: The axis scales are severely inconsistent (left: 0.0–0.5; right: 0–5,000,000), making the comparison misleading. The scales and units should be unified, or the reason for the differing dimensions must be explained.

6. Insufficient Evaluation of "Transparency/Explainability": The authors mainly rely on case studies and custom metrics to claim improvements, but there is a lack of systematic quantification of citation accuracy, coverage, and redundancy, as well as fair comparisons with other methods. The current "relevance score" is strongly coupled with the training reward, which can lead to overestimation.

7. Ambiguity in Prompt-based Setup: The so-called "answering questions directly using references" does not clarify how the references are obtained, whether the methods share the same retrieval results, or whether there is any information leakage. This affects the fairness of comparisons.

8. Incomplete Related Work: The paper lacks a systematic discussion and empirical comparisons with relevant works in the "attributable citation" RAG direction, such as[1][2]

[1] Citation-Enhanced Generation for LLM-based Chatbots (ACL 2024)

[2] Model Internals-based Answer Attribution for Trustworthy RAG (EMNLP 2024)

### Questions
Please address the questions in the Weaknesses.

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
This paper proposes an Adaptive-Reward Reinforcement Learning (AR-RL) framework for improving the transparency and robustness of Retrieval-Augmented Generation (RAG) models. The framework aims to make the RAG process more interpretable and stable by introducing three components:
(1) a Stability KL term to ensure training stability,
(2) a Transparency Module that encourages explicit citation behavior, and
(3) a reward design that evaluates response relevance and citation correctness.
The goal is to produce RAG outputs that are both faithful to retrieved evidence and explainable through citation traces. Experiments on QA and citation-grounded datasets show that the proposed approach improves citation precision and factuality metrics over baseline RAG and reinforcement learning variants.

### Strengths
1. tackling the issue of traceable and interpretable RAG outputs is meaningful and timely.
2. includes modules addressing both stability and traceability, leading to more robust empirical performance.
3. results are consistently presented and compared with competitive baselines.
4. transparency-driven reward optimization aligns with current research trends in trustworthy RAG and factual QA.

### Weaknesses
1.	Weak conceptual connection among modules — The three main components (Stability KL, Transparency module, and Adaptive Reward) do not form a tightly integrated theoretical framework. Stability is for smoother training, transparency is for citation generation, and reward shaping is for content quality; however, the paper does not clearly articulate how these together solve one unified problem. This makes the work appear as a collection of engineering heuristics rather than a cohesive research contribution.
	2.	Reward design vulnerability — The Relevance Reward assigns 0.5 for partial overlaps and 0 for no overlaps, which may incentivize reward hacking: the model can increase citation quantity to maximize overlap probability, reducing citation precision. Introducing a citation-length penalty or constraint reward could prevent this issue.
	3.	Potential bias in ground-truth construction — The relevance labels come from locating paragraphs containing annotated supporting facts. This process assumes that only those paragraphs are relevant, but multiple documents might contain semantically similar information. As a result, the reward may penalize correct but alternative citations, introducing redundancy and bias. A better approach would be sentence-level citation matching and reference-grouping, where one correct reference per knowledge group yields a high reward.
	4.	Limited evaluation scope — The experiments are restricted to simpler QA datasets such as HotpotQA. Modern reasoning-oriented RAG tasks like BC and GAIA involve more complex multi-hop and abductive reasoning. Evaluating on such benchmarks would better demonstrate the framework’s robustness and generality.
	5.	Moderate novelty — Most components (stability KL, adaptive reward, transparency regularization) are known techniques; the innovation mainly lies in combining them, not in introducing new theoretical insights.

### Questions
1.	How are the three components—Stability KL, Transparency Module, and Adaptive Reward—conceptually linked? Could the authors provide a unified view or optimization objective that justifies their joint inclusion?
2.	Have you observed any evidence of reward hacking (e.g., the model generating excessive citations to gain partial-overlap rewards)? If so, how is it mitigated?
3.	Can you clarify how the ground-truth relevance labels are constructed when multiple supporting documents contain similar information?
4.	How would the framework perform on reasoning-heavy RAG benchmarks (e.g., BC, GAIA) where relevance and correctness may not perfectly align?
5.	Would adding a citation-length penalty or dynamic normalization improve reward fairness and prevent over-citation behaviors?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes ARENA, a transparent and robust RAG generator framework trained via RL with adaptive rewards. While ARENA outperforms naive GRPO and other popular methods with better RAG transparency, its overall novelty is limited due to the reliance on existing technical components (e.g., k2 KL estimator, multi-objective linear reward).

### Strengths
S1：ARENA significantly improves the performance of RAG generators compared to naive GRPO and other baseline methods (prompt-based, SFT-based, RL-based) across three multi-hop QA datasets (HotpotQA, 2Wiki, MuSiQue). It successfully balances transparency (explicit evidence-reasoning-answer traces) and stability (mitigated gradient spikes), providing actionable insights for RL-based RAG research.
    
S2：The introduction of a bonus reward effectively incentivizes the model to simultaneously satisfy format adherence, answer accuracy, and evidence relevance throughout training. Ablation studies confirm that this component is critical for maintaining consistent structured outputs, addressing a common challenge in multi-objective RL for language generation.

S3：The structured output protocol (with <relevance>, <analysis>, and <answer> sections) enhances decision traceability, a key requirement for real-world RAG deployment. Quantitative evaluations (Format/Relevance scores in Table 3) demonstrate that ARENA achieves near-perfect format adherence and substantially higher citation accuracy than base models.

### Weaknesses
W1：The "Adaptive Reward" component, while task-specific, relies on a simple linear combination of four reward terms (format, accuracy, relevance, bonus) with equal weights. This multi-objective reward design is not novel. The paper provides no justification for omitting dynamic weight adjustment or more advanced reward shaping strategies.
    
W2：While the paper adopts the k2 KL estimator in the GRPO pipeline to stabilize training, the core advantages of k2 (non-negativity, lower variance, gradient equivalence) are not original to this work (as acknowledged in Section 3.2).
    
W3：Although the paper includes theoretical analysis of the k2 estimator (Appendix B), it lacks in-depth discussion of why gradient stability (achieved via k2) translates to better RAG performance. The link between smoother KL curves and improved evidence selection/reasoning quality in multi-hop QA remains under-explained.
        
W4：Further experiments (e.g., generalization, cross-task transfer) only report results for ARENA-Qwen2.5-7B, neglecting ARENA-Llama3.1-8B and limiting insights into cross-architecture generalization.

### Questions
Q1：For the results reported in Table 2: Are all RL-based and SFT-based baselines trained on the Qwen2.5-7B backbone? If so, does the lack of Llama3.1-8B-based baselines make the performance comparison between ARENA-Llama3.1-8B and other models unfair (due to backbone differences)?
    
Q2：The paper claims that k2 estimator improves training stability, but it does not explain how this stability specifically benefits multi-hop reasoning and evidence tracing. Could the authors elaborate on the mechanism linking gradient smoothness to better RAG performance?
    
Q3：The bonus reward (10 points) is significantly larger than the other three component rewards (1 point each), yet the paper provides no justification for selecting the value "10" or conducting sensitivity analysis on this hyperparameter. Why was the specific value of 10 chosen, and have experiments been conducted with alternative bonus values (e.g., 5, 15) to validate its optimality for RAG multi-hop QA tasks?

### Soundness
3

### Presentation
3

### Contribution
3
