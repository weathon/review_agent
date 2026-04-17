# Unsupervised Prompt Learning with Few-shot Examples for Answering Objective Questions

- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Large language models (LLMs) have been highly successful on diverse tasks, while some applications require specializing general purpose LLMs to meet stricter accuracy or latency targets; here we focus on objective question answering, an important real-world setting in which a nontrivial subset benefits from such specialization. Most existing methods require parameter retraining or human supervision, both entailing high computational and data collection burdens. To handle these challenges, a direct approach is to generate ``high-confidence'' data from unsupervised downstream tasks and use them for prompt learning or in-context learning to efficiently refine pseudo-supervision. We consider combining the two approaches for better performance; however, a naive strategy that learns the prompt first and selects pseudo-supervised examples only at inference creates a mismatch between prompt learning and usage. In this paper, we propose unsupervised few-shot prompt learning (UFPL), which jointly learns the prompt and refines the overall pseudo-supervision. The learning objective aligns prompt training with usage by requiring the learned prompt to produce consistent answers when pseudo-supervised data from the downstream task are used as in-context examples. We optimize the prompt by translating gradient signals into textual critiques, which serve as feedback to iteratively refine the prompt and the pseudo supervision. Theoretical analysis in a simplified classification setting shows that the algorithm implicitly introduces a regularization, supporting its design. Empirical results on diverse benchmarks and a real world molecule optimization task show the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes UFPL, which selects high-confident answers as pseudo labels and uses few-shot demonstrations in the prompt optimization stage.

### Strengths
1. This work bridges the gap between prompt tuning and few-shot ICL. 

2. The experiment is comprehensive.

### Weaknesses
1. This work is incremental to me. The selection of high-confident pseudo labels and the optimization algorithm are from existing works [1,2]. The claimed contribution is adding a demonstration into the prompt tuning process, which only introduces additional information to the input without addressing many technical challenges. Moreover, the authors provide a theoretical analysis in section 3.3 but the analysis is about how pseudo supervision helps optimization instead of supporting the claimed contribution of this work.

2. The figures in this paper is hard to interpret. Is the red D_i in figure 1 in prompt training and prompt using phase representing the same demonstration? The format of the caption in Figure 2 is not very formal. The font used in the Figures for the experiments is too small.

[1] Huang J, Gu S S, Hou L, et al. Large language models can self-improve[J]. arXiv preprint arXiv:2210.11610, 2022.
[2] Yuksekgonul M, Bianchi F, Boen J, et al. Textgrad: Automatic" differentiation" via text[J]. arXiv preprint arXiv:2406.07496, 2024.

### Questions
1. This paper uses self-consistency to produce pseudo labels to guide the optimization. Have the authors tried methods to mitigate the consistent error (confident but erroneous response) in this stage? Is there any solution to this for further improving the ICL performance?

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
4

### Summary
The paper introduces Unsupervised Few-Shot Prompt Learning (UFPL), a method to specialize general-purpose LLMs for objective question-answering tasks without parameter updates or human supervision. By jointly optimizing a textual prompt and refining pseudo-supervision from high-confidence in-context examples, UFPL closes the gap between prompt training and inference-time usage. The approach leverages gradient-based textual feedback (via TextGrad) to iteratively improve both prompt and pseudo-labels. A simplified theoretical analysis links UFPL to implicit ERM regularization, and extensive experiments, including a real-world molecule optimization task, demonstrate strong gains over baselines.

### Strengths
- The joint optimization of prompt and pseudo-supervision with aligned training-inference objectives is novel and addresses a clear practical pain point: the train-test mismatch in prior high-confidence pseudo-labeling pipelines.
- Translating gradients into textual critiques enables prompt refinement without access to model parameters, a clean, scalable mechanism.
- Experiment strengths:
    + Broad coverage across LLMs and standard QA benchmarks, plus a challenging real-world molecule optimization task.
    + Clear ablation studies and comparisons to recent unsupervised/zero-shot baselines.

### Weaknesses
- Unclear Learning Objective in Eq. (4): Eq. (4) describes learning objectives based on high-confidence pseudo-supervised data.
However, the loss term encourages the model with optimized prompt $z$ to match the zero-shot
baseline ($\mathbf{z}_0, \emptyset$). This seems counterintuitive: we often expect specialization to
diverge from zero-shot behavior on selected high-confidence inputs, not converge to it.
- The theoretical analysis rests on Assumption 1 (Leave-one-out correctness) and
Assumption 2 (Uniform stability), both of which appear highly idealized:
  + Assumption 1 requires that for every $i$, $h^{(-i)}(x_i) = y_i$, i.e., the demonstrator
perfectly recovers the true label when excluding $x_i$. This is unlikely in low-data, noisy,
or ambiguous settings.
  + Assumption 2 imposes Lipschitz continuity and bounded deviation of the score function
under leave-one-out, again, strong for real LLMs.
- Moreover, Theorem 1 concludes that UFPL induces classifier separation $\geq \gamma$, but:
   + It is unclear how these assumptions reflect the unique properties of UFPL vs. any ICL-based
method.
    + No empirical validation of these assumptions is provided (e.g., measuring leave-one-out
accuracy or stability).
- The set of baselines is inadequate and seems to be chosen arbitrarily. Please consider recent and stronger prompt optimization approaches. For example, RL-based [1], supervised method [2], and LLM-based [3] methods. At least, please discuss related recent works and explain why your method is different and/or cannot be compared with them.  

[1]  Do et al. "Large Language Model Prompting with Episodic Memory." In ECAI. 2024.

[2] Do, Viet-Tung, Xuan-Quang Nguyen, Van-Khanh Hoang, Duy-Hung Nguyen, Shahab Sabahi, Jeff Yang, Hajime Hotta, Minh-Tien Nguyen, and Hung Le. "Automatic prompt selection for large language models." In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pp. 91-102. Springer, Singapore, 2025.

[3] Yang, C., et al.: Large language models as optimizers. arXiv preprint arXiv:2309.03409 (2023)

### Questions
Please clarify the following issues:
- Please explain the motivation for Eq. (4)
-  Notation: D, D_k, D_l, D_i, \mathcal{D} used interchangeably (esp. Between Eq. 5 and 6)
- S_l: unclear if S_l is a set of indices or inputs x_k
- L93: BDPL is undefined
- L96: TextGrad might not rely on human supervision
- Algorithm 1: Key formulas (e.g., gradient $\frac{\partial L}{\partial z}$, update rule) missing; parameters like $m$, $T$, $\gamma$ appear but are not being used in pseudocode.
- Table 1: PAPO is undefined.
- Other format error: - L218: Zhao et al. (2021) instead of (Zhao et al. 2021).

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes unsupervised few-shot prompt learning (UFPL) to jointly learn the prompt and refine the overall pseudo-supervision. Both theoretical analysis and empirical results are presented.

### Strengths
S1. Well-motivated work

S2. Extensive experiments.

S3. A theoretical analysis, though in a simplified setting.

### Weaknesses
W1. UFPL is inherently more expensive and complex than baselines, as shown in Figure 3 and Figure 4, limiting their use in a wider range of scenarios.

W2. There are several hyperparameters, which may be hard to set in a truly unsupervised setting. I did not find hyperparameter analysis in Appendix C. I only saw some limited analysis on one hyperparameter in  question answering and natural language inference tasks in Figure 6. Not sure whether such a conclusion drawn from limited tasks can be generalized.

### Questions
Q1. Could you quantitatively compare the computational cost of your approach with the baselines? The current description about Figure 3 and Figure 4 is vague (e.g., 'modest increase').

Q2. Could you conduct hyperparameter analysis on all datasets as in your main experiments?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors propose to specialize general-purpose large language models (LLMs) for objective question answering tasks without human supervision or parameter updates, which involves queries with a single verifiable answer. Specifically, they aim to address limitations in existing methods like fine-tuning and supervised in-context learning by generating high-confidence pseudo-supervised data from unlabeled downstream tasks using self-consistency chain-of-thought (CoT) reasoning. Experiments are conducted on several benchmark datasets when the experimental results show the consistency accuracy gains.

### Strengths
* The proposed method aligns training and inference. By jointly optimizing prompts and pseudo-supervision, it avoids mismatches in conventional approaches and ensures few-shot examples to be used consistently.
* The proposed method relies solely on LLM-generated pseudo-labels from unlabeled data, reducing data collection burdens while achieving high-quality refinements applicable to black-box models.
* The authors also provide theoretical analysis showing implicit regularization for generalization when the method is validated on diverse benchmarks.

### Weaknesses
* Iterative refinements and initial distance matrix for example selection add runtime and cost, potentially limiting scalability for large dataset as computational overhead.
* The proposed method relies on self-consistency CoT, so inaccuracies in weak base LLMs could propagate errors.
* The proposed method is limited to objective QA with verifiable answers, so it may not generalize to open-ended or subjective tasks without modifications.

### Questions
* How does the proposed method perform on non-objective tasks (e.g., open-ended or creative writing tasks)?
* What are the specific computational environment used in the experiments (e.g., GPU and memory)?
* How does the proposed method handle noisy or ambiguous unlabeled data in the downstream task?
* How does the proposed method perform when applied to multilingual or cross-lingual objective QA tasks?

### Soundness
3

### Presentation
3

### Contribution
3
