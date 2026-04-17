# Retrieval-Augmented Meta Test-Time Training for Multimodal Reasoning

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Reasoning lies at the core of Large Vision–Language Models (LVLMs). Recent Test-Time Scaling (TTS) methods enhance reasoning by allocating additional computation during inference. However, they primarily exploit the model’s internal knowledge without incorporating new information, which limits their effectiveness under distribution shifts.
While retrieval can introduce new knowledge, existing methods primarily emphasize semantic similarity rather than reasoning utility, leaving LVLMs struggle to effectively leverage the retrieved examples for complex reasoning.
To address these limitations, we propose RAM-TTT (Retrieval-Augmented Meta Test-Time Training), a retrieve–train–generate framework that unifies retrieval with meta-adaptation. RAM-TTT includes two key components: LVLM Aligned Retrieval (LAR), which selects examples for both semantic relevance and reasoning utility, and Meta Test-Time Training (Meta TTT), which casts retrieved examples as alternating support sets and meta-queries, allowing the model to ``learn how to learn'' from external information while mitigating overfitting.
Experiments show consistent gains on MathVerse (+6.4\%), LogicVista (+5.6\%), and We-Math (+8.5\%) with Qwen2-VL-7B, and strong generalization to Phi-3.5-Vision and Pixtral-12B. These results highlight RAM-TTT’s broad applicability in enabling LVLMs to acquire and internalize new information at test time for stronger reasoning under distribution shifts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a multimodal retrieval-augmented generation framework, which consists of two components: LVLM Aligned Retrieval (LAR) and Meta Test-Time Training (Meta TTT). Overall, the paper is well presented. However, several technical shortcomings remain.

### Strengths
1. The paper is well presented and easy to follow.
2. Additional experiment on Phi-3.5-Vision and Pixtral-12B demonstrate the method's generalization across different model architectures.

### Weaknesses
1. The paper proposes LAR and Meta TTT, while the title emphasizes only Meta TTT. If Meta TTT is the core contribution, I recommend focusing solely on it and removing LAR to avoid diluting the main contribution.
2. While both LAR and Meta TTT are shown to be effective, the paper lacks a deeper analysis of how they work. For example, the authors claim that LAR improves reasoning utility, but no ablation or analysis is provided to support this. The same applies to Meta TTT.
3. The work emphasizes multimodal reasoning, but the backbone model (Qwen2-VL) does not support CoT reasoning, which raises concerns about how reasoning is truly being evaluated. Moreover, given the recent release of Qwen2.5/3-VL, experiments on these stronger models would be more compelling.
4. There is almost no comparison with other state-of-the-art methods in multimodal RAG. The comparisons with ICL and TTT in Tables 1 and 2 can be seen as ablation study.
5. I also question the suitability of the selected primary areas "transfer learning, meta-learning, and lifelong learning" for this work, as the paper is fundamentally centered on multimodal RAG.

### Questions
See Weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RAM-TTT, a Retrieval-Augmented Meta Test-Time Training framework aimed at enhancing the reasoning capability of Large Vision-Language Models (LVLMs) under distribution shifts. RAM-TTT unifies two key innovations: (1) LVLM-Aligned Retrieval (LAR) selects support examples based on both semantic relevance and their utility for reasoning, and (2) Meta Test-Time Training (Meta TTT) leverages a meta-learning paradigm to adapt the LVLM with retrieved examples while mitigating overfitting.

### Strengths
Motivational Clarity and Problem Significance. The paper identifies and articulates a genuine gap: standard Test-Time Scaling methods in LVLMs typically fail to incorporate new, external knowledge, which limits adaptation to distribution shifts and complex reasoning scenarios. Targeting this problem is valuable for the field.

Extensive Empirical Validation. Experiments span four strong reasoning benchmarks and three LVLM backbones. Results in Table 1 and Table 2 show RAM-TTT outperforming several reasonable baselines (e.g., Zero-shot, Self-Correction, TTT, ICL) by a notable margin.

Detailed Ablations and Analyses. Ablation results are comprehensive, examining the importance of each module (Meta TTT, LAR, contextual alignment), retriever variants, permutation effects, and computational trade-offs, demonstrating careful empirical study. Figures 3 and 4, for instance, solidly illustrate the method’s robustness to parameter settings and model backbone.

### Weaknesses
1. In Figure 2, essential mathematical annotations are missing. For example, the (e_i) referenced in §2.2 and the candidate set (C_{\text{cand}}) do not appear in the figure.
2. Line 173 uses the inverse of perplexity to measure the LVLM’s confidence, but this lacks a detailed explanation (although I found one in the supplementary material) and omits citations to key references:
   1. *Language Models are Unsupervised Multitask Learners* (GPT-2)
   2. *Calibrating Trust of Large Language Models via Perturbation-Based Calibration*
   3. *Self-Consistency Improves Chain of Thought Reasoning in Language Models*
3. The treatment of overfitting remains insufficient. Although the Meta TTT paradigm is designed to avoid overfitting on tiny support sets, the evidence for robustness is largely anecdotal (see the discussion around Figure 5). The scientific value would be improved by providing more rigorous quantification—e.g., plotting performance/error versus the number of adaptation steps, or comparing against alternative regularization schemes.
4. The ablation tables (especially Tables 3–5) do not report standard deviations or confidence intervals across multiple runs.
5. RAM-TTT introduces additional computation for re-ranking, meta-training, and prompt permutations. However, the paper lacks a detailed discussion of the framework’s complexity and computational overhead.
6. I notice the authors use proprietary prompt designs. It remains unclear how the method performs under unseen or alternative prompts.

### Questions
The same as Weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes a novel Retrieval-Augmented Test-Time Training (RAM-TTT) framework that successfully integrates the benefits of Retrieval-Augmented Generation (RAG) with Test-Time Training (TTT) techniques. The framework’s design moves beyond simple semantic similarity by additionally incorporating "reasoning utility" into the training process, thereby ensuring retrieved results are optimal for complex reasoning tasks. The experiments demonstrate that this new framework effectively enhances answer accuracy. However, the paper lacks sufficient clarification regarding its core motivation, and the claims about the effectiveness of the reasoning utility metric require more substantial evidence.

### Strengths
1）Novel Integration: The paper introduces a novel framework (RAM-TTT) that effectively integrates Retrieval-Augmented Generation (RAG) with Test-Time Training (TTT), which is a promising and timely direction for enhancing model performance, particularly under distribution shifts.

2）Reasoning-Aware Retrieval: The training framework is uniquely designed to optimize retrieval not only based on traditional semantic similarity but also on a new metric, "reasoning utility," ensuring the retrieved examples are highly relevant to the complex reasoning task at hand.

3）Empirical Performance: The experimental results demonstrate that the proposed method significantly improves the accuracy of answers across various benchmarks, validating the effectiveness of the combined approach.

### Weaknesses
1）The core motivation and justification for the proposed method could be elaborated more thoroughly.

2）While the abstract claims improvements in both accuracy and reasoning efficiency, the experimental section heavily prioritizes verification of accuracy, with insufficient evidence or analysis dedicated to demonstrating the gains in reasoning efficiency.

3）The analysis and interpretation of the observed phenomena and hypotheses within the ablation study are too brief and could benefit from deeper discussion.

### Questions
1）Why do you choose high-confidence pairs in stage 1? These pairs are used for training. If the trained data is low-confident which means the model haven't learned them, will the performance be better? Because the reference in RAG is often supplementary material which the model haven't seen, if model have studied it in advance, the performance will be better in most cases.

2）In stage 2, why do you use the leave-one-out strategy to make $M$ random permutations? For data enhancement?

3）Can you briefly describe the difference between TTT and meta TTT?

### Soundness
3

### Presentation
2

### Contribution
3
