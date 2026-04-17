# DUET: Distilled LLM Unlearning from an Efficiently Contextualized Teacher

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
LLM unlearning is a technique to remove the impacts of undesirable knowledge from the model without retraining from scratch, which is indispensable towards trustworthy AI. Existing unlearning methods face significant limitations: conventional tuning-based unlearning is computationally heavy and prone to catastrophic forgetting. In contrast, in-contextualized unlearning is lightweight for precise unlearning but vulnerable to prompt removal or reverse engineering attacks. In response, we propose Distilled Unlearning from an Efficient Teacher (DUET), a novel distillation-based unlearning method that combines the merits of these two lines of work. It learns a student model to imitate the behavior of a prompt-steered teacher that effectively refuses undesirable knowledge generation while preserving general domain knowledge. Extensive evaluations on existing benchmarks with our enriched evaluation protocols demonstrated that DUET achieves significantly higher performance in both forgetting and utility preservation, while being orders of magnitude more data-efficient than state-of-the-art unlearning methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces DUET, a distillation-based LLM unlearning method that transfers the refusal behavior of an in-contextualized teacher into a student model via Top-K logit distillation. Concretely, the teacher is a frozen base LLM steered by a compact unlearning prefix; DUET trains the student to match the teacher’s dominant raw logit shifts on forget queries, while mixing in retain queries under the same objective to preserve utility. DUET requires only query-level forget data without no ground-truth answers or refusal templates, and achieves strong forgetting–retention trade-offs on LLM unlearning benchmarks. Comprehensive ablations also indicate robustness to reverse-prompt attacks and evaluation-format shifts.

### Strengths
1. The paper is well written and easy to follow.

2. Distilling from an in-context teacher to obtain an unlearned student is reasonable, cleanly formalized, and avoids constructing ground-truth answers or refusal templates for forget data.

3. The work proposes and validates the effectiveness of Top-K logit distillation for unlearning.

4. The experimental section demonstrates the method’s effectiveness and provides thorough ablations showing robustness to reverse-prompt attacks and evaluation-format shifts.

### Weaknesses
The paper’s discussion of distillation-based unlearning remains incomplete. Similar ideas have been applied in the LLM unlearning literature, including W2SDefense (weak-to-strong distillation for backdoor removal) [1], UNDIAL (self-distillation with adjusted logits) [2], and UNDO (“distillation robustifies unlearning”) [3]. The manuscript should explicitly discuss conceptual and practical differences from these works, and add an analysis to clarify DUET’s unique contributions.

### Reference

1. Zhao, Shuai et al. “Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation.” ACL (2024).

2. Dong, Yijiang River et al. “UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models.” NAACL (2024).

3. Lee, Bruce W. et al. “Distillation Robustifies Unlearning.” ArXiv.

### Questions
1. The paper states that the Harry Potter (MUSE-Books) evaluation set is expanded to 500 items. How exactly is this expansion performed?

2. Please clarify the source and selection process for the retention data used during training and evaluation.

3. In Table 1, why does NPO with a retain-set KL (w $\text{KL}(\mathcal{D_r})$, line 320) yield worse retain performance than the variant without retain-set KL (w/o $\text{KL}(\mathcal{D_r})$, line 319)? Please explain the underlying cause.

### Soundness
4

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
The paper proposes **DUET**, a distillation-based unlearning framework that transfers a teacher model’s refusal behavior into a student model via top-K logit alignment at the first decoding step. DUET aims to retain general capabilities while suppressing undesirable knowledge. Experiments on MUSE-Books and WMDP show strong forgetting with comparatively small utility loss, and improved robustness to a simple reverse-prompt attack.

### Strengths
1. **Simple method and clear motivation.** The approach is conceptually straightforward and is well justified relative to tuning-based and purely in-context unlearning.
2. **Good empirical performance.** On MUSE-Books and WMDP, DUET achieves lower forgetting scores and stronger utility preservation than most baselines. 
3. **Robustness to attack.** The distilled model is notably less sensitive to a reverse-prompt attack than a purely in-context teacher.

### Weaknesses
1. **Distilling only the first decoding step is not fully convincing.** Many tasks (e.g., math reasoning) often start with stereotyped lead tokens (e.g., _“To solve the problem, I need to…”_), so aligning only the first-step logits may fail to shape downstream generation in a robust way. The paper explicitly trains on **first-position** logits only; a deeper justification and multi-step ablations would help.
2. **Limited experimental breadth.** The forget set and retention set used for training are quite small (e.g.,  100 queries for Harry-Potter forgetting; 100 for retention), which risks overfitting the teacher prefix distribution and may bias results toward DUET’s design. Broader forget/retention sets or more domains would strengthen claims.
3. **Adversarial evaluation is narrow.** “Reverse attacks” are instantiated as a single reverse-prompt. The paper does not cover more systematic jailbreak or targeted relearning attack suites, so robustness claims remain preliminary.

Overall, I like this paper, and if the authors can provide a clear explanation about W1 and W2, I’d be happy to raise my score.

### Questions
1. Why apply the unlearning prefix to _retention_ data in Eq. (3)? A seemingly cleaner design is to prefix only the forget set $D_f$​ and leave $D_r$ unmodified. 
2. Is this aggregate metric "performance shift" standard in the unlearning literature, or specific to this paper?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DUET (Distilled Unlearning from an Efficient Teacher), a novel distillation-based method for large language model (LLM) unlearning—the process of removing undesirable knowledge without full retraining. Existing unlearning techniques either suffer from high computational cost and catastrophic forgetting (tuning-based approaches) or security vulnerabilities such as prompt removal and reverse-engineering attacks (in-context methods). DUET addresses these issues by training a student model to emulate a prompt-steered teacher that suppresses unwanted knowledge while retaining useful domain capabilities. Experiments on benchmark datasets show that DUET delivers forgetting and utility preservation, achieving good performance with higher data efficiency than state-of-the-art methods.

### Strengths
The novelty of DUET lies in its distillation-based unlearning approach, where a student model learns from a prompt-steered teacher to selectively suppress undesirable knowledge while preserving useful information. This method combines efficiency, robustness, and effectiveness, outperforming existing unlearning techniques in both forgetting unwanted content and maintaining model utility.

The research provides:

•	Efficient unlearning without full retraining, saving computational resources.

•	Effective knowledge removal while preserving useful domain information.

•	High data efficiency compared to existing unlearning methods, achieving reasonable performance on benchmarks.

### Weaknesses
The research demonstrates some shortcomings

•	Reliance on teacher quality — form my understanding effectiveness depends on how well the teacher model suppresses unwanted knowledge.

•	The work makes a valuable contribution and builds effectively on current advances. However, including a discussion of remaining challenges and possible avenues for future research would strengthen the paper and highlight its long-term potential.

•	Limited evaluation — generalisation across LLMs is not shown.

### Questions
Please check your references – there are titles that have some “words” like LLM that should be uppercase.

With regard to the use of LLMs, I suggest that you check the claims that are made, such as “comprehensive evaluations” and “significantly superior performance”. It’s up to the reader to judge if the evaluations are comprehensive and whether the performance is “improved” or “significantly superior”. Please discuss.

Please discuss the generalizability of the results across LLMs.

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
This paper proposes DUET, a novel distillation-based unlearning method that combines the merits of two approaches: optimizing a student model to imitate the behavior of a prompt-steered teacher, which effectively refuses the generation of undesirable knowledge while preserving general domain knowledge. Experiments on existing benchmarks demonstrate that DUET achieves strong performance in both forgetting and utility preservation, with greater data efficiency.

### Strengths
1. The proposed approach is novel and sound. The paper reveals that training-based unlearning achieves stronger robustness but risks greater utility degradation, while contextualized unlearning enables more precise unlearning yet can be easily reversed. The proposed method strikes a good balance between these two paradigms.
2. The paper proposes top-k logits distillation to further enhance performance.
3. The paper is enjoyable to read.

### Weaknesses
The authors should compare their method with more baseline methods, such as [1][2][3], as well as additional distillation-based approaches—for instance, distillation from gradient ascent methods.

How about distillation from multiple unlearning teachers?

[1] Unified Gradient-Based Machine Unlearning with Remain Geometry Enhancement, NeurIPS'24
[2] Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation, ICLR'24
[3] Torward Natural Machine Unlearning, TPAMI'25

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
