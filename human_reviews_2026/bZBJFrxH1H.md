# Distillation of Large Language Models via Concrete Score Matching

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Large language models (LLMs) deliver remarkable performance but are costly to deploy, motivating knowledge distillation (KD) for efficient inference. Existing KD objectives typically match student and teacher probabilities via softmax, which blurs valuable logit information. While direct logit distillation (DLD) mitigates softmax smoothing, it fails to account for logit shift invariance, thereby restricting the solution space. We propose Concrete Score Distillation (CSD), a discrete score-matching objective that overcomes both softmax-induced smoothing and restrictions on the optimal solution set. We resolve the training instability and quadratic complexity of discrete score-matching in autoregressive LLMs, and the resulting CSD objective aligns relative logit differences across all vocabulary pairs between student and teacher with flexible weighting. We provide both mode-seeking and mode-covering instances within our framework and evaluate CSD on task-agnostic instruction-following, task-specific, and general chat capability distillation using GPT-2-1.5B, OpenLLaMA-7B, and Gemma-7B-IT, Qwen2.5-7B-IT, and Gemma2-9B-IT teachers. Experiments show that CSD consistently surpasses recent KD objectives, achieves favorable fidelity–diversity trade-offs, and yields complementary gains when combined with on-policy techniques, demonstrating its scalability and effectiveness for LLM distillation. Code: https://github.com/aailab-kaist/CSD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Concrete Score Distillation (CSD), a novel training objective for improved large language model's knowledge distillation. Based on the observation that softmax is invariant to constant addition, the proposed method extends direct logit distillation (DLD) by allowing student's logits to be matched to the teacher's logits shifted by an arbitrary constant. The paper further introduces and experiments with various weighting functions to balance the output accuracy and diversity of the trained student. Experiments show that the proposed method outperforms prior approaches compared to divergence-based loss function, and that it achieves complementary gains when used in combination with recent on-policy KD methods.

### Strengths
1. The paper, including the algorithm, theoretical analyses and proofs, is well-written and easy to follow.
1. The idea of enhancing logit matching distillation by accounting for the logit shift invariance is well-motivated and reasonable.
1. The author's extension to DLD theoretically and empirically achieves the target of logit shift invariance.
1. The experimental results show improvements compared to the baselines, across multiple model families and model sizes.
1. The paper is well placed in existing literature, and prior works considered are thorough.

### Weaknesses
1. The authors claim that one of their motivations is that vanilla KLD gives almost identical and very low probabilities and gradients to the low-probability tokens in the tail. While their method, with uniform weighting (w1,w2=U) would indeed achieve their target of "overcoming the softmax-induced smoothing of teacher knowledge", most of their results use w1,w2 as either student or teacher probabilities. As these values are very small and almost zero, a very similar smoothing is also present in the proposed method.
1. Comparison with DLD: since the method is an extension to DLD, I believe it would be more appropriate to compare the method with variants of logit matching methods. Currently, only the comparison between DLD and CSD on GPT2-0.1B for the five benchmarks is given. It would be great to see how CSD compares against DLD at larger model sizes, different datasets, etc.
1. Weighting function: not using weighting function (corresponding to the (u, u) setting in the paper) performs significantly worse compared to other divergence-based baselines, and the overall performance seems to be heavily impacted by the choice of the weighting function. Furthermore, with S as a weighing function, DLD performance seems to be only slightly worse than the proposed method. This makes me wonder if similar performance improvement can be made to the divergence-based methods by adding similar weighting function to those.
1. The method is only tested on relatively small datasets (eg. 15000 samples), its effectiveness in longer training horizon is unexplored.
1. Most of the experiments were performed on a relatively weak model (GPT2), with only some experiments on Openllama/gemma. It is uncertain if the results will hold with newer/stronger models.

### Questions
1. In figure 4, "Ours" refers to which exact model/setting for the two model sizes? Just CSD, or perhaps ImitKD+Ours, or something else? Could the authors share the GPT evaluation scores of vanilla CSD, GKD+ours, DistilLLM+ours,ImitKD+ours if these were also evaluated?
1. For results in Table 3, w1,w2 were used as (t,s) (as mentioned in line 879) compared to the "default"(as mentioned in line 312) of (s,s) elsewhere. Can the authors elaborate on why were w1,w2 changed for these experiments? If the authors tried (s,s) first, could they share these results?
1. In figure 8 (a) and (b), could the authors also share the KL-divergence/MSE/some other divergence metric between the teacher and student probabilities corresponding to y1 to y10? This will prove that the student probabilities with CSD maintain similarity to teacher inspite of logit shift.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Concrete Score Distillation (CSD), a knowledge-distillation objective for autoregressive LLMs that addresses two core issues: (1) probability-matching objectives (e.g., KL) lose fine-grained information at the logit level due to softmax smoothing; (2) while direct logit distillation (DLD) avoids softmax, its objective implicitly forces the student logits to match the teacher exactly, ignoring logit shift invariance (i.e., an additive constant does not change probabilities), which restricts the optimal solution set. CSD adapts score matching from energy-based modeling to a discrete setting and matches pairwise relative logit differences across the vocabulary, thereby being inherently shift-invariant and expanding the attainable solution set. The paper further addresses training instability and computational overhead when applying score matching to autoregressive LLMs by deriving an O(|V|) analytic gradient (from a naïve O(|V|2) formulation). Experiments show that CSD outperforms recent probability-matching and direct-logit objectives on both task-agnostic instruction following and task-specific distillation (summarization, math, and translation), offers a controllable fidelity–diversity trade-off, and yields complementary gains when combined with on-policy techniques, indicating good scalability.

### Strengths
1. Precise problem framing and strong motivation. The paper clearly identifies fundamental shortcomings of existing approaches—probability matching loses logit information; DLD over-constrains the solution set by ignoring shift invariance.
2. Theory-grounded contribution. CSD is derived from score matching with key guarantees: Proposition 1 (consistency) and Theorem 2 (solution set is a superset of DLD). This provides principled justification for the objective.
3. Clear, controllable objective design. Equation (9) reveals a gradient structure based on centering, factorized weighting, and role swapping; the (w1,w2) design smoothly tunes the fidelity–diversity trade-off (as illustrated in Figure 3).

### Weaknesses
1. Underpowered DLD baselines. DLD does not appear in the main comparison table and is shown only once in ablations (Table 4). Stronger DLD variants—e.g., mean-centered, temperature-scaled, or otherwise shift-aware formulations—are missing, as are several recent logit-distillation variants. This makes it hard to assess CSD’s margin over the strongest DLD baselines.
2. Practical computational burden. Although Theorem 3 reduces complexity from O(|V|2) to O(|V|), real-world LLMs with very large vocabularies (e.g., 50k+) still require computing and normalizing all token logits per step. The paper shows that a Monte-Carlo estimator performs worse, so the efficient implementation currently relies on full-vocabulary passes, which may become a bottleneck at scale.

### Questions
1. Stronger DLD comparisons. If DLD is augmented with mean-centering, temperature scaling, or an explicit shift-invariant formulation, does CSD still retain the advantages reported in Tables 1 and 3? Could you include strong DLD in the main results and report variance?
2. On the solution space. Since CSD allows a global additive shift on logits, how does this offset behave during training—does it converge to a stable value? Are there systematic differences across tokens? Does this flexibility introduce any extra instability compared to DLD?
3. Scaling characterization. For vocabularies of 32k–100k and sequence lengths of 4k–8k, what are the peak memory, step time, and convergence trade-offs of CSD (analytic vs. MC) versus KL/DLD? Is there a practical heuristic for choosing analytic vs. MC estimators at different scales?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the challenge of distilling large language models. To address some of the limitations of Direct Logit Distillation, the authors propose leveraging score matching for discrete variables. They thus propose a new objective for knowledge distillation named Concrete Score Distillation. The main idea is to replace the dataset data distribution ($p_{\textrm{data}$) in the score matching objective with the teacher probability $p_T$. To address training instability, they propose using a logarithm function. They also propose an efficient approach for computing gradients. A complete experimental validation is provided for different setups: task-agnostic instruction-following distillation and task-specific distillation.

### Strengths
**originality**
 + The idea of using the generalized score function for LLM knowledge distillation is new and original.
 + The proposed approach is technically sound, and the claims are supported by both theoretical aspects and by the experimental part.

**quality**
 + The effectiveness of the proposed approach is validated on multiple KD settings.

**clarity**
+ The paper is well-written, with a clear formalization of the proposed approach.
The approach is also well-motivated.
+ The annexes of the paper contain important additional information, such as proofs, derivations, and additional experimental details.

**significance**
+ LLM Knowledge distillation is a crucial issue for the real-world deployment of LLMs. The proposed approach clearly advances the state of the art on LLM Knowledge distillation.

### Weaknesses
+ Some minor remarks on equation 5 and 6 : $p_{\textrm{data}}$ is not defined. 
+ Proposition 1 and Theorem 2 assume a sufficient model capacity. This should be detailed. What is a sufficient model capacity in this particular case? 
+ In Theorem 3, an assumption is that we can write the weighting function $w(y_t,x) = w_1(y_t) w_2(x)$. The authors do not justify this assumption. This rewriting of the weighting function is important in the proposed approach and in the experimental part. The paper does not explain how to build $w_1$ and $w_2$.

### Questions
+ A key assumption of the proposed approach is that the teacher and student models share the same vocabulary. We know that, in practice, this hypothesis is not satisfied due to the differences between the different tokenizers. How can this issue be addressed in the proposed approach?

### Soundness
3

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
3

### Summary
The paper proposes Concrete Score Distillation (CSD), a discrete score-matching objective for LLM distillation that avoids softmax-induced smoothing and is invariant to constant logit shifts, which are two limitations of common KD and direct-logit losses. It resolves the quadratic cost of pairwise score matching with an analytic, linear-time gradient and provides formal guarantees. Experiments on instruction following and task-specific settings (e.g., summarization, translation) show CSD consistently outperforms recent KD objectives and delivers favorable fidelity-diversity trade-offs.

### Strengths
- The paper proves consistency and that CSD’s optimum set strictly contains DLD’s, leveraging invariance to constant logit shifts

- It derives an Algorithm 1 for keeping compute/memory in line with standard KD

- The method consistently surpasses strong KD baselines across tasks and yields complementary gains when combined with on-policy techniques, suggesting good scalability.

### Weaknesses
- The paper claims linear-time gradients, but lacks systematic reporting of GPU memory, tokens/sec, and wall-clock per step versus KL/SKL/DLD across vocabulary sizes (e.g., 32k to 128k). Adding end-to-end cost curves and scaling ablations would strengthen adoption

- While the paper shows that weights modulate mode-seeking/covering, default choices, schedules, or adaptive schemes are not well-justified; more guidance (or learning the weights) would aid practitioners.

- Since CSD replaces softmax normalization with centered logits, its impact on probability calibration warrants measurement. (Related to shift-invariance.)

### Questions
Can you report throughput, peak memory, and step time for CSD vs. KL/SKL/DLD on matched setups, plus scaling with vocab size? (This directly tests the linear-time gradient claim and Algorithm 1 practicality.)

Can you provide defaults, schedules, or an adaptive/learned weighting (e.g., driven by confidence or gradient variance)?

### Soundness
3

### Presentation
3

### Contribution
3
