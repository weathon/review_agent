# Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Large language models (LLMs) achieve strong performance across diverse tasks but suffer from high inference latency due to their autoregressive generation. 
Speculative Decoding (SPD) mitigates this issue by verifying candidate tokens from a smaller draft model in parallel, yet its strict exact-match verification discards many semantically valid continuations.
We propose Training-Free Loosely Speculative Decoding (FLy), a novel method that loosens the rigid verification criterion by leveraging the target model’s own corrective behavior to judge whether a draft–target mismatch remains semantically valid. 
FLy introduces a two-tier mechanism: an entropy-level gate that identifies whether the current token allows multiple plausible alternatives or is nearly deterministic, and a token-level deferred window that distinguishes genuine errors from differently worded yet semantically correct variants. 
To further reduce latency, we design a multi-level acceleration strategy that accelerates not only the target model but also the drafter itself.
Owing to its training-free design, FLy composes seamlessly with arbitrary draft–target pairs and generalizes across models and domains without hyperparameter re-tuning. 
Experiments show that FLy preserves $\geq$99\% of the target model’s accuracy while achieving an average 2.81$\times$ speedup on Llama-3.1-70B-Instruct and 5.07$\times$ speedup on the 405B variant. Notably, on out-of-domain datasets, our method remains highly effective and outperforms the training-based method EAGLE-3 by 1.62$\times$. 
Our code is available at
https://github.com/AMD-AGI/FLy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Training-Free Loosely Speculative Decoding (FLy), which relaxes verification by accepting semantically correct drafts without additional training. Experiments show that FLy preserves ≥99% of the target model’s accuracy while achieving a considerable speedup.

### Strengths
1. This paper presents a clearly written narrative and well-structured exposition that is easy to follow.

2. The two-tier mechanism that relaxes verification for worded-differently yet semantically valid alternatives is reasonable, simple to implement, and empirically effective.

### Weaknesses
1. My primary concern is that with tree-structured verification, the paper’s targeted optimization cases may be diluted; this risk should be explicitly analyzed and validated with focused experiments.

2. The main table integrates PLD’s multi-level acceleration while comparing against methods not adapted with hierarchical speculative decoding; this may be an unfair comparison.

3. The related-work section mischaracterizes training-free SPD: LayerSkip requires training, whereas the earliest training-free approaches are Self-SD, Draft & Verify, and their extensions SWIFT and KNN-SSD; this should be corrected.

4. Please add insights on task sensitivity to relaxed verification: which tasks are less sensitive (and typical improvement gaps between non-relaxed vs. relaxed), which are more sensitive, and why. This would substantively enrich the paper’s contribution.

5. The paper contains interesting ideas; releasing code would make the work more promising and facilitate community adoption and verification.

### Questions
Please refer to the weakness.

### Soundness
2

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
4

### Summary
The paper introduces Training-Free Loosely Speculative Decoding (FLy), a speculative decoding method that relaxes the strict exact-match verification criterion in standard SPD. FLy accepts semantically correct draft tokens even if they do not exactly match the target model’s output, thereby increasing the number of accepted tokens per round and improving speedup. The method uses a two-tier mechanism: an entropy-level gate to determine whether a mismatch is in a high-uncertainty context, and a token-level deferred window to monitor future tokens for signs of error correction. FLy is training-free, model-agnostic, and includes a multi-level acceleration strategy that also speeds up the draft model. Experiments show strong speedups (up to 5.07× on 405B models) while preserving ≥99% of the target model’s accuracy.

### Strengths
- The proposed approach Fly introduces a novel, training-free speculative decoding mechanism that accepts semantically valid mismatches.
- Extensive experiments, ablations, case study and cross-model evaluations demonstrate robustness and effectiveness.
- The paper is clearly written, with intuitive explanations and strong visual aids.
- FLy addresses a real bottleneck in LLM inference and offers a clever, general-purpose acceleration method.

### Weaknesses
- For Figure 1, it is hard to tell whether Fly is more robust against OOD datasets as the degration rate on speedup is similar.
- While outperforming training-based methods on OOD data is a key result, the paper notes that FLy is slightly slower than the heavily optimized EAGLE-3 on ID datasets. An in-depth discussion or analysis of this performance gap beyond the expected trade-off for "training-free" would improve the technical quality of the paper.
- The Multi-level acceleration technique with PLD is borrowed from literature. Its technical contribution is not significant.
- The paper may need a quantitative breakdown of how often FLy’s deferred acceptance mechanism is triggered, and what fraction of accepted tokens are due to semantic mismatch tolerance rather than exact matches. Reporting the percentage of tokens where FLy overrules vanilla SPD and the resulting speedup contribution would directly substantiate the claim that loose verification is the key driver of acceleration. Without this, it remains unclear how much of the speedup is due to FLy’s core innovation versus other factors (e.g., larger K, MLA).
- The ablation studies in Table 4 vary only one hyper-parameter at a time (W, θ, K), while keeping others fixed. This one-at-a-time approach is insufficient for exploring the joint effect of hyper-parameters. A grid search over the hyper-parameter space is necessary to ensure that the reported configurations are globally optimal, not just locally tuned.

### Questions
- The reproducibility of the work may need improvement through open-source sharing. Would you provide your source code?
- Regarding the performance of JudgeDecoding in the paper, please detail the reproduction of the results.
- How do you explain the extremely high recovery rates (≥99.5%, including multiple 100%) reported in Figure 3, Tables 4, 6, and 7? Given that FLy explicitly accepts semantically mismatched tokens, some degradation in task-level accuracy is expected.
  - What is the standard deviation across multiple runs?
  - How many test samples were used per dataset?
  - Could you provide confidence intervals or bootstrap estimates to demonstrate that these values are statistically stable, rather than artifacts of small test sets or coarse metrics?
- Please address the issues mentioned in "Weakness".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies speculative decoding with lossy verification, a topic that has recently garnered some attention in the community. The authors investigate training-free approaches that enable the target model to accept tokens it would usually reject under the standard regime, while maintaining high output quality. This is achieved through a two-stage process as follows: First, an entropy gate is designed, where the rejection of tokens that would get rejected according to the target logits, is deferred in case the entropy of the target distribution was very high, suggesting that multiple tokens could potentially be used instead of solely the most likely one. The deferred token is then put on “probation”, i.e. if another rejection within a certain window occurs within the continuation of said token, then it will be rejected. If no other rejections occur, then the token is accepted. This follows from the insight that target models tend to correct mistaken tokens in subsequent steps, making the likelihood of another rejection significantly higher in case the token truly should have been rejected.

The authors then evaluate their approach in the setup of draft / target speculative decoding, with models of size roughly 8B serving as the draft for targets of size 70B and 405B. The proposed approach is evaluated on a range of benchmarks and compared against recent models such as Eagle-3. On out-of-distribution datasets, the novel approach outperforms the baselines and remains competitive in-distribution as well, while maintaining the output quality.

### Strengths
1. Lossy verification in speculative decoding is a timely topic that enables more efficient decoding for large models. Making progress in this area is very relevant for the field to further push the efficiency gains.
2. The paper is very nicely written and easy to understand. The proposed approach is very well motivated and the experiments are thorough, which is needed to assess its validity. 
3. The technique doesn’t require any training and any model can easily be enhanced with this approach, making it easy for users to experiment with it for their own model.

### Weaknesses
1. The out-of-distribution numbers are quite strong but in-distribution, training-based approaches like EAGLE-3 still keep the upper hand, with quite a discrepancy. What seems a bit problematic however in this experiment design is that different speculative decoding techniques are compared; standard small / large model speculation (Fly) and self-speculation (EAGLE-3) where the draft model is a very small model on top of the features. Usually, more involved like tree-attention are also involved. Thus enhancing small / large model speculation with the new techniques might not be the best comparison. Why not also equip EAGLE-3 with the same techniques as introduced in this paper, or am I missing an incompatibility? Indeed, the approach presented here does outperform the standard small / large model speculation setup without these enhancements.
2. I could not find any tokens/s metrics, arguably the most important metric in LLM inference. What frameworks are the experiments based on in the main paper? I found some results in vLLM in the appendix but they’re unfortunately very limited. I find experiments in this setting way more important as they actually reflect real scenarios more accurately. Many conclusions in the literature have changed in the past when moving to more optimized frameworks, so it would be great to verify that the proposed approach remains superior even in more optimized settings like in vLLM.

### Questions
1. When looking for later mistakes in the window, you seem to not apply the entropy rule anymore, i.e. any direct mismatch counts as a mistake within the window, correct? 
2. The section on Multi-Level Acceleration confused me a bit initially. Of course when drafting longer candidate sequences with high acceptance rates, the generation process starts to spend most time on drafting. But from a pure speculative decoding perspective, this is of course inevitable (in the extreme case we draft the entire sequence once and accept everything with the target in a single forward pass). I would state more clearly that MLA is an idea to further speed-up the entire process, but it’s not a mechanism to improve the speculative decoding setup you’re studying, it’s a way to speed-up drafting itself. Or did I misunderstand something?
3. Can one equip EAGLE-3 with your augmentations to enable it to have lossy verification? In general, does the approach extend to any speculative decoding setup?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose a novel method to accelerate LLMs via speculative decoding. In particular, they relax the rigid assumption of verification in classic speculative decoding and propose a two-tier mechanism. The method comprises first an entropy gate to evaluate whether a mismatch should be directly rejected, and then a token-level defer window to find whether the mismatch is benign. The efficacy of the method is showcased on multiple benchmarks and across different draft and target models.

### Strengths
- The main idea is novel and clearly presented.

- The studied problem is relevant and the proposed solution appropriately motivated.

- There is a plethora of experimental evidence across different models that support the efficacy of the proposed method.

- The proposed method is benchmarked against mutliple SOTA baselines and showcases superior performance while requiring no training.

- The comparison with learning-based approaches, i.e., judge-decoding, is insightful and highlights the capabilities of the proposed method.

- I appreciate the thorough ablations on cross-model drafting.

### Weaknesses
Although I do not see any major flaws, there are some issues that I believe would worth addressing:

- I am missing some discussion and/or analysis of the shortcomings and failure modes of the proposed model.

- For completeness, it would be interesting to see what are the speedups achieved by the proposed and judge-decoding in Fig. 3.

- I am missing some related work, e.g., [1, 2]

 [1]. Learning harmonized representations for speculative sampling

 [2]. Glide with a cape: A low-hassle method to accelerate speculative decoding

### Questions
I would appreciate if the authors address the questions raised in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
