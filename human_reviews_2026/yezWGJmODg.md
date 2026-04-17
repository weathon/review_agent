# Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
Reinforcement learning (RL) has emerged as a powerful post-training paradigm for large language models (LLMs), yet its effectiveness varies significantly across base models. While incorporating a pre-RL supervised fine-tuning (SFT) phase can enhance RL training, key questions remain: how long should the SFT cold-start phase last, and is the SFT objective truly aligned with the requirements for effective RL preparation?
In our analysis of cold-start dynamics, we uncover a key limitation: the SFT checkpoint with the highest evaluation performance often fails to maximize RL potential due to distributional forgetting—a phenomenon where the model drifts excessively away from the base model’s distribution even before traditional overfitting occurs. We identify diversity metrics, such as the entropy and self-BLEU, as more reliable early-stopping criteria than the standard performance-based checkpoint selection. Our findings show that SFT checkpoints with peak diversity consistently lead to superior post-RL results. Building on these insights, we introduce Adaptive Early-Stop Loss (AESL), a lightweight and dynamic cold-start method that balances the acquisition of new patterns with the preservation of the base model's distribution. AESL operates at both the token and subsequence levels, providing finer-grained control over the cold-start process. Experimental results on mathematical reasoning benchmarks demonstrate that diversity-based early stopping surpasses traditional performance-based SFT, while AESL further enhances RL preparation. By steering LLMs toward better initialization points for RL, AESL consistently achieves superior final performance compared to existing SFT and cold-start strategies. The
code is publicly available at \url{https://github.com/LXXXXR/AESL}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors show that when doing SFT before RL, the checkpoint with the best validation performance often isn't the best starting point for RL. The authors find this happens because models drift too far from the base distribution before traditional overfitting occurs. They propose two solutions: (1) using diversity metrics (entropy, self-BLEU) for checkpoint selection instead of validation performance, and (2) Adaptive Early-Stop Loss that balances learning new patterns while preserving the base model's distribution.

### Strengths
1. The observation that best SFT performance does not translate to best RL readiness is valuable and practically very relevant.
2. The paper is well-motivated and easy to follow.

### Weaknesses
1. I feel that in this paper, severely limited scope threatens generalizability of the results. For example, only mathematical reasoning tasks tested that too only on Qwen model family (both 7B variants are closely related). There's no evidence this generalizes to code, instruction-following, creative writing, etc. or is applicable to a different model family like Llama 3 8B.

2. The AESL weighting function in equation 4 appears ad-hoc. Why softmax? Why this specific ratio? Why average log-probability in the denominator? The entropy decomposition (appendix A) shows a decomposition exists but doesn't strongly motivate this particular functional form. Moreover, there's no comparison to simpler alternatives (linear weighting by confidence, threshold-based stopping loss, etc.) An ablation comparing 3-4 different weighting schemes would greatly strengthen the paper.

3. [Minor] There's no significance testing despite small margins (many improvements are 1-2 points in Tables 1-4).

4. AESL requires additional forward passes for computing weights. How much does this add to training time? Is the improvement worth the computational overhead?


Overall, in its current form, this paper feels more like a useful engineering trick for a specific setting (math reasoning with Qwen) rather than a fundamental contribution to post-training research. I'd encourage the authors to either (a) substantially expand the scope to other domains to demonstrate generality, or (b) provide much deeper analysis of the specific math/Qwen setting to get broader insights.
Right now, the contribution in this paper feels incremental for ICLR but with significant additional work, this could become a solid accept.

### Questions
1. I am curious about under what conditions does AESL underperform standard approaches? Are there cases where preserving base distribution hurts?
2. How does AESL compare to simply doing less SFT (fewer epochs)? Is the sophisticated weighting necessary or would early stopping at lower loss achieve similar results?
3. Is this approach applicable when use a model which hasn't been RL'd before? Qwen-Math already has been RL'd before. For example, Olmo-2 provides checkpoints right after pre-training.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper finds SFT checkpoints with best evaluation performance fail for RL due to distributional forgetting. Diversity metrics are better early-stopping criteria. Furthermore, it proposes AESL, balancing new pattern learning and base distribution preservation, outperforming baselines in math reasoning tasks.

### Strengths
(1) It pinpointed the core misalignment between traditional SFT and RL preparation, uncovering "distributional forgetting"
(2) Developed AESL with token/subsequence-level adaptive weighting.
(3) Conducted in-depth mechanism analysis and reported multiple metrics, enhancing result interpretability and credibility.

### Weaknesses
(1) While the paper identifies "distributional forgetting" as a key bottleneck, it lacks a deep investigation into its underlying causes.
(2) It remains unproven whether AESL’s effectiveness translates to other domains (e.g., code generation, scientific writing).

### Questions
(1) Could you provide experimental evidence to verify whether AESL’s mechanism of balancing new pattern learning and base distribution preservation generalizes to other domains where RL post-training for LLMs is critical?
(2) Could you conduct layer-wise parameter analysis or data-specific ablation studies to discuss the root mechanisms of distributional forgetting?

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
This paper explores the best strategy for fine-tuning a base LLM checkpoint in order to achieve the best performance after RL fine-tuning. Currently, when RL fine-tuning is used to improve a reasoning LLM's ability to solve math or coding problems, a short supervised fine-tuning (SFT) is done on a base checkpoint. One reason for this, as mentioned in the paper, is to provide the base LLM with useful reasoning patterns (that the base model does not have, or is quite weak on), which are further enforced via RL.

A key finding of the paper is that the commonly used metrics such as accuracy on downstream task during SFT does not correspond to the best performance after RL fine-tuning. In fact, the paper shows that lower accuracies during SFT (somewhat surprisingly) leads to the best performance after RL fine-tuning. The paper then proposes to use diversity metrics such as entropy or self-BLEU scores to pick the best checkpoints during the SFT stage, in order to maximize performance after RL. 

The paper also proposes a new method of doing SFT before RL fine-tuning that, by using an adaptive weighting on the loss function, maintains diversity of the SFT model while also learning the necessary reasoning patterns for RL. Experiments show that the method improves over existing SFT + RL baselines on math benchmarks, for the Qwen class of models.

### Strengths
1. Empirically showing that commonly used metrics for SFT are not optimal for RL fine-tuning. This is an interesting finding that is somewhat counter to existing conventional wisdom. 

2. Proposed new methods for SFT fine-tuning target. The paper proposes to use diversity metrics such as entropy or self-BLEU to select checkpoints during SFT before RL fine-tuning. This is a well-motivated idea (maintain diversity to avoid model collapse and allow more exploration during RL) and seems to be supported by empirical evidence.

3. Proposed a change to the standard cross-entropy loss during SFT in order to optimize for diversity metrics while allowing the model to learn new useful reasoning patterns.

4. Experimental evidence for the effectiveness of the proposed method, and ablations.

### Weaknesses
1. Lacks analysis of the complexity of the proposed loss. How much additional complexity does the weighting require? Are there significant memory requirements to handle the logits for each training data sequence?

2. Lack of diversity in experiments. Only the Qwen class of models is used in experiments. Why are there no other open-source, small-sized models? The evaluations are also only done on math benchmarks. It is difficult to be convinced of the generality of the proposed method when the breadth of results are quite limited.

3. Lack of details of baselines. What are GEM and PSFT? You especially mention that GEM "emphasizes promoting diversity" and that PSFT is "concurrent work". How similar are they to the method proposed in this paper?

### Questions
Appendix A, equation (7)-(8), what are the \pi(s_t|s_{t-1})'s, are they deltas functions?

Page 5, line 260, what does the index j index over?

Table 1, why are there no error intervals? How many seeds were used?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Reinforcement learning fine-tuning of LLMs typically involves a SFT phase that primes the model for more efficient RL training. In this context, this paper studies the question of how to select the right SFT checkpoint to initialize RL training from. The authors claim that diversity metrics (entropy, self-BLEU) are better correlated with final RL performance than standard CE loss, and propose an adaptive early-stop loss (AESL) for SFT that consists of weighted CE where the weight is designed to scale decreasingly with both the likelihood of the target token and the likelihood of the prefix sequence.

### Strengths
This paper studies the important yet under-studied question of how SFT checkpoint selection influences final RL performance. This is especially critical because all modern LLM training pipelines involve SFT and RL phases for nearly all tasks, including reasoning and human alignment. The paper is clearly written and the proposed method is easy to understand.

### Weaknesses
I found the empirical motivation and evidence to be a bit lacking. In section 3, the authors claim that diversity has a clear correlation with final performance, but only present results for two (related) models and a single SFT dataset. More broadly showing that this is the case on more models (especially non-Qwen ones), different datasets, and different tasks (e.g. RLHF) would immensely strengthen the claims in the paper. 

I am also not strongly convinced by the argument presented in Figure 3, and would also like to see an ablation of the denominator in the loss weight.

### Questions
1. Cold start seems like the wrong term to use here - warm-start is more appropriate. I know Deepseek-R1 paper also uses this term improperly, but it should be corrected here.
2. Table 5 should include vanilla SFT as well. Along with varying t_scaling, I would also like to see a sweep with the average prefix log prob term removed.

### Soundness
2

### Presentation
2

### Contribution
2
