# Direct Token Optimization: A Self-contained Approach to Large Language Model Unlearning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
Machine unlearning is an emerging technique that removes the influence of a subset of training data (forget set) from a model without full retraining, with applications including privacy protection, content moderation, and model correction. The key challenge lies in ensuring that the model completely forgets the knowledge of the forget set without compromising its overall utility. Existing unlearning methods for large language models (LLMs) often utilize auxiliary language models, retain datasets, or even commercial AI services for effective unlearning and maintaining the model utility. However, dependence on these external resources is often impractical and could potentially introduce additional privacy risks. In this work, we propose direct token optimization (DTO), a novel self-contained unlearning approach for LLMs that directly optimizes the token level objectives and eliminates the need for external resources. Given a sequence to unlearn, we identify two categories of tokens: target tokens, which capture critical knowledge for unlearning, and the remaining non-target tokens, which are crucial for maintaining the model utility. The former are used to optimize the unlearning objective, while the latter serve to preserve the model's performance. The experimental results show that the proposed DTO achieves up to 16.8$\times$ improvement in forget quality on several benchmark datasets than the latest baselines while maintaining a comparable level of model utility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Direct Token Optimization (DTO), a self-contained method for unlearning knowledge from fine-tuned large language models (LLMs) without using retain datasets, auxiliary models, or external APIs. The core idea is to identify “target tokens”—tokens most responsible for encoding the forget-set knowledge—using a new metric called delta-score, which measures the influence of each prefix token on predicting the suffix. DTO performs gradient ascent on these target tokens to erase memorized knowledge, while preserving general utility by minimizing KL divergence on the remaining non-target tokens. Experiments on TOFU and MUSE benchmarks show that DTO achieves significantly higher forget quality than prior methods such as FLAT, DPO, NPO, and LLMU, while maintaining reasonable utility, all under a more realistic resource-constrained setting.

### Strengths
1. The method does not rely on retain data, auxiliary models, or external APIs; it only requires the forget set, making it more practical.
2. The method performs targeted, token-level updates, focusing only on tokens that encode forget-related knowledge. This minimizes collateral damage to unrelated knowledge and helps preserve the model’s overall capabilities.
3. The experimental evaluation is thorough, spanning multiple datasets and metrics, and provides convincing evidence of both the method’s effectiveness and its reliability.

### Weaknesses
1. The idea of token-level optimization is not entirely new and appears to be an incremental refinement of TPO [1], rather than a fundamentally different approach.
2. The paper lacks analysis of computational complexity and does not provide experiments on spatial or temporal efficiency (e.g., training time, GPU memory usage), which makes it difficult to assess the method’s scalability.
3. Although the target token selection is improved, treating all remaining tokens uniformly as “utility-preserving language tokens” seems overly simplistic. This assumption ignores that some of these tokens may also contribute to task-relevant knowledge, and thus model utility may not be preserved in a principled way.
4. The method relies heavily on the assumption that forget-related knowledge can be localized through prefix–suffix structures. This may fail when answers are short phrases instead of complete sentences, when the expected ordering of prefix and suffix does not hold, or when linguistic structure is more complex. Moreover, the effectiveness of target token selection is sensitive to hyperparameters, such as the prefix length and the defined delta-score ratio.

[1] Zhou, X., Qiang, Y., Zade, S. Z., Zytko, D., Khanduri, P., & Zhu, D. (2025). Not All Tokens Are Meant to Be Forgotten. arXiv preprint arXiv:2506.03142.

### Questions
1. Have you evaluated this approach when applied to other unlearning methods such as NPO [1] or SimNPO [2]? If the method is compatible in principle with forget-only optimization, how does it perform in those settings?
2. What is the time and computational complexity of the proposed method? Specifically:
(1) What is the complexity of identifying target tokens?
(2) What is the additional training overhead of optimizing only on these tokens?
(3) Are there experiments measuring training time, GPU memory usage, or scaling behavior?
If target token selection is a core component, more detailed ablations or efficiency analyses would be valuable.
3. How would this method extend to multimodal LLMs (MLLMs)? When visual information is integrated (e.g., via cross-attention with images), does target token selection need to be adapted? Would the prefix–suffix assumption still hold in such a setting?

[1] Zhang, R., Lin, L., Bai, Y., & Mei, S. (2024). Negative preference optimization: From catastrophic collapse to effective unlearning. arXiv preprint arXiv:2404.05868.
[2] Fan, C., Liu, J., Lin, L., Jia, J., Zhang, R., Mei, S., & Liu, S. (2024). Simplicity prevails: Rethinking negative preference optimization for llm unlearning. arXiv preprint arXiv:2410.07163.

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
5

### Summary
This paper introduces Direct Token Optimization (DTO), a self-contained method for machine unlearning in large language models (LLMs). Unlike prior work that depends on auxiliary models, retain datasets, or external AI services (e.g., ChatGPT) to remove the influence of a “forget set,” DTO performs unlearning directly on the original model using only the forget data. The core idea is to identify target tokens—those most responsible for encoding the memorized knowledge—using a novel metric called the delta-score, which measures the impact of perturbing each token on the loss of subsequent tokens. DTO performs gradient ascent on these target tokens to “forget” them while preserving general linguistic ability by minimizing KL-divergence between logits of non-target tokens in the unlearned and original models.

Experiments on the TOFU and MUSE benchmarks demonstrate that DTO achieves up to 16.8× improvement in forget quality compared to recent baselines (e.g., FLAT, DPO, NPO), while maintaining comparable utility. The method shows strong empirical performance without relying on any external resources, making it practical and privacy-preserving.

### Strengths
1. Novel and Practical Approach. DTO is the fully self-contained LLM unlearning method that does not require retain sets, auxiliary models, or external AI services. The token-level optimization framework and the introduction of the delta-score for selecting influential tokens are conceptually intuitive.
2. Balanced Optimization Design. The two-stage optimization—gradient ascent on target tokens and KL minimization on non-target tokens—provides a clear mechanism to balance unlearning and utility.
3. Comprehensive Experiments and Clear Gains. Evaluations across TOFU and MUSE benchmarks are extensive and demonstrate significant improvements in forget quality while maintaining reasonable model utility. Ablation and parameter studies (e.g., Figure 2–3) effectively illustrate the effects of target ratio k and suffix ratio on the unlearning trade-off.

### Weaknesses
1. Limited Theoretical Grounding. The paper provides little theoretical analysis of delta-score properties or convergence guarantees of the optimization. The method remains primarily empirical and intuitive.
2. Scalability and Generalization. Experiments are limited to small and mid-sized models (LLaMA 2–7B, LLaMA 3.2–3B). It is unclear whether DTO scales efficiently to larger production-scale LLMs (e.g., 70B+).
3. Heuristic Hyperparameter Choices. The selection of target token ratio (k%) and suffix ratio is based on manual tuning. The method may require domain-specific adjustments, reducing general applicability.
4. Limited Qualitative Analysis. Although quantitative metrics are strong, more qualitative examples of “before-and-after unlearning” would help demonstrate DTO’s linguistic preservation and knowledge removal in practice.
5. Some relevant prior works should be included and discussed [1-3].


[1] Liu, Sijia, Yuanshun Yao, Jinghan Jia, Stephen Casper, Nathalie Baracaldo, Peter Hase, Yuguang Yao et al. "Rethinking machine unlearning for large language models." Nature Machine Intelligence (2025): 1-14.

[2] Tang, Haoyu, Ye Liu, Xi Zhao, Xukai Liu, Yanghai Zhang, Kai Zhang, Xiaofang Zhou, and Enhong Chen. "Learn while unlearn: An iterative unlearning framework for generative language models." arXiv preprint arXiv:2407.20271 (2024).

[3] Patel, Gaurav, and Qiang Qiu. "Learning to unlearn while retaining: Combating gradient conflicts in machine unlearning." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4211-4221. 2025.

### Questions
1. Is there a way to adaptively determine the k% target ratio or suffix ratio during training rather than manual tuning?
2. See the weaknesses.

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
This paper introduces Direct Token Optimization (DTO), a self-contained unlearning method for fine-tuned large language models that avoids reliance on external resources such as retain sets, auxiliary models, or commercial LLM services. DTO directly performs targeted token-level gradient ascent on a model's internal representations to forget sensitive content while minimizing KL divergence on non-target tokens to preserve general utility. A key innovation is Delta-score, an assistance-free token selection strategy inspired by sequence memorization studies. It scores each token in a prefix based on how much its perturbation affects the loss over the suffix tokens. Tokens with the highest Delta-scores are treated as knowledge-bearing and become targets for unlearning, while the rest are regularized to maintain fluency and model performance.

### Strengths
1. DTO requires only the original model and forget set, making it fully compliant with privacy laws and practical for deployment in restricted data environments.

2. By identifying high-impact tokens via Delta-score, DTO performs fine-grained unlearning, allowing selective removal of knowledge without damaging broader model capabilities.

### Weaknesses
1. The paper does not clearly explain how the threshold q is selected or how token scores are computed for the entire sentence, particularly for tokens where r > q. A more detailed description of the scoring mechanism and token selection criteria is needed to ensure reproducibility and clarity.

2. The proposed method is only evaluated on LLaMA-based models. To demonstrate broader applicability, it should also be tested on other popular open-source models, such as Gemma and Qwen. Without this, it is difficult to assess the approach's generalizability across architectures.

3. The method fails to achieve competitive model utility performance compared to other state-of-the-art unlearning techniques. 

4. The paper claims that linguistically meaningful but semantically uninformative tokens tend to have low per-token loss due to frequent exposure during pretraining. However, the analysis without an empirical study does not demonstrate how the proposed method helps. Providing evidence for this claim is essential, as it underlies the core motivation of the approach.

### Questions
Could the authors provide a more detailed explanation of how the threshold q is selected and how token scores are computed and applied across all tokens in a sentence, particularly for those with rank r>q?

Could the authors conduct experiments on other widely used models, such as Gemma and Qwen, to demonstrate the proposed method's generalizability? 

The paper hypothesizes that linguistically important but semantically uninformative tokens tend to have low per-token loss due to frequent pretraining exposure. Could the authors include more quantitative experiments (e.g., token-level loss distributions, category-wise analysis, or case studies) to substantiate this claim and show how the proposed method addresses it?

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
This paper proposes Direct Token Optimization (DTO), a self-contained unlearning method for fine-tuned LLMs that eliminates dependence on retain datasets, auxiliary models, or external AI services. DTO introduces delta-score to identify “target tokens” encoding critical knowledge by measuring how prefix token perturbations affect suffix generation.

### Strengths
1. Table 1 provides a taxonomy of existing LLM unlearning methods, clearly demonstrating that most prior work relies on impractical assumptions (retain datasets, auxiliary models, or commercial AI services).

2. The delta-score metric is an interesting adaptation of memorization analysis. Unlike prior work requiring ChatGPT or human annotators, this approach is self-contained and theoretically motivated by identifying tokens that most influence memorized suffix generation.

### Weaknesses
1. The paper uses reported TPO results rather than running experiments under identical conditions, undermining claims of superiority. Can you run TPO under identical conditions for fair comparison?

2. KL regularization helps on TOFU dataset but gets worse results for MUSE-Books, contradicting the method’s core design. This inconsistency is unexplained.
3. No analysis of when/why gradient ascent on target tokens succeeds, how KL regularization provably preserves utility, or what guarantees the method provides.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
