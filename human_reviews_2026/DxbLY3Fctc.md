# MoECondenser: Finetuning MoE LLMs with Condenser Experts

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Despite MoE models leading in benchmarks, supervised fine-tuning (SFT) for the MoE architecture remains difficult because its router layers are fragile. Methods such as DenseMixer and ESFT mitigate collapse with dense mixing or auxiliary load-balancing losses, but these introduce noisy gradients that often degrade performance. In preliminary experiments, we systematically removed experts and observed that while certain “super experts” are activated far more frequently, discarding less used experts still leads to notable performance degradation. This suggests that even rarely activated experts encode non-trivial knowledge useful for downstream tasks. Motivated by this, we propose a new auxiliary loss free MoE SFT framework that combines router biases with shared condenser experts. Instead of enforcing balanced activation across all experts, our method leverages bias updates to encourage imbalanced and sparse routing, allowing rarely used experts to become inactive while designating two existing experts as shared condensers that aggregate knowledge from the inactive set without increasing the per-token compute budget. Router stability is maintained entirely through bias updates that regulate token-level and expert-level activation, eliminating the need for auxiliary losses. Experiments on large-scale MoE models demonstrate that our approach outperforms state-of-the-art SFT baselines such as DenseMixer and ESFT, achieving 4%+ gain on both mathematical reasoning and commonsenseQA benchmarks. Pruning and interexpert correlation analyses confirm that our condenser experts aggregate knowledge from the long-tail experts, preserving performance under sparse routing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Expert Condenser, a new framework for fine-tuning Mixture-of-Experts (MoE) language models without auxiliary losses. The method introduces bias-driven sparse routing combined with two always-active condenser experts that aggregate knowledge from rarely used experts. The method stabilizes fine-tuning by adjusting router biases only.

### Strengths
The paper begins with a strong empirical motivation in Section 2: although “super experts” dominate activation in MoE models, even rarely activated experts carry indispensable knowledge. This observation is novel. I believe this finding could be valuable for future improvements of MoE models.

The proposed Expert Condenser introduces bias-driven sparse routing without auxiliary losses, avoiding the gradient noise. I think the method is technically reasonable.

The proposed method outperforms both ESFT and DenseMixer on multiple MoE architectures (Qwen, DeepSeek, OLMoE).

### Weaknesses
I think there is a missing link between the analysis in Section 2 and the proposed method in Section 3. It would be better to elaborate on how the findings from Section 2 motivate or inspire the specific design choices made in Section 3. 

The bias-update mechanism and condenser aggregation are mainly motivated empirically; the paper lacks formal theoretical grounding or ablation to isolate why condenser experts effectively “absorb” knowledge.

### Questions
Are the Type-G and Type-B experts different only in terms of their gating mechanism?

How does the routing strategy of MoE differ between pre-training and post-training settings? It seems that the proposed method is not theoretically connected to SFT. Is this method applicable to the pre-training stage as well?

### Soundness
3

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
2

### Summary
Motivated by the observation that even rarely activated experts encode non-trivial knowledge useful for downstream tasks, this paper propose a new auxiliary loss free MoE SFT framework that combines router biases with shared condenser experts.

### Strengths
- Novel and Well-Founded Motivation: This paper claims that long-tail of rarely-used experts is "indispensable". This is convincingly supported by the scaling-law analysis in Section 2, which shows that pruning experts (even a small amount) leads to a substantial, non-recoverable performance gap. This provides a solid empirical foundation for the paper's method.

- The Expert Condenser framework is a clever solution to the stated problem. Rather than fighting router collapse with noisy auxiliary losses , the method embraces and induces a "controlled collapse" via bias updates. The idea of designating specific experts to "condense" knowledge from the inactive tail  is an elegant way to preserve information while maintaining sparsity.

### Weaknesses
- Unclear Role of "Type-G" Experts: Section 3.2 introduces "Type-G (Ungated) Shared Experts" , which are visualized in Figure 2. These experts are described as standard feed-forward layers applied to every token. Their role is confusing and seems disconnected from the main "condenser" idea, which relates to the "Type-B" experts. It is unclear if these are new parameters added during SFT, and their introduction complicates the paper's claim of not increasing the per-token compute budget.

### Questions
Could you clarify the exact selection mechanism for the two Condenser Experts? Based on Appendix M.2 , it appears the correct method is to select the two lowest-bias experts, which aligns with Figure 2 but contradicts the text in Section 3.2.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents Expert Condenser, a auxiliary-loss-free framework for post-training Mixture-of-Experts (MoE) large language models. Instead of enforcing balanced expert activation (as in ESFT or DenseMixer), the authors propose bias-driven sparse routing that encourages rarely useful experts to become inactive, while introducing two always-active “condenser experts” that aggregate gradients and knowledge from inactive experts. This aims to preserve long-tail expert knowledge and stabilize routing without adding auxiliary losses. Experiments on DeepSeek, Qwen, OLMoE, and GPT-OSS models show 4–7% accuracy gains over prior methods on math and commonsense reasoning benchmarks.

### Strengths
- The results in Table 1/2/3 is clearly outperform other baselines.

- The system efficiency in section 5.3 provide convining support for the proposed methods.

- The experiments covers multiple base models as well as various tasks.

### Weaknesses
- The paper lacks ablation studies isolating the individual contributions of different components in the proposed method, such as the bias term and the shared (condenser) experts.

- The idea of shared experts is not new; prior MoE works have already demonstrated their effectiveness. For instance, DeepSeek-MoE explicitly employs shared experts: "Shared Expert Isolation: we isolate certain experts to serve as shared experts that are always activated, aiming at capturing and consolidating common knowledge across varying contexts."

- Section 2 feels unnecessary, as it is natural to me that removing experts would lead to performance degradation. Even dropping less-activated experts can hurt accuracy, as shown in prior works such as https://arxiv.org/pdf/2504.05586.

- Figure 1 contains excessive empty space and would benefit from a more compact and informative redesign.

- The term “base model” in Tables 1–3 is unclear. It appears to refer to results without SFT, but improvements over such a base model are not particularly meaningful. It would be more informative to compare against a vanilla SFT baseline to properly contextualize the gains.

### Questions
Please refer to the weakness part.

### Soundness
2

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
This paper proposes ExpertCondenser, a training-free auxiliary-loss-free framework for supervised fine-tuning (SFT) of Mixture-of-Experts (MoE) large language models. Its core motivation lies in the overlooked value of rarely activated "long-tail experts" in MoE models—existing methods either ignore these experts or introduce noisy gradients via auxiliary losses. ExpertCondenser reconciles sparse routing and knowledge preservation by leveraging router bias updates to drive inactive experts into dormancy and designating two shared condenser experts to aggregate knowledge from dormant experts. Experiments show that it outperforms state-of-the-art baselines (DenseMixer, ESFT) by over 4% on mathematical reasoning and commonsense QA benchmarks, achieving 2.87× inference speedup and retaining performance under sparse routing.

### Strengths
1. Strong compatibility as a post-training module, compatible with various MoE architectures (GPT-OSS, DeepSeek, Qwen, OLMoE) without modifying the core model structure.

2. Balanced performance and efficiency, achieving both performance gains (4%+ on key benchmarks) and computational advantages (2.87× speedup over DenseMixer) via parameter offloading optimization.

### Weaknesses
1. I'm not sure how the results reported in the submission are fair. At least, it is fair to re-implement all baselines and fine-tune them with the same dataset. let alone running into tune and finding the best hyperparameters for all methods. The authors motioned in the paper-"We re-implemented ESFT and DenseMixer following the reported setups in (Yao et al., 2025), and reuse their best-reported results when available.", which is contradictory. It is not an apple-to-apple comparison if we reuse results from other papers, as the training datasets are different. 

2. The cost of condensing MoE is not new to the community. At least this paper has explored this concept: https://arxiv.org/pdf/2412.00069. Probably, I could find more related works if I searched them entirely.

3. Condensing expert knowledges to certain shared experts may limit the learning capacity of the model, since for different tasks there may be different crucial experts, and the condensing may limit the variability of MoE. Have the authors tested the generalization performance of this method? i.e. training on mathematical reasoning datasets and evaluate on commonsense benchmarks.

4. One of the claims in the paper is that the proposed method achieves better stability in training and expert selection. Could the authors provide supporting results (i.e. training details, loss curves compared with baselines) to further support this claim?


5. Regarding the auxiliary-free training strategy: the selection of experts considers the $\beta$ parameter, but the gating weight does not. Could the authors provide reasons of this design? What would be the results of also using $\beta$ in computing the weights? And how different is the selected experts when using $s_i + \beta_i$ for selection compared to only using $s_i$ (i.e. can be shown with rank metrics with different criteria)?

### Questions
If the downstream task domain differs significantly from the pre-training domain (e.g., pre-training on general text but fine-tuning on legal documents), will the knowledge aggregated by condenser experts fail to adapt, leading to performance degradation?

### Soundness
2

### Presentation
2

### Contribution
2
