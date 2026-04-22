# Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 6, 6, 6, 4, 4

## Abstract
Math reasoning has become the poster child of progress in large language models (LLMs), with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as math leaderboards improve week by week, it is worth asking: do these gains reflect broader problem-solving ability or just narrow overfitting? To answer this question, we evaluate over 20 open-weight reasoning-tuned models across a broad suite of tasks, including math, scientific QA, agent planning, coding, and standard instruction-following. We surprisingly find that most models that succeed in math fail to transfer their gains to other domains. To rigorously study this phenomenon, we conduct controlled experiments on Qwen3-14B models using math-only data but different tuning methods. We find that reinforcement learning (RL)-tuned models generalize well across domains, while supervised fine-tuning (SFT)-tuned models often forget general capabilities. Latent-space representation and token-space distribution shift analyses reveal that SFT induces substantial representation and output drift, while RL preserves general-domain structure. Our results suggest a need to rethink standard post-training recipes, particularly the reliance on SFT-distilled data for advancing reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the transferability of reasoning abilities gained from math-specific fine-tuning to other, non-mathematical domains. Through a large-scale audit of existing models and controlled experiments on Qwen models, the authors argue that Reinforcement Learning (RL) is superior to Supervised Fine-Tuning (SFT) for achieving transferable gains. While SFT on math data is shown to cause catastrophic forgetting of general capabilities, RL-based tuning improves math performance while preserving, and even enhancing, performance on other reasoning and non-reasoning tasks. The authors support this claim with in-depth analyses of latent space representations (PCA shift) and token-level distribution shifts (KL divergence), concluding that on-policy RL is a more robust method for improving reasoning in a generalizable way.

### Strengths
1. **Important Research Question:** The paper tackles a critical and timely problem. Understanding whether specialized training on benchmarks like MATH translates to broader capabilities is crucial for guiding future research in LLM development.
2. **In-depth Mechanistic Analysis:** The primary strength of the paper lies in its diagnostic approach. The use of PCA shift, KL divergence, and gradient norm analysis provides compelling evidence for why RL and SFT exhibit different behaviors. This analysis of internal model dynamics is insightful and moves the conversation beyond surface-level performance metrics.
3. **Strong Empirical Evidence (within its scope):** The controlled experiments and the ablation study provides actionable insights, particularly highlighting the importance of on-policy sampling for achieving better generalization.

### Weaknesses
1. **Significant Overlap with Prior Work and Limited Novelty:** The central finding of the paper, that SFT tends to memorize while RL promotes generalization, has been substantially explored in prior work [1]. Notably, Chu et al. (2025), in their paper titled "SFT Memorizes, RL Generalizes," demonstrate this exact phenomenon across both textual and visual reasoning tasks. While the current paper is cited, its contribution is not sufficiently distinguished. Chu et al. already established the core dichotomy using a broader set of tasks (arithmetic and navigation) and modalities. The main novelty of this paper is thus reduced to a mechanistic analysis of a previously identified phenomenon within the specific context of math reasoning transfer.
2. **Over-reliance on the Qwen Model Family:** The controlled experiments are conducted exclusively on Qwen models. This is a major limitation, as recent work by Shao et al. (2025) [2] has shown that Qwen models exhibit anomalous behavior during RLVR, improving even from spurious or random rewards. This raises serious concerns that the observed superiority of RL in transfer might be an artifact of the Qwen architecture or its pre-training data, rather than a fundamental property of the RL algorithm itself. The absence of validation on a different model family, such as Llama, significantly weakens the generalizability of the paper's conclusions.
3. **Insufficient Discussion of Limitations and Scope:** The paper lacks a dedicated limitations section. Beyond the model family bias, other unaddressed limitations include:
    - Source-Domain Specificity: The study is confined to math as the source domain for fine-tuning. It remains unclear whether these findings would hold for other specialized domains like law, medicine, or code, which might have different transfer properties.
    - Nature of SFT Data: The SFT data is generated via distillation from a stronger teacher model. This setup differs from SFT on human-annotated, ground-truth data and may introduce confounding variables related to the teacher's style, which could contribute to the observed distributional shifts and catastrophic forgetting.
4. **Missed Nuance in the RL vs. SFT Dichotomy:** The paper frames RL and SFT as a binary choice. However, it overlooks a more nuanced perspective, such as that presented in "RL Squeezes, SFT Expands" (Matsutani et al., 2025) [3]. This work suggests that RL is inherently more conservative, "squeezing" the solution space to a few reliable paths, while SFT "expands" it with new paths from a teacher. From this viewpoint, RL's superior transferability might be a consequence of its conservatism (it doesn't break what works because it learns less that is new), while SFT's failure is a risk associated with its greater plasticity. The paper would be stronger if it engaged with this trade-off instead of presenting a one-sided conclusion.



[1] Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., ... & Ma, Y. (2025). Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161.

[2] Shao, R., Li, S. S., Xin, R., Geng, S., Wang, Y., Oh, S., ... & Zettlemoyer, L. (2025). Spurious rewards: Rethinking training signals in rlvr. arXiv preprint arXiv:2506.10947.

[3] Matsutani, K., Takashiro, S., Minegishi, G., Kojima, T., Iwasawa, Y., & Matsuo, Y. (2025). Rl squeezes, sft expands: A comparative study of reasoning llms. arXiv preprint arXiv:2509.21128.

### Questions
1. Given the significant conceptual overlap with Chu et al. (2025) [1], could you more clearly articulate the primary novel contribution of your work beyond providing a mechanistic analysis for a known phenomenon?

2. Considering the findings of Shao et al. (2025) [2] on Qwen models, how can you be certain that your conclusions about RL's superior transferability are not specific to this model family? Have you considered replicating even a small-scale version of your controlled experiment on a Llama model?

3. The "RL Squeezes, SFT Expands" framework [3] suggests that RL is more conservative than SFT. Does your analysis of latent space and token shifts support this view? Could RL's preservation of general skills be seen as a side effect of its limited ability to incorporate novel reasoning patterns compared to SFT?

[1] Chu, T., Zhai, Y., Yang, J., Tong, S., Xie, S., Schuurmans, D., ... & Ma, Y. (2025). Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161.

[2] Shao, R., Li, S. S., Xin, R., Geng, S., Wang, Y., Oh, S., ... & Zettlemoyer, L. (2025). Spurious rewards: Rethinking training signals in rlvr. arXiv preprint arXiv:2506.10947.

[3] Matsutani, K., Takashiro, S., Minegishi, G., Kojima, T., Iwasawa, Y., & Matsuo, Y. (2025). Rl squeezes, sft expands: A comparative study of reasoning llms. arXiv preprint arXiv:2509.21128.

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
4

### Summary
This paper examines whether LLMs' progress in math reasoning translates to broader problem-solving abilities or just reflects overfitting. Testing over 20 models, the authors found that math-focused models often fail in other domains. Experiments with Qwen3-14B show that reinforcement learning (RL) preserves general reasoning skills, while supervised fine-tuning (SFT) causes models to lose broader capabilities. The study suggests rethinking training methods, especially SFT, to improve cross-domain reasoning.

### Strengths
1. This paper addresses an intriguing and timely topic: whether the advancements in mathematical reasoning by large language models (LLMs) translate into broader problem-solving abilities or merely reflect task-specific overfitting.
2.The authors conducted extensive experiments across over 20 reasoning-tuned models, exploring their performance not only in math but also in diverse domains such as scientific QA, planning, coding, and instruction-following tasks. The experimental design is thorough and well-executed, providing valuable insights into the limitations of math-focused models, which often fail to generalize their capabilities to other domains. The controlled experiments with Qwen3-14B further highlight the impact of different tuning methods, revealing that reinforcement learning (RL) preserves general reasoning skills better than supervised fine-tuning (SFT).
3.The paper is well-written, with clear explanations of the methodology, results, and implications. 
4. Provide code.

### Weaknesses
While this paper explores an interesting topic and provides a detailed experimental analysis, particularly through controlled studies on Qwen3-14B, its main conclusion—that reinforcement learning (RL) preserves general reasoning abilities better than supervised fine-tuning (SFT)—is already widely recognized within the LLM research community. This observation, while valid, lacks novelty and does not bring new insights to the field.

### Questions
ref Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the transferability of reasoning skills from the mathematical domain to more general capabilities in Large Language Models (LLMs). The authors conduct a large-scale evaluation of over 20 reasoning-tuned models and find that models fine-tuned with Supervised Fine-Tuning (SFT) often fail to transfer their math gains to other domains, and frequently exhibit "catastrophic forgetting" of non-reasoning skills. In contrast, models tuned with Reinforcement Learning (RL) demonstrate superior generalization. Through a controlled study on the Qwen3-14B model, the paper provides strong evidence for this discrepancy. The authors introduce several analyses, including a "Transferability Index" (TI), latent-space PCA, and token-distribution KL-divergence, to show that SFT causes significant representational and output drift, whereas RL better preserves the model's foundational structure. An ablation study on RL components pinpoints the on-policy sampling distribution as the most critical factor for promoting generalization.

### Strengths
1. The study is exceptionally thorough. It begins with a broad audit of over 20 existing models, which motivates the core hypothesis. This is followed by a well-designed controlled experiment that isolates the fine-tuning method (SFT vs. RL) as the key variable. The evaluation spans a diverse set of benchmarks, thoughtfully categorized into "Math Reasoning," "Other Reasoning," and "Non-Reasoning," which allows for a nuanced analysis of transferability.

2. The paper is exceptionally well-written and easy to follow. The central argument is presented clearly from the start and is consistently supported by the experiments. The figures are particularly effective; for instance, Figure 1 and Figure 2 immediately convey the paper's main findings in a powerful and intuitive way.

### Weaknesses
1. The SFT models in the controlled study are trained on data generated via rejection sampling from a teacher model (Qwen3-32B). This distillation process itself might introduce a confounding variable. The performance gap could stem not only from the off-policy nature of the SFT algorithm but also from the properties of the distilled dataset, which may have a narrower distribution or contain subtle artifacts compared to the more diverse data distribution explored by the on-policy RL agent. While the study is well-controlled, it is difficult to completely disentangle the effect of the algorithm from the effect of the data distribution it is trained on.

2. The use of a signed square root to "control extremes" is a non-standard transformation. It would be beneficial to explain why this was chosen over other robust scaling methods (e.g., logarithmic scaling or winsorizing) and to provide an analysis of how this non-linear transformation might affect the relative rankings of the models.

### Questions
1. The ablation study in Table 4 shows that "On-policy SFT" achieves impressive transferability to non-reasoning tasks (TInon = 30.2), significantly outperforming "Off-policy SFT" (-40.5) and even "Off-policy RL" (4.5). This strongly supports your conclusion that the sampling distribution is a critical factor. Could you elaborate on the practical implications of this finding? Does it suggest that a simpler, "iterative SFT" paradigm—where a model is periodically used to generate its own training data—could provide a compelling alternative to full-blown RL, capturing most of the generalization benefits without the associated complexity?
2. The PCA shift analysis aggregates representational changes across all layers into a single distance metric, d(*). Have you investigated whether this representational drift is uniform across the model's depth? For instance, does the large shift induced by SFT primarily occur in the final, task-specific layers, or does it fundamentally alter deeper, more foundational representations as well? A layer-wise analysis could provide a more granular understanding of the mechanism behind catastrophic forgetting.
3. In Appendix A.3.1, you specify a learning rate of 1e-6 for RL and 5e-5 for SFT, a 50-fold difference. While it's common for SFT to use a higher learning rate, could this large discrepancy be a confounding factor in the observed representational drift? Is it possible that the greater shifts seen in SFT models are partially attributable to a more aggressive optimization step, rather than solely to the off-policy nature of the training? Have you experimented with SFT using a lower learning rate, and if so, how did it affect the results?
(4)	Could you provide more detail on the RL reward function? Also, have you tested a sequential SFT-then-RL fine-tuning approach, and how does it affect model generalization compared to using RL alone?
(5)	How would you position Direct Preference Optimization (DPO) in your comparison? Given DPO is also off-policy, would you expect it to suffer from poor generalization like SFT, or would it perform closer to on-policy RL?

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
4

### Summary
This paper examines whether improved mathematical reasoning in LLMs transfers to broader capabilities. Evaluating over 20 open-weight reasoning models across diverse domains, the authors find that math-tuned models often fail to generalize. Controlled experiments on Qwen3-14B show that RL preserves cross-domain reasoning, while SFT leads to catastrophic forgetting. Representation and token distribution analyses reveal that RL maintains a stable internal structure, whereas SFT causes significant drift. The results call for rethinking SFT-dominant post-training practices in reasoning models.

### Strengths
- Comprehensively evaluates reasoning transfer across diverse domains with clear and well-controlled experiments.

- Provides insightful analyses of latent-space and token-level drift, revealing why SFT damages general reasoning while RL preserves it.

### Weaknesses
- The controlled study is limited to the Qwen3 family, reducing generality.

- Evaluation tasks could be broader, and the cost-effectiveness of RL training is insufficiently discussed.

- Some recent studies report that SFT can outperform RL under certain conditions, suggesting the paper’s conclusions may be overly one-sided.

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Researchers evaluate over 20 open-weight reasoning models on diverse tasks (math, science QA, agent planning, coding, instruction-following) to test if math gains generalize or reflect overfitting. Key finding: Most models excelling in math fail to transfer improvements to other domains. Controlled experiments on Qwen3-14B using math-only data show:RL-tuned models generalize well across domains.
SFT-tuned models overfit, losing general capabilities.Analyses reveal SFT causes significant drift in latent representations and output distributions, while RL preserves general-domain structure. Conclusion: Standard post-training (especially SFT with distilled data) needs rethinking to build robust, generalizable reasoning models.

### Strengths
1.Experiments on the effects of math problem for the reasoning ability is conducted.
2.Experiments on Qwen-8B is conducted.

### Weaknesses
1.The effects of solving math problems bring goodness to other reasoning problems is common sense and it widely studied in the community of LLM reasoning. 
2.There is little RL algorithm development work for this topic.
3.The work is mainly on the analysis but the very similar analysis and the very conclusion is already in common sense in the community. 
4.The simple SFT+RL is lack of nolviety.
5.The base model is Qwen-8B, more base model will be good.

### Questions
1.Does the common-sense belief in the LLM reasoning community—that math problem-solving benefits other reasoning tasks—hold true based on broad cross-domain evaluations?
2.To what extent have RL algorithms been specifically developed and explored for improving generalization in reasoning-tuned LLMs?
3.How novel is the analysis in this work, given that similar conclusions about SFT overfitting and RL generalization are already widely accepted as common sense in the community?
4.Does the simple combination of SFT followed by RL offer sufficient novelty as a post-training recipe for advancing reasoning models?
5.Would the findings be more robust and generalizable if the controlled experiments included a wider variety of base models beyond Qwen-8B?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigate whether training LLMs with good math capabilities could improve on broader domains. It finds that the capability transfer dependes heavily on how LLMs are trained. The authors evaluate over 20 open-weight reasoning models on math, other reasoning (science, coding, planning) tasks, shoing that strong math performance often fails to transfer. Some math-specialized models even degrade on general tasks. To study this, the authors fine-tune Qwen3-14B on exactly the same math-only data using either SFT or RL. Both approaches boost math scores, but RL generalize well to other reasoning and non-reasoning benchmarks, whereas SFT models systematically forget general capabilities. Representation and token distribution analyses find that SFT causes large latent and output drift, while RL makes smaller changes that preserve the base model's structure. The ablation shows the importance of sampling distribution, credit assignment and negative gradient.

### Strengths
* The evaluation is broad. The paper evaluates on over 20 open reasoning models on math benchmarks (AIME, ..), other domian-reasoning benchmarks (GPQA, ..) and non-reasoning benchmarks (CoQA, ...). The experiment setting is complete. For example, they compare different training paradigm: Qwen3-14B SFT  (think), SFT (no-think), RL. To study the difference between SFT and RL on math and other domain tasks, they do detailed analysis including token distribution shifts, which provides quantitative numbers to support the claim. The results in the paper seem to be consistent with the claim that RL tends to preserve general abilities better than pure SFT.
* The motivation of the paper is clearly stated in the abstract and introduction. The figures also provide high-level understanding for the paper, e.g., figure 2 shows the transferability of mathematical reasoning models to other reasoning and non-reasoning tasks.
* Many of the current the works use only math scores as proxies for evaluating reasoning capabilities. This paper provides concrete evidence that math-only SFT can hurt general tasks and RL can improve math capability while preserving capabilities on other domains.

### Weaknesses
* The core conclusions from this paper are: (1) Math-only SFT can cause catastrophic forgetting and harm non-math capabilities, (2) RL-style post-training on math tends to preserve (or improve) broader abilities and stays closer to the base model in latent/token space. However, similar themes are discussed in previous papers, e.g., [1-3]. The paper doesn't distinguish what is genuinely new compared to previous works. A lot of the story reads as a strong confirmation of widely observed behavior rather than a clearly novel insight.
* The paper’s analyses of latent PCA shift and token-distribution KL / rank shift are interesting but largely descriptive. They don’t fully connect these diagnostics to concrete training prescriptions beyond the conclusion that RL drifts less. For example, based on these observations and analysis, how can you try to fix the SFT to let LLMs keeps old capabilities?
* The baselines are somewhat limited. The SFT baselines only include SFT-no-think and SFT-think. But more baselines could be added. For example, regularized SFT methods (L2-SP, EWC-like constraints)


References

[1] Chu, Tianzhe, et al. "Sft memorizes, rl generalizes: A comparative study of foundation model post-training." arXiv preprint arXiv:2501.17161 (2025).

[2] Lai, Song, et al. "Reinforcement fine-tuning naturally mitigates forgetting in continual post-training." arXiv preprint arXiv:2507.05386 (2025).

[3] Wu, Yongliang, et al. "On the generalization of sft: A reinforcement learning perspective with reward rectification." arXiv preprint arXiv:2508.05629 (2025).

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
