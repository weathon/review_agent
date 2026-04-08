## Human Reviewer 1

### Summary
This paper studies the problem of identifying valuable preference data for aligning large language models (LLMs) with human preferences. The authors argue that data quality should be viewed as a model-dependent property, not intrinsic to the data itself, and further introduce the Truncated Influence Function (TIF) to evaluate the per-sample effect of preference pairs on validation performance, showing that “medium-IF” data tend to yield the best alignment results.

To make this computation feasible at scale, they propose two lightweight proxies—Loss Difference (LossDiff) and Implicit Reward Margin (IRM)—and a combined rule LossDiff–IRM for preference data selection. Experiments on various LLM models as well as alignment methods show that the proposed selection could achieve better alignment with about half of the dataset, improving winrate over full-data baselines.

### Strengths
1. This paper is well-motivated. The idea that data valuation is model-dependent is insightful and challenges a long-standing assumption in RLHF and preference optimization pipelines.


2. The proposed Truncated Influence Function (TIF) seems new in the literature and provides a principled mechanism to quantify individual data influence. 

3. Results span several LLM families, datasets, and alignment methods, demonstrating generality and robustness.

### Weaknesses
1. The “medium-IF” hypothesis is interesting, but primarily from the empirical observations. It would be better if the authors could elaborate more in a theoretical or formal way. 

2.  It seems a commonly used benchmark, arena-hard, is not included. Can authors elaborate more on it?

3. (Mirror Issue) The presentation could also be improved. For example, the captions of Figure 1 and Figure 3 refer to the materials in the appendix. From my personal perspective, it might be better to put the related paragraphs back into the main paper.

### Questions
Please see the weakness section.

In summary, this paper provides a well-motivated perspective on preference data valuation, supported with solid empirical evidence. I currently tend to recommend acceptance. However, I'm willing to re-evaluate this work according to the further dicusssions.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
The paper reframes preference-data quality as model-dependent and introduces a Truncated Influence Function (TIF) lens showing that medium-IF pairs—not very small or very large ones—drive the most stable alignment gains. To avoid the high cost of exact IF on LLMs, the authors propose two forward-pass proxies—Loss Difference (LossDiff) and Implicit Reward Margin (IRM)—and a combined LossDiff–IRM selector that closely tracks TIF.

### Strengths
1. **Principled, model-aware data valuation.** The paper grounds selection in a TIF-motivated view of *model-dependent* data value and shows that combining **LossDiff** and **IRM** outperforms either proxy alone; moreover, pairs discarded by the selector are empirically low-value for alignment, confirming the criterion’s discriminative power.  

2. **Strong empirical generality.** Across models, objectives (DPO/SLiC), and benchmarks, the method achieves higher win rates using reduced data (e.g., 64% subset with top performance).

### Weaknesses
1. **Incomplete positioning vs. recent work.** The paper discusses Filtered DPO, but does not engage with several highly relevant baselines:

   * [1] margin-based preference selection for alignment quality (ICML 2025),
   * [2] RS-DPO (rejection sampling + DPO for cleaner preference data),
   * [3] active preference learning for LLMs (querying informative pairs instead of passively filtering).
     These works target the same core problem — selecting / curating high-value preference pairs — and should be compared both conceptually and empirically.

2. **Practicality.** The proposed selectors still require rescoring large volumes of pairs with both the current model and an auxiliary model. The paper does not report the real cost (GPU hours, throughput) or show that this is cheaper/more scalable than RS-DPO-style sampling or active preference acquisition.

3. **Robustness.** The “medium-IF is best” claim is convincing on the reported setups but is not stress-tested across broader domains, model sizes, or stages of alignment; it is unclear how stable this curriculum is outside the presented benchmarks.

[1] Larger or Smaller Reward Margins to Select Preferences for Alignment? ICML 2025.  
[2] Rs-dpo: A hybrid rejection sampling and direct preference optimization method for alignment of large language models. NAACL 2024.  
[3] Active preference learning for large language models. ICML 2024.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes a novel data selection method for preference data, inspired by two points:

1. Preference data selection based on influence function (IF);

2. DPO formulation implicitly encodes the reward difference, which can be leveraged for better estimation of IF.

Through preliminary experiments, the authors reveal an interesting phenomenon: data with medium IF values (i.e., within a truncated IF range) yields the best performance. They further propose a new data selection strategy to avoid the heavy computational cost of IF estimation, and demonstrate positive experimental results.

### Strengths
1. The idea is novel and surprising. Traditionally, it is believed that maximizing the margin (i.e., the loss difference) for single-step optimization is most beneficial for learning. However, this paper shows that truncated medium IF actually performs best.

2. The authors also carefully consider the high computational cost of IF computation in practice and propose an approximation strategy.

3. In the preliminary experiment, both DPO training and validation reward margins are computed on subsets from UltraFeedback, which share similar distributions, while in the main experiments, OOD situations are also tested, and the core design of this paper works on both situations, proving its value.

### Weaknesses
1. Although the analysis is thorough, the method itself is relatively simple, and the level of technical innovation is somewhat limited.

2. The proposed LossDiff-IRM takes the intersection of two selection criteria. However, model performance also depends on data scale. If the intersection yields a very small subset, it may pose a risk of data scarcity, which should be addressed.

3. Since pairwise preference data and DPO-style training are becoming less common, the strong coupling between the proposed data selection and model completions might limit its applicability to more recent paradigms such as RL-based methods. It would be helpful to include a discussion on how such approaches might extend to the prompt-level selection process.

### Questions
How are the thresholds set? I did not find sufficient details in the paper regarding the choice or tuning of these thresholds.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper studies the data quality of preference pairs for LLM alignment.
The authors begin with influence function analysis of each preference pair's impact, which measures the consistency of datapoint's gradient with the overall gradient of validation set.
The analysis shows the medium-IF preference pairs are more valuable, so the authors design the TIF metric: $\mathbb I[\delta_{low} < IF < \delta_{high}]$ to select medium-IF data for training.
Furthermore, to reduce the computation cost of IF, the authors introduce two simplified data quality metrics:
1. LossDiff: the difference of loss between $\pi_\theta$ and a validation model $\pi_{val}$;
2. IRM: the implict reward margin.

Both metrics correlate well with IF and can be used to approximate TIF for data selection.
Empirical results on different models, DPO variants, and benchmarks demonstrate the effectiveness of the proposed data selection methods in improving alignment performance.

### Strengths
1. **Clear Motivation and Well-Justified Methods**: Data quality metrics based on model state serve as an intuitive and reasonable approach to improve preference optimization performance. The paper's derivation from influence function analysis to practical data selection metrics is well-motivated and clearly presented.
2. **Stable Improvements**: The proposed data selection methods consistently enhance alignment performance across different models, DPO variants, and benchmarks, demonstrating their robustness and effectiveness.
3. **Comprehensive Ablation**: The paper includes several ablation studies to provide further insights including model-specific selection, noisy robustness, and optimized layers.
4. **Presentation Quality**: The paper is well-written and easy to follow, with clear explanations of the proposed methods and experimental results.

### Weaknesses
My main concern with this work is the lack of comparison with prior model-specific data selection methods.

The authors state in L56-58:
> Our above analysis suggests a reasonable yet seldom-discussed viewpoint: preference data selection
should be performed for specific models and explicitly related to the training process

However, there are already several works discussing model-specific data selection. For example, implicit margin based selection methods [1,2,3,4] also take the model state $\pi_\theta$ into consideration when selecting data. 
So I think it would be better not to claim that this is a "seldom-discussed viewpoint".

Moreover, [3,4] utilize IRM to select data in a similar way: They prioritize preference pairs with the smallest absolute IRM value, i.e., medium-IRM data, which is similar to the TIF metric proposed in this paper.
So it would be better to cite and compare with these prior works.

---

**References**

[1] Morimura, Tetsuro, et al. "Filtered Direct Preference Optimization." Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.\
[2] Deng, Xun, et al. "Less is more: Improving llm alignment via preference data selection." arXiv preprint arXiv:2502.14560 (2025).\
[3] Huang, Kexin, et al. "Larger or Smaller Reward Margins to Select Preferences for LLM Alignment?." Forty-second International Conference on Machine Learning. \
[4] Yang, Sen, et al. "Not All Preference Pairs Are Created Equal: A Recipe for Annotation-Efficient Iterative Preference Learning." Findings of the Association for Computational Linguistics: EMNLP 2024. 2024.

### Questions
Can you provide more comparison with prior model-specific data selection methods?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4