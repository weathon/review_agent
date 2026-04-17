# Flatter Tokens are More Valuable for Speculative Draft Model Training

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Speculative Decoding (SD) is a key technique for accelerating Large Language Model (LLM) inference, but it typically requires training a draft model on a large dataset. We approach this problem from a data-centric perspective, finding that not all training samples contribute equally to the SD acceptance rate. Specifically, our theoretical analysis and empirical validation reveals that tokens inducing flatter predictive distributions from the target model are more valuable than those yielding sharply peaked distributions. Based on this insight, we propose flatness, a new metric to quantify this property, and develop the Sample-level-flatness-based Dataset Distillation (SFDD) approach, which filters the training data to retain only the most valuable samples. Experiments on the EAGLE framework demonstrate that SFDD can achieve over 2$\times$ training speedup using only 50\% of the data, while keeping the final model's inference speedup within 4\% of the full-dataset baseline. This work introduces an effective, data-centric approach that substantially improves the training efficiency for Speculative Decoding. Our code is available at https://github.com/fjm9933/Flatness.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper reveals that tokens inducing flatter predictive distributions from the target model are more valuable than those yielding sharply peaked distributions. Based on this insight, this work introduces flatness, a new metric to quantify the quality of training tokens, as well as the Sample-level-flatness-based Dataset Distillation (SFDD) approach, which filters the training data to retain only the most valuable samples. Experiments on the EAGLE framework demonstrate that SFDD achieves over 2 training speedup using only 50% of the data, while keeping the final model's inference speedup within 4% of the full-dataset baseline.

### Strengths
1. The manuscript is clearly written, with a well-structured narrative, compelling motivation, detailed analyses, and transparent demonstrations that enhance its readability and impact.
2. The motivation of this work is well demonstrated. Existing speculative decoding (SD) approaches such as Eagle series, medusa, and Hydra fine-tunes on all training tokens and treat these tokens equally. Investigating the importance of training tokens to SD is quite an important research question.
3. Through empirical analysis, this work reveals that tokens inducing ≈er predictive distributions from the target model are more valuable than those yielding sharply peaked distributions. The empirical experiments are well designed and clearly demonstrated, with detailed formulations. This research insight may inspire future SD research.
4. The proposed Sample-level-flatness-based Dataset Distillation (SFDD) approach is simple yet effective, which further highlights the utility of the flatness score in practice.

### Weaknesses
1. **The necessity of training efficiency**: The major concern of this work is the training efficiency of SD methods. As we know, training efficiency is also a highlighted point in the paper of the Eagle series. In such an SFT setting with only a small amount of training data, the training overhead of SD is quite small, e.g., the training of EAGLE is completed in 1-2 days on 4x A100 (40G) GPUs. The training of EAGLE on 7B, 13B, and 33B models can even be conducted on an RTX 3090 node in 1-2 days. Considering this, the value of inference efficiency improvement may be much higher than the training efficiency of SD.
2. Some errors that could be fixed in the manuscript:
   - Hinton et al. (2015) should be fixed to \citep in Lines 48-49.
   - insert blanks before citation in Lines 84-89.

### Questions
Please check the weakness part above. In general, this manuscript is well written with valuable research insights. I recommend the publication of this work in the venue.

### Soundness
3

### Presentation
4

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
This paper finds that in training-based speculative decoding, tokens whose target-model distributions are “flatter” (i.e., closer to uniform) contribute more significantly to improving the acceptance rate. Building on this insight, the authors propose SFDD (Sample-level-flatness-based Dataset Distillation), a data selection strategy that trains the draft model only on samples with high flatness. This approach reduces training cost while preserving inference speedup. Experiments show that SFDD outperforms other data importance metrics such as entropy and perplexity.

### Strengths
1. The paper provides a clear, theoretically motivated insight that tokens with flat target distributions are more valuable for SD training, which goes against conventional wisdom in standard KD.
2. The proposed flatness metric is simple, target-model-only, and computable offline. It consistently outperforms multiple baselines across diverse tasks and data retention ratios.
3. SFDD offers a plug-and-play method to significantly reduce training cost for SD without architectural or loss-function changes, which is highly relevant given the growing adoption of SD in LLM serving.

### Weaknesses
1. The Gaussian approximation is elegant but may not fully capture the heavy-tailed, sparse nature of real LLM output distributions. The robustness of conclusions to this modeling choice could be better addressed.
2. While flatness is shown to outperform entropy empirically, the paper does not deeply analyze why cosine similarity to uniform is superior to entropy as a flatness proxy since both measure distributional spread.
3. Although SFDD reduces the training cost of the draft model, it requires a full forward pass over the entire training dataset using the target model to compute flatness. When the target model is significantly larger than the draft model, this data selection step may actually introduce additional computational and time overhead.
4. Some experimental settings are unclear, e.g. the pretrained model used for the draft model and the hyperparameters employed during training.
5. The y-axis label in Figure 2(b) is missing a closing parenthesis “)”.

### Questions
1. The paper claims that low-flatness tokens “saturate quickly”. Could this be quantified (e.g., by showing their gradient norms or loss reduction curves over training steps)?
2. Why use cosine similarity instead of more commonly used metrics for measuring the discrepancy between two probability distributions, such as KL divergence?
3. The paper adopts a sample-level filtering strategy but does not investigate whether a token-level filtering approach would be effective. Specifically, during training, could masking out low-flatness tokens in the loss computation further reduce computational overhead without degrading performance?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on improving the training efficiency of drafter model training via dataset selection in a speculative decoding setup. Based on an analysis involving Gaussian distributions, the paper proposes a **flatness** metric to assess the value of a token towards the drafter LM training. The paper then aggregates the per-token flatness values to obtain a sample (sequence ) level metric for the data selection. The paper then performs experiments by using EAGLE-2 training pipeline with LLaMA3-8B-Instruct to showcase the utility of the proposed *Sample-level-flatness-based Dataset Distillation* (SFDD) method towards improving the training efficiency for the drafter model while preserving most of the inference speedup via speculative decoding.

### Strengths
- The paper aims to develop a theoretical underpinning for the data selection for draft model training. 
- The proposed method, namely SFDD, strikes a good trade-off between training efficiency and inference speedup across multiple datasets. SFDD outperforms other selection criteria from the literature when selecting 50% of the available data from the draft LM training.
- The paper provides ablation studies that highlight the utility of SFDD as one varies the fraction of data selected for the drafter LM training.

### Weaknesses
- The theoretical analysis in the paper is based on the assumption that the underlying distributions are Gaussian, which could be far from the discrete distributions produced by an LM. Did the authors consider working with other distributions such as Exponential and Half-normal distribution.
- The empirical evaluation in the paper is a bit limited. The authors may want to expand their empirical study by exploring more LLMs and training datasets.
- The paper does not provide insights on why other natural selection measures such as entropy do not provide good results (see Questions section as well) as compared to the proposed flatness metric. Adding such an analysis would strengthen the contributions of the paper.

### Questions
- Figure 1a shows that the reduction in $\Delta L\_1$ is proportional to $\sigma$. Since entropy for the Gaussian is a function of $\sigma$, why did the authors not consider entropy as the metric of interest? 
- In Line 252, the paper say ``....we cannot directly compute "continuous variance"``. Did author consider working with variance of the discrete distribution produced by target LM itself as the selection criterion?
- Did you consider other robust aggregation approaches in Eq (8) such as median. Would the conclusions from Table 1 hold with such robust aggregation metrics as well?
- Why did you not include Top-1 probability -- the second best performing baseline in Table 1 -- for your ablation studies in Table 2?
- From Table 1, it appears that the speedup for Random degrades more gently as one decreases the Retain Ration. Did you consider extreme Retain Ratios such 5% and 10%?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The paper introduces a new metric called target model flatness to improve the efficiency of training draft models for speculative decoding.
- Demonstrates that using only 50% of the dataset achieves performance comparable to training on the full dataset.
- Outperforms other baseline methods in experiments.

### Strengths
- Novel metric (target model flatness) that could provide better guidance for draft model training.
- Significant data efficiency: achieves similar performance with half the data.
- Shows better results than existing baselines, indicating practical impact.

### Weaknesses
- Experiments are limited to Llama3.1-8B-Instruct; unclear if results generalize to other model sizes or families.
- Missing discussion of related work on efficient draft model training (e.g., [1] Goel et al., 2024).
- Lack of clarity on training details (epochs, convergence criteria).

Some tables are confusing:
- Table 1 speed-up discrepancies vs. acceptance length raise concerns about hardware or redundancy:
  - For GSM8K, acceptance lengths for No Filter, SFDD, and PPL are 3.28,2.95,and 2.79 respectively.
  - The gap between No Filter and SFDD is 0.33, while between SFDD and PPL is 0.16.
  - However, the speed-up gap between No Filter and SFDD is only 0.02, while between SFDD and PPL is 0.33.
  - This seems inconsistent: a larger acceptance length difference should not result in a smaller speed-up gap. This seems like a possible hardware mismatch or redundancy for non-SFDD methods.

- Table 2 shows unexpected performance drop at 60% retention for MTB and NQ compared to 50%.
- Mathematical proof (Eq. 5) has ambiguity in computing $\sigma_r^*$​ given its dependence on $\sigma_r$​ through $\tau$.

Reference: [1] Goel, Raghavv, et al. "Direct alignment of draft model for speculative decoding with chat-fine-tuned llms." arXiv preprint arXiv:2403.00858 (2024).

### Questions
- Please see weakness section
- Can acceptance plots and target entropy plots be added to Fig. 2? Is there a relationship between entropy and flatness?
- What is the role of acceptance criteria in the top-left of Fig. 3, seems extra?

### Soundness
2

### Presentation
2

### Contribution
2
