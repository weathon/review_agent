# SeedPrints: Fingerprints Can Even Tell Which Seed Your Large Language Model Was Trained From

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Fingerprinting Large Language Models (LLMs) is essential for provenance verification and model attribution. Existing methods typically extract post-hoc signatures based on training dynamics, data exposure, or hyperparameters—properties that only emerge after substantial training. As a result, prior evaluations largely focus on lineage verification after fine-tuning, where detection is considerably easier, potentially giving a false sense of safety.
In contrast, we propose a stronger and more intrinsic notion of LLM fingerprinting: **SeedPrints**, a method that leverages random initialization biases as persistent, seed-dependent identifiers present even before training begins. We show that untrained models exhibit reproducible prediction biases induced by their initialization seed. Although weak in magnitude, these biases remain statistically detectable throughout training, enabling high-confidence lineage verification.
Unlike prior techniques that are unreliable before convergence or vulnerable to distribution shifts, **SeedPrints** remains effective across all training stages and robust under domain shifts and parameter modifications. Experiments on LLaMA-style and Qwen-style models demonstrate seed-level distinguishability and enable birth-to-lifecycle identity verification akin to a biometric fingerprint. Evaluations on large-scale pretrained models and fingerprinting benchmarks further confirm its effectiveness under prolonged training and realistic deployment scenarios.
Together, these results suggest that initialization itself imprints a unique and persistent identity on LLMs, forming a true ``Galtonian'' fingerprint.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates whether large language models (LLMs) inherit statistically detectable traits from their random initialization seeds, a phenomenon termed “Galtonian fingerprints.” The authors propose SeedPrints, a training-free statistical framework that traces model lineage at the seed level by measuring correlations across output distributions. Systematic experiments on 58 models spanning six major LLM families (LLaMA, Qwen, Mistral, Gemma, Phi, and OLMo) show that token preference biases established at initialization remain stable, even after diverse training and deployment interventions such as Instruct tuning, finetuning, PEFT, quantization, model merging, and distillation. These findings indicate that a model’s identity is partially determined at birth (initialization) rather than solely during training, providing a practical tool for model provenance analysis.

### Strengths
The paper proposes a statistical method for detecting and characterizing persistent fingerprints left by random initialization seeds in LLMs.
Specifically,  the authors introduce the novel idea that random seeds can leave stable, measurable fingerprints that persist through training. The findings reveal that seed-level fingerprints remain robust despite continued training across diverse datasets and objectives.

### Weaknesses
1. The paper empirically demonstrates the effect but does not fully explain why seed-level bias persists after optimization.   
2. Reliance on Kendall-Tau significance may be unstable for small sample sets; sensitivity analysis (sample size vs. detection power) should be more explicit.
3. The motivation for using the Kolmogorov-Smirnov (KS) statistic is unclear.

### Questions
1. Could the authors provide a theoretical or intuitive explanation for why seed-induced bias persists after long training? Is it due to early optimization path dependency or weight-space geometry?  
2. How many samples are required to detect a fingerprint reliably (e.g., 95% confidence)? Please quantify this relation to support practical use.    
3. Please conduct experiments on Llemma-7b[c1] in Table 5, because the robustness of the proposed method against large-scale data fine-tuning needs to be verified.
4. The motivation for using the Kolmogorov-Smirnov (KS) statistic needs to be clarified. In such a setting, REEF underperforms ICS, which is contrary to common sense.



[c1] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631, 2023.

### Soundness
3

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
This paper proposes SeedPrints, a stronger and more intrinsic notion of LLM fingerprinting. The SeedPrints methodology is built on a key empirical observation: randomly initialized LLMs exhibit subtle but reproducible token selection biases that are unique to their initialization seed. These biases persist throughout training, creating a detectable signature. Experiments show that SeedPrints remains effective across all training stages and robust under domain shifts or parameter modifications.

### Strengths
- The idea of seed-based intrinsic fingerprinting is interesting.
- The proposed method is validated through extensive experiments.
- It demonstrates a certain degree of practical application value.

### Weaknesses
- Some descriptions in the paper are unclear. For example, the terms "min/max" and "top/bottom" appearing in Figure 1 are not explained.
- There is a lack of explanation as to why models with different seeds possess different "identity indices," which is somewhat counter-intuitive. Is it theoretically possible for two models, initialized with different seeds, to have statistically indistinguishable SeedPrints due to some coincidence?
- The paper does not discuss potential evasion strategies. For instance, if a malicious actor, understanding how SeedPrints work, might attempt to develop methods to "erase" the model's original fingerprint or "forge" a fingerprint to make it appear to originate from a different seed.
- Although the paper mentions its inapplicability to black-box models, it must be acknowledged that this limits its application to some extent.
- There is a typo: in line 136, "10001st" should be "1025th".

### Questions
- In Figure 1 (top-right subplot), why is the baseline at -0.0003?
- In lines 178-179, why is the mean output vector calculated? Will the mean output vector differ for different values of n, thereby affecting identity indices?
- In practical applications, how is the size of m determined? Do different m values affect the stability and discriminative power of the fingerprint? Should the value of m vary for different models based on their size and vocabulary?
- In Table 5, how does the t-test result change? Based on the U-test results, it seems that the correlation (if any) weakens as the number of parameters increases.
- In Algorithm 1, is it possible for no intersection of identity indices to exist?

### Soundness
3

### Presentation
2

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
The paper identifies an interesting phenomenon in model training: randomly initialized language models exhibit some bias in their token predictions and this bias persists even after training. This signature can be leveraged to form the basis of a fingerprinting test, which can test whether a given model has statistically significant correlation with the bias detected in a given randomly initialized model. The key innovation is the introduction of _identity indices_, continuous embeddings which have more persistent signatures throughout training.

### Strengths
1. The paper makes use of a scientifically and technically interesting signature that is present in LLMs. To my knowledge, this is the first time this particular signature has been documented.
2. The introduction of the _identity indices_ is a clever innovation which allows the test to focus on a statistical signature which is more stable (across training).

### Weaknesses
1. The test itself is a bit complicated. This, in itself, is not a significant weakness, but it does makes it difficult to determine if the test is correct or not. The paper itself does not justify why the null distribution used is correct. In particular, due to the selection used to build the intersection $T$, the random variables in $\mathcal{T}$ are not independent. Thus, it seems odd to me that the null distribution treats each simulated index as being independent. I'm not saying the test is incorrect, but I do think some justification here is necessary, given the complexity of the method.

### Questions
1. The paper describes an identity index with a low mean $\bar{g}_j$ to be "unwilling", but in the case of the final embedding, where the dimensions are signed and there is no distinction between "true" and "non-true" classes, it's not clear why choosing with the argmin makes sense here.
2. How many parameters did the LLaMA/Qwen-style models have? 
3. There is an opportunity here to show that this technique persists across scales by leveraging the checkpoints published during the training of the OLMo/OLMo 2 family of models. This is a very interesting experiment, since it shows that the signature can persists during pretraining at the scale of trillions of training tokens without having to train such models from scratch for this work.
4. In Table 2, the negative p-values are on the order of 1e-10 or larger while in Figure 2 they are on the order of 1e-50 or smaller. Why are the p-values so different in these two settings?
5. The results in Figure 1 suggest that token level biases also persist through training, although the rank correlation is quite small. Do you think this could be converted into an effective test by increasing the number of samples until the p-value becomes small enough? If it can be made to work, the token-level setting is desirable because it can be implemented without access to the continuous embeddings of the model.

### Soundness
2

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
4

### Summary
SeedPrints fingerprints LLMs using their random initialization rather than training artifacts. Untrained networks already show reproducible token selection biases set by the seed, and these signatures persist through training. A simple statistical test recovers model lineage with high confidence and stays robust across training stages, domain shifts, and moderate parameter edits, with strong results on LLaMA and Qwen families and public benchmarks.

### Strengths
1. Finding of the biometric fingerprint is interesting, where initialization creates stable token selection biases that are seed-specific and persist throughout training, giving each model a seed-level identity.
2. The authors find an effective algorithm by constructing identity indices from the most down-weighted output dimensions on random inputs, then run a rank-based correlation test with calibrated p-values to decide shared lineage.
3. The experiment setting is comprehensive, which verifies lineage at every checkpoint and stays accurate under domain shift, finetuning, PEFT, quantization, merging, and distillation, outperforming similarity-based baselines.

### Weaknesses
1. Within the same model family, how similar are fingerprints across sizes. For example, what is the similarity between Qwen2.5 7B and 14B, and does similarity grow with parameter sharing or architectural reuse, and will they be distinguishable?

2. For models like deepseek distill llama or deepseek distill qwen, which fingerprint do they most resemble? The teacher, the student base, or a new hybrid, and does SeedPrints remain reliable in these distillation settings

3. The SeedPrints relies on logits or hidden states, which many closed-source APIs do not expose. Are there practical black box surrogates to test for seed-level fingerprints, for example, using only top-k outputs or sampled continuations with controlled prompts

4. Rank-based tests may be sensitive to large vocabularies and temperature scaling. Could authors add more analysis of sensitivity to temperature, alternative normalizations before ranking, and report how these choices affect power and false positive rates

5. In the experiment, the comparisons emphasize p-values. Maybe add more effect sizes, confidence intervals for Kendall tau, could improve the interpretability and calibration

### Questions
Please see the weakness part. I will consider raising the score if the authors address the weaknesses/problems clearly

### Soundness
3

### Presentation
3

### Contribution
3
