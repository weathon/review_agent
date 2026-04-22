# Learnability and Privacy Vulnerability are Entangled in a Few Critical Weights

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 2, 6, 8, 4, 6

## Abstract
Prior approaches for membership privacy preservation usually update or retrain all weights in neural networks, which is costly and can lead to unnecessary utility loss or even more serious misalignment in predictions between training data and non-training data. In this paper, we empirically show that only a very small number of weights are liable to membership privacy vulnerability. However, we also identify that those neurons are not only liable to membership privacy breach but also contribute to generalizability. According to these insights, to preserve privacy, instead of discarding those neurons, we rewind only the weights for fine-tuning. We show that through extensive experiments, this mechanism, plugged into other approaches, shows enhanced resilience against Membership Inference Attacks while maintaining utility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper finds that performance impact and privacy vulnerability are entangled and exist in a very small fraction of weights. The authors propose a fine-tuning method that curates only privacy-vulnerable weights to defend against membership inference attacks. Experiments show the performance of this method on three datasets.

### Strengths
1. The proposed question is interesting. The authors argue that performance impact and privacy vulnerability are entangled and exist in a very small fraction of weights. This is a valuable research problem in the membership inference attack field. 

2. The author makes an attempt to confirm the proposed insights by empirical evidence. Based on these observations, this paper proposes a method to mitigate the privacy vulnerability introduced by MIAs.

### Weaknesses
1. The presentation of this paper could be improved. A well-organised structure makes it easier for readers to understand the paper and follow the arguments. This paper lacks readability, making it difficult for readers to grasp the viewpoints the author wants to convey. For example, the authors should introduce the importance estimation in neural networks in the preliminaries (not as a related work) if they use the tools as an analytical method.

2. The paper does not establish a clear logical connection between the three proposed insights, the motivation, and the method. The authors claim the importance of weights stems from their locations rather than their values. However, the TFO method estimates weight importance via magnitudes of gradients and weights. Moreover, why does the proposed method not consider fine-tune weights at particular locations, even though the importance of weights stems from their locations? In addition, the authors claim that most privacy-vulnerable weights impact utility performance. Why does rewinding privacy-vulnerable weights, while freezing these weights, have little impact on performance?

3. The experimental results provided don’t sufficiently substantiate the conclusions drawn in this paper. For an empirical paper, authors should conduct extensive experiments to evaluate the effectiveness of the proposed method, including the various attack and defence methods, different types of datasets, and more models. In addition, the authors should conduct experiments to analyse the performance of the method under different configurations, such as hyperparameters, ablation studies on key components. Moreover, the authors should plot privacy-utility curves to show the privacy-utility trade-offs, rather than reporting only a single point result as in Table 3.

### Questions
1. In Figure 1(b), does the training loss approach zero, suggesting that the model is severely overfitting? Does CE loss represent privacy vulnerability? 

2. How to formally define privacy-vulnerable weight. As described in Section 4, the author uses the weight importance estimation method Eq.2 to estimate privacy vulnerability. Can importance estimation serve as a proxy for privacy vulnerability?
3. Existing paper[1] also propose a membership-privacy-oriented fine-tuning defence method.
4. How to compute the learnability score? Is this the accuracy of models?
5. What does the PCC in Table 1 represent? For example, does a high PCC indicate both high accuracy and a high MIA AUC?
6. Can CWRF be used independently as a defence method?  Why does Table 3 only show the results when CWRF is plugged into other defence methods?
7. The proposed method is similar to pruning techniques. Why are pruning-based defence methods not included in the comparison?

[1] Defending membership inference attacks via privacy-aware sparsity tuning[J]. arXiv preprint arXiv:2410.06814, 2024.

### Soundness
2

### Presentation
2

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
This paper proposes a new algorithm, Critical Weights Rewinding and Finetuning (CWRF), for privacy preservation [without significantly affecting the utility of the model] based on the observation that there is a relatively small fraction of weights that cause privacy leakage in models. They also demonstrate that a large fraction of these privacy-vulnerable weights are critical to the model's learnability. CRWF works by keeping the learnability/ privacy-critical weights intact while fine-tuning the other privacy-invulnerable weights. This work incorporates the concept of importance estimation of weights in a neural network, which is largely used in the context of model pruning (Frankle & Carbin [1]).

### Strengths
- Unlike previous works, instead of data points, the author(s) in this paper are concerned with addressing model-level privacy vulnerability in neural networks. They propose a new metric for privacy-vulnerability estimation for individual weights in a neural network based on Bourtoule et al.'s [2] work on machine unlearning.
- They provide empirical evidence to support their conjectures: (a) Privacy-critical weights are concentrated in the layers responsible for learnability (Section 4.1), (b) there is a correlation between privacy-critical and learning-critical weights (Section 4.2), and (c) privacy-vulnerable weights are also least updated during training (Section 4.4).
- CWRF is modular as it allows plugging in any privacy-preserving approach during the fine-tuning step, as demonstrated in Table 3 by integration of CWRF with other privacy-preserving methods such as HAMP, DP-SGD and others. Furthermore, in some cases, CWRF not only improves resistance to MIAs but also boosts the utility (measured in terms of the test accuracy) of the models. 
- This work also contributes to the field of Machine Unlearning, where the model trainer aims to minimise/ remove the influence of select data points on a trained machine learning model. With CRWF, the author(s) aim to train a model to mitigate weight-level privacy vulnerability without significantly affecting its utility.

### Weaknesses
- In CRWF, per my interpretation of the approach, the model is fine-tuned using the same training dataset as the one used to train $M_{up}$ from scratch in the privacy vulnerability estimation step. However, the author(s) do not clarify this in Section 4.3.
The author(s) do not clarify what privacy parameters, for example, $(\epsilon, \delta)$, were used in the experiments with DP-SGD. 
- In lines 388-389, the author(s) state that they use a total of 8 shadow models for both LiRA and RMIA. While RMIA is said to work well with a low number of shadow models, comparing it against a powerful version of LiRA would necessitate training more than 8 shadow models. Carlini et al. [3] use 256 shadow models. If the author(s) intent is to convey the strength of their proposed privacy-preservation approach (CWRF), they ought to evaluate it using strong attacks and not its weaker versions.
- Lines 404-405 that "It is verified that our approach successfully boosts the performance of all of them." ought to be framed in the context of the performance of CWRF + other privacy-preserving methods against SOTA MIAs.
- Carlini et al. [4] demonstrated the onion-effect wherein removing privacy-vulnerable data points from training exposes earlier invulnerable data points to replace them. Do the author(s) probe whether something similar happens when using CWRF, which involves rewinding privacy-vulnerable weights to their initial values? Could this cause previously privacy-invulnerable weights to become privacy-critical?
- The author(s) do not share the code for the paper.

[1] Frankle, J., and Carbin, M. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. ICLR 2019.

[2] Bourtoule, L. et al. Machine Unlearning.” IEEE S&P 2021.

[3] Carlini, N., et al. Membership Inference Attacks From First Principles. IEEE S&P 2022.

[4] Carlini, N. et al. The Privacy Onion Effect: Memorization is Relative. NeurIPS 2022.

### Questions
**Questions**: I would urge the author(s) to address the weaknesses detailed above.

**Suggestions**:

Author(s) can improve the presentation of the paper by considering the following suggested edits:

- Minor suggestion #1: It might be good to rearrange the Fig 3 and Tab 1's results by decreasing order of PCC for an easy read.
- Minor suggestion #2: The PCC reported for linear layer weights for ResNet18 in Tab 1 does not appear aligned with the results as depicted in 1st row, 4th column of Fig 3. If possible, use a different y-axis to demonstrate the high correlation between privacy vulnerability and learnability for linear layers (~0.8).
- Minor Suggestion #3: I would urge the authors to use the full form of the abbreviated terms, such as HAMP and CCL, in Line 377, when referencing them for the first time. Thereafter, they can use the abbreviation. It would be erroneous to presume a reader knows the abbreviations beforehand. 
- Minor Correction #1: Line 811, Specifically <-> "Specificly".

### Soundness
3

### Presentation
2

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
The paper provides evidence that privacy vulnerability (as quantified by susceptibility to membership inference attacks, or MIAs) is concentrated in certain locations/weights of a model. Additionally, these weights are also those that correspond to learnability/generalizability. With these insights, the authors propose a method that rewinds and freezes these privacy-vulnerable weights and fine-tunes the other unfrozen weights with (potentially private) information with privacy-preserving approaches such as DP-SGD and more recent approaches. They further corroborate their claims with an extensive array of results across different percentages of rewound weights, different methods, ablating which weights are rewound/fine-tuned, etc. along with an analysis of privacy-utility tradeoff for their method and the reduction in attack success observed for different state-of-the-art MIAs (LiRA and RMIA).

### Strengths
**[S1]** Provides valuable insights on which weights correspond to MIA vulnerability and how they are also largely the same as the ones that are most important to generalizability. The methods used to identify such weights are sound and convincing.

**[S2]** Insights provided in Fig. 5 to motivate the design of CWRF are interesting; it clearly demonstrates why it is key to both rewind privacy vulnerable weights and fine-tune privacy-invulnerable (so to speak) weights. It is also interesting how fine-tuning privacy-invulnerable weights, even if they might not be most correlated with learnability, helps significantly with achieving a good privacy-utility tradeoff/test accuracy. In other terms, the observation that the location of these weights, and not their values, is what matters is fascinating, albeit not very intuitive.

**[S3]** Demonstrates the effectiveness of the proposed method (CWRF) when paired with multiple prominent differentially private training algorithms.

**[S4]** Uses the strongest possible attacks to test their proposed defense against. In addition, they report attack success in low FPR regions, which is absolutely essential to effectively communicate privacy attack success rates (as argued by [1]).

**[S5]** In-depth descriptions of the hyperparameters, libraries/software, and hardware used are provided, providing confidence in the reproducibility of the work.

All in all, this paper appears to provide a valuable contribution to the community: a method that can reliably defend against powerful MIAs while maintaining a good level of test accuracy/utility. I find the contributions of this paper valuable and appealing.

---

## References:

**[1]** Carlini, Nicholas et al. “Membership Inference Attacks From First Principles.” 2022 IEEE Symposium on Security and Privacy (SP) (2021): 1897-1914.

### Weaknesses
**[W1]** The description of the method could be done more clearly. While the motivation of the methodology by discussing prior approaches to rewinding, weights freezing and fine-tuning, and ablation studies is a great choice, it will be highly beneficial to have a concise self-contained discussion about the steps involved in CWRF along with pseudocode to avoid ambiguity and for easy reference.

**[W2]** The text in tables 1 and 2 is very small and not very appropriate for a potential camera-ready version.

---

This paper is pretty solid otherwise.

### Questions
**[Q1]** Can you please address W1 and add a self-contained description of CWRF in the final version of the paper for better presentation?

**[Q2]** Could you also ideally address W2 as well?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a method, Critical Weights Rewinding and Finetuning (CWRF), for optimizing model utility and membership privacy simultaneously. It attempts to resolve the issue of "entanglement", where learnability (utility) and privacy vulnerability are concentrated in the same small set of critical weights. The new approach is applied as a "booster" to existing privacy-preserving training methods (DP-SGD, RelaxLoss, HAMP, and CCL) and is evaluated on both model utility and its resilience to Membership Inference Attacks (MIAs). The algorithmic contribution is demonstrated with a suite of experiments in classic benchmark domains (CIFAR-10, CIFAR-100, and CINIC-10) and on modern architectures (ResNet18 and ViT).

### Strengths
S1. The paper is well written and easy to follow

S2. The insights are interesting and helpful

### Weaknesses
I found the claims in the paper are interesting but unconvincing due to several points:

W1. Lacks of theoretical motivation. The paper's central hypothesis that a weight's importance stems from its location, not its value is a strong one, but it is presented without theoretical proof and is supported only by a specific set of ablation studies. The paper argues that because A3 (CWRF) successfully recovers accuracy while A1 (Remove) fails, the hypothesis is validated. While the result is compelling, it is not a proof. Can the authors provide any formal argument for this? How does this "location-over-value" hypothesis relate to existing work like the Lottery Ticket Hypothesis, which is mentioned but not deeply connected?

W2. Missing datasets. Previous baselines such as RelaxLoss or HAMP include tabular dataset such as Purchase or Texas besides image dataset.

W3. The justification for fine-tuning invulnerable weights (A3) over vulnerable weights (A2) is the paper's most critical experimental result. However, this experiment appears to have been run only using RelaxLoss as the fine-tuning defense. Does this crucial finding hold for DP-SGD, HAMP, and CCL as well?

### Questions
Please refer to the Weaknesses section.

Some minor points:

- The abstract states CWRF "exhibits outperforming resilience", but the actual results show mixed performance on RMIA for ResNet18 e.g RelaxLoss is better without CWRF

- In Figure 3, the paper notes that the axes for the ViT plots are not consistent. While this is acknowledged, it makes visual comparison of the correlations across different layer types (e.g., Att+MLP vs. Norm) very difficult.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates the relationship between model learnability and privacy, finding that privacy vulnerability is "entangled" with utility performance within a very small fraction of critical weights. The authors propose a key insight that the importance of these weights stems from their location rather than their trained values. Based on this, the paper introduces Critical Weights Rewinding and Finetuning (CWRF) , a strategy that identifies this small set of privacy-vulnerable weights. Instead of pruning, CWRF rewinds only these high-risk weights to their initial, privacy-safe values and then freezes them. Finally, the model recovers its utility by fine-tuning only the remaining "privacy-invulnerable" weights, leveraging the fact that the critical weights' locations are preserved. Experiments demonstrate that CWRF successfully boosts the effectiveness of existing privacy-preserving training methods, achieving a superior privacy-utility tradeoff.

### Strengths
1. The work offers a significant contribution by clearly explaining why standard model pruning fails to mitigate privacy risks, linking it directly to this entanglement.
2. The paper proposes CWRF that cleverly combines machine unlearning for vulnerability estimation with weight rewinding, which boosts existing privacy-preserving methods to achieve.

### Weaknesses
1. The paper provides no sensitivity analysis for the hyperparameter rewinding rate $r$ and $\lambda$, making it unclear how to set it efficiently.
2. The empirical validation is limited to small-scale models (ResNet18, small ViT) and datasets (CIFAR, CINIC). It is not demonstrated whether the vulnerability estimation step is computationally feasible or if the core insight scales to LLMs. 
3. It is recommended to include full privacy–utility curves (similar to those reported for RelaxLoss and CCL) rather than only isolated metric points, as these curves may provide a more comprehensive representation of the privacy-utility trade-off.  
4. Previous work [1] also discusses which parameters substantially impact privacy risk. Please add a discussion and comparison with this work. 

[1] "Defending Membership Inference Attacks via Privacy-aware Sparsity Tuning" (2024)

### Questions
1. How are the hyperparameters chosen for the baseline methods reported in this paper? For example, the loss threshold for RelaxLoss and the $\gamma$ for RMIA.
2. How do the authors split the dataset for training the target model and reference models?

### Soundness
3

### Presentation
3

### Contribution
2
