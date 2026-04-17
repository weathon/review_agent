# Membership Privacy Risks of Sharpness Aware Minimization

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
Optimization algorithms that seek flatter minima, such as Sharpness-Aware Minimization (SAM), are credited with improved generalization and robustness to noise. We ask whether such gains impact membership privacy. Surprisingly, we find that SAM is more prone to Membership Inference Attacks (MIA) than classical SGD across multiple datasets and attack methods, despite achieving lower test error. This suggests that the geometric mechanism of SAM that improves generalization simultaneously exacerbates membership leakage. We investigate this phenomenon through extensive analysis of memorization and influence scores. Our results reveal that SAM is more capable of capturing atypical subpatterns, leading to higher memorization scores of samples. Conversely, SGD depends more heavily on majority features, exhibiting worse generalization on atypical subgroups and lower memorization. Crucially, this characteristic of SAM can be linked to lower variance in the prediction confidence of unseen samples, thereby amplifying membership signals. Finally, we model SAM under a perfectly interpolating linear regime and theoretically show that sharpness regularization inherently reduces variance, guaranteeing a higher MIA advantage for confidence and likelihood ratio attacks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the vulnerability of neural networks trained using algorithms that improve generalisation, such as the Sharpness-Aware Minimisation (SAM), to membership inference attacks (MIAs). It provides empirical evidence that suggests SAM-optimised networks are more vulnerable to MIAs when compared to SGD-optimised networks. The author(s) also provide theoretical justification for the high MIA vulnerability of SAM-optimised networks despite improved generalisation, since that goes against the conventional expectation where high MIA vulnerability is associated with low generalisation (and vice-versa).

### Strengths
- The paper presents evidence that suggests that optimising models using SAM encourages memorisation of atypical samples (mid-range memorisation scores), which contribute to their improved generalisation.
- The author(s)' proposed metric of influence entropy $\mathcal{I}_{ent}$ in Eq(6) helps verify the claim that SAM enhances generalisation by encouraging the network to memorization more atypical samples compared to SGD-based optimisation (as seen in Figure 4(a)). 
- The empirical results show that the claims in the paper are not model-dependent but extend to different experimental settings.
- The theoretical justification for the observed phenomenon of improved generalisation leading to high memorisation when optimising a model with SAM is well described.

### Weaknesses
- Computing memorisation scores using Feldman and Zhang's [1] method is computationally expensive as it often requires (re)training hundreds of models. Why did the author(s) not use Ye et al.'s [2] more efficient version to compute these scores?
- Use of balanced accuracy as the attack metric and not TPR at fixed FPR [3], which informs more about an attack's ability to correctly identify membership signal at [preferably] low FPR (or low chances of predicting a member as a non-member).  
- It does not use SOTA MIAs such as LiRA [3] / Quantile-MIA [4] / RMIA [5] to measure the sensitivity of SAM-optimised networks, which are known to provide better estimates of a model's MIA vulnerability compared to attacks that rely on predetermined thresholds, such as the Entropy- or Confidence-based attacks. There is also the case that the theoretical results for Theorem 2 depend on a single [data-dependent] threshold, whereas a factor contributing to the success of SOTA MIAs is that they incorporate sample-level thresholds. 

[1] Feldman, V., and Zhang, C. “What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation.” NeurIPS 2020.

[2] Ye, J. et al. "Leave-One-Out-Distinguishability in Machine Learning." ICLR 2024.

[3] Carlini, N. et al. "Membership Inference Attacks From First Principles." SP 2022.

[4] Zarifzadeh, S. et al. “Low-Cost High-Power Membership Inference Attacks.” ICML 2024.

[5] Bertrán, M. et al. "Scalable Membership Inference Attacks via Quantile Regression." NeurIPS 2023.

### Questions
**Questions**: I would urge the author(s) to address the weaknesses detailed above. I am amenable to updating my initial assessment thereafter.

**Suggestions**: I suggest the following edits to improve the presentation of the paper:
- Minor Suggestion #1: Memorisation scores are measured w.r.t. samples, so this statement, "Motivated by this connection, we analyse the memorisation scores of SAM-trained models...", is somewhat misleading. It would be better to amend it to frame it w.r.t. individual samples.
- Minor Suggestion #2: Can you report the correlation coefficient between mem_SAM and mem_SGD for Figure 1(b) and 1(c)?
- Minor Suggestions #3: Lines 350-352 are written in a complicated and difficult-to-read manner. It would be best to rewrite it focusing on one relationship (for example, lower number bucket is associated with higher memorisation and vice versa).

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
4

### Summary
This paper reveals a critical trade-off in Sharpness-Aware Minimization (SAM): while it improves model generalization, it also increases privacy risk by making models more vulnerable to Membership Inference Attacks (MI). The authors demonstrate that this occurs because SAM's performance gains stem from its enhanced ability to memorize rare sub-patterns within the training data.

### Strengths
- The paper is well written and easy to follow.

- The finding of the relationship between the SAM and memorization is interesting and may shed light to privacy defense.

- The theoretical analysis is interesting and insightful.

### Weaknesses
+ (Major) Experimental Evidence. The central claim regarding SAM's heightened privacy risk is not yet fully convincing. The results in Table 1 show that SAM's privacy risk (measured by ASR) is not consistently or significantly higher than that of SGD (in Purchase-100 and Texas-100). Besides, Table 3 reports only the best attack accuracy, unlike Table 1 which shows results for all MIA methods.

+ (Major) Uncertain connection between ASR and memorization score. The link between the memorization score and the Membership Inference Attack (MIA) success rate is presented as a given. However, this relationship appears to be an assumption rather than an empirically demonstrated fact. This connection should be either validated or more cautiously framed as a hypothesis.

+ (Major) What can the observation bring to the future work on MIA or SAM? A more precise presentation of the study's potential implications for relevant fields would substantially strengthen the significance of this research.

+ (Minor) The solidness of the experiments would be strengthened by evaluating a wider range of modern MIA methods to ensure the findings are not specific to the selected attacks. 

+ (Minor) The experiments are primarily conducted on small datasets. It is unclear if the observed privacy trade-offs persist on larger, more complex benchmarks such as ImageNet.

## Score Justification

The paper is well-organized and introduces valuable theoretical analysis. However, the experimental evidence currently feels incomplete, which undermines the strength of its conclusions. Besides, the most significant shortcoming is the lack of a clear discussion on the broader impact and implications of these findings for the community. While briefly mentioned in the conclusion, a detailed discussion on how this new understanding of SAM's properties (especially concerning long-tailed data) can influence future research and practice is crucial.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a counterintuitive yet rigorously studied phenomenon: Sharpness-Aware Minimization (SAM), an optimization method known for improving generalization, unexpectedly increases vulnerability to membership inference attacks (MIAs). Through extensive experiments on multiple datasets and theoretical analysis, the authors demonstrate that SAM’s generalization gains stem from structured memorization of atypical subclass patterns(e.g., rare features in long-tailed distributions), which simultaneously enhances test performance and privacy risks. The work challenges the conventional belief that better generalization implies lower privacy risk, offering novel insights into the trade-offs between optimization, generalization, and privacy.

### Strengths
1. The paper offers novel insights into the trade-offs between generalization and privacy, which challenges the conventional belief that better generalization implies lower privacy risk.
2. The authors combine empirical evidence with theoretical guarantees to support their claims. The consistent results across datasets and models strengthen their conclusion.
3. The paper is well-written and easy to follow.

### Weaknesses
1. While the paper highlights SAM’s privacy vulnerability, it would be good to propose or evaluate defensive strategies to mitigate this risk to improve practical applicability.
2. The theoretical analysis relies on a simplified linear model and strong assumptions (e.g., perfect interpolation). It lacks both theoretical and empirical validations on more advanced non-linear architectures like Transformer-based or diffusion models.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper challenges the conventional belief that improved model generalization implies better membership privacy by investigating the SAM algorithm. The authors surprisingly find that SAM-trained models, despite achieving better generalization than standard SGD, may be more vulnerable to MIAs. The authors hypothesize this occurs because SAM selectively learn atypical but generalizable sub-patterns more effectively than SGD. This enhanced memorization of rare features improves performance on atypical test samples, but consequently increases the privacy risk by making training samples more distinguishable. The paper empirically validates this mechanism by analyzing memorization and influence scores and provides a theoretical framework showing how a model that better captures minority features can simultaneously achieve high generalization and high MIA vulnerability.

### Strengths
1. The paper first states that SAM increases vulnerability to MIAs.
2. The paper is exceptionally clear, presenting its counter-intuitive finding and complex hypothesis in a logical, well-structured, and easy-to-follow manner.

### Weaknesses
1. The authors claim to challenge the conventional assumption that improved generalization implies stronger privacy. However, this insight has already been discussed in previous works [1,2,3,4].  
2. The paper's claims are based solely on the original SAM algorithm. It is unclear whether these findings on increased privacy risk generalize to other sharpness-aware optimizers (e.g., ASAM, GSAM). An investigation into these variants is recommended.  
3. As a privacy metric, accuracy has been criticized in many works [4,5]. It is recommended to provide TPR at a low FPR and evaluate the privacy risk using recent MIA methods, such as LiRa and RMIA [4,6].
4. The "anti-alignment" assumption means a model that gets better at classifying the majority subclass inherently gets worse at classifying the minority subclass, forcing a direct trade-off. This is a very strong and specific setup.



[1] "Understanding membership inferences on well-generalized learning models" (arXiv, 2018)

[2] "When does data augmentation help with membership inference attacks" (ICML, 2021)

[3] "Bounding Information Leakage in Machine Learning" (Neurocomputing, 2023)

[4] "Membership Inference Attacks From First Principles" (SP 2022)

[5] "Evaluations of Machine Learning Privacy Defenses are Misleading" (CCS, 2024)

[6] "Low-Cost High-Power Membership Inference Attacks" (ICML, 2024)

### Questions
I do not understand why Assumption 4 is reasonable. Could you explain it?

### Soundness
2

### Presentation
3

### Contribution
2
