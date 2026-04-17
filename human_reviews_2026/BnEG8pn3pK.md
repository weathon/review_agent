# Reducing information dependency does not cause training data privacy. Adversarially non-robust features do.

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
In this paper, we challenge the prevailing view that information dependency (including rote memorization) drives training data exposure to image reconstruction attacks. We show that extensive exposure can persist without rote memorization and is instead caused by a tunable connection to adversarial robustness. We begin by presenting three surprising results: (1) recent defenses that inhibit reconstruction by Model Inversion Attacks (MIAs), which evaluate leakage under an idealized attacker, do not reduce standard measures of information dependency (HSIC); (2) models that maximally memorize their training datasets remain robust to MIA reconstruction; and (3) models trained without seeing 97% of the training pixels, where recent information-theoretic bounds give arbitrarily strong privacy guarantees under standard assumptions, can still be devastatingly reconstructed by MIA. 

To explain these findings, we provide causal evidence that privacy under MIA arises from what the adversarial examples literature calls "non-robust" features (generalizable but imperceptible and unstable features). We further show that recent MIA defenses obtain their privacy improvements by unintentionally shifting models toward such features. To establish this causal relationship, we introduce **A**n**t**i **A**dversarial **T**raining (**AT-AT**), a training regime that intentionally learns non-robust features to obtain both superior reconstruction defense and higher accuracy than state-of-the-art defenses. Our results revise the prevailing understanding of training data exposure and reveal a new privacy-robustness tradeoff.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper challenges the prevailing view that information dependency causes privacy leakage in machine learning models. It compellingly argues that vulnerability to reconstruction attacks stems from the model's reliance on "robust," human-interpretable features. Based on this, it introduces Anti-Adversarial Training (AT-AT), a novel method that intentionally trains the model to use "non-robust," human-imperceptible features to obscure visually sensitive information from attackers.


I recommend a Weak Accept. 
- The paper's excellent and fundamental contribution in identifying that robust features are the root cause of privacy leakage. However, I find its proposed solution, which sacrifices model robustness and interpretability for privacy, to be a questionable direction that I do not endorse.

### Strengths
- Fundamental and Novel Contribution: The paper's primary strength is its rigorous and convincing refutation of the widely-held "information dependency → leakage" hypothesis. The identification of the connection between robust features and privacy vulnerability is a fundamental, paradigm-shifting insight that could redirect future research in this area.

- Strong Empirical Evidence: The three initial experiments are exceptionally well-designed and counter-intuitive, providing strong evidence to dismantle the status quo. The findings—that effective defenses don't reduce information dependency metrics, that rote-memorizing models are private, and that data from massively pixel-deleted images can still be reconstructed—are surprising and impactful.


- Revealing a New Trade-off: The discovery and formalization of the "privacy-robustness trade-off" is a significant conceptual contribution in itself. It provides a new and valuable lens for understanding the inherent tensions in designing secure and private machine learning models.

### Weaknesses
- Undermines Model Interpretability and Trustworthiness: A core goal of building reliable AI is for models to learn features that are robust and align with human perception, making them more interpretable. The AT-AT framework intentionally subverts this principle by compelling the model to rely on non-robust, human-imperceptible features for classification. While this may protect visual privacy, it achieves it by making the model's decision-making process opaque and semantically meaningless to humans. This is a significant step backward for explainable and trustworthy AI.

- Creates Intentionally Fragile Models: The direct consequence of relying on non-robust features is an extreme vulnerability to adversarial attacks. The authors acknowledge this as a trade-off, but I argue that deliberately engineering models to be fragile is not an acceptable solution. For years, the community has worked to make models more robust; this approach reverses that progress, trading one major security flaw (privacy leakage) for another (vulnerability to adversarial manipulation). This does not seem like a net gain for model security.

While the paper's contribution in identifying the role of robust features in privacy leakage is excellent, I do not endorse the proposed solution of relying on non-robust features as a viable or "correct" way to build private models. The solution sacrifices too much in terms of robustness and interpretability.

### Questions
- The AT-AT framework appears to encourage models to rely on non-robust, human-imperceptible features for classification. Could the authors clarify how this design aligns with the broader goal of developing interpretable and trustworthy AI systems?

- Have the authors considered integrating interpretability-preserving constraints or perceptual alignment mechanisms to mitigate the opacity introduced by adversarial feature reliance?

- Since the framework intentionally leverages non-robust features, how do the authors ensure that the resulting models are not excessively fragile to adversarial perturbations or distributional shifts?

- In the broader security context, how do the authors justify deliberately reducing model robustness as a viable long-term strategy for privacy protection?


While the paper's contribution in identifying the role of robust features in privacy leakage is excellent, I do not endorse the proposed solution of relying on non-robust features as a viable or "correct" way to build private models. The solution sacrifices too much in terms of robustness and interpretability.

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the challenge of defending against model inversion attacks (MIA). It investigates the relationship between information dependency and privacy, yielding three key findings:
Models that effectively defend against MIA (measured by Attack Acc@1) do not necessarily exhibit reduced information dependency (quantified by HSIC).
Models that memorize training data can still remain robust to MIA (evaluated via L2-Face).
Even models trained on images with 97% of pixels masked can still be reconstructed by MIA (as indicated by Attack Acc@1, @5, and L2-Face).
To interpret these counterintuitive results, the authors introduce the concept of non-robust features—imperceptible perturbations that alter predicted labels. By analyzing robust accuracies under varying perturbation magnitudes, they demonstrate a clear linear relationship between robust accuracy and attack success rate.
Building on this insight, the paper proposes a new training objective, AT-AT, which learns non-robust features to obtain MIA defense.
The paper offers interesting new perspectives, is generally well written (even though this could be further improved, as detailed below), and the experimental evaluation shows some promise (even though some details are unclear).

### Strengths
This work challenges the conventional assumption that reducing information dependency between inputs and latent representations inherently improves privacy. Moreover, it identifies a meaningful correlation between privacy leakage and robust accuracy, opening a new perspective and research direction at the intersection of privacy and robustness.
The experimental design is sound, comprehensive, and statistically validated, lending credibility to the reported observations.
The paper compares against a sufficient range of baseline methods, including those focusing on information dependency as well as those emphasizing gradient-based denfense strategies.
The visualizations and plots are clear, well-structured, and intuitive, effectively supporting the main arguments.

### Weaknesses
Lack of causal evidence: Section 4 establishes a correlation between privacy and non-robust features, which is well supported by experimental results. However, no causal relationship is demonstrated. Consequently, the claim that “recent MIA defenses obtain their privacy improvements by unintentionally shifting models toward such features” remains insufficiently substantiated.
Missing hyperparameter details: In the proposed AT-AT algorithm, the hyperparameter ε (epsilon)—which controls perturbation magnitude—is neither defined nor discussed. The absence of this detail limits reproducibility and interpretability of the results.
Limited dependency metrics: The paper relies solely on HSIC to measure information dependency. Other established measures, such as CLUB, could have been included to provide a more comprehensive and cross-validated analysis.
Restricted model scope: The experiments are conducted exclusively on ResNet-based architectures, with no evaluation on Transformer-based models. This limits generalizability of the findings to such architectures.

Minor comments:
The structure and organization of the paper could be improved. For example, robust accuracy is frequently referenced in Section 4 and the Introduction, but is not clearly defined, and its relationship to non-robust features is insufficiently discussed. Moreover, there is substantial redundancy between the Introduction and the Experiments section.
Terminological consistency should be improved. The terms information dependency and memorization are used somewhat interchangeably, leading to ambiguity. Choosing a single consistent terminology and providing a clearer definition would enhance readability and conceptual clarity.

Ref: Cheng, Pengyu, et al. “Club: A contrastive log-ratio upper bound of mutual information.” International conference on machine learning. PMLR, 2020.

### Questions
First experiment:
Why does training with HSIC regularization result in a higher HSIC value? According to the BiDO paper, HSIC was originally introduced to reduce information dependency, yet this work reports the opposite effect. It would be helpful to clarify this discrepancy. Interestingly, the BiDO paper did not explicitly report post-training HSIC values for comparison.

Second experiment:
How exactly are the labels permuted? If labels are permuted per person (on the images), this would merely rename the classes and should not alter the learned representations. I assume the permutation is performed per image, but this should be explicitly stated.
Additionally, an overfitted model typically learns a direct mapping from input to label rather than meaningful features, effectively reducing the dependency between images and latent representations while increasing the dependency between latent representations and labels. For example, the model can memorize the label of a specific input by memorizing several pixel values. Reporting HSIC values between images and latent spaces for both the permuted and non-permuted models would provide valuable insight.

Third experiment:
In this setting, the available information in the image space is extremely limited, meaning the latent space could potentially encode nearly all pixel-level details. This might explain the model’s vulnerability to inversion attacks. Measuring the HSIC between images and latent spaces in this scenario could help validate this hypothesis.
Furthermore, what datasets are used to train the PPA generative model? If the generative model is trained on the same dataset as the classifier, reconstructing images from only 3% visible pixels may be significantly easier, which would influence the interpretation of the results.
Line 410 and Equation (3):
Why is the objective formulated as a maximization rather than a minimization? In the AT-AT algorithm, ( x + \delta ) corresponds to ( x’ ). Given that the model should predict ( y’ ) for input ( x’ ), it seems that a minimization objective would be more appropriate. Please clarify.

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
5

### Summary
This paper challenges the prevailing paradigm that "reducing information dependency mitigates training data leakage via Model Inversion Attacks (MIAs)" through three counterintuitive experiments. To explain these findings, the authors identify a novel privacy-adversarial robustness tradeoff. Based on these findings, they propose Anti Adversarial Training (AT-AT), a defense that proactively learns non-robust features by reversing standard adversarial training. Experimental results have validated the effectiveness of the proposed method.

### Strengths
- Good performance: the proposed method outperforms baselines in most metrics
- Generalizability: results hold across datasets, architectures, and attacks, demonstrating the tradeoff’s universality and AT-AT’s adaptability.
- Practical Relevance: AT-AT’s tunable λ addresses a key limitation of SOTA defenses (fixed privacy-robustness tradeoffs) and works for deployed models without modifications on the training/generation process.

### Weaknesses
- This work only focuses on the white-box attacks. It is recommended that the author conducts more evaluation on the black-box attacks and label-only attacks to further validate the correlation between adversarial robustness and privacy.
- The compared baselines are not sufficient. Attacks like PLGMI [1] and PPDG [2] and defenses like Trap-MID [3] are expected to compare.
- If a model is fine-tuned post-deployment (shifting its non-robust features), the author does not evaluate whether AT-AT’s privacy guarantees persist or require retraining.


[1] Yuan X, Chen K, Zhang J, et al. Pseudo label-guided model inversion attack via conditional generative adversarial network[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2023, 37(3): 3349-3357.

[2] Peng X, Han B, Liu F, et al. Pseudo-private data guided model inversion attacks[J]. Advances in Neural Information Processing Systems, 2024, 37: 33338-33375.

[3] Liu Z T, Chen S T. Trap-MID: Trapdoor-based Defense against Model Inversion Attacks[J]. Advances in Neural Information Processing Systems, 2024, 37: 88486-88526.

### Questions
Refer to Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper argues that training-data leakage is driven by adversarially robust features instead of information dependency, and it supports this claim with experiments that falsify the dependency hypothesis. The authors therefore propose Anti-Adversarial Training (AT-AT) which intentionally shifts learning toward non-robust features to reduce leakage, since these are hard for MIAs to turn into clear pictures.

### Strengths
Strong MIA defense, AttAcc@1 fell from ~84% to ~6.5% (Tab. 21). On RN-152 accuracy vs. privacy (Fig. 3), AT-AT outperforms baseline methods across datasets and attacks.


The paper combines causal reframing, quantified trade-off, and practical SOTA method. The authors clearly test and challenge a popular theory using simple, targeted experiments, and turn the insights into AT-AT that lowers SOTA inversion success while keeping accuracy high.

The paper is generally well written, with a decent amount of details provided in the appendix.

### Weaknesses
Non-robust cues may still be highly predictive when added up and even sufficient for generalization. That means an adaptive MIA that directly optimizes the target’s representations could still extract identity/membership information. The paper does not systematically evaluate such features or prior-agnostic inversion attacks, so its privacy gains may depend only on the current setup.


Narrow empirical scope. The paper lacks tests with broader attackers, other modalities, and offers no formal guarantees.

### Questions
Does AT-AT trade privacy gains for interpretability? The proposed method might help privacy, but it may suffer in terms of interpretability compared to robustly trained models. If the model leans on tiny, high-frequency cues spread across many pixels (rather than dog’s head or a face), attribution maps may become visually unintuitive.

### Soundness
3

### Presentation
2

### Contribution
3
