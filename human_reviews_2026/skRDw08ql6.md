# Private and interpretable clinical prediction with quantum-inspired tensor train models

- Avg Score: 4.00
- Decision: Reject
- Scores: 8, 2, 4, 2

## Abstract
Machine learning in clinical settings must balance predictive accuracy, interpretability, and privacy. Models such as logistic regression (LR) offer transparency, while neural networks (NNs) provide greater predictive power; yet both remain vulnerable to privacy attacks. We empirically assess these risks by designing attacks that identify which public datasets were used to train a model under varying levels of adversarial access, applying them to LORIS, a publicly available LR model for immunotherapy response prediction, as well as to additional shallow NN models trained for the same task. Our results show that both models leak significant training-set information, with LRs proving particularly vulnerable in white-box scenarios. Moreover, we observe that common practices such as cross-validation in LRs exacerbate these risks. To mitigate these vulnerabilities, we propose a quantum-inspired defense based on tensorizing discretized models into tensor trains (TTs), which fully obfuscates parameters while preserving accuracy, reducing white-box attacks to random guessing and degrading black-box attacks comparably to Differential Privacy. TT models retain LR interpretability and extend it through efficient computation of marginal and conditional distributions, while also enabling this higher level of interpretability for NNs. Our results demonstrate that tensorization is widely applicable and establishes a practical foundation for private, interpretable, and effective clinical prediction.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Authors first identifies vulnerabilities with privacy of LR models, and then propose a quantum-inspired defense using tensor train (TT) models to mitigate those vulnerabilities by tensorizing LR to obfuscate parameters while preserving accuracy and interpretability.

### Strengths
- Work tackles important issue of privacy of some of the widest used ML models in critical applications like medicine LR proposing a private method that remains interpretable and minimises leakage of personal data. Authors studied specifically membership inference attack based on shadow-model training.
- Authors reconstructed coefficients of the publically accessible model LORIS showing directly mentioned vulnerabilities

### Weaknesses
- The paper successfully demonstrates the TT defense on LR models, particularly LORIS, and notes that the approach only requires Black-Box (BB) access, making it generalizable to arbitrary models. This claim is a major potential contribution but currently lacks empirical validation beyond the linear model used.
- Preliminary theoretical framework would benefit the paper.

### Questions
The paper advises that averaging models for deployment should be avoided as common practices like cross-validation compromise privacy. Given that averaged models are often standard practice for stability, what specific alternative strategies do the authors recommend for the practical deployment of stable, yet privacy-preserving, models in sensitive clinical domains?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper shows that a linear regression (LR) models are vulnerable to membership inference attacks, and proposes a defense technique based on tensor trains (TT). The experimental results on LORIS, a publucly available immunotherapy response prediction model, shows that an attacker can accurately identify which datasets (from the nice candidates) were used to train the model. Then, the paper shows that applying the tensor train representation to the LR model makes this membership inference attack far less effecitve. Finally, the paper shows that the proposed TT approach still maintains the explainability of the original LR models.

### Strengths
The privacy concern in medical AI models is an important concern. The paper makes clear observations on the privacy risk of LR models and the potential privacy benefit of using TT. The experimental results suppor the main claims.

### Weaknesses
While demonstrating the privacy risk in a real-world immunotherapy prediction model is a notable contribution, the observation is not necessarily new or surprising from the technical point of view. Membership inference attacks have been shown for many types of models including more complex deep-learning models. It is relatively well-known that LR models are even more vulnerable given their simplicity and linear nature. In that sense, while the observation on the vulnerabiltiy adds another datapoint to what is known, it does not represent a new risk that was not known before. Also, the main privacy risk in medical models will be for individual personal records instead of entire datasets. In that sense, the study will be more compelling if the authors demonstrate attacks on individual records.

The proposed defense based on TT seems promising. However, the current security evaluation is only based on a specific type of an attack. Given that previous studies on other types of low-rank approximation (such as LoRA) did not provide sufficient privacy protection, it is unclear how robust the privacy protection from TT will truly be. Ideally, a privacy protection scheme needs to provide a mathematical guarantee. If not, the evaluation needs to be more comprehensive, including stronger learning based attacks as well as other privacy risks such as input reconstruction attacks, property inference attacks, model inversion attacks, etc. 

There exist a large body of work on privacy risks in ML models. For related work, it will be helpful if the paper provides more comprehensive discusions. In particular, given that the paper proposes the TT-based defense, I would suggest including previous studies on low-rank approximation techniques and more explicitly state how this work is different.

### Questions
See above.

### Soundness
2

### Presentation
3

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
This paper studies privacy vulnerabilities in logistic regression models used for clinical prediction, focusing on the publicly available LORIS model for immunotherapy response. The authors design membership inference attacks under white-box and black-box access and show that cross-validation exacerbates leakage. As a defense, they propose tensorizing LR models using quantum-inspired tensor train representations, claiming that TT models retain interpretability and predictive performance while enhancing privacy, roughly comparable to Differential Privacy.

### Strengths
- The paper tackles a critical and timely issue at the intersection of clinical machine learning, privacy, and interpretability. Protecting sensitive medical data while maintaining transparent model behavior is a fundamental challenge for real-world deployment, and this work contributes a novel and practical perspective on it.
- The experimental setup is creative and insightful. The authors not only design realistic membership inference attacks but also demonstrate how such attacks can be applied to publicly available clinical models like LORIS. This empirical analysis illustrates the vulnerability of logistic regression models to privacy breaches, even under limited (black-box) access.
- The quantum-inspired tensor train representation provides an elegant defense mechanism that obfuscates model parameters while preserving the predictive accuracy and interpretability characteristic of logistic regression. The ability to maintain interpretability while improving privacy is a key contribution.

### Weaknesses
- The paper's primary contributions, including both the attack analysis and the proposed defense, are validated *exclusively* on Logistic Regression (LR) models. While the authors claim the tensorization method can be "applied to arbitrary models" because it only requires black-box access, this crucial claim is entirely unsubstantiated by the experiments. It remains unknown how effective this defense would be for more complex, non-linear models (e.g., deep neural networks).
- The Tensor Train (TT) defense mechanism proposed in the paper provides what is effectively empirical privacy; that is, it successfully thwarts the specific attacks designed by the authors. This is essentially a form of privacy by obfuscation, which lacks the rigorous, attack-agnostic mathematical guarantees of a framework like Differential Privacy (DP). It remains uncertain whether a more powerful adversary or an attack using different techniques could still break this defense.
- The paper's direct comparison between the TT method and DP is fundamentally inequitable because their privacy-utility mechanisms are non-equivalent. DP's trade-off is governed by the addition of calibrated noise, formally controlled by the $\epsilon$ parameter. The TT method's trade-off, however, stems from a different operation: information loss via discretization, controlled by the heuristic parameter $b$. This disparity makes a "fair" comparison inherently difficult, as it raises the question: what $\epsilon$ value in DP constitutes an equitable benchmark against the TT's $b=6$ setting? While the paper itself attempts to equate the two by suggesting $\epsilon=10$ offers a similar trade-off in empirical utility and robustness, this comparison is based on post-hoc performance observation. It misleadingly equates a formal, provable privacy guarantee with a heuristic, obfuscation-based one, simply because their empirical utility and robustness to a single attack happen to align at that specific setting.
- The paper's experimental comparison is largely limited to the original LR model and DP. This is insufficient to properly contextualize the contribution of the Tensor Train approach. The study would be significantly stronger if it benchmarked against other established privacy-preserving machine learning methods, even those mentioned in the related work , such as knowledge distillation or model pruning.

### Questions
- Could the authors provide preliminary evidence or at least a conceptual discussion of how the method would extend to non-linear or deep neural network architectures?
- The comparison between TT and DP relies on heuristic equivalence between the discretization parameter $b$ and the privacy budget $\epsilon$. Could the authors clarify how they define or justify this correspondence? Is there a principled way to calibrate $b$ so that it aligns with a given $\epsilon$, or are the two simply matched by post-hoc performance similarity?
- The study’s experimental comparisons are limited to LR and DP baselines. Have the authors considered benchmarking against other privacy-preserving strategies mentioned in the related work, such as pruning-based obfuscation or knowledge distillation?
- Table 1 shows that a WB attack on the TT cores results in random guessing, but an attack on reconstructed LR-TT coefficients is much more successful. Doesn't this imply that the function learned by the TT is still vulnerable to reconstruction and attack?

### Soundness
3

### Presentation
3

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
This work uses quantum-inspired “tensor train” model representations to produce black-box tensorized representations of logistic regression (LR) models. The work is motivated by the tension between the transparency and the attack vulnerability of simple ML models used within medical/clinical decision making contexts. The authors: 
* provide empirical evidence of logistic regression model vulnerabilities to adversarial attacks and access,
* demonstrate dataset leakage by an LR-based clinical model, LORIS (logistic regression-based immunotherapy response score),
* propose a defense training process using tensor train models that maintains accuracy and LR interpretability while decreasing attack vulnerability in the context of LORIS.

### Strengths
* Sec 3.3 Experimental Setup provides a detailed overview of the datasets, models, and implementation details. Suggested improvement under “Weaknesses”. 
* Feature sensitivity analysis is interesting and supports model interpretability, which aligns with the problem motivation and justification for focusing on LR models within this work.
* Preliminary exploration seems potentially promising. (Reviewer believes more is needed, please refer to `Weaknesses`.)

### Weaknesses
* At a high-level, the problem this work centers upon seems important for the application area in question (logistic regression for immunotherapy response prediction). However, it’s unclear how broadly applicable the proposed approach (quantum-inspired defense using tensor train models) is for other tasks or domains. The paper currently reads, at least to this reviewer, as being better suited for a more targeted audience or venue than the broader ML community reflected at ICLR.
    * The scoping centers largely on LORIS, “a publicly available LR model for immunotherapy response prediction”, which is a very specific application of logistic regression models (which themselves are a very specific, simple, and classical ML model). This results in the paper’s findings being very tightly scoped with limited broader applicability (even if the proposed method may be more generally useful).
    * If the authors believe the scope of this work aligns with the ICLR community, a clearer introduction, background, and motivation of the setting is needed. 
* In response to the last line of the abstract: `Although demonstrated on LORIS, our approach generalizes broadly, positioning TT models as a practical foundation for private, interpretable, and effective clinical prediction.`
    * While this may be the case, we need to have some evidence of the broad generalizability and applicability of the method for this claim to hold.
* Sec 3.3.1 Datasets would benefit from a table to provide an outline the datasets used, both for greater ease of understanding for the reader. Additional details on these datasets would be helpful to contextualize the tasks, so that these are not simply shorthand identifiers. In other words, why do these tasks matter, and what do they map to (e.g. `clinical, pathological, and genomic features with a binary treatment-response label`)? This would also help build greater intuition around the vulnerabilities of LORIS and the risks for applications in clinical settings to motivate employing a method as described in this work. 
* Empirical results (in 3.4 and tables) are hard to follow. Suggest reorganizing to clarify the evaluation setup, individual tasks, results, and discussion.
* More discussion around the choice of metrics used to assess the adversarial attacks and their definition(s) is needed.

### Questions
* In terms of metrics, attack success rate is an important metric in evaluating adversarial attacks. Is this something you explored in the tasks evaluated here?
    * Additionally, adversarial robustness score would be useful, particularly to demonstrate the difference between LORIS and the increased robustness of using TT models.
* Under what basis can you support the claim of `Although demonstrated on LORIS, our approach generalizes broadly, positioning TT models as a practical foundation for private, interpretable, and effective clinical prediction.`? Demonstrating broader applicability would add more value and substance to this work.

### Soundness
2

### Presentation
2

### Contribution
2
