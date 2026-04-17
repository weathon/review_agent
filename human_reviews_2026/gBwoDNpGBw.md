# Turning Uncertainty into Control: Bi-level Training with Editable Bayesian Layers

- Decision: Reject
- Scores: 4, 4, 6, 2, 4

## Abstract
As deep learning systems are increasingly deployed in high-stakes settings, it is
essential not only to quantify predictive uncertainty but also to use it to steer train-
ing and improve downstream decision. Yet dense parameter updates often cause
collateral interference due to the entangled nature of parameter. In this paper,
We propose Bayesian Feature Reweighting (BFR), a framework that turns cali-
brated, instance-level uncertainty into training-time control signals while enforc-
ing sparse, localized parameter updates. Concretely, BFR employs a Bayesian
Last Layer to obtain well-calibrated predictive uncertainty with negligible over-
head and imposes sparsity on the decision layer to concentrate changes on task-
relevant parameters, thereby limiting unintended perturbations. It then tinkers
the model via uncertainty-guide bi-level optimization. Across vision and lan-
guage benchmarks, BFR consistently improves group robustness while maintain-
ing competitive accuracy and yields smaller parameter changes. Code is provided
in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present Bayesian Feature Reweighting, a two-step procedure which first uses probabilistic pre-training with a Bayesian last layer to then perform uncertainty-guided optimization. They show that their method achieves competitive accuracy and worst-group accuracy across a diverse set of benchmarks.

### Strengths
- The authors do a thorough job of benchmarking their method against a large number of related methods on various spurious correlation benchmarks, and BFR has competitive results.
- The method is thoroughly explained, and each step has sufficient detail. The intuitions behind the method are presented well, and the code is also available for reproducibility.
- The visualizations and ablations provided are interesting, and Figure 4 demonstrates how the model is able to shift from spurious correlation to core features for high-uncertainty samples.

### Weaknesses
- I do not think that the method is well-motivated, although I may have missed this in the paper (see next point). Here is my understanding of the main claims of the paper, and why I don't think the evidence is sufficient to support these claims.
  1. **The sparsity prior is important**: The prior over $\theta$ and $\Phi$ is fixed to $Gamma(1, 1)$ throughout the paper, and there are no ablations with other choices of priors. It's unclear to me what the benefits of this prior are.
  2. **BLL is a good method of measuring uncertainty**: BLL is the only probabilistic method which is tested, and the authors do not ablate with other low-cost methods like Dropout. I would also be interested in seeing an ablation where epistemic uncertainty is not considered, and the bi-level optimization is performed directly on the original's model's aleatoric uncertainty, since that would demonstrate the importance of the probabilistic pre-training component of the method.
  3. **Uncertainty-guided bi-level optimization outperforms other instance-level uncertainty methods**: While the proposed method is intuitive, there seem to be simpler methods to balancing high-uncertainty and low-uncertainty loss, such as minimizing a weighted sum of the two groups. It would be helpful to see how this minmax problem is beneficial. I see an ablation with "Direct-U" in the appendix (not referenced in the main text), but I don't understand what this weighting is, or why there would be an inner learning rate for these problems.
- The analysis of the empirical results is weak, and makes the result difficult to interpret. The methods which BFR is compared against are not explained. Even if there is insufficient space to discuss all of the results in detail, it would be helpful to directly compare the differences between BFR and a few related works so I can understand why BFR seems to be performing well. Instead, the paper concludes (L484) "Additional ablations isolate the contributions of each uncertainty component and of the retraining strategy", but I don't know which experiments this refers to.

### Questions
- Could you elaborate on what ERM-BFR means? This seems like an important ablation, but the only explanation I could find was L355: "We replace the last layer of the base ERM models (Vapnik, 1999) for the base model training". How does this differ from BFR? 
- Do you have runtimes for your method? I see that there's a complexity analysis, but it's difficult to understand how significant the big O terms are in practice.

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
3

### Summary
This paper introduces Bayesian Feature Reweighting (BFR), a framework that uses calibrated predictive uncertainty as a control signal for model training. It combines a Bayesian Last Layer with sparsity-inducing priors and a bi-level optimization procedure that reweights samples based on uncertainty, promoting robust and localized parameter updates. BFR is tested on several vision and NLP benchmarks and improves worst-group accuracy and out-of-distribution robustness without requiring group labels.

### Strengths
1. **Motivation:** The paper moves beyond using uncertainty only for post-hoc evaluation and instead leverages it as a training signal to guide model updates, and allows robust fine-tuning. This provides an effective approach to improving robustness and editability.

2. **Evaluation:** The method is evaluated across diverse vision and NLP benchmarks, demonstrating consistent gains in worst-group accuracy and robustness over strong baselines.

3. **Visualization:** Results in Fig 4 shows an interesting observation that how BFR produces more interpretable model changes, aligning with its goal of controlled and explainable fine-tuning.

4. **Theoretical and complexity analysis:** This work also introduces a theoretical and complexity analysis, increase the soundness of the proposed method.

### Weaknesses
1. **Motivation:** 
The method performs well without group annotations but remains weaker than baselines that use them. The paper does not clearly justify why annotation-free settings are particularly important or how frequently they occur in practice. A clearer motivation and discussion of use cases would strengthen the contribution.

2. **Scalability:** Although the authors claim negligible computational overhead, the probabilistic formulation and bi-level optimization still introduce nontrivial costs. It remains unclear how the method scales to larger models and datasets, what the typical training time and computational resources are, and whether these factors limit broader applicability.

3. **Last Layer fine-tuning:** The Bayesian Last Layer is efficient but inherently limited in scope since only the final layer is adapted. This may restrict representational flexibility and the model’s ability to capture deeper interactions compared to methods that fine-tune intermediate representations. A discussion or comparison illustrating this trade-off would be valuable.

### Questions
1. Could the authors elaborate on the practical importance of group-annotation-free settings? In which real-world applications is this assumption most relevant? Clarifying why this scenario deserves particular attention would strengthen the paper’s motivation, especially since methods with group labels still outperform BFR.

2. While the Bayesian Last Layer is presented as having negligible overhead, it only adapts the top layer during fine-tuning. Could the authors discuss whether this limits representational flexibility compared to fine-tuning other layers?
Computational scalability.

3. It would be helpful to include or discuss results on how sensitive BFR’s performance is to hyperparameters. Which components contribute most to the observed robustness gains?

### Soundness
2

### Presentation
2

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
The paper proposes Bayesian Feature Reweighting (BFR), which uses a Bayesian Last Layer to estimate calibrated uncertainty, partitions validation data into low- and high-uncertainty groups, and then applies bi-level optimization to stably reweight training and improve robustness.

### Strengths
1. Combining the Bayesian Layer for Uncertainty and using the uncertainty as a proxy for group label is interesting
2. The experimental results are solid and extensive.

### Weaknesses
1. Using a Bayesian last layer for uncertainty estimation and reweighting the loss via (pseudo) group labels is established in prior work; the novelty here appears to lie primarily in the specific combination.

### Questions
1. Is the Bayesian last layer strictly necessary? For group partitioning, could alternative uncertainty proxies (e.g., loss, confidence, MC uncertainty) work?
2. What is the computation efficiency of the proposed methods?
3. Figure 2 shows the correlation between uncertainty and group labels in the visualization. Is it possible to calculate the correlation quantitatively?

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
5

### Summary
The paper introduces an uncertainty-driven training framework that leverages probabilistic pretraining and statistical test-based uncertainty estimation. Rather than relying on a fixed uncertainty-weighting scheme, the approach uses the estimated uncertainties to partition the training data into support and target sets. A bi-level optimization strategy is then employed, where the inner loop estimates sample weights and the outer loop updates the last-layer parameters. The proposed method is evaluated across diverse datasets spanning image, language, and distribution-shift scenarios.

### Strengths
**1. Well-motivated:** The paper is well-motivated and presents a well-crafted motivating discussion on both classical and SOTA methods of uncertainty estimation and spurious correlation. 

**2. Well-presented:** The paper is very well-organized in terms of describing its technical parts such as probabilistic pretraining, uncertainty estimation, and the proposed bi-level optimization scheme. 

**3. Diverse experimental settings:** The paper employs diverse datasets including image, language, and distribution shift to validate their proposed method.

### Weaknesses
**1. Absence of Innovation in Sect 3.2:** This section essentially reiterates the theory proposed by [1]. The derivation of Eqn. 6 in Appendix C is already presented in detail in Eqn (13) of [1]. Therefore, Eqn. (6) can be directly cited from [1] without the claimed novelty in the writing. 

**2. Limited novelty in Sect 3.3.** This section is essentially a derivative of the original p-valued based uncertainty estimation of [2]. The bi-level optimization was also originally proposed by [3] which also adopts similar bi-level training strategy to weight the samples based on uncertainty. 

[1] Xinyue Hu, Zhibin Duan, Bo Chen, and Mingyuan Zhou. Enhancing uncertainty estimation and interpretability with bayesian non-negative decision layer. In The Thirteenth International Conference on Learning Representations, 2025.

[2] Xinjie Fan, Shujian Zhang, Korawat Tanwisuth, Xiaoning Qian, and Mingyuan Zhou. Contextual dropout: An efficient sample-dependent dropout module. In International Conference on Learning Representations, 2021 

[3] Xiao Zhou, Yong Lin, Renjie Pi, Weizhong Zhang, Renzhe Xu, Peng Cui, and Tong Zhang. Model agnostic sample reweighting for out-of-distribution learning. In International conference on machine learning, pp. 27203–27221. PMLR, 2022.

### Questions
The paper is clearly written and well-presented, with comprehensive experimental results. However, its main contribution appears to be an effective engineering integration of the theories proposed in [1], [2], and [3], rather than a fundamentally novel methodological advancement. I recommend that the authors strengthen the technical and theoretical aspects of the work to make it more suitable for theory-oriented venues such as ICLR, ICML, or NeurIPS.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Bayesian Feature Reweighting (BFR), which replaces the classifier with a Bayesian Last Layer (BLL) to get calibrated, instance-level uncertainty under sparsity-promoting priors, then uses that uncertainty to control training via a bi-level optimization that upweights high-uncertainty examples while keeping parameter edits localized to the head. Evaluated on Waterbirds, CelebA, MultiNLI, CivilComments, and an ImageNet-9 to ImageNet-A shift. BFR improves worst-group accuracy and shows smaller fractions of changed parameters vs last-layer retraining baselines.

### Strengths
Experiments demonstrate some improvement in worst-group and OOD accuracy across vision (Waterbirds, CelebA, ImageNet-A) and language (MultiNLI, CivilComments) tasks.

### Weaknesses
- The novelty of this paper is incremental: BLLs and bi-level reweighting are established methods. The contribution is mainly coupling a known BLL with a bi-level sample-reweighting loop. There is little new probabilistic machinery beyond the proposed method.
- The paper claims to "turn uncertainty into control", but the control mechanism is largely heuristic partitioning of data into "support" vs "target" via that p-value, followed by a standard min–max bi-level update. There is no derivation that this procedure optimizes worst-group risk under the proposed uncertainty.
- The introduction argues that principled ways to use uncertainty as a training signal to enhance representation learning remain underexplored, yet the method offered does not provide calibration guarantees or a decision-theoretic objective tying the p-value to group-robust risk, so the “principled” claim is not met.
- The paper asserts aleatoric and epistemic uncertainty modeling, but validates only a unified predictive uncertainty.
- Computational cost (e.g., time, FLOPs, or memory usage) is not reported to support the "lightweight" claim.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
