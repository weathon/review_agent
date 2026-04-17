# Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Generalization in EEG-based motor imagery (MI) brain-computer interfaces (BCIs) is hampered by cross-subject and cross-session variability. Although large-scale EEG pretraining has advanced representation learning, their practical deployment is hindered by the need for costly fine-tuning to overcome significant domain shifts. Test-time adaptation (TTA) methods that adapt models during inference offer a promising solution. However, existing EEG-TTA methods either rely on gradient-based fine-tuning (suffering from high computational cost and catastrophic forgetting) or data alignment strategies (failing to capture shifts in temporal predictive embeddings). To address these limitations, we propose BTTA-DG, a novel Bayesian Test-Time Adaptation framework that performs efficient, gradient-free adaptation by modeling the distribution of temporal predictive embeddings. Our approach first employs a lightweight SincAdaptNet with learnable filters to extract task-specific frequency bands. We then introduce a novel Dirichlet feature projection that maps temporal embeddings onto a compact and interpretable parameter space, effectively capturing the concentration of time-varying predictive evidence. Adaptation is achieved via a GMM-driven Bayesian inference mechanism, which models the historical distribution of these Dirichlet parameters and fuses this evidence with the model's prior predictions to calibrate outputs for the target domain. Extensive experiments show that BTTA‑DG significantly outperforms previous EEG‑TTA methods, achieving state‑of‑the‑art accuracy while running at real‑time speed. Furthermore, visualizations confirm the physiological interpretability of our learned filters and the robust class separability of our Dirichlet feature space.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel method for cross-subject (and cross-session) Test Time Adaptation (TTA) of motor imagery BCIs, that is based on Bayesian-inference calibration of predictions, with the likelihood being given by the (GMM) distribution of test trials in the Dirichlet parameter space of their class probability distributions across time. This method is not gradient-based, avoiding high computational cost and catastrophic forgetting. Compared to other non-gradient-based methods in prior literature, authors claim that this approach can adequately capture high-dimensional (deep feature distribution) domain shifts. Results show state-of-the-art performance (+0.8%-2.6% absolute accuracy improvement) in cross-subject and cross-session accuracy compared to prior TTA methods, evaluated in an online setting across 4 public motor imagery datasets. Computational efficiency is the second best among all compared methods, and visualization of the GMM clusters of test trials in Dirichlet space shows good separation of classes. Finally, additional ablation studies are provided concerning the comparison of probability calibration vs. projection only and the impact of hyperparameters in estimating the GMM likelihood, thus explaining performance and design choices.

### Strengths
This paper offers substantial novelty to the field of TTA in BCIs, through creative combination and application of existing statistical modeling methods (Dirichlet distribution + GMM estimation + Bayesian inference) which advances the state of the art considerably on multiple datasets, while being interpretable and efficient at the same time.

The paper is of high quality; the methods proposed are well thought out and well explained, the experiments and figures provided support (most of) the claims well, and figures are both pleasant and informative (especially figure 1). Writing is very clear, convincing, and easy to follow, even when going through complex technical details.

### Weaknesses
1. **High-level statements misaligned with method details.** In various instances throughout the paper it is either implied (lines 20, 61, 63, 70, 74, 250) or directly stated (line 22) that the method “directly models the distribution of deep features”. However, this is not true from this reviewer’s understanding of the method - the Dirichlet parameter space, although reflecting the distribution of class probabilities across trial time (which is indeed more than just static trial predictions), is still only modeling low dimensional class probability features and not high-dimensional deep features of the trials as stated (or implied). If by “high-dimensional” the authors referred to the temporal dimension, that should be made more clear as the current phrasing can be misleading (usually when saying high-dimensional / deep features of a DNN this cannot be the output layer even if it makes predictions per token).
2. **DNN model choice not well motivated.** While the introduction states that “recent advances in large-scale EEG pretrained models [cited: labram, egg-gpt, etc.] trained on massive datasets have demonstrated unprecedented capabilities in learning general […] representations”, these models are not used as the backbone model or as baselines in the current paper’s experiments. Instead, a much shallower model is used (only 4 layers) and it is trained from scratch in the down-stream datasets. This choice should be better motivated (ideally also showing the performance of those large-scale pretrained models as baselines), and depending on what that motivation is, it could also be good to show TTA results using those models as backbones.
3. **Useful additional comparisons desired.** First, it would be useful to be able to compare the TTA performance with the performance achieved with full fine-tuning to the target subject, to quantify how much this gap is starting to close; for example EA+Fine-tuning achieves 82.6 in BNCI2014001 (see [1]). Additionally, it would be informative to know the (k-fold) within-subject performance of each of the subjects here (perhaps in the Appendix), to provide insights into the “decodeability” of each subject’s data and how it may relate to the effectiveness of TTA with this data. For example in [1], pretraining on the rest of the subjects was shown to especially benefit “bad-performing” subjects compared to their within-subject performance - here it could conversely be that those subjects are not very effective for TTA. Second, an additional ablation that performs the GMM-BTTA without the Dirichlet projection, using instead the feature space of the mean class-probability across trial time (red dot in Figure 1; e.g. (d) would be represented by (0, 0.5, 0) instead of (1, 5, 1)) would also be good to demonstrate the usefulness of the projection (uncertainty modeling) in the calibration.
4. **Presentation issues.** The conclusion paragraph needs some more work. Sentence “While the framework assumes” (line 481) is either left unfinished or has some grammatical error. Sentence “our framework demonstrates robustness under mild departures” (line 484) is not evidenced in the results (or is not made clear which result this refers to). Same line, “would” does not make sense. Figure 4 would benefit from having the scale of all y axes be the same so that the difference in fluctuations is more clear and comparable. The order of tables and figures in that page is also inconsistent with the flow of the text, should be Table 7, Table 6, Figure 4. The paper would benefit from having Table 13 in the main text. There should also be some comment in the text on why “BN-adapt” is that much faster (Table 7) than all others. On line 173 there is a mistake in the figure caption, “(b) and (d) illustrate” should be just “(d) illustrates”.

[1] Sartzetaki, C., Antoniadis, P., Antonopoulos, N., Gkinis, I., Krasoulis, A., Perdikis, S. and Pitsikalis, V., 2023. Beyond Within-Subject Performance: A Multi-Dataset Study of Fine-Tuning in the EEG Domain. In 2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC) (pp. 4429-4435). IEEE.

### Questions
Please address the points outlined above in the “Weaknesses” section. If those are substantially addressed with a change in the paper’s content or with a convincing reason why this change is not feasible / desirable, the paper will be recommended for acceptance. 

Especially addressing the 1st point is necessary for recommendation of acceptance. Depending on how the rest of them are addressed, the rating could be increased to either 6 or 8.

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
3

### Summary
The paper proposes BTTA-DG, a Bayesian Test-Time Adaptation (TTA) framework for motor imagery (MI) EEG decoding. The method aims to address cross-subject and cross-session variability by enabling gradient-free adaptation during inference. It consists of three core components:
(1) a Sinc-based adaptive network (SincAdaptNet) for frequency-selective representation learning,
(2) a Dirichlet feature projection that probabilistically models deep feature uncertainty, and
(3) a GMM-driven Bayesian inference mechanism that calibrates model predictions without updating weights.

The framework is theoretically motivated and computationally efficient. By introducing Dirichlet modeling into the EEG-TTA setting, the paper provides a probabilistic view of feature dynamics and uncertainty estimation. Experiments demonstrate competitive accuracy and improved interpretability compared with prior TTA approaches.

### Strengths
- The authors successfully introduce Dirichlet modeling into the EEG-TTA decoding paradigm and achieve strong results on four public datasets, validating the method’s empirical effectiveness.

- The proposed BTTA-DG framework leverages GMM-driven Bayesian inference to avoid parameter updates during test time, achieving fast inference speed and making it suitable for real-world BCI deployment.

- The model effectively mitigates cross-subject and cross-session variability, demonstrating that a single model can generalize to new subjects and sessions at test time without gradient updates.

### Weaknesses
- The paper employs SincAdaptNet for pretraining, which contains only four convolutional layers. Given the limited size of EEG datasets, it remains unclear how such a lightweight backbone can ensure strong representational capability. If the contribution mainly lies in “extracting task-specific frequency bands,” this may be insufficient as a novel methodological point. The authors are encouraged to clarify the specific purpose and benefits of frequency-band extraction in the context of TTA and practical BCI applications.

- The Dirichlet distribution is introduced to parameterize the probabilistic distribution over class predictions. However, since the scale parameter α₀ significantly affects the concentration of the distribution, the authors should explain how α₀ is chosen or tuned, and analyze its impact on model stability and uncertainty estimation.

- Although Table 7 reports the average inference time, it would be beneficial to include a comparison of runtime and memory usage, to quantitatively demonstrate the computational efficiency of the Dirichlet modeling and strengthen the claim of practical advantage.

- While the GMM-driven Bayesian inference is theoretically sound, the interaction between Dirichlet feature projection and GMM posterior fusion remains insufficiently explained. The authors should further clarify how the historical GMM distribution influences the Dirichlet-projected features during test time, and whether it could cause bias toward recent samples. Providing additional conceptual explanation or visualization would help improve clarity.

- The ablation study is not sufficiently comprehensive. A more systematic evaluation, such as testing under cross-session and cross-subject settings and examining whether the improvements are consistent across multiple datasets, would substantially strengthen the empirical validity and generalizability of the proposed framework.

### Questions
- The paper claims that BTTA-DG achieves lower computational overhead than existing EEG-TTA methods. Quantitative evidence—such as comparisons of inference time, FLOPs, or memory usage—would be helpful to substantiate this claim.

- The proposed online TTA framework appears to operate only with a batch size of 1. It would be useful to clarify whether this is a methodological constraint or a design choice, as well as the rationale for this setting and its consistency with real-world BCI scenarios where input configurations may vary.

- Further clarification is needed on how the “historical distribution” in the GMM-based Bayesian inference is estimated and updated during test time. If this distribution is continuously adapted, it may lead to potential distribution drift or overfitting toward recent samples.

-The paper presents visualizations of learnable Sinc filters and claims physiological interpretability. A quantitative evaluation of whether the learned frequency bands align with known EEG rhythms (e.g., μ, β, γ bands) would strengthen the interpretability argument.

### Soundness
3

### Presentation
3

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
This paper presents a Bayesian test-time adaptation framework for EEG motor imagery decoding that addresses cross-subject and cross-session generalisation challenges. It leverages Euclidean Alignment (EA) to align EEG trials into a common domain, followed by a gradient-free Bayesian adaptation on low-dimensional Dirichlet-distributed sequential embeddings, extracted from pretrained SincAdaptNet models.  This method offers a gradient-free approach that avoids catastrophic forgetting and adapts robustly during test time.

### Strengths
The method emprically demonstrates its consistent outperformance across four well-known and diverse motor imagery EEG datasets with rigorous leave-one-subject-out and cross-session validation.

The projection of high-dimensional sequential EEG embeddings onto a compact Dirichlet-distributed parameter space is an innovative approach to capture predictive uncertainty and temporal evidence concentration.

The learned filters and Dirichlet feature spaces align with neurophysiological expectations, which supports the model’s interpretability and trustworthiness

### Weaknesses
The lack of an ablation study without EA precludes understanding of the standalone adaptation capabilities. This reliance could limit practical application scenarios where EA cannot be applied, e.g., variable or missing channels, high artifact environments.


The current approach processes entire trials for adaptation and inference, which is less practical for online BCIs requiring low latency and continuous feedback.

### Questions
Given the practical constraints on EEG hardware, benchmarking performance on datasets with fewer EEG channels is advisable. While BNCI and SHU datasets are well established, inclusion of standard datasets like BCI Competition IV Dataset 2b (3 channels) or datasets with low-density configurations would strengthen practical relevance.

For more realistic online BCI applications, investigating windowing or sliding window strategies, which segment data into shorter temporal windows for quick iterative adaptation and inference, would address practical BCI applications.

A recommended discussion point to include in the paper is addressing scalability and memory cost concerns for large-scale deployment across thousands of individuals, given the fact that each trial from each participant is represented and stored.

### Soundness
3

### Presentation
3

### Contribution
3
