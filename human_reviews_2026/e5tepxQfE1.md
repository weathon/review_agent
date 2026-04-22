# Generative Diffusion Prior Distillation for Long-Context Knowledge Transfer

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
While traditional time-series classifiers assume full sequences at inference, practical constraints (latency and cost) often limit inputs to partial prefixes. The absence of class-discriminative patterns in partial data can significantly hinder a classifier’s ability to generalize. This work uses knowledge distillation (KD) to equip partial time series classifiers with the generalization ability of their full-sequence counterparts. In KD, high-capacity teacher transfers supervision to aid student learning on the target task. Matching with teacher features has shown promise in closing the generalization gap due to limited parameter capacity. However, when the generalization gap arises from training-data differences (full versus partial), the teacher’s full-context features can be an overwhelming target signal for the student’s short-context features. To provide progressive, diverse, and collective teacher supervision, we propose Generative Diffusion Prior Distillation (GDPD), a novel KD framework that treats short-context student features as degraded observations of the target full-context features. Inspired by the iterative restoration capability of diffusion models, we learn a diffusion-based generative prior over teacher features. Leveraging this prior, we posterior-sample target teacher representations that could best explain the missing long-range information in the student features and optimize the student features to be minimally degraded relative to these targets. GDPD provides each student feature with a distribution of task-relevant long-context knowledge, which benefits learning on the partial classification task. Extensive experiments across earliness settings, datasets, and architectures demonstrate GDPD’s effectiveness for full-to-partial distillation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Generative Diffusion Prior Distillation (GDPD), a novel knowledge distillation framework that treats the teacher’s full-context representation as a generative prior and uses diffusion models to approximate the distribution of teacher features. During distillation, the student’s partial observations are regarded as degraded evidence, and posterior sampling from the diffusion prior produces plausible teacher feature reconstructions that serve as soft supervision. This approach effectively bridges the gap between full-context teachers and partial-context students in time-series or incomplete-modality settings. Extensive experiments on UCR, UEA, and PhysioNet datasets demonstrate consistent gains over standard KD and feature-based methods, accompanied by detailed ablation studies and reproducibility details.

### Strengths
1. The paper frame the teacher’s representation as a learnable generative prior, leveraging diffusion modeling to capture the intrinsic uncertainty and diversity of knowledge transfer. This view enriches the typically deterministic KD paradigm with a stochastic, distributional interpretation.
2. The experiments are extensive and systematically designed, with convincing improvements and well-controlled ablations.

### Weaknesses
1. While GDPD is conceptually elegant, the paper’s treatment of the diffusion prior remains largely phenomenological—its success is empirically shown, yet its mechanistic role in shaping the student’s feature space is underexplored. The work stops short of analyzing what kind of uncertainty the diffusion prior captures (epistemic vs. aleatoric), or how this aligns with the teacher’s representational manifold. Consequently, the “why” behind the gains remains somewhat opaque.
2. Moreover, the approach introduces additional training overhead through diffusion sampling and warm-up phases, but the paper does not quantify these costs or discuss trade-offs between sample diversity and efficiency. Finally, while the method is positioned as general, all evaluations are within classification tasks; its potential limitations on forecasting, regression, or multimodal fusion remain untested, leaving open questions about scalability beyond the current scope.

### Questions
1. To what extent does the diffusion prior capture epistemic uncertainty about the teacher’s manifold, as opposed to merely adding stochastic regularization? Could visualizing the learned diffusion trajectories help clarify this distinction?
2. The GDPD assumes the teacher’s feature space forms a coherent generative manifold. How sensitive is the approach to teacher architectures where intermediate representations are poorly structured?

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
The paper tackles classification from partial time-series prefixes by distilling knowledge from a teacher trained on full sequences.
Extensive experiments across earliness settings, datasets, and architectures demonstrate GDPD’s effectiveness for full-to-partial distillation

### Strengths
Generally clear and readable; figures and tables are informative; Codes are publicly available.

### Weaknesses
1. Training a diffusion model may additionally introduce significant computation and memory
2. compared baselines (KD and FitNet) are significantly out-of-dated
3. the paper lacks a formalization of the posterior being sampled

### Questions
see weakness

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
3

### Summary
This paper proposes a new method to train time series classifiers that operate on partial sequence prefixes to achieve similar generalization performance to those trained on full sequences. The authors argue that the generalization gap in this case is due to a representation gap, as the full context features from teacher models cannot be directly distilled into the partial context features from student models. To solve this, a diffusion model is trained to obtain a generative prior over full context features, which can then be used to posterior-sample completed features given the partial features from the student model. The student model is then optimized to produce partial features such that its corresponding posterior full feature samples have high classification performance.

### Strengths
1. This paper introduces an interesting formulation for the partial time series prefix classification problem, which identifies the representation gap as a key reason why conventional distillation methods do not work as expected.

2. The proposed method is novel, which leverages powerful diffusion priors to obtain posterior full features given partial features for the downstream classification task. The learning objective is effective since it directly optimizes to student model to achieve highly discriminative performance given the posterior full feature samples.

3. The authors provide some reasonable justifications for why the proposed method works, which is much appreciated.

4. Extensive experiments with ablation studies are performed the validate the proposed method's empirical performance. However, I cannot evaluate the experimental results, since I'm not familiar with time series classification benchmarks and baselines.

### Weaknesses
1. My main concern is the computational complexity of the proposed method in all training stages. First, in the pre-training stage, it involves pre-training (i) a teacher classifier on full sequences; (ii) a diffusion model on the full features extracted from one of the middle layers of the teacher classifier. Then, when training the student model, each training step requires sampling from the posterior distribution, which involves simulating the reverse diffusion process of the diffusion model for multiple steps. This is significantly more costly than traditional distillation methods.

2. It is unclear why the noise-fusing approach is used for the conditioning mechanism for the diffusion model, given that classifier-guided/free guidance is more principled and widely used in many applications.

3. Limitations of the proposed method is not discussed in the paper.

### Questions
1. Could the authors comment on and provide some analysis of the trade-off between model performance and computational cost for the proposed method and other conventional distillation methods?

2. Could the authors clarify why the noise-fusing approach is used for diffusion model conditioning? How does it compared to classifier-free/guided guidance in this setting?

3. Could the authors discuss limitations of the proposed method in the paper?

4. Does z_long and z_short have the same dimensionality? From the paper, it is unclear whether the posterior sampling procedure is an impainting process that fills in the missing postfix in z_short or a transformation that modifies the entirety of z_short into z_long.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies classification from partial time-series prefixes and proposes GDPD, a distillation framework that learns a diffusion prior over teacher features and uses posterior samples—conditioned on the student’s short-context features—to supervise the student. The aim is to provide progressive, diverse, and distributional targets rather than single-point matches. Across UCR/UEA benchmarks and a PhysioNet case study, GDPD improves accuracy and teacher–student fidelity under multiple earliness levels, channel-wise partialness, compression, and self-distillation.

### Strengths
- The work focuses on full-context → partial-context distillation and explains why direct feature/logit matching can overwhelm the student under representation mismatch.
- Treating teacher representations as a diffusion-learned prior and supervising with distributional, progressive targets is a substantive shift from standard KD.
- Consistent gains over Base, Fits, and other KD baselines across earliness settings; better teacher–student agreement; robustness to channel-wise partialness; and utility for compression and self-distillation.
- Warm-up scheduling, loss-weight trade-offs between task and GDPD, and the effect of sampling multiplicity J provide insight into what drives improvements.
- The PhysioNet ICU study suggests the approach remains effective under heterogeneous conditions (partial channels, imbalance, cross-task).

### Weaknesses
- Training adds a diffusion prior and posterior sampling within student optimization. Please report wall-clock and step-level overhead versus Fits/Logit-KD (e.g., epochs × diffusion steps, training/inference cost).
- Performance may depend on which teacher layer is modeled and which student features are aligned. Clear guidance, cross-layer experiments, or robustness checks would help practitioners.
- Please discuss applicability beyond time series (e.g., images or text under partial inputs) and clearly state assumptions and limitations to contextualize the results and guide follow-up work.

### Questions
- How is $z_{\text{long-hint}}$ defined for a given example—what subset of the teacher feature manifold counts as useful hints of $z^*_{\text{long-ideal}}$? If $z_{\text{long-hint}}\sim p(z_{\text{long}})$ is just a generic draw, how does it differ from standard teacher features, and why doesn’t this construct appear later in the method/experiments? Please clarify the relationship between the posterior reconstruction $\hat z_{\text{long}}$ and the hints, or state that $z_{\text{long-hint}}$ is expository only.
- Which teacher/student layers define $z_{\text{long}}$ and $z_{\text{short}}$ across architectures? Have you tried cross-layer modeling (multiple blocks) or multi-scale priors?
- How much training time does GDPD add on typical settings, and how sensitive are results to diffusion steps/samples and key hyperparameters? Any guidance on trade-offs between speed and accuracy?

### Soundness
3

### Presentation
3

### Contribution
3
