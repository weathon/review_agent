# Stochastic Optimal Control for Continuous-Time fMRI Representation Learning

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Learning robust representations from functional magnetic resonance imaging (fMRI) is fundamentally challenged by the temporal irregularity and noise inherent in data from heterogeneous sources. Existing self-supervised learning (SSL) methods often discard critical temporal information by discretizing or averaging fMRI signals. To address this, we introduce a novel framework that reframes SSL as a Stochastic Optimal Control (SOC) problem. Our approach models brain activity as continuous-time latent dynamics, learning a robust representation of brain dynamics by optimizing a control policy that is agnostic to the temporal irregularity. This SOC framework naturally unifies masked autoencoding (MAE) and joint-embedding prediction (JEPA) to extract compact, control-derived representations. Furthermore, a simulation-free inference strategy ensures computational efficiency and scalability for large-scale fMRI datasets. Our model demonstrates state-of-the-art performance across diverse downstream applications, highlighting the potential of the SOC-based continuous-time representation learning framework.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes BDO, a deep latent model for self-supervised representation learning from fMRI time series with heterogeneous temporal resolutions. The approach formulates representation learning as a stochastic optimal control (SOC) problem over continuous-time latent dynamics governed by SDEs, where the control policy acts as an encoder. The model combines masked autoencoding (MAE) and joint-embedding prediction (JEPA) in a single SOC framework and introduces a simulation-free inference scheme for computational efficiency. Trained on large-scale resting-state fMRI (UK Biobank) and evaluated on multiple public datasets (HCP-A, ABIDE, ADHD200, HCP-EP), BDO achieves higher downstream task performance (e.g., age and diagnosis prediction) compared to prior SSL baselines.

### Strengths
- Tackles a highly relevant and timely topic: representation learning for continuous, irregularly sampled neuroimaging data.
- The SOC formulation is mathematically elegant and could offer a principled way to model latent neural dynamics.
- Integration of MAE and JEPA in a single control-theoretic objective is technically interesting.
- Results on multiple benchmarks are empirically strong and suggest robustness across datasets.

### Weaknesses
1.	Limited insight into learned dynamics.
The latent SDE component is central, but no qualitative or quantitative analysis of the learned latent trajectories or their neurobiological structure is provided. As a result, it is unclear whether the model learns meaningful neural dynamics or primarily acts as an advanced denoising / feature-compression scheme.

2.	Clarity and accessibility.
The exposition is mathematically dense and often assumes familiarity with SOC and variational inference. The connection to reinforcement learning, while conceptually appealing, is not sufficiently explained or justified. Figures are mostly schematic and do not substantially aid understanding of the pipeline.

3.	Positioning and novelty.
The work seems closely related to recent latent SDE and neural ODE formulations for neuroimaging (e.g., ElGazzar & van Gerven, 2024), yet these are not cited or compared. It remains unclear whether the main novelty lies in the SOC framing or in the MAE-JEPA combination, and whether either leads to qualitatively different behavior.

4.	Evaluation design.
The chosen downstream tasks (e.g., age regression, gender classification) are convenient but not strongly motivated as demonstrations of dynamic modeling. More suitable evaluations—such as temporal forecasting, dynamic functional connectivity analysis, or representational alignment across TRs—would have made the contribution clearer.

5.	Presentation and writing.
The introduction and motivation sections are somewhat opaque; key intuitions are missing. While the results are broad, the paper feels like a mathematically strong model attached post-hoc to standard neuroimaging tasks rather than driven by neuroscientific questions.

ElGazzar, A., & Gerven, M. van. (2024). Generative Modeling of Neural Dynamics via Latent Stochastic Differential Equations (No. arXiv:2412.12112). arXiv. https://doi.org/10.48550/arXiv.2412.12112

### Questions
- Did you examine or visualize the inferred latent trajectories to assess whether they capture meaningful brain dynamics (e.g., temporal smoothness, subject-specific signatures)?
- How sensitive are the results to the dimensionality of the latent space?
- How do your “simulation-free” results compare against a full latent-SDE baseline with numerical solvers?
- Could the same results be achieved by a neural SDE-VAE or GRU-ODE-Bayes setup without the SOC formalism?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a self-supervised learning framework called Brain Dynamics with Optimal Control (BDO) particularly developed to meet the challenges arising when dealing with fMRI data. It consists in explicitly modelling continuous-time latent brain dynamics using stochastic optimal control (SOC). Unlike existing methods that discretize or average fMRI signals and lose temporal information, BDO treats brain activity as a stochastic differential equation (SDE), capturing multi-scale temporal structure while being robust to heterogeneous sampling protocols. Within the SOC formulation, the framework unifies two major SSL paradigms, Masked Autoencoding (MAE) and Joint Embedding Predictive Architecture (JEPA), allowing the model to learn compact, control-based representations of brain dynamics. Furthermore, to ensure scalability, the authors introduce a simulation-free inference strategy based on locally linear SDE approximations. Experiments on large multi-site fMRI datasets show that BDO achieves state-of-the-art performance and strong transferability across diverse downstream tasks.

### Strengths
First, I find paper exceptionally well written. It posses all key qualities: clarity, smooth flow of ideas, good balance presenting both intuition and technical complexity, appendix complements adequately the main text. 

Second, the paper tackles an important problem and proposes a method that is applicable beyond targeted fMRI applications. 

Third, the innovation to use SOC in SSL context is, up to my knowledge, clear and it also consists in combing MAE and JEPA.  While, focusing on linear drifts and exploiting Gaussian distributions to scale-up is relatively standard idea, it is executed well and is a very good first strategy for the proposed method.

Finally, the experimental setup and baselines are adequate, and clearly shows the main benefits coming from BDO.

### Weaknesses
While I do not find any strong weakness of the paper, here is a few minor issues:

**Minor**
- While full names of acronyms such as JEPA and MAE are introduced in the abstract, it would be helpful to still write them in the main text when the concepts are introduced.

- Typo in Eq. (3): in drift part should read $\alpha^\theta$

### Questions
Ca authors add positioning of their work w.r.t. following two nicked references, and potentially add this in Appendix A?

- Hassan et al. *Identifying Latent Stochastic Differential Equations*.IEEE Signal Processing 2021

- Bartosh et al.  *SDE Matching: Scalable and Simulation-Free Training of Latent Stochastic Differential Equations*. ICML2025

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Brain Dynamics with Optimal control (BDO), a novel self-supervised learning
(SSL) framework for learning representations from fMRI time-series. The core problem this work
addresses is the temporal irregularity and noise inherent in multi-site fMRI datasets, which is often
handled by existing methods by discarding or discretizing temporal information.

BDO reframes fMRI representation learning as a Stochastic Optimal Control (SOC) problem. It models
latent brain activity as a continuous-time stochastic differential equation (SDE) and learns a control
policy (parameterized by a Transformer) that steers a prior SDE to a posterior SDE that best explains
the observed, noisy fMRI data. This control policy itself serves as the learned representation.

The framework provides a principled unification of Masked Autoencoding (MAE) and Joint-Embedding
Predictive Architectures (JEPA). The SOC objective is shown to be equivalent to an ELBO (MAEstyle reconstruction), while a JEPA-style latent target (from an EMA-updated encoder) is introduced
to prevent the model from overfitting to signal noise. To ensure scalability, the paper introduces a
simulation-free inference method based on a locally linear SDE assumption, which admits a closed-form
solution. In addition, the model is pre-trained on the large-scale UKB dataset and demonstrates stateof-
the-art performance on a wide range of downstream tasks, including trait and diagnosis prediction,
on both internal (UKB held-out) and external (HCP-A, ABIDE, etc.) datasets.

### Strengths
1. Novel and Elegant Formulation: Reframing SSL for fMRI as a Stochastic Optimal Control
problem is a highly novel and theoretically compelling contribution. This approach is a natural
and powerful fit for modeling continuous-time, noisy, and irregularly-sampled data, which is a
key challenge in fMRI analysis.

2. Principled Unification of MAE and JEPA: The paper does an excellent job of motivating
and technically unifying two powerful SSL paradigms. Deriving the MAE-like objective from
the SOC formulation (Prop 2.1) is elegant. The subsequent introduction of a JEPA-style latent
target (Eq. 7-9) to regularize this objective and avoid overfitting to noise is a well-reasoned and
technically sound improvement.

3. Impressive Computational Efficiency: A major contribution is the simulation-free inference
(Theorem 2.2). By using a locally linear SDE approximation, the authors derive a closed-form
Gaussian posterior for the latent states, which can be computed in parallel with O(log k) complexity.
This makes a theoretically complex SDE/SOC framework computationally practical for
massive datasets like UKB. The empirical result of 15 GPU hours for pre-training (Table 10) is
extremely strong compared to baselines.

4. Strong and Comprehensive Empirical Validation: The experimental results are state-of-the-art and convincing. BDO outperforms a wide array of baselines (both task-specific and
fMRI-specific SSL models) across numerous diverse tasks (age, gender, diagnosis) and datasets
(UKB, HCP-A, ABIDE, ADHD200, HCP-EP). The generalization from resting-state pre-training
to task-based fMRI (Appendix D.5) is also a strong result.

### Weaknesses
1. Inconsistency in Control Policy Formulation: There appears to be a critical inconsistency
between the theoretical formulation and the practical implementation of the control policy α.
The SOC theory (Eq. 3, Prop 2.1) defines a closed-loop control policy α(t,Xθt ;Y), which depends
on the current latent state Xθt . This is standard for optimal control. However, the simulation-free implementation (Eq. 10) and the architecture diagram (Fig. 5) use what appears to be an
open-loop policy αθti(Y), which is pre-computed from the fMRI data Y and does not depend on
the current state Xθt . This is a significant simplification that deviates from the theory. While it
enables the efficient closed-form solution, the paper needs to explicitly state this simplification
and justify how the SOC theory (Prop 2.1) still holds for this open-loop policy.

2. Simplistic Final Representation: The model’s universal feature A is derived by a simple
mean-pooling of the control policy sequence {αt}. This design choice seems to contradict the
paper’s primary motivation, as it discards the rich temporal structure that the continuous-time
SDE model worked so hard to capture. It is surprising that this simple pooling outperforms
other methods, and it feels like a missed opportunity.

3. Clarity on Linear SDE Parameters: The locally linear SDE in Eq. 10 (dXθ
t = [−DtiXθt +αθti ]dt + dWt) is key to the model’s efficiency. However, it is unclear how the drift matrix Dti
is parameterized and learned. Theorem 2.2 states Dti = V ΛtiV ⊤, which implies it changes over
time. How are the shared eigenbasis V and the eigenvalues Λti generated by the encoder network?
This is a crucial, missing architectural detail.

4. Baseline Reproducibility: The authors note in Appendix D.11 that they could not reproduce
the Brain-JEPA baseline with their own preprocessing, which is a minor red flag. While they
provide a fair comparison using the original Brain-JEPA preprocessing (where BDO still wins),
it would be more convincing to understand *why* the reproduction failed.

### Questions
1. Can you please clarify the apparent inconsistency in the control policy? Is the implemented
policy αθti(Y) (open-loop) or α(t,Xθt ,Y) (closed-loop)? If it is open-loop, how is this reconciled
with the SOC theory in Proposition 2.1, which is based on a state-dependent policy?

2. The final feature A is a mean-pooling of the control policy, which discards all temporal information.
Why was this chosen? Have you experimented with using the learned latent states Xt
or a more temporally-aware aggregation of {αt} (e.g., attention, final hidden state) as the final
representation?

3. How is the drift matrix Dti = V ΛtiV ⊤ in the linear SDE (Eq. 10) parameterized? Does the
Transformer encoder output V and all {Λti}? Is V shared across all time steps and subjects?

4. In Appendix D.11, you mention a difference in per-sample zero-mean normalization might be
responsible for the failed Brain-JEPA reproduction. Could you elaborate on this? This seems
like a subtle difference to cause a full failure, and understanding it would strengthen the baseline
comparisons.

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
The authors propose a stochastic optimal control framework as a way to view fMRI pre-training. The model they end up with is intuitively similar to a variational auto encoder that is trained with masked auto encoding and is regularized with a Joint Embedding Predictive Architecture (JEPA) objective function. The model outperforms baselines across a variety of tasks and datasets, and the model exhibits good scalability.

### Strengths
The authors perform a wide range of experiments, describe an interesting framework for fMRI pre-training, and thoroughly ablate their model.

### Weaknesses
Major weaknesses:
1) No comparison to recent foundation models, and limited related work in the main text. The authors have moved most of their related work to the Appendix. Especially given how quickly the foundation model field is evolving, I believe it is important to have a more substantial related work section than is currently present in the paper. Moreover, a few key foundation models are missing from the evaluations, nor do they discuss in depth differences between their approach and BrainJEPA for example. Namely, BrainJEPA [1] and BNT [2]. I cannot recommend the paper for publication without fair comparisons to these works.
2) The authors claim that their stochastic optimal control framework solves heterogeneity in repetition time (TR) for the fMRI data, but it is unclear why this is true given the authors' model choices. Specifically, they mention on Lines 42-45 that "... existing SSL methods often fail to fully capture the inherent temporal dynamics of fMRI by treating the time-series data as segmented patches or static connectivity graphs as described in Figure 2.". However, this is not necessarily true. Transformers can still capture dynamics even when they use temporal patching. For example POYO [3] and NDT [4] that show that transformers can still capture dynamics even with temporal patching in the computational neuroscience domain. Moreover, the authors say their framework solves heterogeneous TRs, but still use a temporal transformer (see Figure 5) to create embeddings. It is unclear to me how the authors fix heterogeneous TRs when the transformer is pre-trained on a dataset with a certain spacing between patches. The current transformer does not seem to solve this issue even when incorporated into a stochastic optimal control policy.

Minor weaknesses/types/grammar:
- The authors keep mentioning (e.g. on Lines 76-79) that fMRI data is high-dimensional, which is true, but then also use anatomical regions of interest (ROIs) to reduce the data to 450 spatial dimensions. It is unclear to me why the authors seem to focus so much on the high dimensionality of the data when they do not contribute a specific way to address this fact.
- To the previous point, the authors do not ablate the ROI choice.
- Theorem 2.2 can be moved to the Appendix because it is not a novel theorem but a direct consequence of the papers cited in the previous paragraph. Especially given the fact that the authors moved their related work to the Appendix, I believe these types of derivations can be swapped for a more substantial related work section. If anything some of the mathematical derivations, including but not limited to this theorem, take away from the main contributions this paper makes.
- The "Masked auto encoders" paragraph on page 5 repeats many of the ideas/claims in paragraph 2.2, I think either of these paragraphs can be shortened.
- Lines 152-153: "The SOC proves ..." -> SOC provides
- Lines 242-243: "..., we use the MAE ..." -> we use an MAE

[1] Dong, Z., Li, R., Wu, Y., Nguyen, T. T., Chong, J., Ji, F., ... & Zhou, J. H. (2024). Brain-jepa: Brain dynamics foundation model with gradient positioning and spatiotemporal masking. Advances in Neural Information Processing Systems, 37, 86048-86073. \
[2] Kan, X., Dai, W., Cui, H., Zhang, Z., Guo, Y., & Yang, C. (2022). Brain network transformer. Advances in Neural Information Processing Systems, 35, 25586-25599. \
[3] Azabou, M., Arora, V., Ganesh, V., Mao, X., Nachimuthu, S., Mendelson, M., ... & Dyer, E. (2023). A unified, scalable framework for neural population decoding. Advances in Neural Information Processing Systems, 36, 44937-44956. \
[4] Ye, J., & Pandarinath, C. (2021). Representation learning for neural population activity with neural data transformers. arXiv preprint arXiv:2108.01210.

### Questions
1) On lines 84-85 what do the authors mean by "..., which encodes multi-scale brain activity onto a single continuous real-time axis."?
2) Line 121-122: Why do the authors assume the data is being governed by an Ito diffusion process? There is a rich literature of biophysical dynamical models for fMRI data, including the FitzHugh–Nagumo model [1], the Wilson-Cowan model [2], and the Kuramoto model [3]. The authors provide no biophysical or neuroscientific explanation for this design choice.
3) In the same paragraph on Lines 131-132 the authors mention "... we initially assume the underlying changes are purely random, acknowledging the difficulty in modeling the intricate nature of fMRI signals beforehand.". Why do the authors not just use a learnable prior or infer an initial condition?



[1] Ghosh, A., Rho, Y., McIntosh, A. R., Kötter, R., & Jirsa, V. K. (2008). Cortical network dynamics with time delays reveals functional connectivity in the resting brain. Cognitive neurodynamics, 2(2), 115-120. \
[2] Deco, G., Jirsa, V., McIntosh, A. R., Sporns, O., & Kötter, R. (2009). Key role of coupling, delay, and noise in resting brain fluctuations. Proceedings of the National Academy of Sciences, 106(25), 10302-10307. \
[3] Cabral, J., Hugues, E., Sporns, O., & Deco, G. (2011). Role of local network oscillations in resting-state functional connectivity. Neuroimage, 57(1), 130-139.

### Soundness
3

### Presentation
3

### Contribution
2
