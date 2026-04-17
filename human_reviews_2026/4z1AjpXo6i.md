# Continuous Time Series Generation with Irregular Observations

- Decision: Reject
- Scores: 6, 4, 8, 2

## Abstract
Time series generation (TSG) contributes to diverse fields (e.g., healthcare), but most methods assume regularly sampled data and fixed outputs—mismatched with real-world settings where observations are irregular and sparse. 
This mismatch is especially problematic in domains such as clinical monitoring, where irregularly recorded data must support downstream tasks with continuous and high-resolution data.
Neural Controlled Differential Equations (NCDEs) show significant potential in handling irregular time series, but still face challenges in learning dynamic temporal patterns and continuous TSG.
To address this, we propose MN-TSG, a framework that explores MOE (Mixture of Experts)-NCDE and integrates it with existing TSG models for irregular or continuous TSG tasks. 
The key designs of MOE-NCDE are the dynamic functions with mixture of experts and the decoupled design to better optimize the MOE dynamics. 
Further, we employ the existing TSG model to learn the joint distribution of the mixture of experts and the time series. In this way, the model can not only generate new samples but also produce suitable experts for them to enable MOE-NCDE for refined continuous TSG tasks. 
We have validated the effectiveness of our method on ten public and synthetic datasets, outperforming advanced TSG baselines in both irregular-to-regular and irregular-to-continuous generation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MN-TSG, a framework for continuous time-series generation with irregularly sampled observations, which extends Neural Controlled Differential Equations (NCDEs) with a mixture-of-experts and diffusion generative models. It proposes a Mixture-of-Experts NCDE (MoE-NCDE) method that can capture varied temporal dynamics via a learnable router for weightedcombination.
It employs a decoupled training strategy by pretraining a channel-wise autoencoder before learning dynamics.
Finally, the model integrates a joint diffusion generator that simultaneously models observations and expert weights.

### Strengths
1. The MoE-NCDE extends NCDE approach by incoroporating a MoE method. Standard NCDEs jointly train the encoder, dynamics, and decoder, making optimization unstable. Here, the authors pre-train an autoencoder to help NCDE focuses solely on learning the MoE-dynamics.

2. The model trains a diffusion generator G not only on time-series samples O but also on their expert-weight vectors S.

3. Experiment designs are comprehensive. The results show improved dynamic realism and smoother interpolation compared to NCDE, latent ODE, and imputation-based baselines.

4. The visualization of tsne and density plot helps interpret the generated data, enhancing its interpretability.

### Weaknesses
1. The paper use existing NCDE, MoE, decoupled training, and diffusion models. The main novelty lies in combining them with NCDEs rather than introducing new theoretical mechanisms.

2. It is not clear how to select the expert number and how does this hyperparameter impact to the generalization. 

3. This paper may incorporate the privacy analysis about whether we can recover the patient records.

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MoE-NCDE, a framework combining a mixture-of-experts neural controlled differential equation (CDE) with a diffusion model to generate continuous-time trajectories from irregularly sampled observations. The CDE component is used to construct the continuous dynamics from irregular observations, while the diffusion model learns the joint distribution of the regular observations and expert weights.

### Strengths
1. The proposed method integrates a continuous-time dynamic layer (MoE-NCDE) on top of discrete diffusion generation, helps address diffusion’s discretization constraint. 
2. The paper includes multiple datasets, metrics, and ablation studies, with reproducible implementation details given.

### Weaknesses
1. Lack of justification for the CDE choice:

1.1 The CDE component requires spline-interpolated control paths to perform integration, meaning it cannot generate trajectories by itself and must rely on the diffusion model. However, continuous-time formuations such as ODEs or SDEs can already generate continuous trajectories directly from initial conditions, fully addressing the irregular-to-continuous generation task stated in the title. No comparison with these alternatives Is provided. 

1.2 The spline-interpolation imposes a strong smoothness prior on the control path, which conflicts with the target domains (e.g. healthcare and finance) where real processes often involve abrupt events or shocks. As a result, the generated trajectories may appear smooth but fail to reflect the irregular and event-driven nature of such data. 

2. The proposed MoE-NCDE introduces multiple experts, but without explicit constraints or diversity objectives, there’s no guarantee that different experts learn distinct dynamic behaviors. Each expert is optimized jointly under the same loss, receiving similar gradients through soft gating. Consequently, the MoE may effectively function as a single, large network, and the reported performance gains could primarily reflect increased model capacity rather than meaningful expert specialization. 
3. In Appendix B.2, the generated trajectories appear jagged, while there’s no ground-truth data for the interpolated intervals, so the additional dynamics cannot be verified and may simply be artifacts of the model’s interpolation.

### Questions
Please see the Weaknesses part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper tackles the problem of continuous time series generation from irregularly observed data, a scenario highly relevant to domains such as healthcare and finance where data points often arrive at uneven intervals. The authors introduce MNTSG, a framework that innovates on Neural Controlled Differential Equations (NCDEs) by introducing a mixture-of-experts (MOE) dynamic function architecture and a decoupled training strategy. This structure is further integrated with diffusion-based generative models to jointly model time series samples and their dynamic function assignments. Extensive experiments across public, medical, and synthetic datasets show the proposed approach outperforming established baselines, with both quantitative and qualitative evaluation.

### Strengths
1. The motivation and research gap is clearly articulated. 
2. The proposed method design is clearly motivated.
3. The paper presented strong and comprehensive evaluation including various qualitative and quantitive aspects.

### Weaknesses
1. The motivation choose MoE to learn the complex various temporal dynamics is not well described. Is there any hidden correlations? Or do you observe any specific temporal dynamics can be learnt from MoE?
2. About the generated data evaluation, although current format is quite strong and follow previous standard evaluation setting, why not apply to real irregular downstream tasks to evaluate the quality of the generated data? 
3. The paper uses channel-wise autoencoder for reconstruction task. What's the trade-off between channel-wise versus channel-correlated stucture?
4. The overall framework still looks like assembly of existing modules, the key innovation of proposed methods is somehow hidden.

### Questions
see above

### Soundness
3

### Presentation
4

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
The paper proposes a new model called MOE-NCDE for generating continuous time series from observations over an irregular time grid. MOD-NCDE consists of three main components: 1) a Mixture of Neural Controlled Differential Equations (NCDE), 2) a routing function to weight the experts, and 3) a diffusion model. The authors propose a decoupled training mechanism where the vector field of NCDE is trained, freezing the weights of the pretrained reconstruction model. MOE-NCDE is used to convert irregular time series into regular time series, and then a diffusion model is trained on the regular time series jointly with the MoE weights. Experimental results on various generation and prediction tasks demonstrate that the proposed model has superior performance.

### Strengths
1. The authors attempted to address an important task of generating continuous time series from Irregular data. It has wide applications in interpolation, where the distribution of the missing data can be computed.

2. Combining multiple components, such as MoE, NCDE, and Diffusion Models, is new.

3. Experimental results on multiple datasets for various tasks demonstrate that the proposed model is superior compared to the baseline models.

### Weaknesses
1. While the paper is understandable at a high level, many components are understated.

i) A clear problem statement is missing. It is not clear what exactly the task is aimed at: probabilistic interpolation, probabilistic forecasting, or purely generating new sequences.

ii) An explanation of the pretraining part for computing $z(0)$ and the reconstruction function is not provided.

iii) The training process of the MOE router $\mathcal{R}$ is not described.

2. Several related works are missing. While the main comparisons are with NCDE-based models, other works [1,2] can also perform the generation task. Both models can predict the distribution of observations at any time point given irregularly sampled data.

[1] Shukla et al., "Heteroscedastic Temporal Variational Autoencoder for Irregularly Sampled Time Series," ICLR 2022

[2] Yalavarthi et al., "Probabilistic Forecasting of Irregularly Sampled Time Series with Missing Values via Conditional Normalizing Flows," AAAI 2025

### Questions
1. What exactly are the channel-wise encoder and decoder? Given that the notation on p. 3 suggests univariate time series, how does “channel-wise” apply here?

2. In line 7 of Algorithm 1, is only one expert being used? It is unclear how the top-K experts are utilized, and Eq. 3 does not clarify this.

3. The model seems to work only for univariate irregular time series. Is there a way to extend it to multivariate series?

4. How does the diffusion model ensure that the generated ranks for the router are positive and sum to 1?

### Soundness
2

### Presentation
1

### Contribution
2
