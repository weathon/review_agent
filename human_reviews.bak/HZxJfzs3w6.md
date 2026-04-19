# Stochastic Flow Matching for Resolving Small-Scale Physics

- Decision: Reject
- Scores: 5, 6, 5, 5, 5

## Abstract
Conditioning diffusion and flow models have proven effective for super-resolving small-scale details in natural images. However, in physical sciences such as weather, super-resolving small-scale details poses significant challenges due to: $(i)$ misalignment between input and output distributions (i.e., solutions to distinct partial differential equations (PDEs) follow different trajectories), $(ii)$ multi-scale dynamics, deterministic dynamics at large scales vs. stochastic at small scales, and $(iii)$ limited data, increasing the risk of overfitting. To address these challenges, we propose encoding the inputs to a \textit{latent} base distribution that is closer to the target distribution, followed by flow matching to generate small-scale physics. The encoder captures the deterministic components, while flow matching adds stochastic small-scale details. To account for uncertainty in the deterministic part, we inject noise into the encoder's output using an adaptive noise scaling mechanism, which is dynamically adjusted based on maximum-likelihood estimates of the encoder’s predictions. We conduct extensive experiments on both the real-world CWA weather dataset and the PDE-based Kolmogorov dataset, with the CWA task involving super-resolving the weather variables for the region of Taiwan from 25 km to 2 km scales. Our results show that the proposed stochastic flow matching (SFM) framework significantly outperforms existing methods such as conditional diffusion and flows.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper tackles a difficult variant of super-resolution applied to weather data where the goal is to retrieve small-scale dynamics of (potentially different) physics variables from a coarser observation grid. The authors build upon Mardani et al., 2023, proposing to disentangle the deterministic part (modeled with a U-Net or a 1x1 Conv.) and the stochastic part (modeled with flow matching). Conversely to Mardani et al., 2023, the authors jointly trained both parts by dynamically scaling the influence of the stochasticity in order to prevent overfitting.

### Strengths
The paper is well written (despite some minors typos) and easy to follow. The method is well explained and sounds reasonable. The authors conducted an impressive set of experiments, anticipating well potential questions of the reader.

### Weaknesses
**Major**
- The authors claims that their method is *specifically tailored for data-limited regimes*, and mention this as a key limitation of Mardani et al., 2023 compared to their approach. Yet, no experiment explore low data-regime. The ERA5-CWA dataset contains 4 years of hourly sampled data, and the "toy" dataset contains 100k points.
- I found section 4.2 / 4.3 rather unconvincing, especially the intuitive motivation for the adaptive noise scaling. I think these sections could benefit from a clarification from the authors (see related questions).

**Minor**
- In the introduction, the authors stated that the method is designed to match *spatially* misaligned data, yet in the rest of the paper, the term "misalignement" seems to refers to difference between the PDE governing the low-resolution data and the high-resolution one. It is not clear to me why such misalignment can be considered spatial. If *spatial* misalignement does corresponds to differences in PDEs, then the first paragraph of A.4.1 describes it very well and probably should be in the main paper.
- Replaying the proof of Proposition 1, when injecting the proposed form of $v_\theta(x,t) = (x_t - D_\theta(x_t, t)) / (1-t)$ into equation (7), I found:
$$\|v_\theta(x_t, t) - (x-z)\| = \left\|\frac{x_t - D_\theta(x_t, t)}{1-t} - (x-z)\right\|$$
then using equation (8):
$$\|v_\theta(x_t, t) - (x-z)\| = \left\|\frac{(1-t)z + tx - D_\theta(x_t, t)}{1-t} - (x-z)\right\| = \left\|2z + \frac{2tx - x}{1-t} - \frac{D_\theta(x_t,t)}{1-t}\right\|$$
So I think the correct expression for $v_\theta$ should rather be $v_\theta(x,t) = \frac{x_t - D_\theta(x_t, t)}{t-1}$

**Grade motivation**
My grade is motivated mainly by my difficulties to fully understand what the proposed method actually learns, and consequently how it compares with pure deterministic or pure stochastic baselines. I suspect degeneration of the model toward a fully stochastic method, which seems supported by ablations, but contradicts main results in table 2. The "data-limited regime" claims is also confusing. The math typo is certainly a simple sign mistake, but also plays a role.

**Typo**
- Missing parenthesis in score function after equation (5)
- Repeated $\theta$ in A.1 (expression of $v_\theta$)

### Questions
**Major**
- Which parts of the model support your claim that it is *specifically tailored for data-limited regimes* ? Is there an experiment in the paper exposing the limitation of Mardani et al, 2023 in this context ?
- During training, how is $\sigma_z$  computed (using training or validation error) ?
- Both noise scaling and encoder regularization are motivated by the need to counter overfitting. Is there any proof in the paper that vanilla methods does suffer from overfitting that I could have missed ?
- If I understand correctly, the adaptive noise scaling will increase the noise amplitude if the deterministic part has large error. Then could it leads to a degenerate solution where $\sigma_z$ remains very large and most of the work is done by the stochastic part ? This seems supported by several results in the paper:
	- The simpler 1x1 conv. performs often better than UNet, which is unexpected, since UNet is more powerful. Could it indicates that the model is actually using more the stochastic part, and learns to ignore the deterministic part ?
	- Using the encoder regularization should prevent the degenerate case. And indeed, Table 6 shows that the regularization is helpful mostly for UNet. 
I think the paper could benefit from a study on the strength of $\sigma_z$, for instance by analyzing how corrupted is the output of the deterministic part (if the noise is of the same order of magnitude, then the model has degenerated ?). It would also be interesting (if available) to see the evolution of $\sigma_z$ during training of the model for both UNet and Conv 1x1.

**Minor**
- Can you please confirm the typo in the proof of proposition 1 ?
- Could you please elaborate on Sec.4 (Deterministic Dynamics) "it spatially aligns the base and target distribution *by matching their large-scale dynamics*" ? The sentence is unclear to me.
- The authors suggest that $\sigma_z$ prevents overfitting by increasing when the validation error increases. This intuition is only valid when $\sigma_z$  is computed from a validation loss, hence not online. Then, what is the intuition behind *online* method for scaling the noise ?
- If the temperature is deterministic, how the authors explains that the UNet Reg. baseline does not outperforms the other models on this variable ?

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
2

### Summary
The paper proposes stochastic flow matching for super-resolving small-scale physics in weather data, addressing challenges like data misalignment, multiscale dynamics, and limited data availability. The method combines an encoder that maps coarse-resolution inputs to a latent space with flow matching to generate fine-resolution outputs, while using adaptive noise scaling to balance deterministic and stochastic components. Experiments demonstrate that SFM outperforms existing methods like conditional diffusion and flow models.

### Strengths
- The presentation and writing are good

- The paper presents a novel combination of deterministic encoding with stochastic flow matching for solving the task

- The method's ability to handle misaligned data and limited data scenarios makes it valuable for certain applications

- Experiments on synthetic and real-world datasets demonstrate the effectiveness of the method

### Weaknesses
A theoretical analysis of SFM's convergence properties would be interesting, especially regarding the impact of the adaptive noise scaling mechanism on convergence guarantees

### Questions
- Could you provide more details about the hyperparameter selection process?

- Have you identified specific failure cases or limitations of the method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a framework that becomes stochastic stream matching for learning to recover small-scale physical fields. Specifically, they first use an encoder to learn to regress deterministic dynamics, and then add stream matching to learn the distribution of fine-grained stochastic dynamics. They validate the algorithm on multi-scale Kolmogorov flow and regional weather downscaling datasets.

### Strengths
1. The presentation is apparent and the description of the proposed methodology is accurate and easy to follow.

2. The figure and quantitative comparisons are good, the experiment is sufficiently adequate, the experimental setup is described in detail with methodological details and it should be easily reproducible.

### Weaknesses
1. The problem of novelty, due to the well-known connection between flow matching and diffusion modeling. The approach proposed in this paper seems to resemble CorrDiff. Although the authors emphasize the difference in Section 4.4. They claim that CorrDiff is trained in two stages, i.e., the regression encoder is trained first and the diffusion of the residual components is trained afterwards. In contrast, it seems that in this paper, the two components are only trained jointly, and the final loss is the sum of these two components.

2. In addition, the authors claim that the first stage of CorrDiff may run the risk of overfitting. This reasoning seems insufficient to me, and the authors don't seem to have relevant evidence. There is also no guarantee that joint training avoids this risk, besides this can be solved perfectly well using simpler ways, such as early stopping.

3. The idea of co-training encoders and diffusion models is also not new, and in fact he has already proposed it for tasks such as speech synthesis [1,2] and precipitation nowcasting [3].

[1] Popov V, Vovk I, Gogoryan V, et al. Grad-tts: A diffusion probabilistic model for text-to-speech[C]//International Conference on Machine Learning. PMLR, 2021: 8599-8608.

[2] Chen Z, He G, Zheng K, et al. Bridge-TTS: Text-to-Speech Synthesis with Schrodinger Bridge[J].


[3] DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a stochastic flow matching approach for super-resolving small-scale details. Specifically, it uses an encoder to transform inputs into latent images that are more aligned with the ground truth images. The additional noise is added to the latent images to introduce uncertainty for the following flow matching. The authors further present a modified flow matching objective that involves the noisy latent in training. Moreover, the maximum likelihood learning is proposed to tune the noise scale and an additional regularization is used in encoder learning for better generalization. The proposed method is evaluated on multi-scale weather datasets and synthetic PDE datasets.

### Strengths
1. The overall idea for cross-modality image translation is promising.
2. The proposed stochastic encoder-denoiser pipeline is interesting. 
3. Extensive Experiments illustrate the effectiveness of the proposed method, and the results on most datasets are good.

### Weaknesses
1. If the authors want to add uncertainty to the deterministic part, maybe the better way is to apply approaches like VAE to output both mean and variance (the downscale layers can be removed if there is no resolution change). Using VAE can also avoid the tuning of noise scales.
2. Also, calling this method "stochastic flow matching" is somehow improper, since the flow matching is usually an ODE. Note that the original flow matching also accepts noise input.   Adding the "stochastic" to "flow matching" makes it confusing and readers might think it's a general method converting flow matching to an SDE.
3. In Proposition 1, how do you define the velocity field $v$? Moreover, I don't think the term $(1-t)$ can be simply ignored since it changes over time. At least the authors should add some experiments to show that ignoring the $1-t$ could achieve the same performance.
4. As stated in Sec 4.2, the noise level is highly related to x and y and thus this work proposes a maximum likelihood approach to tune it. My question is why not learn it through the encoder directly? It should be easier and can be trained in an end-to-end manner as it is also involved in the training objective.
5. Adding the encoder regularization is too intuitive and would definitely affect the learning of flow matching. More details and ablation experiments should be provided.
6. The 1 x 1 conv layer encoder performs surprisingly well compared to the complex UNet encoder. Can the authors provide more analysis on it? Why a linear transform can lead to better performance?
7. In appendix A.1, the last paragraph, it should be D_{\theta}.

### Questions
See "Weaknesses".

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces the Stochastic Flow Matching (SFM) framework, a novel approach to super-resolve small-scale physics in meteorological data, particularly in downscaling weather variables for regions covering Taiwan. The authors address three key challenges: spatial misalignment between input and output distributions, deterministic vs. stochastic dynamics across scales, and limited data, which can lead to overfitting. SFM leverages an encoder to map coarse-resolution inputs to a latent distribution aligned with the target fine-resolution, which is refined using flow matching to reconstruct small-scale stochastic details. An adaptive noise scaling mechanism, based on maximum-likelihood estimates, helps balance deterministic and stochastic components. The framework is tested on real-world CWA weather data and the PDE-based Kolmogorov dataset, consistently outperforming existing methods, such as conditional diffusion and flows, in terms of RMSE and fidelity across several variables and spatial resolutions.

### Strengths
- The authors conducted comprehensive experiments on both synthetic and real-world datasets.

- The paper includes ablation studies and spectral analysis, which reveal the impact of different architectural choices and underscore SFM's robustness in capturing high-frequency details.

- The dynamic adjustment of noise levels based on the encoder’s prediction error provides a robust method to balance deterministic and stochastic data components.

### Weaknesses
- It is not clear why the method is named and formalized as "flow matching" because the actual implementation is based on EDM. The perturbation and the denoising model both follow the diffusion custom instead of flow matching. Please justify whether the method especially the probability path is closer to flow matching or diffusion.

- In Algorithm 2. the inputs to the denoising model $\mathcal{D}_\theta$ are $(x, t)$ but elsewhere $(x, \sigma)$. This may lead to confusion on how the adaptive noise scaling can affect sampling.

-  Application of the proposed method to other domains is unexplored, limiting the generalizability of the findings. It seems that the method can be applied to general image-to-image translation tasks, for example restoration of natural images which also involves stochasticity in generation of local details.

- As acknowledged in the paper, SFM relies on paired input-output data, which may not always be feasible in real-world applications, particularly for unpaired or missing datasets.

- Improvement upon baseline method CorrDiff is not consistent (especially temperature in Table 2).

- A minor point that does not affect the score (if the method still modelled as flow matching).  SFM/SfM is commonly used in the broad field of AI to refer to Structure from Motion, which could lead to confusion, particularly if the paper is targeted at a broader audience that includes computer vision researchers.

### Questions
- See weaknesses 1 and 2.

- I wonder how the noise variance in the adaptive noise scaling varies in the training run, and if the effectiveness is sensitive to the hyperparameter setup.

### Soundness
2

### Presentation
3

### Contribution
2
