# CountsDiff: A diffusion model on the natural numbers for generation and imputation of count-based data

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Diffusion models have excelled at generative tasks for both continuous and token-based domains, but their application to discrete ordinal data remains underdeveloped. We present \emph{CountsDiff}, a diffusion framework designed to natively model distributions on the natural numbers. CountsDiff extends the Blackout diffusion framework by simplifying its formulation through a direct parameterization in terms of a survival probability schedule and an explicit loss weighting. This introduces flexibility through design parameters with direct analogues in existing diffusion modeling frameworks.
Beyond this reparameterization, CountsDiff introduces features from modern diffusion models, previously absent in counts-based domains, including continuous-time training, classifier-free guidance, and churn/remasking reverse dynamics that allow non-monotone reverse trajectories. 
We propose an initial instantiation of CountsDiff and validate it on natural image datasets (CIFAR-10, CelebA), demonstrating the benefits of the proposed design space and that the framework scales to complex, high-dimensional data domains. We then highlight biological count assays as a natural use case, evaluating CountsDiff on single-cell RNA-seq imputation in a fetal cell and heart cell atlas. Remarkably, we find that even this simple instantiation matches or surpasses the performance of a state-of-the-art discrete generative model and leading RNA-seq imputation methods, while leaving substantial headroom for further gains through optimized design choices in future work.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces **CountsDiff**, a diffusion framework whose state space is the set of natural numbers. The forward process is a continuous-time pure-death birth–death process parameterized by a _survival_ schedule p(t)p(t)p(t); the reverse process is generalized to allow **attrition** (deaths) during sampling, enabling non-monotone trajectories akin to remasking/churn. The objective is a weighted NLL with a principled time weighting; the authors also add predictor-free guidance and propose randomized rounding to avoid small-count collapse. Experiments cover (i) toy sparse counts, (ii) CIFAR-10/CelebA generation, and (iii) scRNA-seq **imputation** (fetal and heart cell atlases), where CountsDiff is competitive with discrete diffusion (ReMDM/D3PM) and specialized imputation baselines.

### Strengths
- *Well written easy and structured:* The forward process and binomial marginals/conditionals are derived cleanly with an intuitive p(t)p(t)p(t) parameterization; the link to Gaussian schedules (via matched SNR) is useful for design transfer.

### Weaknesses
- *Novelty*: CountsDiff’s core forward process is a reparameterized pure-death process; the main novelty is the _design space framing_ (survival schedule + weights) and the **attrition** reverse sampler. This is solid but somewhat incremental relative to Blackout diffusion and continuous-time discrete denoising models.
- *Empirical results:* While the aim isn’t SOTA on CIFAR-10/CelebA, reported FIDs are modest and sometimes worse under cosine schedule without tuning; training steps for CelebA are halved due to compute, weakening claims there.

### Questions
- Gaussian diffusion on the toy dataset seems to be unexpectedly bad given that the authors were "unable to match the performance ..., even by varying model capacity". Can the authors provide some more details on the setting? And what was used for dequantization (i.e. unfiorm dequnatization, bitwise channel encodings,...)

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors develop a continuous-time diffusion framework (CountsDiff) that is suited for modeling natural numbers (count data). Specifically, the proposed framework extends a prior model, Blackout diffusion, by introducing features adapted from modern continuous diffusion models, such as continuous-time training, classifier-free guidance, and non-monotone reverse dynamics (attrition). The authors validate the framework on natural images and RNA sequences, showing it can match or outperform a SOTA discrete generative model .

### Strengths
1. The formulation of a pure death process as the forward corruption process and a generalized birth-death process (attrition) as the reverse sampling process provides an elegant and principled way of modeling distributions on the natural numbers.

2. The proposed framework successfully bridges the gap between modern Gaussian-based continuous diffusion models and discrete diffusion models in count-based domains, specifically by translating and incorporating powerful techniques such as continuous-time training, classifier-free guidance, and non-monotone reverse dynamics. This greatly unlocks the modeling potential of diffusion models for count data.

### Weaknesses
1. The experiments are relatively weak and somewhat unconvincing. Despite all the technical innovations, the experimental results of CountDiff do not fully demonstrate its advantages. For example:

   i. On natural images, there are no other baselines except for the variants of CountDiff itself. It is difficult to tell how well the proposed method performs without a direct comparison to other methods under the same setting (*e.g.*, same model architecture and training steps).

   ii. In Table 1, it seems that to achieve good sampling results (FID and IS), both guidance $\gamma$ and attrition schedule $\eta_{\text{rescale}}$ have to be tuned. It would be helpful to include ablation results showing the individual effect by just changing one of them.

2. The paper fails to mention some highly-relevant prior work [1]. Upon a quick research, it appears that Chen et al. (2023) [1] developed a similar discrete-time forward and backward processes that are suited for modeling count data, which the authors termed binomial thinning and Poisson thickening processes respectively. In [1], an identical loss function was introduced to train the model. However, it does not explicitly acknowledge nor discuss the established foundation laid by this prior work.

[1] Chen, Tianqi, and Mingyuan Zhou. "Learning to jump: Thinning and thickening latent counts for generative modeling." International Conference on Machine Learning. PMLR, 2023.

### Questions
1. Although the paper formulates CountDiff as a continuous-time diffusion framework for count data. It is unclear what the additional benefits are to adopt the continuous-time training. Unlike the Gaussian diffusion, to my understanding, CountDiff still have to go through a fixed number sampling steps as described by Algorithm 2. Consequently, the major drawback of slow sampling remains unresolved in this new framework.

2. Can the CountDiff be adapted for modeling non-negative continuous data like [1]? In [1], a trick named the Poisson-based data randomization is introduced to handle both natural numbers and non-negative real values.

3. The main quantitative comparison (Table 1) reports the best FID and IS for different combinations of guidance ($\gamma$) and attrition ($\eta_{\text{rescale}}$). Could the authors clarify how to determine a good combination of the two hyperparameters in practice?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a discrete diffusion model that builds off of Blackout diffusion with a reparameterization of the forward process, which enables the large toolkit from Gaussian diffusion (for reals) to be applied in the natural number case. The corresponding denoising process (with generalization over birth-death processes), guidance, and rounding procedures are worked out. On simulated counts, discrete diffusion methods easily capture the data distribution where Gaussian diffusion fails. On CIFAR10, countsdiff produces competitive qualitative and quantitative results. Finally, the authors test the method on imputation for transcriptomics data.

### Strengths
I will contextualize both my positive and critical comments by acknowledging that I’m not very well versed in the details of existing discrete diffusion methods, and so may not be able to judge the conceptual advances (or the lack thereof) of the current work:

- the paper tackles an important problem in developing diffusion models more appropriate for discrete data with an elegant and more general framework building off of existing works.
- the results are solid, and while not overwhelmingly superior than existing methods, provide some interesting insights on the method, such as the effect of attrition scale on blurring and the newly incorporated noise schedules
- the paper is overall clearly written with concise language, and clearly motivates the problem being tackled relative to existing solutions

### Weaknesses
- While the framework developed here is elegant, it’s ultimately unclear to me how well it performs compared to existing approaches without the additional proposed generalization. See questions for specific comments. To be clear, I’m not saying that this work is inferior or uninteresting because it does not beat the SOTA on those benchmarks, but that it’s unclear to me as a relatively naive reader what the value of the proposed generalization is aside from conceptual elegance (which I appreciate)

### Questions
- It’s great that the method was validated on simulated data, but it’s somewhat expected to outperform a Gaussian model and that’s not really the interesting comparison here. Are there some synthetic experiments (e.g., with higher dimensional toy data) one can come up with that better emphasize the benefits of the generalization compared to existing discrete diffusion methods?

- Similarly, while qualitative results on CIFAR10 look fine and show the value of guidance, attrition, and a cosine p-schedule, it’s unclear to me how they compare quantitively to existing methods (esp. in Table 1). This may be solved, for example, by simply noting that one of the previous methods (e.g. Blackout) is equivalent to which noise schedule.

- On the RNA seq data as well, the current method is comparable but not clearly superior to existing methods, in particular remasking diffusion. The authors reason why some metrics are less meaningful. But as a non-domain expert, a more explicit demonstration of these points (e.g., via a downstream task) may help the authors’ case.

### Soundness
2

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
2

### Summary
This paper presents CountsDiff, a diffusion framework designed to model distributions on the natural numbers. It builds upon the Blackout Diffusion framework, simplifying its formulation through a direct parameterization in terms of a survival probability schedule and an explicit loss weighting.

### Strengths
- Introduces a diffusion framework designed to handle discrete ordinal data using birth/death processes.

- The survival-probability-based parameterization provides a potentially simpler formulation than prior discrete diffusion frameworks.

- The method could be useful for applications such as modeling scRNA-seq data, where count-valued distributions naturally arise.

### Weaknesses
- The motivation for applying the method to natural image datasets is unclear and misaligned with the model’s intended domain.

- The novelty relative to existing discrete diffusion models (e.g., Santos et al., 2023) appears limited.

- The presentation and related work discussion are difficult to follow and lack clear differentiation from prior methods.

### Questions
I do not understand why CountsDiff is applied to natural image datasets (CIFAR-10, CelebA). Existing Gaussian diffusion models already perform very well, and there is no comparison with these established methods on image datasets. The DDPM paper by Ho et al. (2020) achieved an FID score of 3.17, which is much better than the results reported in Table 1. 

The scRNA-seq data seem to make much more sense for this framework, and the authors also mainly emphasize related work on scRNA-seq in Section 6. Therefore, I think the paper should be significantly revised to make this emphasis much clearer, so that the audience can better understand the main contribution. Moreover, the related work section is unclear—for example, the authors mention that methods such as Forest-Diffusion can also be adapted to the scRNA-seq task, but it is not explained what the limitations of that method are or what advantages the proposed approach offers. In addition, Forest-Diffusion is not included as a baseline in the numerical results.

The presentation of the proposed method and its novelty compared to existing work such as Santos et al. (2023) are also unclear. It seems that the main difference from Santos et al. (2023) is that this work considers a general schedule p(t), whereas Santos et al. (2023) focuses on the special case of a fixed p(t).

### Soundness
2

### Presentation
2

### Contribution
2
