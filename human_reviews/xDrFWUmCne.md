# Learning to Discretize Denoising Diffusion ODEs

- Decision: Accept (Oral)
- Scores: 8, 8, 8

## Abstract
Diffusion Probabilistic Models (DPMs) are generative models showing competitive performance in various domains, including image synthesis and 3D point cloud generation. Sampling from pre-trained DPMs involves multiple neural function evaluations (NFEs) to transform Gaussian noise samples into images, resulting in higher computational costs compared to single-step generative models such as GANs or VAEs. Therefore, reducing the number of NFEs while preserving generation quality is crucial. To address this, we propose LD3, a lightweight framework designed to learn the optimal time discretization for sampling. LD3 can be combined with various samplers and consistently improves generation quality without having to retrain resource-intensive neural networks. We demonstrate analytically and empirically that LD3 improves sampling efficiency with much less computational overhead. We evaluate our method with extensive experiments on 7 pre-trained models, covering unconditional and conditional sampling in both pixel-space and latent-space DPMs. We achieve FIDs of 2.38 (10 NFE), and 2.27 (10 NFE) on unconditional CIFAR10 and AFHQv2 in 5-10 minutes of training. LD3 offers an efficient approach to sampling from pre-trained diffusion models. Code is available at https://github.com/vinhsuhi/LD3.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes LD3, a method to discover timesteps that will yield good samples for diffusion models when the number of forward passes is very small at inference time, through optimization. The authors propose a distillation-like objective where a global notion of distance between the original model and a student model with learned timesteps is minimized. Two variants are presented, one "soft" objective achieving best results that only requires samples to be within a ball of hyperparameter radius $r$ to the teacher's samples. The soft objective additionally is linked theoretically to also upper bounding the KL divergence between the two models. Various experiments are conducted to demonstrate that LD3 achieves better FID scores than prior methods on few-step sampling, and an ablation study is included to demonstrate the importance of different components of LD3, the most important seemingly being the decoupling of timestep choice and step size, and the use of a perceptual distance (LPIPS) as opposed to pixel-based L2.

### Strengths
- The paper includes proofs of soundness of their proposed minimization objectives, going beyond purely empirical contribution.
- The number of experiments is substantial across both datasets, baselines from prior work, and choice of pretrained models.
- The experiments conducted include notoriously difficult datasets in the literature of diffusion step reduction like ImageNet, and shows improvement in more complex settings such as a text to image model.
- The objective is cheap to train compared to prior work; the key being that a very low batch size of 2 is permissible to use.
- Ablation studies demonstrate the importance of the proposed changes separately.
- The samples presented qualitatively look very reasonable and show clear improvement over the usual hand-crafted timestep schedules, and they fix the random seed so the same samples can be comapred.

### Weaknesses
There are two main areas around which the paper could be much stronger. The first is in comparisons to distillation methods, which are among the strongest in the literature. The paper includes a comparison to progressive distillation and consistency distillation in Table 9, but it is really difficult to compare these methods apples-to-apples. There are details missing (please correct me if I missed these e.g. in the supplementary material) such as what models were compared and where are the baseline scores taken from; ideally the same model should be post-trained with the different techniques. The number of forward passes across methods also doesn't match, making it difficult to draw any conclusions. One conclusion that can be drawn, however, is that progressive distillation remains better than LD3 in FID score at NFE=8, albeit requiring much more compute to distill.

The other major weakness is the lack of careful qualitative comparisons to other step reduction methods. The vast majority of the qualitative samples are compared to hand-crafted schedules, which are the weakest baselines. This is really important, especially because prior work has shown that very low FID scores can be achieved somewhat adversarially, resulting in strange samples (e.g., consider the CIFAR10 samples in the GGDM paper), so quantitative results are insufficient to truly demonstrate that LD3 improves over all prior work. Careful side-by-side comparisons of different step reduction methods, derived from the same pre-trained model and using the same initial noise and matching NFE would be significantly more convincing.

Overall, the work is strong, and the quantitative results already put this paper as a valuable contribution to the literature that should be accepted at the conference. I am opting for a weak accept, because the comparisons to distillation methods seem improper and incomplete, and the qualitative comparisons require more care. But even if so, due to the very low cost of the proposed technique and the achieved scores, the work already has intrinsic value. I strongly encourage the authors to address the concerns outlined above as it would make the work excellent.

### Questions
Near the introduction, the paper suggests the approach should be seen as complementary to distillation methods as opposed to going head-to-head, but the goal of both said approaches and LD3 is to achieve good sample quality with the least number of steps possible. Why do the authors argue this is the case? It is not clear to me (and it is not demonstrated in the paper) that distillation methods are compatible with optimizing the timesteps used, e.g. progressive distillation is trained to match two steps of the teacher with the student, so it's not clear a priori whether changing the timesteps later will completely break a model distilled this way.

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
5

### Summary
The paper tackles the challenge of optimizing timestep schedules for sampling in diffusion and flow-based generative models. The authors propose optimizing the schedule in order to minimize the global truncation error of multi-step sampling by reducing the discrepancy between a high-NFE teacher model and a low-NFE student model. They demonstrate that a relaxed version of this optimization problem yields even better results. Impressively, their proposed algorithm runs very quickly, achieving convergence in under 1 hour on a single GPU. A comprehensive evaluation is done using multiple pretrained diffusion models and ODE solvers. The proposed algorithm is compared against several hand-designed and learnt sampling schedules, and is shown to considerably improve the image quality in the low-NFE regime.

### Strengths
- The LD3 algorithm is extremely lightweight, requiring only 100 samples and less than 1 hour on a single GPU to learn optimized sampling schedules.
- The method is evaluated on a comprehensive set of pretrained models and compared against several baseline, showing improved quality in the majority of cases
- A proper ablation study is done on the various choices/hyperparameters.

### Weaknesses
- There are several typos in the paper. See some examples below:
    - Algorithm 1 Line 6: $x'_T ← x'_T + ...$  must be $x'_T ← x_T + ...$
    - Line 251: $x'T \rightarrow x'_T$
    - Line 251: $\Psi*(x_T) \rightarrow \Psi_*(x_T)$
- Theorem 1 requires more explanation on its invertibility assumption. Specifically, if the NFE is small, functions $\Psi_*, \Psi_\xi$ invertibility is a non-trivial fact which requires some justification on its assumption. 
- The method relies on a learned perceptual distance (LPIPS) to achieve optimal results, as shown by the significant quality drop in Table 7 when switching to a standard Euclidean loss. This raises questions about how well the method might generalize to other data types beyond images.

### Questions
- Can the invertibility assumption in Theorem 1 be relaxed and still achieve an upper bound on the KL divergence?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces a sampling method for diffusion models in order to reduce the sampling time required to generate an image. The paper proposes to learn the sampling steps from a teacher model that accurately solves the ODE by taking small step sizes. Extensive experiments show the effectiveness of the method while only requiring small amounts of training time.

### Strengths
- The paper is well-written and easy to follow.  
- It presents an easy solution to the sampling problem of diffusion models that only requires limited training time while obtaining. 
- The soft teacher loss is effective and simple to implement. 
- The evaluation is thorough and includes multiple models, multiple datasets, and multiple sampling strategies.

In general, I liked the paper and I lean toward acceptance. However, since this is not my area of expertise, I would wait for the discussion with the authors and other reviewers to increase the score to Accept and recommend borderline Accept for now.

### Weaknesses
Although I liked the paper, there are some concerns that, if addressed, would improve the paper. In the following paragraphs, I describe my concerns in detail:

- In the table with the main results, sometimes it is not clear what the metrics are computed against. I suppose the metrics in table 2, 3, 4, and 5 are computed against random samples of the model using the accurate estimation of the ODE. However, if this metric is computed against the true distribution, the performance of the teacher with the accurate computation of the ODE should be shown (1000 steps). I think the evaluation protocol needs to be more clearly defined.

- In a similar direction, Table 6 shows the performance of a teacher model using 8 steps. Why only 8 steps are used here? Would not the teacher use a higher number of samples?

- The model used is quite simple being only composed of a single vector (or two in the decoupled version). From the results in Table 7, increasing the number of parameters leads to better results. Would increasing the complexity of the model lead to better results?

- In the limitations section I found missing that the proposed method needs to be retrained for the target number of sampling steps. One model trained to generate images with 2 samples, would not be useful for 3 and a new model would need to be trained. This might be a problem since different images might necessitate a different number of steps to achieve good quality.

### Questions
I would like to hear the opinion of the authors on the concerns I raised in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
