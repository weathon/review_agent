# Universal Inverse Distillation for Matching Models with Real-Data Supervision (No GANs)

- Decision: Accept (Oral)
- Scores: 8, 4, 6, 6

## Abstract
While achieving exceptional generative quality, modern diffusion, flow, and other matching models suffer from slow inference, as they require many steps of iterative generation. Recent distillation methods address this by training efficient one-step generators under the guidance of a pre-trained teacher model. However, these methods are often constrained to only one specific framework, e.g., only to diffusion or only to flow models. Furthermore, these methods are naturally data-free, and to benefit from the usage of real data, it is required to use an additional complex adversarial training with an extra discriminator model. In this paper, we present **RealUID**, a universal distillation framework for all matching models that seamlessly incorporates real data into the distillation procedure without GANs. Our **RealUID** approach offers a simple theoretical foundation that covers previous distillation methods for Flow Matching and Diffusion models, and is also extended to their modifications, such as Bridge Matching and Stochastic Interpolants. The code can be found in https://github.com/David-cripto/RealUID.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This is a paper about generative modeling, in which the main contribution is the incorporation of real data into the loss function or the training process. The goal is to overcome the slow sampling process in diffusion models and flow matching models, which are collectively called match models in the paper. The proposed method incorporates real data into the distillation loss, thus, it removes the need to use GAN for assessing the adversarial loss. Experiments were performed on CIFAR-10 and showed that the proposed method did have faster inference.

### Strengths
Stengths include the incorporation of real data into the distillation loss, thus avoiding the need for extra adversarial loss, which in turn, requires a discriminator network as in GAN. This is a logic and novel idea of implementing the distillation loss, as compared with existing approaches. Another strength is the reduced inference time, as shown in Table 3, second row from the bottom.

### Weaknesses
It seems Equation 17 is the key formula of the work as it defines the new distillation loss, which weighs the loss or contribution from real data by hyperparameters alpha and beta. It is unclear why alpha and beta are, first, confined to a small range such as between 0.94 and 1.0, and, second, why alpha and beta are so close to each other in most cases. What does that mean for understanding the distillation loss? Does it mean that the contribution from the real data is typically given a small weight since it is multiplied by (1-alpha)? 
In the ablation study, Table 2, second row, when alpha = 1, beta=1 in this case, shouldn't the RealUID's performance equal that of the reference (the first row, Teacher Flow) since according to equation 17, when alpha=1, the second term is 0?

### Questions
Please see weaknesses.
It is unclear why on line 294, the paper requires that "A key constraint is that the loss must still yield the same teacher upon minimization on the real data." With the inclusion of real data, shouldn't the teacher model be somehow different from the teacher model without using the real data? 
The goal of equation 13 is to maximize it, which is the last term of equation 10. Does maximizing equation 13 guarantee that equation 10, as a whole, will be minimized as equation 10 has two other terms that depend on f_t^star and f_t^theta?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the RealUID framework, with two primary contributions: (1) it theoretically unifies prior distillation methods for different matching models (e.g., SiD, FGM) and extends them to Bridge Matching and Stochastic Interpolants via a novel UID loss based on a linearization technique; (2) it proposes a novel method for incorporating real data into the distillation process without relying on GANs. Experiments on CIFAR-10 provide empirical support for these theoretical claims.

### Strengths
1.The authors constructed a concise unified form through a linearization technique, and the theoretical derivation is fundamentally sound; 
2.The extension of the framework to bridge matching and stochastic interpolants is commendable; 
3.Replacing complex GANs with simple loss interpolation to incorporate real data is a straightforward yet effective approach.

### Weaknesses
1.The proposed unified framework appears more as a formal re-formulation than a substantive theoretical breakthrough. This raises concerns that the core contribution may essentially be a "re-parameterization" of prior methods like SID and FGM, thereby undermining its claimed novelty and independence.
2.The proposed RealUID method suffers from a critical dependence on two newly introduced hyperparameters, α and β. Firstly, the selection of these parameters lacks clear theoretical guidance, and their design appears more as a heuristic afterthought rather than a principled derivation. Secondly, as evidenced in Table 1 and Appendix Table 6, the method exhibits extreme sensitivity and pronounced randomness to these parameters.
3.Lemma 2 relies on the assumption that the data probability density is positive everywhere (pt∗(xt) > 0, ∀xt, t). This assumption typically fails in high-dimensional spaces.
4.The most prominent weakness of this work is its narrow experimental setup:
1)The evaluation is narrowly based on small-scale datasets like CIFAR-10. The absence of results on standard large-scale, high-resolution benchmarks such as AFHQ and ImageNet severely limits the empirical grounding of the paper's claims.
2)The experimental evaluation is confined to  flow-based models, omitting tests on diffusion models. This limitation raises serious concerns about the method's competitive advantage and the framework's generalizability.
3)The experiments lack comparisons against state-of-the-art baselines from 2025.

### Questions
1.Beyond mathematical re-formulation, what are the new insights and practical improvements uniquely enabled by your unified UID loss perspective that were not apparent or easily achievable within the prior frameworks.
2.Is there any adaptive criterion for tuning α and β? What is the rationale behind using different parameter values during the training versus fine-tuning stages? Do these hyperparameters generalize across datasets, or do they require costly re-tuning for each new dataset?
3.Why does RealUID use a lightweight model architecture different from the baseline methods? Could increasing the parameter count further improve performance?
4.Could the authors provide additional results of RealUID on other datasets and models? This is my primary concern.

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
5

### Summary
The paper tackles the challenge of distilling diffusion/flow-based models into fast one-step generators. The authors build on works such as SiD and FGM, which optimize the student by minimizing the $\ell_2$ distance between the teacher and student score/velocity functions, referred to as $f$-functions in the paper. The authors propose a simple but clever modification to the training objective of the student $f$-function that includes both real and generated data, allowing for an alternative way to incorporate real data into the distillation pipeline of these methods. They theoretically justify this change by proving that it has a proper global minimizer and experimentally demonstrate its usefulness on CIFAR-10, achieving impressive results.

### Strengths
- The premise of the paper is very interesting and important. Distillation methods that rely on Reverse-KL-like objectives (SiD, FGM, DMD, …) are widely used in practice but are known to suffer from a diversity collapse problem. This issue arises mainly because the loss is computed only on student samples (reverse-KL). Therefore, being able to effectively incorporate real data into training without resorting to GANs (which have famously unstable training dynamics) is potentially very valuable.
- The reasoning that leads to the RealUM and RealUID objectives is clever.
- The supplementary material is extensive and includes additional variants of the approach, as well as theorems and proofs.

### Weaknesses
- As mentioned above, loss of diversity is a common drawback of distillation approaches with Reverse-KL-like objectives. Incorporating real data into the training pipeline can, in principle, fix this issue to some extent. Therefore, including diversity metrics in the experiments would be very helpful and could strengthen the paper’s argument.
- The paper broadly argues that recent diffusion distillation methods either rely solely on synthetic data or use real data only in GAN-like objectives. While this is true for most Reverse-KL-like methods, it does not hold for consistency- or flowmap-based methods[1,2,3,4,5]. Including a discussion of such methods and adding them as additional baselines would help the paper better position itself within the literature.
- Although the authors explicitly note that the limited scope of the experiments is due to computational constraints, the fact remains that the proposed approach is primarily evaluated on a small dataset using a lightweight network architecture. As a result, it is unclear whether the conclusions drawn will generalize to larger datasets and models. For example, as shown in Table 4, the proposed approach performs considerably worse than prior SiD and SiDA baselines.
- The method seems highly sensitive to the values of $\alpha,\beta$, as shown in Table 1.


[1] Song, Yang, et al. "Consistency models." (2023). \
[2] Lu, Cheng, and Yang Song. "Simplifying, stabilizing and scaling continuous-time consistency models." arXiv preprint arXiv:2410.11081 (2024). \
[3] Frans, Kevin, et al. "One step diffusion via shortcut models." arXiv preprint arXiv:2410.12557 (2024). \
[4] Geng, Zhengyang, et al. "Mean flows for one-step generative modeling." arXiv preprint arXiv:2505.13447 (2025). \
[5] Sabour, Amirmojtaba, Sanja Fidler, and Karsten Kreis. "Align Your Flow: Scaling Continuous-Time Flow Map Distillation." arXiv preprint arXiv:2506.14603 (2025).

### Questions
- Why does the setting too low values for the coefficients $\alpha,\beta$ lead to collapse? Is this just experimentally seen or is there a theoretical justification for it?

### Soundness
4

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
4

### Summary
This paper addresses a core problem in generative model distillation: existing data-free methods (e.g., SiD, FGM) cannot leverage real data to improve the student model, and the only current way to do so relies on complex and unstable GANs.The paper's contributions are twofold:
1. **Theoretical Unification (UID)**: It introduces a general min-max distillation framework (UID) and demonstrates that recent methods like SiD and FGM are special cases of this unified structure. 
2. **Methodological Innovation (RealUID)**: This is the paper's main contribution. The authors propose RealUID, a novel, non-adversarial (GAN-free) mechanism to integrate real-data supervision. This is achieved by cleverly redesigning the inner loss function of the min-max game ($\mathcal{L}_{R-UM}^{\alpha,\beta}$) to include terms from both the generated distribution ($p_0^\theta$) and the real-data distribution ($p_0^*$).

Experiments (on CIFAR-10) show that RealUID not only outperforms the data-free baseline in FID but also achieves comparable or superior results to GAN-based methods, all while significantly accelerating convergence and avoiding the complexities of adversarial training.

### Strengths
1. **Significant Methodological Innovation (GAN-free Supervision)**: The paper's strongest contribution is RealUID, a novel and elegant non-adversarial method to incorporate real-data supervision into the distillation process. This directly addresses a major pain point of previous work, avoiding the need for extra discriminator architectures and the instability of adversarial training.
2. **Clever Loss Function Design ($\mathcal{L}_{R-UM}^{\alpha,\beta}$)**: The mechanism of innovation is particularly commendable. Instead of simply adding a new top-level loss, the authors redesigned the inner-loop loss of the min-max game (Eq. 17). The "fake model" $f$ is forced to optimize on both $p_0^\theta$ (generated data) and $p_0^*$ (real data) during its $\max_f$ step. This allows $f$ to serve as a "bridge," passing the supervisory signal from the real data to the student generator $G_\theta$ in a natural, non-adversarial way.
3. **Strong Theoretical Unification (UID)**: Prior to RealUID, the paper's UID framework is a valuable theoretical contribution in its own right. It unifies seemingly separate works (SiD, FGM) under a single min-max perspective using a clear and general mathematical derivation (the linearization technique). It clarifies the underlying game structure: the "fake model" $f$ (the $\max_f$ player) tries to approximate the "theoretical student function" $f^\theta$, while the generator $G_\theta$ (the $\min_\theta$ player) tries to minimize the distance between $f^\theta$ and the teacher $f^*$.
4. **Excellent Empirical Efficiency**: The experimental results strongly support the practical utility of this new method. RealUID not only achieves strong FID scores but also demonstrates significantly faster convergence (approx. 3x) compared to the data-free baseline (Fig. 2), proving that this GAN-free supervision is both effective and efficient。

### Weaknesses
1. **Severely Limited Experimental Scope (Main Flaw)**: All experiments are conducted exclusively on the CIFAR-10 (32x32) dataset. While resource constraints are mentioned, this leaves the scalability of RealUID to higher resolutions (e.g., 256x256 or 512x512) completely unknown. The need for distillation is most acute for these larger, slower models.
2. **Not State-of-the-Art (SOTA) Performance**： While the results are strong, the best-reported FID (e.g., 1.91 for conditional generation) still lags behind SOTA adversarial distillation methods (e.g., $SiD^2A$ @ 1.39). The authors hypothesize this is due to their lightweight architecture and teacher model, but this is unproven. It is unclear if the performance gap is due to the architecture or an inherent ceiling of the RealUID method itself compared to adversarial approaches.
3. **New Hyperparameter Sensitivity**: RealUID avoids GAN instability but introduces new hyperparameters $\alpha$ and $\beta$. The fine-tuning ablation (Table 6) shows high sensitivity to these, with many configurations failing to converge (marked with "-"). This may offset the claimed simplicity, as tuning $\alpha$ and $\beta$ could be as difficult as tuning a GAN.

### Questions
1. **On Scalability (W1)**: Can the authors provide even preliminary results on a single higher-resolution dataset (e.g., FFHQ-64 or CelebA-256)? How does RealUID converge, and are the $\alpha, \beta$ hyperparameters stable across different resolutions?
2. **On $K=5$ (W3)**: The experimental details state the fake model $f$ is updated 5 times for every 1 generator update ($K=5$). How critical is this ratio? Does the RealUID framework become unstable at $K=1$? This seems to hide a 5x computational cost for the fake model's training loop relative to the generator's.
3. **On $\mathcal{L}_{R-UM}^{\alpha,\beta}$ Coefficient Design (S2)**: The specific coefficients $\frac{\beta}{\alpha}$ and $\frac{1-\beta}{1-\alpha}$ in Eq. 17 are clearly deliberate. The paper claims this is to ensure $f=f^\*$ is the optimal solution when $p_0^\theta = p_0^*$.
Is this specific mathematical form the only one that satisfies this property? Would the framework fail if simpler, unscaled $L_2$ losses were used (e.g., setting $\beta/\alpha = 1$ and $(1-\beta)/(1-\alpha) = 1$)? This seems highly relevant given the hyperparameter sensitivity seen in (W3).
4. **On "Correcting" the Teacher (S2)**: A potential benefit of RealUID is that the fake model $f$ sees both the teacher $f^\*$ and the real data $p_0^\*$. If the teacher $f^\*$ is flawed (e.g., suffers from mode collapse) but $p_0^\*$ is complete, $f$ faces a conflict during its training: should it match $f^\*$ or the function implied by $p_0^\*$?  Can RealUID truly "correct" a flawed teacher, or would this conflict lead to training instability?

### Soundness
3

### Presentation
3

### Contribution
3
