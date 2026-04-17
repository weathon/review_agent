# Unified Latent Steering and Residual Refinement for Online Improvement of Diffusion Policy Models

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Imitation learning has driven major advances in robotic manipulation by exploiting large and diverse demonstrations, yet policies trained purely by imitation remain brittle under distribution shift and novel scenarios, making online improvement essential. 
Directly finetuning the parameters of modern large policies is prohibitively sample inefficient and computationally expensive,
while recent finetuning-free adaptation methods either fail to exploit the multimodal distributions learned by pretrained policies or remain confined to the coverage of demonstrations.
We propose USR, a Unified framework for latent Steering and residual Refinement that enables efficient online improvement of diffusion policy models. A lightweight actor jointly outputs latent noise to steer the diffusion process toward promising modes and residual corrections to adapt beyond the diffusion policy's support, combining stable mode selection with flexible refinement. This unified design stabilizes training and fully leverages both components. Experiments on standard benchmarks and our MultiModalBench demonstrate USR's state-of-the-art performance. Furthermore, we validate its real-world applicability by improving a Vision-Language-Action (VLA) model on a physical robot, setting a new paradigm for sample-efficient adaptation of diffusion-based policies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents USR, a lightweight framework for online improvement of pretrained diffusion robot manipulation policies. It combines latent noise steering to exploit multimodal base policy behaviors and residual corrections to enable adaptation beyond base policy support. Experiments on MultiModalBench, a new benchmark for evaluating multimodal manipulation, AdroitHand and ManiSkill show USR outperforms strong baselines in sample efficiency and task success.

### Strengths
- This paper is overall clearly written.
- The experiments cover a range of test settings, and the ablation studies help understand each component of the method.
- Overall the experiment results are good, which demonstrates the effectiveness of the proposed method.

### Weaknesses
**Weakness 1: Real-world validation is absent.**

There are no real-robot experiments to validate the effectiveness of the proposed method in policy learning in the real world. Incorporating such experiments would provide a more comprehensive understanding of USR's performance in real-world policy learning.


**Weakness 2: The analysis of USR’s performance in long-horizon tasks is insufficient.**

The maximum episode steps across all experiments are only 650 (for SortYCBStrict), and there is no validation of the method’s adaptability to longer-cycle manipulation tasks (e.g., multi-step assembly, continuous object transportation) where cumulative errors or mode drift might occur.

**Weakness 3: Limited discussion of constraints and theoretical failure cases:**

The paper notes the actor-critic mechanism but does not delve into what happens if the critic overfits to spurious correlations early in adaptation, or how USR mitigates bounce-back or drifting due to over-reliance on the residual path.

### Questions
**Question 1:**

Can the authors clarify how the dimensions and sampling distributions for $ w_t $ are chosen and normalized relative to the base diffusion model? Is there any adaptive mechanism (e.g., temperature, scale) for action/noise selection beyond a static bound?

**Question 2:**

How does USR perform in settings in which demonstration coverage is incomplete or severely imbalanced, e.g., where certain modes are missing or underrepresented? Are there experiments quantifying such regime limitations?

### Soundness
3

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
This paper proposes an unified framework for online fine-tuning diffusion policies with a combination of sample steering and residual refinement. The idea is simple and straightforward with reasonably good performance on three simulation benchmarks. Main limitations including lack of ablation study on design choices (i.e., sample steering alone, residual refinement alone, etc.) and lack of real-world experiments.

### Strengths
1. The idea is simple yet efficient, quite easy to understand and follow
2. The figures (esp. Fig. 1 & 2) are really helpful
3. Abundant simulation experiments are conducted with various baseline methods

### Weaknesses
1. No real-world experiments, I'm curious about the impact of learned residual actions on sim-to-real transfer
2. It appears to me that the proposed method is a simple combination of two basic ideas, how does each component contribute to the final performance improvements? I didn't see ablation studies on each design choices
3. The policy’s performance shows sensitivity to hyperparameter settings, indicating that task-specific tuning is required and potentially limiting the method’s generalizability

### Questions
1. Can authors elaborate more on the conclusion of "the noise action steers sampled trajectories toward the most promising mode, while the residual action enables further refinement beyond the support of the base policy" from figure 7?
2. Why DPPO is omitted on ManiSkill?
3. How does equation (7) differ from policy distillation or value decomposition in practice?

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
This work, USR, augments a frozen diffusion policy with a lightweight RL actor that performs two tasks simultaneously:
1. Latent Steering: The actor outputs latent noise vectors that are fed as the diffusion model’s initial noise. This biases sampling toward promising modes in the base policy’s multimodal distribution, improving behavior selection.
2. Residual Refinement: The actor outputs a residual correction applied to the diffusion output. A scale factor $\alpha$ controls the strength of refinement, enabling fine-grained adaptation beyond the base policy’s support.

The work also releases a dataset, MultiModalBench. Results are shown in sim, on their own dataset as well as Adroit and some tasks from ManiSkill.

### Strengths
Paper is well motivated, with a simple method proposed to solve the task.

### Weaknesses
1. Lack of Real-World Validation. All experiements are in sim.
1. Unclear Critic Training Distribution: The description of the combined critic (which distills from the environment critic) omits details about the sampling distribution used for random latent-residual actions pairs.
1. Potential Policy-Critic Mismatch: The progressive gating of the application of the residual (via hyperparam $\epsilon$) during rollouts in a form of curricular learning, but it introduces a distributional gap between training and inference for the critic. The paper doesn’t analyze or justify this mismatch.
1. Less Human-generated datasets / Evaluation Design:
    * MultiModalBench rewards selecting the correct demonstration mode, a setup that inherently favors a method centered on mode steering. This could inflate the comparative advantage of USR relative to general-purpose baselines.
    * MultiModalBench, the main benchmark introduced, uses multimodal trajectories that are synthetically generated, primarily via motion planning. By virtue of using classical TAMP techniques, each “mode” is a low-variance Gaussian cluster (low Kolmogorov complexity), not a truly multimodal human-behavior distribution. Under such conditions, latent steering can easily select the correct cluster, and residual refinement only needs to make small local adjustments.
    * The paper’s only human-generated dataset results, Adroit, shows no clear advantage for USR over DSRL, the very baseline it aims to supersede. This pattern (visible in Figure 5) suggests that USR’s benefit may not generalize beyond synthetic multimodality.
    * To address this, the authors should evaluate on RoboMimic (as DSRL did). This helps test robustness on naturally diverse human demonstrations where intra-mode variance and cross-mode overlap are much higher.
1. Residual actor expressivity question: For USR to succeed on truly multimodal human data, the residual actor must itself represent multiple behavior modes. However, the paper omits implementation details. If it is a simple MLP producing a unimodal residual distribution, it cannot express multimodal refinements, which would limit generalization in richer human-collected settings.

### Questions
1. Considering the use of additional (2 critics and a unified actor) models, What are the training times and computational needs of the method?
1. What are the implementation details of the lightweight actor and the 2 critics? Some details, such as architecture, are missing in appendix D.
1. What is the implication of having a UTD ratio <1 ? With UTD 0.25, does that mean you discard $\frac{3}{4}$ of the training data? Please explain this further.

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
The paper proposes USR, a method that uses RL to steer and improve a pre-trained frozen diffusion policy. USR provides a unified framework that combines concepts from latent-noise steering and residual-RL approaches. Specifically, it trains an RL actor to output the concatenation of an initial noise term and a residual action. The initial noise can guide the policy toward promising modes, and the residual action refines the output to extend performance beyond the support of the pre-trained policy. Experiments on simulation manipulation benchmarks, including a newly proposed MultiModalBench, show that USR can outperform both DSRL and Residual RL.

### Strengths
The paper is clearly written overall, and the idea of unifying initial-noise steering with residual RL is simple yet well motivated. The practical algorithm is well-designed and combines dual critics with distillation. The simulated evaluation seems promising, with the USR outperforming all the baselines.

### Weaknesses
The main concern is the selection of the task benchmarks. Most evaluations are performed on the newly introduced MultiModelBench, making it difficult to calibrate task difficulty and judge how prior methods perform. Notably, DSRL does better on Adroit and ManiSkill (Figure 5) but struggles on MultiModelBench, raising the concern that the benchmark may have been constructed in a way that disadvantages baselines while favoring the proposed approach. I recommend adding additional benchmarks such as RoboMimic, which is widely used in prior work (e.g., DPPO and DSRL). Please see the next section for further questions.

### Questions
- I’m a bit confused about the design of MultiModalBench. The authors mention that the benchmark is designed to “evaluate the ability of adjusting multimodal policies toward the desired single-mode behavior.” Then why does DSRL struggle to improve? Isn’t this a mode-selection problem, since the dataset already contains all modes?

- The results note that DSRL is "prone to collapse during training". Given that DSRL is conceptually similar and USR shares similar hyperparameters, why is DSRL unstable while USR is stable?

- While both USR and DSRL use the dual-critic version, have the authors tried naively training a single critic in the latent space (similar to the DSRL-SAC variant mentioned in their paper)?

- Related to the question above, the motivation for noise-aliased DSRL is that it can leverage offline demonstrations to further improve sample efficiency. Have the authors tried combining offline demonstrations with USR to see if it provides additional benefits? Including this ablation would make the analysis more complete and interesting.

- The implementation details of the unified actor and critic are missing. What is the architecture?

- Why is a discount factor (gamma) of 0.97 used across the experiments? Have the authors performed a sweep over this value? It seems quite low compared to the values typically used in prior work (0.99 or 0.999).

I’d be happy to reconsider my score once these points are addressed.

### Soundness
2

### Presentation
3

### Contribution
2
