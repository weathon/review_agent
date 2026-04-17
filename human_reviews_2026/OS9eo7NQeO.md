# WARPD – World-model Assisted Reactive Policy Diffusion

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
With the increasing availability of open-source robotic data, imitation learning has become a promising approach for both manipulation and locomotion. Diffusion models are now widely used to train large, generalized policies that predict controls or trajectories, leveraging their ability to model multimodal action distributions. However, this generality comes at the cost of larger model sizes and slower inference, an acute limitation for robotic tasks requiring high control frequencies. Moreover, Diffusion Policy (DP), a popular trajectory-generation approach, suffers from a trade-off between performance and action horizon: fewer diffusion queries lead to larger trajectory chunks, which in turn accumulate tracking errors. To overcome these challenges, we introduce WARPD (World model Assisted Reactive Policy Diffusion), a method that generates closed-loop policies (weights for neural policies) directly, instead of open-loop trajectories. By learning behavioral
distributions in parameter space rather than trajectory space, WARPD offers two major advantages: (1) extended action horizons with robustness to perturbations, while maintaining high task performance, and (2) significantly reduced inference costs. Empirically, WARPD outperforms DP in long-horizon and perturbed environments, and achieves multitask performance on par with DP while requiring only ∼ 1/45th of the inference-time FLOPs per step.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduce world-model assisted reactive policy diffusion, which learns a closed-loop controller rather than a trajectory predictor. Apart from learning a policy parameter encoder and decoder with VAE, this paper’s major contribution is introduce a co-trained world model to provide auxiliary loss to ensure policy consistency and robustness.

### Strengths
- the proposed warpd demonstrate significant speed up for diffusion model, highlight the efficiency of reactive policy.
    
- the proposed method also preserve the diversity of expert demonstrations, indicating the importance of VAE encoder.
    
- the modeified ELBO is nealty and the decomposed loss function is well motivated.

### Weaknesses
- since world model is merely trained on data collected with expert, it might be unreliable for OOD states, which is precisely where stability is needed. This paper doesn’t clearly show how the world model improve robustness under such conditions. Quantitive analysis of modelling error v.s. policy performance would help.
    
- WARDP encoder trajectories into latent rather than policy weights used in standard hypernetwork. Clarifying what the latent represents and comparing to baselines like [1] would make contributino cleaner.
    

[1] Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation. Xue et al.

### Questions
- can the authors evaluate WARPD on more challenging tasks (e.g., Robomimic Tool Hang, Transport) or with varied expert quality and input modalities? The current tasks (Can, Lift) are relatively easy and don’t fully test robustness since both of them are relative easy to solve even with regression.
    
- PushT and Robomimic rely on high-frequency position control. Can the authors show that WARPD still outperforms diffusion policies when this stabilizing layer is removed or control frequency is reduced? (i.e. directly do torque control, where the station becomes unstable)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on imitation learning for robotics—specifically in manipulation and locomotion—using diffusion models. It notes that while diffusion policies effectively capture multimodal action distributions and support generalized policy behavior, they suffer from large model sizes and slow inference speeds, which are critical limitations in tasks requiring high control frequencies. To address these issues, the authors propose WARPD (World Model Assisted Reactive Policy Diffusion), a method that integrates a learned world model to assist a diffusion-based policy, enabling fast and reactive control while retaining the advantages of diffusion-based action generation. The work seeks to bridge diffusion-based imitation learning with model-based prediction to achieve more responsive policies. It clearly identifies inference latency and model size as key bottlenecks of diffusion policies in high-frequency robotic control settings and underscores the limitations of trajectory-generation Diffusion Policy approaches in such scenarios.

### Strengths
The paper demonstrates significant originality by reframing diffusion-based imitation learning to generate closed-loop policies directly in parameter space, a conceptual shift that effectively preserves multimodality while overcoming the latency and trajectory-tracking pitfalls of prior methods. It astutely diagnoses the underexplored but critical trade-off between action horizon and tracking accuracy in trajectory diffusion under high-frequency constraints. The integration of a learned world model with latent diffusion to enable reactive control is a creative synthesis that cohesively addresses inference speed and stability.

### Weaknesses
1. **Novelty and positioning need sharper evidence**
The core idea—use latent diffusion plus a world model to generate closed‑loop policies directly in parameter space, bypassing trajectory generation—is interesting but overlaps conceptually with prior lines on context/hypernetwork-conditioned policies and model-assisted imitation learning. The paper should more clearly articulate what is fundamentally new versus: (a) Diffusion Policy variants that reduce query counts via action chunking or one step diffusion methods; (b) hypernetwork/meta-learning approaches that generate policy parameters from context; (c) model-based IL/RL with world-model regularization. Concretely: add head-to-head comparisons and an explicit “differences vs. closest work” table. Include works on policy parameter generation via hypernetworks/meta-learning and diffusion-based control to delineate WARPD’s distinct contribution. When claiming to “bypass trajectory generation,” clarify how this differs in practice from generating shorter horizons plus feedback. Provide a targeted ablation replacing diffusion with simpler generators (e.g., VAE or normalizing flows) for the same parameter-space policy generation to show diffusion is necessary for the claimed benefits, not just any latent generator.
2. **Real-world deployment readiness not yet demonstrated**
If all results are in simulation, add at least one real-robot evaluation or a sim-to-real transfer test, reporting latency, success, and safety incidents. Even a short, controlled study would substantiate the practical relevance of the claimed speed and reactivity.

### Questions
The main issues and suggestions have been outlined in the Weakness section. Additional points include:

The experimental section should be enhanced by adding comparisons with other state-of-the-art Diffusion Policy and Model-based RL methods.

Please use vector-based figures for all illustrations to ensure optimal clarity and scalability.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes WARPD, a diffusion-based imitation learning framework that generates closed-loop policies (network weights) rather than open-loop action trajectories. Instead of predicting a sequence of actions like Diffusion Policy (DP), WARPD samples policy parameters via latent diffusion and decodes them using a hypernetwork conditioned on states or task identifiers. A learned world model is used during training to align policies with realistic system dynamics and correct drift from the dataset distribution.

WARPD aims to address three main limitations in trajectory diffusion approaches:

(1) Slow inference from repeatedly denoising long action sequences,

(2) Tracking errors from large action horizons, and

(3) Poor robustness under perturbations due to open-loop control.
The authors show experiments on PushT, Robomimic (Lift & Can), and 10 MetaWorld tasks, reporting better long-horizon robustness, ~45× lower inference FLOPs, and competitive success rates.

### Strengths
Novel shift from action-space to parameter-space diffusion
Instead of diffusing trajectories, WARPD diffuses policy weights via a hypernetwork, which naturally enables closed-loop control and removes the need to regenerate trajectories at every timestep.

Addresses the core weakness of diffusion policies (latency & drift)
By producing a policy that runs reactively at high frequency, WARPD avoids action-horizon drift and reduces the number of diffusion queries required per episode.

Integration of world model is meaningful (not just architectural trick)
The world model guides the policy to stay in-distribution and helps correct state deviations during training. This makes the approach more grounded than purely behavior cloning + diffusion. 

Strong empirical evaluation (robustness + efficiency)

Handles long-horizon perturbed control better than Diffusion Policy (Fig. 3 & Fig. 5)

On MetaWorld multitask settings, achieves ~45× lower inference FLOPs at comparable success rates (Fig. 6)

Demonstrates both state-based and vision-based versions, which increases credibility of generalization.

Clear theoretical formulation (modified ELBO + hypernetwork objective)
The paper derives a structured learning objective combining behavior cloning, reconstruction via world model, and latent diffusion. This helps show it’s not just a heuristic but grounded in generative modeling. 

Behavior diversity captured in latent space
The t-SNE result showing skills clustered by human demonstrators (Fig. 7) suggests WARPD can learn structured behavioral variability without explicit supervision.

### Weaknesses
Still relies on low-dimensional state observations in main experiments
Vision-based results only appear in a small section (Table 1), using frozen encoders rather than full end-to-end visual policy generation.

World model quality is critical but not deeply analyzed
If the learned dynamics are inaccurate, does WARPD degrade? The paper does not provide failure cases or show robustness to model inaccuracies.

Training complexity and compute overhead are downplayed
Although inference is cheaper, training requires three components (VAE + world model + diffusion). No wall-clock training cost or GPU hours are reported.

Policy generation via hypernetworks may limit expressiveness
Only MLP policies are considered. It's unclear whether this method scales to transformers or visuomotor architectures like RT-1, Octo, or Diffusion Policy with image inputs.

Comparison baselines could be broader
Missing comparisons with model-based policy generation approaches like DreamerV3, HyperPPO, UniZero, or transformer-based policy distillation.

Limited real-robot or real-world experiments
All experiments are simulated environments (PushT, MetaWorld, Robomimic). No deployment on physical robots, making practicality uncertain.

### Questions
see weaknesses

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
2

### Summary
WARPD proposes a diffusion-based world-model-assisted controller that diffuses latent variables used to generate policy weights via a hypernetwork, rather than directly diffusing actions. The approach is claimed to reduce inference compute while maintaining or improving performance across multi-task manipulation and perturbation settings. A modified ELBO objective combines imitation, world-model rollout, and latent regularization terms.

### Strengths
- The paper tackles an underexplored bottleneck in diffusion-based control and provides a plausible path toward efficiency via policy-space generation.

- WARPD combines ideas from world models, latent diffusion, and hypernetwork-based policy generation, forming a coherent if complex system that extends recent model-based imitation learning work.

- The experimental coverage is broad, with consistent reporting of FLOPs, ablations, and hyperparameters.

- The reported performance gains in long-horizon and perturbed settings are practically meaningful.

- The work addresses a timely and relevant problem in large-scale robot policy learning.

### Weaknesses
- The system chains together a world model, VAE encoder, diffusion model, and hypernetwork, with partial joint optimization. This makes it difficult to isolate where improvements originate. The theoretical story suggests unified probabilistic modeling, but the actual implementation is a series of independent training stages.

- The paper’s modified ELBO derivation is central to its conceptual framing. However, the KL term is effectively disabled (β ~ 1e-10), meaning the latent space is unregularized. The diffusion model thus learns to imitate arbitrary encoder codes, not a well-defined prior. Sampling from the prior (as the ELBO suggests) would not produce coherent behaviors. Therefore, the method is not truly a probabilistic generative model, and the ELBO justification is more rhetorical than operative. The paper’s central theoretical claims (generative nature, latent consistency, principled regularization) would collapse under this setting.

- I’m not convinced the central architectural choice of diffusing policy weights via a hypernetwork instead of diffusing actions is necessary or clearly beneficial. The paper provides no ablation or evidence that this design improves over simpler schemes (e.g., latent trajectory diffusion). In fact, several implementation choices (KL≈0, world-model rollout, multi-stage training) seem to exist primarily to make this complicated setup trainable. As a result, the method feels overengineered to support this representation.

### Questions
1) Given β≈0, are you truly sampling from a generative prior or simply predicting deterministic latents? How does this affect the validity of the modified ELBO derivation?
2) Is the hypernetwork really necessary? Have you compared against diffusing actions or latent trajectories directly, and can you clarify what specific benefit weight diffusion brings?

### Soundness
2

### Presentation
3

### Contribution
2
