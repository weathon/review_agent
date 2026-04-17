# BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Building Behavioral Foundation Models (BFMs) for humanoid robots has the potential to unify diverse control tasks under a single, promptable generalist policy. However, existing approaches are either exclusively deployed on simulated humanoid characters, or specialized to specific tasks such as tracking. We propose BFM-Zero, a framework that learns an effective shared latent representation that embeds motions, goals, and rewards into a common space, enabling a single policy to be prompted for multiple downstream tasks without retraining. This well-structured latent space in BFM-Zero enables versatile and robust whole-body skills on a Unitree G1 humanoid in the real world, via diverse inference methods, including zero-shot motion tracking, goal reaching, and reward inference, and few-shot optimization-based adaptation. Unlike prior on-policy reinforcement learning (RL) frameworks, BFM-Zero builds upon recent advancements in unsupervised RL and Forward-Backward (FB) models, which offer an objective-centric, explainable, and smooth latent representation of whole-body motions. We further extend BFM-Zero with critical reward shaping, domain randomization, and history-dependent asymmetric learning to bridge the sim-to-real gap. Those key design choices are quantitatively ablated in simulation. A first-of-its-kind model, BFM-Zero establishes a step toward scalable, promptable behavioral foundation models for whole-body humanoid control. Webpage:
https://lecar-lab.github.io/BFM-Zero/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose an approach for Unsupervised RL pretraining of Behavior Foundational Models (BFMs) using a recently proposed technique called Forward-Backwards (FB) representation. The approach presented by the authors can essentially be framed as putting together existing techniques and scaling FB method to humanoid whole-body control, so that it could be prompted for different tasks without retraining. Moreover, they even propose a technique for fast adaptation, finetuning and high-level planning of their model.

### Strengths
The authors successfully put together a method from existing related approaches, such that it scales well to complex humanoid whole-body control. Their approach can successfully be used in zero-shot manner, as well as being used for fast adaptation and finetuning. This is a good empirical contribution since existing methods do not enjoy such properties.

### Weaknesses
I think authors could do a better job for a reader to make them understand how this work differs from existing approaches such as [1] and what are the most important differences which make their method work. Currently, the proposed approach differs in a few components:
* Different RL algorithm
* Section 2.2 also proposes a few techniques such as Domain Randomization (DR) and Reward Regularization (RR)

I think disentangling these design choices is important for a reader to understand why this approach works better. I think having additional ablations on DR and RR will greatly improve paper's technical presentation.


[1] Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models, Andrea Tirinzoni, Ahmed Touati, Jesse Farebrother, Mateusz Guzek, Anssi Kanervisto, Yingchen Xu, Alessandro Lazaric, Matteo Pirotta, 2025

### Questions
What are design choices in this approach which make it work significantly better than previous approaches?

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
4

### Summary
This work presents BFM-Zero, an unsupervised RL method that learns a well-structured latent representation of whole-body motions. The authors hereby demonstrate zero-shot task performances on a physical humanoid, bringing unsupervised RL and forward-backward models to actual hardware. BFM-Zero is trained using a motion capture dataset to regularize the large space of behaviors towards more useful behaviors, reducing the search space immensely. It then allows for solving various tasks such as motion tracking or goal reaching in a zero-shot fashion, as well as allows for few-shot adaptation in the real world, where the environment can be modified, for example, by adding additional weight onto the robot. The paper shows that adding common tricks in sim-to-real transfer, such as domain randomization or privileged information into FB-CPR, can successfully transfer to the real world. 

**Main contributions:** 
- Extending the FB-CPR work to real robots, which includes adding Domain Randomization (DR) and an auxiliary rewards
- Presenting a few-shot adaptation via sampling-based optimization for hard-to-achieve tasks.

**Explanation of rating:** Overall, the paper is a first of its kind and shows zero-shot RL on a humanoid robot for whole-body control. However, from a technical perspective, it is not fundamentally different from previous work, such as MetaMotivo (Tirinzoni et al., 2024), and applies previously known techniques for sim-to-real transfer. While its main contribution lies in the transfer to the physical world, it lacks quality in evaluating results there. Given that this is one of the first attempts to bring unsupervised RL and FB to real humanoids, I still lean towards acceptance, but advise the authors to significantly improve the quality of the exposition and extend the real-world evaluation to provide a more substantial contribution.

### Strengths
- The authors show promising results for Zero-shot RL on humanoids. The video presents target pose tracking and continuous dancing, as well as few-shot adaptation in the real world on a challenging hopping example.
- The results are interesting. Especially the natural recovery, the few-shot adaptation, and motions with whole-body contact are impressive.

### Weaknesses
In general, the paper seems rushed and contains many sections with obvious errors (see below for minor issues caught during the read). This is also true for the evaluation, which raises more questions than it answers (see the question section). 

- **Major:** The main contribution of this work is the presentation of a method that works on a real humanoid. However, beyond the few-shot adaptation task, there is little quantitative evaluation in the real world. Furthermore, the results presented on the robot do not demonstrate the method's limits. While it is clear that this formulation comes with many new advantages, I would be interested in understanding where the limits and gaps are from a motion tracking perspective. We have seen quite expressive motion-tracking capabilities in, e.g., He et al., ASAP; how close do we get to such performances? How much expressivity is encoded? Different experiments could help understand the performance in the real world as well as help understand/interpret the gap between the results presented in Fig. 3 and real-world results, e.g., how much can we trust those numbers for the real world? For example, to better understand the sim-to-real gap, it would be helpful to pick 1-2 examples and compute the MPJPE in the real world. This would give a sense of how much the metrics might vary between simulation and reality.
- A Related Work: In general, the related work is incomplete, rushed, and oversimplifies the progress in the field of robotics. There has been significant work before 2025 that has investigated and enabled versatile human motion on robots.
    - DR is a well-explored technique with methods dating back to 2017, such as (Peng et al., Sim-to-Real Transfer of Robotic Control with Dynamics Randomization)
    - Bringing bigger datasets of whole-body human motion beyond locomotion to robots has been investigated before 2025 in many works, some also identified by the authors (He et al., 2024, HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots), but also some not mentioned in this section, such as (Cheng et al., 2024, Expressive Whole-Body Control for Humanoid Robots), (He et al., 2024, OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning) or not included at all, such as (Serifi et al., 2024, VMP: Versatile Motion Priors for Robustly Tracking Motion on Physical Characters), (Fu et al., 2024, HumanPlus Humanoid Shadowing and Imitation from Humans), (Dugar et al., 2024, Learning Multi-Modal Whole-Body Control for Real-World Humanoid Robots), (He et al., 2024, Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation), most of this work is from robotics conferences.

- Minor:
    - L139, L996: broken reference.
    - Figure 3: missing values.
    - Some tables are figures.

### Questions
- In the videos, the robot seems a bit unstable, often drifting in a direction. The authors report MPJPE metrics; however, this does not include the root linear and angular velocities, which seem quite off, judging from the videos. It is not clear to me why the root linear velocity is not part of the observable state. I would assume this could mitigate some of this behavior. Could the authors elaborate on this design choice?
- The Auxiliary critic is a new component added to the framework. However, it is not ablated in any of the results. How important is this part for the sim-to-real transfer at the end? In general, how important is DR? While reading, I was wondering how FB-CPR would perform on a robot, and which components in this work represent the main changes?
- The distribution of the rewards seems very interesting (Fig. 3), especially the observation that there seem to be severe collapses. The authors hypothesize that domain randomization has led to this. move-ego-0-0 appears to be a static pose (judging from the supplementary video where the motion is shown). If the hypothesis of the authors is true, why do we only see this in some of the motions? Investigating for which behaviors this occurs would be interesting.
- The evaluation of the latent space is minimal. e.g., the interpolation result is not surprising; most likely, all those states in between for raising the hand are part of the dataset as well. What happens if we interpolate between dancing and crouching motions? How does the robot's behavior change in those more challenging scenarios where we interpolate trajectories in space and time? 
- The results on the Booster T1 robot are incomplete. It would be interesting to understand if there are weaknesses of the method when applied to this robot, beyond the obvious mechanical limitations. For example, do the robot's specific capabilities affect the training of BFM-Zero in any way (e.g., do the discriminators remain similarly stable)?
- It would also be helpful to see the selected AMASS goal poses for the goal-based evaluation (Fig. 13).
- It is not clear to me why the authors selected 175 motions from the CMU subset of AMASS. This subset contains many walking cycles, which, according to the latent-space plots, appear to be well covered. In simulation, it would be straightforward to test on the entire AMASS dataset, thereby improving understanding of out-of-distribution behavior. 
- Why did the authors not train on the much larger AMASS data? How do the discriminators scale with more data, or do we observe mode-collapse?

### Soundness
3

### Presentation
2

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
The paper trains a single latent-conditioned policy built on a forward-backward (FB-CPR) backbone, that claims to handle motion tracking, sparse-reward tasks, and goal-pose reaching with no task-specific fine-tuning. Training is fully off-policy in Isaac Sim with heavy domain randomisation. The same network is then deployed zero-shot on a Unitree G1, with optional latent-space CEM/dual-loop annealing (Dial-MPC) for quick adaptation. Evaluations span simulations and real hardware, highlighting robustness in balance, recovery, and payload handling.

### Strengths
## Strengths

- Real World Demonstrations: Impressive zero-shot performance on the Unitree G1, including balance maintenance, push recovery from large perturbations, and handling a 4 kg payload, showcases practical sim-to-real transfer.
- Clear Description of Method: The FB backbone, critics, reward shaping, and auxiliary components are explained adequately, though familiarity with prior FB-CPR work is helpful for full understanding.
- Thorough Ablations: Quantitative evaluations of privileged vs. proprioceptive sensing, domain randomization effects, and sim-to-sim robustness (e.g., Isaac to MuJoCo) provide strong evidence for the design choices.

### Weaknesses
## Weaknesses

- Lack of Competitive Baselines: The paper does not compare against state-of-the-art methods like Ex-Body 2, OmniH2O, or Puppeteer on identical tasks and metrics, making it hard to gauge relative advancements.

- Under-Quantified Latent Space Claims: T-sne plot of latents looks interpretable, But assertions about the "promptable" and semantic nature of the latent space are not backed by quantitative measures, weakening the foundation model claims.

### Questions
## Questions / Requested Revisions

- Please add comparisons to competitive baselines (e.g., re-run Ex-Body 2, OmniH2O, Puppeteer on the same tasks and report identical metrics like MPJPE for tracking).
- Expand the evaluation to include error tracking, and behaviour diversity metrics. Provide full breakdowns in the appendix.
- Quantify the latent space semantics with metrics such as cluster-purity scores, linear-probe accuracy for task prediction, or mutual information to substantiate claims.
-  Release training artifacts, including scripts, Docker images, MPC optimization parameters, and safety limits used on the G1, to facilitate reproducibility.


Overall, the work extends BFM to real humanoids with solid engineering, but substantial overlap with prior BFM reduces novelty.
Stronger sim-to-real evidence via detailed real-robot metrics, baselines, and stress tests is needed to convincingly demonstrate progress.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
BFM-Zero introduces a framework to learn humanoid control from unsupervised RL. As opposed to most approaches that define motion tracking rewards over a dataset, BFM-Zero allows the model to explore in an unstructured way and pushes its action distribution toward human-like motions with a GAN-style discriminator trained on motion capture data. The authors demonstrate real world deployments of BFM-Zero.

### Strengths
- The method contrasts the dominant PPO over a motion dataset approach and demonstrates that it works. This may present a path forward to locomotion models trained with less reward engineering.
- Smooth and structured latent space is likely helpful for multimodal prompting or other downstream tasks.
- Pretraining seems scalable, subject to limits of the simulation model.
-Real world sim-to-real deployment of off policy-trained model.

### Weaknesses
- "Foundation model" itself may be overclaiming since the model is essentially just a low level controller without rich vision or touch sensing.
- Prompting occurs in a human-uninterpretable latent space rather than something like language.
- More detailed comparisons of computational cost and representation quality with typical on-policy methods would be more convincing to validate the usefulness of such an approach.
- Policy quality is still upper-bounded by simulation environment

### Questions
I'm curious to learn more about the latent space design decisions. How does the dimensionality affect the ease of prompting and performance of the model?

Does the model learn diverse behaviors or does it mode-collapse like on-policy algorithms?

### Soundness
3

### Presentation
3

### Contribution
3
