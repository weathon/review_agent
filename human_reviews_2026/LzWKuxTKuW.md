# Much Ado About Noising: Dispelling the Myths of Generative Robotic Control

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Generative models, like flows and diffusions, have recently emerged as popular and efficacious policy parameterizations in robotics. There has been much speculation as to the factors underlying their successes, ranging from capturing multimodal action distributions to expressing more complex behaviors. In this work, we perform a comprehensive evaluation of popular generative control policies (GCPs) on common behavior cloning (BC) benchmarks. We find that GCPs do not owe their success to their ability to capture multimodality or to express more complex observation-to-action mappings. Instead, we find that their advantage stems from iterative computation, provided that intermediate steps are supervised during training and this supervision is paired with a suitable level of stochasticity. As a validation of our findings, we show that a minimal iterative policy (MIP), a lightweight two-step regression-based policy, essentially matches the performance of flow GCPs. Our results suggest that the distribution-fitting component of GCPs is less salient than commonly believed and point toward new design spaces focusing solely on control performance. Videos and supplementary materials are available at https://anonymous.4open.science/w/mip-anonymous/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper questions why generative policies, specifically diffusion-based and flow-based imitation learning models, appear to outperform regression-based policies in robot learning. It argues that their success is not due to better distribution modeling or multimodality capture, but rather to algorithmic structure. The paper examined a few hypotheses and then concluded that GCPs are not fundamentally superior distribution learners; their strength arises from how their training and inference processes structure learning.

### Strengths
- The paper conducted comprehensive experiments and had careful comparison with architecture parity and identical backbones, ensuring fairness and validating the hypothesis. It has a large-scale benchmark study across 27 imitation learning datasets that isolates the actual causes of GCP success.
- It separates generative modeling goals from control objectives and proposes a new non-generative baseline that rivals flow policies MIP.
- It shows that distribution fitting is largely irrelevant for single-task control, which is a major conceptual correction.
- The insight that a two-step regression model can rival state-of-the-art flow policies is surprising.

### Weaknesses
- The experiments fix inference to deterministic modes, although guided sampling or diffusion-based control might behave differently.
- The study focuses on a single-task setting, while multi-task or goal-conditioned GCPs might still benefit from distributional modeling.
- The paper does not test whether C2+C3 advantages persist under RL fine-tuning or long-horizon planning, which is crucial for robotic applications.

### Questions
- The paper empirically shows that the combination of stochasticity and iteration works. Is the improvement due to smoother gradient landscapes, implicit ensemble effects, or noise-induced curvature exploration or is there any intuition?
- Can MIP generalize across different goals or objects without retraining? This would test whether its simplicity generalizes or overfits. Compared to it, flow-based policies often work well in multi-task settings where distributional conditioning is crucial.
- How does MIP perform under camera noise, lighting changes, or slight object translation? Does MIP maintain robustness to sensor noise or domain shift? Since MIP is deterministic, it may degrade fast.

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
3

### Summary
This paper investigates why generative control policies (GCPs)—such as flow- or diffusion-based action models—perform better than standard regression policies in behavior cloning (BC).
Rather than assuming the benefit arises from distributional modeling (learning multimodal 
𝑝
(
𝑎
∣
𝑜
)
p(a∣o)), the authors dissect GCPs into three conceptual components:

C1: Distributional learning (learning a stochastic action distribution)

C2: Stochastic training (injecting noise during training)

C3: Supervised iterative computation (multi-step prediction with intermediate supervision)

Through controlled experiments across 27 BC benchmarks (Robomimic, Adroit, MetaWorld, etc.) with matched architectures, they show that C1 is not important, while C2 and C3 are the true drivers of GCP performance.
Based on this, they propose a Minimal Iterative Policy (MIP) that keeps only C2 + C3—achieving parity with flow models but with deterministic inference and far lower compute cost.
They also introduce manifold adherence as a diagnostic for how closely policy outputs stay near expert data manifolds.

### Strengths
Rigorous experimental isolation of C1, C2, C3 effects.

Unusually broad benchmark coverage (27 tasks).

Strong ablations: flow vs regression vs MIP.

Negative result clearly demonstrated and well-reasoned.

Practical takeaway: simple two-step deterministic policy matches flow performance.

“Manifold adherence” metric offers a novel lens on robustness.

### Weaknesses
All evaluations are offline imitation; no fine-tuning or real-robot rollouts to confirm closed-loop stability.

MIP’s hyperparameters (e.g. noise scale) are fixed; sensitivity analysis would help.

### Questions
Have you tried learning $\(t^*\)$ or using a schedule instead of a fixed value (0.9)?

Could MIP’s iterative refinement be stacked (3–4 steps) without re-introducing flow losses—does performance saturate after two steps?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates why generative control policies often outperform regression-based ones in imitation learning. The authors aim to show (through controlled evaluations) that this advantage is not due to multimodal behaviors or higher expressiveness of generative models, but rather due to manifold adherence. They argue that, since control policies mainly map observations to actions, such complexity is unnecessary; what matters is incorporating some degree of stochasticity and supervised iterative computation. Based on this insight, they propose a minimal approach that achieves strong results on several continuous control benchmarks.

Overall, I find the contribution valuable for the community. The main evaluation, while not entirely conclusive, is well-designed and offers interesting (and somewhat counterintuitive) insights, showing that generative policies’ strength may not come from multimodality or expressiveness. The second contribution, the MIP framework, also seems practical and empirically validated. I have mixed opinions overall: some parts of the evaluation are not fully rigorous, but the results and reasoning are promising. I would give this paper an initial rating of 6 and look forward to further discussion with the authors and other reviewers.

### Strengths
1. I like the motivation of this work. Decomposing and analyzing the underlying reasons behind the success of generative policies is valuable, especially for RL and imitation learning, where understanding why a model works often matters more than just scaling it up. As often said, RL doesn’t necessarily benefit from ever-larger networks like LLMs or VLMs do  (IMO since the right actions or behaviors come from a subtle, structured distribution that doesn’t simply improve with more data or capacity..).

2. For the most part, the evaluation design is solid, and the conclusions drawn from the hypothesis testing seem meaningful and informative.

3. The proposed minimal algorithm looks sound and provides fresh insights into behavior cloning and how to simplify it (with stochasticity, and iterative computing but not necessarily model the whole distribution) without losing performance.

### Weaknesses
I list both my (tentative) weaknesses and questions together here, since some of them overlap, and a few points might just come from my not fully understanding certain parts of the evaluation.

1. On the conclusion that GCPs cannot really produce multi-modal actions, I have some concerns about the setup of evidence C. Even in deterministic environments and with deterministic policies, the underlying mapping from observation to action can still be stochastic or even multi-modal at the distribution level (e.g., multiple valid actions for the same state). So using this setting to claim the absence of multi-modality might be a bit too strong.

2. Related to that, the theoretical part on policy expressiveness seems to lean heavily on the assumption that there is no multi-modality (so that you can use the k-log-concave unimodal distribution arguments). But this then fully depends on the correctness of the earlier empirical conclusion about multi-modality. In practice, we do observe multimodal behaviors in pretrained GCPs, e.g. in the DP-like setups, so I’d like to see more clarification here.

3, For the claim that C2 supports C3 (i.e., noise injection can replace data augmentation), I’m not fully convinced. The noise here is injected on the action/control side or in the dynamics, while “data augmentation” in imitation learning can also mean visual, semantic, or context-level augmentation, which GCPs can implicitly provide. It might help to define more precisely what kind of “augmentation” you mean.

4. For the pixel-based control part where you show RCP can match GCP, I suspect this may partly come from the architectural alignment you used. If that’s the case, then H1 may not be as strong a takeaway as presented, it could just be an architecture effect. It would be good to clarify whether there is another implication here about the observation-to-action mapping.

5. To make the conclusions more universal, it would help to also test diffusion-based and transformer-based imitation policies. Even if they share the spirit of flow models, it would make the claim solid that the result is not specific to one generative family.

6. About the choice of $t_{*}=0.9$ is this environment-specific or a general setting? If it’s a trade-off hyperparameter, it would be good to say so and show a small sensitivity analysis

### Questions
Other than the points above, I also have a more general question. From the perspective of the true action generation process, it’s inherently a distribution. So, in principle, learning this distribution directly (as GCPs do) still seems like the optimal approach. The paper focuses on which components of the objective function work best, but from a first-principles standpoint, modeling the full action distribution still feels like the most appropriate formulation IMO

### Soundness
2

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
This paper seeks to understand the "secret sauce" behind the recent success of Generative Control Policies (GCPs) in robotics. While the field has largely attributed this success to the power of generative models (like diffusion) to handle multi-modality or complex representations, this paper systematically analyzes those common assumptions.

The authors argue that the generative, distributional aspect of these models is their least important feature. Instead, the performance leap comes from a powerful and previously under-appreciated combination of two other design choices: Supervised Iterative Computation (C3) and Stochasticity Injection (C2).

There're several claims in the paper

1. Analysis on multimodality of GCPs
The paper's argument is built in two parts. Firstly, the authors show that on tasks known to be multi-modal, GCPs show no significant performance advantage. Using a set of ablation experiments (e.g., showing that taking the mean action doesn't cause performance to collapse), they provide evidence that the models are not, in fact, learning distinct action modes.
The paper challenges the idea that the iterative nature of GCPs makes them deeper or more expressive. Through a theoretical argument (Theorem 1) and an empirical measurement of the Policy Lipschitz Constant, the authors show the exact opposite: successful GCPs learn smoother, less complex functions than their regression counterparts.

2. the authors introduce three core components of any GCP:
C1: Distributional Learning 
C2: Stochasticity Injection (adding noise during training)
C3: Supervised Iterative Computation (refining an action over multiple steps)
The paper proposes simplified algorithm they call the Minimal Iterative Policy (MIP). MIP is designed to completely discard the complex generative component (C1) and rely only on C2 and C3.
The results show that this simple MIP matches the performance of the full, state-of-the-art Flow-based GCPs. This experiment conclusively demonstrates that the C2+C3 combination is the true engine of performance.

3. Finally, the paper provides a powerful explanation for why this C2+C3 combo is so effective.
They find that while MIP is no better than regression at simple imitation (i.e., they have the same validation loss), it is dramatically better at producing plausible actions when it encounters unfamiliar states. This iterative refinement process learns to fail gracefully, always steering its actions back toward the "manifold" (the space of valid expert actions). The iterative process (C3) fails on its own. The authors find that the injected noise (C2) is the critical ingredient that stabilizes the training, allowing the iterative refinement to learn its task without being derailed by compounding errors.

### Strengths
1. A key finding of this paper is that when using the same network architecture, traditional Regression Control Policies (RCPs) are highly competitive with modern Generative Control Policies (GCPs). This is an important contribution, as it helps correct a potential misconception in the field (stemming from earlier works that used different architectures for baselines) that diffusion policies inherently offer a large performance gain over standard behavior cloning on these tasks.

2. A major strength of this paper is its systematic methodology. The authors first propose a novel taxonomy to deconstruct GCPs into three key components, and then create 'ablation' models (MIP, RR, and SF) to empirically isolate and test the precise performance contribution of each component.

### Weaknesses
1. The paper's use of the Lipschitz constant as a metric for "expressivity" is a potential point of confusion. As you noted, the Lipschitz constant measures smoothness or robustness (how much the output can change for a small input change), not necessarily the representational capacity (the class of functions a model can learn). While their point (that GCPs find smoother solutions) is valid, their terminology can be debated.

2. The paper's theoretical argument, presented as Theorem 1 (Informal), feels underdeveloped. It attempts to formally decouple the number of iterative steps from an equivalent network's depth, but the premise of this connection to a diffusion policy's true expressiveness is not rigorously established. As presented, it functions more as a high-level intuition than a solid proof, making it a weak pillar for the paper's claims.

3. A significant limitation is the paper's narrow scope. The strong conclusion that Distributional Learning (C1) is the "least important factor" is derived entirely from a single-task, high-precision setting. This overlooks the primary strength of generative models: capturing complex, diverse distributions, which is fundamental for multi-task learning. The success of diffusion models in other domains (e.g., vision) is precisely their ability to model complex, multi-modal data distributions. It is highly plausible that C1 is, in fact, the critical component for scaling to multi-task behavioral distributions, making the paper's central finding a potential artifact of its limited, single-task evaluation.

4. A particularly counter-intuitive finding is the diffusion policy's failure to produce multi-modal behavior, given that diffusion models are expressly designed to capture complex data distributions. This observed uni-modal output may not be a failure of the model itself, but rather an artifact of the DDPM sampling process.
DDPM functions as a stochastic sampler (an SDE), and its Langevin dynamics term, while beneficial for error correction, could be the mechanism responsible for mode concentration. This stochastic term might be "correcting" the trajectory so effectively that it collapses different potential paths into a single, high-density outcome.
A crucial follow-up experiment would be to test the policy with a deterministic DDIM sampler (an ODE). This would remove the influence of the Langevin term, offering a clearer view of the true learned distribution. It's possible that this would reveal the multi-modality the model did learn, even if it simultaneously lowers task performance by removing the stochastic error correction.

Overall, this is an interesting paper with provocative findings. However, its conclusions feel brittle. They contradict the established behavior of diffusion models in other domains, a discrepancy likely stemming from the paper's narrow, single-task focus. The arguments lack full rigor, relying on conflated terminology (Lipschitz vs. expressivity) and informal theorems. The paper successfully isolates a powerful mechanism (C2+C3) for this specific setting, but it does not provide a generalizable understanding of generative policies in robotics.

### Questions
Besides the questions I raised in the weakness part. 

Minor Point::
The paper's citation on line 159 for the concept of "straight interpolation" (or similar flow matching/rectified flow concepts) is incomplete. The authors cite only "Flow Matching" (Lipman et al. [3]), but this was one of several foundational works that introduced this concept concurrently.

[1]Albergo, Michael, and Eric Vanden-Eijnden. "Building Normalizing Flows with Stochastic Interpolants." ICLR 2023 Conference. 2023.
[2] Heitz, Eric, Laurent Belcour, and Thomas Chambon. "Iterative α-(de) blending: A minimalist deterministic diffusion model." ACM SIGGRAPH 2023 Conference Proceedings. 2023.
[3] Lipman, Yaron, et al. "Flow Matching for Generative Modeling." The Eleventh International Conference on Learning Representations.
[4] Liu, Xingchao, and Chengyue Gong. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." The Eleventh International Conference on Learning Representations.

### Soundness
2

### Presentation
3

### Contribution
2
