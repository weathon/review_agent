# TraSCE: Trajectory Steering for Concept Erasure

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Recent advancements in text-to-image diffusion models have brought them to the public spotlight, becoming widely accessible and embraced by everyday users. However, these models have been shown to generate harmful content, such as not-safe-for-work (NSFW) images. While approaches have been proposed to erase such abstract concepts from the models, jail-breaking techniques have succeeded in bypassing such safety measures. In this paper, we propose TraSCE, an approach to guide the diffusion trajectory away from generating harmful content. Our approach is based on negative prompting, but as we show, the widely used negative prompting strategy is not a complete solution for concept erasure and can easily be bypassed in a simple corner case. To address this issue, we introduce two techniques. We first borrow the idea of concept negation from compositional generation and propose to use it instead of the conventional negative prompting. Second, we introduce a localized loss-based guidance that enhances the modified negative prompting technique by steering the diffusion trajectory. We demonstrate that our proposed method achieves state-of-the-art results on various benchmarks in erasing harmful content, artistic styles, and objects, including on benchmarks introduced by red teams without impacting unrelated concepts. Our proposed approach does not require any training, weight modifications, or training data (either image or prompt), making it easier for model owners to erase new concepts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a method to erase harmful content from diffusion models. They identify a corner case where vanilla negative prompting fails and replace it with a safer variant that better suppresses the targeted concept. They also introduce a localized, loss-based guidance term that nudges sampling toward the unconditional trajectory when the prompt behaves like the banned concept. The paper evaluates the approach on adversarial attack benchmarks and reports results for erasing artistic styles and objects.

### Strengths
1. The paper tackles a timely and important safety challenge: removing harmful content from generative models. 
2. The paper is clearly written, easy to follow, and the proposed method is both simple and elegant.

### Weaknesses
1. The method assumes that falling back to the unconditional prior is a safe default, which may not always hold. If the base model’s unconditional distribution already leans toward risky or unwanted content in certain contexts, anchoring to it won’t prevent leakage. The approach can fail in such cases.
2. I also have sme concerns regarding the generation quality.  In diffusion, coarse scene layout forms early, while objects solidify mid-trajectory and fine details emerge late. So, when it comes to erasing the objects, the loss in the paper only activates when $$ \hat{\epsilon}_\theta(x_t, e_p) - \hat{\epsilon}_\theta(x_t, e_{np}) $$
 becomes small, typically mid/late. By that time, the scene and object placement are largely decided. Nudging the latent then can remove or weaken the target object only by distorting the image. This may be true even for complex safety concepts. For instance, the qualitative result shown in Figure 4, where the authors aim to “erase a car,” looks distorted and challenges the claim that the method preserves generation quality.

### Questions
1. Can the authors provide a plot that shows how the difference term varies across timesteps during the erasure of objects? This may give some clarity on the concern in 2 above.

### Soundness
3

### Presentation
4

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
The paper proposes TraSCE, an inference-time method for concept erasure in text-to-image diffusion. It combines (i) a modified negative prompting formulation (replacing the conventional “negative as base” with an unconditional base plus a difference term) and (ii) a localized loss–based guidance that nudges the denoising state away from regions aligned with the undesired concept each step. Experiments report lower attack-success rates on several adversarial NSFW/violence benchmarks and show applications to artistic styles and object erasure.

### Strengths
1.  Addresses safety in diffusion models, especially robustness to prompt‑based jailbreaks.
2. Integrates a geometric “trajectory steering” view into the diffusion process, offering intuitive control over latent evolution.

### Weaknesses
1. While “trajectory steering” offers a coherent new perspective, the implementation closely resembles classifier-free or loss-based guidance mechanisms already explored in prior works (e.g., SLD). The novelty primarily lies in problem framing and loss design rather than theoretical advancement.
2. The per-step gradient update increases sampling time by 2–3×, which may limit deployment for large-scale or real-time use.

### Questions
1. How robust is the method to partial or semantically related prompts (e.g., “bikini” vs. “swimsuit”)? Does the trajectory steering generalize to paraphrased adversarial prompts?
2. Can the authors quantitatively characterize how the diffusion trajectories differ between baseline CFG, SLD, and TraSCE?
5. How does TraSCE behave when multiple negative concepts are specified simultaneously (e.g., “no nudity, no gore”)? Does the trajectory steering remain stable?

### Soundness
3

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
5

### Summary
This study introduces TraSCE, a method aimed at mitigating the generation of harmful content (e.g., sexual elements) in T2I diffusion models. The authors propose guiding the diffusion trajectory away from problematic generation paths. The proposed method combines two part: a modification of traditional negative prompting based on the classifier-free guidance, and a localized loss-based guidance, which further enhance the guidance performance. The performance of the proposed method is outstanding, demonstrating effectiveness on multiple models and concept erasure tasks.

### Strengths
1. The method is very clear and easy to understand.
2. The proposed method performs excellently and shows outstanding results on multiple evaluation benchmarks. 
3. The authors' experimental setup is comprehensive, taking into account various evaluation tasks, erasure robustness, and different base models.

### Weaknesses
1. The application of the proposed method seems to be based on an unreasonable setting: that the specific category of harmful content must be predefined for the current generation. This is impractical in real-world scenarios. In contrast, recent related works [1, 2, 3] adopt a "detect-then-erase" mechanism, which first determines if a specific concept has been generated and only then performs concept erasure. This appears to be a more reasonable setup.
2. The additional generation time introduced by the proposed method appears to be unacceptable. For instance, in a standard classifier-free guidance model, the application of Eq. 3 directly incurs an extra 50% inference time, and the feedback optimization in Eq. 4 is even more time-consuming.
3. I appreciate that the authors adapted their experiments to the new FLUX model. However, related to the previous point, the authors must report the additional inference time overhead that the proposed method introduces when applied to FLUX. Given that FLUX is a  model with a very large number of parameters (15x than sd1.4), it is difficult to believe that the optimization mechanism from Eq. 4 can be practicably applied to it.

[1] SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image and Video Generation, iclr25

[2] Detect-and-Guide: Self-regulation of Diffusion Models for Safe Text-to-Image Generation via Guideline Token Optimization, cvpr25

[3]  Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation, cvpr25

### Questions
Please address the weaknesses

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
3

### Summary
The paper proposes TraSCE, a training-free method for concept erasure in text-to-image diffusion models. TraSCE combines a modified negative prompting strategy with a localized loss-based guidance to steer diffusion trajectories away from target concepts. TraSCE is evaluated on benchmarks that include adversarial prompts and tasks such as erasing NSFW content, artistic styles, and specific objects, achieving state-of-the-art reductions in attack success rates while preserving image quality across unrelated concepts.

### Strengths
**S1:** TraSCE operates at inference time, eliminating the need for costly retraining or data collection. This makes it easily deployable for model owners to adapt to new concepts.

**S2:** TraSCE shows significant reductions in attack success rates against black-box adversarial attacks, with minimal degradation in general image quality.

**S3:** Experiments cover diverse erasure tasks using multiple metrics, providing a broad assessment of the method's applicability.

### Weaknesses
**W1:** The main concern about this paper is its limited novelty and insufficient distinction from prior work. The core component of TraSCE, the modified negative prompting, is adapted from Liu et al. (2022) on concept negation but lacks adequate justification for its novelty. While the addition of localized loss-based guidance is claimed as new, it fails to be differentiated from existing guidance techniques, such as classifier guidance. For instance, Schramowski et al. (2023) also employ trajectory steering for safety, but TraSCE's ablation studies only compare against basic negative prompting, lacking a deep analysis of how the loss function advances beyond prior art. Therefore, the authors should conduct a comparative analysis with recent inference-time methods, highlighting theoretical or empirical distinctions, and perform ablations to quantify the independent contribution of the loss guidance versus the negative prompting.

**W2:** The experimental results lack dynamic adversarial testing, white-box attack settings, and human evaluation. The paper does not test against adaptive white-box attacks that exploit TraSCE's gradient information, thereby limiting its claims of robustness. For artistic style and object erasure, assessments depend solely on automated classifiers without human evaluation or diversity metrics, risking overestimation of erasure effectiveness due to dataset biases. The authors should incorporate dynamic adversarial testing, add human evaluations for subjective tasks (e.g., artistic styles), and report diversity scores to ensure that erasure does not harm output variety.

**W3:** TraSCE increases inference time by approximately 2.6 times, which could hinder real-time applications. Hyperparameters are task-specific and tuned empirically, yet no sensitivity analysis or guidance for adaptive selection is provided. To enhance practicality, the authors could benchmark TraSCE on resource-constrained devices and suggest approximations (e.g., reducing gradient steps or using sparse updates).

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
