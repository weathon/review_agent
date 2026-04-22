# From Language to Locomotion: Retargeting-free Humanoid Control via Motion Latent Guidance

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 2, 8, 6

## Abstract
Natural language offers a natural interface for humanoid robots, but existing text-to-motion pipelines remain cumbersome and unreliable. They typically decode human motion, retarget it to robot morphology, and then track it with a physics-based controller. However, this multi-stage process is prone to cumulative errors, introduces high latency, and yields weak coupling between semantics and control. These limitations call for a more direct pathway from language to action, one that eliminates fragile intermediate stages. Therefore, we present RoboGhost, a retargeting-free framework that directly conditions humanoid policies on language-grounded motion latents. By bypassing explicit motion decoding and retargeting, RoboGhost enables a diffusion-based policy to denoise executable actions directly from noise, preserving semantic intent and supporting fast, reactive control. A hybrid causal transformer–diffusion design further ensures long-horizon consistency while maintaining stability and diversity, yielding rich latent representations for precise humanoid behavior. Extensive experiments demonstrate that RoboGhost substantially reduces deployment latency, improves success rates and tracking accuracy, and produces smooth, semantically aligned locomotion on real humanoids. Beyond text, the framework naturally extends to other modalities such as images, audio, and music, providing a general foundation for vision–language–action humanoid systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RoboGhost, a framework for language-guided humanoid locomotion that eliminates the need for conventional motion retargeting. Instead of decoding explicit motion sequences and manually adapting them to specific robot morphologies, RoboGhost leverages language-conditioned motion latents as direct control signals to a diffusion-based policy. The method combines a causal transformer-based motion latent generator with a teacher-student policy setup (using a Mixture-of-Experts RL teacher and a diffusion-based student policy) to enable retargeting-free, semantically guided robot control. Extensive experiments across simulation and a real humanoid platform (Unitree G1) demonstrate improved efficiency (lower latency), tracking, and robustness compared to traditional pipelines.

### Strengths
- The paper convincingly argues for eliminating the retargeting step in language-to-motion pipelines, highlighting both latency and cumulative error benefits.
- The paper proposes a thoughtful shift in control pipeline design by the hybrid of a causal transformer for latent generation and a diffusion policy for action denoising .
- Multiple quantitative tables provide a comprehensive performance landscape. Results show clear improvements in motion generation quality (precision, FID), tracking accuracy, and especially latency (17.85s → 5.84s).

### Weaknesses
- While the student policy avoids decoding/retargeting at deployment, the teacher still tracks explicit reference motions with privileged info, then the student distills from that oracle. This leaves open whether gains come mainly from inference-time design or from the strong teacher and filtering/curation choices.
- The central claim is a retarget-free latent-to-policy pipeline for humanoid control, but the current experimental evidence only partially targets this claim; stronger head-to-head comparisons with explicit-retargeting SOTAs, swap-in pretrained latents on control metrics, and detailed diffusion-student ablations are needed to firmly establish that the retarget-free design—not ancillary choices—drives the gains.

### Questions
1. The paper shows that implicit retargeting with distillation outperforms a PHC-style teacher, but it does not compare against **state-of-the-art explicit retargeting approaches** (e.g., ProtoMotions, GMR) that report substantial tracking gains from carefully engineered retargeting; please add these as strong baselines under matched settings.
2. Please justify the need to **learn your own human motion latent space** rather than using a pretrained motion latent (e.g., MLD, MotionGPT, PULSE, SMAP) by providing swap-in experiments where your latent is replaced with these pretrained alternatives (both frozen and fine-tuned) and comparing not only generation metrics but also control-side metrics.
3. The **diffusion student policy** section is ambiguous (e.g., “noise scale and the MDP share a time step $t$”); please provide a clear **algorithm box or pseudocode** detailing the DAgger data-collection and aggregation procedure (rollout scheme, teacher query frequency, buffer management), the supervision target ($\epsilon$ v.s. $x_0$ prediction), the noise schedule and time-step sampling, conditioning injection (history/ proprioception/ latent), and the online distillation loss composition and stabilization tricks, and add fine-grained ablations on denoising steps, noise scale, sampling strategy, and teacher query ratio with success-rate and latency curves.
4. Current baselines are predominantly **human motion generation methods**; to substantiate advantages for humanoid motion generation/control, please include or align representative **humanoid-specific baselines** on the same tasks and metrics and discuss where your method excels.
5. Please specify exactly how $E_{mpjpe}$ is computed in Table 3 Implicit Tracking, and clarify the conditioning: is the student driven solely by the text-derived latent from the HumanML/Kungfu test captions while errors are measured against the paired dataset reference motion? If not, please detail the source of the latent and the reference used for scoring.

### Soundness
2

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
This paper proposes RoboGhost, a retargeting-free framework for language-guided humanoid locomotion. The key idea is to bypass motion retargeting by using language-conditioned motion latents to guide a diffusion-based humanoid policy. The system follows a teacher–student distillation scheme where the teacher policy is trained with retargeted motion data and privileged simulation information, and the student policy learns to act directly from latent representations, eliminating explicit retargeting during deployment. Experiments show improved inference speed and comparable or better motion-tracking accuracy compared to retargeting-based baselines.

### Strengths
- **System design clarity:** The two-stage training (teacher-student) and latent-driven diffusion policy are clearly described, supported by architectural and algorithmic details.
- **Efficiency gains:** The results show a substantial improvement in inference time.
- **Broad applicability:** The framework generalizes to different modalities (text, image, audio), providing a foundation for multimodal humanoid control.

### Weaknesses
- **Motivation and logic gap:**
The core motivation is to bypass retargeting, yet the teacher policy still relies on retargeted motion data. Hence, the student’s performance remains bounded by retargeting quality. Logically, distilling from a retargeted teacher should not outperform the original retargeted paradigm. However, the results show the opposite. The paper lacks a clear explanation of how or why the student surpasses its teacher, which raises questions about evaluation fairness or metric alignment. The main advantage of retargeting-free I acknowledge here is it can improve inference speed. But there is only very limited experiments on this. No break down on which module cost the most times, as well as lack comprehensive study on different retargeting methods (e.g. GMR [1]), and different iteration times of PHC.

- **Experiments:** 
The paper evaluates motion generation metrics (e.g., FID, Diversity) even though the final task is humanoid control, making these metrics less meaningful. Most performance improvements (e.g., success rate, tracking error) are minor; the main advantage comes from inference speed. And no real-world or simulation qualitative comparison which is more intuitive to see whether this method is useful.

- **Evaluation inconsistency:**
The “Evaluation of Motion Tracking Policy” section uses “ground truth” motions that seem derived from retargeted data. If so, evaluating a non-retargeted policy against a retargeted reference introduces bias and makes interpretation difficult. The authors should clarify where the ground truth comes from and justify why this comparison is meaningful.

- **Limited comparative discussion:**
No discussion and comparison with very related works, LeVERB [2] and UH-1 [3], which also pursue latent-based control or retargeting-free. A direct comparison or discussion of conceptual differences is needed.

- **Presentation issues:** 
Table 1 (MM-Dist) highlights the wrong value.

[1] https://github.com/YanjieZe/GMR

[2] LeVERB: Humanoid Whole-Body Control with Latent Vision-Language Instruction

[3] Learning from Massive Human Videos for Universal Humanoid Pose Control

### Questions
- Since the student policy is distilled from a teacher trained with retargeted data, how can it outperform the retargeting-based pipeline? What factors (e.g., noise injection, diffusion regularization) contribute to this paradoxical result?
- What exactly constitutes the ground truth for evaluating the motion tracking policy?
- Can you provide a stage-wise breakdown of inference time (e.g., motion generation, diffusion denoising, policy forward) to show where the latency reduction originates?
- How does RoboGhost compare to LeVERB and UH-1 in terms of architecture, goal, and performance?
- Could the authors include additional ablations on PHC iterations and different retargeting methods to substantiate the efficiency advantage?

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
3

### Summary
"From Language to Locomotion: Retargeting-free Humanoid Control via Motion Latent Guidance" simplifies the deployment stack of text to humanoid robot locomotion. Whereas prior approaches generate human motion then retarget onto a robot, this method learns motion representations that can be decoded to robot action directly. This improves inference latency and removes error-prone components of existing pipelines. The authors demonstrate real-world deployments on a Unitree G1 which show physically stable and semantically plausible robot actions corresponding to various text prompts.

### Strengths
- This method tackles a real, error-prone piece of prior works' deployment pipelines. The proposed solution is simple and practical, reducing inference latency and compounding errors.
- Most aspects of the method are well-justified and follow conventions from appropriate subfields. For example, AdaLN is a well-tested conditioning scheme for image diffusion models.
- Real world robot deployment clearly works and generates locomotion that resembles the input prompt.
- Paper is well-written and the components of the method are clear.

### Weaknesses
- Mixture of Experts is simply a compute efficiency optimization for transformers to maintain inference cost while scaling parameter count. I appreciate the expert count ablation in the appendix, but the obvious comparison is simply making the model larger. I find this part of the method poorly justified.
- There is minimal analysis of the diffusion model sampling latency and tradeoffs with increasing/decreasing the number of sampling steps. Many existing diffusion distillation methods could reduce the number of NFEs (neural function evaluations) necessary at test-time as well.
- Since the authors propose a new motion representation learner, it would be helpful to probe those representations to understand them better.
- It's unclear to me why the text should be encoded through LaMP instead of just using a generic text embedding model like T5. For example, Stable Diffusion 3 moved from CLIP encoding to T5 encoding for text.

### Questions
I see that prior works call their motion FD metric FID, but I find this misleading since an Inception network is not used. I think it's good to clarify this and potentially define a new term for this metric.

Note: typo at line 259

### Soundness
3

### Presentation
3

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
This paper presents RoboGhost, a novel framework for language-guided humanoid locomotion that eliminates the explicit motion retargeting step. The authors identify that conventional multi-stage pipelines (text-to-motion, decoding, retargeting, tracking) suffer from high latency and cumulative errors. RoboGhost addresses this by proposing a latent-driven approach. A motion generator, based on a hybrid transformer-diffusion architecture , produces a motion latent from a text prompt. This latent directly conditions a diffusion-based student policy , which is trained to denoise executable actions using supervision from an MoE-based teacher policy. This retargeting-free pipeline is shown to significantly reduce deployment latency and improve tracking success rates in both simulation (IsaacGym, MuJoCo) and on a real Unitree G1 humanoid.

### Strengths
1. Novel and Elegant Framework: The primary strength is the "retargeting-free" pipeline. Instead of the conventional "generate-retarget-track" approach, RoboGhost proposes a "generate-control" paradigm. Using motion latents as a direct conditioning signal for a diffusion-based action policy  is a effective idea. It tightly couples semantic intent from language with low-level control, addressing the "weak coupling" problem of prior work.
2. Technically Sound and Thorough: The work is technically solid. The two-stage teacher-student architecture is well-designed, featuring a strong MoE-based teacher for robust supervision and a novel diffusion-based student for generalization. The inclusion of Causal Adaptive Sampling (CAS) to focus on difficult motion segments  further strengthens the training methodology.
3. Real-World Relevance and Efficiency: The reported reduction in deployment latency (from 17.85s to 5.84s) and improved success rates (+5%) are compelling for real-time robotic applications. The sim-to-real transfer without manual tuning is particularly impressive.

### Weaknesses
1. Limited Scope of Baseline Comparisons: The paper's primary evidence for the superiority of its latent-driven framework stems from the comparison in Table 3, which contrasts "Ours-Implicit" against "Ours-Explicit" (an author-implemented traditional pipeline with retargeting). While this serves as an effective internal ablation study that validates the advantages of the latent-driven paradigm over the explicit one, it is insufficient for fully contextualizing the work. The paper lacks a direct, end-to-end performance comparison against other external, published, state-of-the-art pipelines for language-guided humanoid control. It is unclear how RoboGhost would perform against a system with a highly optimized SOTA tracking policy, such as GMT (Chen et al., 2025) or ExBody2 (Ji et al., 2024).
2. Strong Dependency on the Initial Motion Dataset and Limited Extrapolation: The effectiveness of the entire RoboGhost framework, from the semantic content of the motion latent ($l_{ref}$) to the capability of the MoE Teacher Policy ($\pi_t$), is fundamentally bound by the diversity and quality of the initial human motion dataset used for training. While the paper demonstrates success on provided motions, its ability to genuinely handle open-ended language is questionable when commands necessitate novel combinations or extrapolation far outside the training distribution (e.g., highly stylized, physically novel, or long-horizon composite actions). A failure in the dataset translates directly into a failure of the Stage 1 latent generator to encode valid semantics, thereby constraining the overall potential of the open-ended language interface.

### Questions
Could the authors provide a more detailed analysis, or ideally, a direct comparison, of the real-world deployment performance between the Diffusion Policy and a conventional RL-Distilled Student Policy (e.g., a simple, optimized MLP policy trained to imitate $\pi_t$ using non-privileged observations)?

Specifically, I am interested in the practical trade-offs introduced by the Diffusion architecture upon deployment:

1. Inference Latency/Throughput: Does the iterative nature of the Diffusion sampling (e.g., DDIM steps) impose a practical latency penalty (higher per-step inference time) on the real robot compared to a single forward pass of a distilled MLP policy? If so, is this increased latency a limiting factor for high-frequency control?
2. Real-World Benefits: Beyond the simulation-based generalization shown in Table 4, what specific additional benefits (e.g., smoother motion, greater robustness against sensor noise, better handling of physical disturbances due to the mode-covering nature of diffusion) does the Diffusion Policy provide in the real-world deployment that justifies its computational complexity over a simpler, conventionally distilled policy?

### Soundness
3

### Presentation
3

### Contribution
3
