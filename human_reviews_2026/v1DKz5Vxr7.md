# MotionStream: Real-Time Video Generation with Interactive Motion Controls

- Avg Score: 5.50
- Decision: Accept (Oral)
- Scores: 6, 6, 6, 4

## Abstract
Current motion-conditioned video generation methods suffer from prohibitive latency (minutes per video) and non-causal processing that prevents real-time interaction. We present MotionStream, enabling sub-second latency with up to 29 FPS streaming generation on a single GPU. Our approach begins by augmenting a text-to-video model with motion control, which generates high-quality videos that adhere to the global text prompt and local motion guidance, but does not perform inference on the fly. As such, we distill this bidirectional teacher into a causal student through Self Forcing with Distribution Matching Distillation, enabling real-time streaming inference. Several key challenges arise when generating videos of long, potentially infinite time-horizons -- (1) bridging the domain gap from training on finite length and extrapolating to infinite horizons, (2) sustaining high quality by preventing error accumulation, and (3) maintaining fast inference, without incurring growth in computational cost due to increasing context windows. A key to our approach is introducing carefully designed sliding-window causal attention, combined with attention sinks. By incorporating self-rollout with attention sinks and KV cache rolling during training, we properly simulate inference-time extrapolations with a fixed context window, enabling constant-speed generation of arbitrarily long videos. Our models achieve state-of-the-art results in motion following and video quality while being two orders of magnitude faster, uniquely enabling infinite-length streaming. With MotionStream, users can paint trajectories, control cameras, or transfer motion, and see results unfold in real-time, delivering a truly interactive experience.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents MotionStream, a real-time motion-controlled video generation system that achieves sub-second latency at 24 FPS on a single GPU, representing a two-order-magnitude speedup over existing methods. The approach distills a bidirectional motion-conditioned text-to-video teacher model into a causal student model using Self Forcing with distribution matching loss, enabling streaming inference. To address the challenges of infinite-horizon generation, the system introduces sliding window causal attention combined with attention sinks and KV cache rolling during training, which allows it to generate arbitrarily long videos at constant speed while maintaining quality and preventing error accumulation. This enables truly interactive experiences where users can control videos through trajectory painting, camera movements, or motion transfer and see results unfold in real-time, achieving state-of-the-art results in both motion following and video quality.

### Strengths
1. This paper achieves sub-second latency for motion-controlled video generation at 24 FPS on a single GPU, representing a two-order-magnitude speedup over existing methods through causal distillation of a bidirectional teacher model using Self Forcing with distribution matching loss.

2. This paper enables infinite-length video generation at constant speed by introducing sliding window causal attention with attention sinks and KV cache rolling, which prevents quality degradation and error accumulation while maintaining fixed computational costs regardless of video length.

3. This paper delivers truly interactive motion control experiences by allowing users to paint trajectories, control cameras, or transfer motion with real-time visual feedback, achieving state-of-the-art results in both motion following and video quality.

### Weaknesses
1. The technical contributions are somewhat limited. The proposed MotionStream seems to be a straightforward combination of existing full-attention controllable video generation with the self forcing framework.
2. While the paper claims to enable long video generation, only a single long-form demonstration is provided; the authors should present more extensive examples to substantiate the robustness and consistency of their approach across diverse scenarios.

### Questions
1. The authors acknowledge that fixed attention sinks limit the model's ability to handle complete scene changes, favoring initial scene preservation; they should provide concrete visual examples demonstrating this failure case to better illustrate when and how the method breaks down under drastic environmental transitions.
2. In Figure 5, when conducting the guidance ablation study, the authors use different guidance scales for the same conditioning across experiments, such as w_m=1.5 in Hybrid versus w_m=5.0 in Motion Only; could the observed performance differences be primarily attributed to these inconsistent hyperparameter settings rather than the conditioning strategies themselves?

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
This paper presents MotionStream, a real-time, motion-controlled video generator that achieves sub-second latency and up to 24 FPS on a single GPU, enabling interactive dragging, motion transfer, and camera control with effectively unbounded video length. The model conditions on sparse 2D tracks using lightweight sinusoidal embeddings and a learnable track head, and employs joint classifier-free guidance to balance strict trajectory adherence with text-prompted natural dynamics. Long-horizon stability is maintained via sliding-window causal attention supplemented by attention sinks and rolling KV caches, which anchor early frames while keeping computation constant. The authors substantiate the approach with extensive comparisons, ablations, and a user study, demonstrating clear advantages for efficient motion-controlled video generation.

### Strengths
1. The manuscript’s motivation is clear and compelling, and the proposed solution appears technically sound and promising.
2. The authors provide thorough comparative evaluations and ablation studies that isolate the contributions of each component.
3. The presented demos are persuasive, showcasing the method’s practical impact.

### Weaknesses
1. The manuscript now reads like a composition of prior work—self-forcing, attention sinks, and motion conditioned video generation—assembled to achieve real-time, controllable video generation. However, specific contributions are not clearly presented.
2. For motion conditioning, the method employs channel-wise concatenated embeddings and fine-tunes pretrained image-to-video models. While computationally efficient, this design may degrade the pretrained model’s general generative capabilities. The manuscript lacks analysis or experiments that quantify the trade-offs introduced by this specialization.
3. The manuscript does not include comparisons of general image-to-video quality with the original pretrained backbone. Such an evaluation would provide a more complete picture of the method’s impact on baseline capabilities.
4. The evaluation omits challenging corner cases—such as abrupt, large-amplitude motion changes or trajectories that poorly match the source subject. Moreover, because the approach combats long-horizon self-forcing with attention sinks and a sliding-window scheme tied to the teacher’s training horizon (e.g., position embeddings capped at ~81 frames), the model may be constrained in realizing large or sudden motion transitions. The authors should explicitly test and discuss these scenarios to clarify whether the architectural choices impose limits on motion magnitude or adaptability.

### Questions
1. In the guidance ablation, why do the text-only generations still exhibit a clear motion trajectory in the visualizations? Is this because the training data contain many similar side-to-side sway patterns, or is there another cause?
2. In Table 1, models with more parameters perform worse than smaller ones. The manuscript does not explain this.

### Soundness
3

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
4

### Summary
This paper introduces *MotionStream*, a real-time video generation framework with interactive motion control. The authors distill a bidirectional diffusion teacher into a causal autoregressive model, achieving streaming generation up to 24 FPS on a single GPU. The paper further proposes attention sinks and rolling KV caches to enable stable long-sequence generation without increasing latency. Experiments demonstrate competitive visual quality and strong speedups across several motion-controlled generation tasks.

### Strengths
1. Addresses a timely and important problem: bringing real-time interaction to diffusion-based video models.  

2. The proposed attention sink and KV cache mechanisms are intuitively sound and practically effective.  

3. Extensive experiments and ablations support the framework's stability and efficiency.  

4. Overall writing and presentation are polished and easy to follow.

### Weaknesses
1. The evaluation datasets appear limited to short human-action or camera-move clips; it's unclear how the model performs on scenes with large physical transformations (e.g., object deformation, rapid perspective change). 

2. Limited theoretical analysis; most findings are empirical.

### Questions
See  Weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method named Motionstream, which helps interactive video  generation with flexible motion control and efficient inference speed. Specifically, Motionstream trains a teacher video generation model with both text and motion control and Motionstream distills the teacher with Diffusion strategy to the student with auto-regressive strategy. The student with auto-regressive strategy can generate the interactive video at constant high speed.  To further bridging the domain gap from training on finite length and extrapolate, and preventing error accumulations, MotionStream incorporates self-rollout with attention sinks and KV cache rolling during training. Finally, MotionStream is evaluated from respective of the motion following, video quality and inference efficiency.

### Strengths
1. Complete "Teacher-Student" Framework: The entire pipeline, from designing an efficient motion-controlled teacher model to distilling a causal student model, is rigorously and systematically designed.
2. Lightweight Track Encoder: The use of sinusoidal positional encoding with a learnable track head, compared to the VAE-based RGB encoding method, achieves a 40× speedup in encoding while maintaining high track adherence accuracy, which is crucial for real-time systems.
3. Attention Sinking with Rolling KV Cache: This is one of the core innovations of the paper. The authors adapt the "attention sink" concept from LLMs and successfully apply it to video diffusion models. By using the initial frame as a fixed "anchor" combined with a local rolling context window, it effectively solves the drift problem in long-video generation, ensuring long-term stability. The paper explicitly highlights that its method eliminates the train-test discrepancy by simulating inference-time extrapolation during training, which offers an advantage over contemporaneous work.

### Weaknesses
1. It is worth noting that utilizing distillation to enhance model inference efficiency is an established technique in model optimization, including the video generation domain. Several contemporaneous works, such as Hunyuan-Gamecraft and notably Matrix-Game 2.0, have adopted a highly similar paradigm—combining a distillation framework with an autoregressive student model and Self-Forcing training. While the integration presented here is non-trivial, the core methodological concept bears a strong resemblance to these existing approaches, which consequently lowers the perceived novelty of the contribution
2. Furthermore, while the paper highlights motion control as a key contribution alongside the distillation framework, the synergistic relationship between the two remains unclear. They are presented as somewhat decoupled components. The authors should articulate a more cohesive narrative that explicitly links the design of the motion control mechanism to the requirements and success of the causal, distilled student model.

### Questions
I find the overall approach of MotionStream to be very similar to that of Matrix-Game 2.0, which leads me to question the novelty of this work's contribution.

### Soundness
2

### Presentation
2

### Contribution
2
