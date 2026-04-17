# End-to-end Listen, Look, Speak and Act

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Human interaction is inherently multimodal and full-duplex: we listen while watching, speak while acting, and fluidly adapt to turn-taking and interruptions. Realizing these capabilities is essential for building models simulating humans. We present ELLSA (End-to-end Listen, Look, Speak and Act), which, to our knowledge, is the first full-duplex, end-to-end model that simultaneously perceives and generates across vision, text, speech, and action within a single architecture, enabling interaction patterns previously out of reach, yielding more natural, human-like behaviors. At its core is a novel SA-MoE architecture (Self-Attention Mixture-of-Experts) that routes each modality to specialized experts and fuses them through a unified attention backbone. This provides a generalizable solution for joint multimodal perception and concurrent generation, leveraging strong pre-trained components while enabling efficient modality integration and mitigating modality interference. On speech-interaction and robot-manipulation benchmarks, ELLSA matches modality-specific baselines, while uniquely supporting advanced multimodal and full-duplex behaviors such as dialogue and action turn-taking, defective instruction rejection, speaking-while-acting, context-grounded visual question answering, and action barge-ins. We contend that ELLSA represents a step toward more natural and general interactive intelligence, contributing to the broader pursuit of artificial general intelligence. A demonstration is available at https://anonymous.4open.science/r/LLSA-E821.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose ELLSA, a unified, full-duplex multimodal model capable of simultaneous perception and generation across speech, vision, text, and action modalities. Specifically, the model proposes SA-MoE that routes inputs to modality-specific experts and fuses them through shared attention. Multimodal streams are partitioned into 1-second segments to support full-duplex interaction. The experiments show that ELLSA achieves performance comparable to or surpassing modality-specific baselines on multiple benchmarks covering speech interaction, speech-conditioned robot manipulation, and multimodal full-duplex tasks.

### Strengths
- The paper makes the first attempt to explore end-to-end full-duplex multimodal interaction, pointing to a promising new direction.
- The paper conduct experiments on benmarks of various task type and demonstrate competitive performance.

### Weaknesses
- The novelty of SA-MoE is limited, as prior works such as EVE-v2 [1] and Mono-InternVL [2] already adopt modality-specific MoE structures.
- The ablation study is insufficient and could be expanded to better demonstrate the contributions of individual components.
- The discussion of computational efficiency and real-time constraints is limited and should be addressed more thoroughly.
- The experiments on VLA tasks are limited, lacking experiments on more benchmarks and real world experiments.

References
[1] Diao, Haiwen, et al. “EVE-v2: Improved baselines for encoder-free vision-language models.” arXiv preprint arXiv:2502.06788 (2025).
[2] Luo, Gen, et al. “Mono-InternVL: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training.” Proceedings of the Computer Vision and Pattern Recognition Conference, 2025.

### Questions
1. The paper would benefit from clarifying the novelty of SA-MoE compared to existing methods such as EVE-v2 or Mono-InternVL.
2. I encourage the authors to include additional ablation studies, for example:
    1. Evaluate the impact of time block duration on both performance and latency.
    2. Investigate whether increasing the number of experts—by splitting vision and action modalities into separate experts—would further improve overall performence , while the action model is currently responsible for processing both vision and action inputs.
3. Given the full-duplex streaming interaction setup, it is important to analyze the response latency of the entire system.
4. The VLA evaluation is currently limited to the LIBERO dataset, and the author should evaluate the model on additional benchmarks and real world experiments to better assess the model’s generalizability.

### Soundness
3

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
3

### Summary
This paper introduces ELLSA (End-to-end Listen, Look, Speak and Act), a novel end-to-end, full-duplex model. The motivation is to bridge the gap between disembodied conversational agents ("talkers") and non-conversant robotic agents ("doers") by creating a unified system that can simultaneously perceive multimodal inputs (vision, speech) and generate multimodal outputs (speech, actions). The core technical contribution is the Self-Attention Mixture-of-Experts (SA-MoE) architecture. This design routes different modalities to specialized, pre-trained expert models (a speech expert for speech/text and an action expert for vision/action) and integrates them through a shared self-attention mechanism. The authors shows ELLSA's capabilities on a wide range of tasks. It achieves competitive or superior performance on standard benchmarks for speech interaction and speech-conditioned robot manipulation.

### Strengths
- The paper addresses a fundamental and ambitious goal in AI: creating truly interactive embodied agents that can communicate and act in a fluid, human-like manner. 

- The proposed SA-MoE architecture is a well-motivated to the problem of modality interference, a common challenge in multimodal model development. The ablation study comparing SA-MoE to a dense model (Table 7) provides compelling evidence of its superiority in performance.

- The paper shows its evaluation of the unique capabilities enabled by its full-duplex, multimodal design. The successful demonstration of action turn-taking, barge-ins, and speaking-while-acting (Tables 3 and Table 4) directly validates the central claims of the paper and showcases interaction patterns that are qualitatively more natural and complex than what previous systems could achieve.

### Weaknesses
- The results for "speaking-while-acting" (Table 4) show a noticeable drop in performance for both the manipulation and speech interaction tasks compared to their single-task counterparts (Tables 1 & 2). The authors note that the model "may be distracted," which is an honest assessment. This performance hit raises questions about the scalability of this approach to more complex, simultaneous tasks and the true cost of concurrent generation.

- The system relies on distinct modality-specific components and different backbones. It will be better with a more unified architecture which  could enable deeper cross-modal fusion and represent a more fundamental approach to generalized interactive AI.

### Questions
- Could you elaborate on the performance degradation observed during the speaking-while-acting task? Is this an issue of model capacity (i.e., would a larger backbone help?), an architectural limitation of SA-MoE where attention is being split, or a data-related artifact?

- The concept of full-duplex interaction is intrinsically tied to low latency.  Could you provide any measurements for key interaction loops, such as speech-to-action or speech-to-speech-response times?

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
4

### Summary
This paper introduces ELLSA, an end-to-end, full-duplex model that seamlessly integrates listening, looking, speaking, and acting. Its novel core is the SA-MoE architecture, which routes modality-specific experts through a shared attention backbone to mitigate interference and enhance fusion. Evaluated on speech QA, robot manipulation, and VQA, ELLSA not only achieves strong performance but also demonstrates unique capabilities like speaking-while-acting and action barge-in.

### Strengths
1. This work presents a novel and practically significant real-time, full-duplex system that seamlessly integrates the speech, vision, and action modalities. 
2. The proposed SA-MoE architecture offers a clean and modular solution to the challenge of modality interference by routing modalities to specialized components. 
3. The system demonstrates remarkable versatility by tackling a diverse range of tasks from dialogue to manipulation.

### Weaknesses
1. The SA-MoE architecture is primarily evaluated empirically; new conceptual insight and theoretical analysis of its properties are lacking. 
2. The study does not investigate the model's ability to generalize to unseen tasks or domains, no evidence of zero-shot or cross-task generalization is fully shown. 
3. The broader implications for AGI remain speculative, as the model's capabilities are confined to the specific multimodal tasks presented. This seems to be an overstated narrative about AGI.

### Questions
1. Please explain the specific mechanism through which SA-MoE's cross-modal attention prevents interference. Is there quantitative evidence to support this?
2. What is the impact of the 1-second block size on real-time performance and the quality of the model's responses in a duplex setting?
3. Has the model been evaluated on unseen task or modality combinations to rigorously test its generalization capabilities?

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
This paper presents ELLSA (End-to-end Listen, Look, Speak and Act), claiming to be the first full-duplex, end-to-end model that simultaneously processes vision, speech, text, and action within a unified architecture. The core technical contribution is SA-MoE (Self-Attention Mixture-of-Experts), which routes modalities to specialized experts (Speech Expert and Action Expert) while enabling cross-modal fusion through shared attention. The model is trained in three stages and evaluated on speech interaction benchmarks and robotic manipulation tasks (LIBERO), demonstrating both competitive basic performance and novel full-duplex capabilities.

### Strengths
- This paper addresses full-duplex multimodal interaction for embodied AI is timely and important. The motivation for combining conversational abilities with physical action is well-articulated.

- SA-MoE is an elegant solution that leverages pretrained experts while mitigating modality interference. 

- The introduction of context-grounded VQA during manipulation and systematic evaluation of full-duplex behaviors adds value beyond existing benchmarks.

### Weaknesses
- This paper omit discussion of prior work inlcuding: 
   - Unified-IO 2 (Lu et al., 2023) which is also native support vision, laugnage, audio and action. The authors need to clearly differentiate from Unified-IO 2, likely arguing that their full-duplex streaming capability is the key difference.
  - Gato (Reed et al., 2022, DeepMind) which A generalist agent. 
  - PaLM-E: Embodied multimodal model combining vision-language understanding and robotic manipulation. 

- Limited Technical Novelty of SA-MoE: While effective, SA-MoE is essentially routing + shared attention between existing pretrained models.

- Speaking-while-acting shows notable performance drops (Table 4), especially on harder tasks (LIBERO LONG: 73.2% vs 84.4%)

### Questions
- Why specifically 2 experts rather than 4 (one per modality)? Have you experimented with different expert configurations? What is the computational overhead of SA-MoE vs a single model

- Can the model handle truly simultaneous inputs (e.g., receiving new speech while still processing previous speech and executing action)? The 1-second time block seems to discretize the interaction.

- What are the main challenges anticipated for real-world deployment? Have you done any preliminary real robot experiments?

### Soundness
2

### Presentation
2

### Contribution
2
