# MoTVLA: A Vision-Language-Action Model with Unified Fast-Slow Reasoning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Integrating visual-language instructions into visuomotor policies is gaining momentum in robot learning for enhancing open-world generalization. Despite promising advances, existing approaches face two challenges: limited language steerability when no generated reasoning is used as a condition, or significant inference latency when reasoning is incorporated. In this work, we introduce MoTVLA, a mixture-of-transformers (MoT)–based vision–language–action (VLA) model that integrates fast–slow unified reasoning with behavior policy learning. MoTVLA preserves the general intelligence of pre-trained VLMs (serving as the generalist) for tasks such as perception, scene understanding, and semantic planning, while incorporating a domain expert, a second transformer that shares knowledge with the pretrained VLM, to generate fast domain-specific reasoning (e.g., robot motion decomposition), thereby improving policy execution efficiency. By conditioning the action expert on decomposed motion instructions, MoTVLA can learn diverse behaviors and substantially improve language steerability. Extensive evaluations across natural language processing benchmarks, robotic simulation environments, and real-world experiments confirm the superiority of MoTVLA in both language reasoning and manipulation task performance. We refer to https://motvla.github.io/MoTVLA-website/ for the demonstration videos and corresponding descriptions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a MoT based VLA model to address the challenging gap between reasoning lantency and language sterrability. It conducts VQA and Manipulation experiments to verify its model's capability.

### Strengths
The writing is clear, and the paper is exceptionally well organized. The figures are thoughtfully designed, with harmonious colors and a clean, balanced layout.

### Weaknesses
I doubt the necessity of introducing such large models (generalist and domain expert) merely to perform manipulation tasks like pick and place. As I understand it, the current work aims to achieve an integration of high-level reasoning and low-level control, but with the introduction of such a large number of parameters, I would expect to see tasks that cannot be accomplished by basic VLAs, in order to truly highlight the role of the generalist—rather than just demonstrating VQA-like capabilities, which are useless in actual manipulation. I believe the following types of tasks could better reflect the value of MoT based design, such as long-horizon tasks (where the “slow system” handles instruction decomposition and subtask transition) or zero-shot generalization to reflect the slow system reasoning ability.

### Questions
1. As I understand it, the generalist and domain expert modules are connected through a shared latent feature space, rather than by having the generalist’s output directly feed into the domain expert. Moreover, the generalist itself is not trained—it mainly serves to provide original pretrained vision-language model (VLM) features. Is my understanding correct?

2. In Table 1, there is no comparison with existing baselines, so I cannot assess how strong the VQA capability actually is.

3. In Table 3, I noticed that the domain expert runs at 9 Hz, while the generalist runs at only 2 Hz. My understanding is that the domain expert has roughly the same number of parameters as the generalist, plus an additional action expert module. Is the frequency difference due to the computation involving action chunks during inference?

### Soundness
3

### Presentation
4

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
The paper proposes MoTVLA, a mixture-of-transformers vision–language–action model that unifies fast and slow reasoning to improve both language steerability and real-time performance in robot control. It retains a pretrained VLM “generalist” for slow, autoregressive reasoning (perception, semantic planning) and introduces a shared-knowledge “domain expert” that performs fast, token-wise motion decomposition to condition a diffusion-based action expert (DiT) for continuous control. Experiments across NLP/VQA benchmarks, ManiSkill simulations, and real robots show superior reasoning metrics and higher task success rates versus strong baselines (e.g., π0/π0.5 KI, DP), with markedly lower reasoning latency. Ablations confirm that both the pretrained generalist and the fast domain expert are critical for stable reasoning and policy success.

### Strengths
- Originality: The unified fast–slow reasoning architecture via a Mixture-of-Transformers is a creative synthesis that removes a key limitation in prior VLA systems—either poor steerability without explicit reasoning or high latency with autoregressive CoT—by sharing global attention between a pretrained generalist and a token-wise domain expert. The decomposition–composition–decomposition design and the use of fast motion decomposition to condition diffusion policies is a novel and pragmatic formulation.

- Quality: The paper presents a thorough empirical evaluation across complementary fronts (robotic simulation, real-world manipulation, and general VQA), with strong, consistent gains over strong baselines (π0/π0.5 KI, DP, GR-MG) and clear latency advantages. 

- Clarity and Significance: The model components and inference pipeline are clearly described, supported by pseudo-code, figures, and metric choices that align with the intended behaviors.

### Weaknesses
- Switching between the slow and fast models is not yet autonomous. The generalist is used only for optional high-level reasoning or dialogue at the outset. During execution, it can be activated only by the operator. 
- Ablations are conducted on a simple cube-stacking task, where explicit reasoning appears less necessary.

### Questions
- Does the model exhibit error-recovery behavior? If so, is additional annotation required?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
MoTVLA proposes a Mixture-of-Transformers vision-language-action model that unifies general and robotic domain reasoning with behavior policy learning to balance language steerability and inference latency. The model is trained via SFT for the domain expert and diffusion policy learning for the action expert, and is evaluated on NLP benchmarks, ManiSkill simulations, and real-world manipulation.

### Strengths
S1. Using bidirectional token-wise reasoning can effectively accelerate the inference process.

S2. This paper conducts benchmarks on both general-domain and robotic-domain reasoning to evaluate their respective performances.

### Weaknesses
W1. **How are slow reasoning and fast reasoning defined?** I don’t find it convincing that general-domain reasoning is inherently slow while robotic-domain reasoning is fast. Moreover, reasoning in the robotic domain is more crucial for manipulation. Wouldn’t adopting token-wise prediction potentially lead to insufficient reasoning capability?

W2. **The paper lacks both motivation and ablation studies.** It does not verify why and how general and robotic-domain reasoning improve manipulation performance. Which aspect of manipulation is enhanced (e.g., fine-grained control, long-horizon planning, or generalization)? How can the authors justify that preserving science or other general reasoning abilities helps manipulation? This contradicts common understanding in robotics, where retaining such reasoning only increases parameter redundancy and inference latency for real-time control systems. Finally, during policy inference, the DiT head relies solely on fast reasoning features (plus visual observations, etc.) and does not use outputs from slow reasoning as conditions.

W3. **What is the real innovation of the MoT design compared with Bagel (Deng et al., 2025a)?** Is it merely distinguishing general-domain and robotic-domain datasets based on Bagel’s MoT structure and then adding a DiT policy head? Furthermore, the use of bidirectional token-wise reasoning is not motivated by characteristics of robotic tasks.

W4. As a robotic VLA paper, the model’s inference speed should be compared with more VLA baselines such as Fast [1], Pi_0, and CogACT [2]. Although these models do not explicitly generate language, the latent features provided by their LLMs also serve as high-level conditions for action generation.
[1] Fast: Efficient Action Tokenization for Vision-Language-Action Models
[2] CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation

### Questions
Q. I think the authors need to clarify the innovation of the MoT design and the overall method design, as well as clearly explain the motivation of the paper, e.g., why and how general-domain reasoning can benefit robotic manipulation, and what exact aspects of manipulation it is expected to improve.

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
3

### Summary
In this paper, the authors address two key challenges in existing Vision-Language-Action models for robot learning: limited language steerability without reasoning and high inference latency with reasoning. They propose MoTVLA, a MoT-based VLA model integrating unified fast-slow reasoning with behavior policy learning. MoTVLA includes a generalist, a domain expert, and an action expert. Its effectiveness is validated via NLP benchmarks, robotic simulations, and real-world tasks, outperforming SOTA baselines.

### Strengths
1. This paper proposes a novel MoT-based unified fast-slow reasoning framework, addressing the critical trade-off between language steerability and inference latency in existing VLA models. The "generalist-domain expert" design with knowledge sharing is conceptually sound, filling a gap in current VLA research.

2. To ensure both general intelligence retention and policy efficiency, the authors adopt a "decomposition-composition-decomposition" p
ipeline for reasoning and a two-stage training recipe. This methodological rigor lays a solid foundation for MoTVLA’s performance.

3. Extensive experiments across simulation, real-world, and reasoning tasks corroborate MoTVLA’s superiority in reasoning accuracy, manipulation success rate, and inference latency, providing strong empirical support.

### Weaknesses
1. The real-world dataset used for training the action expert is excessively small and relies heavily on manual annotation, which poses significant challenges for data scaling. This not only increases the risk of overfitting but also restricts the model to a narrow range of action skills (e.g., simple pick-and-place, stacking, and pulling tasks). Consequently, the model’s reliability and scalability in diverse real-world scenarios are compromised.

2. The results of reasoning tasks (Section 4.2) lack comparative analyses with SOTA VLMs, making it difficult to accurately assess the competitive advantage of MoTVLA in terms of reasoning capabilities.

3. The paper focuses exclusively on simple short-horizon tasks (e.g., cube stacking) and lacks evaluations on complex long-horizon tasks (with only one qualitative result provided) or tasks involving multi-arm systems or cross-embodiment scenarios. This limitation undermines the generalizability of the MoTVLA framework.

### Questions
1. Table 4 contains a mislabeling issue: its title claims to present "dataset composition," while the actual content displays "reasoning latency" data, which may cause confusion for readers. Will the authors correct the label of Table 4 to align with its content?

2. Could the authors add comparative analyses of slow reasoning performance between MoTVLA and SOTA VLMs (e.g., Qwen2.5-VL)? Additionally, would it be possible to supplement quantitative results comparing MoTVLA with baselines on multi-stage (semi-long-horizon) tasks?

3. How does MoTVLA handle unseen skills or complex multi-step instructions in open-world scenarios? Could the authors provide quantitative or qualitative results to demonstrate the model’s performance in such cases?

4. There is an inconsistency in the description of motion merging: the appendix mentions merging the "picking up" and "moving toward the destination" motions, while Figure 4 illustrates the merging of "picking up" and "placing into" motions. Could the authors explain the reason for this discrepancy, clarify the criteria used to determine which motions to merge, and elaborate on how consistency is maintained throughout the motion decomposition process?

5. The paper lacks detailed descriptions of the random states and seeds employed in the experiments. Could the authors supplement this information, including the specific values of the random seeds used and the methods by which initial object poses and robot initial states were randomized?

6. To fully validate the language steerability of MoTVLA, could the authors add verification experiments where the model is instructed to grasp different objects via varying language prompts in the same scene? For example, in a scene containing eggplants, carrots, and corn, the model should perform pick-and-place operations on the target object specified by each distinct instruction.

7. Regarding the identified limitation that MoTVLA exhibits "strong reasoning ability but insufficient execution capability," have the authors attempted to transfer the decomposed sub-instructions (generated in the second stage) to other VLA policies (e.g., π0.5 KI) for execution? If such attempts have been made, what results were obtained? If not, what are the reasons for not pursuing this approach?

### Soundness
2

### Presentation
3

### Contribution
3
