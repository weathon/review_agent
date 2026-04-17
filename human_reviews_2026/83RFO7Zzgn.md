# Think Proprioceptively: Compact Subgoal Traces for Vision-Language-Action Model

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Vision-language-action (VLA) models translate visual observations and language instructions to robot actions, yet current architectures regard proprioception as a passive input rather than an active reasoning component. Without proprioceptive guidance, VLA models process multimodal features in isolation from the robot’s physical configuration, and hierarchical approaches often encode subgoals in high-dimensional visual or textual spaces that are ungrounded in the robot’s embodiment. We present SubgoalVLA, a framework built on the \textit{think proprioceptively} paradigm that redefines how multimodal information is processed. SubgoalVLA leverages proprioception in two ways. First, proprioceptive states serve as cross-attention queries to select vision-language features, enabling configuration-aware feature extraction. Second, subgoals are encoded as compact sequences of joint configurations that eliminate the need for cross-modal translation. Through a two-stage training protocol that begins with supervised learning on ground-truth subgoals and then fine-tunes with self-predicted subgoals, we mitigate distribution shift between training and inference. On the CALVIN benchmark, SubgoalVLA achieves state-of-the-art performance with an average task completion length of 3.32, demonstrating that proprioceptive reasoning provides the critical bridge between high-level task understanding and embodied control.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors discuss recent VLA models, categorizing them by how they adapt a pretrained VLM to output robot actions (autoregressively, via diffusion, in the model vs a separate trained expert, etc.). Within this, the authors propose another VLA variation.

Assume we have a robot trajectory. Typically, a VLA model is trained to predict the actions $a$ in that trajectory from visual input (cameras) and a natural language instruction. But, we may also have proprioceptive information within our robot data. Prior works have included this in the context, or added it as contextual information for an action expert (either concatenated or added via cross attention.) This paper's proposal, SubgoalVLA, continues to incorporate proprio information this way, but makes some small variations on how the proprio information is incorporated into the MHA, as well as including an intermediate subgoal prediction head.

For the proprio information, the cross attention mechanism involves 2 residual connections for just the projected $Q_{proprio}$ info: once right after the MHA, and once right after the MLP, to make as sure as possible that the proprio information is preserved through the attn layers.

For the subgoal prediction head, we treat this as a higher-level diffusion policy. Given a ground truth trajectory we pick out intermediate robot positions as subgoals. (I believe this is $M$ even choices over the entire trajectory, but am not sure.) The subgoal head is trained to predict these subgoals via diffusion. The action head is then modified to condition subgoals, proprio, and visual-language features (as mediated by the cross-attn). At training time, the true golden subgoals are used for the first stage of training, then we switch to training against predicted subgoals in the second stage.

The authors argue that these proprioceptive based subgoals are lower dimensionality, more interpretable than latent visual embeddings, and improve robot success rates.

### Strengths
Proprioceptive information is a modality of robot data that does not appear elsewhere, which can make it especially high leverage to consider. The authors have done a variety of ablations studying the architecture design decisions they have made, and the diagrams described the modified cross-attention mechanism is helpful for understanding the paper.

### Weaknesses
I believe the authors' changes are ones that are unusually well suited to simulated environments, and given that all experiments are only in sim, I'm unsure on the effectiveness of the method in general. Sim proprio information is inherently perfect, making it especially useful to condition on. Real robot proprio is not necessarily as nice to use. Although the model is trained with noised subgoals, to me it is unclear if this is enough to make intermediate subgoal prediction a good idea.

Predicting subgoals feels a bit like a hack, since in theory, once could modify the action expert to predict the entire trajectory from current timestep to end, which would include the subgoals by definition. In practice, we do not do this due to inference time concerns, but this approach feels like one which would quickly become obsolete in a world with more compute.

Predicting subgoals via diffusion before doing action chunk diffusing would also increase inference time right? Given that actions need to wait for subgoals to be denoised before we can start that stage of denoising? Would like clarification on this, especially because this is another potential blocker for real robot usefulness.

Adding more subgoal prediction may affect generalization of the VLA. If we assume that VLA's perform best in-distribution and degrade as they are forced further from their data manifold, then subgoal prediction increases how tightly VLA performance is bound to the data manifold. By adding another prediction stage, we are increasing the surface of how OOD predictions can hurt us (i.e. if the subgoal prediction is poor then the action chunk predictions will also become off). It would be interesting to see how the model handles out of distribution subgoals, but I don't think this is tested. It would also be interesting to see how the model handles changes during the trajectory (i.e. how much the model retains closed-loop behavior compared to just getting better at open-loop prediction). Again this is not studied.

In general the baselines are a bit old - no OpenVLA, no SpatialVLA, no Octo, no Magma-8B, no pi_0 etc.

### Questions
Is inference time affected by adding the subgoal head before it predicts the action head?

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
2

### Summary
This paper proposes SubgoalVLA, a vision–language–action (VLA) model that explicitly leverages proprioceptive information as an active reasoning signal. The method introduces (1) proprioceptive cross-attention, where joint states query visual-language features to yield configuration-aware perception, and (2) compact proprioceptive subgoal traces, which represent intermediate goals directly in the robot’s joint space rather than as images or text. A two-stage training strategy mitigates the train–test distribution gap between ground-truth and predicted subgoals. Experiments on the CALVIN long-horizon benchmark show clear improvements over several strong baselines, especially on multi-task and long-horizon settings.

### Strengths
1. Treating proprioception as an active reasoning signal rather than a passive input is conceptually elegant and aligns well with embodied intelligence principles.

2. Representing intermediate goals directly in joint space avoids cross-modal translation and significantly reduces dimensionality, which is both efficient and biologically inspired.

3. Results on CALVIN LH-5 and ablations clearly demonstrate that both proposed components (cross-attention + subgoal traces) are essential. The gains are consistent across multiple metrics.

### Weaknesses
1. All experiments are simulation-based; it remains unclear how proprioceptive traces generalize to noisy or delayed sensor feedback on real robots.

2. The method’s reliance on accurate proprioceptive signals may restrict applicability to robots with different morphologies or partially observed states.

3. The model combines a vision–language backbone, diffusion-based planners, and multiple transformers. Runtime or resource requirements are not discussed.

### Questions
No

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
4

### Summary
The paper proposes SubgoalVLA, which claims to use proprios both for cross-attention to select VL representations, also served as a middle stage predict objective. Finally, the SubgoalVLA conditioned both VL representtaions and predicted subgoal trace to predict the action.

### Strengths
1. The paper is easy to read and follow.

2. The paper gives very clear method level comparison of different ways the current VLA or ACT use proprios for, compared with the previous ones, which maybe condition on proprios use a learn encoder or not condition on proprios, it is kind of novel the paper propose to use proprios both to select VL latent and serve as a middle prediction subgoal.

### Weaknesses
1. I am confused about the deisgn. It seems the second stage, the user train a DiT to pedict the trace, which the trace is the chunk of future proprios, so what is the exact difference of the paper's seond stage and third stage, the subgoal trace is basically the same stuff of ACT (predicted goal in third stage), in that way, i think the design is weird, it is kind of repetitve because the tuitive is if the model can learn to predict an accurate subgoal trace in the second stage, it will predict good results for future action, that (maybe conceptual different a little bit), but in the implementation, the trace is the same stuff.

2. The concern in 1 seems further verified in Table 4, seems not a big difference of the abaltion between these two modules/stages, which the increase is very small, and as we know the robotic evaulation itself is with high variance, so the conclusion that the "Subgoal design works better" is not convincing.

### Questions
See the above weakness. Besides those, I have further concern that:

1. I might need the author both intuitively and systematicall calrify the motivation of the design of the subgoal module, which now confuses me a lot, since i feel technically it is the same information of the DiT Action chunk model.
2. The paper does not have through experiments to support the conclusion, especially, no real robot experiment, and only have CALVIN simulation experiments.

### Soundness
2

### Presentation
2

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
The paper presents SubgoalVLA, a vision-language-action framework that introduces a “think proprioceptively” paradigm for robotic manipulation, treating the robot’s proprioceptive state (joint configuration and motion state) as an active reasoning component rather than a passive input. SubgoalVLA contributes two key designs: (1) proprioception-driven cross-attention, which uses the current kinematic state as queries to select configuration-aware features from vision-language representations; and (2) subgoal traces, compact sequences of joint configurations over time that encode motion dynamics and eliminate the need for high-dimensional visual or textual subgoal representations. The framework adopts a two-stage training protocol: supervised learning on ground-truth subgoals followed by fine-tuning on self-predicted subgoals to mitigate train–test distribution shift. On the CALVIN benchmark, it achieves state-of-the-art performance compared with other methods evaluated in the experiments.

### Strengths
Originality: Using proprioceptive states as cross-attention queries for configuration-aware feature selection, combined with joint sequence subgoals, provides a compact hierarchical representation that avoids costly cross-modal translation from high-dimensional visual/textual subgoals.

Quality: The two-stage training protocol offers a reasonable engineering solution to address distribution shift; the reported performance improvements on CALVIN tasks demonstrate measurable gains.

Clarity: The architecture, temporal chunking, and training pipeline are clearly described, with Figure 1's architectural comparison effectively highlighting the methodological differences from existing approaches.

Significance: Highlights "think proprioceptively" as a complementary direction to "think visually/textually" paradigms, potentially reducing inference latency and improving executability, with promising prospects for extension to real-world systems.

### Weaknesses
Limited Experimental Scope: Evaluation is restricted to the CALVIN benchmark only, which is insufficient for embodied AI applications. The absence of real robot experiments and other standard benchmarks (RLBench, Meta-World) makes the experimental validation inadequate. Additionally, the paper lacks comparisons with recent hierarchical methods (e.g., π0.5, CoT-VLA) that are directly relevant to the proposed approach.

Insufficient Ablation Studies: The ablation analysis is incomplete, lacking systematic study of cross-attention layer selection (only k=m/2 tested), missing analysis of subgoal trace length M, and no ablation of the two-stage training protocol. These are critical design choices that require proper validation.

Unclear Methodological Boundaries: The distinction between the proposed approach and existing subgoal-based dual-system VLA paradigms is not sufficiently clear, making it difficult to assess the true novelty and contribution of the work.

### Questions
How does SubgoalVLA perform on other manipulation benchmarks? Can you provide results on RLBench or real robot experiments to demonstrate broader applicability beyond simulation?
Why is k=m/2 optimal for feature extraction? Have you experimented with other intermediate layers or adaptive layer selection strategies? What is the sensitivity of performance to this architectural choice?
Can you provide direct comparisons with recent hierarchical VLA approaches (π0.5, CoT-VLA, DiffusionVLA) that also employ intermediate representations? How does your method compare in terms of both performance and computational efficiency?
Under what circumstances does the proprioceptive approach fail? Are there specific task categories where visual or textual subgoals would be more appropriate than joint-space representations?
How much demonstration data is required for effective training? How does performance scale with dataset size compared to end-to-end methods, and what are the sample complexity trade-offs?

### Soundness
2

### Presentation
3

### Contribution
2
