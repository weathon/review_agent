# ViPRA: Video Prediction for Robot Actions

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6, 6

## Abstract
Can we turn a video prediction model into a robot policy? Videos, including those of humans or teleoperated robots, capture rich physical interactions. However, most of them lack labeled actions, which limits their use in robot learning. We present *Video Prediction for Robot Actions* (**ViPRA**), a simple pretraining-finetuning framework that learns continuous robot control from these actionless videos. Instead of directly predicting actions, we train a video-language model to predict *both future visual observations and motion-centric latent actions*, which serve as intermediate representations of scene dynamics. We train these latent actions using perceptual losses and optical flow consistency to ensure they reflect physically grounded behavior. For downstream control, we introduce a chunked *flow-matching decoder* that maps latent actions to robot-specific continuous action sequences, using only 100 to 200 teleoperated demonstrations. This approach avoids expensive action annotation, supports generalization across embodiments, and enables smooth, high-frequency continuous control upto 22 Hz via chunked action decoding. Unlike prior latent action works that treat pretraining as autoregressive policy learning, ViPRA explicitly models both what changes and how. Our method outperforms strong baselines, with a 16% gain on the SIMPLER benchmark and a 13% improvement across real world manipulation tasks. We have released models and code [here](https://vipra-project.github.io/).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ViPRA, a hierarchical video-based robot control framework that learns discrete latent action representations from large-scale passive human and robot videos and then maps these to continuous robot actions via a flow-matching decoder. The approach jointly predicts future visual observations and latent action sequences during pretraining, and requires only a small amount of teleoperated data for downstream adaptation. The method is evaluated on SIMPLER simulation tasks and several real-world tabletop manipulation tasks, showing improved performance over recent latent-action and vision-language-action baselines.

### Strengths
- The work provides evidence that hierarchical visuomotor control can benefit from large-scale passive human and robot videos, reducing dependence on action-labeled demonstrations.

- The system attains competitive performance on physical manipulation tasks with only 100–200 teleoperated demonstrations, suggesting good data efficiency in downstream adaptation.

- The integration of flow-matching for action decoding leads to stable, high-frequency command generation suitable for real hardware, addressing common control smoothness limitations in discrete latent policies.

- The paper includes both simulation (SIMPLER) and real-world evaluations, with comparisons against several strong recent methods in the latent-action and VLA literature.

### Weaknesses
- **Limited technical novelty.**
Aside from data sacle, this work appears to be primarily an engineering integration of existing components rather than a new conceptual contribution (i.e. latent action learning, chunked action decoding, and flow-matching control) Similar hierarchical designs have been explored in recent systems such as UniVLA, LAPA, and UniPI.

- **Scaling claim not sufficiently validated.**
The paper attributes improvements to pretraining with large-scale passive human videos, but no controlled comparison is provided against training with robot-only video or reduced data subsets. Such experiments would help confirm that data scale, rather than architecture or training strategy, drives the gains.

- **Generalization scope is limited.**
The approach is framed as cross-embodiment and generalist, yet evaluations focus on similar 7-DoF manipulators and relatively simple tabletop tasks (pick and place). Broader embodiment diversity or clearer claim boundaries would improve alignment between motivation and evidence.

- **Ablation coverage.**
The system includes multiple engineered components (optical flow consistency, VQ bottleneck, multi-stage training). More detailed ablations isolating the effect of each would aid in understanding which design elements are critical to performance.

- **Compute and practicality not discussed.**
Although labeled data requirements are low, the approach requires substantial large-scale video pretraining. Reporting approximate compute/time requirements would help practitioners assess feasibility and compare with alternative methods.

Overall, this work offers a valuable scaling demonstration with solid empirical results, but the methodological novelty is limited, placing it at a borderline accept.

### Questions
- One of the most exciting implications of this work is that dynamics priors learned from human interactions may transfer across embodiments. Do the authors have any preliminary evidence or insights into what kinds of human video content (e.g., fine manipulation vs. gross motion) most help downstream control? Any failure cases that suggest limits of human-to-robot transfer?

- Have the authors attempted to manipulate latent action tokens intentionally to achieve specific motion patterns (e.g., alter motion direction, smoothness, or speed)? Observing consistency or interpretability here could reveal whether the learned latents are physically meaningful beyond reconstruction.

- Since the method leans heavily on scaling with passive data, at what point does additional video stop helping? Are there any observed regressions or negative-transfer effects when including noisier human videos from less structured environments?

### Soundness
3

### Presentation
3

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
This work proposes a pretraining-finetuning framework, ViPRA, that learns discrete latent actions from videos without action labels by minimising perceptual and optical flow consistency objectives. Furthermore, this work proposes to use a flow matching decoder to convert the discrete actions to smoother continuous actions, which greatly prevent local discontinuities of previous works. The authors experiments on both simulation and real-world manipulation tasks and shown that the proposed method can achieve stronger performance.

### Strengths
- Very comprehensive related works, greatly aiding the readers in positioning the paper's contributions. 
- The writing was easy to follow and clear, and while the background section is in the appendix, the context provided is enough to have a intuitive understanding of the proposed method. 
- The proposed method regarding representation learning that leverages joint-prediction of pixels and latent actions seems intuitive and shows good performance.
- The proposed method regarding converting discrete actions to continuous control via flow matching is to my knowledge novel. 
- Conducted comprehensive experiments and analysis, and the experiment results seem impressive (if we can ignore a major weakness for the moment).

### Weaknesses
- Major weakness: For Table 1 and Figure 3, which are the main results in the main manuscript, it is unclear how many seeds/runs/rollouts are experimented and how these values are calculated, so it is not possible to draw any conclusion about the (statistical) significance of the results presented at the moment. 
- "Up to 22Hz" in the abstract and introduction is a bit misleading in my opinion, because the real-world experiments were done in 3.5Hz. (i.e. experiments with 22Hz were not done to show the efficacy). 
- Since this work listed on real-world speed as a contribution, several comparison works (LAPA[1]) are not experimented on real world evaluations (i.e. Figure 3) but only in simulation, so it’s unclear how much speed is gained compared to previous works.
- Minor issues:
  - Figure 5 is introduced in the manuscript before Figure 4.
  - Table 6 is not referred nor explained (I assume it’s for appendix E.3, E.4). Furthermore, in Table 6 ViPRA is labelled but should be ViPRA-FM I think.
  - Apart from KV caching, which is more of an engineering technique for me, it is not obvious why inference speed is gained, since at inference the proposed method performs an extra flow-matching step compared to previous works. 
  - The robot for the real world experiments in Figure 6/8 is not clearly written (I assume it’s Franka).
  - Not sure how to interpret the results where in LIBERO tasks, $\pi_0$ and UniVLA both outperform the proposed method ViPRA.

### Questions
I will repost some questions that were mentioned in the weakness section here for clarity.


$$\textbf{Suggestions}$$
S1: Clarify about how the evaluation metrics is calculated in Table 1 and Figure 3.     
S2: It would be great if the authors can either amend the claim of 22Hz a bit, or perform some real world experiments at 22Hz.     
S3: Could provide some clarity about ViPRA-AR since it's not really explained anywhere.    
S4: Minor: I assume the optical flow model RAFT is pretrained and frozen although it is not explained anywhere. It would be great if the authors could clarify.     
S5: Minor: Could improve page 22’s readability a bit by aligning them properly (e.g. at the top).    

$$\textbf{Questions}$$
Q1: Some related works suggest that pixel-level reconstruction is somewhat not as efficient [1][2], what do the authors think about this? Since this work proposes to learn latent actions, can we argue that it would be more effective to learn latent representation features rather than the full pixel reconstruction?     
Q2: If I understand correctly, in Table 1, compared to LAPA[3] and UniVLA[4], the proposed ViPRA performs quite better in full success rate, showing that leveraging video prediction is quite helpful to learn robust representations for SIMPLER tasks. However, in LIBERO’s experiments, UniVLA performs best. What do the authors think about this?    
Q3: Following up on Q2, UniVLA is the best performing framework in LIBERO-10, the authors attributed this to UniVLA being optimised for LIBERO. Can the authors further clarify what this means? For example, what kind of performance can we expect on SIMPLER if we somehow optimise UniVLA for it?    
Q4. Can we also use KV caching for other related works (LAPA, UniVLA) to get improved speed as well (I believe UniVLA already uses action chunking)? If so, it’s unclear to me how much speed is gained compared to previous works.     

---

[1] G. Zhou et al., DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning, arXiv 2024  
[2] R. Sun et al., Learning Latent Dynamic Robust Representations for World Models, ICML 2024   
[3] S. Ye et al., Latent Action Pretraining from Videos, ICLR 2025   
[4] Q.Bu et al., UniVLA: Learning to Act Anywhere with Task-centric Latent Actions, RSS 2025

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ViPRA, a framework for learning robot policy from action-free videos by incorporating video prediction and latent policy learning. The key idea is to pretrain a video–language model to jointly predict both (i) future visual frames and (ii) motion-centric latent actions that summarize local dynamics, guided by perceptual and optical flow consistency losses. These latent representations are mapped to continuous action space through a flow-matching decoder trained on a small number of teleoperated demonstrations.
The paper claims that this “video prediction + latent action” pretraining allows robots to leverage large-scale unlabeled human and robot videos, achieving improvements on both simulation benchmark and real-world tasks.

### Strengths
The paper tackles a major challenge in robot learning—leveraging large-scale actionless videos for control. Using video prediction to inject physical dynamics into latent actions is conceptually coherent and builds upon trends in world-model-based control. Evaluation spans both simulation (SIMPLER benchmark) and real-world manipulation (Franka bimanual setup). ViPRA gets better performance comparing against plausible baselines including LAPA, UniVLA, π0, and diffusion-policy variants.

### Weaknesses
1. **Contribution**
- Unclear novelty: It is not fully clear whether ViPRA introduces a fundamentally new paradigm, or whether it can be viewed as a hybrid of LAPA (latent action tokenization) and UVA (unified video and action prediction).

2. **Codebook Design**
- Codebook size (|C| = 8) appears extremely small compared to typical VQ-based latent action works (e.g., 128–8192).
- No ablation or justification is provided for this choice, nor evidence that such a small capacity suffices to capture motion diversity.
- The paper should include:
(i) Ablations over codebook size (8 / 32 / 128 / 512 / 8192).
(ii) Quantitative codebook utilization metrics (entropy, perplexity, diversity).


3. Data Scale, Composition, and Scalability
- Scaling behavior: Although the model claims to leverage “large-scale actionless videos,” the dataset (~400K clips) remains moderate.
There is no scaling analysis showing how performance varies with the number of pretraining videos.
- A performance–vs–data-size curve would clarify whether ViPRA is still data-limited.
- Human–robot ratio: The impact of mixing human and robot videos is not studied. Different ratios may have large effects on transfer and generalization; sensitivity curves or ablations are needed.
- Generalization limitation: Without scaling or compositional studies, it is unclear if the model can extend to larger internet-scale video corpora.

4. Latent–Action Semantics and Alignment
- The alignment between learned latent actions and ground-truth actions (on datasets where GT is available) is not quantified, which would verify whether latent tokens actually capture actionable dynamics rather than visual motion.
- Would the latent action learned from multiple sources enable a unified action space between multiple embodiments? 

5. Optical-Flow Supervision and Robustness
- The $L_{flow}$  loss is claimed in the contribution but lacks ablation and robustness analysis.
- How much performance degrades without L_flow?
- How stable is RAFT-based flow under high ego-motion, blur, or occlusion?
- Would learned or multi-frame flow estimators perform better?

6. Missing or Incomplete Baselines
- UVA is cited but not directly compared, despite strong overlap in video-conditioned policy learning. Including UVA as an explicit baseline would help contextualize improvements.

### Questions
Please refer to the weakness part. I would like to raise my score once the concerns are resolved.

### Soundness
3

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
4

### Summary
This paper presents ViPRA, a framework that converts video prediction models into robot policies by learning motion-centric latent actions from unlabeled human and robot videos. Its core contributions are a method to extract these physically-grounded latent actions using perceptual and optical flow losses, a pretraining strategy that jointly predicts future video frames and action sequences, and a flow-matching decoder that enables smooth control.

### Strengths
1. Creatively combines video prediction, latent actions, and flow matching in a novel "what" (future state) + "how" (latent action) pretraining paradigm.

2. The paper is well-structured and logically presented, with a clear narrative.

3. The qualitative analysis on latent action representations is interesting.

### Weaknesses
1. The authors did not perform a systematic ablation study on the loss components for the latent action model.

2. The real-world tasks, while commendable, are primarily table-top pick-and-place variants. The paper does not demonstrate generalization to tasks requiring significant non-prehensile manipulation (e.g., pushing, sliding, re-orienting), dynamic environments.

### Questions
1. How does the model reconcile the fundamental kinematic and dynamic differences between human and robot arms when transferring latent actions? Is there a negative transfer from the "noise" of human motion?

2. The entire framework is dependent on the visual perspective of the training videos. Latent actions are also inherently grounded in pixel space. How would performance degrade if the test-time camera viewpoint is different from the training data？

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors address the question of how to convert a video prediction model into a robot policy.  They show that instead of directly predicting actions, the video model can predict future visual observations and motion-centric latent actions.  A flow matching decoder to map  these latent actions to robot-specific action sequences.  The entire system runs at 22Hz for low-latency control.  Authors show that the method outperforms baselines.

### Strengths
- The method is view agnostic, gets data from humans or robots without the need for action labels, can be applied across robot embodiments, and enables low-latency control.

- The appendix provides a lot of detail on the approach and results in the paper.  This is great for reproducibility.

### Weaknesses
- While the paper claims that the method can generalize across embodiments, I didn't seen strong evidence to substantiate this.  Experiments seem to be on one manipulator arm.

- It will be good to include the limitations of the proposed work in the paper - this is currently missing.

### Questions
- Fig. 2: The connections between the left / right figures is unclear.  Where does the left figure fit in on the right?  Where does the output of the left figure (the latent actions) go as input on the figure on the right? Is the "Latent Action Embedding E_\phi" block in the right figure represented by the entire left figure?

- If I understand correctly, the paper mentions both discrete latent actions as well as continuous latent actions.  If this is right, then it's unclear what is the difference between these and why the need for both discrete and continuous latent actions.  (The continuous latent actions are then decoded to continuous action chunks.)

### Soundness
3

### Presentation
3

### Contribution
3
