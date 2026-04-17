# On the Representation Degradation in Vision-Language-Action Models

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Vision-Language-Action (VLA) models have become a promising paradigm for robotic decision-making, yet their application remains limited by generalization bottlenecks. In this paper, we conduct a layer-wise representation analysis and uncover a previously overlooked phenomenon of representation degradation: deeper layers tasked with action generation exhibit diminished generalization to both semantic information and environmental dynamics. To mitigate this issue, we introduce hidden Space WOrld modeLing (SWOL), a lightweight but efficient approach that aligns degraded deep-layer features with more generalizable mid-layer representations extrapolated from future observations. SWOL enforces temporally consistent, action-grounded representations without modifying model architecture or inference procedures. Extensive experiments in simulation and real-world settings demonstrate that SWOL alleviates representation degradation, leading to improved policy effectiveness and stronger generalization across modalities of vision, language, and dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work isolates the issue of representation degradation within vision-language action models, where deeper layer fail to carry rich information (semantic and dynamical) that is useful for generalization. These metrics are defined in this context and three VLA models are analyzed to exhibit these shortcomings. The solution proposed to this issue, SWOL, consists in encouraging the deeper perceptual features to match mid-level features from the next observation in time. This is done as to force some sort of world model representation learning at those deeper layers. This is evaluated in simulation as well as in the real world and some performance benefits are shown.

### Strengths
The paper topic is very relevant to the current approaches to robot learning, and the problem highlighted is clearly identified.

The metrics to evaluate the degradation phenomenon are clear and the case made for this being an issue is sound and convincing.

The design with the self-supervised loss aligning representations between timesteps is quite elegant.

The experiments run are adequate to test the hypothesis made and solution suggested.

The performance gains, though not miraculous, are sufficiently beneficial for this to be an interesting result.

### Weaknesses
**Insight and implementation**
- The authors resort to world modeling as the general idea behind the solution for mitigating representation degradation in deep VLA layers. Just in terms of presentation, the insight should not be a question (lines 231-232) rather an observation.
- The insight on its own is insufficient to directly lead to the solution proposed: current deep layer features aligned with future mid layer features. This warrants more explanation as to why this specifically is the best way and not just one way to do things.
- The mid level representations seems to also vary substantially in quality both within the same architecture (e.g. pi_0's 8th layer does very poorly on semantic classification) and among VLAs (Very noisy for openVLA while very clean for pi_0-Fast). Admitting that the observation is general and empirical, this is still not discussed to a sufficient extent, and the solution does not seem to directly account for this.
- The presentation of the partitioning of the hidden features into perceptual and action parts is not clearly presented. It is unclear to the reader why such a partitioning is taken as a postulate. The authors cite Gandelsman et al., 2024, but the text is in no way self-contained in presenting this decomposition method and relating it to the policy architectures considered.
- Along these lines, the update rules (lines 112-124) are hard to decipher both because of the above point, but also due to the cumbersome notation. I would suggest rewriting this section as well as creating a more technical figure that exhibits the decomposition and update rules in a clear fashion, even should it be in the appendix.

**Presentation**
- I did not find figure 1 very useful, especially considering the area/information ratio.
- In table 1 the best per data fraction and per task score should be in bold as it is fatiguing on the eyes to decipher such a big block of numbers. This is done for the average length but I suspect practitioners care more about success rate.
- The paper fails to report a reproducibility statement as well as a statement on the use of LLMs which I believe are required

### Questions
- Why does it make sense to do deep-layer to mid-layer of next step matching?

- What is the "target" mid-layer selection protocol?

- Why is the average length an interesting/meaningful metric (table 1) ?

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
The paper “On the Representation Degradation in Vision-Language-Action Models” presents an in-depth study of how internal representations evolve across layers in Vision-Language-Action (VLA) models. The authors uncover a consistent and concerning trend: representation degradation, where deeper layers,  responsible for generating actions,  lose generalization capacity to both semantic and dynamic aspects of the environment.
To address this, the paper proposes SWOL (Hidden Space World Modeling), a lightweight auxiliary objective that aligns degraded deep-layer features with mid-layer representations from future observations. SWOL effectively introduces a self-supervised “world modeling” signal in hidden space without architectural modifications or inference overhead.
Extensive experiments on CALVIN (simulation) and real-world robotic tasks (Aloha/ARX5 platform) show that SWOL improves policy generalization, particularly in low-data settings, enhancing both semantic grounding and dynamic awareness of VLAs.

### Strengths
- The discovery of representation degradation fills an important analytical gap. By decomposing the forward pass and probing layer-wise generalization, the authors provide valuable interpretability into how semantic and dynamic information dissipates through depth.

- SWOL stands out for its conceptual clarity: it uses mid-layer representations from future observations to “rejuvenate” degraded deep-layer embeddings. This plug-and-play auxiliary loss is elegant, general, and requires no changes to model architecture or inference.

- The authors test SWOL across multiple VLA architectures (π₀, π₀-fast, OpenVLA-OFT), different data regimes (1%, 10%, 100%), and both simulated and real-world environments. The consistent improvements across setups strengthen the empirical claim.

- Unlike many representation analysis papers confined to simulation, this work extends experiments to real manipulation tasks like folding a towel, plugging in a cable, and cleaning a table. These practical gains significantly bolster the contribution’s impact.

- The study includes careful ablations on loss weight, target layer, and predictor architecture, along with visualization of improved deep-layer representation quality. This rigor demonstrates strong experimental maturity.

### Weaknesses
- The connection between SWOL and formal notions of world modeling remains intuitive rather than mathematically grounded. The paper could benefit from a clearer theoretical explanation of why aligning to future mid-layer features enhances generalization beyond empirical observation.

- All experiments focus on imitation learning scenarios. While the method should, in principle, generalize to reinforcement learning or planning-based agents, this is not tested or discussed in detail.

- Although the authors claim “no inference overhead,” SWOL roughly increases training GPU hours by 25–30%. This discrepancy could be better contextualized.

- Similar auxiliary consistency losses (e.g., temporal or cross-view prediction) have been explored in visual representation learning. The paper’s differentiation from these paradigms could be more explicit.

- Aligning deep features to mid-layer targets risks encouraging representational homogeneity. While empirical results show gains, a discussion on possible over-regularization effects is missing.

### Questions
- How does SWOL compare against standard temporal consistency or contrastive predictive coding baselines in representation learning?

- Does the improvement persist if the target mid-layer is randomly sampled instead of fixed (e.g., layers 5–9)?

- Could SWOL interfere with the diversity of action representations by enforcing excessive alignment across time?

- Could the authors provide qualitative visualization (e.g., t-SNE) of how representation structure changes before vs. after SWOL?

### Soundness
4

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
4

### Summary
This paper discovers that semantic and dynamic representations degrade with network depth in VLA models, reducing generalization. To verify this, the authors use task intent classification for semantic generalization and inverse dynamics prediction for dynamic generalization. They propose SWOL, a simple yet effective approach, which makes deep-layer features predict mid-layer representations from future observations, helping to recover lost information and mitigate degradation.​ The authors conduct extensive experiments on the CALVIN simulation benchmark and ARX5 robotic platform, testing $\pi_0$, $\pi_0$-fast, and OpenVLA-OFT and showing SWOL's superiority. Ablation studies analyze key design choices, and further analysis of semantic and dynamic prediction results confirms SWOL's effectiveness. The paper's main contributions are identifying representation degradation as a key issue in VLA models and proposing the plug-and-play SWOL method to address it.

### Strengths
1. This paper rigorously identifies the representation degradation phenomenon in fine-tuned VLA models, which is an overlooked issue in prior VLA representation research.

2. The authors design two well-targeted evaluation protocols: semantic intent classification and inverse dynamics regression, systematically measure the distribution of semantic and dynamic information across layers, clearly revealing that mid-layers maintain strong representational quality while deeper layers suffer from significant degradation.

3. The proposed SWOL method is innovative in its design. By performing future mid-layer representation prediction in the hidden space, it avoids the computational inefficiency and appearance sensitivity of raw visual-space future prediction, while forcing the model to re-learn degraded representations. It's plug-and-play, enabling seamless integration with various existing VLA models.

### Weaknesses
1. The paper lacks quantitative comparative experiments with model-based methods discussed in the Related Works section.

2. Typo: The fourth legend in Figure 5 should be "Dyn. w. SWOL"

### Questions
1. The paper posits that direct visual-space modeling is highly susceptible to appearance variations. However, it lacks robust quantitative evidence demonstrating the superiority of SWOL. Are there comparative experiments showcasing SWOL's performance edge over conventional visual future prediction methods? Specifically, tests should involve varying critical factors such as lighting conditions, object textures, and background clutter to comprehensively validate the robustness claims.

2. Given that SWOL adds an auxiliary loss during training, does it introduce any unintended side effects, such as overfitting to mid-layer representations or compromising the original action generation capability of VLA models? If so, how are these trade-offs managed?

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
The paper finds a layer-wise representation degradation phenomenon in fine-tuned Vision-Language-Action (VLA) models, losing task semantics and dynamics information in the deep layers. Then, this paper proposes SWOL (Hidden Space World Modeling), training an alignment between deep-layer features to mid-layer features of the next observation with a simple MLP predictor. SWOL has no additional inference cost, and yields consistent gains on CALVIN and in real-robot manipulation tasks.

### Strengths
- Generalization of VLA models is an important research problem.
- The paper empirically observes representation degradation and low-complexity intervention that efficiently solves the problem with improvements in simulated and real world settings.
- Results span multiple VLA backbones, low-data regimes, long-horizon tasks, and real-robot experiments.
- The method has no inference overhead, making it attractive and easy for applied use.

### Weaknesses
- Unclear necessity of correcting representation degradation: Many VLA architectures condition action decoding on all intermediate features of VLM. If the action expert can access earlier semantically rich features, it is not obvious why degradation in some deep layers must be corrected since there could be semantics agnostic behavior in the deep layers such as precise refinement of actions. Performance gain could be purely from integration of dynamics in the world model objective.

- Insufficient comparison to prior world-modeling baselines:  From prior works [1, 2], performance gains from predictive auxiliary objectives like implicit or explicit next-state prediction is not that surprising. This work does not convincingly separate SWOL’s benefits from these approaches. Direct empirical comparisons to at least a couple of simple representative baselines are necessary.

[1] Zhao, Qingqing et al., CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models.

[2] Zheng, Ruijie et al., FLARE: Robot Learning with Implicit World Modeling.

### Questions
- How well does representation semantics and dynamics experiments perform with pretrained VLM weights? Discrete action tokens tend to less disturb LLM’s representation space but also appear in discrete action models, so representation degradation could be related to biased behavior from the pretrained weight.
- In ablation, using the first layer as alignment target shows best performance. Using the first layer as target is nearly equivalent to CoT-VLA, except output tokens are from the perception tokens and targets are from layer 1 (which is close to input representation), questioning the need of mid-layer alignment where task semantics and dynamics are theoretically upper bounded by that of input representations.
- The result of target layer 5 in Table 3 appears missing.

### Soundness
3

### Presentation
3

### Contribution
2
