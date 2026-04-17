# SketchEvo: Leveraging Drawing Dynamics for Enhanced Image Synthesis

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Sketching represents humanity's most intuitive form of visual expression -- a universal language that transcends barriers. Although recent diffusion models integrate sketches with text, they often regard the complete sketch merely as a static visual constraint, neglecting the human preference information inherently conveyed during the dynamic sketching process.This oversight leads to images that, despite technical adherence to sketches, fail to align with human aesthetic expectations. Our framework, SketchEvo, harnesses the dynamic evolution of sketches by capturing the progression from initial strokes to completed drawing. Current preference alignment techniques struggle with sketch-guided generation because the dual constraints of text and sketch create insufficiently different latent samples when using noise perturbations alone. SketchEvo addresses this through two complementary innovations: first, by leveraging sketches at different completion stages to create meaningfully divergent samples for effective aesthetic learning during training; second, through a sequence-guided rollback mechanism that applies these learned preferences during inference by balancing textual semantics with structural guidance. Extensive experiments demonstrate that these complementary approaches enable SketchEvo to deliver improved aesthetic quality while maintaining sketch fidelity, successfully generalizing to incomplete and abstract sketches throughout the drawing process.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes SketchEvo, with two parts:
- SGPO: extend SPO-style pairwise preference optimization to sketch-guided diffusion by building the candidate pool with different sketch stages (not just noise like the prior works) and picking top/bottom by a pretrained preference scorer.

- SGR: an inference-time sequence-guided rollback that blends conditional/unconditional scores with text+sketch (two γ’s), arguing increased “information gain” when text and sketch diverge.

evaluated on Sketchy (train), and tested also on QuickDraw, AnimeDiffusion, FSCOCO. metrics: ImageReward / HPSv2 / PickScore (aesthetics), CLIPScore (text alignment), and a sketch-LPIPS variant (sketch fidelity). baselines include ControlNet, T2I-Adapter, VersaGen, and ControlNet with DPO/SPO.

### Strengths
- Simple, intuitive idea: use sequence of sketches as a structured source of diversity ->  stronger positive/negative pairs -> better gradients than noise-only SPO. this feels original within sketch-conditioned alignment.

- Method framing: clear diffusion-probability ratio objective (SPO/DPO family) and a roll-back mechanism adapted to dual conditions (text+sketch).

- Decent empirical lift: consistent gains on preference metrics, with some qualitative improvements in composition/position/color, and cross-dataset demos.

- Ablations: show candidate-pool size effects and performance vs. which sketch stage guides rollback; also a tiny “scoring model” swap test.

- SGPO vs prior alignment: prior DPO/SPO/LPO/D3PO target text-only or step-aware preferences but don’t, to best of my knowledge, explicitly use sketch-sequence stages to create candidate diversity at each diffusion step. that is a novel and plausible extension for multimodal control.

### Weaknesses
- SGR vs rollback work: Rollback/zig-zag/self-reflection sampling has been explored for text-to-image; adapting it to joint text+sketch guidance with a sequence-aware blend is a reasonable incremental contribution. the theoretical “information gain increases with text–sketch divergence” is interesting, but depends on approximations and fixed γ’s.

- Baselines: ControlNet, T2I-Adapter, VersaGen are appropriate; ControlNet-DPO/SPO adaptations are helpful. but given 2024–2025 progress, additional strong controls (e.g., ControlNet++ [1] /SmartControl [2] or more recent) would be useful; at least include one recent sketch-aware control baseline if feasible.

- Possible circularity / reward hacking: The same family of preference scorers appears to be used both for pair selection during training and for evaluation, risking overfitting to those scorers rather than to human judgment. Cross-metric generalization is only partially demonstrated.

- Limited human studies / no significance testing: No human pairwise evaluation targeted to sketch-conditioned synthesis. No confidence intervals, multiple seeds, or statistical tests are reported for the main tables.

- Writing clarity: Background part (L126-149), is not clear and detailed enough. All the terms, should be explained clearly for readers to better grasp the discussion. For example, what is $x^w_t$, $x^l_t$ ?

[1] https://arxiv.org/abs/2404.07987

[2] https://arxiv.org/abs/2404.06451

### Questions
On reward hacking and evaluation metrics: My primary concern is the potential for "reward hacking" or circularity in the evaluation. The paper states that SGPO uses a "pretrained scoring model" (L195) to select the winning $x^w_t$ and losing $x^l_t$ samples during training. The evaluation in Table 1 then relies on automated preference metrics like ImageReward, HPSv2, and PickScore.

- Could you please clarify which specific "pretrained scoring model" was used for training?
- If this training scorer is one of the evaluation metrics (or from the same family, e.g., another CLIP-based preference model), how can we be sure the model hasn't simply learned to optimize for the biases of this specific scorer, rather than for genuine, generalizable human aesthetic preference?


Necessity of human evaluation: Following the previous point, the gains on automated preference scores are strong, but their correlation to actual human judgment is not guaranteed, especially in a dual-condition (sketch+text) setting.
 - Given the risk of reward hacking, a human evaluation study (e.g., pairwise preference tests comparing "Ours" vs. "ControlNet-SPO" and "ControlNet") seems crucial to validate the paper's central claims. Would it be possible for the authors to provide such a study, even on a smaller scale, to confirm that the observed metric improvements translate to tangible, human-perceived quality gains?


Missing SOTA baseline comparisons: The related work section and introduction cite several strong, recent methods for controllable generation, such as SmartControl  and ControlNet++, which are highly relevant.

- Could the authors provide a quantitative or qualitative comparison against these (or similarly strong) recent baselines? If this is not feasible, could you please provide a more detailed discussion on how SketchEvo's contributions differ from and are expected to perform against these methods, which also aim to improve fidelity in controllable generation?


SGPO Mechanism and Learning Signal: In the SGPO formulation (Eq. 4), the model is trained on pairs ($x^w_t$, $x^l_t$) generated from different sketch conditions ($c_{s_w}$,  $c_{s_l}$)
How does this process ensure the model is learning a general "aesthetic preference" rather than simply learning to "prefer more complete sketches"? For instance, if $x^w_t$ is frequently generated from a late-stage sketch ($s_N$ ) and  $x^l_t$ from an early-stage sketch ($s_1$), is it possible the model is just learning to assign a higher likelihood to more detailed conditions?


Justification for SGR using $s_1$: The ablation (Fig. 5a) suggest that using the most abstract sketch ($s_1$) for the rollback mechanism (SGR) is optimal, attributing this to maximized "information gain" from text-sketch divergence.
- This is an interesting finding, but the theoretical justification in Appendix D relies on several approximations. Could you provide more intuition on why maximal divergence (using $s_1$) is superior to using a more structurally-informative intermediate sketch (e.g., $s_{0.6N}$ )? One might expect a more complete sketch to provide a better structural prior during the rollback.

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
5

### Summary
The paper introduces SketchEvo, a sketch-guided image generation framework that leverages the drawing sequence—from early strokes to the completed sketch—rather than treating the final sketch as a static constraint. The core claim is that existing text+sketch methods underperform because they ignore preference signals implicit in the sketching process and struggle to produce sufficiently diverse positive/negative samples for preference alignment under multimodal constraints. 

SketchEvo tackles this with two components: 

* Sequence-Guided Preference Optimization, which uses sketches at different completion stages to create meaningfully divergent training pairs and yield stronger gradients for preference learning; 

* Sequence-Guided Rollback, an inference-time mechanism that balances textual semantics with sketch structure by integrating the sketch sequence into rollback updates so that learned aesthetic preferences transfer to generation while preserving structural fidelity.

### Strengths
* Clever use of the sketching sequence (early→late strokes) to create structured candidate diversity for preference learning; pairing SGPO with SGR is a fresh, problem-driven combo.

* Method is well-specified; ablations on candidate-pool diversity and rollback settings support the mechanism(however, they didn't mention the number of GPUs that they used for training and just mentioned the type of GPU); consistent wins on preference and sketch-fidelity metrics across multiple datasets.

### Weaknesses
* **Lack of related work & baselines for hand-drawn sketches.** The paper mostly compares to ControlNet-style direct sketch conditioning. It omits stronger, recent sketch-specific approaches such as Democratising Sketch Control [2] and KnobGen [1], which explicitly address amateur/sparse sketches—neither discussed nor used as baselines. Add these to Related Work and include at least one competitive replication.

* **Bad Experiments, need more robust results**: As the author shows in Figure 6, s_i affects the generation. ControlNet and the T2I-Adapter have coefficients that the client can control to receive a different level of condition on spatial information. Did the author change the coefficient for those baselines?

* **Proxy metrics only for aesthetics.** Claims rely on ImageReward/HPSv2/PickScore; no human A/B tests or significance. 

* **Requires stroke sequences.** Many users provide only the final sketch. Assess a “no-sequence” mode (pseudo-sequences via edge reveals/stroke segmentation) and report the performance drop.

* **Cost underreported**. SGPO candidate pools and SGR rollbacks add compute. Report wall-clock per step, GPU hours, VRAM, and quality–cost curves vs. K and rollback steps.

* **Effect of text on generation**: I think text has a lot of effect on the generation. We can see this effect in Figure 10. What is the effect of text on generation? Do they have any experiments without text and only rely on the sketch?







[1] Navard, Pouyan, et al. "KnobGen: controlling the sophistication of artwork in sketch-based diffusion models." arXiv preprint arXiv:2410.01595 (2024).

[2] Koley, Subhadeep, et al. "It's All About Your Sketch: Democratising Sketch Control in Diffusion Models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
* **Metric validity for structure:** Sketch-LPIPS can be biased by edge density. Can you add a complementary metric (e.g., edge chamfer distance, keypoint F-score) and/or a small human “structure-match” study?

* **Training data specifics:** How many unique stroke sequences per class were used? Any filtering for noisy/misaligned strokes? Please clarify sequence quality controls and their impact.

* **Robustness to sketch noise:** How sensitive are results to (a) shuffled stroke order, (b) dropped early/late strokes, (c) jittered stroke geometry, and (d) off-prompt doodles? 


**I am open to changing my score based on the author's responses.**

### Soundness
2

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
3

### Summary
SketchEvo seeks to improve sketch-to-image generation by analyzing the drawing process rather than treating sketches as static blueprints. SketchEvo captures the evolutionary stages of drawing—from initial rough strokes to the completed sketch—to understand what the artist prioritizes and intends. The claim is that by recognizing that the sequence and refinement of strokes reveal important information about the drawing's intended meaning, SketchEvo generates images that better align with human preferences and aesthetic sensibilities.

The authors state that the system achieves these improvements through two innovations: a training method that uses sketches at various completion stages rather than random variations to learn aesthetic preferences more effectively, and a generation process that employs initial sketch strokes to guide a "rollback" mechanism balancing structural fidelity with natural beauty. 

The results are stated as demonstrating superior aesthetics with higher human preference scores.

### Strengths
The notion of capturing artist **intent** in the generation and refinement of an image is a novel idea that this reviewer appreciated. This is a well-written paper. The techniques employed seemed sound. Results were interesting. Ablation studies were good.

### Weaknesses
What seemed to be missing is an explicit capturing of the artist's intent with a view towards having the model provide explanations and also learn high level concepts.

### Questions
It wasn't clear whether or how the model and the representations captures the artist's intent in an explicit manner. An explanation of that would be helpful.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the SketchEvo framework, which harnesses the dynamic evolution of sketches for enhanced image synthesis. The key insight is that intermediate sketches from different drawing stages represent varying levels of abstraction and detail. These sketches offer meaningful semantic and structural divergence while maintaining a connection to the user's intent.

### Strengths
- The paper proposes a novel framework that leverages the sketch sequences for preference-based optimization.
- The experimental results demonstrate improvements over baseline methods.

### Weaknesses
- The paper does not clearly justify why using intermediate sketches from different drawing stages enhances human preference for generated images. While intermediate sketches offer more structural divergence and variation, it remains unclear how this divergence and variation improve human aesthetic preference. The early sketch strokes actually contains very little information of the full structure.
- It would be interesting to see the results if the sampling sketch condition $c^k_{s_n}$ were replaced with a constant value $c^k_{s_N}$ while sampling different noise z to ensure sample diversity. To my understanding, this sampling method differs from ControlNet-SPO. This experiment would verify whether the performance gain comes from using intermediate sketches rather than simply increasing the number of samples.

### Questions
- Why using intermediate sketches from different drawing stages enhances human preference for generated images?
- Does the performance gain come from using intermediate sketches or simply from increasing the number of samples?

### Soundness
2

### Presentation
3

### Contribution
3
