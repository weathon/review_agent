# Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-Language Navigation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 8

## Abstract
While recent large vision-language models (VLMs) have improved generalization in vision-language navigation (VLN), existing methods typically rely on end-to-end pipelines that map vision-language inputs directly to short-horizon discrete actions. Such designs often produce fragmented motions, incur high latency, and struggle with real-world challenges like dynamic obstacle avoidance.
We propose DualVLN, the first dual-system VLN foundation model that synergistically integrates high-level reasoning with low-level action execution. System 2, a VLM-based global planner, "grounds slowly" by predicting mid-term waypoint goals via image-grounded reasoning. System 1, a lightweight, multi-modal conditioning Diffusion Transformer policy, "moves fast" by leveraging both explicit pixel goals and latent features from System 2 to generate smooth and accurate trajectories.
The dual-system design enables robust real-time control and adaptive local decision-making in complex, dynamic environments. By decoupling training, the VLM retains its generalization, while System 1 achieves interpretable and effective local navigation. DualVLN outperforms prior methods across all VLN benchmarks and real-world experiments demonstrate robust long-horizon planning and real-time adaptability in dynamic environments.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents DualVLN, a dual-system foundation model for vision-language navigation (VLN). The proposed architecture decouples slow, global reasoning from fast, local control. System 2, built upon QwenVL-2.5 (7B), performs low-frequency, image-grounded reasoning to predict mid-term waypoints via pixel goal grounding and latent goal representation. System 1, a lightweight diffusion transformer policy, generates smooth, real-time trajectories conditioned on these goals and high-frequency RGB inputs.
This asynchronous slow-fast design enables both high-level reasoning and agile local control. DualVLN achieves state-of-the-art results on VLN-CE and VLN-PE benchmarks and shows strong real-world generalization across multiple robot embodiments. The authors further introduce Social-VLN, a new benchmark evaluating social awareness and dynamic obstacle avoidance.

### Strengths
1. Novel dual-system design that elegantly separates global reasoning and real-time control, addressing key challenges of latency and fragmented motion in VLN.
2. Pixel goal grounding is a well-motivated and technically sound intermediate representation, providing interpretable supervision and improving semantic grounding.
3. Strong empirical results: DualVLN consistently outperforms existing VLM-based and diffusion-policy baselines on VLN-CE, VLN-PE, and real-robot experiments.
4. Comprehensive evaluation, including the new Social-VLN benchmark, ablation studies, and cross-embodiment real-world tests, demonstrating robustness and scalability.

### Weaknesses
1. The paper frames DualVLN as a reasoning–acting foundation model, but the reasoning component is mostly implicit. System 2 performs spatial grounding and waypoint prediction rather than explicit multi-step or interpretable reasoning, making the “foundation-level reasoning” claim somewhat overstated.
2. The main novelty lies in the dual-system and asynchronous architecture, which is well-engineered but incremental. Similar slow–fast paradigms have appeared in StreamVLN, Helix, and Hi-Robot; the paper could clarify what new learning principle is introduced beyond system integration.
3. The Social-VLN benchmark is valuable yet briefly analyzed. Deeper quantitative or behavioral analysis would better support its claimed generalization and practical significance.

### Questions
Could the authors provide qualitative analysis or visualizations showing how System 2 grounds the pixel goal (e.g., attention maps or reasoning traces)?

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
4

### Summary
This paper proposes a dual-system framework (DualVLN) for VLN, consisting of System 2 and System 1. The authors mention that this framework can leverage the strong reasoning capabilities of VLMs and the fast inference speed of diffusion policies. While the implementation is described comprehensively, some concepts lack clarity. DualVLN is evaluated on multiple benchmarks, including VLN-CE, R2R, RxR, and the proposed Social-VLN benchmark. Results from both benchmark tests and real-world experiments demonstrate the effectiveness of DualVLN.

### Strengths
- The dual-system design appears well-designed, and the results demonstrate that DualVLN achieves both a high success rate and fast execution speed. 
- DualVLN has been extensively evaluated in both synthetic and real-world environments. 
- The proposed Social-VLN benchmark could benefit the research community by providing more realistic VLN environments.

### Weaknesses
The major weakness of this paper is that many statements and descriptions are unclear or confusing:
 - L173 mentions "self-directed view adjustment capabilities," but no details are provided on how this capability is implemented. It remains unclear how the VLA models are trained to possess this ability or how it is utilized during inference.
- L83 states that the training of System 1 and System 2 is decoupled. This raises the question of how the "Trajectories Query" in Figure 2 is learned within QwenVL. Specifically, what supervision signal is used for the latent trajectories?
- L425 references a "Table X," which appears to be missing. This table seems crucial for evaluating the effectiveness of pixel goal prediction.
-  The format of the pixel goal is not specified. Is it represented using special tokens or simple coordinate text?

### Questions
- Please refer to the 'Weaknesses' section for detailed concerns.
- Providing visualizations of the pixel goal in both synthetic and real-world environments would help readers better understand this concept.
- The rationale for using a pixel goal instead of directly predicting waypoints relative to the robot. Waypoints may seem more straightforward.

If the authors can address these concerns, I would be happy to raise my score.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes **DualVLN**, a **dual-system foundation model for vision-language navigation (VLN)**. The core insight is to decouple high-level semantic reasoning (System 2) from low-level control (System 1) in a slow–fast, hierarchical architecture.
System 2 (“Ground Slow”) is a **7 B-parameter vision-language model (VLM)** that performs mid-term waypoint planning through **pixel-goal grounding** and **self-directed view adjustment**.
System 1 (“Move Fast”) is a **multi-modal diffusion transformer policy** that conditions on both explicit pixel goals and latent features from System 2, enabling high-frequency (30 Hz) trajectory generation and dynamic obstacle avoidance.

The paper introduces a new **Social-VLN benchmark** to test navigation under dynamic, human-like obstacles, and evaluates DualVLN extensively across **VLN-CE**, **VLN-PE**, and real-world robotic platforms (wheeled, quadruped, humanoid).
DualVLN achieves new state-of-the-art performance on VLN-CE and VLN-PE and demonstrates real-time adaptability and strong cross-embodiment generalization.

### Strengths
**1. Clear and Novel Architectural Concept – Dual-System Slow/Fast Design:**
Decoupling a slow, deliberative VLM planner from a fast, diffusion-based controller is an elegant and well-motivated solution to the latency and fragmentation issues of end-to-end VLN models.
The “Ground Slow, Move Fast” principle is conceptually intuitive and technically grounded, providing a bridge between symbolic reasoning and reactive control.

**2. Solid Methodological Design and Implementation:**
The formulation of **pixel-goal grounding** and **latent-goal extraction via learnable queries** is novel.
System 1’s **flow-matching diffusion objective** (Eqns 1–3) and **multi-modal conditioning pipeline** (Q-Former + ViT + DiT) are clearly described and reproducible.
The asynchronous inference pipeline (2 Hz for System 2, 30 Hz for System 1) demonstrates engineering maturity and real-time feasibility.

**3. Comprehensive Empirical Evaluation:**
DualVLN surpasses all prior methods—particularly **StreamVLN** and **NaVILA**—on both **R2R-CE** and **RxR-CE** unseen splits (SR ↑ 64.3%, SPL ↑ 58.5%), and maintains superiority under physical dynamics (VLN-PE).
Zero-shot real-world tests across three robot platforms validate strong sim-to-real transfer.

**4. Introduction of Social-VLN Benchmark:**
Extending VLN to dynamic, human-populated settings (with humanoid agents) is a meaningful contribution. The proposed **Human Collision Rate (HCR)** metric provides a new perspective on social safety and trajectory recovery.

**5. Careful Ablations and Analysis:**
Ablation studies (Figure 6, Table 4) effectively dissect the roles of pixel goals and latent goals, showing tangible drops when each is removed.
The **data-scaling analysis** for System 1 provides insights into data efficiency, demonstrating near-saturation at ≈ 10% of training data.

**6. Strong Real-World Validation:**
Results on TurtleBot 4, Unitree Go2, and G1 confirm robust transfer and low navigation error, with detailed qualitative analyses (Figure 5).
Few VLN papers provide this level of real-robot evaluation.

**7. Writing and Clarity:**
The paper is clearly organized, with high-quality figures (Figures 1–2, 5–7) that make the dual-system workflow and results intuitive.

### Weaknesses
**1. Conceptual Novelty Is Weak and Superficial.**
The “dual-system” framing (slow = planner, fast = controller) has been explored extensively in prior works on hierarchical RL, slow–fast control, and modular VLA models (e.g., Helix 2025, Hi-Robot 2025, RoboPoint 2025). The present paper mostly *repackages* a conventional high-level planner + low-level policy pipeline with new terminology (“Ground Slow, Move Fast”) rather than introducing new algorithmic insights. The design feels more like narrative framing than a fundamental contribution.

**2. Limited Theoretical Insight:**
While the framework presents an appealing “fast–slow” dual-system design, the paper offers limited theoretical grounding for how latent-goal conditioning from the VLM stabilizes or aligns representations within the DiT-based controller. The motivation for using latent queries as a bridge between perception and action remains heuristic rather than principled. A more formal analysis, e.g., how latent conditioning improves stability, credit assignment, or planning coherence, would considerably strengthen the work’s theoretical foundation.

Furthermore, the central problem this paper addresses appears to be the offline–online training gap in modular VLN systems, similar to how ETPNav bridges DuET, ScaleVLN bridges HAMT, GridMM & BEVBert, etc., bridge their methods into the continuous environment through waypoint models and online finetuning. It is unsurprising that decoupling the LLM (planner) and controller (executor), then finetuning the latter for error correction, leads to improved downstream performance. However, this design also highlights a deeper limitation: the “fast” and “slow” systems do not jointly plan toward a shared latent goal. Instead, the fast system functions primarily as an error-corrective compensator for the slow system rather than as a hierarchically consistent planner. This undermines the claimed cognitive analogy to hierarchical reasoning and weakens the novelty of the dual-system framing.

**3. Missing Discussion of the System Mismatch.**
System 2 predicts **explicit pixel goals**, while System 1 consumes **latent goals**. The paper lacks explanation:

* Why the latent goal is preferable to directly using the pixel goal;
* How consistent or accurate the pixel-goal grounding is;
* Why System 1 actually benefits from the latent embedding.

Although L428-L431 try to explain this, it is far from convincing. To me, the results in Figure 6 and Table 4 show that it is necessary to compensate for the offline training gap for a large system during online inference. The diffusion policy could compensate for such a discrepancy by converting the action space into a continuous trajectory and correcting suboptimal pixel goals (by using a latent goal instead). However, these hypotheses are lacking evidence in the paper; no error analysis for why LLM + iPlanner/ DP are inferior, and what error system 2 corrects.


**4. Clarity of Data Generation and Training Pipeline:**
Details of **System 2 training data**, the projection from 3D to 2D pixel goals, and the **view-adjustment supervision** are somewhat compressed.
More transparency (e.g., dataset size, sample generation rate) would aid reproducibility.

**5. Lack of Real Justification for the “Fast” System.**
The argument that System 1 must operate at 30 Hz is not substantiated with latency or performance ablation. The experiments show improved results but do not prove that *asynchrony* itself yields the gains. No evidence is given that the fast controller corrects or compensates for sub-optimal pixel goals.

**6. Incremental Empirical Improvement.**
Although numbers on VLN-CE/VLN-PE slightly surpass baselines, the margin is modest and could easily stem from larger model capacity (7 B QwenVL backbone) or training data rather than the proposed dual-system design.

**7. Limited Analysis and Interpretability.**
There is no diagnostic evaluation of the intermediate goals, no visualization of latent features, and no discussion of why the dual representation improves reasoning. The Social-VLN benchmark, while new, does not isolate the contribution of the architecture. It simply provides another testbed.


**8. Potential Compute and Accessibility Concerns:**
Although System 1 is lightweight, System 2 uses a 7 B VLM (QwenVL-2.5), which may limit accessibility.
A discussion on inference cost, scalability to smaller models, or open-source readiness would be beneficial.

### Questions
1. Why is a **latent goal** required at all if explicit pixel goals already encode spatial intent?
2. How does System 1 benefit from the VLM’s latent state? Can you show feature alignment or ablation?
4. What empirical evidence supports the need for a **“fast” 30 Hz controller** rather than standard synchronous inference?
5. Is there any analysis showing that the **asynchronous design** contributes more than simply using a smaller local policy?
6. **System 2 → System 1 Coupling:** How are temporal inconsistencies handled when latent goals become outdated during fast inference? Is there a re-synchronization or interpolation strategy beyond using the last frame?
7. **Self-Directed View Adjustment:** How is the view-adjustment policy trained—supervised with labeled actions or via reinforcement from coverage gains?
8. **Benchmark Generalization:** Can Social-VLN agents interact with multiple moving agents simultaneously, or only single humanoid trajectories?
9. **Failure Modes:** What are common failure cases in dynamic scenes (e.g., oscillation, stuck behaviors)?
10. **Scalability:** Could System 1 be extended to 3D point-cloud inputs or depth channels without retraining System 2?
11. **Resource Footprint:** What is the average inference latency per navigation cycle (end-to-end) and GPU memory footprint during deployment?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This article proposes a dual-system framework called DualVLN for the vision-language navigation (VLN) task. DualVLN employs Qwen2.5-VL as System 2 to encode visual observations and language instructions, predicting pixel-level goals based on the current observation. In addition, System 2 introduces learnable latent queries to enhance its interaction with System 1.
System 1 is a diffusion-based model designed to generate short-horizon, high-frequency, and collision-free trajectories. System 2 is first trained through a pixel-goal grounding task and then fine-tuned on VLN datasets, while System 1 is trained using flow matching.
Extensive experiments are conducted on multiple benchmarks, including R2R-CE, RxR-CE, VLN-PE, and Social-VLN. DualVLN achieves state-of-the-art performance across all of these benchmarks, and real-world experiments further demonstrate its strong performance compared to existing baselines.

### Strengths
1.	The proposed dual-system framework is novel for the VLN task, offering a promising approach to handle high-level planning and low-level control separately while allowing joint optimization.
2.	The strong performance across multiple benchmarks demonstrates DualVLN’s remarkable capability and its potential to scale effectively with larger datasets.
3.	The dual-system design significantly enhances control frequency, which is crucial for real-world deployment.
4.	Real-world experiments conducted on wheeled, humanoid, and quadruped robots further highlight DualVLN’s strong cross-embodiment generalization ability.

### Weaknesses
1.	The authors should provide additional details about the ground-truth data collection process used to train the two systems.
2.	More methodological details should be included, particularly regarding the pixel-goal grounding component and the self-directed view adjustment mechanism.
3.	Ablation studies and further analyses on the self-directed view adjustment module should be provided to better understand its contribution and effectiveness.

### Questions
1.	How is visibility from the agent’s position measured when projecting trajectories onto 2D observations? The authors are encouraged to provide a detailed description of the procedure for generating pixel-grounding training samples.
2.	The authors should offer more explanation on the design of the self-directed view adjustment module, including how this ability is trained and its overall effectiveness.
3.	In the ablation study section, Line 425 references Table X, which does not exist in the article. In fact, Figure 6 illustrates the role of different goal representations, but it is not cited anywhere in the text. The authors are advised to clarify and revise this part for better consistency and readability.

### Soundness
4

### Presentation
2

### Contribution
4
