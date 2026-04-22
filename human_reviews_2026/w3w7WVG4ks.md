# Building spatial world models from sparse transitional episodic memories

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
Many animals possess a remarkable capacity to rapidly construct flexible cognitive maps of their environments. These maps are crucial for ethologically relevant behaviors such as navigation, exploration, and planning. Existing computational models typically require long sequential trajectories to build accurate maps, but neuroscience evidence suggests maps can also arise from integrating disjoint experiences governed by consistent spatial rules. We introduce the Episodic Spatial World Model (ESWM), a novel framework that constructs spatial maps from sparse, disjoint episodic memories. Across environments of varying complexity, ESWM predicts unobserved transitions from minimal experience, and the geometry of its latent space aligns with that of the environment. Because it operates on episodic memories that can be independently stored and updated, ESWM is inherently adaptive, enabling rapid adjustment to environmental changes. Furthermore, we demonstrate that ESWM readily enables near-optimal strategies for exploring novel environments and navigating between arbitrary points, all without the need for additional training. Our work demonstrates how neuroscience-inspired principles of episodic memory can advance the development of more flexible and generalizable world models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Paper proposes a novel representation of spatial memory for mapping and navigation. A transformer-based model is trained via mask-prediction self-supervised method to predict the unseen region/path. Visualization of the neural network with ISOMAP projection shows that it is able to learn to infer the relative position of disjoint regions. The framework is further extended to the exploration task and goal-seeking task. It can also effectively handle changing environment for the goal-seeking task.

### Strengths
The inspiration from neuroscience has been a common thread in the major breakthroughs in computational science. In particular, the neural network is proven to be instructional in how computational theories are advanced through such inspiration. While this paper does not directly model brain episodic memory, the concept being borrowed/inspired shows good potential in the application of mapping and navigation.

One of the major challenges facing machine learning and AI is the effective and efficient representation and modelling of memory within/coupled with neural network training. This work may lead to some interesting ideas for other researchers. S1. Claims are supported by the strong experimental results for the exploration tasks and the goal-seeking task (without global map).

S2. Proposed methods are sensible and applicable for the problem.

S3. Experimental designs and analyses are sound and valid. There are sufficient experiments and ablation studies to support the central claim of the paper. There are additional experiments to compare transformer variants of the model, ESWM-T, against transformer-based such as TEM-T.

### Weaknesses
Minor: The proposed solution may not generalized to continuous environments. However, given the highly theorical framing of the paper, this is not a major concern.

### Questions
1. Does the authors have specific plans for generalizing their methods for real-world environment, i.e. physical embodiment, for their future work?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces the Episodic Spatial World Model (ESWM), a framework that
constructs spatial representations from sparse, disjoint episodic memories. The
model is meta-trained to predict unseen transitions from minimal one-step
transition memories across diverse environments. The authors demonstrate that
ESWM forms geometric latent representations, enables zero-shot exploration and
navigation, and adapts rapidly to environmental changes.

### Strengths
1. Well-documented implementation: The paper provides comprehensive technical details including pseudocode (Algorithms 1-4), extensive appendices, and thorough experimental setup descriptions. This supports reproducibility.
2. Strong downstream performance: ESWM demonstrates impressive zero-shot capabilities on exploration (96.48% of oracle performance) and navigation tasks (96.8% success rate, 99.2% path optimality), outperforming the task-specific EPN baseline by substantial margins.
3. Interesting architectural insights: The finding that transformer attention mechanisms are critical for this task (while LSTM and Mamba struggle in Open Arena) provides valuable insights about the role of content-addressable memory in spatial reasoning.
4. Adaptability demonstration: The ability to handle environmental changes by simply updating memory banks (93% success rate with new obstacles) is a genuine advantage over weight-based world models.
5. Emergence of spatial structure: The spontaneous emergence of environment-like manifolds in latent space (Figures 3, 9-13) is compelling, especially the model's ability to infer spatial relationships from obstacle observations and boundary shapes.

### Weaknesses
## Major Concerns
1. Insufficient Justification of Core Premise.
The abstract and introduction claim that animals/humans build spatial maps from "disjoint experiences governed by consistent spatial rules," but no neuroscience evidence is provided to support this specific claim. The cited disruption studies (lesions, amnesia) show that MTL is important for spatial cognition and episodic memory, but don't demonstrate that humans actually construct maps from genuinely disjoint, fragmented experiences.
In reality, human and animal spatial experience is largely continuous and spatiotemporal. We don't teleport randomly through environments. The authors should either:
  * Provide evidence that biological systems actually face and solve this "disjoint memory" problem, or
  * Reframe their contribution as addressing a computational/AI challenge rather than claiming biological inspiration for this specific aspect

2. Unclear Relationship to Prior Work on Transition Systems.
The memory bank structure (graph of one-step transitions) closely resembles successor representations and transition graphs that have been extensively studied in both:
  * Hippocampal/entorhinal modeling: Stachenfeld et al. (2017), and related work on spatial transition representations in grid cells
  * General cognitive maps: Work from Behrens' lab on relational structure

The paper mentions Stachenfeld et al. only briefly and doesn't adequately distinguish ESWM's contribution from this and other related substantial body of work. How does ESWM's approach differ conceptually from learning successor representations or other transition models? This needs clear articulation.

3. Limited Scope and Generalization Concerns.
Environment Scale: All tested environments are very small (19-37 locations). With such limited state spaces and the model seeing many different configurations during meta-training, it's unclear whether ESWM is:
  * Learning generalizable spatial reasoning principles, or
  * Memorizing patterns across the finite set of possible small-scale configurations

The one generalization test (size-19 -> size-37) shows substantial performance drops (source->action->end accuracy: 59%->90%->55%), suggesting the model may not scale well.
Egocentric Setting: While MiniGrid experiments (Figure 8) show promise, they're presented quite briefly. Given that real-world spatial cognition is fundamentally egocentric, this deserves more thorough investigation.
Why hexagonal grids? The choice of hexagonal structure is never justified. This seems arbitrary and limits comparability with standard benchmarks.

4. Missing Statistical Rigor.
Many key results lack error bars or significance tests:
  * Figure 2: Training curves show variance bands, but test results don't
  * Figure 4a: No confidence intervals on entropy vs. path length
  * Figure 5: Some comparisons report significance, others don't
  * Exploration and navigation comparisons need statistical testing across multiple seeds
For a model claiming to learn generalizable principles, demonstrating statistical reliability is essential.

5. Insufficient Ablation Studies.
The paper would benefit from:
  * Ablating the "minimality" constraint more thoroughly: Figure 7 shows non-minimal banks work, but how does performance scale with memory bank size? Is there an optimal density?
  * Analyzing what makes transformers work here: Beyond showing LSTM/Mamba fail, what properties of self-attention are critical? Number of heads? Depth? Context length?
  * Testing robustness to noise: Real-world transitions are noisy—how sensitive is ESWM to observation or transition noise?


## Minor Concerns
1. Conceptual and Framing Issues.
Section 2.2's key question: "Can spatial maps also emerge in general-purpose models not explicitly trained for navigation?"
This has essentially been answered affirmatively by prior work (TEM itself, Behrens lab work, emergence in language models). The authors should clarify what new dimension their work adds to this question.

Section 2.1: The first paragraph of Related Work feels unfocused, jumping between human cognition, RL, and vision without clear connections.

2. Technical Clarifications Needed.
Graph theory terminology (Section 3.1): Please verify that "disjoint" is used consistently with standard graph theory definitions. In particular, "disjoint" typically refers to vertex or edge-disjoint subgraphs, but here seems to mean "non-sequential transitions". If that's the case, please state this somewhere.

"I don't know" classification: The handling of unobservable regions (17% of queries in Random Wall) is clever, but how is the threshold for outputting "I don't know" determined? Is it learned or hand-tuned?



## Minor Comments

* Line 124-125: "TEM factorizes sensory and structural information". Clarify how ESWM's approach differs
* Figure 1c: The masking procedure is clear, but consider showing an example with actual states/actions
* Section 4.4: The "zig-zag exploration strategy" is interesting. Can you characterize this more formally?
* Appendix B.2: The guided imagination approach using latent space geometry is valuable and could be elevated to the main paper

### Questions
1. Can you provide evidence that biological systems construct spatial maps from genuinely disjoint (not just sparsely sampled) experiences?
2. How does your approach differ fundamentally from learning successor representations or other graph-based transition models? What unique capability does ESWM provide?
3. Have you tested on substantially larger environments (100+ states)? What is the scaling behavior?
4. Why is transformer attention specifically required? Have you tried other relational reasoning architectures (e.g., Graph Neural Networks, which seem natural for transition graphs)?
5. Can you clarify the relationship between your work and existing transition system models, particularly for grid cells and hippocampal representations?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the Episodic Spatial World Model (ESWM), a framework for building coherent spatial world models from **sparse and disjoint one-step transitional memories**. Inspired by the dual role of the Medial Temporal Lobe (MTL) in processing both episodic memory and spatial information, ESWM infers the underlying structure of an environment from a 'memory bank'—a set of independent transitions ($s_s​,a,s_e$​)—unlike prior models that rely on continuous trajectories.

The authors demonstrate the efficacy of ESWM through experiments in various grid-world environments (Open Arena, Random Wall, and MiniGrid). The results show that ESWM achieves higher prediction accuracy than the sequence-based model, TEM-T, and exhibits strong generalization to environments with unseen structures. Through ISOMAP visualizations, the authors reveal that ESWM's latent space forms a geometric map that accurately reflects the physical topology of the environment, including obstacles. This learned model enables high performance on downstream tasks such as zero-shot exploration and navigation. Furthermore, by decoupling memory from reasoning, the proposed architecture allows for rapid adaptation to environmental changes by simply updating the memory bank without retraining the model.

The paper's claims are well-supported by a detailed appendix that enhances reproducibility. It includes specifics on the model architecture, data generation pipeline, and additional results, such as the extension to the high-dimensional, vision-based MiniGrid environment (Figure 8), providing substantial evidence for the method's effectiveness.

### Strengths
- **Important Research Question and Interdisciplinary Connection:** This paper addresses a significant limitation of prior world models: their difficulty in inferring robust spatial knowledge from continuous sequences and adapting to environmental changes. The authors propose the Episodic Spatial World Model (ESWM), a novel approach inspired by the dual role of the Medial Temporal Lobe (MTL) in processing both episodic and spatial memory. This represents an excellent example of leveraging insights from neuroscience to advance machine learning research.
    
- **Comprehensive and In-depth Experiments:** The authors validate their claims through a diverse and rigorous set of experiments. By evaluating prediction accuracy in various environments (Open Arena, Random Wall), visualizing the latent space with ISOMAP, and demonstrating strong performance in zero-shot exploration and navigation, the paper not only shows that ESWM is effective for spatial reasoning tasks but also provides intuitive evidence for why it works.
    
- **High Reproducibility and Detailed Analysis:** The manuscript enhances the credibility and reproducibility of its results by providing meticulous details on model architectures and experimental setups in both the main paper and the appendix. The authors offer specific descriptions of the query generation (Appendix A.4), the ISOMAP visualization process (Appendix A.6), and provide extensive additional analyses (Appendix B) and visualizations (Appendix C), which lend strong support to their findings.

### Weaknesses
- **Insufficient Discussion of Closely Related Work and Unclear Novelty:** The core mechanism of the proposed method shares significant similarities with Generative Temporal Models with Spatial Memory (GTM-SM) [1], particularly the idea of leveraging one-step transitions to build a spatially-aware model. GTM-SM also demonstrated that its inferred state representations align with the true geometry and enable high-quality long-term predictions (e.g., Figure 2 in [1]). However, this paper lacks any discussion comparing ESWM with GTM-SM. I believe a thorough comparison is crucial for contextualizing the contributions of this work and clarifying its novelty. I strongly encourage the authors to incorporate this discussion.
    
- **Limited Evaluation Outside of Grid-World Environments:** While the authors test their model in several environments, all of them are grid-based. The discrete actions and cyclic nature of grid-worlds may provide implicit structural cues that simplify the task of learning spatial relationships. It is unclear whether ESWM's strong performance would generalize to more realistic, continuous environments (e.g., DMLab [2]) where such cues are absent. The lack of experiments in these settings makes it difficult to assess the true robustness and scalability of the proposed method.
    
- **Compliance with LLM Usage Policy:** The authors do not explicitly state whether or how they used Large Language Models (LLMs) in preparing their submission, as required by the ICLR 2026 Author Guide.
    
- **Minor Presentation Issues:**    
    - In Figure 1a (middle), the caption "Specific parts of the environment may be changed dynamically" could be misinterpreted as changes occurring mid-trajectory. It would be clearer to state that wall configurations are randomized across different episodes.
    - In Figure 2, the font size is too small, and the x-axis labels and ticks are occluded. In the caption, `Open Arena` should be $\texttt{Open Arena}$, and the formatting for ”-T” is inconsistent.
    - In Figure 3, the font sizes are too small for easy reading.

[1] Fraccaro, Marco, et al. "Generative temporal models with spatial memory for partially observed environments." _International conference on machine learning_. PMLR, 2018.

[2] Beattie, C., Leibo, J. Z., Teplyashin, D., Ward, T., Wainwright, M., Kuttler, H., Lefrancq, A., Green, S., Vald ¨ es, ´ V., Sadik, A., Schrittwieser, J., Anderson, K., York, S., Cant, M., Cain, A., Bolton, A., Gaffney, S., King, H., Hassabis, D., Legg, S., and Petersen, S. DeepMind Lab. CoRR, abs/1612.03801, 2016.

### Questions
- In the Figure 2 results, the performance of Mamba and LSTM architectures is notably poor in Open Arena but significantly better in Random Wall. This is counter-intuitive, as the Random Wall task seems more complex due to obstacle wall. Could the authors provide an explanation for this performance discrepancy?
    
- Regarding the experiments on non-minimal memory banks (lines 293-295), you show that ESWM-T performs well. Does this finding hold for other architectures like ESWM-Mamba and ESWM-LSTM, as well as for the TEM-T baseline? Additionally, could you elaborate on why a duplicated memory item might be problematic? Is it because it reduces the diversity of transitions sampled within a finite memory capacity?
    
- In Appendix A.4, you describe three different query types used for the Random Wall experiments (unseen, seen, and unsolvable). Could you please provide a performance breakdown for each query type? I suspect this data might help explain why ESWM-Mamba and LSTM perform better in the Random Wall setting.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a novel framework Episodic Spatial World Model (ESWM) for constructing cognitive maps from disjoint episodic memories, akin to human learning behavior. Instead of using long consecutive trajectories over the environment, ESWM uses a meta-learning algorithm across many environments to infer unseen transitions from a sparse memory bank of one-step experiences, learning spatial regularities on the way. Experiments demonstrate useful properties of the framework, including the ability to learn an internal representation that closely mirrors the environment layouts with minimal observations, and being sample-efficient and adaptive.

### Strengths
1. The empirical evaluation is thorough, rigorous, and insightful. The authors test ESWM on multiple tasks and environment variants (open arenas vs. maps with random walls/obstacles) and over multiple base models (transformers, LSTM, and Mamba) to assess its prediction accuracy, representation quality, and control performance. The set of experiments covers many insightful explorations of the ESWM behavior, including memory integration, navigation, adaptation to environment change, and scaling. Open Arena training also reveals nontrivial performance difference between the transformer ESWM and the other two variants. 
2. This work has substantial implications for both reinforcement learning and cognitive modeling. By showing that an agent can rapidly form a useful world model from a handful of potentially disjoint experiences, it highlights a pathway toward more generalizable and lifelong learning in AI. In practical terms, the ability to plan efficient exploration and navigation with less training could be impactful for robotics or embodied AI, where an agent might get only sparse observations of a new environment before needing to act.

### Weaknesses
1. The innovation is a bit limited, and there is no baseline outside of the ESWM model to compare with, making it hard to ground the model properties in context. Given the nature of the base models and the environment, the key innovation seems to be the training scheme over multiple memory banks. On the grounding, it would be interesting to see how the model performs against ordinary baselines or training schemes. For example, one may compare ESWM-T with ordinary transformers (with a single memory bank). Comparing ESWM-T with a model trained with longer trajectories would also provide insight into how sparse episodic memory effects learning. 
2. While the results are strong, the experiments are confined to relatively simple, synthetic environments. Additionally, the environments considered all follow consistent spatial rules. If an environment had more complex dynamics or non-grid connectivity, it’s unclear if the current approach would generalize as well (beyond scale).

### Questions
1. Since ESWM essentially learns to infer spatial structure, have you considered comparing it against a non-learned mapping baseline?
2. How crucial is it for the memory bank to form a perfect spanning tree of the environment? In cases where the episodic memories are incomplete, how might ESWM perform or adapt?

### Soundness
3

### Presentation
4

### Contribution
4
