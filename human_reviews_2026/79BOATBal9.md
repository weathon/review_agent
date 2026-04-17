# Planning with an Embodied Learnable Memory

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
We develop a novel memory representation for embodied planning models performing long-horizon mobile manipulation in dynamic, large-scale indoor environments. Prior memory representations fall short in this setting, as they struggle with object movements, suffer from computational deficiencies, and often depend on the heuristic integration of multiple models. To overcome these limitations, we present the Embodied Perception Memory (EPM), a learnable memory designed for embodied planning. EPM is implemented as a unified Vision-Language Model (VLM) that uses egocentric vision to maintain and update a textual environment representation. We further introduce two complementary methods for training planners to leverage the EPM: an imitation strategy that uses human trajectories for natural exploration and interaction, and a novel reinforcement learning approach, Dynamic Difficulty-Aware Fine-Tuning (DDAFT), which improves planning performance via difficulty-aware exploration. Our memory representation, when integrated with our planning training methods, leads to significant improvements on planning tasks, showing up to a 55% increase in success rate on the PARTNR benchmark compared to strong baselines. Also, our planning method outperforms these baselines even when they have access to groundtruth perception.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Embodied Perception Memory (EPM), a learnable memory representation for long-horizon embodied planning in dynamic indoor environments. This representation enables end-to-end scene information operations (addition, removal, update), adopts object-centric encoding (3D coordinates + textual state/context descriptions), and eliminates explicit task-based queries by generating LLM-compatible textual scene representations. In addition, a two-stage tranining recipe is proposed to finetune LLM planners to leverage the EPM: first train the model on human demostration data to improve exploration and task execution, then leverage a novel online RL-based approach, named Dynamic Difficulty-Aware Fine-Tuning (DDAFT), to optimize planning via online sampling of difficult instructions.

### Strengths
1. The writing is clear and logically coherent, ensuring easy readability for readers.
2. The proposed memory representation is well-suited for tracking object states in dynamic indoor environments, and it exhibits several key advantages:
 (1) Easy maintenance: Unlike complex scene graphs that detail the relative relationships between all objects, EPM adopts a simpler spatial representation—one that focuses only on the relationships between dynamic objects and infrequently movable furniture.
 (2) Lightweight design: Compared with explicit semantic mapping techniques, EPM eliminates the need to perform segmentation, open-vocabulary encoding, and dynamic map updates.
3. The performance results are robust, particularly when planners are paired with the proposed memory under learned perception settings.

### Weaknesses
1. In Section 4.2, the use of human demonstration data for training is introduced; however, the description of how to acquire human demonstration data based on tasks and memory lacks clarity.
2. How is the instance association problem in the perception process addressed? It appears that the same instance may be recorded as multiple instances due to perception and localization errors. When applied to real world, has consideration been given to adding an operation type for merging entities?
3. In the inference phase, how is the information about which specific room each piece of furniture is located in (stored in memory) acquired? In the planning example in Appendix A.4, the descriptive information of the furniture is inconsistent with the content described in the Entity List and Embodied Perception Memory Input in Appendix A.2.

### Questions
See weakness above.

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
This paper proposes an embodied planning framework that introduces a *dynamically learnable memory* to assist high-level task planning. The key idea is to maintain a textual memory that records the positions and states of currently observed objects, which is then injected into the input context of a large language model (LLM) planner. The planner conditions on this memory to produce high-level action decisions such as *Navigate*, *Pick*, and *Place*, allowing the system to reason about dynamic environments through language-conditioned planning.

### Strengths
- Incorporating a memory mechanism into embodied planning is conceptually valuable. It aligns with the broader goal of enhancing situational awareness and long-horizon reasoning in embodied agents.
- The proposed pipeline is relatively complete, combining human demonstration data and DDAFT to improve the planning capability of the LLM. 
- The authors provide detailed implementation descriptions, which improve the reproducibility of the proposed approach.

### Weaknesses
The overall method is straightforward and lacks sufficient evidence to justify the necessity of the proposed memory mechanism. Essentially, the “memory” represents the textual description of observed object positions, which is appended as additional prompt context for the LLM planner to make decisions. This design raises questions about its true effectiveness: a Vision-Language Model (VLM) could directly perceive and infer object locations from images without requiring a separate process to extract and feed this information in textual form.

In the experiments, the authors compare models of different scales (7B and 70B) and their fine-tuned variants. The results show that supervised fine-tuning and DDAFT (a DPO-like reward fine-tuning method) indeed enhance model performance, but such improvement mainly reflects the benefit of additional data and optimization rather than the specific contribution of the memory mechanism. From the presented evidence, it remains unclear why this kind of dynamic memory is indispensable for embodied task planning.

Moreover, the evaluation is limited to the PARTNR benchmark. While this environment is a reasonable testbed, the empirical validation would be more convincing if the authors could extend experiments to other widely used embodied task planning benchmarks such as **ALFWorld** or **EmbodiedBench**. Including results from multiple environments would help demonstrate the generality and robustness of the proposed approach.

### Questions
1. **Why use an LLM planner instead of a VLM?**
   Since the memory content is purely a textual description of visual states, a Vision-Language Model could directly access equivalent spatial information from images. What is the rationale for choosing a text-only LLM over a multimodal model that already encodes spatial and visual grounding?
2. **Can you compare LLM w/ memory vs. VLM w/o memory vs. VLM w/ memory for the task planning?**
   Such a comparison would more convincingly isolate the contribution of the memory mechanism and clarify whether the observed gains come from the memory design 
3. **Can you include additional benchmarks?**
   Evaluating on environments like **ALFWorld** or **EmbodiedBench** would strengthen the claim that the proposed approach generalizes across different embodied planning settings.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the Embodied Perception Memory (EPM), a textual list of entities (objects and furniture) and their relationships that uses a Vision-Language Model (VLM) to maintain and update. For planning, the authors employ a two-stage training strategy: (1) Imitation learning on planning trajectorirs derived from human demonstrations, which are post-processed to incorporate perception-specific exploration and robustness to memory errors; (2) A reinforcement fine-tuning method named Dynamic Difficulty-Aware Fine-Tuning (DDAFT), which adapts the DART-Math idea to the planning task during online policy improvement. The primary experimental validation on embodied planning is conducted on the PARTNR benchmark.

### Strengths
- The EPM provides a unified and interpretable text-based memory that seamlessly integrates with LLM planners, avoiding the need for complex querying mechanisms used in some prior works (e.g., feature matching in point clouds).

- The proposed DDAFT method effectively improves planning performance by dynamically focusing on difficult tasks during training.

### Weaknesses
- The idea of using a unified VLM to convert visual observations into a textual memory for planning has been explored in prior works [1-3].
- The experiments are conducted only on the PARTNR validation set. It is unclear how the method would perform on the test set, which is important for assessing generalization and fairness in benchmarking.
- The training data for EPM relies on simulator-privileged information (e.g., ground-truth object states and poses), which limits the scalability and applicability of the method to real-world settings where such information is unavailable.

[1] Context-aware planning and environment-aware memory for instruction following embodied agents, ICCV 2023   
[2] STMA: A Spatio-Temporal Memory Agent for Long-Horizon Embodied Task Planning, arxiv 2025.02    
[3] KARMA: Augmenting Embodied AI Agents with Long-and-short Term Memory Systems, ICRA 2025

### Questions
1. How does the system handle multi-object disambiguation? For example, if there are four apples and five milk bottles in the environment, how does EPM distinguish and track each instance in its textual representation?
2. For the Spot-Indoor dataset, are the annotations for adding/removing objects manually labeled, or are they constructed using the method described in Section 4.2 (i.e., via simulation replay and heuristics)?

### Soundness
2

### Presentation
3

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
This paper introduces Embodied Perception Memory (EPM), a unified vision–language model based memory system for embodied planning that represents the environment as a textual, object-centric world model updated from egocentric observations. EPM dynamically tracks objects and changes in long-horizon and enables seamless integration with LLM-based planners. To train planners to effectively leverage this representation, the authors propose two complementary methods: (i) imitation learning from human demonstration trajectories and (ii) Dynamic Difficulty-Aware Fine-Tuning (DDAFT), a value-function-free reinforcement learning algorithm that prioritizes training on harder tasks by sampling episodes based on failure rates. They evaluate EPM both as a perception module in isolation and as part of full embodied planning tasks. In particular, EPM outperforms existing memory architectures such as DynaMem and also surpasses LLM-based planners trained solely on the PARTNR dataset.

### Strengths
1. The writing is generally clear, with strong motivation and context provided for the problem setting.

2. The paper introduces a novel memory mechanism for embodied planning which is a conceptual advance over prior approaches that rely on separate perception modules, graph-based representations, or retrieval interfaces. The integration of a single VLM that simultaneously maintains, updates and exposes a textual world representation represents a meaningful rethinking of memory design for embodied agents.

3. The proposed approach tackles an increasingly important problem in embodied A, long-horizon planning in dynamic environments. The ability to maintain and evolve a textual world model from egocentric observations, while integrating seamlessly with language-based planners, has direct implications for future real-world embodied AI systems.

4. By unifying perception and memory within a single model, the approach reduces reliance on heavy symbolic querying or multi-module pipelines, which can be brittle and computationally expensive. This end-to-end design offers a promising direction toward scalable, real-time embodied planning systems.

5. The paper evaluates both perception and planning. The perception system is tested on both simulaton and real-world egocentric data, demonstrating promising transfer beyond simulation. Planning performance is assessed in challenging simulated environments, where EPM outperforms prior memory architectures (e.g., DynaMem) and LLM planners trained solely on PARTNR.

### Weaknesses
1. Real-world perception results do not show clear advantage over DynaMem: While the perception module is evaluated on real-world egocentric data, the reported results do not show a clear performance improvement over DynaMem in this setting. The authors attribute this to out-of-distribution objects that appear in real environments but are absent in simulation training. However, this highlights a practical limitation of the current approach: the system’s ability to generalize in open-world settings remains uncertain. It would be valuable to explore whether fine-tuning EPM on diverse real-world embodied datasets could reduce this gap. For example, large-scale egocentric datasets such as Ego4D, BEHAVIOR-1K real-world captures, or RT-1/RT-2 robot interaction datasets contain a wide variety of real objects and household interactions that could support domain adaptation. 

2. No real-world validation of the planning pipeline: While the perception module is evaluated on real-world egocentric data, the planning results are restricted to simulation. Consequently, it remains uncertain how well the complete EPM-based planning pipeline would transfer to physical robots, where challenges such as sensor noise, actuation inaccuracies, and occlusions are more pronounced. Even small-scale real-robot experiments, or analysis demonstrating robustness to real-world observation and control noise, would significantly strengthen the claim that the method scales beyond simulation.

3. Incomplete comparison to recent memory systems and lack of computational analysis: While the paper compares against DynaMem, the evaluation omits other recent memory representations that also maintain object-centric world models, such as ConceptFusion, ConceptGraph etc. which would provide a more complete picture of performance relative to the state of the art in embodied memory systems. Additionally, although success rates and perception F1 scores are reported, the paper does not quantify computational efficiency or update overhead, which is central to the claimed scalability benefits. Reporting metrics such as time per memory update, insertion cost for new objects, query latency, and overall inference load per planning step would clarify the practical efficiency gains of EPM over modular graph-based or retrieval-based memory architectures.

4. Limitations of a text-only memory representation: A purely textual memory, while clean and LLM-friendly, may fail to capture fine-grained perceptual details needed for precise manipulation. Object shape, texture, grasp points, and continuous spatial relationships are difficult to encode in text, and tasks requiring such information may exceed the capabilities of the current representation. Although the authors acknowledge hybrid memories as future work, the paper does not explore mechanisms to integrate visual or geometric cues, limiting applicability to more physically detailed settings.

5. DDAFT lacks formal justification and comparison to standard RL fine-tuning techniques: While DDAFT intuitively prioritizes difficult trajectories, the method is introduced without strong grounding, and its advantages over established RL-from-feedback or policy fine-tuning approaches remain unclear. In particular, comparisons to more standard baselines such as RFT-style updates, GRPO, or PPO-based fine-tuning would help determine whether DDAFT offers a principled improvement or functions primarily as a heuristic curriculum strategy. Without these comparisons, it is difficult to isolate the contribution of the proposed RL component.

6. Limited analysis of failure cases and memory degradation over long horizons: The paper reports aggregate success metrics but offers limited qualitative analysis of failure modes. In particular, it remains unclear how EPM behaves when memory errors accumulate over long sequences (e.g., misplaced objects, overwritten states, hallucinated object persistence). Without studying memory drift, error correction, or mechanisms for uncertainty estimation, it is difficult to assess reliability in extended, multi-room or multi-task settings.

7. Ambiguity in update policies and memory maintenance strategy: EPM continuously rewrites textual memory, but the rules governing updates (e.g., when to delete stale information, how to handle conflicting observations, how to resolve object identity uncertainty) are not fully articulated. For instance, if an object is temporarily occluded or repositioned off-camera, the system may incorrectly maintain outdated state. An explicit treatment of object permanence, confidence tracking, or uncertainty-aware updates would strengthen the method’s robustness claims.

8. Task and domain diversity is limited: Although the results on PARTNR are compelling, the generality of EPM to broader embodied domains such as navigation, deformable object manipulation, tool use, or embodied household tasks in other environments like BEHAVIOR-1K or Habitat 2.0 is not established. Without cross-domain evaluation, the system’s scalability across diverse embodied tasks remains open.

### Questions
1. How do you plan to evaluate the complete EPM planning pipeline in real-world robotic settings, and what challenges do you anticipate in transferring beyond simulation?

2. Have you considered fine-tuning EPM on large-scale real-world embodied datasets (e.g., Ego4D, RT-1/RT-2, BEHAVIOR-1K) to improve generalization for real objects not present in simulation?

3. How does EPM handle conflicting observations, object occlusions, or reappearances? Is there an object permanence or confidence mechanism to avoid stale or incorrect memory states?

4. Can you provide memory drift or long-horizon error analyses? How does the textual memory behave when small memory errors compound over long sequences?

5. What motivates DDAFT over more established RL fine-tuning methods, and have you compared against RFT, PPO-based finetuning, or GRPO?

6. Could you report computational metrics such as update latency, memory insertion cost, and query time, to substantiate the claimed scalability benefits?

7. Why were recent concept-centric memory systems (e.g., ConceptFusion, ConceptGraph) not included as baselines, and how do you expect EPM to compare conceptually and empirically?

8. Do you anticipate difficulty scaling EPM to domains requiring fine-grained perceptual grounding (e.g., grasp point reasoning, deformable objects, tool use), and how might hybrid visual-textual memory help?

9. Have you explored mechanisms to reset, compress, or prune memory to avoid uncontrolled growth or hallucinated persistence in long-horizon tasks?

### Soundness
3

### Presentation
3

### Contribution
2
