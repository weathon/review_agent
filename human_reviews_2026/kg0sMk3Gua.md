# DORAEMON: Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 2, 6

## Abstract
Adaptive navigation in unfamiliar environments is crucial for household service robots but remains challenging due to the need for both low-level path planning and high-level scene understanding. While recent vision-language model (VLM) based zero-shot approaches reduce dependence on prior maps and scene-specific training data, they face significant limitations: spatiotemporal discontinuity from discrete observations, unstructured memory representations, and insufficient task understanding leading to navigation failures. We propose DORAEMON (Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation), a novel cognitive-inspired framework consisting of Ventral and Dorsal Streams that mimics human navigation capabilities. The Dorsal Stream implements the Hierarchical Semantic-Spatial Fusion and Topology Map to handle spatiotemporal discontinuities, while the Ventral Stream combines RAG-VLM and Policy-VLM to improve decision-making. Our approach also develops Nav-Ensurance to ensure navigation safety and efficiency. We evaluate DORAEMON on the HM3D, MP3D, and GOAT datasets, where it achieves state-of-the-art performance on both success rate (SR) and success weighted by path length (SPL) metrics, significantly outperforming existing methods. We also introduce a new evaluation metric (AORI) to assess navigation intelligence better. Comprehensive experiments demonstrate DORAEMON's effectiveness in zero-shot autonomous navigation without requiring prior map building or pre-training.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces DORAEMON, a novel cognitive-inspired zero-shot navigation framework operating in unseen/unfamiliar environments.
The proposed system integrates two parallel pathways: a Dorsal Stream for spatial reasoning via a hierarchical semantic-spatial fusion network and topological memory, and a Ventral Stream that combines two VLM modules, CoDe-VLM for task decomposition and Exec-VLM for decision execution guided by chain-of-thought reasoning.
In addition, a Nav-Ensurance module is designed to ensure robustness through multimodal struck detection, context-aware escape strategies, and adaptive precision navigation.
A new metric, AORI, is proposed to quantify the exploration redundancy. 
Experiments conducted on multiple datasets demonstrate that DORAEMON achieves state-of-the-art performance in both SR and SPL without pretraining or scene-specific adaptations.
Overall, the paper offers an interdisciplinary attempt to merge cognitive science principles with VLMs for autonomous navigation

### Strengths
The paper draws inspiration from the “Decentralized Ontology” in cognitive science and successfully instantiates it in a dual-stream framework. It has the potential to offer a structured yet flexible alternative to (traditional) VLM-based zero-shot approaches.
The decomposition into CoDe-VLM and Exec-VLM within the ventral stream is well-motivated, allowing explicit separation between semantic compilation and action execution.
The hierarchical semantic-spatial fusion unifies episodic observations and spatial priors through a topological map, mitigating temporal consistency.
The Nav-Ensurance system introduces practical reliability mechanisms (e.g., multimodal stuck detection) that try to address real-world navigation pitfalls. The AORI metric provides an interesting new point for evaluating route redundancy.
The model achieves state-of-the-art results across multiple datasets and remains effective when integrated with different backbone VLMs, demonstrating “plug and play” ability.

### Weaknesses
*Clarity on “Decentralized Ontology”*

While conceptually appealing, the notion of “decentralized ontology” remains under-formalized. It is not rigorously shown how this concept is represented in the graph structure or contributes quantitatively to generalization.

*Lack of temporal stability analysis*

Although the topological map incrementally updates nodes, no analysis is presented on long-term memory scalability or potential degradation (e.g., graph explosion, out-of-date information, stale nodes).

### Questions
How is “Decentralized Ontology” structurally encoded within DORAEMON? Can this concept be explained more or measured (e.g., via entropy or mutual information)?

Does the CoDe-VLM dynamically update the knowledge graph during exploration, or is it compiled only once per task? If dynamic, how does it synchronize with the Dorsal Stream’s evolving topology?

In the Nav-Ensurance module, what is the false-positive rate of stuck detection in narrow or cluttered environments? Has the system been evaluated for recovery reliability?

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
This work proposes a zero-shot navigation framework composed of a ventral and dorsal stream, inspired by decentralized ontology principles and cognitive neuroscience. The approach integrates a hierarchical topological memory with VLM-based task reasoning and includes additional mechanisms for stuck detection and escape. Experiments show improved performance on several standard indoor navigation benchmarks, along with a newly introduced metric aimed at measuring exploration redundancy. The paper also provides ablations and a brief sim-to-real demonstration to support the system’s design choices.

### Strengths
- The paper presents a well-engineered navigation framework that combines spatial memory and VLM-based semantic reasoning into a cohesive system.

- Strong empirical results are demonstrated on multiple established datasets, showing improvements over recent zero-shot and end-to-end baselines.

- The hierarchical memory design is intuitive and may help support longer-horizon reasoning.

- Practical concerns such as getting stuck or failing to stop correctly are explicitly addressed through the Nav-Ensurance module.

- The proposed AORI metric reflects an interesting attempt to evaluate exploration redundancy beyond standard SR/SPL metrics.

- A sim-to-real example is included, suggesting potential for deployment beyond simulation.

### Weaknesses
- **Weak novelty; overly conceptual framing**
The cognitive metaphors (decentralized ontology, ventral/dorsal streams) provide an interesting narrative direction, but their influence on the actual technical design remains unclear. Many of the core components, such as hierarchical spatial memory, semantic graph–based task representation, and VLM-guided action reasoning, are already present in recent zero-shot navigation systems. It would strengthen the contribution to more explicitly identify what aspects of the architecture or behaviors are genuinely enabled by the decentralized ontology perspective, beyond established techniques like graph-structured memory and chain-of-thought reasoning.

- **Performance attribution is unclear**
Much of the performance gain might stem from the power of the underlying VLM. More targeted ablations that quantify the incremental contribution of each stream, or evaluations using a wider range of foundation models, would help isolate where the improvements truly come from.

- **Fairness of action-space comparison:**
The proposed agent uses continuous actions that cover significantly longer distances per step than the discrete baselines. Even with post-hoc “step conversion,” the agent effectively has greater traversal capacity per episode, which can inflate SR/SPL. 

- **On the use of AORI**
The idea of analyzing exploration redundancy is promising. However, AORI assumes a fully static environment, whereas realistic navigation may require returning to previously visited areas due to changes in scene configuration or visibility. Adjusting the metric to better capture navigation success in dynamic contexts would make it more broadly applicable.

- **“End-to-end” branding seem misleading**
The current system incorporates many hand-designed components, making it less end-to-end than suggested. Positioning the work more as a flexible integration framework, or demonstrating pathways toward learning more of these components jointly, could help align expectations.

### Questions
- Could the authors justify that improvements in SR/SPL are not influenced by differences in traversal capacity per episode?
- How might AORI be adapted or complemented to remain informative in dynamic environments where beneficial revisiting is necessary due to changes in object visibility or scene configuration?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents DORAEMON, a decentralized, ontology-aware framework for zero-shot embodied navigation. It draws inspiration from human cognitive neuroscience, introducing two interacting components: a Ventral Stream (for semantic reasoning via CoDe-VLM and Exec-VLM) and a Dorsal Stream (for spatial processing via Hierarchical Semantic-Spatial Fusion and a Topological Map). The framework is complemented by a Nav-Ensurance module for safety and a new evaluation metric (AORI) to measure spatial redundancy.

### Strengths
The dual-stream structure (ventral/dorsal) offers a biologically motivated yet technically meaningful decomposition of perception and reasoning. The integration of semantic graphs and spatial topology is novel for zero-shot navigation.

The AORI metric provides a more nuanced measurement of spatial redundancy than SPL/SR, aligning well with embodied intelligence evaluation trends.

### Weaknesses
Distilling human reasoning ability into navigation tasks is not a new direction, as many works have already explored this idea through various approaches such as NavGPT[1], NavigateDiff[2], and Navid[3]. In addition, numerous studies have leveraged large language models for navigation, which the authors should review more comprehensively and discuss in greater depth.

Some parts of the paper are written rather roughly. For example, the **caption of Figure 3** should include a detailed explanation of the overall workflow, and the **Methods** section is somewhat unclear—it requires multiple readings to fully understand.

The proposed framework is quite complex, yet the experiments focus only on relatively simple **object navigation tasks**. It would be valuable to include **Vision-Language Navigation (VLN)** and **Image Navigation** tasks to better demonstrate the framework’s generality.

Moreover, for the current **object navigation** benchmarks, strong reasoning ability is generally not required, so the reported improvements may not convincingly validate the effectiveness of the framework. The **real-world experiments** also lack sufficient implementation details, and the scene shown in **Figure 5** appears too simple—basic object navigation methods could achieve similar results. I would encourage the authors to include **more challenging tasks** to more convincingly demonstrate the robustness and reasoning capability of the proposed pipeline.


[1] NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models

[2] NavigateDiff: Visual Predictors are Zero-Shot Navigation Assistants

[3] NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation

### Questions
See Weakness.

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
This paper proposes DORAEMON, a cognitive-inspired, end-to-end, zero-shot navigation framework for household service robots to address the challenges of adaptive navigation in unfamiliar environments—such as spatiotemporal discontinuity from discrete observations, unstructured memory, and insufficient task understanding faced by existing vision-language model (VLM)-based approaches. Mimicking human navigation, DORAEMON comprises two core streams: the Dorsal Stream, which handles spatial information via a Topology Map and Hierarchical Semantic-Spatial Fusion to resolve spatiotemporal discontinuities, and the Ventral Stream, which enhances task understanding and decision-making through CoDe-VLM (converting tasks into structured knowledge graphs) and Exec-VLM (graph-based reasoning for action selection). It also integrates the Nav-Ensurance system for stuck detection, context-aware escape, and precision navigation, and introduces a new evaluation metric AORI to quantify exploration efficiency by penalizing redundant paths. Evaluations on HM3Dv1, HM3Dv2, MP3D, and GOAT datasets show DORAEMON achieves state-of-the-art performance on Success Rate (SR) and Success weighted by Path Length (SPL), outperforming existing end-to-end and non-end-to-end baselines, with valid sim-to-real generalization; the code is publicly available to support reproducibility.

### Strengths
### 1. Innovative Cognitive-Inspired Architecture  
DORAEMON mimics human "dorsal (spatial)-ventral (semantic)" dual pathways to address core VLM navigation flaws: the Dorsal Stream resolves spatiotemporal discontinuity via a hierarchical Topology Map and semantic-spatial fusion, while the Ventral Stream (CoDe-VLM + Exec-VLM) converts unstructured tasks into knowledge graphs for interpretable reasoning—outperforming baselines with fragmented memory.  


### 2. End-to-End & Zero-Shot Capability  
It enables navigation in unseen environments without prior map building or task-specific pre-training. Unlike non-end-to-end methods relying on discrete actions, its continuous action space (polar coordinates) ensures smoother paths, and it maintains compatibility with any VLM (e.g., Gemini, Qwen) for flexibility.  

### 3. Robust Safety & Efficiency Guarantees  
The Nav-Ensurance system detects stuck states (via progress efficiency/rotation-translation ratio) and applies context-aware escapes (e.g., large turns for dead ends). The novel AORI metric quantifies redundant exploration, while precision navigation mode (scaled action distance near targets) boosts final positioning accuracy.  


### 4. Strong Experimental & Generalization Performance  
It achieves SOTA on SR/SPL across HM3Dv1/v2, MP3D, and GOAT datasets, outperforming end-to-end (e.g., VLMNav) and non-end-to-end baselines. Sim-to-real tests in novel offices validate its generalization, with public code ensuring reproducibility.

### Weaknesses
### 1. Limited Novelty in Topological Map Representation  
The work’s use of a topological map for spatial memory lacks strong originality, as topological structures for robot navigation have been extensively explored in prior studies. For instance, recent work (e.g., Mem4Nav, TopoNav) already integrated semantic topological graphs with spatial memory to address navigation continuity. While this paper optimizes node updating/merging, these are incremental tweaks to existing topological paradigms, not breakthrough innovations in spatial representation.  


### 2. Heavy Dependence on External Strong Models  
Its performance heavily relies on high-capacity external tools (e.g., GPT-4o for reasoning, GLEE for segmentation). Ablations show removing these models causes sharp SR/SPL drops, suggesting the framework’s success may stem more from external models’ power than its own core design. This also limits applicability in resource-constrained scenarios where such models are unavailable.  


### 3. Inadequate Dynamic/Large-Scale Scenario Validation  
Experiments are mostly in static simulation datasets (HM3D, MP3D) with no testing on dynamic scenes (e.g., moving obstacles, relocated targets). Real-robot tests only cover simple small-scale tasks, failing to validate robustness in practical long-horizon or large-scale environments (e.g., multi-floor buildings).  


### 4. Ambiguous Topological Node Rules  
The logic for creating/merging topological nodes is under-specified. Key details like "waypoint selection criteria" (distance? semantics?) and "merge thresholds" (spatial proximity metrics?) are unclear. This ambiguity may lead to inconsistent node generation and makes it hard to replicate or extend the work.

### Questions
1. Your framework depends heavily on high-capacity external models like GPT-4o and GLEE. How would its performance (e.g., SR, SPL) change when using lighter, open-source alternatives (e.g., LLaVA-1.5) for resource-constrained scenarios?  

2. The experiments only cover static environments (HM3D, MP3D) with no dynamic elements (e.g., moving obstacles). Does your topological map have real-time update mechanisms to adapt to environmental changes, and if so, how would you quantify this adaptability?  

3. The logic for topological node creation/merging lacks specific details (e.g., distance thresholds, semantic similarity criteria). Could you specify these key parameters and explain how they were determined or validated?

### Soundness
3

### Presentation
3

### Contribution
2
