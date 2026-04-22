# ChangingGrounding: 3D Visual Grounding in Changing Scenes

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 4, 8

## Abstract
Real-world robots localize objects from natural-language instructions while scenes around them keep changing. Yet most of the existing 3D visual grounding (3DVG) method still assumes a reconstructed and up-to-date point cloud, an assumption that forces costly re-scans and hinders deployment. We argue that 3DVG should be formulated as an active, memory-driven problem, and we introduce ChangingGrounding, the first benchmark that explicitly measures how well an agent can exploit past observations, explore only where needed, and still deliver precise 3D boxes in changing scenes. To set a strong reference point, we also propose Mem-ChangingGrounder, a zero-shot method for this task that marries cross-modal retrieval with lightweight multi-view fusion: it identifies the object type implied by the query, retrieves relevant memories to guide actions, then explores the target efficiently in the scene, falls back when previous operations are invalid, performs multi-view scanning of the target, and projects the fused evidence from multi-view scans to get accurate object bounding boxes. We evaluate different baselines on ChangingGrounding, and our Mem-ChangingGrounder achieves the highest localization accuracy while greatly reducing exploration cost. We hope this benchmark and method catalyze a shift toward practical, memory-centric 3DVG research for real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose ChangingGrounding, a new 3D visual grounding task that uses memory from past observations.
To evaluate the performance, the authors provide a dataset with corresponding evaluations.
The proposed zero-shot agent-based pipeline Mem-ChangingGrounder outperforms the baselines on this newly-proposed task.

### Strengths
1. The motivation to explore 3D visual grounding in changing scenes is novel and interesting.
2. The scope is comprehensive. The authors (i) propose the task, (ii) provide the dataset and evaluation for the new task, and (iii) compare against several baselines for this new task.

### Weaknesses
1. The motivation and task formulation in the introduction section require clearer presentation. In particular, the comparison shown in Figure 1 is not sufficiently explained. If the reviewer understand it correctly, it seems that current problem formulation is the previous setting (one full scan) + several new explored image demos, so that it could avoid repetitive rescan of entire scenes mentioned in the motivation (line 041). The authors should consider make better clarification compare with the previous setting.
2. The task formulation in Sec. 3.1 is not well-constructed. It's not clear what are in $S_p$ and $M_p$. Are $S_p$ only a set of images, or it also contains the old 3D object bounding boxes? Also, if the output $B$ is the predicted 3D bounding boxes, then the evaluation could only conducted on accuracy. Its not clear how to use bounding boxes to evaluate motion and action costs.
3. As the task if formulated as $⟨S_p, S_c, M_p, D_c⟩ → B$ (line 158), it is not clear why $S_p$ would help with the localization accuracy. It seems that the past memory could either be (i) the same as current views or (ii) different but has a wrong position. The correct location should be included in $S_c$. It's not intuitive why the accuracy of the Wandering Grounding baseline is worse (Table 2).
4. Other than the actions and memory costs, the proposed zero-shot pipeline heavily relies on VLM to do reasoning and inference. From table 10, it seems the inference cost is huge for the proposed method. How would the inference time compared with the traditional 3D method? The authors only show the action and motion cost comparison in Table 8. The inference time comparison is missing.
5. In the pipeline figure (Figure 3), it is not clear which images are from $S_p$ and which are from $S_c$. It is better to have the corresponding variable notations on the figure to help readers understand.

### Questions
Please refer to the weakness section.

During rebuttal, the reviewer would like to see the authors' responses of the weaknesses, including the following:

1. The further clarification on the motivation and problem formulation (Weakness 1, 2, 3).
2. Additional experimental results analysis (Weakness 3).
3. Discussion and results on inference time comparison (Weakness 4).
4. A clearer explanation of the pipeline (Weakness 5).

The reviewer will update the evaluation after seeing the clarification from the authors.

### Soundness
2

### Presentation
2

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
This paper introduces a new setting for 3D Visual Grounding (3DVG), termed Changing Grounding, and proposes a corresponding baseline method called Mem-ChangingGrounder (MCG). The motivation is that traditional 3DVG assumes full-scene reconstruction in advance, which is unrealistic in embodied environments. In contrast, embodied agents must actively explore, incrementally reconstruct local regions, and rely on memory for reasoning. The proposed MCG baseline adopts an agent-based design, integrating memory retrieval, VLM-based analysis, and GroundingDINO + SAM + projection to obtain 3D bounding boxes. Experiments demonstrate its overall effectiveness.

### Strengths
1. The motivation is highly valuable, the proposed active, memory-enabled, and incremental reconstruction setting aligns well with emerging embodied scenarios, especially in robotic navigation.
2. The paper contributes a Changing Grounding dataset and introduces new metrics that balance grounding accuracy and exploration cost, which are both practical and meaningful.
3. The proposed baseline is comprehensive and functional, showing promising performance advantages in experiments.

### Weaknesses
1. The baseline (MCG) is an agent-based pipeline with considerable complexity, and obtaining 3D bounding boxes via 2D SAM + projection may introduce significant localization errors compared to direct 3D detection.
2. Only an agent-based baseline is provided; there are no learnable or fine-tuned baselines (e.g., Transformer-based models such as DETR-style 3D grounding).
3. The memory mechanism may face scalability issues in long-horizon tasks, as the current design lacks memory compression or forgetting strategies, potentially leading to high computational overhead.
4. There is a minor ambiguity in the accuracy metric, when multiple candidate bounding boxes exist, it is unclear how accuracy is computed or which box is selected.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper first introduces a benchmark that evaluates how well a system can localize target semantic objects based on past memory, the current scene, and a language query. It then proposes a method that performs query classification, memory retrieval, fallback, and multi-view fusion to generate 3D bounding boxes. Experiments show that the proposed approach outperforms several baselines.

### Strengths
This paper tackles the problem of 3D visual grounding in dynamics scenes, a setting that more closely reflects real-world scenarios than the static environments commonly studied in prior work. Reasoning about changing scenes is indeed an interesting and important research direction.

### Weaknesses
1: The paper proposes to store all past frames as a database and feeds them directly into the VLM, which does not represent realistic memory management. Instead, this treats all past frames as in-context examples for the VLM rather than selective memory. A realistic setup should include mechanisms for memory updating, such as deciding which frames to retain or discard. Without this, the memory grows indefinitely and can causes efficiency issues. 

2: This paper’s definition of the “robot” is not realistic. If the robot is assumed to navigate indoor environments, collisions and geometric constraints should be considered. The current formulation of most cost does not generalize to real robots. If the work is intended as a pure vision paper, it would be better to remove or de-emphasize robot-related claims. 

3: The proposed approach relies heavily on VLMs. It’s unclear how robust the method is to noise or uncertainty in VLM outputs. 

4: The paper does not present and analyze failures cases.

### Questions
See the weakness section.

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
In this work, the authors present a data-structure centric method called ChangingGrounding for doing 3D visual grounding with large vision language models. They also present a related benchmark, where the agent needs to identify bounding boxes of objects in environment where scenes change with objects being added or removed. The method itself is inspired by biological memory – where the agent maintains a cache of past observations. Then, using the past observation cache and the query, the method uses one of two different scanner modules to collect more observations and add them to the memory cache. Finally, the 3D bounding boxes are constructed using multiple views and a semantic segmentation module. In the ChangingGround benchmark, the method shows progress over the baselines, and the ablations show which components are important in the success of this method.

### Strengths
1. This work identifies the problem of object grounding in scenes as an active perception problem rather than a static or passive problem, which makes the setting realistic.
2. Making the VQA problem rely on both memories and new, obtainable observations makes the problem more tractable under uncertainty about observations.
3. Using the accuracy and exploration cost both makes the benchmark a valid test for real world robotic applications.
4. The modular architecture makes it easier to understand the failure modes of different parts.

### Weaknesses
1. While the architecture is modular, the resultant accuracy is not broken down by the accumulated errors from different components.
2. The work does not give enough details about the ChangingGrounding benchmark, for example, how the queries are sampled, how the scene changes across observations, and what would be a rough corresponding score from a human.
3. While it is not essential, a bit more formalism in defining the grounding problem in the changing world would be helpful in recreating the method by follow up works.

### Questions
1. How can a follow-up paper reproduce the ChangingGround benchmark?
2. What are the primary ways this system shows brittleness to different ways of prompting the underlying VLMs (i.e. not just different versions of GPT)?
3. How do different underlying LLM/VLMs affect the performance of the Mem-ChangingGround?

### Soundness
2

### Presentation
3

### Contribution
3
