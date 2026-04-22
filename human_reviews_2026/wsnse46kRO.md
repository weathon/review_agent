# Visual Planning: Let's Think Only with Images

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 8, 6, 4, 6

## Abstract
Recent advancements in Large Language Models (LLMs) and their multimodal extensions (MLLMs) have substantially enhanced machine reasoning across diverse tasks. However, these models predominantly rely on pure text as the medium for both expressing and structuring reasoning, even when visual information is present. In this work, we argue that language may not always be the most natural or effective modality for reasoning, particularly in tasks involving spatial and geometrical information. Motivated by this, we propose a new paradigm, Visual Planning, which enables planning through purely visual representations for these "vision-first'' tasks, as a supplementary channel to language-based reasoning. In this paradigm, planning is executed via sequences of images that encode step-by-step inference in the visual domain, akin to how humans sketch or visualize future actions. We introduce a novel reinforcement learning framework, Visual Planning via Reinforcement Learning (VPRL), empowered by GRPO for post-training large vision models, leading to substantial improvements in planning in a selection of representative visual navigation tasks, FrozenLake, Maze, and MiniBehavior. Our visual planning paradigm outperforms all other planning variants that conduct reasoning in the text-only space. Our results establish Visual Planning as a viable and promising supplement to language-based reasoning, opening new avenues for tasks that benefit from intuitive, image-based inference.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This study proposes a visual planning approach that performs task planning entirely through visual representations. In this paradigm, planning is carried out via a sequence of images, which encode the step-by-step reasoning process in the visual domain, similar to how humans make sketches or visualize future actions.

### Strengths
It performs planning purely within the visual modality as a holistic process, where the actions are not explicitly predicted but instead implicitly represented by transitions between visual states.

### Weaknesses
The experimental validation is restricted to discrete, low-dimensional grid-world navigation tasks, where the visual states are comparatively simple and straightforward to encode.
Visual planning implicitly expresses actions by generating a sequence of visual states. Although this avoids modality switching, when planning fails, the absence of an explicit action sequence (such as textual CoT) makes the model’s decision process difficult to debug and understand.

### Questions
1. Visual planning involves generating high-dimensional image sequences, which can be computationally more expensive than searching in low-dimensional text/action spaces as in language models. Please elaborate on how  it's computational efficiency and search space complexity compare to textual CoT methods?

### Soundness
4

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
2

### Summary
The paper takes visuals as a medium for expressing and structuring reasoning, which potentially learn spatial and geometrical information. To realize this, the author proposes Visual Planning, planning through purely visual representations for “vision-first” tasks. The proposed GRPO demonstrates improvement in planning.

### Strengths
The author investigates the potential of visual representation as a medium, which expands the research of LLMs to a broader area.

The presentation of the paper is great, with a clear statement and an appropriate graph.

The paper is the first attempt to investigate whether models can achieve planning purely through visual representations.

### Weaknesses
Can you provide any figures to clearly show the difference between language as a medium and visual as a medium in certain cases?

It will be better if we can discuss any advantages of visual as a medium in real CV tasks, such as visual grounding. And the proposed methods, whether they can be easily transferred to 3D?

If we finally want to get an MLLM, how do we add GRPO to the regular training receipt? When to align visuals with other modalities?

### Questions
See Weakness.

### Soundness
3

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
4

### Summary
This paper introduces Visual Planning, a new paradigm that performs reasoning through visual representations rather than language. Using VPRL (Visual Planning via Reinforcement Learning), a GRPO-based post-training framework for large vision models, our approach generates image sequences that visualize step-by-step inference. Experiments on visual navigation tasks (FrozenLake, Maze, MiniBehavior) show that VPRL surpasses text-based reasoning, highlighting visual reasoning as a powerful complement to language reasoning for spatial and geometry-driven problems.

### Strengths
1. New Paradigm: The paper introduces "Visual Planning" as a genuinely new paradigm for reasoning.

2. Good Empirical Results: The proposed method, Visual Planning via Reinforcement Learning (VPRL), significantly outperforms a wide range of baselines.

3. Methodological Robustness: The two-stage VPRL framework is well-designed and justified.

### Weaknesses
1. Reliance on an External Oracle for Rewards: A significant weakness in the method's detail is its reliance on non-learned, external modules to provide the reward signal. The VPRL framework depends on a "dynamics interpreter" and a "progress estimator". The appendix reveals this estimator is a Breadth First Search (BFS) algorithm —an oracle that has already solved the task and knows the optimal path from every state. The interpreter also uses rule-based pixel and IoU comparisons. This means the model isn't learning the environment's dynamics or the concept of progress; it's learning to generate images that satisfy an external oracle that already has the answers.

2. Insufficient Justification for a "Purely Visual" Paradigm: This paper needs to justify why exploring a purely non-verbal, vision-only paradigm is a necessary or superior research direction. The authors' decision to "eliminate language as a confounding factor" is a research-scoping choice, but it is not a strong argument for the paradigm's utility.

3. Limited Task Complexity and Scalability: The paradigm is validated only on simple, 2D, discrete grid-world environments (FROZENLAKE, MAZE, MINIBEHAVIOR). It is highly questionable if this approach can scale to complex, 3D, photorealistic, or continuous-state environments (e.g., robotics). In such settings, autoregressively generating a perfect, step-by-step sequence of future images is computationally expensive (a point the authors concede ) and the rule-based reward function (which relies on pixel comparison ) would be far too brittle to work.

### Questions
All of my qeustions are listed in the weakness section. If my concerns are well addressed, I will raise my rating.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on visual planning tasks in MLLMs. It proposes a novel way to perform visual planning by generating images during the planning process. Specifically, given an input image, the model reasons about the next state after performing certain actions by generating the corresponding images. To achieve this, the VLM is first trained to generate next-state images, then trained through reinforcement learning to encourage actions toward the target state. Experiments show that the proposed methods outperform strong open-source and private models in various environments.

### Strengths
1. Generating images makes the reasoning occur in the visual space rather than the textual space. Thus, the proposed method has the potential for more direct and better reasoning performance.
2. Through qualitative analysis of intermediate outputs, the paper shows that the proposed method can generate reasonable intermediate images, which is key for correct visual reasoning.
3. The experiment result demonstrates the proposed method outperforms strong baselines including private MLLMs.

### Weaknesses
1. It can be observed that the intermediate images are not perfect (e.g., in Fig. 3, first row, the player and goal tokens have artifacts). Thus, it would be interesting if the paper could show performance when the model reasons over high-quality images. For example, each time the model generates a new image, the corresponding high-quality image (rendered by the engine rather than generated by the model itself) is fed into the model. Would this lead to better performance? If so, the performance gap could quantify the importance of generating high-quality (precise) intermediate images.
2. If I understand correctly, $v$ stands for one of the intermediate images, as stated in L132–133. As such, how is it determined whether two images are an exact match in the EM metric in L300-L302? Or is it actually comparing in the action space (e.g., left/right) rather than the image space?
3. Similar to this work, some recent studies also explore generating intermediate images during reasoning, such as [1]. It is good that the paper discusses these related works, but it should consider directly comparing with these baselines.

[1] Imagine while Reasoning in Space: Multimodal Visualization-of-Thought

### Questions
Please check the weaknesses section:
1. (for weakness 1) How important it is to generate high-quality images? Would current intermediate images quality good enough?
2. (for weakness 2) If I understand correctly about EM measure, How to compute the exact match in the image space?
3. (for weakness 3) Comparison with similar methods such as MVoT?

### Soundness
3

### Presentation
3

### Contribution
3
