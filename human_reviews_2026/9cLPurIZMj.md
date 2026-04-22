# Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 4, 8, 8, 6

## Abstract
Memory is crucial for enabling agents to tackle complex tasks with temporal and spatial dependencies. While many reinforcement learning (RL) algorithms incorporate memory, the field lacks a universal benchmark to assess an agent's memory capabilities across diverse scenarios. This gap is particularly evident in tabletop robotic manipulation, where memory is essential for solving tasks with partial observability and ensuring robust performance, yet no standardized benchmarks exist. To address this, we introduce MIKASA (Memory-Intensive Skills Assessment Suite for Agents), a comprehensive benchmark for memory RL, with three key contributions: (1) we propose a comprehensive classification framework for memory-intensive RL tasks, (2) we collect MIKASA-Base -- a unified benchmark that enables systematic evaluation of memory-enhanced agents across diverse scenarios, and (3) we develop MIKASA-Robo -- a novel benchmark of 32 carefully designed memory-intensive tasks that assess memory capabilities in tabletop robotic manipulation. Our work introduces a unified framework to advance memory RL research, enabling more robust systems for real-world use.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper “Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning” introduces MIKASA (Memory-Intensive Skills Assessment Suite for Agents), a unified benchmark framework designed to systematically evaluate memory mechanisms in reinforcement learning (RL). Recognizing the absence of a standard evaluation suite for memory-based RL, especially in robotic manipulation, the authors propose a taxonomy of four memory types: object memory, spatial memory, sequential memory, and memory capacity. Building upon this taxonomy, they release two complementary components: MIKASA-Base, a consolidated collection of existing memory-intensive RL environments unified under a common API, and MIKASA-Robo, a new suite of 32 tabletop robotic manipulation tasks that require various forms of temporal and spatial memory. Experimental evaluations with online RL, offline RL, and VLAs demonstrate that current models, such as PPO-LSTM, Decision Transformer, and Octo, struggle to generalize across tasks with strong memory dependencies. The results highlight that while RL agents can perform well under full observability, their performance deteriorates sharply when memory retention is required, validating MIKASA as a rigorous benchmark for developing and assessing memory-centric RL algorithms.

### Strengths
The paper introduces a benchmark that systematically studies memory in reinforcement learning for tabletop manipulation, extending ManiSkill3 with new task variants emphasizing temporal dependency and spatial recall. The motivation is relevant and well-grounded, as memory remains a key bottleneck for long-horizon robotic reasoning. The paper provides clear organization, a well-documented codebase, and thorough evaluations across multiple RL and imitation learning baselines. By categorizing tasks based on distinct memory demands, the benchmark offers a structured framework for analyzing policy retention and forgetting behaviors. The authors also contribute useful diagnostic discussions and visualization tools for understanding agent performance over time. Overall, the work is timely and thoughtfully executed, offering a meaningful step toward standardized evaluation of memory mechanisms in robotic learning, even though its scope remains largely limited to simplified simulation settings.

### Weaknesses
There are some weaknesses arise:

1.This benchmark foucs on solving memory issuse encuntered in the table top manipulation scenarios. The benchmark is build upon maniskill3, where there is only a franka panda arm as the embodiment. And the manipulation task distribution is limited at pick-place blocks with various colors. However, in the real-world, more complex spatial resoning and operations are required. The lack of more diverse, real-world tasks and embodiments (such as retrieving a can of cola from a closed refrigerator, or a humanoid robot sequentially using both arms to assemble components collaboratively) undermines the overall contribution of this benchmark. Therefore, this benchmark can be considered merely an incremental supplement to the previous Maniskill3.

2.This benchmark does not conider the sim2real gaps that encounter in the real-world deployment, even though the authors claimed that their unified framework that can enable more robust systems for real-world use. Lack of real-world performance as evidence undermines the overall contribution of this benchmark. The RL memory problems that authors analyzed occur only in simulation or also in real-world? If memory problems (such as occuled objects, spatial reasoning) also encounter in the real-world, current evidence reported in the paper is not enough to support the claim about real-world application. Although I acknowledge simplfied simulation task setup is useful for controlled testing, these settings may not capture the sensory noise, continuous dynamics, and unstructured complexity found in real robot environments.

3.As this benchmark is an incremental work of maniskill3, current task distribution is limited in pick-place various blocks. However using such tasks for spatial/object evaluation is already covered in previous benchmarks, such as Ravens, LoHoRavens. Moreover, benchmark like LIBERO involve both object/spatial reasoning tasks and also include long-horizon tasks that require long-temporal dependencies, and the operated objects are more close to the real-world setting (e.g., coffee machine).  

4.Although the benchmark divides tasks by memory type, the evaluation mainly reports success rates. It lacks diagnostic tools to analyze why an agent fails—whether due to perception errors, memory decay, or decision confusion. This makes it difficult to understand failure mechanisms for each memory type.

5.A minor point as the paper mentions that the evaluation of models like Octo and OpenVLA was restricted due to computational constraints, and fine-tuning was minimal. This means the reported performance may not reflect their full capability, reducing the fairness and depth of comparison across architectures. More extensive large-model adaptation is needed for a complete evaluation.

In summary, while the benchmark extends ManiSkill3 with 32 tasks, its contribution to memory-intensive RL remains limited because it does not address the sim-to-real gap and relies solely on success rates without richer diagnostic metrics. Most tasks center on pick-and-place cubes. Therefore, it is unclear how this benchmark provides substantial improvement over existing RL manipulation benchmarks.

### Questions
1.How well do the tabletop tasks in MIKASA-Robo reflect real-world robotic manipulation challenges? Since most tasks are simplified color or position memory exercises, can performance on these synthetic environments meaningfully predict how well an agent will perform in real, noisy physical settings?

2.The paper proposes four memory categories (object, spatial, sequential, capacity). How are the metrics and evaluation protocols designed to isolate these different memory mechanisms? Could overlapping factors (e.g., attention or perception) interfere with measuring pure memory ability?

3.MIKASA currently uses RGB and joint states for perception. Would incorporating other sensory modalities such as depth, tactile feedback, or language instructions change the evaluation of memory skills? 

4.Has the author considered scenarios where occlusion exists from the outset (e.g., retrieving a bottle of cola from a closed refrigerator)? Can the task be accomplished solely with RGB+joint data in such cases? Additionally, can this benchmark support extensions like dual-arm coordination for humanoid robots and other heterogeneous embodied systems? Could the authors provide a unified interface to facilitate such expansions?

5.Have the authors considered adding diagnostic metrics or analyses beyond success rate, such as error decomposition or memory recall tracking, to better reveal the failure causes across different memory types?

6.The evaluation of large-scale vision-language-action models like Octo and OpenVLA was limited due to computational constraints. How might more thorough fine-tuning or longer training horizons change the conclusions about these models’ memory abilities? Does the benchmark currently support scalable training for such large architectures (provide scalable dataset or distributed training)?

I reserve the possibility of raising my score if the authors can adequately address all the concerns raised during the rebuttal phase.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a comprehensive classification framework for memory-intensive RL tasks, integrating multiple previous tasks into a unified benchmark (MIKASA-Base). The authors also develop a novel benchmark of 32 memory-intensive tasks for tabletop robotic manipulation (MIKASA-Robo). They evaluate several current algorithms on their benchmark, demonstrating a clear need for the development of more effective algorithms for memory-intensive tasks.

Overall, I believe these contributions are of significant interest to the RL community and I strongly recommend acceptance. The comprehensive suite of tasks and thorough documentation make this a valuable resource for future research.

### Strengths
The paper is well-motivated, addressing an important gap in existing RL benchmarks. The comprehensive suite of 32 tasks provides comprehensive coverage of different types of memory requirements in physical robotic tasks.

The categorization of previous tasks and integration into a single framework is an important contribution that will facilitate further research.

The authors evaluate both online and offline algorithms and motivate the need for both in memory-intensive tasks.

The authors convincingly demonstrate that the tasks are challenging for current algorithms, showing the benchmark's value for driving future research.

The benchmark codebase is well-documented and the authors provide an easy startup notebook.

### Weaknesses
This is not the first benchmark for physical memory-intensive tasks. However, the authors acknowledge this limitation and clearly state that their benchmark captures more types of memory tasks than previous work.

The connection to cognitive processes and memory mechanisms could be more fully developed.

### Questions
**Questions for clarification:**

1. You mention that "The link between physical interaction and memory remains underexplored, motivating a framework for spatio-temporal memory in real-world tasks." What kinds of memory mechanisms do physical memory tasks require? What specific memory capabilities are not considered in the current framework? A more detailed discussion of this would strengthen the motivation.
2. What are your plans for continued support of the benchmark? Are you planning on adding new environments or algorithms? A section on future work would be welcome.
3. In Section 4.2, what does each memory task correspond to with regard to cognitive processes? 

---

**Minor comments:**

Figure 6 is difficult to digest. I’m not sure what the legend items for Sequential Memory, Memory Capacity, Spatial Memory, and Object Memory correspond to on the plot since the background shading of the different tasks doesn’t match the colors in the legend. Additionally, Sequential Memory and Memory Capacity have different legend icons than Spatial Memory and Object Memory.

---

**Suggestions for improvement (not factored into score):**

1. The authors could elaborate more on the connection between cognitive processes and agentic memory requirements, maybe through a more systematic categorization. This would strengthen the theoretical foundation of the benchmark and help researchers understand what types of memory mechanisms are needed for different task categories.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a benchmark for testing the capabilities of an agent's memory in the context of robotic manipulation with reinforcement learning (RL).
They distinguish different types of memory tasks as 1) object memory, 2) spatial memory, 3) sequential memory, and 4) memory capacity, which seems intuitive and, to the best of my knowledge, hasn't been done before.
They propose a set of Gymnasium-based tasks for memory-enhanced RL and a set of 32 robotic tasks.
They also release a set of offline visual-based datasets for sparse-reward manipulation tasks, for offline RL.
It's built upon ManiSkill3, which offers fast GPU parallelisation.
However, it is worth noting that while I do work on RL, I do not have experience with memory mechanisms in RL.

### Strengths
The proposed benchmark addresses spatial and temporal memory -- an important and timely topic in RL. The authors evaluate a broad set of baselines, and since none achieve strong performance across all tasks, the benchmark appears to have an appropriate level of difficulty, leaving room for future progress. It is built on the ManiSkill3 simulation framework, enabling fast GPU parallelisation -- a valuable design choice for scalability. Overall, the paper is well written and easy to follow, with only minor issues to fix.

### Weaknesses
- The main text lacks details on the quality of the offline RL datasets. The authors should explicitly state that these are expert datasets, as this information is important for readers and enables potential use of imitation learning methods.
- TD-MPC2 is cited as an arXiv preprint, but it has been published at ICLR 2024.
- The bibliography has inconsistent capitalisation (e.g., “Td-mpc2” should be “TD-MPC2”). This can be fixed by correctly using braces {} in BibTeX.
- It is unusual to use "Subsection". I would advise the authors to use "Section".
- Formatting issues:
    - Figure 1 needs more space around it.
    - Table 1 requires additional white space below it.
    - The text in Table 3 is too small; it should be made bigger.
    - The text in Figure 6 is too small and difficult to read; it should be enlarged.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MIKASA (Memory-Intensive Skills Assessment Suite for Agents), a comprehensive benchmark suite aimed at systematically evaluating memory capabilities in reinforcement learning (RL) agents. The authors address a major gap in the field — the lack of standardized benchmarks to assess memory across diverse settings, particularly in tabletop robotic manipulation, where partial observability and temporal dependencies are critical.

### Strengths
1. The benchmark offers robust and comprehensive coverage, incorporating a diverse set of tasks that rigorously test models across a spectrum of memory requirements.

2. This benchmark successfully strikes a critical balance, bridging the gap between current, often over-simplified evaluation methods and the complexity of real-world, human-level tasks. By incorporating challenges that mirror the cognitive demands of human problem-solving while maintaining experimental tractability, it establishes a new, more relevant standard for assessing model performance

### Weaknesses
1. The benchmark is specialized for tabletop robotic manipulation tasks. While this area is critical for memory in robotics, the benchmark does not cover other key areas of robotic memory, such as continual learning where the agent must adapt to changing task distributions or physical environments over extended, sequential periods.

2. By design, the tasks are complex and memory-intensive, leading to high computational demands for training. This can limit the accessibility of the benchmark to researchers without substantial computational resources.

### Questions
1. How do you envision extending the benchmark to evaluate memory mechanisms in continual or lifelong learning settings, where robots must adapt to evolving task distributions or environments while not forgetting previous ones?

2. Given that the benchmark’s tasks are both complex and memory-intensive, have you explored ways to reduce the computational cost? How do you balance the benchmark’s realism and difficulty with inclusivity for a wider research community?

### Soundness
2

### Presentation
3

### Contribution
2
