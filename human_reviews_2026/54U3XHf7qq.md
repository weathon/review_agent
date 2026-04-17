# MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Temporal context is essential for robotic manipulation because such tasks are inherently non-Markovian, yet mainstream VLA models typically overlook it and struggle with long-horizon, temporally dependent tasks. Cognitive science suggests that humans rely on working memory to buffer short-lived representations for immediate control, while the hippocampal system preserves verbatim episodic details and semantic gist of past experience for long-term memory. Inspired by these mechanisms, we propose MemoryVLA, a Cognition-Memory-Action framework for long-horizon robotic manipulation. A pretrained VLM encodes the observation into perceptual and cognitive tokens that form working memory, while a Perceptual-Cognitive Memory Bank stores low-level details and high-level semantics consolidated from it. Working memory retrieves decision-relevant entries from the bank, adaptively fuses them with current tokens, and updates the bank by merging redundancies. Using these tokens, a memory-conditioned diffusion action expert yields temporally aware action sequences. We evaluate MemoryVLA on 150+ simulation and real-world tasks across three robots. On SimplerEnv-Bridge, Fractal, LIBERO-5 suites and Mikasa-Robo, it achieves 71.9%, 72.7%, 96.5%, and 41.2% success rates, respectively, all outperforming state-of-the-art baselines CogACT and pi-0, with a notable +14.6 gain on Bridge and +11.8 gain on Mikasa-Robo. On 12 real-world tasks spanning general skills and long-horizon temporal dependencies, MemoryVLA achieves 84.0% success rate, with long-horizon tasks showing a +26 improvement over state-of-the-art baseline. Project Page: https://shihao1895.github.io/MemoryVLA

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MemoryVLA, a Cognition-Memory-Action framework for robotic manipulation that addresses the limitation of mainstream VLA models in handling long-horizon, temporally dependent tasks. Inspired by human memory systems, the approach features a Perceptual-Cognitive Memory Bank that consolidates low-level visual details and high-level semantic information, which working memory retrieves and fuses with current observations to condition a diffusion-based action expert.

### Strengths
1. The paper identifies a concrete limitation in existing VLA models (ignoring temporal context) and proposes a practical memory bank mechanism with retrieval, fusion, and consolidation operations. The dual-stream design (perceptual + cognitive tokens) with gated fusion is technically sound and the memory consolidation strategy effectively manages computational costs.

2. The method achieves substantial improvements over strong baselines, with comprehensive experiments across 3 robots, and thorough ablations validating each design choice. The real-world deployment on both Franka and WidowX robots demonstrates practical applicability beyond simulation.

### Weaknesses
1. The paper does not report inference time and memory footprint comparison with baselines. The memory retrieval via cross-attention at each timestep, especially with memory lengths up to 256, likely incurs significant computational overhead that could limit real-time deployment.

2. The paper lacks visualization or analysis of retrieved memory contents. It's unclear whether the memory bank actually retrieves semantically/temporally relevant contexts or if the gains simply come from having more visual history. Attention weight visualization or case studies showing retrieved frames would strengthen the claims.

3. Ablations (Tables 5-6) are only conducted on SimplerEnv-Bridge. It's unclear if the design choices (e.g., memory length=16, gate fusion, token merge) generalize to other benchmarks like LIBERO or real-world tasks where temporal dependencies may differ.

4. Limited generalization analysis:
* Task generalization: While OOD robustness is tested with visual variations (backgrounds, lighting, occlusion), there's no evaluation of zero-shot generalization to unseen task categories, which is crucial for a general-purpose VLA model.

* Memory capacity analysis: For long-horizon tasks, is memory length of 256 sufficient? The paper doesn't analyze what happens when task horizon exceeds memory capacity, or whether the consolidation strategy causes information loss for very long sequences.

* Benchmark selection for ablations: Tables 5-6 conduct ablations only on SimplerEnv-Bridge, which arguably doesn't require strong temporal dependencies (as the tasks are relatively simple pick-and-place). Ablations should be conducted on benchmarks where temporal reasoning is critical (e.g., real-world long-horizon tasks) to validate design choices.

### Questions
See weaknesses.

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
4

### Summary
The paper tackles core gap in VLA which is weak temporal reasoning for long-horizon, non-markovian manipulation, and they do so by drawing on cognition-memory-action architecture with two complementary memory systems. A pretrained VLM converts observations into (i) perceptual tokens and (ii) higher-level cognitive tokens that together serve as working memory for immediate control. In parallel, a Perceptual-Cognitive Memory Bank accumulates both low-level details and high-level semantic “gist.” At each step, working memory retrieves decision-relevant entries from the bank, adapts/fuses them with current tokens, and updates the bank by merging redundancies. A memory-conditioned diffusion action expert then produces temporally aware action sequences.

### Strengths
1. The working-memory vs. long-term (episodic + semantic) split is directly inspired by human memory and mapped cleanly to a VLA stack (perceptual/cognitive tokens + Perceptual-Cognitive Memory Bank). This makes the temporal modeling choice easy to justify and reason about.

2. Converting observations into perceptual and cogntiive tokens enable lightweight retrieval, fusion and consolidations.

3. Good performance on SimplerEnv-Bridge benchmark and LIBERO.

### Weaknesses
1,Benchmark mismatch (memory not actually required).

Fundamentally, the simulation benchmark used does not evaluate memory: the tasks appear in-distribution, short-horizon, and solvable without non-Markovian reasoning. I recommend evaluating on a benchmark that explicitly requires memory, such as Memory-Bench (from SAM2Act), to substantiate the paper’s claims.

2.Inadequate baselines (no memory or long-context retrieval).

The chosen baselines are not memory-enhanced and do not leverage long context or retrieval, making it difficult to attribute gains to the proposed method. Please compare against baselines that incorporate memory or retrieval to fairly assess effectiveness.

### Questions
The paper should address its two major weaknesses; without doing so, its claims will remain insufficiently supported. I would consider change the rating if my questions can be addressed.

### Soundness
2

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
5

### Summary
The paper introduces a new model, MemoryVLA, which employs a specialized memory bank designed to better handle temporal dependencies.

### Strengths
1. Incorporating memory mechanisms into VLAs is a highly relevant and important research direction.
2. The paper presents a large number of experiments, including those conducted on a real robot.
3. The work is well written and easy to read.

### Weaknesses
1. Despite the large number of experiments, the main drawback of the paper is that most of the tasks used do not actually require a memory mechanism.  The authors should conduct comparisons on specialized robotics benchmarks focused on memory-based tasks, such as Mikasa-Robo [1] and MemoryBench [2]. Without these experiments, it is impossible to properly evaluate the effectiveness of the proposed memory mechanism.
2. The results on LIBERO outperform Discrete Diffusion VLA [3] by only 0.3, even though the latter does not use any memory mechanisms. This again raises questions about the suitability of the chosen benchmarks for evaluation.
3. The functioning of the memory mechanism is not demonstrated clearly, it can only be inferred indirectly from the overall model performance. It is important to show that the memory bank retrieves relevant elements, providing direct evidence of how the mechanism contributes to task solving.

In its current state, I believe that, despite addressing an important direction, the paper does not sufficiently demonstrate that the proposed memory mechanism truly helps solve complex memory-dependent tasks. This is primarily due to testing mostly on simple tasks that do not require memory, as well as the lack of an in-depth analysis of the memory mechanism itself.

I am willing to reconsider my evaluation if these shortcomings are addressed.

References:
1. Cherepanov, Egor, et al. "Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning." arXiv preprint arXiv:2502.10550 (2025). 
2. Fang, Haoquan, et al. "Sam2act: Integrating visual foundation model with a memory architecture for robotic manipulation." arXiv preprint arXiv:2501.18564 (2025).
3. Liang, Zhixuan, et al. "Discrete diffusion vla: Bringing discrete diffusion to action decoding in vision-language-action policies." arXiv preprint arXiv:2508.20072 (2025).

### Questions
1. The main question is how the proposed model performs on tasks from Mikasa-Robo [1] and MemoryBench [2].
2. In [1], it was shown that using an action chunk of large size can circumvent the need for a memory mechanism to solve tasks. In MemoryVLA, a chunk of size T = 16 is used, predicting 16 steps ahead. How would the model behave if this value were reduced? In which cases does performance improvement come from using a long chunk, and in which cases from the memory mechanism itself?
3. What is the number of steps (actions) required to solve the tasks (mean, minimum, maximum, and median), including on the real robot?
4. How do the elements retrieved from the memory bank correspond to the task being solved at a given moment?
5. Why does the proposed model perform so much better than CogACT in real-robot experiments compared to simulation? Were the models trained under comparable conditions?

### Soundness
1

### Presentation
3

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
The paper tackles non-Markovian, long-horizon robotic manipulation where single-frame VLAs fail. It proposes MemoryVLA, a Cognition-Memory–Action framework that (i) converts a single RGB frame + instruction into perceptual tokens (DINOv2+SigLIP compressed via SE-bottleneck) and a single cognitive token (EOS from LLaMA-7B), (ii) stores both streams in a Perceptual–Cognitive Memory Bank (PCMB), and (iii) performs retrieval (cross-attention + timestep PE), gate fusion, and consolidation (adjacent-pair merge by cosine similarity) before a DiT+DDIM action head predicts a 16-step continuous 7-DoF trajectory. The authors conduct an extensive evaluation across 150+ tasks in simulation (SimplerEnv, LIBERO) and the real world.

### Strengths
* Clear structure with good motivation: the task of handling of non-Markovian tasks is interesting and significant in robotics, where the motivation fused memory term within the architecture design.
* Extensive evaluation: the authors evaluate MemoryVLA across three different robots, three distinct simulation benchmarks (SimplerEnv-Bridge, SimplerEnv-Fractal, LIBERO), and a set of 12 real-world tasks. This comprehensive evaluation on 150+ tasks with 500+ variations provides high confidence in the method's effectiveness and generalizability.
* Significant performance gain: the model achieves state-of-the-art performance across all benchmarks. The standout result is the +26 point improvement over the next-best baseline on real-world long-horizon tasks.

### Weaknesses
* The ambiguity of optimal memory length: from the ablation study in Table 5, it suggests that a memory length of $L=16$ is optimal (71.9% success), while the performance worsens at $L=64$ (67.7%). However, in the Appendix, the authors state that a memory length of $L=256$ was used for real-world long-horizon tasks. There lacks of in-depth analysis of how memory length is associated with the actual performance.
* The mechanism of using single cognitive token: for complex tasks require multiple latent hypotheses, one EOS token could be lossy. The paper doesn’t probe whether more cognitive capacity helps or hurts.

### Questions
* Although the model shows strong robustness to many OOD variations, the performance drops sharply when viewpoint changes in Appendix (Sec. C). Does that suggest the learned perceptual features are highly view-dependent and that the memory module may be memorizing visual details rather than an abstracted spatial representation?
* How does the memory length affect the performance of the model?
* How does different memory modules work independently? Can you provide a qualitative example of a task failure that occurs with only cognitive memory (i.e., lacking perceptual detail) and a failure that occurs with only perceptual memory (i.e., lacking cognitive gist)?

### Soundness
3

### Presentation
4

### Contribution
3
