# Scaling up Memory for Robotic Control via Experience Retrieval

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Humans rely on memory to perform tasks; our goal is to endow robot policies with the same ability. Naively conditioning on long observation histories is computationally expensive and brittle under covariate shift, while indiscriminate subsampling of history leads to irrelevant or redundant information. We propose a hierarchical policy framework, where the high-level policy is trained to select and track previous task-relevant keyframes from its experience. The high-level policy uses selected keyframes and the most recent frames when generating text instructions for a low-level policy to execute. This design is compatible with existing vision-language-action (VLA) models and enables the system to efficiently reason over long-horizon dependencies. In our experiments, we fine-tune Qwen2.5-VL-7B-Instruct and $\pi_{0.5}$ as the high-level and low-level policies respectively, using demonstrations supplemented with minimal language annotations. Our approach, MemER, outperforms prior methods on three real-world long-horizon robotic manipulation tasks that require minutes of memory. Videos and code can be found at https://jen-pan.github.io/memer/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work addresses the problem that robotic policies often rely only on the current observation or a few recent frames and thus lack long-term memory. It proposes a hierarchical policy framework: the high-level policy selects keyframes or memories from past experiences and generates language-based subtasks or instructions based on both the current observation and the retrieved keyframes; the low-level policy then receives the current image, the robot’s current state, and the subtask produced by the high-level policy to execute the concrete actions.

### Strengths
1. The system design is simple, effective, and scalable.

2. The results show a significant improvement in long-horizon task success rates.

3. The writing is clear and easy to follow.

### Weaknesses
1. The evaluation of scalability aspects such as memory size and retrieval latency could be more detailed.

2. Providing a conceptual comparison among different types of approaches addressing the long-horizon problem would offer readers deeper insights.

### Questions
1. How does the high-level policy determine which frames are keyframes? When the task types or scenes vary significantly, is this keyframe selection mechanism still generalizable, or does it require task-specific tuning?

2. The paper includes baselines such as “Short History (8 frames)” and “Long History (32 frames).” I’m curious why the long-history setup leads to such a large improvement. Moreover, since simply adding more historical frames performs worse than MemER, why does “more history” not yield the same gains as the keyframe retrieval mechanism?

3. After the high-level module generates subtasks and keyframes, isn’t the way the low-level policy directly uses this information somewhat too simple?

4. Although the paper proposes using memory to tackle long-horizon problems, it does not compare MemER conceptually with other VLA-based or related long-horizon approaches [1, 2, 3]. Even if experimental comparison is difficult, a conceptual discussion would provide readers with deeper insights.

5. The paper mentions that keyframes are accumulated into memory but currently lacks a mechanism for “deletion” when the memory becomes too large. Are there any promising methods to address this issue?

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
2

### Summary
This paper presents MemER, a hierarchical architecture for Vision-Language-Action (VLA) modeling. The low-level policy is a general VLA model, while the high-level policy employs a Vision-Language Model (VLM) to predict primitives and identify key frames. The core contribution is the proposed experience retrieval mechanism, which operates at the high-level policy, enabling the VLM to leverage critical historical information for more accurate primitive prediction.

### Strengths
- The paper is easy to follow, and the figures and tables are clear and easy to understand.
- There is a strong motivation for this work. Memory is essential for real-world, long-horizon tasks, a need that is often overlooked in existing VLA literature.
- The approach is simple and effective. The low-level VLA model requires minimal or even no training, with the bulk of the effort focused on fine-tuning the high-level VLM. This results in a low overall training cost.
- The experimental validation provided is thorough and comprehensive.

### Weaknesses
- The architecture of MemER appears overly simple. The core advancements appear to lie primarily in the training methodology and data preparation for the high-level VLM, and these specific improvements do not seem to be particularly novel. This raises concerns about the overall innovation of the paper.

- The data preparation phase seems to be highly resource-intensive, relying on manual annotation of primitives and the segmentation of trajectories.

### Questions
- Could the authors more clearly articulate the specific innovative contributions of this paper? Given that the architecture is simple and the high-level training methods appear familiar, a clearer distinction from existing work is necessary to establish the technical merit.

- The authors discuss several related VLA models that utilize memory in the Related Works section. Why are these models not included as baselines for comparison in the experimental evaluation? Including these relevant memory-based approaches would provide a more rigorous validation of MemER's claimed state-of-the-art performance in memory-intensive tasks.

### Soundness
3

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
4

### Summary
- A hierarchical VLA (Vision-Language-Action) framework where the high-level policy retrieves and tracks keyframes from past experience.

- Efficient memory management via online keyframe selection and filtering, reducing redundancy and computational cost.

- Real-world evaluation on three long-horizon tasks requiring minutes of memory: Object Search, Counting Scoops, and Dust & Replace.

### Strengths
- Successfully tackles real-world robotic tasks that require reasoning over several minutes of past experience (hundreds of frames), a significant step beyond prior work limited to a few dozen frames.
- The hierarchical design with intelligent keyframe selection avoids the high cost of processing long, raw video sequences, enabling low-latency inference (~1 Hz for high-level policy) suitable for closed-loop control.

### Weaknesses
- The framework was evaluated on a single robot arm and on memory within a single task. Its scalability to mobile manipulation, multi-room navigation, and cross-task memory recall remains unexplored.
- The approach is inherently limited to the specific task it was fine-tuned on and lacks the capacity for broader scaling.

### Questions
- How does the keyframe extraction algorithm proposed in this work compare to frequency domain-based clustering methods (such as Fourier transform or wavelet transform followed by clustering), for example, with *UniDomain*?
- As tasks become longer, does the computational overhead of maintaining an explicit visual memory buffer adversely impact the performance of the high-level policy by reducing its inference frequency?
- Does the approach presented in this paper offer significant advantages compared to existing long-context strategies, such as introducing sink tokens?

### Soundness
3

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
This paper proposes MemER, a hierarchical framework that endows VLAs with long-term memory via experience retrieval. A high-level VLM policy processes recent frames and retrieved keyframes to generate language primitives and nominate new keyframes for storage. A low-level VLA policy executes these primitives based on the current frame. This sparse retrieval mechanism allows the robot to successfully perform complex, multi-minute tasks that depend on recalling distant past events.

### Strengths
- The hierarchical design, where a high-level policy learns to explicitly nominate salient keyframes for retrieval, is a novel and highly scalable architecture for managing truly long-horizon (multi-minute) dependencies.
- The experiments are well-designed for long-horizon tasks, and the ablation comparing visual memory ($K_{img}$) to textual memory ($K_{text}$) provides a crucial insight into the limitations of using language as a lossy memory representation.

### Weaknesses
- The method's generalizability is unproven, as it was only evaluated on three custom, in-domain tasks and lacks benchmarks on standard, multi-task datasets like RoboCasa or LIBERO.
- The framework introduces significant system complexity and data annotation overhead, as it requires labeling both language primitives and ground-truth keyframes for training the high-level policy.
- The paper provides no computational analysis, making it impossible to assess the inference latency or memory cost of this dual-policy system, which is a critical factor for real-world deployment.
- The comparison to GPT-5 was only conducted offline (Table 2), and it's unclear if this setup accurately reflects the challenges of closed-loop control or simply highlights the need for finetuning.

### Questions
- The most important concern is that the dual-policy architecture is computationally heavy and overly complex, requiring two separate, large models (a VLA and a VLM) to be run in parallel. The paper fails to justify this separation or explore a more efficient, unified architecture where a single VLM backbone could perform both high-level planning (keyframe nomination, primitive generation) and low-level feature extraction, which is a significant missed opportunity.

### Soundness
3

### Presentation
3

### Contribution
2
