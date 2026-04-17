# MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 8

## Abstract
The rapid progress of large language models (LLMs) has laid the foundation for multimodal models. Nevertheless, visual language models (VLMs) still face significant computational overhead when scaled from images to the video domain.
When video data is too large (due to high frame rates and long durations), the inference cost of models increases sharply. This severely hinders their deployment and application in environments that require rapid responses and have limited computation resources.
Token compression for input videos is one of the promising directions, as effective compression schemes can significantly reduce computational overhead.
Most existing compression methods are based on training-free token merging strategies in either the spatial or temporal dimension. Although these methods reduce computational overhead, their training-free nature inevitably leads to information loss during token compression, resulting in a significant performance drop.
To address these challenges, we propose a Memory-Augmented Reinforcement Learning-based Token Compression (MARC) method for efficient video understanding that integrates structured retrieval with RL-based distillation. 
Our proposed MARC is a retrieve-then-compress method, which employs a Visual Memory Retriever (VMR) tool and a Compression Group Relative Policy Optimization (C-GRPO) training strategy.
The Visual Memory Retriever first segments videos into event-level fragments and selects query-relevant clips. The C-GRPO distills reasoning ability from a Teacher Network to a Student Network by encouraging the output of the student network to match the performance of the teacher network. 
Extensive experiments on six video benchmarks demonstrate that our compression method achieves nearly identical accuracy to the 64-frame Qwen2.5-VL-3B baseline while using only one frame’s worth of tokens as input, resulting in a 95% reduction in visual tokens. Moreover, our approach reduces GPU memory usage by 72% and generation latency by 23.9%. 
These results demonstrate the strong potential of our compression method as a robust solution for RL-based post-training compression of large-scale models, enabling practical deployment in latency-sensitive and resource-constrained applications such as real-time video question answering, surveillance, and autonomous driving.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MARC, a retrieve-then-compress framework for efficient video understanding. The authors introduce the Compression Group Relative Policy Optimization (C-GRPO) algorithm, which transfers the complex reasoning ability from a high-token "Teacher" network to a low-token "Student" network. The method achieves performance comparable to full-token models while drastically reducing GPU memory consumption.

### Strengths
1. The proposed method is effective, successfully achieving performance with only 1 token that rivals models using 64 tokens.  
2. The introduced C-GRPO algorithm is interesting and demonstrates successful transfer of complex reasoning capabilities from a high-token "Teacher" network to a low-token "Student" network.

### Weaknesses
1. MARC appears to have significant limitations in long video understanding. Has the method been evaluated under scenarios where a high compression ratio is maintained but the input token count is increased to better accommodate longer videos?  
2. While the reduction in the number of tokens is impressive, the end-to-end latency improvement seems relatively limited.

### Questions
Related to the weaknesses above, have you explored strategies to better adapt MARC to long video inputs, possibly by increasing the input token count while maintaining a high compression ratio?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents MARC, a new framework for efficiently understanding videos. First, MARC segments videos into event-level clips. Then, it employs a Visual Memory Retriever to select relevant video segments, which are compressed using a Reinforcement Learning-based Compression Group Relative Policy Optimization (C-GRPO) strategy to reduce tokens. C-GRPO distills the Teacher Network, which takes 64 frames as input, into the Student Network, which takes just one frame's worth of tokens as input. This significantly reduces peak GPU memory usage while achieving comparable performance in some benchmarks (41.6G $\rightarrow$ 11.5G).

The novelty of this work lies in the C-GRPO, which is the first method to compress video tokens using reinforcement learning. Extensive experiments prove its obvious advantage over other token compression methods, despite the extra training required. I believe this will pave the way for a new approach to token compression for video understanding. 

However, I still hold some concerns:
1. Given the significant reduction in visual tokens, why is the inference latency nearly identical to that of the baseline? What is the main bottleneck limiting inference efficiency?
2. As shown in Table 1, the performance gap between MARC and the baseline (Qwen-VL 2.5) varies dramatically across benchmarks—MARC either substantially outperforms or noticeably underperforms the baseline. A more detailed analysis of these discrepancies across different benchmarks would be helpful.

Overall, I like this work and advocate its acceptance if my concerns are properly answered.

### Strengths
This work presents MARC, a new framework for efficiently understanding videos. First, MARC segments videos into event-level clips. Then, it employs a Visual Memory Retriever to select relevant video segments, which are compressed using a Reinforcement Learning-based Compression Group Relative Policy Optimization (C-GRPO) strategy to reduce tokens. C-GRPO distills the Teacher Network, which takes 64 frames as input, into the Student Network, which takes just one frame's worth of tokens as input. This significantly reduces peak GPU memory usage while achieving comparable performance in some benchmarks (41.6G $\rightarrow$ 11.5G).

The novelty of this work lies in the C-GRPO, which is the first method to compress video tokens using reinforcement learning. Extensive experiments prove its obvious advantage over other token compression methods, despite the extra training required. I believe this will pave the way for a new approach to token compression for video understanding.

### Weaknesses
1. Given the significant reduction in visual tokens, the inference latency is nearly identical to that of the baseline, indicating there is a bottleneck limiting inference efficiency
2. As shown in Table 1, the performance gap between MARC and the baseline (Qwen-VL 2.5) varies dramatically across benchmarks—MARC either substantially outperforms or noticeably underperforms the baseline. A more detailed analysis of these discrepancies across different benchmarks would be helpful.
3. The overview figures (Figure 1 and Figure 2) lack essential annotations and explanatory elements, making them difficult to interpret. I recommend enriching the figure with additional clarifying details to help readers better understand the proposed method.

### Questions
1. Given the significant reduction in visual tokens, why is the inference latency nearly identical to that of the baseline? What is the main bottleneck limiting inference efficiency?
2. As shown in Table 1, the performance gap between MARC and the baseline (Qwen-VL 2.5) varies dramatically across benchmarks—MARC either substantially outperforms or noticeably underperforms the baseline. A more detailed analysis of these discrepancies across different benchmarks would be helpful.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a "retrieval-first then compression" framework called MARC to address the significant computational cost and latency challenges faced by video language models (VLMs) when processing long videos. The framework first uses a "visual memory retriever" (VMR) to segment the video into event-based clips and identify the most relevant clips for the user's query. It then employs a novel reinforcement learning-based distillation strategy called "Compressed Group Relative Policy Optimization" (C-GRPO) to distill the inference capability of a "teacher model" that processes 64 frames into a "student model" that uses only the token count equivalent to one frame.

### Strengths
Achieve a 95% visual token compression rate with almost no loss of model average accuracy, and significantly reduce memory and latency.
Originality: Core Contribution C-GRPO performs a clever "hack" on the existing GRPO algorithm by designing a new "performance retention reward" (retention alignment reward) specifically for compression tasks.

### Weaknesses
1.The paper lacks some comparisons with token compression works such as visionzip, videoxl, etc., and the author used some data and content from video-r1, but why there is no comparison with video-r, making the results less persuasive.
2. The motivation introduced in the intro is not strong. The author mentions that previous work caused significant information loss due to being training-free. First, can this conclusion be determined with 100% certainty? For example, if the compression ratio of training-free methods is not that large, would there still be substantial information loss? The essence of this information loss is not necessarily caused by being training-free. Second, previous methods are not all training-free; many can be trained and fine-tuned. The author introduces the use of RL training in this work, but the motivation is not very strong. For instance, why use RL and not the previously mentioned trainable methods?
3. The author can provide some use cases at the end of the main text or in the appendix to help readers better understand.

### Questions
1.MARC writes in the implementation part that 1 fps sampling is used with a maximum of 64 frames, but the final evaluation results table shows 1 frame. Generally, the maximum input frame number is used, not the compressed frame number. The author can check the previous token compression work paper.

### Soundness
2

### Presentation
2

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
The MARC framework addresses the significant computational overhead and memory bottlenecks faced by Visual Language Models (VLMs) when processing long-duration videos, aiming to overcome the performance degradation common in prior training-free compression methods. 

The MARC framework successfully solves the VLM computational bottleneck by achieving robust performance retention despite an extreme 95% reduction in visual tokens, driven by the innovative VMR and C-GRPO components. It can be a natural extension of various token compression or frame selection based on RL.

However, as all of its preceeding works, while highly efficient and accurate across most tasks, its fixed retrieval limit causes notable performance sacrifice in scenarios involving very long, complex temporal contexts. Given its substantial practical contributions to VLM deployment efficiency, the overall conclusion is a **accept**.

### Strengths
*   **Exceptional Efficiency and Resource Optimization:** MARC achieves a **95% reduction in visual tokens** by compressing input to the equivalent of a single frame. This results in practical gains, including a **72% reduction in GPU peak memory usage** and a **23.9% reduction in generation latency**, enabling deployment in resource-constrained, real-time applications.
*   **Performance Maintenance at Extreme Compression:** The framework maintains **nearly identical mean accuracy** (42.20 vs. 42.21) compared to the 64-frame baseline and surpasses the baseline on several complex benchmarks (MMVU, MVBench, TempCompass). It significantly outperforms prior training-free compression strategies.
*   **Novel Reinforcement Learning for Distillation:** C-GRPO ensures performance robustness under aggressive compression by using a specialized **retention alignment reward** to explicitly match the student network's output to the teacher's reasoning quality.
*   **Structured Retrieval:** The VMR segments video into semantically coherent events, confirming that this approach eliminates redundant information and even boosts baseline accuracy by focusing on query-relevant context before compression.

### Weaknesses
*   **Performance Degradation on Extremely Long Videos:** The model incurs performance loss when dealing with videos of average duration exceeding several hundred seconds (e.g., VideoMME), retaining only **74% of the baseline score**. This occurs because the model cannot recover information once most of the temporal chain has been discarded.
*   **Static Compression Strategy:** The framework’s commitment to a **fixed, single-frame equivalent token budget** lacks dynamic adaptability. This static approach struggles when crucial temporal context is scattered across many fragments, as the VMR's restricted retrieval inevitably discards necessary information.

### Questions
See summary for my main concern.

### Soundness
4

### Presentation
4

### Contribution
3
