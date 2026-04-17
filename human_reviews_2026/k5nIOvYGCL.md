# MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent

- Decision: Accept (Oral)
- Scores: 8, 8, 6, 4

## Abstract
Despite improvements by length extrapolation, efficient attention and memory modules, handling infinitely long documents without performance degradation during extrapolation remains the ultimate challenge in long-text processing. To solve this problem, We introduce a novel agent workflow, \method, which processes text in segments and updates memory through an overwrite strategy, addressing the challenge of long-context task through enhanced memory management. We further extend the DAPO algorithm to directly optimize memory ability in an end-to-end fashion, facilitating training via independent-context multi-conversation generation. Experimental results demonstrate that MemAgent has superb long-context capabilities, being able to extrapolate from an 8K context to a 3.5M QA task with a performance loss of less than 10\% and achieving over 95\% on the 512K NIAH test.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents MEMAGENT, an agent that processes long documents chunk-by-chunk, using an RL-trained policy to update a fixed-size text buffer that serves as its memory. This sidesteps quadratic attention costs, achieving linear-time inference. The method demonstrates SOTA performance on several long-document QA benchmarks.

### Strengths
* This paper proposed a novel agent-based framework with clever memory design that gets excellent performance on long-document QA 
* The approach reduce the complexity to linear-time inference complexity.

### Weaknesses
* The paper claims solving the problem of long-context LLM but mostly tested on synthetic or  semi-synthetic NIAH, RULER-HQA and extractive QA benchmarks. Form the memory architecture and prompt setup, this solution should well-suit the benchmarks given the major capability tested in these benchmarks are locating discrete facts. However, the same memory design may not work well with other tasks like summarization, etc.
* One major concerns regarding to the memory design is the error propagation. If there is a single step injecting error in the memory, it will propagate through the memory and finally affect the final results. More analysis on the failure patterns would make this draft more sound. 
* The comparison is not sufficiently fair. The comparison mainly shows the performance diff between iterative-memory and single-pass attention. The experiment is hard to conclude that MEMAGENT is a "better" long-context model, but rather that an iterative approach is superior for these specific retrieval-heavy tasks. A more sound comparison would involve other iterative or memory-based techniques.

### Questions
* Can you provide a failure mode analysis, especially in the 3.5M token QA task. Does the distribution of errors related to the distance of required context? 
* To better show the generalization of MemAgent, Could you provide results on a high-quality long-document summarization benchmark? 
* The performance comparison is against an independent LLM out-of-box or fine-tuned. Could you discuss the performance comparison with other agent solution, too?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces MemAgent, a novel agent-based workflow designed to allow LLMs to process arbitrarily long documents with linear time-complexity and minimal performance loss. Instead of processing an entire document at once, MemAgent divides the text into smaller chunks. It then iteratively reads each chunk alongside a fixed-length "memory" buffer. After each chunk, the model uses a simple "overwrite" strategy to update this memory with what it deems important.The core innovation is in its training.

The model is trained using a new Reinforcement Learning (RL) algorithm called Multi-conv DAPO. This method treats the entire sequence of memory updates for a single document as one long trajectory. A reward is given only for the final answer's correctness, and this reward is used to optimize all preceding memory-writing decisions. Experiments show that this approach is highly effective. A model trained with an 8K context window (using a 5K chunk and 1K memory) can extrapolate to answer questions on documents as long as 3.5 million tokens with less than 10% performance degradation—a task where standard long-context models fail completely.

### Strengths
1. **Strong Performance:** Extrapolating from an 8K training context to 3.5M tokens shows only minimal performance loss,  effectively solves the crucial problem in long-context modeling. The strong results on RULER-HQA, LongBench-QA, and NIAH demonstrate state-of-the-art performance and generalization.
2. **Novel and Effective RL Framework:** The Multi-conv DAPO algorithm is a clever solution to a difficult credit-assignment problem. By propagating the final answer's reward back to all intermediate memory-update steps, the model is directly optimized to learn what to remember to succeed at the final task. This shows an easy way to improve current model's ability.
3. **Well written paper**: The paper is well written and easy to follow.

### Weaknesses
1. Memory management seems to be a new lost-in-the-middle: In the ablation shown in Figure 9, increasing memory size does not seem to improve performance but rather decrease, and so this still demonstrate the model inherent problem with handling long-context (in this case the context window of the memory).
2. The memory length is different for different tasks, and this may seem an additional hyper-parameter which may be costly to tune.

### Questions
1. Do the authors have any intuition how this may be adapted to other tasks, like long-form summarization, where there is no verifiable outcome reward?
2. I am wondering if the authors have explored the lost-in-the-middle for MeMAgent, i.e. do the models still perform worse when the crucial information is in the middle or can this strategy effectively fix this problem?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MEMAGENT, a novel agent-based framework designed to enable LLMs to process arbitrarily long contexts with linear time complexity. The core idea is to process a long document in discrete segments. The model maintains a fixed-size "memory" buffer within its context window, which it iteratively updates after processing each segment. This overwrite strategy avoids the quadratic complexity of standard attention mechanisms.

The key contribution is the training methodology. The authors formulate the memory update mechanism as a policy to be learned through RL. They propose Multi-Conv DAPO, an extension of DAPO, to optimize the entire multi-step agent workflow based on a final outcome reward. In experiments, a model with an 8K context window trained using the MEMAGENT workflow demonstrates remarkable extrapolation capabilities, successfully handling QA tasks on documents up to 3.5 million tokens with minimal performance degradation, significantly outperforming existing long-context models.

### Strengths
1. The agent-based memory workflow is an elegant and practical solution to the long-context problem, sidestepping the quadratic complexity of attention.
2. The experimental results are outstanding. The ability to extrapolate from an 8K training context to a 3.5M token QA task with less than 10% performance drop is great.
3. By design, the method scales linearly with the length of the input document in terms of both time and memory, making it highly efficient for real-world deployment on extremely long texts.
4. The paper successfully demonstrates that RL can be used to train an LLM to perform a complex, multi-step reasoning task like dynamic memory management, with the proposed Multi-Conv DAPO algorithm being a key enabler.

### Weaknesses
1. The fixed-size memory is the source of the method's efficiency, but it's also a potential bottleneck. For tasks that require synthesizing many disparate pieces of information from across a long document, the model might discard critical information prematurely. The paper could discuss this trade-off more explicitly and analyze failure cases where this occurs. Moreover, I think a better choice is to use variable-sized memory based on the amount of information contained in the context. How to implement this in the current framework should be explained and experimented with, if possible.
2. The evaluation is heavily focused on extractive QA. It is unclear how well this memory management strategy would generalize to other long-context tasks like summarization, multi-document comparison, or creative writing, which may require different memory retention patterns.
3. The RL-based training setup is sophisticated, involving a custom algorithm (Multi-Conv DAPO) and multiple stages. This poses a high barrier to reproduction for researchers without access to similar frameworks and expertise. More detailed pseudocode or implementation guidance would be beneficial.
4. The choice to apply the final advantage value uniformly to all preceding memory-update steps is a simple credit assignment strategy. This could be suboptimal, as some memory updates are likely more critical than others. This represents a potential area for future improvement.

### Questions
Section 2.3 presents a theoretical latent-variable view of the process. How does this formal interpretation connect to the practical implementation of the RL agent? Is the agent learning to approximate a posterior distribution over memory states, or is it learning a more deterministic policy to generate a single memory trajectory?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents MEMAGENT to enable LLMs to process arbitrarily long contexts efficiently by learning to maintain and overwrite a fixed-length memory.
Rather than increasing the model’s context window or modifying its architecture, MEMAGENT processes long documents in segments (chunks) and uses a learned memory update policy to selectively retain and discard information relevant to the final task.
Empirical results show that MEMAGENT significantly outperforms strong long-context baselines such as Qwen2.5-Instruct-1M, QwenLong-L1, and DeepSeek-Distill.

### Strengths
1. Memory update is formulated as a reinforcement learning setting.
2. The experiments across RULER-HQA, LongBench-QA, and NIAH show the effectiveness of the proposed method.
3. Detailed ablation study is conducted.

### Weaknesses
1. The comparison is weak. There are some other long-context modeling methods such as FocusLLM (FocusLLM: Precise Understanding of Long Context by Dynamic Condensing) and E2LLM (E2LLM: Encoder Elongated Large Language Models for Long-Context Understanding and Reasoning) which also use chunks. Moreover, there are also some memory-based methods such as Mem0. The proposed method should be compared with them.
2. Despite inference efficiency, the RL training process is computationally heavy. The time complexity of the proposed method is not discussed.
3. While the model overwrites memory continuously, the paper does not empirically analyze failure cases where crucial early information is overwritten, nor propose mitigation strategies.

### Questions
What is the comparison with other long-context modeling methods and memory-based methods?

### Soundness
3

### Presentation
2

### Contribution
2
