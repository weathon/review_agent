# CORE: Lossless Compression for Retrieval-Augmented LLMs via Reinforcement Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a promising approach to enhance the timeliness of knowledge updates and the factual accuracy of responses in large language models. However, incorporating a large number of retrieved documents significantly increases input length, leading to higher computational costs. Existing approaches to document compression tailored for RAG often degrade task performance, as they typically rely on predefined heuristics in the absence of clear compression guidelines. These heuristics fail to ensure that the compressed content effectively supports downstream tasks. To address these limitations, we propose CORE, a novel method for lossless context compression in RAG. CORE is optimized end-to-end and does not depend on predefined compression labels, which are often impractical to obtain. Instead, it leverages downstream task performance as a feedback signal, iteratively refining the compression policy to enhance task effectiveness. Extensive experiments across four datasets demonstrate the effectiveness of CORE. With a high compression ratio of 3\%, CORE not only prevents performance degradation compared to including full documents (i.e., without compression) but also improves the average Exact Match (EM) score by 3.3 points. The code for CORE is available at \url{https://anonymous.4open.science/r/CORE-28B4}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CORE, a method that trains a compressor model that compresses long evidence documents into concise summaries, using end-to-end reinforcement learning that directly optimizes downstream task performances. Experiments adopted four QA tasks and showed that the compressed evidence achieves better or on-par performances with using the original evidence on different downstream models, with a high compression rate (~95% reduction in tokens). Ablations also suggest that CORE may transfer across compressor model families and downstream model families.

### Strengths
This paper is well written with clear presentation and logical comprehensiveness. It proposes a strong engineering implementation of an end-to-end RL framework (and models) for evidence compression, which achieved strong empirical results on multiple common QA datasets with a very high compression rate. The experiments contained multiple baselines for future work to reference. The provided implementation details support reproducibility.

### Weaknesses
My main concern with this paper is the academic contribution to the ICLR community. While CORE presents a strong engineering implementation that may benefit future research and applications, the general idea that uses RL for compression optimization can be found in papers from years ago, such as  [1], and some newer works from months ago use task performances as rewards, such as [2] and [3]. The authors need to better discuss key differences between these works (other than better engineering implementation), and the current related works section is insufficient in my view. Moreover, the authors should consider some e2e RL baselines in experiments. 

At the same time, using task performances as rewards itself is somewhat concerning from a high level, because the compressor model is leaked with task labels, and will optimize towards "solving the problem", rather than "being faithful to the evidence". In other words, it seems to me that the compressor model will tend to solve the problem itself, and may limit task transfer performances on more challenging reasoning tasks (which I suggest the authors experiment with) rather than direct IR tasks such as those evaluated in the paper. 

Some misleading terms exist, such as "loselss," which typically refers to information lost from the original evidence instead of performance losses. In order to claim lossless, the authors should conduct studies to show that key information related to the (comprehensive and unbiased, not just performance-oriented) reasoning of the problem is preserved. 


[1] Discrete Prompt Compression with Reinforcement Learning, Jung and Kim, 2023
[2] Oreo: A Plug-in Context Reconstructor to Enhance Retrieval-Augmented Generation, Li and Ramakrishnan, 2025
[3] TACO-RL: Task Aware Prompt Compression Optimization with Reinforcement Learning, Shandilya et al. 2024

### Questions
Please refer to the weakness section. I do not have particular confusions about the current manuscript as it is well written.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes CORE which is an end-to-end framework for context compression in RAG for LLMs using RL. CORE uses downstream task performance as direct reward when training the compressor. The method is evaluated on four QA datasets with EM and F1 as the evaluation metrics, with RECOMP and NoiseFilter as baselines.

### Strengths
- The paper focuses on the challenges of context explosion in RAG paradigms, providing a good empirical and conceptual motivation.
- The ablation studies on document # and cross backbone LLMs are comprehensive and demonstrate effectiveness.
- The paper is mostly clear in the presentation and has an anonymous code repository.

### Weaknesses
- Missing experiments with strong compressor baselines and datasets on context compression. The paper mostly tested on QA datasets with short retrieved document length. It would be neccessary to experiment on QA benchmark with long document context, including ZeroScrolls and LongBench. Also compare with baselines including LLMLingua-2, LongLLMLingua.
- Could the authors clarify whether their approach is robust against adversarial retrievals or irrelevant/noisy context (e.g., randomly shuffled or negative docuements)?

### Questions
See the weakness.

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
4

### Summary
The paper proposes CORE, a novel method for lossless context compression in Retrieval-Augmented Generation (RAG) models. It aims to address the inefficiencies in current RAG systems, where increasing the number of retrieved documents leads to a higher computational cost without necessarily improving the model's performance. CORE leverages reinforcement learning to optimize a compression policy that improves task performance without requiring predefined compression labels. The method uses downstream task performance as feedback, iteratively refining the compression strategy to prevent performance degradation. The authors demonstrate its effectiveness with experimental results across four benchmark datasets.

### Strengths
1. The paper introduces an innovative approach by combining reinforcement learning with end-to-end optimization to address the lossless compression problem in RAG models. This method is distinct from previous heuristic-based approaches that fail to align compression with downstream task performance.

2. The use of Group Relative Policy Optimization (GRPO) for end-to-end training of the compressor without requiring predefined summaries or labels is a clear strength. This approach ensures the compression is aligned directly with the task's requirements.

3. The paper claims that CORE not only prevents performance degradation but also improves task performance (up to 3.3 EM points) with significantly reduced computational costs. The method demonstrates strong generalization across different datasets and model architectures. The ablation study shows that both the distillation warm-up and GRPO phases are critical, and the results remain consistent across multiple architectures (e.g., LLaMA, Qwen).

### Weaknesses
1. The method's computational efficiency is presented in terms of compression ratios, but there is little discussion of how the approach scales when both the retrieved documents and the LLMs are larger. For example, the computational overhead of training the compressor and fine-tuning on new tasks could be prohibitive for very large-scale deployment.

2. While the results are promising, the paper lacks a detailed discussion of the interpretability of the compressed summaries generated by CORE. In real-world applications, understanding why certain documents or parts of documents are prioritized for compression could help improve trust in the system. The case studies presented (on NQ and 2Wiki datasets) are useful, but further case studies or qualitative analysis of failure cases could strengthen the argument for the method's robustness.

3. The paper mostly compares CORE with a few existing compression methods (e.g., RECOMP, NoiseFilter-IB) and traditional RAG. While these are important baselines, the method could be better contextualized within the broader field by incorporating more recent compression techniques or exploring how CORE performs against other forms of task-specific compression like query-guided methods.

### Questions
1. Could you provide additional insights into how CORE scales with the size of both the LLM and the number of retrieved documents? How would it perform in environments where the retrieved document pool and LLM models are very large?

2. What is the computational cost of training the CORE model compared to traditional RAG methods? Specifically, how does the fine-tuning process scale when switching between different LLMs?

3. How can we better understand the decisions made by the compressor during the document selection and compression process? Is there a way to visualize or explain why specific content is prioritized?

4. Does CORE have any potential applicability in non-textual domains (e.g., image captioning, multimodal RAG systems)?

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
The paper introduces CORE, a method designed to address the computational challenges faced by Retrieval-Augmented Generation systems in large language models. The central aim is to achieve lossless compression of retrieved documents, improving task performance while maintaining computational efficiency. The method uses reinforcement learning for end-to-end optimization without the need for predefined compression labels, which is an advantage over previous heuristic-based approaches.

### Strengths
1. The core idea of using RL for context compression, leveraging downstream task performance as a feedback signal, is innovative.
2. The methodology is presented with a detailed explanation of the training process, including the use of distillation and reinforcement learning. The figures and experimental setup are clear and easy to follow, making the approach easy to replicate.

### Weaknesses
1. The paper does not compare CORE to end-to-end reinforcement learning models designed for question-answering tasks. Including such an ablation study would be valuable, as it would provide insight into whether the additional context compression mechanism proposed by CORE is necessary or whether end-to-end RL models could achieve similar or better results without the need for a separate compression step. 
2. The paper does not test the model on domain-specific datasets and lacks an evaluation of whether the CORE can learn a generalized context compression ability through reinforcement learning.
3. The paper claims that CORE reduces computational overhead, but reinforcement learning itself can be computationally expensive, especially during the training phase. The paper does not provide an in-depth analysis of the trade-offs in terms of training time and efficiency compared to other approaches.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
