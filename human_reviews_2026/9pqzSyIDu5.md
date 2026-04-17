# CP Merging: Joint LoRA Merging using Canonical Polyadic Decomposition

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Large language models (LLMs) are often fine-tuned for specific tasks using Low-Rank Adaptation (LoRA), an efficient method that adds small, task-specific modules called LoRA adapters to a pre-trained base model. However, a major challenge arises when merging multiple LoRA adapters trained on different data sources for a specific task: it often leads to task interference, which degrades the model's performance. While recent SVD-based LoRA merging methods have shown promise by decomposing adapters into orthogonal components and keeping only the most important ones, they have an important limitation: These methods process each adapter independently,  overlooking potential interactions between different tasks. To address this, we propose a novel LoRA merging method using joint Canonical Polyadic (CP) decomposition (CP merging). We first combine the LoRA adapters into a single third-order tensor. Then, we apply CP decomposition to this tensor to disentangle factors that are unique to each task from those that are shared across tasks. This joint factorization method helps reduce cross-task interference without losing important information. Our extensive experiments on NLP tasks demonstrate that CP merging yields superior performance compared to the existing SVD-based baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the problem of task interference when merging multiple Low-Rank Adaptation (LoRA) adapters in large language models, which often leads to performance degradation. It proposes a novel method called CP Merging, which uses Canonical Polyadic (CP) decomposition to jointly factorize LoRA adapters into a third-order tensor, disentangling task-specific and shared components. The core contribution is a unified factorization approach that reduces interference compared to existing SVD-based methods. Experimental results on held-in, held-out, and zero-shot tasks using models like Phi-3 and Mistral show that CP Merging outperforms baselines in metrics like Rouge-L and accuracy, particularly in question answering and coding tasks.

### Strengths
1. The paper is well-organized and easy to read, with intuitive figures illustrating key concepts.

2. The CP decomposition framework is derived rigorously as a generalization of SVD, with clear explanations of how factor matrices represent task-specific and shared components, supported by tensor operations.

3. Comprehensive experimental evaluation across diverse scenarios with consistent comparisons to SOTA methods. I really appreciate that the author conduct both held-in and held-out tasks, which is a significant part that is always ignored by existing model merging methods.

### Weaknesses
1. The baselines are somewhat out-of-date, and more recent LoRA-based[1,2] merging methods should be compared to validate the effectiveness of the proposed method.

2. Experiments are only conudcted on LLM. Extending the application scenarios like MLLMs would be of great significant since there is no difference between parameter-efficient tuning of LLM and MLLM other than the structure.

3. Different LoRA fine-tuning strategies like DoRA and so on are suggested to be employed to demonstrate the effectiveness.

4. Although the concept of CP is well defined by previous work, the authors had better explain in detail the concrete adaptation of how to fit it for model merging, or just an application of previous work. For example, the method relies on assumptions on task interdependence that CP decomposition naturally disentangles shared/task-specific factors, but no failure cases or scenarios where factorization degrades performance are discussed. 

5. limited scale of model: extending experiments to larger models (e.g., >7B parameters) would better illustrate the effectiveness.

### Questions
1.  What does oracle mean in Table 1? If it represents the results of individual training, what causes some model merging methods to surpass the theoretical upper bound in some datasets? 

2. How does CP Merging perform under extreme task interference scenarios (e.g., adversarial tasks or safety tasks), and are there failure modes not covered in current experiments?

3. What is the computational overhead of CP tensor operations compared to SVD-based methods, and could this be quantified in terms of time/memory complexity?

4. Generalization limitations: On held-out tasks, uniform merging sometimes outperforms CP Merging, suggesting unresolved generalization challenges, but this is not deeply analyzed. Could the author provide a deeper analysis?

[1] RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness.

[2] LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition.

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
This paper aims to improve the performance of merging LoRA-trained models. While previous SVD-based merging methods show improvement, their limitation is that they decompose each adapter independently and do not utilize the interaction among tasks. In this paper, the authors propose to use Canonical Polyadic decomposition before merging the models. Extensive experiments show the effectiveness of their method. Ablation studies provide multiple insights based on their method.

### Strengths
- The proposed method is simple and elegant, but effective and compatible with other merging methods.
- The overall writing of this paper is good and easy to follow. Visual elements are useful to help understanding.
- Experiments are comprehensive, including diverse ablation studies.

### Weaknesses
- In Sec.4.1, the author discussed that in-domain multi-task learning will be evaluated on unseen tasks, which I think does not match the conventional setting of multi-task learning. For multi-task learning and model merging, the main goal is still to improve the in-domain performance. Could you provide more literature to support the claim in the paper?
- The motivation to use Canonical Polyadic decomposition is not clear to me. The author claims that previous methods decompose the task adapters separately (lines 220-221), but actually KnOTs decomposes all the adapters together to find a shared subspace for all adapters. Moreover, why decomposing all adapters together can improve the performance is not well studied or explained.
- Since larger models are more difficult to (re-)train, it is more necessary to achieve better performance on larger models. This leads to two challenges. First, while this paper mainly uses a 3B model in the experiments, the effectiveness of CP merging under larger models is not well studied. Second, the efficiency of CP merging when the model size and task number increase is not studied.

### Questions
Please see the weaknesses.

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
This paper introduces CP-Merging, a novel method for merging LoRA adapters. Unlike existing approaches that apply Singular Value Decomposition (SVD) independently to each adapter, CP-Merging first stacks multiple adapters into a third-order tensor and then jointly factorizes them using Canonical Polyadic (CP) decomposition. The authors argue that this joint factorization more effectively disentangles shared components from task-specific ones, thereby mitigating task interference. Experimental results across multi-task learning, zero-shot generalization, and skill composition show that CP-Merging outperforms SVD-based baselines.

### Strengths
The application of tensor decomposition, specifically CP decomposition, to LoRA merging is a novel and well-motivated approach. It provides a promising alternative to SVD-based methods for tackling task interference.

The paper includes a comprehensive evaluation across a diverse set of tasks and scenarios, convincingly demonstrating the method's effectiveness.

The ablation studies on the CP rank R and the scaling factor α are thorough and provide valuable insights into the method's behavior and sensitivity.

### Weaknesses
1. There appears to be an error in Table 2. There are duplicate entries for the "TSV Merging" method, each presenting different results. It is unclear if this is a typographical error or if they represent different experimental settings, as this is not clarified in the text.

2. The evaluation on the Flan dataset (Table 1) relies heavily on the ROUGE-L metric. While useful, ROUGE-L primarily measures lexical overlap and may not adequately capture semantic correctness or logical coherence. The inclusion of other metrics could provide a more comprehensive picture of performance.

3. The paper demonstrates that performance is sensitive to key hyperparameters like the CP rank R and the scaling factor α. However, it lacks a discussion on a principled method or heuristic for selecting these hyperparameters in practice, which is a notable limitation for real-world applications.

### Questions
1. The analysis of task interference in Figure 4 is limited to the first 11 layers (0-10). The paper suggests interference is higher in lower layers. Could the authors comment on or provide data for the trend of the CP-STI score in deeper layers of the model?

2. The proposed CP-Merging method is presumably more computationally expensive than baselines like uniform averaging or independent SVD. The paper would be strengthened by an analysis of the time and space complexity of CP-Merging, especially as the number of tasks (N) grows. This is crucial for assessing its scalability.

### Soundness
2

### Presentation
2

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
The paper tackles the problem of merging multiple LoRA adapters - each fine-tuned on different tasks - into a single model without large performance degradation from task interference.
Prior SVD-based merging decomposes each adapter independently, potentially losing cross-task structure. The proposed method (“CP merging”) instead stacks LoRA adapters into a third-order tensor and performs joint CP decomposition, separating shared vs. task-specific factors.
Experiments on NLP tasks show improved performance over SVD-based baselines, notably +3.19 Rouge-L on held-in tasks and +5.69 Rouge-L on held-out tasks.

### Strengths
- CP decomposition naturally models multi-task structure via shared and task-specific factors.
- Repeated gains over SVD methods across held-in, held-out, and skill-composition tasks.
- Handles multi-adapter reuse in modern LoRA libraries.

### Weaknesses
- Technical novelty is moderate. CP decomposition is standard; applying it to LoRA merging is incremental.
- No detail in parameter overhead or computational cost of CP. Tradeoffs need discussion.
- SVD approaches are compared, but unclear whether more recent/strong merging approaches (e.g., LoRI) were included.
- Evaluation scope is only within NLP. Unclear generality to other modalities.

### Questions
- CP decomposition scales poorly. What are practical computational costs vs. SVD?
- Does performance degrade when tasks are highly dissimilar?
- How does method behave when merging >10 LoRAs (real LoRA hubs)?

### Soundness
2

### Presentation
3

### Contribution
2
