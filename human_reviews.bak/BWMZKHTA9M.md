# Suppressing recency bias through implicit task in task-agnostic continual adaptation for foundation language models

- Decision: Reject
- Scores: 3, 6, 3, 5, 6, 1

## Abstract
Foundation language models have significantly advanced natural language processing but face challenges such as catastrophic forgetting when adapting to dynamic environments with diverse tasks. Recently, among the continual learning (CL) methods for these models, model architecture expansion methods have been spotlighted due to the growth of parameter-efficient fine-tuning (PEFT) methods. However, these methods need to store past PEFT adapters for each task and require task identifiers (task IDs) to distinguish each task, thus limiting their applicability in task-agnostic settings. They also overlook recency bias, where models focus overly on current tasks at the expense of past knowledge. 
To address these issues, we propose suppressing recency bias (SRB) by using the concept of implicit tasks. SRB assigns a fixed-size adapter to an implicit task, recursively storing historical knowledge through arithmetic operations with current adapters at every time step instead of task IDs. This arithmetic mitigates recency bias by integrating non-overlapping information between historical and current adapters. 
Our approach requires only simple arithmetic operations without backpropagation, minimizing additional computation, and allocates a fixed-size adapter to the implicit task, resulting in low memory requirements. We evaluate SRB on CL benchmarks for foundational LMs. Experimental results demonstrate that SRB outperforms state-of-the-art methods, achieving superior generalization performance across various task sequences and models by effectively mitigating recency bias.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper addresses the problem of recency bias in foundation models, which causes models to prioritize recent tasks at the expense of past knowledge. The main contribution of this work is a novel method called "Suppressing Recency Bias"(SRB), which combines current and past knowledge using arithmetic operations without requiring additional back-propagation, thus ensuring minimal computational overhead. The key highlights of SRB that authors claim include:
1. SRB eliminates the need for task IDs by using an implicit task that aggregates past knowledge through simple arithmetic.
2. SRB maintains historical knowledge from past tasks while learning new tasks by integrating only unique information, thus reducing redundant learning.
3. SRB requires minimal additional memory space and computations to SOTA methods, outperforming methods like LoRA and IncLoRA effectively.

### Strengths
1. It is a novel approach to handling one of the major challenges in continual learning: recency bias. By leveraging an implicit task to integrate historical knowledge, SRB addresses the common problem where models tend to overly focus on recent tasks at the expense of retaining knowledge from earlier ones. This approach could be beneficial across a wide range of continual learning applications.

2. SRB operates without requiring task IDs, which is a significant advantage in real-world scenarios where tasks are not explicitly defined. Task-agnostic continual learning is particularly challenging, and the SRB method makes a good improvement by showing it’s possible to achieve effective continual adaptation without relying on task-specific information.

3. The experimental results show SRB outperforms or is competitive with state-of-the-art methods like LoRA, IncLoRA, and O-IncLoRA on some standard CL benchmarks. SRB demonstrates superior generalization and knowledge retention across tasks, particularly in comparison to methods prone to catastrophic forgetting.

4. The authors provide a clear description of the experimental setup, benchmark datasets, and hyperparameters settings. This transparency supports reproducibility of this work.

### Weaknesses
The paper presents a novel approach to addressing recency bias in task-agnostic continual learning. However, there are some potential weaknesses and limitations in this work.

1. The SRB method introduces several hyperparameters (such as a, b, and c for controlling the influence and regularization of task vectors). The paper notes that hyperparameter tuning is essential for SRB’s performance. This reliance might limit the model’s robustness, as it may require fine-tuning for different tasks or models, reducing its practicality in real-world, dynamic settings where such tuning isn’t feasible.

2. Although SRB is designed for task-agnostic settings, the scalability to larger or longer task sequences is not thoroughly explored. For instance, the implicit task mechanism might become less efficient or struggle to represent historical knowledge accurately when handling an extensive range of tasks. A larger-scale experiment would provide insights into how SRB performs with extensive, varied task sequences.

3. Since SRB relies on arithmetic operations to balance historical and current knowledge, it may struggle in environments where task characteristics change quickly or drastically. The implicit task representation could fail to adapt promptly in such settings, potentially limiting SRB’s performance on tasks that require quick, context-sensitive adaptation.

### Questions
1.  In the regularization process (Equation 13), how to determine the effectiveness of the orthogonal projection for suppressing recency bias? Were there other regularization techniques or projections you considered, and if so, what made this one preferable? In addition, if the implicit task vector is orthogonal to \tau_t, then the projections becomes zero, is there any additional strategy to prevent this situation?

2.  In task-agnostic settings, are there specific application domains where SRB’s approach to suppressing recency bias is particularly valuable? Conversely, are there domains where the implicit task approach may struggle, such as with tasks that have low overlap or are highly distinct?

3. You mention using average accuracy for performance evaluation, but did you also measure other indicators like backward and forward transfer? If so, how did SRB perform on these metrics, especially in preserving knowledge from earlier tasks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates task-agnostic continual learning using foundational language models. To tackle the issues associated with previous PEFT methods, the authors introduce a novel approach called Suppressing Recency Bias (SRB), which enables the model to adapt to current data while retaining historical knowledge. By leveraging the design of implicit task vectors, SRB-based models can be trained to adapt without the need for task IDs. Experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1. The idea of employing a fixed-size adapter to recursively store the historical knowledge of an implicit task appears to be interesting and novel.  The proposed method can adapt to task-agnostic datasets even without task IDs.
2. The authors have conducted a thorough series of experiments to validate their proposed method.
3. The results regarding information retention indicate that the SRB significantly outperforms other models, showcasing its ability to adapt to new tasks while preserving performance on previous ones with minimal degradation.

### Weaknesses
1. The paper lacks a detailed explanation of the motivation behind the model design, particularly concerning the implicit task vector and the regularization term.
2. The authors should apply the proposed method to additional foundational models (e.g., BERT) to further validate its effectiveness, similar to previous studies.
3. The explanation of how to calculate the task vector is somewhat unclear. For instance, as stated in line 288, the task vector for the current task is defined as $\tau_t=w_t-w_0$. However, the process for calculating the task vector for the next task is not elaborated upon.

### Questions
1. Why does the implicit task vector act as a low-pass filter, limiting diversity?
2. What are the results of LLaMA3 and LLaMA3-chat on Orders 4, 5, and 6?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper presents **Suppressing Recency Bias (SRB)**, a method designed for task-agnostic continual learning in foundation language models (LMs). SRB introduces the concept of an implicit task that integrates knowledge recursively, minimizing the reliance on task identifiers and addressing recency bias—an issue where models disproportionately prioritize current tasks at the expense of previous ones. This approach achieves low memory overhead by requiring only fixed-size adapters and using simple arithmetic operations for updating, without the need for backpropagation. The paper demonstrates that SRB outperforms existing continual learning (CL) methods in both standard and extended task sequences by maintaining superior performance across diverse tasks and reducing recency bias.

### Strengths
- **Task-Agnostic Adaptability**: SRB excels in task-agnostic settings, removing the dependency on task identifiers and preserving performance across tasks.
- **Computational Efficiency**: The method achieves this while maintaining minimal computational and memory overhead, only needing simple arithmetic operations.
- **Reduction of Recency Bias**: The introduction of implicit task vectors effectively mitigates recency bias, ensuring that historical information is preserved during adaptation.
- **Empirical Validation**: Experimental results on standard and long CL benchmarks show that SRB outperforms other state-of-the-art methods in accuracy and efficiency.

### Weaknesses
- **Hyperparameter Sensitivity**: The approach relies on specific hyperparameters for performance, which might affect its generalizability without tuning.
- **Comparison Scope**: While the paper benchmarks against key methods, additional comparisons with more diverse baseline approaches, such as advanced replay-based strategies, could strengthen its conclusions.
- **Task Transition Analysis**: The paper could benefit from deeper analysis of how SRB handles transitions between tasks, especially in complex sequences involving highly dissimilar tasks.
- **Overhead of Hyperparameter Tuning**: Although SRB shows robustness, the paper notes fixed hyperparameters, implying that different task sequences might require adjustment for optimal performance.

### Questions
- Can the authors elaborate on how the method performs under highly diverse task sequences, such as tasks involving distinct domains (e.g., code, legal text, medical literature)?
- What strategies can be used to fine-tune the hyperparameters (a, b, c) without extensive trial and error?
- How does SRB handle tasks that might involve conflicting objectives (e.g., creative writing vs. technical report summarization)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper propose a continual learning algorithm that projects LoRA parameter updates to a subspace defined by an "implicit task" to mitigate recency bias and mitigate forgetting. Experiments on CL benchmarks demonstrate performance improvements over baselines, especially those that also learn LoRA.

### Strengths
- The performance improvement is clear compared to the baselines.
- The description of the proposed approach is clear

### Weaknesses
1. The intuition behind the performance improvement is not quite clear to me.

In a high level, the approach finds a direction of model parameter update that represents previous tasks (termed as "implicit tasks" in this work), and apply regularization in weight updates while learning new tasks by "pushing back" the update a bit towards a direction orthogonal to the "implicit tasks".

- It seems Orthogonal LoRA by Wang et al. 2023 does a similar job. What is the design that makes the approach improve over Orthogonal LoRA?  Please highlight the key methodological differences and explain how these differences contribute to the improved performance observed in the experiments. 

- The approach conceptually shares similar idea to regularization based CL approaches like L2 regularization (which pushes back parameter updates to their initial states before fine-tuning). But in Table 2 L2 regularization performs very poorly. What could be the reason? Please provide a more in-depth analysis of why the proposed approach outperforms L2 regularization.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Suppressing Recency Bias (SRB) method to mitigate catastrophic forgetting during the continuous learning process of language models (LMs) without task IDs. SRB introduces an additional implicit task adapter and designs an update mechanism to appropriately integrate knowledge learned from the current task into the implicit adapter. The updated implicit adapter is then used to initialize the learning of new data. The designed update mechanism reduces duplicated information in classical Model Architecture Expansion (MAE) methods while balancing the increase in adapter diversity and the reduction of recency bias. The method outperforms previous task-agnostic methods and MAE methods using task IDs on benchmark tasks. Ablation studies demonstrate the effects of hyperparameters on reducing recency bias and increasing diversity, and show that SRB is not sensitive to hyperparameters within a certain range.

### Strengths
- The paper is clearly written and easy to understand. Figure 1 effectively contrasts SRB with other methods, and Figure 2 helps in understanding the purpose of the update mechanisms in SRB.
- SRB achieves a balance between increasing adapter diversity and reducing recency bias through a carefully designed update mechanism, which is validated by detailed ablation studies.
- SRB is simple and easy to use, and it outperforms existing MAE methods in terms of computational and memory costs.

### Weaknesses
- The paper lacks a clear explanation of the division and training situation for $(D_1, …, D_T)$ in the SRB setting. Does each $D_t$ correspond to the data contained in a default training batch, or is it a manually divided portion of the dataset?
  - If it corresponds to data in a single batch, how many times will this data be updated? What is the impact of different batch sizes on the relevant hyperparameters?
  - If it is a manually divided portion of the dataset, how is the division performed? If the division ensures that data from one task forms a single $D_i$, then SRB actually leaks task ID information. If data from each task is evenly divided into N parts, it still contains some task information. In a realistic task-agnostic scenario, it is difficult to ensure that data from different tasks are divided into different groups. The reviewer would like to see the performance of SRB when data from sequential tasks are included in the same $D_t$.
- The results section lacks methods such as multi-task learning or task experts as performance upper bounds for reference. Adding this reference would help readers better understand the improvements and limitations of the proposed method. Additionally, Figure 3(a) in Section 5.1 could also provide the performance of task experts as a reference.

### Questions
- Please provide more detailed explanations regarding the division of $D_i$ as mentioned in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
1

### Rating Number
1

### Confidence
2

### Summary
The paper proposed a method for continual learning. But I am unsure what problem it is solving.

### Strengths
The proposed method is different.

### Weaknesses
w-1. Your writing at the beginning sounds like you are working on continual learning of language models and adapting them to different domains. But I believe you are really doing normal continual learning using LMs at feature extractors since your tasks are text classification tasks. 

w-2. The paper did not state what setting of continual learning it works on. I think that this method works on task incremental learning (TIL). However, the first three baselines are commonly used for class incremental learning. Please clarify. For task incremental learning, the problem of forgetting is largely solved. Please check out [1, 2, 3, 4, 5, 6, 7].

w-3. Related to w-2. The paper says that the approach is task-agnostic, which means that no task-id is given, but which means it does class-incremental learning (CIL). But for CIL, by definition, there is no task-id information given. It is very confusing. Please make it clear which continual learning problem you are solving. If you are solving CIL, you should compare your method with another set of SOTA baselines. 

w-4. It is not true that architecture approaches add one adaptor for each task. Also see [1, 2, 3, 4, 5, 6, 7]. Most of the existing methods do not need to task-id either, assuming that you are solving the TIL problem. 

w-5. What is the difference between recency bias and catastrophic forgetting? Catastrophic forgetting is about focusing on the present and forgetting the past, which is the recency bias. 

w-6. Regarding baselines, your non-LoRA baselines, EWC,  Replay, and LwF, are very old and not the state of the art. Again, please check out [1, 2, 3, 4, 5, 6, 7]. 

w-7. What is average accuracy? Please give the definition. In continual learning, there are at least two accuracy measures.  

w-8. Having a separate adaptor for each model is not in the spirit of continual learning, which aims to use the same network learning multiple tasks. Please give the memory requirement of your approach. 

w-9. Please give the performance upper bound for the continual learning problem that you are solving. 

w-10. The writing of the paper needs significant improvement. The paper is confusing. I am not even sure what problem you are solving. If you are solving CIL, how do you deal with documents that may belong to two different classes in two different tasks? For example, a topic-specific document may contain a positive sentiment and a review of a problem may be classified to its product category.  

     [1]. Serra et al. Overcoming catastrophic forgetting with hard attention to the task. ICML-2018.
     [2]. Wortsman et al. Supermasks in superposition. NeurIPS-2020.
     [3]. Ke et al. Achieving Forgetting Prevention and Knowledge Transfer in Continual Learning.  NeurIPS-2021.
     [4]. Lin et al. TRGP: Trust region gradient projection for continual learning. ICLR-2021.
     [5]. Lin et al. Beyond not-forgetting: Continual learning with backward knowledge transfer. NeurIPS-2022.  
     [6]. Ke et al.  Sub-network Discovery and Soft Masking for Continual Learning of Mixed Tasks.  EMNLP-2023.
     [7]. Dissecting learning and forgetting in language model finetuning. ICLR-2024.

### Questions
See the previous section

### Soundness
1

### Presentation
2

### Contribution
2
