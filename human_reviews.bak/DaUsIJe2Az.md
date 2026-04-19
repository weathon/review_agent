# Continual Learning via Continual Weighted Sparsity and Meta-Plasticity Scheduling

- Decision: Reject
- Scores: 3, 6, 5, 3

## Abstract
Continual Learning (CL) is fundamentally challenged by the stability-plasticity dilemma: the trade-off between acquiring new information and maintaining past knowledge. To address the stability, many methods keep a replay buffer containing a small set of samples from prior tasks and employ parameter isolation strategies that allocate separate parameter subspaces for each task, reducing interference between tasks. To get more refined, task-specific groups, we adapt a dynamic sparse training technique and introduce a continual weight score function to guide the iterative pruning process over multiple rounds of training. We refer to this method as the continual weighted sparsity scheduler. Furthermore, with more incremental tasks introduced, the network inevitably becomes saturated, leading to a loss of plasticity, where the model's adaptability decreases due to dormant or saturated neurons. To mitigate this, we draw inspiration from biological meta-plasticity mechanisms, and develop a meta-plasticity scheduler to dynamically adjust these task-specific groups' learning rates based on the sensitive score function we designed, ensuring a balance between retaining old knowledge and acquiring new skills. The results of comparison on popular datasets demonstrate that our approach consistently outperforms existing state-of-the-art methods, confirming its effectiveness in managing the stability-plasticity trade-off.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper presents an approach to address the stability-plasticity dilemma in continual learning by introducing two main components: a continual weighted sparsity and a meta-plasticity scheduler. The method also uses a replay buffer. This makes it a hybrid continual learning approach that combines parameter isolation, regularization, and replay. The continual weighted sparsity component iteratively prunes the network with gradually increasing sparsity over multiple training rounds to identify task-specific neuron groups. The meta-plasticity scheduler then dynamically adjusts learning rates for different neuron groups based on their sensitivity scores to balance knowledge retention and acquisition. The authors evaluate their method on standard benchmarks (CIFAR-10, CIFAR-100, and Tiny-ImageNet) in both class-incremental and task-incremental learning settings.

### Strengths
- The paper is well-structured, making it easy to follow.
- The paper explores a combination of connection rewiring with individually adjusted learning rates at the neuron level. While both approaches have been explored separately in prior work, their integration provides an additional data point in addressing the stability-plasticity dilemma in continual learning.

### Weaknesses
## Limited Discussion of Related Literature (Why I gave 1 to Presentation)
While the paper presents interesting ideas, there are several important concerns regarding the positioning and novelty of the work:

- The proposed "meta-plasticity" mechanism is similar to existing regularization methods (e.g., elastic weight consolidation, synaptic intelligence, and memory aware synapses). These methods similarly modulate learning rates across different connections. A more thorough discussion differentiating the proposed approach from these established methods would be valuable, particularly given that these regularization approaches are explicitly categorized as "meta-plasticity" in the "Biological underpinnings for lifelong learning machines" paper (page 204) they cite to motivate the idea.

- The paper's claimed novelty in iterative pruning appears to overlap significantly with existing work:
  * NISPA (https://arxiv.org/abs/2206.09117, ICML 2022), which presents similar ideas about iterative connection rewiring while maintaining constant sparsity Also see, Space-Net (https://arxiv.org/abs/2007.07617) and AFAF (https://arxiv.org/abs/2110.05329) - this is not an exhaustive list.

- While the current work introduces more sophisticated rewiring strategies and meta-plasticity mechanisms compared to NISPA's random selection approach (or other mentioned works), the fundamental principles appear derivative. The paper would benefit from a clearer justification for why these more complex mechanisms are necessary and advantageous.

## Experimental Design and Methodology Concerns (Why I gave 1 to Soundness)
Results appear to be identical to the values in the TriRE paper, which would be acceptable with proper attribution (but I do not see any attribution). However, a more serious concern is the architectural discrepancy: while TriRE uses ResNet-18, this work claims to use ResNet-50, potentially creating an unfair comparison due to significantly different parameter counts.

While the paper includes ablation studies comparing different parameter settings for individual components, it lacks a comprehensive analysis justifying the necessity of combining all three approaches (replay, parameter isolation, and meta-plasticity). The ablations focus on tuning parameters within each component rather than demonstrating why the full combination of components is required to achieve the reported performance gains.

Critical details about buffer sizes in experiments are missing, making it difficult to properly evaluate the replay component

### Questions
- Could you provide an analysis of the computational cost for each component of your method (replay, regularization, and parameter isolation) and explain why all these components are necessary for the overall approach?
- Could you clarify how neurons are defined in convolutional layers (e.g., whether they represent channels/feature maps) and explain specifically how connection rewiring is implemented in convolutional layers?

### Soundness
1

### Presentation
1

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
The paper presents a framework for Continual Learning (CL) that addresses the stability-plasticity dilemma through two key innovations: the Continual Weighted Sparsity Scheduler and the Meta-Plasticity Scheduler. The former iteratively prunes neurons and connections to create task-specific groups, while the latter dynamically adjusts learning rates based on sensitivity scores. Experimental results demonstrate that this approach outperforms existing methods across various datasets, effectively balancing knowledge retention and adaptability to new tasks.

### Strengths
1. The proposed framework integrates a Continual Weighted Sparsity Scheduler and a Meta-Plasticity Scheduler, effectively addressing the stability-plasticity dilemma in Continual Learning (CL). This allows the model to retain previously learned knowledge while adapting to new tasks, leading to improved overall performance.

2. The use of iterative pruning and dynamic learning rate adjustments based on sensitivity scores enables the model to maintain flexibility and adaptivity. This approach allows for fine-tuning of connections, ensuring that the model can efficiently learn new information without significant interference from prior tasks.

3. Experimental results demonstrate that the proposed method consistently outperforms state-of-the-art CL techniques across various datasets, including CIFAR-10, CIFAR-100, and Tiny-ImageNet. This highlights the robustness and effectiveness of the framework in handling complex and incremental learning scenarios.

### Weaknesses
1.Network Saturation Challenges: The proposed framework may face limitations as the number of tasks increases, leading to network saturation. This saturation can result in reduced adaptability and performance on new tasks, as the model may struggle to allocate sufficient resources to accommodate additional knowledge without interference from previously learned tasks.

2.Complexity of Implementation: The integration of both the continual weighted sparsity scheduler and the meta-plasticity scheduler adds complexity to the implementation. This complexity may pose challenges in terms of computational resources and tuning hyperparameters, making it less accessible for practical applications compared to simpler continual learning methods. Therefore, this paper should add a hyper-parameter analysis.

3.Limited Experiments: The experimental evaluation in this paper appears to be insufficient. The authors should extend their experiments to include more diverse datasets to better demonstrate the robustness and generalizability of their proposed method. 

4.Limited Improvement over SPARCL (NeurIPS 2022): The enhancements made in this framework compared to SPARCL (Sparse Continual Learning on the Edge) seem relatively incremental. A more detailed comparison could clarify the specific contributions and advantages of the proposed method over SPARCL, especially in terms of practical improvements and scalability.

5.Lack of Open-Source Code: The absence of code or implementation details in the supplementary material limits the reproducibility and practical applicability of the proposed framework.

### Questions
Q1: How does the proposed framework address the network saturation problem when scaling to a larger number of tasks, and what are the potential solutions to maintain performance without compromising the model's ability to learn new tasks?

Q2: Given the complex integration of both continual weighted sparsity scheduler and meta-plasticity scheduler:
i) How do different hyperparameters affect the model's performance?
ii) What is the optimal configuration of hyperparameters for different scenarios?
iii) How can we balance the trade-off between implementation complexity and model performance?

Q3: To what extent does the proposed method generalize across different datasets and task domains?
i) How does the method perform on more challenging and diverse datasets?
ii) Can the performance advantages demonstrated in the current experiments be replicated across a broader range of scenarios?
iii) What are the potential limitations or strengths of the method when applied to different types of data and learning tasks?

Q4: Time Overhead from Sparsity Scheduling and Neuron Selection: The proposed Continual Weighted Sparsity Scheduler and neuron selection process likely introduce additional computational time. It would be beneficial to include a comparative analysis of training time with and without these components to better understand the time cost associated with the framework.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a parameter-isolation algorithm that combines a continual weighted sparsity scheduler with a meta-plasticity scheduler to address the stability-plasticity trade-off in continual learning.

### Strengths
- Overall, the paper is well-organized and easy to understand.
- Detailed experiments support the effectiveness of the proposed algorithm.

### Weaknesses
- What is the primary contribution of the proposed $\textit{continual weighted sparsity scheduler}$ compared to [1,2]? Is it simply a novel combination applied to CL? Why is it necessary to gradually reduce $S_{t,n}$ during epoch training? During successive epochs, are previously selected neurons fixed, with only remaining neurons being considered for selection?
- In Eqn (6), what does $\delta$ represent? I checked paper [1], and it seems this should be $\partial$, please clarify.
- Based on steps 1 and 2, the algorithm obtains a set of groups of neurons and connections $\mathcal{G}=\{g_1,...,g_T\}$. I am curious about the extent of overlap among these groups across different layers- does this provide any insight? During testing, is the task ID required to be predicted and then select the corresponding $g_i$? If not,  does the algorithm need to store all groups $\mathcal{G}=\{g_1,...,g_T\}$, or would it be sufficient to store only $g_T$?
- In Eqn (7), $\omega_t^e$ is determined only after completing task t. Is the adjusted learning rate used for training task t+1, or is there a retraining step required here?
- The ablation study on the continual weighted sparsity scheduler is insufficient, and the description of the static condition is ambiguous. Does "fixed sparsity from scratch" mean the sparsity remains fixed throughout each training epoch or that the same fixed sparsity is applied across all layers? The authors should clarify these differences and interpret the performance under each condition in the ablation study.

[1] To prune, or not to prune: exploring the efficacy of pruning for model compression, 2017.
 
[2] Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science, 2018.

### Questions
See weakness 1-5. If my questions are effectively addressed, and after considering feedback from other reviewers, I would be open to increasing my score.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a pruning-based continual learning algorithm consisting of two steps. After learning the current task, multiple rounds of network pruning are conducted to form task-specific groups. In this process, essential neurons and connections are selected based on high activation and a continual weighted score, and the selected components are grouped by task. This approach allows for more effective preservation of knowledge for each task. In the second step, after completing the learning for each task, a meta-plasticity scheduler is updated, applying an independent learning rate for each neuron by increasing the learning rate for relatively inactive groups. Experimental results demonstrate that the proposed algorithm achieves superior performance compared to other baselines.

### Strengths
1. The proposed algorithm introduces the idea of using pruning methods to overcome the trade-off between stability and plasticity in continual learning scenarios. It consists of two steps, each with a clear objective that is specifically designed to achieve its respective goal.

2. Experimental results across various datasets and continual learning scenarios demonstrate that the proposed algorithm outperforms other baseline algorithms.

3. The algorithm's effectiveness is validated through visualizations of actual neuron groups, comparisons of plasticity and stability, experiments in long task scenarios, and an ablation study, providing a comprehensive analysis from multiple perspectives.

### Weaknesses
1. The proposed algorithm emphasizes the importance of carefully setting the hyperparameters, such as pre-defined sparsity, $\lambda$, $\alpha_1$, $\alpha_2$, $\beta_1$, and $\beta_2$, for effective utilization. However, this seems to involve a considerable number of hyperparameters.  

1-1) Although $\alpha_1$, $\alpha_2$, $\beta_1$, $\beta_2$, and pre-defined sparsity utilize already reported values, and experiments have been conducted on various values for pre-defined sparsity and $\lambda$, how can these parameters be configured in actual continual learning situations? This hyperparameter issue has been discussed as a significant concern in continual learning research, as highlighted in studies [1, 2, 3], so I believe a discussion on this aspect is essential.  

1-2) How sensitive is the proposed algorithm to changes in the values of $\alpha_1$, $\alpha_2$, $\beta_1$, and $\beta_2$?  

1-3) In studies such as [1,2], scenarios where many classes are learned in the first task and the remaining classes are learned evenly across multiple tasks are considered. In such scenarios, can the proposed algorithm still achieve superior performance using the same hyperparameters, or would it require finding the new best hyperparameter values?  

2. All experiments appear to be conducted based on ResNet-50. Recently, research using Vision Transformers in the computer vision domain has been actively used, reporting superior performance [4]. In this context, I wonder if the proposed algorithm would function effectively with Vision Transformers, which have entirely different characteristics. I think additional experiments are needed to verify this aspect.  

3. The proposed algorithm is structured in two stages, and particularly, I believe that calculating Equation (6) incurs additional computational costs. From this perspective, how does the computation cost of the proposed algorithm (e.g., FLOPs or training time) compare to other algorithms in Table 1? As shown in studies [3, 5], it is crucial to compare not only the performance of each continual learning algorithm but also their practical computation costs.  

4. All experiments were conducted using small image datasets with sizes of 32x32 or 64x64. As noted in studies [3, 6], each algorithm exhibits different trends depending on the dataset. In this light, I believe the authors should conduct experiments on at least the ImageNet-100 dataset (image size of 224x224) and compare the results with baselines to validate additional effectiveness.  

5. What was the memory buffer size used in the experiments? When varying this size, can the proposed algorithm consistently achieve superior performance?  

6. Considering the results and experiments in [6] (refer to GitHub), does the proposed algorithm outperform well-established algorithms in class-incremental learning, such as WA, BiC, and FOSTER? To concretely validate the superiority of the proposed algorithm, a comparison of performance and costs with these methods is necessary.

[1] Online hyperparameter optimization for class-incremental learning, AAAI 2024.

[2] Hyperparameter Selection in Continual Learning, CoLLAs 2024 Workshop.

[3] Hyperparameters in Continual Learning: A Reality Check, CoLLAs 2024 Workshop.

[4] A Comprehensive Survey of Continual Learning: Theory, Method and Application, TPAMI 2024.

[5] Computationally Budgeted Continual Learning: What Does Matter?, CVPR 2023.

[6] PyCIL: A Python Toolbox for Class-Incremental Learning, SCIENCE CHINA Information Sciences.

### Questions
I have included all questions, along with weaknesses, in the Weakness section, so please refer to it. I believe this paper proposes an interesting pruning-based algorithm to overcome the trade-off between stability and plasticity. However, the complexity of the algorithm and the need to set numerous hyperparameter values, along with various experimental concerns, make it difficult to provide a favorable assessment. I look forward to the authors' response in which they address any misunderstandings on my part and provide feedback on my review.

### Soundness
2

### Presentation
2

### Contribution
2
