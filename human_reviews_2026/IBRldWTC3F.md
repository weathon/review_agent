# Purifying Task Vectors in Knowledge-Aware Subspace for Model Merging

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Model merging aims to integrate task-specific abilities from individually fine-tuned models into a single model without extra training.
In recent model merging methods, task vector has become a fundamental building block, as it can encapsulate the residual information from finetuning. 
However, the merged model often suffers from notable performance degradation due to the conflicts caused by task-irrelevant redundancy in task vectors. 
Existing efforts in overcoming redundancy by randomly dropping elements in the parameter space involves randomness and lacks knowledge awareness. 
To address these challenges, in this study, we propose $\textbf{P}$urifying T$\textbf{A}$sk $\textbf{VE}$ctors (PAVE) in knowledge-aware subspace. 
Concretely, we sample some training examples from each task, and feed them into their corresponding fine-tuned models to acquire the covariance matrices before linear layers.
We then perform a context-oriented singular value decomposition, which accentuates the weight components most relevant to the target knowledge.
As a result, we can split fine-tuned model weights into task-relevant and redundant components in the knowledge-aware subspace, and purify the task vector by pruning the redundant components.  
To induce fair pruning efforts across models, we further introduce a spectral rank allocation strategy by optimizing a normalized activated pruning error. 
The task vector purification by our method as a plug-and-play scheme is applicable across various task vector-based merging methods to improve their performance.
In experiments, we demonstrate the effectiveness of PAVE across a diverse set of merging methods, tasks, and model architectures.
Remarkably, when integrated on the state-of-the-art merging method EMR-Merging with RoBERTa, our method yields a significant performance improvement of 4.1\% (from 80.18\% w/o PAVE to 84.28\%) on the GLUE benchmark, approaching to the average performance of 8 individual models, 85.55\%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed purifying task vectors (PAVE) to eliminate the task-irrelevant redundancy in task vectors, aiming to address performance degradation in model merging. As previous methods rely on randomness and lack knowledge awareness, PAVE utilizes data to filter out the task-relevant and redundant subspaces in parameters. They further proposed a rank allocation strategy to adaptively assign ranks to models. The experiments on diverse models and datasets show the effectiveness of their method.

### Strengths
- The motivation of this paper is novel: as previous methods focus on removing the conflicts in the whole task vectors, this paper proposes to eliminate noise and redundancy in the subspaces.
- The experiments are comprehensive in terms of models and datasets.
- The empirical results show the effectiveness of the main proposed components in their methods, including the purification and rank allocation.

### Weaknesses
- This method requires access to data from the target tasks, where the models come from. However, this assumption may not be valid in some cases (e.g., using HuggingFace models or when the data is private) and limits the application scenarios of this method.
- If my understanding is correct, the motivation of Sec.3.2 is to reduce the task conflicts. However, it is unclear to me why the proposed method in Sec.3.2 can help reduce the task conflicts. I think it would be better to clarify further the connection between the motivation and the proposed method.
- The motivation to use spectral rank allocation is not clear. Please see the questions.
- Though the performance is improved, PAVE introduces more computational cost. The trade-off between performance and computing should be studied to better justify the benefit of PAVE. Also, such a trade-off of using the rank allocation strategy should be studied in Tab.4, where the performance improvement is so limited.
- It seems that ThanoRA [1] proposed a similar method, but it is not discussed in this paper.

[1] Liang, Jian, et al. "ThanoRA: Task Heterogeneity-Aware Multi-Task Low-Rank Adaptation." arXiv preprint arXiv:2505.18640 (2025).

### Questions
- In lines 262-263, why is assigning the rank $r$ to each model non-trivial? I think it would be better to clarify the reason.
- As the author assumes the data access, I think it is better to evaluate the performance on some data-dependent merging algorithms, such as Fisher merging and RegMean.
- In Tab.3, the performance improvement is not as significant as that in Tab.1 and 2. Does it mean that as the model scale increases, the impact of PAVE becomes less? If PAVE cannot perform well with the scaling law, its contribution could be limited, as larger models are more necessary to be merged.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces PAVE (Purifying and Amalgamating Task Vectors), a novel method designed to enhance model merging performance by purifying task vectors. The core premise is that task vectors, which represent the difference between a fine-tuned model and a pretrained model, often contain redundant and noisy components that degrade performance when multiple models are merged. PAVE addresses this by employing a knowledge-aware subspace analysis. It computes a task-specific covariance matrix using data from the target task and then performs Singular Value Decomposition (SVD) to decompose the task vectors. This process allows for the isolation and pruning of task-irrelevant, low-rank components, thereby retaining only the most salient, high-rank information. A key advantage of PAVE is its modular, plug-and-play design, enabling its integration with various existing model merging methods, such as EMR-Merging, Ties-Merging, and Task Arithmetic. The authors demonstrate that this purification step leads to improved performance across a range of tasks and models.

### Strengths
PAVE is presented as a plug-and-play method, which can benefit various existing model merging methods.
The paper is well-organized and easy to follow.

### Weaknesses
The strategy of retaining high-rank components is quite similar to the approach used in the TSV method [1], which also focuses on keeping the high-rank components of task vectors. It would be beneficial to compare PAVE with TSV through an ablation study, such as replacing the SVD approach with WC by directly performing SVD on the task vectors T, where T represents the task vectors.
Although the pruning strategy that selectively retains high-rank components is reasonable, the paper does not fully justify the choice of rank for each model and each layer. Further explanation of how the rank is chosen or how it varies across tasks and layers would strengthen the argument.
The theoretical grounding for the minimization objective in Equation 7 is weak. The paper fails to provide a clear justification for why minimizing this objective correlates with improved final performance. Consequently, its use as a greedy metric for selecting task parameter ranks seems arbitrary, as there is no guarantee that a larger objective value for a specific subspace corresponds to its greater importance for the task.
The performance improvements with PAVE are not consistent across methods. While it shows significant improvement with EMR-Merging and Ties-Merging, the gains with Task Arithmetic are relatively small. This variability in adaptability reduces the robustness of PAVE as a plug-and-play solution, which is a critical characteristic for practical deployment.
As shown in Figure 4, the method relies heavily on the availability of task-specific data for the decomposition process. This could be a limitation in scenarios where only limited samples or no task-specific data are available, making the method less flexible and potentially unsuitable for some use cases.
 Also, the computational cost of performing SVD on high-dimensional task vectors for large-scale models can be significant. For models with many layers and parameters, this could lead to high memory and processing requirements, which may limit the scalability of the method in practical applications.

[1] Gargiulo A A, Crisostomi D, Bucarelli M S, et al. Task singular vectors: Reducing task interference in model merging[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 18695-18705.

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
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PAVE, a general plug-in method for improving model merging by removing redundancy at the task-vector level. The core steps are: for each fine-tuned model, a small number of examples are sampled from its corresponding task, layer input activations are obtained, and then used in any task-vector-based merging strategy. The paper also proposes a greedy spectral rank assignment algorithm that adaptively assigns the preserved rank to each layer/model to minimize the normalized "activation truncation error" at a given overall preservation rate. Experiments report robust performance improvements on multiple benchmarks.

### Strengths
1. The writing is quite easy to read and it was well-written
2.  The motivation is clear.
3.  The paper studies multiple base models
4.  The gains from their proposed method of sampling are convincing and comprehensive
5.  The method outputs a "cleaned task vector", which can directly replace the original task vector in the existing pipeline, resulting in low engineering migration costs.

### Weaknesses
1. The method requires sampling a certain number of training examples for each task to construct the covariance matrix. Although the authors emphasize that the number of samples can be small, this differs from strictly data-free scenarios, and the usability of the method is limited when real-world scenarios are constrained by privacy/inaccessibility of examples. The paper should discuss this more explicitly and provide degenerate behavior and alternatives for scenarios with no or very few samples.

2. The sensitivity of sample selection to task relevance was not adequately assessed: the paper points out that performance degrades when using irrelevant tasks, but does not provide quantitative guidance. In practical applications, the selection of these samples should be more clearly defined.

3. Compared to some methods based on subspace alignment or projection, PAVE focuses more on engineering and empirical evidence and lacks theoretical safeguards.

### Questions
1. As a plug-and-play method, this paper compares a limited number of methods. I want to know if this method can bring consistent improvements when applied to some of the latest SOTA methods, such as wudi-merging[1], tsv-merging[2], doge-merging[3], and iso-merging[4].
2. As both are plug-and-play methods, how does pave compare to awd merging[5] in performance?
3. How efficient is pave in actual computation?

[1] Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors

[2] Task Singular Vectors: Reducing Task Interference in Model Merging

[3] Modeling Multi-Task Model Merging as Adaptive Projective Gradient Descent

[4] Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces 

[5] Multi-task model merging via adaptive weight disentanglement

If the author can solve the question and the weakness well, i will raise my score.

### Soundness
3

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
4

### Summary
This paper presents PAVE, a plug-and-play framework for refining task vectors within a knowledge-aware subspace to enhance model merging. Departing from methods that employ random sparsification, PAVE constructs these subspaces using a context-oriented singular value decomposition (CO-SVD) guided by task-specific activations. This allows for the principled pruning of task-irrelevant components via a new spectral rank allocation strategy. PAVE's modular design enables its direct integration with existing task vector-based merging schemes. Comprehensive evaluations on GLUE benchmarks with RoBERTa and DeBERTa, alongside generative and computer vision tasks, confirm that PAVE achieves superior merging performance over strong baselines, including DARE, Task Arithmetic, and EMR-Merging.

### Strengths
1. The proposal to leverage task-specific activations and CO-SVD for decomposing fine-tuned weights is a principled choice. This offers a more targeted alternative to previous random or sparsity-based pruning approaches.
2. The introduction of a spectral rank allocation policy, optimizing for normalized activated pruning error across models, is a coherent and technically sound addition to enable fair pruning and avoid over-pruning critical tasks.
2. The method’s ability to serve as a plug-in for existing merging approaches (Task Arithmetic, Ties-Merging, EMR-Merging) increases its applicability and practical significance.
3. The manuscript is well-organized, the exposition is generally clear, and references to equations, figures, and tables are explicit and helpful.

### Weaknesses
1. The current literature review lacks the necessary depth to properly contextualize the paper's contributions. While it correctly identifies the limitations of DARE's random dropping policy, it fails to engage with a body of contemporary work exploring complementary or alternative solutions (e.g., trust-region based task vector merging, subspace boosting, and concrete subspace learning). To strengthen the paper, the authors should expand introduction to evaluate these methods, highlighting the specific challenges they do not fully address. Subsequently, this analysis should be used to more clearly frame how PAVE's knowledge-aware subspace approach offers a more effective solution. Failure to engage with them is a significant oversight, affecting the positioning of the novelty.
2. A significant practical limitation of the proposed method is its reliance on sampled activations from the training data to construct the covariance matrices. This design choice deviates from the principles of truly data-free model merging. While PAVE is not a training-based method, it is no longer strictly "data-free," a property that is highly desirable for many real-world applications where original training data may be private, proprietary, or otherwise inaccessible.
3. While PAVE demonstrates improvement on ViT models, its primary weakness is the marginal performance improvement on LLaMA, necessitating more extensive experimentation on LLMs to validate its practical utility for the generative AI landscape.

### Questions
1. What is the performance of PAVE when the number of tasks or models is much larger (e.g., 30)?

### Soundness
2

### Presentation
3

### Contribution
2
