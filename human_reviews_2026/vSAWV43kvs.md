# Towards a Foundation Model Approach for Causal Graph Learning

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Due to its human-interpretability and invariance properties, Directed Acyclic Graph (DAG) has been a foundational tool across various areas of AI research. %, leading to significant advancements. However, DAG learning remains highly challenging, due to its super-exponential growth in computational cost and identifiability issues, particularly in small-sample regimes. To address these two challenges, we leverage the recent success of transformers and develop a foundation model approach for discovering multiple DAGs across tasks. In particular, we propose Attention-DAG (ADAG), a novel attention-mechanism-based architecture for learning multiple linear Structural Equation Models (SEMs). ADAG learns the mapping from observed data to both graph structure and parameters via a nonlinear attention-based kernel, enabling efficient multi-task generalization of the underlying linear SEMs. By formulating the learning process across multiple domains as a continuous optimization problem, the pre-trained ADAG model captures the common structural properties as a shared low-dimensional prior, thereby reducing the ill-posedness of downstream DAG tasks in small-sample regimes. We evaluate our proposed approach on benchmark synthetic datasets and find that ADAG achieves substantial improvements in both DAG learning accuracy and zero-shot inference efficiency. To the best of our knowledge, this is the first practical approach for pre-training a foundation model for unsupervised DAG learning, representing a step toward more efficient and generalizable down-stream applications in causal discovery.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Attention-DAG (ADAG), a transformer-based model designed for zero-shot causal discovery. The model is pre-trained across numerous domains to learn a direct mapping from observational data to a linear Structural Equation Model (SEM), including both the graph structure and its parameters. The key idea is to amortize the cost of discovery, enabling extremely fast inference on new, unseen tasks.

### Strengths
The paper's ambition to create a foundation model for unsupervised causal discovery is timely and novel. The proposed ADAG architecture is a non-trivial application of transformers to this inverse problem, and the resulting zero-shot inference speed is empirically impressive on the tested benchmarks.

### Weaknesses
The paper's central claims are undermined by severe methodological limitations that challenge its viability as a true "foundation model" for causal discovery.
1. The model's foundation is critically narrow, as it is pre-trained exclusively on data generated from linear SEMs. Real-world causal mechanisms are complex, diverse, and often highly nonlinear. By only learning to recognize linear relationships, the model is theoretically unprepared to handle unseen, novel causal functions. While the model can be extended to handle nonlinear data, this capability should not be conflated with the far more difficult challenge of accurately recovering the structure with diverse and unknown nonlinear causal functions that actually generated the data.
2. The paper's framing as a "foundation model" is misleading due to its choice of training objective.ADAG is optimized using a data-reconstruction loss, which is the exact same objective used by single-task, optimization-based methods like NOTEARS. This forces the model to solve the same problem: find a DAG that best fits the linear data-generating process. The underlying causal function depends on the selected loss, which currently can not learn multiple causal functions at the same time. Hence, it does not learn the more general task of mapping data to its corresponding abstract structure.
3. Because the objective is data-fitting, the pre-training process is not learning a general causal discovery skill. Instead, it is simply moving the computational cost of traditional structure learning into a pre-training phase. The model becomes a fast, amortized solver for one specific problem (linear DAG fitting under similar data scales and structures), but it does not learn a fundamentally more general capability.
4. A true foundation model should generalize across a vast diversity of tasks. However, ADAG's methodology suggests that to handle different causal functions (nonlinear, heterogeneous), node scales, and graph structures, one would need to pre-train an entirely new "foundation model" for each specific class of problems. This defeats the purpose and promise of a single, general-purpose foundation model.

### Questions
1. Given that the model is trained exclusively on linear SEMs, why should the community trust its output on real-world datasets where the functional forms are unknown, not linear, and shares diverse causal functions? How does this not represent a critical failure of generalization, which is the primary promise of a foundation model?
2. Your training objective is identical to that of NOTEARS—minimizing data reconstruction error under an acyclicity constraint. Why did you not opt for a direct structural loss (e.g., supervising the model to match the ground-truth adjacency matrix)? Doesn't using a data-fit loss simply force the model to become an "amortized NOTEARS," rather than learning the more abstract and generalizable task of mapping data characteristics to a graph structure?
3. The very concept of a foundation model is built on generalization from massive, diverse data. Your approach seems to require a new "foundation model" for each class of causal functions (linear, polynomial, etc.) and potentially for different graph scales. How do you reconcile this with the claim of building a generalizable foundation model for causal discovery?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel causal graph learning method named Attention-DAG (ADAG) , whose core idea is to introduce the concept of a 'Foundation Model' to address the challenges faced by traditional causal discovery methods, such as high computational cost (super-exponential space) and poor performance in small-sample regimes. ADAG utilizes a Transformer-based attention mechanism to learn a nonlinear 'kernel map' , which can directly convert observational data into the corresponding weighted adjacency matrices (i.e., graph structure and parameters) of linear Structural Equation Models (SEMs). This model, through joint pre-training across multiple domains (tasks) , learns a shared low-dimensional structural prior in an unsupervised manner (using a data reconstruction loss and an acyclicity constraint ). This pre-training enables ADAG to capture commonalities across tasks , effectively mitigating the ill-posedness problem in small-sample learning. Once trained, ADAG can perform efficient 'zero-shot inference'.

### Strengths
Once the model is trained, its inference speed during the testing phase is extremely fast. As shown in Table 1, ADAG's inference time is far lower than that of baselines (like NOTEARS, DAGMA) which require re-optimization for each task. This holds immense practical value in scenarios requiring the rapid processing of numerous different causal tasks.

By pre-training on multiple tasks to learn a shared low-dimensional prior, ADAG is designed to solve the small-sample problem. Experiments fully demonstrate its robustness in small-sample contexts, significantly outperforming other baseline methods.

The method is trained using a data reconstruction loss and does not require ground-truth DAG structures as labels. This makes the method highly practical, as ground-truth causal graphs are almost impossible to obtain in the real world.

### Weaknesses
The experimental results of this paper are intriguing. The authors claim their method learns a "low-dimensional structural prior". Yet, in the "General Data" experimental setting, the DAG for each domain is randomly generated. The paper fails to clearly explain what "shared low-dimensional structural prior" the model could possibly be learning in this scenario. If the graphs are random, no shared structure should theoretically exist. The model's surprisingly strong performance under this setting (Figure 4) is puzzling. This raises the suspicion that what is being learned is not a structural prior, but rather a general "algorithmic prior"—akin to a universal solver for this class of linear problems.



Furthermore, even if we accept the algorithm's validity in principle, its practical feasibility is questionable. As shown in Figure 6, in nonlinear cases, joint training on at least 20,000 domains is required to achieve a converged result. Leaving aside the computational cost, obtaining such a vast number of training domains that share a common structure is highly unrealistic in most real-world applications. The paper's two "real dataset" experiments further underscore this, as the models were trained on synthetic data, not real-world domains. Consequently, I hold significant skepticism regarding both the practical reasonableness and the overall contribution of the proposed method.

### Questions
The paper's mathematical formulation (e.g., Equation 9) appears to use a reconstruction loss consistent with a standard Gaussian noise assumption, but it does not explicitly incorporate identifiability conditions such as 'equal variances' or 'non-Gaussian noise'. In a linear Gaussian SEM, the model can only identify a Markov Equivalence Class. The paper does not clearly explain how ADAG selects a specific DAG from within this equivalence class.

In all experiments, the authors use a fixed threshold of 0.3 to extract the final graph structure from the weighted adjacency matrix. This is a highly sensitive hyperparameter. Could the model be designed to automatically learn an appropriate threshold? Alternatively, a threshold-independent metric (e.g., AUC-PR for edge prediction) should be reported during evaluation to demonstrate that the model's performance is not sensitive to this choice.

The model's input tokenization method ($X^{n}(1:n)\in\mathbb{R}^{d\times n}$) is unique, treating each variable (across n samples) as a single token. This differs from the more common approach in Transformers, which typically treats each sample (across d variables) as a token. What is the justification for this choice?

The initial training time of the proposed method, and how it scales with the dataset size, is also a critical metric. It is insufficient to only report the inference runtime; this information should be supplemented.

### Soundness
1

### Presentation
2

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
This paper proposes an attention-driven DAG learning framework, ADAG, which aims to build a generalizable basic model for causal discovery. The model maps observation data to a weighted adjacency matrix by pre-training a nonlinear kernel mapping, thereby jointly learning DAG structure and causal mechanisms in multiple tasks. ADAG supports zero-shot inference, eliminating the need for additional training on unseen tasks, and performs well in data-scarce and small-sample scenarios. Experiments show that ADAG outperforms existing single-task, multi-task, and amortized DAG learning methods on both synthetic data and real data, especially in terms of inference efficiency and accuracy.

### Strengths
This paper proposes a pre-trained foundational model for unsupervised DAG learning, breaking through the limitations of traditional single-task DAG learning and addressing the key challenges of cross-domain generalization and small-sample adaptation. It employs an attention mechanism to construct a nonlinear kernel mapping, leveraging the Transformer's ability to model complex dependencies while balancing computational efficiency through linear attention blocks. By incorporating acyclic constraints through an enhanced Lagrangian method and optimizing the data reconstruction loss, the paper theoretically guarantees the validity of the model output and provides a theoretical analysis of parameter identifiability.

The dataset covers synthetic data, real biological datasets, and nonlinear data. The comparison baseline is comprehensive: eight state-of-the-art methods across three categories: single-task, multi-task, and amortized. Fairness is ensured through hyperparameter tuning and the recommended settings in the original paper. Key metrics are outstanding, with significant accuracy advantages in small-sample scenarios.

The application value is clear. To address the problems of data scarcity (such as the difficulty in obtaining biological experimental samples) and variable tasks in real scenarios, ADAG's pre-training paradigm can be directly transferred to downstream tasks without retraining, providing practical tools for the implementation of causal reasoning in medical, biological and other fields.

### Weaknesses
Incomplete nonlinear SEM adaptation: While the paper mentions that ADAG can be extended to nonlinear SEM, the final Training Objective stage still uses a linear form. It does not propose a complete architectural design for nonlinear SEM, nor does it compare specialized nonlinear DAG learning methods. Its adaptability to nonlinear scenarios lacks systematic verification. Excluding pre-training on data from multiple systems, it appears to simply replace the linear optimization in Notears with Attention.

Training Optimization Objective: Although Notears obtains an adjacency matrix through reconstruction and constraints, it does not claim to be a causal graph. In ADAG, does the A obtained by reconstructing the observed data truly represent a "causal graph" of causal relationships? This point is not discussed alongside identifiability.

Generalizability of the "pre-trained model": The paper does not provide a detailed explanation of the training data construction process for the pre-trained model. Furthermore, the performance of the pre-trained model is likely to be highly dependent on the distribution and diversity of the pre-training data. Performance may decline if the test data differs significantly from the training distribution, a point that is not fully discussed. Appendix C.2 shows that model performance stabilizes when M ≥ 70,000 training domains. However, the paper fails to analyze the model's degradation when the number of domains is insufficient (e.g., M < 10,000), nor does it explore the impact of domain distribution differences on zero-shot generalization. This results in insufficient support for the model's practicality in real-world scenarios with limited training data.

### Questions
Is ADAG a universal model, meaning it can be adapted to any test data without any modifications or fine-tuning? How is its training data constructed? Does it support mixed data types?

For nonlinear SEM, how should the nonlinear transformation layer of the input data be designed to optimally adapt to the attention kernel mapping? Do the acyclicity constraints or loss functions need to be adjusted to accommodate nonlinear data generation mechanisms? Can specific nonlinear DAG learning methods be compared to more comprehensively validate ADAG's advantages in nonlinear scenarios?

Can the paper provide in-depth analysis of the physical meaning of attention weights—for example, the differences in the contributions of different attention heads and layers to the DAG structure (e.g., key causal edges)? Can visualizations (e.g., attention heatmaps) demonstrate how the model "focuses" on key variables to infer causal relationships?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This article builds a very limited foundation model, ADAG, for causal discovery. By assuming different causal structures of the same set of variables in different domains, ADAG conducts pre-training on a few domains to learn the shared features of causal structures among different domains, and then generalizes to other domains that have not been seen before. The test results show that ADAG can efficiently and accurately complete downstream multi-DAG learning tasks.

### Strengths
1. This paper utilizing a foundation model to directly mapping observed data to the causal graph, which can rapidly discovery causality and meet the requirements of real-world engineering applications.
2. This paper achieves small-sample causal discovery by training a foundation model, which has certain innovation.
3. The description in the method section of this article is clear.
4. When there are significant commonalities in across-domain data, the proposed model has been demonstrated to be effective and stable.

### Weaknesses
1. This paper presents a highly limited fundamental model for causal discovery, which can only learn linear causal relationships within the same set of variables, and may be far from real-world problems.
2. ADAG trains the foundation model by minimizing the reconstruction loss. Referring to previous causal discovery methods, although this strategy is theoretically effective, there is a risk of learning incorrect causal structures, which may seriously undermine the training of the foundation model.
3. Nonlinear expansion is immature. Simply concatenating nonlinear transformations with weighted adjacency matrices may cause the foundation model to fail to learn the mapping from observed data to causal structures.
4. The experimental results are confusing. In Figure 2, comparing the covariance matrix and principal components of the estimated A and the true A may not be as intuitive as directly comparing the two matrices. The caption and the figure of Figure 2 (b) seem inconsistent, which may mislead readers.
5. The experimental analysis is insufficient. In the experimental setup of this paper, general data is introduced, in which different domains have random DAGs. However, the experimental analysis overlooked how ADAG learns causal structure from general data. Given that the DAGs in different domains are random, ADAG may not be able to learn the shared features of causal structure from general data.
6. The efficiency and scalability of ADAG are limited. ADAG only offers slight improvements in accuracy and test efficiency, but it may have expensive training costs (under the same experimental conditions, it can only identify causal graphs with less than 50 nodes).
7. Real experiments might be unreasonable. This article sets that the variable set may have different causal structures in different domains. Take the dataset Sachs as an example. It contains multiple sub-datasets with a total of 7,746 observations. Different sub-datasets were observed after different interventions. Therefore, these sub-datasets can be regarded as coming from different domains and are highly suitable for verifying the effectiveness of ADAG. However, this paper only used one of the sub-datasets and was unable to verify ADAG's ability to identify causal graphs across domains. In addition, it is necessary to introduce other real datasets such as fMARI.
8. Lack of convincing theoretical proof. It seems that this paper merely references the identifiability analysis of existing methods, but lacks a mature theory to prove that ADAG can identify causal graphs and their weighted adjacency matrices that vary with the domain.

### Questions
1. Why refer to the mapping from the observed data to the weighted adjacency matrix as the kernel mapping? In my opinion, the foundation model of this article is merely a stack of Transformers and does not involve kernel operations.
2. Why say that strategies like DAGMA are not applicable to ADAG? Take DAGMA as an example. Although DAGMA uses the central path method for optimization, the more efficient acyclic constraints he proposed can also be optimized within the augmented Lagrange framework. I think methods like DAGMA do not conflict with the optimization process of ADAG. 
3. Can you describe in more detail how ADAG extends to nonlinear structural equation models?
4. How does ADAG identify causal structures under general data? According to the description in this article, ADAG can learn the shared features of causal graphs in cross-domain data and then generalize to domains it has never seen before. However, the causal structure of general data in different domains is random. How does ADAG learn shared features from the random causal structure?
5. How long is the training time for ADAG? Although ADAG has improved testing efficiency, its training phase may consume a significant amount of resources and time. Since ADAG identifies linear causal relationships within the same set of variables, I think the expensive training cost might be unacceptable.
6. What are the metrics reported in Table 3? I think it's SHD, but this article doesn't explain the metrics reported in Table 3.
7. Can you validate the proposed model on other real-world datasets with domain-varying causal graphs?

### Soundness
1

### Presentation
2

### Contribution
2
