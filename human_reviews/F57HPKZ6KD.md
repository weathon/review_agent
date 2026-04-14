# Efficient and Robust Neural Combinatorial Optimization via Wasserstein-Based Coresets

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Combinatorial optimization (CO) is a fundamental tool in many fields. 
Many neural combinatorial optimization (NCO) methods have been proposed to solve CO problems.
However, existing NCO methods typically require significant computational and storage resources, and face challenges in maintaining robustness to distribution shifts between training and test data.
To address these issues, we model CO instances into probability measures, and introduce Wasserstein-based metrics to quantify the difference between CO instances. 
We then leverage a popular data compression technique, \emph{coreset}, to construct a small-size proxy for the original large dataset.
However, the time complexity of constructing a coreset  is linearly dependent on the size of the dataset. Consequently, it becomes challenging when datasets are particularly large.
Further, we accelerate the coreset construction by adapting it to the merge-and-reduce framework, enabling parallel computing. Additionally, we prove that our coreset is a good representation in theory.
{Subsequently}, to speed up the training process for existing NCO methods, we propose an efficient training framework based on the coreset technique. We train the model on a small-size coreset rather than on the full dataset, and thus save substantial computational and storage resources. Inspired by hierarchical Gonzalez’s algorithm, our coreset method is designed to capture the diversity of the dataset, which consequently improves robustness to distribution shifts.
Finally, experimental results demonstrate that our training framework not only enhances robustness to distribution shifts but also achieves better performance with reduced resource requirements.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors represent CO instances as probability measures and utilize a variant of Wasserstein distance that accounts for rigid transformations, allowing the model to better handle similar instances.

They construct a small, representative subset of the original data using hierarchical clustering. This process is optimized using a "merge-and-reduce" framework, enabling parallel computing and faster training times.

The proposed framework replaces large datasets with the coreset for training, reducing computational and storage needs. For inference, the framework aligns test data along a hierarchical tree, which further accelerates the process.

Through experiments on tasks such as the Traveling Salesperson Problem (TSP), the authors demonstrate that the coreset-based framework achieves robust performance and efficiency gains, especially under distribution shifts.

### Strengths
Originality

This paper introduces a novel approach to neural combinatorial optimization (NCO) by leveraging Wasserstein-based coresets to address resource and robustness challenges in training models for combinatorial optimization problems. The originality stems from: 1. Modeling combinatorial optimization instances as probability measures and applying the Wasserstein distance under rigid transformations (RWD) is innovative. This framework addresses the challenges of data redundancy and distribution shifts, providing a fresh perspective in the NCO space. 2. While coresets are common in clustering and other machine learning tasks, adapting this technique for combinatorial optimization with merge-and-reduce frameworks for accelerated parallel computing is both creative and practical.

Quality

The paper is well-executed, with rigorous theoretical foundations and a thoughtful experimental setup: 1. The authors provide a solid mathematical foundation for the proposed coreset construction, including formal definitions, theoretical guarantees, and proof sketches that validate the coreset’s ability to represent large datasets with minimal error. 2. The experiments cover various TSP instances under both uniform sampling and coreset techniques, testing robustness to distribution shifts and different dimensions (2D and 3D) with insightful comparisons across sample sizes and distributions. This comprehensive approach substantiates the paper's claims.

Clarity

The paper is well-structured and communicates its ideas effectively: 1. The paper follows a clear progression from defining the problem and introducing the method to presenting experimental results. Definitions, such as the RWD and coreset construction, are introduced at appropriate points, aiding understanding. 2. Algorithmic steps are outlined in detail, making it easier to understand implementation specifics, especially for constructing coresets.

Significance

This work has substantial potential significance for both research and practical applications: 1. By enabling training on compact coresets, the paper addresses one of the most significant limitations in neural combinatorial optimization—computational inefficiency with large datasets. This is a notable advancement that could make NCO more feasible in real-world, large-scale applications like logistics, robotics, and manufacturing. 2. The paper’s robustness improvements are important for applications where data distributions vary, making it applicable across domains where data acquisition is not consistently distributed or is prone to changes over time.

### Weaknesses
1. The coreset construction approach offers impressive results in reducing data size while preserving accuracy. A deeper discussion on the trade-offs between dataset compression and accuracy loss would be beneficial. Although error bounds are mentioned, it would strengthen the paper if experiments explicitly analyzed performance degradation as coreset size is reduced.

2. It’s unclear how the coreset method performs across diverse types of CO instances (e.g., sparse vs. dense graphs or varying clustering characteristics). Adding experiments with different dataset properties could clarify the method’s robustness and practical applicability.

3.  Aligning instances under rigid transformations for each test instance could be computationally expensive, especially for larger datasets or higher-dimensional instances (e.g., TSP-3D and beyond). Although a heuristic for alignment is provided, it would be helpful to quantify the computational cost and compare it to baseline methods.

4. The paper does not include a performance comparison between exact and heuristic alignment. Such an evaluation would help clarify the practical efficiency and accuracy trade-offs of the heuristic.

5. The experimental validation focuses heavily on TSP or MIS dataset, which, while effective, may limit the generalizability claims. While these are well-known CO problems, the approach could benefit from validation on other types of CO tasks with varying structures, such as graph partitioning problems. Since these problems exhibit different graph characteristics, they would better demonstrate the adaptability of the coreset method.

6. Current experiments primarily focus on tour length and runtime as evaluation metrics. Additional measures, such as robustness to noise or perturbations in the data, could further demonstrate the coreset’s value in handling real-world data variability.

7. The merge-and-reduce framework is crucial for scaling the coreset method, but the paper provides limited guidance on its implementation and scalability implications: Although the paper briefly mentions time complexity, a more detailed breakdown of the merge-and-reduce framework’s complexity across layers and for different dataset sizes would be helpful.

8. There is limited discussion on how the framework’s parallelization could be optimized or applied to larger datasets. This discussion would be especially useful for practitioners seeking to apply the method in large-scale real-world settings.

9. While the paper claims improved robustness to distribution shifts through the use of coresets, this aspect is not rigorously analyzed or compared. The paper could benefit from quantifying the robustness improvements by comparing the method’s performance across significantly different distributions and measuring accuracy decay.

### Questions
see weakness

### Soundness
3

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
3

### Summary
In this article, the author(s) develop a framework for accelerating neural combinatorial optimization (CO) methods. The basic idea is to construct a small-size coreset to represent the whole data set, and only train models on the coreset. The coreset is constructed based on a clustering algorithm and the Wasserstein distance under rigid transformations (RWD).

### Strengths
The overall motivation of this article is clear. Efficiently solving CO problems are of great importance, and this article provides a potential direction.

### Weaknesses
1. Although the main topic of this article is about combinatorial optimization (CO) problems, it seems that the whole article ignores the structure and discreteness of CO problems, and only considers their graph embedding. Then the author(s) only consider objects that lie in an Euclidean space, which leaves me the impression that the proposed method is developed for continuous problems, and CO only appears in the preprocessing stage (i.e., converting a CO instance into a continuous object). I am not sure if this is the proper way to deal with CO problems, but at least the author(s) should discuss the relationship between the original CO problem and the embedding. For example, is there any information loss after converting into embeddings? Are the results sensitive to the choice of embedding methods?

2. I think the equation in Definition 2.1 is incorrect. The $C_{ij}$ term should be $P_{ij}$, and the cost matrix $C$ is never used.

3. It seems that the Wasserstein distance under rigid transformation (RWD) is a core component of the proposed method, but I do not see the method to compute it. In Remark 3.1, the author(s) claim that RWD can be solved within $\tilde{O}(n^2)$ time, but I do not see why. Computing RWD should be much more difficult than the classical optimal transport (OT), as it involves optimization over an additional object $e$. But even for OT, I wonder how the complexity $\tilde{O}(n^2)$ can be achieved without using approximation methods such as the entropic regularization, as it is well known that a linear programming solver for OT takes $O(n^3\log(n))$ [1].

4. If computing RWD is expensive, then I wonder whether it is meaningful to construct the coreset at all. The author(s) are suggested comparing the cost of training on the whole data set and the cost of constructing the coreset plus the time of training on the coreset.

[1] Pele, O., & Werman, M. (2009). Fast and robust earth mover's distances. In 2009 IEEE 12th international conference on computer vision.

### Questions
See the "Weaknesses" section.

### Soundness
2

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
3

### Summary
The paper presents a novel approach to enhance neural combinatorial optimization (NCO) by introducing Wasserstein-based coresets, which efficiently compress large datasets into smaller, representative proxies. By modeling combinatorial optimization (CO) instances as probability measures and utilizing a specialized Wasserstein distance under rigid transformations (RWD), the authors quantify differences between CO instances effectively. To address the computational challenges of constructing coresets for large datasets, they adapt the merge-and-reduce framework, enabling parallel processing and theoretical guarantees of representation quality. Additionally, the proposed training framework leverages these coresets to reduce computational and storage requirements while maintaining and even improving robustness to distribution shifts between training and testing data. Experimental results on Traveling Salesperson Problem (TSP) and Maximum Independent Set (MIS) instances demonstrate that their method outperforms uniform sampling and other existing techniques, achieving better performance and enhanced robustness with reduced resource usage.

### Strengths
The paper effectively addresses key challenges in neural combinatorial optimization (NCO) by introducing Wasserstein-based coresets that reduce dataset size without compromising essential information. The use of Wasserstein distance under rigid transformations (RWD) provides a robust metric for comparing combinatorial optimization instances, enhancing the method's ability to handle distribution shifts. By adapting the merge-and-reduce framework, the authors achieve scalable and parallelizable coreset construction, making the approach feasible for large datasets. Theoretical guarantees ensure that the coresets accurately represent the original data, while the proposed training framework demonstrates reductions in computational and storage requirements. Experimental results on the Traveling Salesperson Problem (TSP) and Maximum Independent Set (MIS) depict the method's superior performance and increased robustness compared to uniform sampling.

### Weaknesses
While the paper presents a rigorous analysis of Wasserstein-based coresets to enhance NCO, it exhibits several weaknesses. In particular, the English language is somewhat poor, leading to unclear statements $-$ for instance, "Therefore, how to train a competitive model by using limited resources while guaranteeing its robustness to distribution shift is a deserving problem" is awkwardly phrased. Technically, the justification for modeling combinatorial optimization instances as probability measures using Rigid Wasserstein Distance (RWD) seems incomplete, especially since the assumptions like low doubling dimension may not hold in practical, high-dimensional settings (e.g., problems with large graphs or high-dimensional feature spaces). Despite the reduced algorithmic complexity, it is unclear whether the proposed merge-and-reduce framework (to accelerate the coreset construction algorithm) may introduce additional practical overhead due to the increased cost of data transfer (via partitioning and merging) and possible synchronization delays.

### Questions
Does the proposed merge-and-reduce framework to accelerate coreset construction introduce additional practical overhead due to increased data transfer costs (via partitioning and merging) and possible synchronization delays?

### Soundness
3

### Presentation
2

### Contribution
3
