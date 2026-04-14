# SSGNN: Simple Yet Effective Spectral Graph Neural Network

- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
Spectral GNNs leverage graph spectral properties to model graph representations but have been less explored due to their computational challenges, especially compared to the more flexible and scalable spatial GNNs, which have seen broader adoption. However, spatial methods cannot fully exploit the rich information in graph spectra. Current Spectral GNNs, relying on fixed-order polynomials, use scalar-to-scalar filters applied uniformly across eigenvalues, failing to capture key spectral shifts and signal propagation dynamics. Though set-to-set filters can capture spectral complexity, methods that employ them frequently rely on Transformers, which add considerable computational burden. Our analysis indicates that applying Transformers to these filters provides minimal advantage in the spectral domain. We demonstrate that effective spectral filtering can be achieved without the need for transformers, offering a more efficient and spectrum-aware alternative. To this end, we propose a $\textit{Simple Yet Effective Spectral Graph Neural Network}$ (SSGNN), which leverages the graph spectrum to adaptively filter using a simplified set-to-set approach that captures key spectral features. Moreover, we introduce a novel, parameter-free $\textit{Relative Gaussian Amplifier}$ (ReGA) module, which adaptively learns spectral filtering while maintaining robustness against structural perturbations, ensuring stability. Extensive experiments on 20 real-world graph datasets, spanning both node-level and graph-level tasks along with a synthetic graph dataset, show that SSGNN matches or surpasses the performance of state-of-the-art (SOTA) spectral-based GNNs and graph transformers while using significantly fewer parameters and GFLOPs. Specifically, SSGNN achieves performance comparable to the current SOTA Graph Transformer model, Polynormer, with an average 55x reduction in parameters and 100x reduction in GFLOPs across all datasets. Our code will be made public upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes SSGNN, a simple and effective GNN model which can achieve good performance with much reduced parameters compared to transformers. With basic spectral encoder-decoder structure, it further incorporates a REGA module to strengthen the representational capabilities and the robustness against spectral perturbation. Experiments demonstrate its effectiveness and superiority on model parameters.

### Strengths
1-	The method is simple and effective on multiple graph downstream tasks.

2-	The experiments are comprehensive and solid.

3-	The theoretical analysis of ReGA is interesting and novel.

### Weaknesses
1-	The time and space cost in pre-computation and training stage of SSGNN seems to be a bottleneck for large graphs. Though top-k techniques can be applied, it’ll lose spectral frequencies which is essential for down-streaming tasks. When comparing parameter amounts and GFLOPS, it’ll be fair to show the training and pre-computation cost for baselines, transformers and other GNNs.

2-	It needs experiments to demonstrate the contribution of ReGa, which is the most novel part in SSGNN. Will the removal of it greatly influence the performance?

3-	Presentation can be improved. “. .” appears in line 373 and line 398 appears incomplete sentences “* means” what?

### Questions
In the abstract, “Our analysis indicates that applying Transformers to these filters provides minimal advantage in the spectral domain.” While such analysis seems to be missing in the content. Could you elaborate it more?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a Simple Yet Effective Spectral Graph Neural Network (SSGNN), which simplifies the set-to-set graph filter, e.g., Specformer, without performance degeneration. The key component is a parameter-free Relative Gaussian Amplifier (ReGA), which not only improves model performance but also maintains the robustness against graph perturbations. Extensive experiments on both node-level and graph-level datasets demonstrate the superiority of SSGNN in terms of effectiveness and efficiency over baselines.

### Strengths
1. The design of the ReGA component in SSGNN has some novelty. It not only avoids negative eigenvalues but also improves the robustness of SSGNN.

2. This paper conducts extensive experiments and SSGNN also shows competitive performance. For example, in the ZINC dataset, SSGNN has a RMSE value of 0.0592.

### Weaknesses
1. This paper claims that SSGNN uses a **simplified set-to-set approach** to capture key spectral features. However, there is no interaction between different eigenvalues in SSGNN. Specifically, the matrix $Z_{eig} \in \mathbb{R}^{N \times (d+1)}$ indicates the $(d+1)$ dimensional representation of each eigenvalue. The transformation $W_{eig}$ is applied on the channels of a single eigenvalue, i.e., $Z_{eig}W_{eig}$. Therefore, SSGNN is not a set-to-set approach.

2. SSGNN involves a lot of tricks in the training process, such as eigen-correction. However, both Specformer and Polynormer do not use eigen-correction for data preprocessing. In this case, it is necessary to make a comprehensive ablation study to validate the roles of each trick. We need to verify whether the performance improvement mainly comes from ReGA rather than eigen-correction.

3. It would be better if the authors could provide a comparison of the time and space overhead between different methods.

4. How many parameters does SSGNN have in Table 5? Generally, we will control the number of parameters around 50K in the ZINC dataset.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies the improvements of spectral GNNs, which aims to design an expressive filter based on spectral graph theory for effective graph representations. 
The paper points out that existing SOTA spectral GNNs bring more computational burden, though they can learn the filters better. Thus, the paper proposes a novel efficient framework, namely SSGNN, which only applies simple linear transformation instead of Transformers on the spectrum. Moreover, SSGNN incorporates a parameter-free Relative Gaussian Amplifier to the decoder to enhance adaptive filter learning and maintain stability. The paper conducts extensive experiments on synthetic and real-world datasets to demonstrate the effectiveness of SSGNN.

### Strengths
1、	The motivation of this paper is significant and valuable for Spectral GNNs.
2、	SSGNN achieves significant efficiency in terms of computation and parameters.
3、	Experimental results in most cases seem promising.

### Weaknesses
1. The novelty of the method is somewhat limited, especially the direct application of existing eigen-correction, eigenvalue encoding, and convolution framework without any transfer challenge.
2. The description of “set” of “set-to-set filtering” is ambiguous. Specformer applies Transformer on eigenvalue encodings, which enables filters to capture relative dependencies among the eigenvalues. Thus, the “set” of “set-to-set” in Sepcformer means the set of eigenvalues. However, SSGNN learns the spectral filter through linear transformations, so eigenvalues don’t interact with each other. Thus, what’s the meaning of “set”?
3. The roles of the two linear transformations (namely W_{eig} and W_1W_h) respectively playing in encoder and decoder are not clear. In other words, why do authors include W_1 and W_h in the decoder instead of the encoder?
4. The paper ignores some essential experiments.
（1）As mentioned in Line 217, different heads allow the decoder to learn diverse spectral filtering patterns. However, there is no visualization of the diverse spectral filters learned by different heads to verify this conclusion.
（2）There is no ablation study on the effectiveness of re-center adjustment in Equation 4 and the effectiveness of Relative Gaussian Amplifier in Equation 6.
（3）Authors don’t verify the stability of SSGNN on OOD benchmarks such as DrugOOD[1], where many model stability studies are validated on this dataset.
5. The symbols are ambiguous. For example, \epsilon is used in Line 182, Line 240-241, and Line 307 simultaneously, making the paper more difficult to read. Moreover, it’s not clear on which \epsilon the ablation experiments reported in Figure 4 are conducted.

[1] Ji, Yuanfeng, et al. "Drugood: Out-of-distribution dataset curator and benchmark for ai-aided drug discovery–a focus on affinity prediction problems with noise annotations." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 7. 2023.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose SSGNN, a simple yet effective spectral-based Graph Neural Network that captures rich spectral information through an adaptive set-to-set filtering approach, offering a more efficient alternative to transformer-based methods. The method introduces a parameter-free Relative Gaussian Amplifier (ReGA) module for robust spectral filtering, and demonstrates superior or comparable performance to state-of-the-art models while using significantly fewer parameters (55x reduction) and computational resources (100x reduction in GFLOPs) across 20 real-world datasets.

### Strengths
The paper presents an interesting transformation from scalar-to-scalar to set-to-set methodology, building upon the Spectral Former framework. By introducing a learnable parameter W to capture relationships between different frequency domain eigenvalues, the authors aim to enhance model performance through better consideration of inter-frequency domain relationships.

### Weaknesses
- However, the methodology lacks clarity regarding the eigenvalue computation process, which appears to rely on time-consuming SVD operations, raising concerns about computational efficiency.

- Given that your proposed method emphasizes simplicity and reduced learnable parameters, it would be particularly valuable to demonstrate its effectiveness on large-scale graphs. While the reduction in parameter count is noteworthy, the real advantage of a simpler model should be its ability to scale effectively to larger, real-world graph applications. Therefore, I strongly recommend including comprehensive experiments on large-scale graph datasets to validate the method's practical utility. This would not only strengthen your contribution but also clearly differentiate your work from existing methods that may struggle with scalability.

- From a comparative standpoint, although the spectral approach shows promise, the evaluation lacks comprehensive comparisons with important baseline methods, particularly SGC and SSGC. These baselines are especially relevant as they also prioritize simplicity and efficiency. To make your contribution more compelling, consider expanding the experimental section to include: (1) comparisons with these relevant baselines, (2) clear documentation of the eigenvalue computation process and its efficiency, and (3) thorough scalability analysis on large-scale graphs that would demonstrate the practical advantages of your simplified approach. This would help readers better understand the unique benefits of your method in real-world applications where scalability is crucial.

### Questions
- How to compute the specturm? It appears to rely on time-consuming SVD operations.
- The feasibility of applying this method to large-scale graphs is uncertain.
- Runing time comparsion need to be added
- Notable baseline methods are missing from the evaluation, particularly:
  - SGC (Simple Graph Convolution)
  - SSGC (Simple Spectral Graph Convolution)

### Soundness
3

### Presentation
2

### Contribution
3
