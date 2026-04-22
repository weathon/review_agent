# GradPCA: Leveraging NTK Alignment for Reliable Out-of-Distribution Detection

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8, 6

## Abstract
We introduce GradPCA, an Out-of-Distribution (OOD) detection method that exploits the low-rank structure of neural network gradients induced by Neural Tangent Kernel (NTK) alignment. GradPCA applies Principal Component Analysis (PCA) to gradient class-means, achieving more consistent performance than existing methods across standard image classification benchmarks. We provide a theoretical perspective on spectral OOD detection in neural networks to support GradPCA, highlighting feature-space properties that enable effective detection and naturally emerge from NTK alignment. Our analysis further reveals that feature quality—particularly the use of pretrained versus non-pretrained representations—plays a crucial role in determining which detectors will succeed. Extensive experiments validate the strong performance of GradPCA, and our theoretical framework offers guidance for designing more principled spectral OOD detectors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work takes a Neural Tangent Kernel (NTK) perspective to investigate the Out-of-Distribution (OOD) detection problem. A novel detection method named GradPCA is proposed. In GradPCA, PCA is executed on the class-wise gradients of in-distribution (IND) data to obtain a low-dimensional subspace, and IND and OOD data are expected to exhibit well separability on this low-dimensional gradient subspace. Theoretical results are presented about the sufficient and necessary conditions for identifying a sample as OOD in spectral OOD detection, with discussions on how to select the feature mapping. Extensive empirical results validate the effectiveness of GradPCA. Besides, the relationships between OOD detection and some other issues such as consistency and feature quality are discussed.

### Strengths
1.	The NTK perspective and the associated GradPCA method are novel and beneficial to the OOD detection community.  
2.	Discussions on the consistency and feature quality issues are appreciated and can provide new insights for OOD detection.  
3.	The writing is good and easy to follow.   
4.	Theories and extensive empirical results are provided.

### Weaknesses
**Major concerns**  

1.	Unclear descriptions on the algorithm implementation  

Some descriptions in Algorithm 1 are confusing, and thereby related details are suggested to be supplemented.   

1.1	In line 4-6 in the offline training stage, what are the detailed executions and differences between line 4 and line 5? I guess that here PCs are obtained through eigendecomposition on the covariance matrix and its dual matrix, and please supplement detailed mathematical equations. But why there are two times of eigendecomposition? Then, given line 4 and line 5, which PCs are adopted to obtain the projection matrix in line 6? Please specify the calculations of the projection matrix clearly.  

1.2	In line 9 in the online inference stage, I guess the PCA reconstruction error on gradients of a new sample is set as the detection score. But the definition for the projection matrix is missing, causing confusion on the detection score. Besides, why the norm of the projected gradients is normalized by that of the original gradients? Calculating reconstruction errors does not need the normalization of input norms. Please clarify this issue.

2.	Insufficient experiments.  

While the experiments include several general detection methods, the comparisons lack more directly relevant, subspace-based baselines, as GradPCA itself is based on PCA and model gradients. The suggested baselines have been reviewed in the submission but are missing in empirical comparisons: (i) PCA [1] and Kernel PCA [2] applied to penultimate layer features, and (ii) the low-dimensional gradient subspace method explored in [3].

3.	Extended discussion on the consistency.  

The inconsistency issue in OOD detection is rarely explored. The experiments in Figure 1 of this submission are executed across diverse benchmarks. Meanwhile, the inconsistency across multiple independently-trained modes (local optima) is highlighted in [4]. It would be appreciated to evaluate GradPCA on such independent modes to validate its consistency from the model side, which will further strengthen this work.

[1] Revisit pca-based technique for out-of-distribution detection. ICCV 2023.  
[2] Kernel PCA for out-of-distribution detection. NeurIPS 2024.  
[3] Low-Dimensional Gradient Helps Out-of-Distribution Detection. TPAMI 2024.  
[4] Revisiting Deep Ensemble for Out-of-Distribution Detection: A Loss Landscape Perspective. IJCV 2024.  

**Minor concerns**  
4.	In the basic settings, the neural network function $f$ outputs a real number. Please specify the calculations of this outputted real number.  
5.	The GradOrth method demonstrates quite poor detection performance across almost all datasets. It would be helpful for authors to clarify whether the hyper-parameters of GradOrth have been carefully tuned. If so, some explanations into the potential reasons for these results would be valuable.

### Questions
My questions are the five concerns listed above. I will raise my rating given all the concerns are well addressed.

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
This work proposes a new method for OOD detection by leveraging NTK alignment over a well-trained network, where both analytical discussions and  empirical evidence are both provided.

In general, the work is easy to follow and the NTK alignment technique is new to the task with different perspectives, even though the PCA technique and its operation on the gradients are not.

### Strengths
This method is easy to implement and shows good efficiency (no training). The empirical performance is on average good and partially near the state-of-the art performances. In particular, the involved NTK perspective is note fully investigated in this field, which may bring some new potential inspiring future work.

### Weaknesses
Some weakness (or points to be clarified) as below:

1. what if the labels of training data are not accessible? This method requires the labels/classes of training samples to construct the projection in PCA, but this is not really requested in other methods and might not be feasible in practical settings, especially considering data privacy. Could the authors make some discussions and remedies? Further, it would be good to mark out in the table whether this requirement  is taken in each compared method.

2. Up to section 3.2, why line 9 and line 10 in algorithm 1 can successfully enable OOD detection? In previous context, it mentions separability of features, which is yet still not direct to the exact rationale of such detection scores. 

In line 149-155, the motivation is rather intuitive and based on existing work (He & Su, 2020). Could you also specify the motivation or evidence? Otherwise, it seems that this wok gives a new technique to do OOD with NTK and PCA, without a clear, strong and verifiable motivation/rationale to do so.

This is very important for the readability of this work and the clarification of its novelty.

3. Why not present discussions with (Guan et al., 2023)  and (Wu et al., 2024a) in the main context, but appendix, and why not present the empirical comparisons in the experiments? These works are highly related. 

4. This work presents quite some discussions on the impact of "feature quality". It still seems unclear and not specific enough for the reviewer accessing the so-called "feature quality" quantitatively. Is it possible to have a clear definition  or metric? Please be more strictly technically precise.

### Questions
Please view the Weakness.

### Soundness
2

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
The paper uses NTK alignment theory (that neural networks are low-rank subspaces) to effectively detect OOD samples. GradPCA is an OOD detection framework that applies PCA to neural network gradients.The authors use NTK Alignment to make this computationally tractable.

### Strengths
The paper establishes strong theoretical motivation behind the GradPCA framework along with some reasoning as to why it works and when the OOD detectors can be thought of as reliable, along with complementing empirical results.

The proposed method is computationally tractable and therefore more practical. O(NP) to O(C).

Decent insights on feature quality explaining when regularity based methods outperform abnormality based methods and vice-versa.

Good empirical results.

### Weaknesses
How does GradPCA compare to the similar recent work [1] which also does PCA on the gradients? Would be good if the authors clarify the major differences and their contributions.

[1] Wu, Yingwen, et al. "Low-dimensional gradient helps out-of-distribution detection." IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).


Only 3 models seem to be used in the experimentation. A greater diverse set might have introduced models that might not exactly follow the NTK alignment theory and cause GradPCA to degrade, and a brief insight into that would have been nice.


Sensitivity to parameter subset selection is important since the method is primed at exploiting the low-rankness of the neural network. So, just using the final hidden layer might be sub-optimal, and a brief insight on this would be helpful.


Ablation study as to how to approximate class-means (smaller batch-sizes) would also have been helpful as they are primary in the proposed algorithm.

### Questions
Please see weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces GradPCA, a novel and principled method for Out-of-Distribution (OOD) detection. The core idea is to exploit the low-rank structure of neural network gradients, which the authors connect theoretically to the Neural Tangent Kernel (NTK) alignment phenomenon. The method applies Principal Component Analysis (PCA) to the class-means of the gradients to define a low-dimensional "in-distribution" subspace. At inference, inputs whose gradients fall outside this subspace are flagged as OOD.
The authors provide a theoretical framework for spectral OOD detection, including a sufficient condition that yields per-sample OOD certificates. Empirically, the paper demonstrates that GradPCA achieves highly consistent and competitive performance across several standard benchmarks (CIFAR-10, CIFAR-100, ImageNet). A significant additional contribution is a systematic analysis showing that OOD detector performance critically depends on feature quality (neural collapse property).

### Strengths
- Strong Theoretical Contribution: The paper offers a formal framework for spectral OOD detection in NNs. Theorem 4.1, which provides a "sufficient condition for spectral OOD detection" , is a strong theoretical result that provides a deterministic, per-sample OOD certificate.

- Excellent Empirical Results & Consistency: GradPCA achieves SOTA or near-SOTA results, but more importantly, it demonstrates the most consistent performance across all benchmarks. This directly addresses a major problem in the OOD field, where methods often fail erratically in different settings.

- Valuable Analysis of Feature Quality: The paper's distinction between "regularity-based" and "abnormality-based" detectors is insightful. The finding that their effectiveness is tied to whether features are pretrained (general-purpose) or not (task-specific) is a practical and important contribution that helps reconcile inconsistencies in prior work.

### Weaknesses
- Memory Scalability: The primary weakness is that the memory cost scales with the number of classes, $C$. The method stores $O(C)$ gradient vectors, which "can be costly for large C" like in ImageNet15. While the paper shows this is manageable (e.g., 7.5GB for ImageNet in the worst case, but often less  ), it could be a barrier for datasets with thousands or tens of thousands of classes.

- Core Assumption: The method's success relies on the assumption that NTK alignment provides a low-rank structure that effectively separates ID and OOD points in the gradient space . The authors note this "may not hold universally," though they rightly argue this assumption is explicit and empirically supported.

- Some missing references and comparisons that similarly leverage the gradient space:

Lee, Jinsol, et al. “Gradient-Based Adversarial and Out-of-Distribution Detection.” arXiv:2206.08255, arXiv, 4 July 2022. arXiv.org, http://arxiv.org/abs/2206.08255.

Sun, Jingbo, et al. “Gradient-Based Novelty Detection Boosted by Self-Supervised Binary Classification.” arXiv:2112.09815, arXiv, 17 Dec. 2021. arXiv.org, http://arxiv.org/abs/2112.09815.

ElAraby, Mostafa, et al. "GROOD: GRadient-Aware Out-of-Distribution Detection." Transactions on Machine Learning Research. https://arxiv.org/abs/2312.14427

### Questions
The paper mentions that GradPCA defaults to using only the last hidden layer parameters. It would be beneficial to include a more comprehensive ablation study on this choice. How does the method's performance vary when using gradients computed from the output layer as in GradNorm, or from specific intermediate blocks?

To further strengthen the empirical validation, the authors are encouraged to benchmark GradPCA within the OpenOOD framework. This would enable a standardized comparison and, more importantly, provide valuable insights into the method's performance on challenging near-OOD datasets versus far-OOD datasets.

For improved readability, especially for black-and-white printing, the authors should reconsider the highlighting scheme in the results tables. The current use of color to highlight the top-3 performers  can be confusing. A simpler, more standard convention, such as bolding the top-performing method and underlining the second, would significantly enhance clarity.

The paper's analysis of feature quality versus detector performance is a key contribution . To further explore this relationship, could the authors include an ablation study where feature quality is degraded in a more controlled manner? For instance, how do the "regularity-based" and "abnormality-based" detectors compare when the in-distribution (ID) training dataset is corrupted with increasing levels of label noise or input noise?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes GradPCA that applies PCA to class-mean gradients of a trained neural network. The approach is motivated by Neural Tangent Kernel (NTK) alignment, which empirically yields a low-rank, approximately block-diagonal structure in the gradient covariance of well-trained models. GradPCA formalizes this link between NTK alignment and spectral OOD detection, derives sufficient and necessary conditions for detection guarantees, and demonstrates consistent, near–state-of-the-art performance across CIFAR-10/100 and ImageNet under pretrained and non-pretrained settings.

### Strengths
Overall, this is a well-written and sophisticated paper with clear motivation, background, and rationale.  The writing is made to be accessible to a general deep-learning audience.

The paper has a clear theoretical grounding that connects OOD detection to NTK alignment and covariance low-rank structure, offering a mathematically elegant view. The method essentially captures a low-rank representation of the NTK kernel / gradient covariance matrix. Kernel-based reasoning is well-founded; kernel and spectral methods have a strong theoretical pedigree and proven reliability for measuring similarity in DL to model complex latent spaces of features. The approach aligns conceptually with sparse / low-rank matrix representation theory, which is also a mature and robust field.
Experiments have been conducted on multiple benchmark datasets. Empirically results are generally sound and demonstrate performance suggested by the theory.

### Weaknesses
The choice of NTK, while natural given the theoretical link, is not unique. Other kernels (e.g., Fisher information, feature-space kernels) could also exhibit similar low-rank behavior, so the generality of the method is not fully demonstrated. The paper would benefit from a clearer articulation of why the NTK is the most appropriate or insightful kernel for connecting spectral structure with OOD behavior. In other words, it needs to “ring a bell” by making the NTK–OOD link feel both necessary and intuitively strong.
	
Experimental coverage is confined to image classification. It remains unclear whether the same assumptions and empirical stability extend to other domains such as text, time series, etc. While full experiments are not required, a discussion of applicability and limitations across modalities would strengthen the paper.
	
The theoretical results rely on simplified assumptions (rank-C covariance matrix, small residual, etc.) that may not strictly hold in real neural networks. What if some of the classes have similar, or even partially overlapping feature semantics? 
	
Related comments are provided in the questions section below.

### Questions
Could the method be extended to characterize OOD detection in terms of Type I and II errors, rather than a hard binary boundary? Specifically, how would the approach behave if in-distribution (ID) and OOD supports overlap, as might occur with visually similar classes?

As a related question, the theoretical assumptions (low-rank structure, small residual \xi) may not hold when classes are mixed or poorly separated. Have the authors tested how sensitive GradPCA is to such violations?

How is the kernel chosen? Could alternative kernels (e.g., Fisher, feature-space, or adaptive kernels) improve detection? Is the NTK selection fixed or adaptive to data/model characteristics?

For the covariance (aka $FF^\top$) matrix, are gradients centered before computing PCA? Clarification is needed on whether centering affects the eigenstructure and stability of the projection.

### Soundness
3

### Presentation
3

### Contribution
2
