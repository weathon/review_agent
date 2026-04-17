# Multi-ReduNet: Interpretable Class-Wise Decomposition of ReduNet

- Decision: Accept (Poster)
- Scores: 2, 2, 6, 6

## Abstract
ReduNet has emerged as a promising white-box neural architecture grounded in the principle of maximal coding rate reduction, offering interpretability in deep feature learning. However, its practical applicability is hindered by computational complexity and limited ability to exploit class-specific structures, especially in undersampled regimes. In this work, we propose Multi-ReduNet and its variant Multi-ReduNet-LastNorm, which decompose the global learning objective into class-wise subproblems. These extensions preserve the theoretical foundation of ReduNet while improving training efficiency by reducing matrix inversion costs and enhancing feature separability. We provide a concise theoretical justification for the class-wise decomposition and show through experiments on diverse datasets that our models retain interpretability while achieving superior efficiency and discriminative power under limited supervision. Our findings suggest that class-wise extensions of ReduNet broaden its applicability, bridging the gap between interpretability and practical scalability in deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper improves the objective of ReduNet under the assumption of “undersampled regimes” and proposes two extensions, Multi-ReduNet and Multi-ReduNetLastNorm, for computational efficiency and representation separability

### Strengths
1. The paper is well-structured.
2. The mathematics is generally written in a professional way.

### Weaknesses
Major:

1. The motivation is quite weak. In the introduction, the authors claim representation learning in the “undersampled regimes” is challenging, but without citing any literature or explaining whether this is a widely-accepted concern.

2. Similarly, the questions to be tackled are also unclear. The authors do not define and thoroughly explain “class-specific structures” which they highlight as missing components in previous works.

3. The contributions seem to be incremental and trivial. This paper only performs slight improvements on ReduNet under niche and small-scale settings. It does not bring in novel tools or appealing theoretical insights other than decomposing ReduNet’s objectives, either. In fact, the authors said in line 255:  “Although the class-orthogonality property of MCR2 optima has been established in prior work (Chan et al., 2021), our proof leverages a simpler and more streamlined argument.”, meaning the findings have already been established. These all make me question its significance.  

4. The technical soundness is also questionable. The gradients derived in lines 274 and 277 are basically the same, subject to different scaling. I hardly believe they have a significant functional difference, which makes the decomposition in line 269 less compelling.

Minor:

1. No experimental details and discussion on limitations.

2. Figures are quite hard to interpret immediately due to font size and colorization.

### Questions
1. I’m confused about the last two sentences from lines 32 to 35. Why scenarios where the number of features exceeds the number of samples will lead to overfitting and unstable generalization. Is there literature supporting this claim? The explanations of the background are missing.

2. In Theorem 1, why does $Z^i( Z^j )^T= 0$ mean class-orthogonality? This seems to be misaligned with line 172. Should it be $( Z^i )^T Z^j= 0$?

3. How does the last equality hold in Eq.(2) under the assumption $(Z^{*j_1})^TZ^{*j_2} \neq 0$?

4. The whole analysis in the paper assumes $m \ll d$. Is this assumption even practical for most tasks? I won’t buy it if it’s just an idealistic setting, and rare in practical scenarios. 

5. Why compare to variants with a random forest classifier?

6. How much does training time improve numerically? From figure 1, the proposed model is almost the same with ReduNet.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper extends ReduNet — a theoretically grounded, interpretable architecture based on the principle of maximal coding rate reduction — to address its limitations in scalability and class-specific representation. The authors propose Multi-ReduNet and Multi-ReduNet-LastNorm, which decompose the global learning objective into class-wise subproblems. This decomposition maintains ReduNet’s theoretical interpretability while substantially improving computational efficiency by lowering matrix inversion costs. Moreover, it enhances feature separability, making the model more effective in undersampled  regimes. The paper provides theoretical justification for this class-wise formulation and demonstrates empirically, across multiple datasets, that the proposed models preserve interpretability and achieve better efficiency and discriminative power.

### Strengths
1.	The empirical results align well with the theoretical analysis, and Multi-ReduNet demonstrates significant performance improvements in the undersampled regime.
2.	The writing is clear, well-structured, and easy to follow.
3.	The theoretical analysis seems rigorous, and I did not find errors in the proofs.

### Weaknesses
1.	The experiments in this paper are primarily conducted on toy datasets such as MNIST. I believe it is necessary to include experiments on more realistic datasets, such as CIFAR. In addition, I am curious about the performance of Multi-ReduNet in the oversampled regime — is it still competitive under such conditions?
2.	The motivation behind Multi-ReduNet is clear and intuitive. However, the rationale for introducing Multi-ReduNet-LastNorm is somewhat unclear, as there is no theoretical comparison between the two. It would strengthen the paper to include a clearer discussion on the logical progression from Multi-ReduNet to Multi-ReduNet-LastNorm.
3.	I noticed that the main proofs are presented in the main text. This makes the paper rather math-heavy and potentially difficult to follow. It might be preferable to include simplified versions of the proofs in the main paper and move the detailed derivations to the appendix.
4.	In Figure 2, the representations learned by ReduNet appear well-clustered. However, Table 2 reports relatively low classification accuracy for the same model. Did I miss something here? It would be helpful to clarify the reason behind this apparent discrepancy between the results.
5.	The authors state that “Although the class-orthogonality property of MCR² optima has been established in prior work (Chan et al., 2021), our proof leverages a simpler and more streamlined argument.” I view this as one of the main theoretical contributions of the paper. However, it remains unclear in what sense the proof is simpler. A more detailed explanation or discussion would strengthen this claim.
6.	The presentation quality of the paper could be improved. For example, the title in Figure 2 is too small and difficult to read. Moreover, if the detailed proofs are moved to the appendix, it would be beneficial to include more discussion or intuitive explanations in the main text to improve readability and accessibility.
7.	In summary, this paper presents a theoretically grounded and interpretable extension of ReduNet with promising results. However, the work could be significantly strengthened through better presentation, clearer motivation for the proposed variants, and more comprehensive experiments on realistic datasets.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes two class-wise extensions to ReduNet, namely, Multi-ReduNet and Multi-ReduNet-LastNor, to improve the performance under undersampled, high-dimensional conditions. The authors theoretically justify that the global MCR² objective can be decomposed into class-specific subproblems and leverage this result to design more efficient and class-discriminative models. Experiments on multiple datasets show consistent improvements in classification accuracy and training efficiency, while maintaining interpretability.

### Strengths
1. The paper gives a theoretical analysis showing that the MCR² objective under certain conditions can be equivalently decomposed into class-wise subproblems. 
2. The proposed models maintain the white-box property of ReduNet and preserve its forward-only optimization strategy. This makes the approach more transparent and easier to analyze compared to conventional backpropagation-based deep networks.
3. The class-wise decomposition reduces the cost of matrix inversion from high-dimensional global matrices to smaller class-specific ones, which is computationally advantageous in settings with limited data and large feature dimensionality.

### Weaknesses
1. The method assumes that the class-wise decomposition is meaningful under undersampled regimes. However, the paper does not investigate how the approach behaves when this assumption is less valid, e.g., when the sample size is moderately large.
2. The evaluation focuses on relatively simple or small-scale datasets (e.g., MNIST, Fashion-MNIST), which may limit the conclusions.
3. The experimental comparisons are restricted to ReduNet and its variants. The paper could be strengthened by comparing against a broader set of interpretable or class-structured learning methods, to better situate the approach within the existing literature.
4. Although Multi-ReduNet-LastNorm performs slightly better in most cases, the role of the final-layer-only normalization is not fully analyzed. It would be helpful to understand under what circumstances this variant is preferable.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors improve the scalability of the classifier ReduNet, which is specialized for solving tasks where there are a lot more features $d$ than samples $m$. Given $K$ classes and $m_j$ samples for class $j$, new algorithms scales as $\mathcal{O}(\sum_{j=1}^Km_j^3)$ compared to the $\mathcal{O}(Kd^3)$ of the original algorithm. The scalability and performance is demonstrated on 4 real datasets.

### Strengths
The paper does a great job at explaining the ReduNet method to readers without prior knowledge on the field.

The theoretical contribution of the paper is significant since the new algorithm scales as $\mathcal{O}(\sum_{j=1}^Km_j^3)$ compared to the original $\mathcal{O}(Kd^3)$.

The improvements in accuracy are also significant compared to prior work.

### Weaknesses
## Orthogonality Constraints

At line 270 it is stated that the optimization problem outlined at line 269 is solved "subject to class-wise orthogonality and norm constraints".
I don't think the manuscript clarifies how these constraints are applied in practice. The proposed algorithm iteratively solves the equation at line 269 in order to infer the representation $Z_j$ of each class. The $Z_j$ are updated independently by maximizing a separate objective so it is not clear how orthogonality is enforced.

## A smoother-introduction to Multi-ReduNet

The paper actually has two contributions : Imp-ReduNet which changes the inversion of a $d\times d$ matrix to a $m\times m$, and
Multi-Redunet which furthers improves this to $K$ inversions of $m_j\times m_j$ matrices. But the order in which these contributions are presented in confusing : multi-redunet is presented first and then imp-redunet. I think that describing imp-redunet first would help motivate the need to separate the objective into $K$ objectives.

By introducing imp-redunet first (via lemma 1) it is clear to the reader that the computational bottleneck is inversion of a $m\times m$ matrix to compute the gradient of $\text{log det}(I+\alpha Z Z^T)$. If we were able to replace this with $K$ terms $\text{log det}(I+\alpha Z_j Z_j^T)$, we could inverse $K$ matrices of shape $m_j\times m_j$ instead, which is a lot better. After this initial high-level motivation, and hinting that $Z Z^T=\sum_{j=1}^K Z_j Z_j^T)$, the main theorems can be presented.

This is a subtle change, but I think it will improve the flow of the paper significantly.

## Technical Overload

On a similar topic, the paper would benefit from moving some technical content to the appendix and focusing more on high-level ideas in the main manuscript. Notably, the proof of the Theorem could be moved in the appendix, and replaced with a high-level proof description.
This description would only need to accentuate the most crucial parts of the proof e.g. that $\text{log det} (I+\alpha Z Z^T)= \text{log det} (I+\alpha \sum_{j=1}^K Z_j Z_j^T) =  \sum_{j=1}^K \text{log det} (I+\alpha Z_j Z_j^T)$ whenever class representations are orthogonal.
This clarifies to the reader that class-orthogonality is the key assumption to separate the objective into $K$ sub-objectives.

The freed space in the main manuscript could be used to introduce intuition for ReduNet e.g. extended content from the Appendix B.

## Figures 1

This is a very minor point, but I think that Figure 1 is hard to read. It is hard to see the color of the markers because it is very dark. Also, the marker color does not match with the line, so I constantly have to read the legend to be sure. Moreover, methods based on RF perform so bad that they hide the differences between ReduNet and Multi-Redunet (the main contributions of the paper). I would suggest removing the RF methods from the plot and simply indicate in the text that they take orders of magnitude more time.

### Questions
What is the $d_j$ in theorem 1? It is not introduced before being used.

Theorem 1 defines class-orthogonality as $Z_j Z_i^T=0$. Shouldn't is be $Z_j^T Z_i=0$ instead following Corollary 1? 

Why is the Frobenius norm $||Z_j||^2_F\leq m_j$ bounded in theorem 1 but the optimization algorithm projects to the unit sphere? Projecting $m_j$ points on the unit sphere guarantees that $||Z_j||^2\leq m_j$ , but the converse is not true. Perhaps a more in-depth introduction to ReduNet in the main manuscript would help.

From my understanding, multi-redunet an improvement in terms of scalability : it solves the same problem as ReduNet but more efficiently. Then how can multi-redunet perform better than redunet in terms of accuracy? Is ReduNet reaching a different optimum?

### Soundness
3

### Presentation
3

### Contribution
4
