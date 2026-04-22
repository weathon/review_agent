# Aligning Collaborative View Recovery and Tensorial Subspace Learning via Latent Representation for Incomplete Multi-View Clustering

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Multi-view data usually suffer from partially missing views in open scenarios, which inevitably degrades clustering performance. The incomplete multi-view clustering (IMVC) has attracted increasing attention and achieved significant success. Although existing imputation-based IMVC methods perform well, they still face one crucial limitation, i.e., view recovery and subspace representation lack explicit alignment and collaborative interaction in exploring complementarity and consistency across multiple views. To this end, this study proposes a novel IMVC method to Align collaborative view Recovery and tensorial Subspace Learning via latent representation (ARSL-IMVC). Specifically, the ARSL-IMVC infers the complete view from view-shared latent representation and view-specific estimator with Hilbert-Schmidt Independence Criterion regularizer, reshaping the consistent and diverse information intrinsically embedded in original multi-view data. Then, the ARSL-IMVC learns the view-shared and view-specific subspace representations from latent feature and recovered views, and models high-order correlations at the global and local levels in the unified low-rank tensor space. Thus, leveraging the latent representation as a bridge in a unified framework, the ARSL-IMVC seamlessly aligns the complementarity and consistency exploration across view recovery and subspace representation learning, negotiating with each other to promote clustering. Extensive experimental results on seven datasets demonstrate the powerful capacity of ARSL-IMVC in complex incomplete multi-view clustering tasks under various view missing scenarios. The source code is publicly available at https://github.com/caoyu110/ARSL-IMVC.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel method, ARSL-IMVC, to tackle the task of incomplete multi-view clustering. This method combines collaborative view recovery (CVR) and tensor subspace learning (TSL) through a shared latent representation, aiming to enhance cross-view consistency and complementarity while strengthening the synergistic interaction between view recovery and subspace representation. Extensive experimental results on seven datasets validate the effectiveness and robustness of the method. Overall, the paper is well-structured, with clear motivations and promising results, and some concerns require further exploration.

### Strengths
1).This paper proposes a method for aligning view recovery and tensor subspace learning using latent representations, which is well-structured.

2).The paper's research motivation is clear, the model design logic is clear, and it is easy to understand.

3).The experiments are detailed, and the method is compared with multiple baseline methods on multiple datasets at different missing rates and the experimental results show the method's good performance.

### Weaknesses
1).The background knowledge about tensors should be added with more details, which can help readers better understand the methods proposed in the paper.

2).The model involves alternating optimization of multiple variables, and the computational cost of tensors is high, lacking analysis of computational complexity.

3). The convergence of optimization process should be analyzed.

4). Does the tensor low-rank regularization preserve local manifold structures within each view, or mainly capture global correlations? Would adding a local structural constraint improve the model?

### Questions
See Weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The IMVC is an important topic and relies on discriminative representation learning. Most of existing imputation-based IMVC methods perform well, but they still face one crucial limitation, i.e., view recovery and subspace representation lack explicit alignment and collaborative interaction in exploring complementarity and consistency across multiple views. These issues are crucial for IMVC. This study proposes a novel ARSL-IMVC that aligns collaborative view recovery and tensorial Subspace Learning via latent representation. It is a simple yet effective method, and this is interesting design for IMVC. Extensive experimental results on seven datasets demonstrate the
powerful capacity of ARSL-IMVC in complex incomplete multi-view clustering tasks under various view missing scenarios. Overall, this paper has a clear and novel contribution to IMVC, and some concerns also need to be clarified.

### Strengths
1.The paper is well-written and easy to follow，ARSL-IMVC provides a new and cohesive framework that aligns consistency and complementarity across views. 

2.The method analysis is relatively comprehensive, and the optimization steps are clearly explained. The experimental design is comprehensive, validating the performance of the method on seven datasets covering various missing rates, and comparing it with several representative baseline methods.The ablation and visualization analyses are well designed and strongly support the effectiveness of the proposed modules.

### Weaknesses
1.What is HW dataset? There is a lack of introduction to the statistical information of the used dataset. This is beneficial for readers.
Some experimental settings need to be further provided.

2. Some additional concerns can be found in **Questions**.

### Questions
1.When stacking view-shared and view-specific subspace representations into a tensor, does the optimization enforce any independence between shared and specific subspaces, making the features tend to be the same?

2.Besides the current averaging strategy, are there other fusion strategies explored to integrate shared and view-specific subspace representations? For example, learning an adaptive weight.

3.There is a lack of detailed analysis of the severe performance degradation of certain datasets under high missing rates. Further explanation and improvement strategies are recommended.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work addresses incomplete multi-view clustering by introducing an aligned representation and subspace learning model. Unlike conventional imputation-based methods, ARSL-IMVC employs a shared latent representation to guide view recovery and tensor subspace construction simultaneously, thus promoting collaborative information exchange across views. The framework is novel and effective, enabling the model to capture both global coherence and view-level diversity. Comprehensive experiments across multiple datasets demonstrate the model's robustness and superior clustering performance under different missing view scenarios. The idea is novel, clear, and provides a meaningful advancement for incomplete multi-view clustering.

### Strengths
1. The paper proposes a novel framework for joint view recovery and tensor subspace learning. It explicitly aligns the two modules through a shared latent representation, which is relatively uncommon in existing IMVC methods.

2. The experimental design is comprehensive, covering various missing rates and including comparisons with multiple baseline methods. The results are superior and clearly demonstrate the effectiveness of each module.

### Weaknesses
1. Lack of introduction to datasets and baseline methods.

2. While results are averaged over 10 runs, reporting the standard deviation would make the experimental conclusions more statistically convincing.

3. Two main differences as induced in this paper need to be more detailed. And the authors are suggested to give more explanations related to some key words, e.g., "limited structural fidelity", "insufficient diversity and consistency reshaping", etc. This makes it easier for readers to understand these opinions. And some empirical or theoretical evidence is more favoured.

4. Notations are not consistent. Please see my questions.

5. The motivations of some equation formulations are not clear. Please see my questions.

6. Complexity analysis is lacking.

### Questions
Apart from the above weaknesses, I still have some questions:

1. The semantic alignment between view recovery and subspace representation is new, its innovation needs further clarification, especially the difference from existing methods.

2. Exploring the criteria of cross-view consistency and complementarity has been widely used in multi-view learning. What is the contribution of this study in this regard?

3. If the latent representation has already been trained to capture shared semantics, does the introduction of a low-rank constraint on the subspace representation tensor result in redundant regularisation?

4. The HSIC regularisation term can use different kernel functions. Why does ARSL-IMVC choose to use the inner product kernel?

5. The latent representation H and the centralised matrix H use the same notation.

6. The authors introduced a view-specific orthogonal matrix $P^{(v)}$ to recover the view-specific features. I have a concern about whether $(P^{(v)})^\top P^{(v)} = I$ can still hold when the dimension of view-specific features is lower than $k$.

7. The choice of HSIC: why this paper chooses HSIC as a diversity measurement here rather than some others, e.g., Gromov-Wasserstein distance.

8. What are the global semantics and local cluster structures in Eq. (4) ?

9. This paper utilises a rotation operator to transfer a $N\times N\times (V+1)$ tensor to a $ N\times (V+1) \times N$  tensor, making the results of the tensor low rank constraint sensitive to the orders of samples arranged in the input features. How does this paper address this issue?

10. The datasets used in this paper are small-scale.

The scores of this paper will be raised after addressing the above weaknesses and questions.

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
4

### Summary
This paper proposes a novel ARSL-IMVC method for the classic incomplete multi-view clustering task, which aligns the collaborative view recovery and tensorial subspace learning in cross-view complementarity and consistency exploration. It is interesting, which is the inverse idea of classical multi-view clustering. It uses an unknown latent shared representation to reverse infer missing multi-view data and act on the subspace representation learning. Latent representation as a bridge helps to explicitly realize cross-view correlation exploration. The extensive experimental results verify its effectiveness.

### Strengths
(1) This paper is clearly motivated and proposes to achieve the "semantic alignment" of view recovery and subspace learning through a shared latent representation in incomplete multi-view clustering, which is innovative.
(2) The logic and writing of the paper are satisfactory.
(3) Experiments on several datasets with multiple missing ratios provide convincing evidence of the clustering performance robustness.

### Weaknesses
(1) The choice of latent representation dimensions remains empirical, lacking selection guidance or systematic analysis. 
(2) For clustering tasks on large-scale datasets, runtime efficiency is crucial. But the paper does not provide an analysis of runtime, leaving its practical scalability unclear.

### Questions
(1) How sensitive is the model to the dimension of the latent representation? Could a too-small latent space limit the recovery ability? Please elaborate and explain further.
(2) The low-rank tensor constraint may lead to over-smoothing of the subspace representations. How does the model maintain clustering discriminability under such constraint?
(3) The proposed model seems to be complex; does it have an advantage in training time compared to existing methods? Please provide relevant comparative experiments.

### Soundness
4

### Presentation
3

### Contribution
3
