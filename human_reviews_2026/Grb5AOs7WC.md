# Spurious Correlation-Aware Embedding Regularization for Worst-Group Robustness

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 8, 4, 6

## Abstract
Deep learning models achieve strong performance across various domains but often rely on spurious correlations, making them vulnerable to distribution shifts. This issue is particularly severe in subpopulation shift scenarios, where models struggle in underrepresented groups. While existing methods have made progress in mitigating this issue, their performance gains are still constrained. They lack a theoretical motivation connecting the embedding space representations with worst-group error. 
To address this limitation, we propose Spurious Correlation-Aware Embedding Regularization for Worst-Group Robustness (SCER), a novel approach that directly regularizes feature representations to suppress spurious cues. We theoretically show that worst-group error is influenced by how strongly the classifier relies on spurious versus core directions, as identified from differences in group-wise mean embeddings across domains and classes. 
By imposing theoretical constraints at the embedding level, SCER encourages models to focus on core features while reducing sensitivity to spurious patterns. Through systematic evaluation on multiple vision and language tasks, we show that SCER outperforms prior state-of-the-art methods in worst-group accuracy. Our code is available at \href{https://github.com/MLAI-Yonsei/SCER}{https://github.com/MLAI-Yonsei/SCER}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a regularization scheme for GroupDRO, called SCER, which minimizes intra-class inter-domain alignment and maximizes inter-class intra-domain alignment in the embedding space. Theoretical results are presented which justify the worst-group error decomposition into these two alignment terms, and experiments are performed on a variety of vision and language benchmarks.

### Strengths
1. Embedding-space regularization is an under-explored technique for group robustness. More generally, the community’s understanding of feature learning in the presence of spurious correlations is incomplete -- so methods which probe embedding space phenomena to design robust algorithms are particularly welcome.

2. The quantitative analyses and ablation studies are well-done, particularly under omission of one subpopulation and integration with the EIIL algorithm. The quantity and diversity of benchmark datasets is also a strength of the paper.

### Weaknesses
1. The comparison to previous methods (across all tables in the paper) is improperly organized and obscures the true performance differential of the proposed SCER method.

    a. The number of previous methods in Table 1 is somewhat excessive: as this is not a benchmarking paper, there is no need to compare to 19 previous methods, especially older algorithms like IRM whose performance has been broadly superseded (a citation is sufficient).

    b. Many of the compared methods utilize different data and annotation assumptions than SCER, and therefore do not represent fair comparisons. Specifically, SCER requires full group annotations on the training dataset since it is an extension of GroupDRO. Comparisons to methods like ReWeight (which does not use group annotations at all) and DFR (which uses group annotations only on a small held-out set) inflate the perceived performance of SCER relative to previous work. Overall, these methods should not be included in the comparison and/or whether group annotations are utilized in the training set, held-out set, or validation set should be clearly denoted.

2. Since SCER is essentially a proposed regularizer for GroupDRO, the primary comparison should be to GroupDRO. However, the paper lacks evidence that SCER represents a substantial improvement over GroupDRO.

    a. The comparison with this paper’s implementation of GroupDRO is somewhat obfuscated. It should be made clear that SCER with both regularization hyperparameters to zero reduces to GroupDRO, and the numbers from Table 6 and Table 11 included as such in Table 1. This is important because implementation details can vary performance: for example, the GroupDRO WGA in Table 1 for ColorMNIST is 73.1%, but in Table 6 one can see it decreases to 70.7% in this paper’s implementation.

    b. The paper [1] which this paper draws GroupDRO numbers from does not represent the most competitive GroupDRO baseline. In particular, it is well-known that GroupDRO requires early-stopping to achieve the best performance [2, 3]. Compared to the early-stopped GroupDRO numbers of [3], the proposed SCER method barely improves performance. On Waterbirds, SCER improves from 90.7% to 91.2% (Table 1). On CelebA, SCER improves from 90.6% to 91.4% (Table 1). On CivilComments, SCER worsens from 80.4% to 74.0% (Table 5). On MultiNLI, SCER improves from 73.5% to 76.8% (Table 5). Keeping in mind that SCER tunes two additional hyperparameters compared to GroupDRO, or one additional hyperparameter including tuning the early-stop threshold, its improvement over GroupDRO is not very significant.

    c. This is less important, but the Grad-CAM visualization in Figure 4 should also compare to GroupDRO. It is possible that the alignment of the gradient magnitude and the core feature is a property of GroupDRO and not of the SCER regularization scheme.

3. Justification of theoretical assumptions:

    a. In Theorem 1, it is assumed that the _data_ covariance matrices $\Sigma^{d_R}=\Sigma^{d_G}=\Sigma$, but in the SCER algorithm, the _feature_ covariance matrices are utilized to compute the Mahalanobis norm (lines 203-204). Since neural networks are nonlinear models, the feature covariances may not be equal even if the data covariance matrices are. This observation has been confirmed empirically in the literature, e.g., [4] observed a greater spectral imbalance in the empirical covariance matrix of the majority group features compared to the minority group.

    b. The extreme/complete correlation assumption that $\mathbb{P}(y,d)\approx 0$ for some groups $(y,d)$ is fine (as theory papers often decay the number of minority group data with dimension, e.g., [5]), but it should be justified more carefully. In particular, none of the datasets except ColorMNIST with an omitted subpopulation (Table 3) exhibit an extreme correlation.

[1] Yang et al. Change is hard: a closer look at subpopulation shift. ICML 2023.

[2] Sagawa et al. Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. ICLR 2020.

[3] Izmailov et al. On Feature Learning in the Presence of Spurious Correlations. NeurIPS 2022.

[4] LaBonte et al. The Group Robustness is in the Details: Revisiting Finetuning under Spurious Correlations. NeurIPS 2024.

[5] Lai and Muthukumar. Sharp analysis of out-of-distribution error for "importance-weighted" estimators in the overparameterized regime. ISIT 2024.

### Questions
1. How do the embedding space dynamics and the performance of SCER change under optimal early-stopping, and how does it compare to standard early-stopped GroupDRO?

2. Is there any connection between the embedding alignment terms proposed by this paper and recent studies on neural collapse and group robustness [1, 2]? Specifically, the neural collapse (NC1) phenomenon suggests that the proposed $\Delta_{spur}$ metric is naturally minimized during training as embeddings collapse to the class means. I’m also curious whether SCER has any relation to the “group mean alignment loss” of [2].

3. There are several grammatical errors in the paper:

    a. Line 24

    b. Figure 1 lines 64-65

    c. Line 76

    d. Figure 2 lines 167, 171

    e. Section 5.1 and 5.2 have repeated titles

    f. Equation references are malformed in the Appendix

[1] Papyan et al. Prevalence of neural collapse during the terminal phase of deep learning training. PNAS 2020.

[2] Lu et al. Neural Collapse Inspired Debiased Representation Learning for Min-max Fairness. KDD 2024.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a novel method for mitigating spurious correlations by developing clear constraints on the embeddings learned from different subpopulation groups. The core feature is calculated as the average difference of embeddings from samples with the same domain but different labels, while the spurious feature considers the difference for samples with the same label but different domains. Next, the alignment between classifier weights and the directions of the normalized features is calculated and used as an additional loss (constraint) alongside the common worst-group classification loss. Experiments are conducted on popular datasets for studying spurious correlations, and the proposed method is compared with many baseline methods. The results demonstrate that the proposed method works as designed and is very promising.

### Strengths
(S1) The proposed constraints on the embeddings are explicit and look novel to me, especially the correlation terms defined in Eq. (1), which are well supported by Theorem 1. I believe the research directions enlightened by this paper are very promising. Besides, the Introduction and Related Works sections also provide a good discussion about this paper's contributions, which places it well among the literature on spurious correlations. 

(S2) Comprehensive experimental settings provide strong evaluations of the performance of the proposed methods, although some improvements are marginal in Table 1. In general, I believe the evidence is enough to demonstrate that the proposed method is very promising. In particular, I really appreciate the results in Figure 3, which demonstrate that the proposed constraints on the correlation terms are working and work as designed. 

(S3) In general, this paper is well written and easy to follow.

### Weaknesses
(W1) While a schematic diagram/representation of the embeddings learned from the proposed methods is presented in Figure 1, a comprehensive evaluation of the structure and clusters of the learned embeddings, in comparison to baseline methods, is not included in this paper, which weakens the evaluation of a key contribution of this paper: the explicit constraints on the embeddings. In particular, it would be very helpful for readers to examine the differences between the embeddings learned by the proposed methods and those learned by works focusing on learning invariant features.

### Questions
(Q1) The authors are encouraged to respond to (W1) I described above. Additional simple experimental results in the appendix would be much appreciated and could have a big impact on my final opinion, such as figures of t-SNE plots labeling different subpopulations, especially in comparison to baseline methods that also operate in the embedding space and focus on invariant features. 

(Q2) (Minor) In Figure 1, instead of using mathematical notations for the proposed core/spurious features, plain language descriptions might improve readability. Personally, I had a difficult time understanding that part before reading the Methods section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a new method to improve worst-group robustness by encouraging models to rely on core features while suppressing spurious correlations in the embedding space. It introduces a regularization term that rewards alignment with class-discriminative directions and penalizes reliance on non-causal relations. Experiments on vision and language benchmarks show consistent gains in worst-group accuracy over existing methods.

### Strengths
- The method is concisely proposed with theoretical analysis
- The results are clearly explained with interpretable evidence

### Weaknesses
- The core idea is to encourage models to rely less on spurious correlations and more on invariant features. This has been explored in a substantial body of prior work both empirically and theoretically. The contribution here, while sound, does not clearly break new conceptual ground, considering the improvements over baselines appear modest.
- I believe this work, as well as similar ones, should be applied to datasets with a larger number of groups. For example, where the number of groups is greater than 50, the spurious correlations are not very obvious to humans under the given group divisions.

### Questions
What do you think are the potential reasons for ElRep’s significant performance drop on MetaShift and ColorMNIST?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Spurious Correlation-Aware Embedding Regularization (SCER) as a method to improve on worst-group robustness by proposing embedding-level regularization. The authors provide theoretical justification for decomposing worst-group error into spurious and core components. The method identifies spurious directions as differences in embeddings across domains within the same class, while core directions are differences across classes within the same domain and combines them into the SCER objective that penalize classifier alignment with the spurious direction while encouraging alignment with the core direction. Experiments on vision and language datasets show improvements over existing methods, particularly in extreme settings where a subpopulation is completely missing during training.

### Strengths
The paper precisely defines core and spurious mean-embedding differences, aggregates them into global directions, and integrates their alignment with classifier weights into a composite loss term that is transparent and straightforward to implement.
 
The authors conduct comprehensive experiments 6 datasets across vision and language domains, comparing against 19 baseline methods. The settings also cover different spurious correlation strengths and extreme scenarios.

Notably, the method substantially outperforms baselines when one subpopulation is entirely absent during training on ColorMNIST.
The authors conduct sensitivity and component analyses to support specific design choices and validate their theoretical decomposition.

### Weaknesses
When formulating the theorem for error decomposition, the paper is restricted to binary labels and two domains and assumes constant core and spurious differences between domains and classes. The paper does not discuss how violations of these assumptions affect the method's validity, particularly for multi-class problems, which appear in experiments like MultiNLI with 3 classes. 

It can be argued that correctly specified group labels are required to operationalize SCER in terms of its component losses. While this is briefly explored with EIIL in a two-environment setup, it would be important to understand how SCER behaves when group labels are noisy. For example, how does robustness vary when EIIL’s inferred groups are partially misaligned to 70-80% agreement instead of 95%?

The paper computes group-wise mean embeddings but doesn't discuss key considerations, such as how many samples are needed for stable estimates, what happens when group sizes are highly imbalanced, and what happens if spurious and core features are not well-separated in embedding space.

Furthermore, aggregating across all classes or groups into a single spurious and single core direction may lead to underfitting in settings with multiple heterogeneous spurious factors. A discussion and validation of how this is handled is missing.

### Questions
Would it be possible to discuss how extending the error decomposition theorem beyond the two-class/groups assumption can be justified for the sake of generality?

How does SCER behave in terms of robustness and performance when group labels are noisy, for instance, when inferred groups are partially misaligned to 70-80% agreement instead of 95%? 

How would you handle when multiple spurious or core directions exist? Can you provide evidence that consolidating into a single global direction is optimal?

How many training steps does SCER typically require compared to baselines? The per iteration time is provided, but not the full training cost

### Soundness
3

### Presentation
3

### Contribution
3
