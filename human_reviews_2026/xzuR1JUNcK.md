# Stabilizing Heterogeneous Federated Learning via Feature Decorrelation and Bidirectional Alignment

- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Data heterogeneity poses a major challenge in federated learning, leading to significant degradation in global model performance. Prior studies have shown that heterogeneity induces dimensional collapse and biased classifiers, which hinder the learning of both feature extractors and classifiers. To tackle these issues, existing approaches apply feature decorrelation to mitigate dimensional collapse and adopt a synthetic classifier with a projector to reduce classifier bias. However, these decorrelation methods fail to prevent small singular values from collapsing to zero, slowing the mitigation of dimensional collapse. Besides, the synergy among the feature extractor, projector and synthetic classifier is overlooked, leading to divergent optimization across clients. To overcome these limitations, we propose FedBlade, a federated learning framework with bilateral alignment and feature decorrelation. Our feature decorrelation method accelerates the mitigation of dimensional collapse by yielding exponential gradients, while the bilateral alignment method enhances synergy among model modules and ensures consistency across clients. Extensive experimental results demonstrate that FedBlade outperforms relevant baselines and achieves faster convergence of the global model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- Connects the the non-IID problem in Federated Learning (FL) to the collapse of singular values. While the connection of the singular values and the collapse that happens in non-IID FL is well documented in prior work, the authors make a point that previous studies do not focus on preventing collapse of the smaller singular values.

- Introduces a new decorrelation method of minimizing the negative log-determinant of the representation correlation matrix. They argue that this approach prevents the collapse of the of the smaller singular values, which expands the useable feature space. With the expanded space, they argue that their alignment method improves performance.

### Strengths
- The authors do well in emphasizing the need to focus on the small singular values.
- The design of the decorrelation loss is quite clever. I don't think I've seen the determinant being used for a decorrelation loss, but this makes sense when we're focusing on the smaller singular values.
- Decent results in terms of performance and speedup.

### Weaknesses
- The main problem I have with this paper is the lack of transparency for the efficacy of the method. The authors' central claim is that their method increases effective rank, as their loss more harshly penalizes the small singular value. Since the authors claim this, they MUST also provide the effective rank for other methods, so readers can truly see that their method is effective not only in increasing the effective rank, but also the performance (which they do show). From Figure 4, they only show ablations of their own method. I could not find the effective rank for other methods.
- I also think the authors have missed two key citations [1], [2]. [1] introduced the idea of keeping the classifier fixed, because the classifier leads to much bias. [2] made connections to the singular values, debiasing, and increasing feature space.
- The core idea of this paper is also quite similar to that of [2]. FedUV uses Uniformity to increase the feature space, and Variance to debias the classifier. I feel this paper needs to frame their paper in a different light to highlight their novelty (likely the decorrelation loss).

- I feel the name of 'bidirectional alignment' is quite poor. It's alignment to the projector and the output of the encoder. There's no 'bi-directional' anything going on. Maybe 'bi-focused'? I'm sure there would be a better name. The goal is the convey that you are targeting two places in the network, not two directions.

[1] Fedbabu: Towards enhanced representation for federated image classification, ICLR 2022.
[2] FedUV: Uniformity and Variance for Heterogeneous Federated Learning, CVPR 2024

### Questions
1. Can the authors provide more transparency into the effective rank for other methods? They should also provide their rationale backed by as much data as they can provide regarding this point.
2. Can the authors frame their method in a more novel light, by first citing the very relevant papers mentioned above?

### Soundness
3

### Presentation
2

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
This paper proposes FedBlade, a federated learning framework addressing label skew via two main components: (1) LDDecorr, a log-determinant based feature decorrelation regularizer that produces exponential gradients to impose stronger penalties on small singular values, thereby mitigating dimensional collapse better than FedDecorr; and (2) PBA, a prototype-guided bidirectional alignment mechanism that aligns the feature extractor and projector with a shared ETF classifier through global prototypes. The whole method builds upon FedETF and FedDecorr, aiming to enforce neural collapse and reduce dimensional collapse, under label skew FL.

### Strengths
1. The motivation and intuition of this work are clear and sound, which provides a well-articulated motivation for addressing dimensional collapse and classifier bias in federated learning. 

2. This work has solid theoretical grounding. Theoretical analysis connecting the log-determinant term to spectrum isotropy and effective rank is insightful.

3. Using global prototypes as bridges between modules (extractor–projector–ETF classifier) is a natural and elegant idea that improves interpretability.

4. The paper is well-structured, and the presentation is clear and easy to follow.

### Weaknesses
1. The proposed method is an incremental improvement over FedETF and FedDecorr, mainly combining a modified decorrelation loss and a prototype-based alignment. The novelty is moderate.

2. The improvement is limited and often small, especially under full client participation.

3. The ablation results are somewhat puzzling: on CIFAR-100, LDDecorr improves accuracy while PBA reduces it; on Tiny-ImageNet, the opposite trend occurs. This suggests dataset-specific sensitivity or complex interactions between the two modules that are not well explained in the paper.
Interestingly, when both components are applied together, the overall performance improves significantly and consistently across datasets, implying that LDDecorr and PBA may complement each other in a deeper way. For instance, LDDecorr might enhance the representation isotropy that PBA relies on for effective alignment, while PBA may in turn stabilize the projection space required for LDDecorr to operate effectively. A more detailed and theoretical analysis of this potential synergy would strengthen the paper’s understanding of why the two modules jointly work better than each alone.

4. Although LDDecorr theoretically reduces computational cost via Cholesky decomposition, there exists a bad case:  on Tiny-ImageNet with α=0.5, where FedDecorr converges faster. Authors should explain why.


5. The experimental scope is narrow. Experiments use only MobileNetV2 and three vision benchmarks. Testing on additional architectures or non-vision datasets would strengthen the generality claim.

6. Lack of cost metrics. No wall-clock, FLOPs, or communication-overhead analysis is provided to substantiate claims of improved efficiency.

### Questions
1. The improvements under full client participation seem limited. Could the authors provide a detailed analysis of why FedBlade’s advantages diminish in this setting?

2. The ablation trends are inconsistent across datasets (Table 3). Can the authors explain why LDDecorr and PBA sometimes have opposite effects? Is this due to prototype drift, sensitivity to γ/β, or data distribution differences?

3. Could the authors show convergence curves or wall-clock time comparisons for α=0.5 to better support the claim of “faster convergence”?

4. How sensitive are the results to hyperparameters β, γ, and τ?

5. Does PBA introduce additional communication overhead when aggregating global prototypes, and if so, how significant is it?

6. The paper mentions that FedBlade “enforces neural collapse.” Could the authors provide quantitative evidence (e.g., NC1–NC4 metrics) to substantiate this claim?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a two-pronged approach to dealing with label skew in Federated Learning for natural image classification. The first component modifies the regularization term introduced in FedDecorr (which encourages different dimensions of representations to be uncorrelated) from the Frobenius norm of the correlation matrix to the negative log-determinant of the same. The second component consists of two sub-components: (i) the projector & frozen ETF classifier head first proposed in FedETF that leverages the neural collapse phenomenon to reduce classifier bias (ii) global class prototypes (like in FedProto) that both the projector and main network are aligned with via separate alignment losses. Experiments are conducted on CIFAR10/CIFAR100 and TinyImageNet, under Dirichlet non-IIDness with alpha values 0.05, 0.1, 0.5, using an untrained MobileNetV2, against 8 baseline methods. There is also an ablation experiment of the components, and sensitivity analysis of the introduced hyper-parameters.

### Strengths
The paper benefits from a clear motivation, namely mitigating dimensional collapse and addressing classifier bias at the same time. Figure 1 helps build intuition to this effect. The idea itself is interesting even if its not that novel (see W1). The experimental setup for the most part falls in line with similar papers, and the range of compared baselines is pretty diverse. The paper is generally well-written, with good pacing, and the target problem is an important one.

### Weaknesses
I appreciate the authors' efforts in tackling the important problem of heterogeneity in federated learning. However, I have some concerns that I hope can be addressed to strengthen the contribution:

1. Theoretical novelty and experimental depth: The paper presents an interesting combination of previous work, but I found myself wanting more depth in certain areas:
* The approach builds upon FedDecorr (replacing the Frobenius norm with log-determinant) and FedETF (incorporating prototypes similar to FedProto/FedNH [3]). Given this foundation, I believe the paper would benefit from either theoretical justification for why this combination is effective or more comprehensive empirical validation.
* The log-determinant technique for matrix rank minimization is well-established [1], so it would be helpful to clarify the specific novelty here.
* While the methodology section presents various formulas, some appear to go unused in the final approach. Additionally, formal analysis (e.g., convergence guarantees or theoretical properties) would strengthen the contribution.
* The experimental scope could be broadened to include more diverse heterogeneity scenarios (e.g., domain shift, real-world datasets, less extreme α settings). I also noticed the model backbone doesn't leverage recent FL research showing that pre-trained models and architectures without BatchNorm can significantly mitigate non-IID performance issues.

2. Related work positioning: The paper would benefit from a broader contextualization within the federated optimization literature:
* While the focus on dimensional collapse and classifier bias is valuable, the heterogeneity challenge in FL has been extensively studied from multiple angles. I'd suggest considering [2] as well, which shares conceptual similarities.
* Recent work has demonstrated that BatchNorm usage and non-pretrained models contribute significantly to performance degradation under non-IID data. Acknowledging this literature and clarifying when/where the proposed approach is most advantageous would help readers understand the method's positioning. Happy to provide some starter references on this if needed.

3. Experimental design considerations: I have several concerns about the experimental setup that might affect the interpretation of results:

* The choice of E=5 local epochs for 100 rounds is interesting. Since higher E under non-IID conditions often degrades performance, could you provide an ablation study (similar to FedDecorr's analysis) or justification for this choice versus E=1? I also noticed in Figures 3, 4, and 7 that algorithms haven't converged at T=100. Would a lower E with longer training change the conclusions and result ranking?

* Under full participation (Table 6), improvement is marginal, with baselines outperforming or performing within the variance range of the proposed method, despite being simpler. Could you discuss these results?
* I noticed FedDecorr's $\beta$ is set to 10 here, while their paper used 0.1 under similar settings. Could you clarify this 100x difference to ensure fair comparison?
* Regarding Figures 5 and 6: the 10% Y-axis increments make the sensitivity appear much lower than it really is, can the authors provide a rendering where the increments are more in line with the performance difference between the proposed method and baselines (e.g. 2% increments)? This would help contextualize the sensitivity better. If test set tuning was used for these hyperparameters, as it appears to be the case, this might disadvantage simpler baselines with fewer hyperparameters. Can the authors comment on this?

## Minor corrections
- Typo line 108 Relate Work -> Related Work
- Typo line 237 definded -> defined

I'm happy to discuss these points further and would be open to reconsidering my assessment if these concerns can be addressed.

## References

[1] Fazel, M., Hindi, H. and Boyd, S.P., 2003, June. Log-det heuristic for matrix rank minimization with applications to Hankel and Euclidean distance matrices. In Proceedings of the 2003 American Control Conference, 2003. (Vol. 3, pp. 2156-2162). IEEE.

[2] Guo, Y., Tang, X. and Lin, T., 2023, July. Fedbr: Improving federated learning on heterogeneous data via local learning bias reduction. In International conference on machine learning (pp. 12034-12054). PMLR.

[3] Dai, Y., Chen, Z., Li, J., Heinecke, S., Sun, L. and Xu, R., 2023, June. Tackling data heterogeneity in federated learning with class prototypes. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 6, pp. 7314-7322).

### Questions
1. The authors present accuracy averaged over the last 10 rounds (10% of training). Can they explain the reasoning behind this decision?
2. I'd appreciate access to the code, which is mentioned by the authors, to better understand the implementation details.

### Soundness
1

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
Data heterogeneity in federated learning, particularly label skew, poses a significant challenge by causing dimensional collapse (where features become low-rank) and classifier bias, ultimately degrading global model performance. To address these issues, this paper proposes FedBlade, a novel framework integrating two key components: LDDecorr and PBA. LDDecorr is a feature decorrelation method that maximizes the log-determinant of the feature correlation matrix; this yields exponential gradients that, unlike previous linear methods, apply an "infinite penalty" to small singular values, effectively and rapidly mitigating dimensional collapse. PBA (Prototype-guided Bidirectional Alignment) enhances the synergy between the model's feature extractor, projector, and a fixed ETF classifier by using global prototypes as a common reference, ensuring these modules are aligned and consistent across clients. Extensive experiments show that FedBlade outperforms existing baselines in accuracy and achieves substantially faster convergence.

### Strengths
1. The paper proposes LDDecorr, a novel feature decorrelation method. Unlike previous approaches like FedDecorr that use linear gradients, LDDecorr uses a log-determinant formulation to produce exponential gradients ($\nabla_{\lambda_{i}} = -1/\lambda_{i}$). This is a significant strength because it imposes an "infinite penalty" on small singular values, preventing them from collapsing to zero much more effectively and accelerating the mitigation of dimensional collapse.

2.  The paper identifies a key weakness in prior work (like FedETF) that uses fixed classifiers, which lacks synergy between the feature extractor, projector, and classifier. Its second component, PBA (Prototype-guided Bidirectional Alignment), directly solves this. It uses global prototypes as a common "bridge" to simultaneously align the feature extractor with the prototypes and the projector with the fixed classifier, ensuring all model parts work together coherently.

3. FedBlade consistently outperforms a wide range of relevant baselines (including FedAvg, FedDecorr, and FedETF) in final accuracy, especially on complex datasets like CIFAR-100 and Tiny-ImageNet. Furthermore, it shows FedBlade achieves substantially faster convergence.

### Weaknesses
1. Feature decorrelation method (LDDecorr) is sensitive to the decorrelation strength,. 

2.  The proposed PBA method requires an extra communication step in each round. Clients must compute and send their local prototypes to the server, and the server must aggregate the global prototypes and send them back to the clients. This adds to communication costs, a key bottleneck in federated learning.

3. The FedBlade framework adds computational overhead on both the client and server. Clients must perform extra calculations for the new loss terms: $\mathcal{L}_{LDDecorr}$ which involves a determinant calculation and the two PBA losses. The server also has the new task of aggregating all client prototypes.

### Questions
1. What are the limitations of the proposed approach? 

2.  How do the computational costs of LDDecorr and the communication costs of PBA scale, especially with high-dimensional feature spaces and a large number of classes? At what point do these added costs make FedBlade less practical than faster, simpler methods?

### Soundness
3

### Presentation
3

### Contribution
3
