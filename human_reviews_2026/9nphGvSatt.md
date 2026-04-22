# ELViS: Efficient Visual Similarity from Local Descriptors that Generalizes Across Domains

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Large-scale instance-level training data is scarce, so models are typically trained on domain-specific datasets. Yet in real-world retrieval, they must handle diverse domains, making generalization to unseen data critical. We introduce ELViS, an image-to-image similarity model that generalizes effectively to unseen domains. Unlike conventional approaches, our model operates in similarity space rather than representation space, promoting cross-domain transfer. It leverages local descriptor correspondences, refines their similarities through an optimal transport step with data-dependent gains that suppress uninformative descriptors, and aggregates strong correspondences via a voting process into an image-level similarity. This design injects strong inductive biases, yielding a simple, efficient, and interpretable model. To assess generalization, we compile a benchmark of eight datasets spanning landmarks, artworks, products, and multi-domain collections,
and evaluate ELViS as a re-ranking method. Our experiments show that ELViS outperforms competing methods by a large margin in out-of-domain scenarios and on average, while requiring only a fraction of their computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ELViS (Efficient Local Visual Similarity), a novel model for instance-level image retrieval. The core problem addressed is single-source domain generalization: training a model on one domain (e.g., landmarks) and having it perform well on retrieval tasks in unseen domains.

**Method Overview:**  
- Input: Local descriptors are extracted from two images using a foundation model (like DINOv2).  
 - Similarity Matrix Refinement: The local descriptor similarity matrix is refined using Optimal Transport with a key innovation: descriptor-dependent dustbin gains. This allows the model to learn to ignore uninformative descriptors (e.g., from the background).  
- Vote Aggregation: For each descriptor, the strongest correspondence (similarity) is selected as a `"vote." A small, learned function \$f\$ transforms these vote strengths, and they are summed to produce a final, global image similarity score.  
- Training: The model uses a modified Binary Cross-Entropy loss with a second learned function g to reshape the penalty curve during training, which is discarded at inference.  


**Main Results:**  
The authors demonstrate that ELViS achieves state-of-the-art performance on a comprehensive benchmark of 8 datasets, showing superior generalization to out-of-domain data while being significantly faster and more parameter-efficient than competing transformer-based methods (RRT, R²Former, AMES).

### Strengths
- The paper is well written and clear to follow.  
- SOTA Cross-Domain Generalization: The primary strength of ELViS is its ability to perform robustly on domains unseen during training. It consistently outperforms all competitors on out-of-domain (OOD) datasets.  
- The method is efficient on both parameters count and latency aspects.  
- Ablation studies on the contribution of each component is provided and well demonstrated.

### Weaknesses
- Justification for \$g\$: The use of the learned function \$g\$ in the loss, which is discarded at inference, is an unconventional and somewhat non-standard technique. While it works well empirically, a more rigorous theoretical explanation for why this is necessary and why discarding it is valid would strengthen the method.  
- Performance on In-Domain Data: In some in-domain (ID) settings, ELViS is outperformed by other methods. This suggests that while its bias towards generalization is powerful, it might come at the cost for optimal performance on the training domain itself.  
- Novelty: The method primarily relies on existing components, which somewhat limits its degree of novelty.

### Questions
-

### Soundness
2

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
4

### Summary
This paper introduced a image-to-image similarity model that promotes cross-domain transfer, namely Efficient Local Visual Similarity (ELViS). In order to facilitate faster and more explainable image retrieve in cross-domain scenarios, ELViS leveraged optimal transportation to refine the local description similarity matrix S’, and then aggregate a learnable voting process to transfer the local similarity to the global similarity for the further image retrieval. 
Their major technical contribution of their work lies in (1) the construction of refined local-description similarity matrix S’, which contains traditional similarity matrix S, data-dependent gains (learned by parametric method) that suppress uninformative descriptors (dustbins) and the learnable scalars that stands for transportation mass for dustbins. (2) the voting mechanism that transfer local descriptions to global description. (3) The authors built a benchmarking protocol that unified 8 existing datasets across various domains.

### Strengths
(1) The construction of refined local description similarity matrix is novel and intuitive. Especially the introduction of dustbin that avoids the hard comparison between different image instance. And the implementation of optimal transportation to refine the similarity matrix is intuitive and reasonable. Additionally, compared to the former strategy, this paper used the parametric method to empower the model with the ability to adjust the dustbin with self-supervision, making the model more interpretative and flexible. 
(2) They introduced 8 benchmarks that contains various kinds of domains, and they are the first work to conduct such an extensive evaluation of single-source domain generalization in instance-level retrieval. And the model performed well on those datasets, especially on out-of-domain retrieval scenarios.
(3) Most of the figures and narrations in this paper (except for Introduction) is good and logical, stating their motivation and insights.  
(4) The experiments and further analysis is abundant and showed the model’s effectiveness.

### Weaknesses
(1) Compared to their technical contribution, their narration in the introduction is less satisfying and cannot specify their motivations. 
a. The authors should detailed the reason why the focus on local descriptors is better than the global descriptors for the cross-domain image retrieval. Even though the author mentioned the intepretability and time cost, they can delve deeper into the explanation in representation space, making their statement and motivation less trivial and more solid.
b. It would be better for the authors to add a figure for this comparison, making their motivation more intuitive.
c. It would be better for the author to itemize and highlight their contribution in end of the introduction, making their contribution more clear for the readers.
(2) The authors should simplify the Section.2 (Related Work), instead explicitly detail their motivation, contribution and analysis on previous researches in the introduction.
(3) Even though model’s in-domain performances didin’t achieve state-of-the-art performances, it is acceptable.

### Questions
See weakness

### Soundness
2

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
4

### Summary
The paper presents ELViS, a re-ranking technique which is robust across domains. Elvis can be applied on top of common vision foundation models like DINOv2, DINOv3 and SigLIP2, and uses the transformer's local features by applying a lightweight post-processing to output a similarity score between two images, which is then used to re-rank a shortlist of retrieved candidates.

### Strengths
The paper outlines, demonstrates, and tackles a clear problem, which is that re-ranking methods trained on one domain underperform on others.
Elvis shows overall improvement when used on OOD data.
The paper is well written

### Weaknesses
1. Can Elvis work when images are of different resolutions? Given that f is an MLP it seems like Elvis would require images at a fixed resolution.

2. I don't fully understand Figure 3. A better caption would be helpful

3. Splitting results on ROxford and RParis would make results clearer and more comparable with other literature. Also which sets of ROxford and RParis are used (easy, medium, hard)?

4. The dataset table should report also the sizes of train/val/test sets for the two datasets used for training

5. A couple of images per dataset would help the reader to understand the domain gap between any two datasets

6. I believe the retrieval is performed with the same model of which the local features are used? I don't see this explicitly stated in the paper

7. Most importantly, comparing with image matching methods would be really helpful for the reader. Are the presented methods relevant, or should methods like SuperGlue be used for re-ranking in these domains?

### Questions
See the weaknesses stated above

### Soundness
4

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
5

### Summary
Instance-level image retrieval struggles with poor cross-domain generalization—existing models overfit to domain-specific data and fail on unseen domains, due to scarce cross-domain training data and reliance on representation space.
Specifically,
1.Refine the local descriptor similarity matrix using entropy-regularized optimal transport (OT) with descriptor-dependent dustbin gains to filter uninformative patches
2.Aggregate global similarity: select strongest local similarities per descriptor, weight them via a learnable function f, and sum—trained with modified BCE loss.

### Strengths
The paper shifts instance-level image retrieval from representation space to similarity space for stronger cross-domain robustness, refines optimal transport (OT) with descriptor-dependent dustbin gains , and creates the first unified benchmark for single-source cross-domain retrieval—uniting 8 datasets across 5 domains to standardize generalization evaluation.

### Weaknesses
Chowdhury (2022) also employed optimal transport (OT) to address the instance-level image retrieval task. Therefore, I have concerns about the technical innovation of this paper.
•	Optimal Transport (OT) Parameters: The paper uses 10 iterations of the Sinkhorn-Knopp algorithm and a regularization term \(\lambda = 0.1\), but it does not explain: i) Why 10 iterations (not 5 or 20)? ii) Why \(\lambda = 0.1\)? Cuturi (2013) shows \(\lambda\) directly impacts OT’s accuracy-efficiency tradeoff
•	Extreme Domains: The paper does not evaluate on domains with radical visual differences from natural images (e.g., infrared images, underwater photos, remote sensing imagery)—scenarios where cross-domain retrieval is highly valuable (e.g., satellite image matching for disaster response).
•	Small-Sample Training: The paper uses large training sets (GLDv2 has 762K images, SOP has 60.5K), but many real-world domains have only 100–1000 labeled samples. It is unknown if ELViS’s small parameter count (96K) translates to good small-sample performance.

 
Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26.
Chowdhury, P. N., Bhunia, A. K., Gajjala, V. R., Sain, A., Xiang, T., & Song, Y. Z. (2022). Partially does it: Towards scene-level fg-sbir with partial input. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2395-2405).

### Questions
For the questions , please refer to the above summary of weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
