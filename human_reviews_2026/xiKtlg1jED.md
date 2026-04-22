# PartCo: Part-Level Correspondence Priors Enhance Category Discovery

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Generalized Category Discovery (GCD) aims to identify both known and novel categories within unlabeled data by leveraging a set of labeled examples from known categories. Existing GCD methods primarily depend on semantic labels and global image representations, often overlooking the detailed part-level cues that are crucial for distinguishing closely related categories. In this paper, we introduce PartCo, short for Part-Level Correspondence Prior, a novel framework that enhances category discovery by incorporating part-level visual feature correspondences. By leveraging part-level relationships, PartCo captures finer-grained semantic structures, enabling a more nuanced understanding of category relationships. Importantly, PartCo seamlessly integrates with existing GCD methods without requiring significant modifications. Our extensive experiments on multiple benchmark datasets demonstrate that PartCo significantly improves the performance of current GCD approaches, achieving state-of-the-art results by bridging the gap between semantic labels and part-level visual compositions, thereby setting new benchmarks for GCD. Code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the task of generalized category discovery - given a labelled training set up some categories learn a model to automatically cluster an unlabelled test set which contains both seen and unseen categories. Particularly this paper focuses on part-level correspondence aiming to capture relationships between categories in terms of shared components/parts to aid generalized category discovery. This is achieved by performing PCA on DINO features to obtain fine-grained features and then clustering these features into parts. The resulting part-level features can then be integrated into existing GCD methods.

### Strengths
- The paper is generally well-presented, with visuals that effectively aid understanding of the proposed approach.
***

- The idea of leveraging object parts for GCD is interesting and particularly relevant for fine-grained scenarios where category discovery is more realistic and challenging.
***

- The method is well-motivated, making good use of the richer feature representations available in DINO beyond the CLS token.
***

- The evaluation includes both generic and fine-grained datasets, demonstrating reasonable generality of the approach.
***

- The proposed framework is flexible and can be combined with different existing GCD methods, increasing its potential applicability.
***

- The use of DINOv3 features is appreciated as it helps keep the experiments aligned with the current state of the field.
***

- The comparison between implicit and explicit part-level learning in Table 4 is a nice inclusion and provides some insight into the benefits of the explicit formulation.

### Weaknesses
- An existing work in GCD also uses part-level correspondence [A]. The difference to this work in terms of approach and performance needs to be made clear
***

- It remains unclear whether explicit part modelling is necessary. Could similar benefits be achieved simply by removing background features? Including a “single-part” or “foreground-only” variant in the ablation (e.g., Figure 7) would help clarify this.
***

- It would be useful to better understand how the method performs on more challenging or long-tailed datasets such as Herbarium. While Herbarium and Oxford Pets are tested in the supplementary material, the comparison is only made against SimGCD. This feels incomplete, especially since other works such as SelEX and Flipped Classroom report results on these datasets and include more extensive baselines.
***

- SelEX is no longer the current state of the art on fine-grained datasets as this paper claims since there are newer works in GCD [A,B,C,D] with [A,B,C] all demonstrating to outperform Selex. Therefore framing comparisons primarily against selex  misrepresents the true standing of the proposed method. These newer works should be added in the comparison.
***

- While the proposed approach performs well on the fine-grained datasets, outperforming all methods compared to, the results are more mixed on the generic datasets where on some metric FlipClass outperforms.
- I don't think this is a major issue as the performance is still strong, however it would have been interesting to see if better results could be achieved by combining the proposed approach and FlipClass
***

[A] Dai et al. Adaptive Part Learning for Fine-Grained Generalized Category Discovery: A Plug-and-Play Enhancement. CVPR 2025.
***
[B] Liu et al. Hyperbolic Category Discovery. CVPR 2025.
***
[C] Tang et al. Dissecting Generalized Category Discovery: Multiplex Consensus under Self-Deconstruction. ICCV 2025.
***
[D] Xu et al. A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention. ICCV 2025.

### Questions
- Comparison to [A]: Could the methodological and numerical difference to [A] be clarified?
***

- Necessity of parts: Could the reported improvements be explained by background suppression rather than true part reasoning? Have you tested a variant using only a single (foreground) part to isolate this effect?
***

- More challenging datasets: Why are other baselines (e.g., Flipped Classroom, SelEX) omitted from the comparisons on Herbarium and Oxford Pets, given that results for these datasets are already publicly available?
***

- Further comparisons: Why are more recent methods e.g. [A,B,C,D] omitted from the state-of-the-art comparison?
***

- Understanding features: DINOv2 consistently outperforms DINOv3 in your experiments. Can you provide an explanation or hypothesis for this surprising trend?

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
This paper introduces PartCo, a framework that enhances Generalized Category Discovery (GCD) by incorporating part-level correspondence priors extracted from ViT patch tokens.

### Strengths
- Clear motivation for why part-level information helps GCD, especially for fine-grained categories with shared global features but different local compositions
- Simple and effective approach that leverages existing foundation models without requiring additional annotations or complex architectural changes
- Genuinely plug-and-play—demonstrated compatibility with multiple GCD baselines (SimGCD, SelEx, SPTNet, etc.) with consistent improvements

### Weaknesses
- The novelty is somewhat limited. Using patch tokens from foundation models to extract fine-grained features is increasingly common. The specific formulation of first and second-order correspondences is the main technical contribution, but this feels incremental.

- The reliance on DINO features for correspondence labels means PartCo inherits DINO's biases and limitations. If DINO fails to capture relevant part correspondences (e.g., due to severe occlusion or unusual viewpoints), PartCo will struggle. This dependency is acknowledged in limitations but not thoroughly analyzed.

- The choice of second-order correspondence is not well-justified theoretically. Why stop at second-order? Have you tried third-order or higher? Is there a principle for selecting the order, or is it empirical?

- Limited analysis of failure cases or when part-level features don't help. The paper shows PartCo improves performance on average, but are there categories or scenarios where it hurts? For example, what about categories distinguished by global shape rather than local parts?

- Computational cost analysis is incomplete. How much does PartCo add to training time and memory? The paper mentions it's "lightweight" but doesn't provide concrete numbers or scalability analysis.

- The part correspondence labels are treated as pseudo-ground-truth, but their quality is not validated. How accurate are these correspondences? Some analysis comparing against manually annotated part correspondences (available in CUB) would strengthen the paper.

- The paper claims part-level features help with novel category discovery, but the analysis mostly focuses on fine-grained datasets where this is intuitive. The gains on ImageNet and CIFAR are smaller, suggesting limited benefit for coarse-grained categories. This limitation deserves more discussion.

### Questions
1. Have you validated the quality of correspondence labels against ground-truth part annotations in CUB? What's the alignment accuracy?

2. Why second-order specifically? Can you provide ablations on first-order only, second-order only, and third-order to justify this choice?

3. How does PartCo perform when the foundation model (DINO) fails to capture good part correspondences? Can you show failure cases?

4. What is the computational overhead? Please provide training time, memory usage, and FLOPs comparisons.

5. For coarse-grained categories (ImageNet, CIFAR), the gains are modest. Can you analyze which types of categories benefit from part-level features and which don't?

6. How sensitive is PartCo to the hyperparameter λp (correspondence loss weight)? The ablation shows some variation but doesn't analyze why certain values work better for different datasets.

7. The attention maps show PartCo focuses on parts, but how does this translate to better novel category discovery specifically? Can you show examples where part-level features helped correctly cluster novel categories that global features confused?

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
This paper presents a new learning framework for Generalized Category Discovery (GCD) by integrating explicit part-level visual feature correspondences. In contrast to traditional GCD methods, the proposed approach enhances category understanding and discovery by leveraging compositional object features. Experimental results on benchmark datasets demonstrate that the method achieves improved performance in most cases.

### Strengths
1. The motivation of the paper is clear, and the content is easy to understand.
2. The idea of incorporating explicit part-level visual feature correspondences is interesting.
3. The experimental results indicate that the proposed method outperforms existing approaches across different benchmark datasets in most scenarios.

### Weaknesses
1. The technical novelty appears somewhat limited, particularly as the work is built upon the existing SimGCD framework.
2. The generalization capability of the method has not been fully validated, as only two baseline methods are tested. Evaluation with more benchmark methods is recommended.
3. Most existing GCD methods use pre-trained models different from DINOv2 and DINOv3, making it difficult to fairly compare the results with those reported in original studies. It is suggested to maintain consistency in the pre-trained model for a more equitable comparison.

### Questions
See weaknesses

### Soundness
2

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
4

### Summary
This paper proposes a novel framework, PartCo that introduces a part-level correspondence prior to improve category discovery by leveraging fine-grained visual relationships beyond global image features. It integrates seamlessly with existing GCD methods and enhances both supervised and unsupervised learning through a novel part-level contrastive loss. Experiments across multiple benchmarks show significant accuracy gains and state-of-the-art performance, validating its effectiveness and adaptability.

### Strengths
(1) The proposed method, PartCo,  effectively integrates part-level visual priors into existing GCD frameworks, enhancing fine-grained category discrimination without altering model architecture.

(2) It consistently achieves state-of-the-art performance across multiple benchmark datasets, demonstrating strong robustness and generalizability.

### Weaknesses
Related works are not discussed properly. More recent works are needed to be cited.

(1) Adaptive Part Learning for Fine-Grained Generalized Category Discovery: A Plug-and-Play Enhancement, CVPR 2025

(2) Hyperbolic Category Discovery, CVPR 2025

(3) Cdad-net: Bridging domain gaps in generalized category discovery, CVPR 2024

(4) MOS: Modeling Object-Scene Associations in Generalized Category Discovery, CVPR 2025

### Questions
(1) Is the selection of between 1st- and 2nd-order labels dataset-dependent? How could you decide the best in general?

(2) What are the effects of the hyperparameters i.e. PCA threshold $\tau$ and the balancing factor $\lambda_{b}$, as these are empirically chosen?

### Soundness
3

### Presentation
3

### Contribution
3
