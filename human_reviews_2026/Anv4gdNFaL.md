# One-Shot Exemplars for Class Grounding in Self-Supervised Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 8, 4, 4, 6

## Abstract
Self-Supervised Learning (SSL) has recently achieved remarkable progress by leveraging large-scale unlabeled data. However, SSL pretrains models without relying on human annotation, so it usually does not specify the class space. This inevitably weakens the effectiveness of the learned representation in most downstream tasks that have the intrinsic class structure. In this work, we introduce the new easy setting of One-Shot Exemplar Self-Supervised Learning (OSESSL), requiring only one instance annotation for each class. By introducing this extremely sparse supervision, OSESSL provides the minimum class information to guide the exploration of unlabeled data, achieving significant performance boosts with neglectable annotation cost (i.e., a complexity of $\mathcal{O}(1)$ w.r.t. the sample size). In this OSESSL setting, we propose a simple yet effective framework that leverages the single-labeled exemplar to build the class-specific prototype for learning reliable representations from the huge unlabeled data. To this end, we also build a novel consistency regularization, which extends the sparse exemplar supervision into the decision boundaries, thus improving the robustness of the learned representation. Extensive experiments on real-world datasets clearly validate the reliability of this simple and practical setting. The proposed approach successfully outperforms the state-of-the-art methods, achieving gains of approximately 3\% and 6\% $k$-NN accuracy on CIFAR-100 and ImageNet-100, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on one-shot exemplar self-supervised learning, a new problem to self-supervised learning that uses one labeled example per class. To address this issue, the paper incorporates a prototype learning algorithm and an interpolation consistency module. Extensive experimental results validate the effectiveness of the proposed method.

### Strengths
The paper is well-written.

The problem of self-supervised learning with one-shot exemplars is interesting and novel to the literature. Utilizing the available supervision information for self-supervised learning is a critical research problem. Therefore, it will create new opportunities in the field.

The proposed overall algorithm is simple yet effective. The different components of the algorithm are effective and reasonable. The design of the algorithms for exemplar-guided prototype construction, exemplar-guided prototype learning, and exemplar-guided interpolation consistency is coherent and interesting.

Extensive experiments validate the effectiveness of the proposed method.

### Weaknesses
Is the proposed approach sensitive to the choice of the selected example for each class? Since one exemplar is very few, will it cause big variations on the algorithm performance?

Since there are different losses for the proposed methods, it would be beneficial to conduct ablation studies to confirm the effectiveness of each component, although the experimental results are already very comprehensive.

### Questions
I have no major questions on this paper.

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
This paper introduces a novel and practical learning setting called One-Shot Exemplar Self-Supervised Learning (OSESSL). The core motivation is derived from that self-supervised learning (SSL) fails to specify the class space, which inevitably weakens the effectiveness of the learned representations for downstream tasks with intrinsic class structures. Instead of relying on self-generated signals, the authors propose using a single annotated instance per class to provide minimal yet crucial supervision, guiding the model toward meaningful semantic structures while retaining the scalability of SSL. The central mechanism to tackle the new setting is the "Exemplar-Guided Prototype Construction," where each single-labeled exemplar is used to build a robust class prototype. This is achieved not by using the exemplar in isolation, but by enriching it with its most discriminative neighbors from the vast pool of unlabeled data. These prototypes then provide semantic guidance for a clustering-based SSL objective. The framework is further enhanced with a prototype dispersion loss to prevent collapse and an "Exemplar-Guided Interpolation Consistency" loss, which regularizes the model on mixed samples to improve decision boundary robustness. Extensive experimental results on standard benchmarks demonstrates that the proposed method significantly outperforms a wide array of state-of-the-art SSL methods across multiple evaluation protocols and demonstrates strong performance on both CNN and Vision Transformer backbones.

### Strengths
1. The introduction of the OSESSL setting is a significant contribution in itself. The authors provide a very lucid argument for its necessity. Unlike traditional SSL that lacks explicit semantic grounding, OSESSL leverages a single labeled exemplar per class to guide representation learning in a scalable manner. The OSESSL setting offers semantic grounding with a truly negligible annotation cost (a complexity of $\mathcal{O}(1)$ with respect to dataset size, as the authors rightly point out). This is an extremely practical scenario for many real-world applications where identifying and labeling one canonical example of each class is far more feasible than labeling thousands of instances.

2. The proposed method is simple yet effective, combining exemplar-guided prototype alignment, prototype dispersion, and interpolation consistency to jointly promote discriminative and stable representation learning. The design is also supported by theoretical reasoning. The method is evaluated across diverse datasets, and significant performance makes the empirical evidence convincing. Ablation studies further confirm the complementary effects of the proposed components and the stability of key hyperparameters.

3. The paper is clearly written and well-structured, making it easy to follow. The authors effectively present the motivation, identify the limitations of prior work, and justify their proposed approach. Figures and tables are well-designed to illustrate key results, and the appendices provide useful supplementary details that enhance the overall clarity and completeness of the work.

### Weaknesses
1. The neighbor selection mechanism is key to the method's success, as it relies on a meaningful feature space to identify semantically similar instances. However, in the early stages of training, the encoder is not yet powerful, and the feature space is likely to be poorly structured. I concern that the initial neighbor selections could be noisy or incorrect, reinforcing incorrect associations early on. 

2. Lack of ablation studies of exemplar-guided prototype construction. It would be helpful to analyze how the choice of exemplars influences performance and to evaluate the sensitivity or benefit of the discriminative-neighbor weighting parameter. Such studies would clarify the robustness and generality of the proposed mechanism.

### Questions
Have the authors evaluated the robustness of OSESSL under exemplar noise, or considered mechanisms to mitigate its effects? If such analysis has not been conducted, what challenges or design choices would be most critical for extending the method to noisy one-shot supervision?

### Soundness
4

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
3

### Summary
The paper addressed self-supervised learning, but with a new paradigm called One-Shot Exemplar Self-Supervised Learning (OSESSL). In OSESSL, the challenge is to use the sparse supervision to guide representation learning. First, the authors build class-specific prototypes and use the prototypes to guide representation learning. Moreover, the authors proposed an interpolation consistency loss to provide the regularization and improve the decision boundaries. The authors have conducted extensive experiments to validate the effectiveness of the  proposed method in various settings, such as SSL, semi-supervised learning and transfer learning.

### Strengths
1. The paper is well motivated

Considering representation learning in a semantically grounded manner is likely to encourage the model to learn more informative representations.

2. The presentation is clear

3. The paper is easy to follow

### Weaknesses
1. The proposed OSESSL setting seems unnecessary

2. The proposed exemplar-guided alignment seems similar to PAWS

3. The ablation study is missing

4. The semi-supervised learning setting is strange

Please see the Question section below for details.

### Questions
1. The proposed OSESSL setting seems unnecessary

While the authors claim this OSESSL as one of their contributions, I am not convinced that OSESSL brings new insights to the community. Its entire definition is covered by semi-supervised learning; the one-exemplar assumption is still under the few-shot semi-supervised learning paradigm. Having one or 3 samples does not change the nature of sparse supervision. 

2. The proposed exemplar-guided alignment seems similar to PAWS

I would love to see a direct comparison between the proposed exemplar-guided alignment with PAWS. The major difference is that PAWS uses labeled data as anchors, and here the authors use the reconstructed exemplars. Under such sparse supervision, I am not sure how much more information the reconstructed exemplar gathers, as it has to stay close to the labeled data to avoid the loss of semantic meaning. Therefore, I find these two methods very similar.

3. The ablation study is missing

While the ablation study is promised in line 312 in the experiment section, there is no ablation study provided in this section, making it hard to validate the effectiveness of the proposed method.

4. The semi-supervised learning setting is strange

I find the semi-supervised learning setting in line 425 very strange. It is more like a transfer learning rather than semi-supervised learning, if the model is first trained in SSL on ImageNet-1k, and then fine-tuned on labeled data. In general, semi-supervised learning provides both labeled and unlabeled data at the same time. I hope the authors could elaborate on the reason for setting up the experiments in this way.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper introduces a new setting called One-Shot Exemplar Self-Supervised Learning (OSESSL), where self-supervised learning is augmented with just one labeled example per class to ground the representations in the true class space. The authors argue that this minimal supervision—essentially O(1) annotation cost relative to dataset size—can significantly boost downstream performance without losing SSL's scalability. They propose a framework that builds class-specific prototypes from the exemplar plus selected unlabeled neighbors, enforces alignment across views, adds dispersion to prevent collapse, and includes an interpolation consistency regularization for robustness near decision boundaries. Experiments on CIFAR-10/100, ImageNet-100/1K show consistent gains over SSL baselines like ReSA, DINO, etc., with notable improvements in k-NN (e.g., +6% on ImageNet-100) and linear classification. Transfer to semi-supervised, detection, and fine-grained tasks also looks strong

### Strengths
1. The OSESSL idea is clever and addresses a real gap in SSL—lack of class grounding—while keeping annotations negligible. It's a nice middle ground between pure SSL and semi-supervised learning, especially since class count grows slower than data volume in big datasets like LAION. This could inspire more work on "minimal supervision" hybrids.
2.The prototype construction (using discriminative scores for neighbors) and interpolation consistency are straightforward extensions to clustering-based SSL. The math derivations (e.g., gradient analysis of alignment loss) provide good intuition on why it works. No overly complex bells and whistles, which makes it reproducible.
3

### Weaknesses
1.The method relies on one high-quality exemplar per class, but real-world data often has noise or ambiguity. The conclusion mentions extending to noisy scenarios, but no experiments here—would be good to test robustness with mislabeled or atypical exemplars.
2. While gains over pure SSL are clear, the semi-sup baselines (PAWS, Suave) use 1% labels (~12k on ImageNet), which is 12x more than yours (1k classes = 1k labels). Claiming superiority feels a bit stretched without ablating how performance scales with more exemplars. Also, some citations are to 2025 papers (ReSA, SOP)—fine for anon review, but ensure they're public.
3. Mostly image classification-focused; more on non-vision (e.g., if applicable) or diverse domains (medical, satellite) would strengthen generality. Detection results are good, but only on COCO—VOC or others? Also, no analysis on failure cases, like classes with high intra-variance.
4. Appendix shows stability, but temperatures τs/τt fixed at 0.1/0.04—why these? And α=0.75 in discriminative score; a sweep there might reveal more.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces One-Shot Exemplar Self-Supervised Learning (OSESSL), a novel setting that leverages only one annotated instance per class to provide semantic grounding while maintaining the scalability of self-supervised learning. The authors propose a framework that constructs exemplar-guided prototypes augmented with discriminative neighbors from unlabeled data, and introduces exemplar-guided interpolation consistency to smooth decision boundaries. Extensive experiments on CIFAR and ImageNet benchmarks demonstrate state-of-the-art performance, with significant improvements in k-NN accuracy (e.g., +3% on CIFAR-100, +6% on ImageNet-100) over strong baselines.

### Strengths
**Originality**: The OSESSL setting is novel and fills an important gap between fully unsupervised SSL and semi-supervised learning. The exemplar-guided prototype construction and interpolation consistency mechanism are creative and well-motivated.

**Quality**: Experimental evaluation is extensive and convincing, covering linear evaluation, k-NN classification, semi-supervised learning, and transfer learning. The method consistently outperforms strong baselines across all settings.

**Clarity**: The paper is exceptionally clear in both writing and technical exposition. The gradient analysis provides valuable theoretical insight.

**Significance**: The work addresses the important problem of incorporating minimal supervision to guide SSL toward semantically meaningful representations, with practical implications for real-world applications where annotations are scarce.

### Weaknesses
1.	The method assumes clean exemplars are available, but real-world scenarios often involve noisy annotations. The paper briefly mentions this limitation but provides no experiments on noisy exemplars.

2.	The paper could benefit from more analysis on how the method performs with different qualities of exemplars (e.g., easy vs. hard examples).

### Questions
1.	How sensitive is the method to the quality of the single exemplar per class? Have you experimented with different strategies for selecting the exemplar (e.g., cluster centers vs. random samples)?

2.	For the neighbor selection in prototype construction, did you consider using more sophisticated metrics beyond cosine similarity, such as incorporating density estimation?

### Soundness
4

### Presentation
4

### Contribution
4
