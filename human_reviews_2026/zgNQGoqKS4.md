# Structuring Hidden Features via Clustering of Unit-Level Activation Patterns

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
We propose a self-supervised learning framework that organizes hidden feature representations across layers, thereby enhancing interpretability. The framework first discovers unit-level structures by comparing activation patterns across data samples. Building on these structures, we introduce a structure-aware regularization objective that (i) promotes feature reuse across layers via identity mappings and (ii) encourages the emergence of representative units that serve as anchors for related features. This regularization yields clearer and more structured feature pathways, enhancing the interpretability of the learned representations. Experiments demonstrate that our method induces structured feature pathways on synthetic data, improves interpretability on CIFAR-10 as measured by Grad-CAM++ metrics, and maintains competitive performance with slightly improved mean accuracy on both CIFAR-10 and ImageNet-1K.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Deep neural networks develop complex and unstructured internal representations, often creating redundant features that are difficult to interpret. This paper introduces a self-supervised regularization method to better organize hidden features, enabling their reuse across layers and increasing feature diversity within layers. This approach improves interpretability, makes better use of network resources, and may enhance generalization performance.

The method has two main components: First, it identifies redundant features through cross-layer clustering. Second, it implements a structure-aware regularization that encourages the reuse of one unit per cluster through residual connections while allowing other units to learn complementary features.

The authors tested their approach on three datasets: a synthetic task, CIFAR-10, and ImageNet, using variants of the ViT architecture. They developed new metrics to measure feature reuse, diversity, while utilizing previously proposed metrics for interpretability, and performance. Compared to standard training methods, their results showed better feature organization and interpretability.

### Strengths
- The paper presents its ideas clearly and comprehensively, with excellent organization and complete details. 
- The approach is novel, introducing efficient methods to reduce computational costs without compromising effectiveness. The use of group ranked transformation for clustering helps reduce sensitivity to magnitude differences. The evaluation framework and analysis metrics are well-designed. 
- The concept of enabling precise unit-level feature reuse across layers while utilizing residual layers is particularly novel.

### Weaknesses
- The paper's primary weakness lies in its limited experimental scope. While the presented results are promising, a broader evaluation across diverse datasets, model architectures, and network layers would better demonstrate the method's generalizability and practical impact. Enhanced visualizations of feature organization across multiple layers would also strengthen the paper's empirical validation.

- The introduction of multiple hyper-parameters without detailed ablation studies makes it challenging to determine optimal settings for future applications.

### Questions
- Is the structure loss calculated at the sample level?
- When clustering flattened representations across token positions and layers, multiple units from different token positions can end up in the same cluster, but only one anchor unit is selected per cluster. Would selecting multiple anchor units per cluster for each unique token position improve results?
- How does the method handle cases where the same unit position from different layers appears in different clusters but has the lowest index in each? This could create multiple loss computations for the same residual stream position. 
- How does this method perform with text inputs, where token counts vary and token positions can have significantly different representations? A discussion is required on the applicability across domains.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a self-supervised learning framework aimed at improving the interpretability of deep neural networks by structuring hidden feature representations. The method operates at the hidden-unit level, clustering activation patterns across data samples and imposing a structure-aware regularization that encourages cross-layer feature reuse and the emergence of representative anchor units.

### Strengths
1. Structured feature representation is an interesting topic. The paper does introduce a hidden-unit-level approach to organize features though it may remain complex.
2. The combination of clustering hidden units and enforcing structure via a regularization objective is conceptually interesting and aligns with efforts to improve interpretability through learned representations.

### Weaknesses
1. The evaluation relies heavily on Grad-CAM++ metrics. Gradient-based attribution methods are known to have limitations (especially in deep networks) and can produce misleading explanations. This raises concerns about whether the reported scores in interpretability are meaningful.

2. The paper does not adequately position itself relative to prior explanation methods. Many traditional explanation methods are missing, such as TCAV (Kim et al.) and DINO. While these methods are not specifically relevant to "structure", they are helpful for understanding features.

3. The paper does not convincingly demonstrate that structured representations are helpful for downstream tasks. If not, the advantage of structured features over existing feature characterization methods can be the key. But this part is missing in the current scope.

4. I am curious about the impact of the structure-aware regularization on feature dynamics and learning efficiency.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose a novel way to align the feature across many different layers of a neural network. The author propose to collect a latent embedding buffer during training, which contains the embedding across different sample, position, and layers, then cluster these embedding to create "feature anchor" points. Then the author define an auxiliary loss to constraint the latent of the neural network to match these feature anchor accordingly. The author claims the this auxiliary loss makes the model more explainable in two way: 1. features across different layers are more aligned. 2. model trained auxiliary loss when applied with grad-cam produces better unsupervised segmentation map.

### Strengths
The author present a novel way to make neural network more interpretable. 
Method presented by the author is not post-hoc, unlike many other interpretability works.
The author shows their method aligned with class-level segmentation map better, when applied grad-CAM.

### Weaknesses
The author only provide the baseline against VIT trained with standard classification loss, but did not compare their method with other method that improves model's interpretability.
The model does 
The author does not provide standard evaluation (classification accuracy) between standard VIT training and model trained with their auxilary loss. 
There are many moving part of the design, ranking as preprocessing, and group rank, the author did not provide enough ablation study to show these design are necessary.

### Questions
Can author read my summary to see if my understanding is correct? If not, please tell me and also explain to me how the model actually works. 
I would guess the training with auxiliary will improve model's interpretability but hurt the model's performance, if so, how much?
The ranking part is confusing to me. Why do the author use "rank" to preprocess the latent embedding, then something like normalization?

### Soundness
2

### Presentation
2

### Contribution
2
