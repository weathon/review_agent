# Samples Are Not Equal: A Sample Selection Approach for Deep Clustering

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4

## Abstract
Deep clustering has recently achieved remarkable progress across various domains. However, existing clustering methods typically treat all samples equally, neglecting the inherent differences in their feature patterns and learning states. Such redundant learning often drives models to overemphasize simple feature patterns in high-density regions, weakening their ability to capture complex yet diverse ones in low-density regions. To address this issue, we propose a novel plug-in designed to mitigate overfitting to simple and redundant feature patterns while encouraging the learning of more complex yet diverse ones. Specifically, we introduce a density-aware clustering head initialization strategy that adaptively adjusts each sample's contribution to cluster prototypes according to its local density in the feature space. This strategy mitigates the bias towards high-density regions and encourages a more comprehensive attention on medium- and low-density ones. Furthermore, we design a dynamic sample selection strategy that evaluates the learning state of samples based on the feature consistency and pseudo-label stability. By removing sufficiently learned samples and prioritizing unstable ones, this strategy adaptively reallocates training resources, enabling the model to consistently focus on samples that remain under-learned throughout training. Our method can be integrated as a plug-in into a wide range of deep clustering architectures. Extensive experiments on multiple benchmark datasets demonstrate that our method improves clustering accuracy by up to $\textbf{6.1}$\% and enhances training efficiency by up to $\textbf{1.3$\times$}$. Code is available at [https://github.com/notoaudrey/Samples-Are-Not-Equal](https://github.com/notoaudrey/Samples-Are-Not-Equal).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper identifies a key problem in deep clustering: existing methods treat all samples equally, causing them to overfit to simple, redundant feature patterns found in high-density regions of the feature space. To solve this, the authors propose a two-part plug-in designed to be integrated into existing deep clustering models:
* Density-Aware Clustering Head Initialization (DACHI): This strategy addresses the initialization bias. Instead of a standard prototype (which is just the average feature of all samples in a cluster and is thus dominated by high-density samples ), it computes a density-weighted prototype.
* Dynamic Sample Selection (DSS): This strategy adaptively manages training resources. It identifies and temporarily removes "sufficiently learned" samples from training batches. This allows the model to reallocate its capacity toward more "unstable" or under-learned samples.

### Strengths
1. The dynamic sample selection (DSS) strategy provides a novel and sensible method for curriculum learning in a fully unsupervised setting.
2. The proposed method successfully improves both clustering accuracy and training efficiency.
3. The method is evaluated as a plug-in for four different deep clustering baselines, demonstrating its general applicability. The ablation studies validate the contribution of both DACHI and DSS.

### Weaknesses
1. The paper claims a training speedup of 1.3, but this only accounts for the reduced batch size during the model update step. It fails to discuss the significant overhead of the selection mechanism itself.
2. The method introduces new and sensitive hyperparameters. The DSS strategy's pruning threshold, $\epsilon$, is particularly problematic. As shown in Table 5, performance is highly dependent on this value: moderate pruning ($\epsilon=0.1-0.3$) helps, but aggressive pruning ($\epsilon=0.5$) degrades performance. The paper provides no clear heuristic for setting this critical value, which was manually set to 1e-1 or 1e-2 depending on the dataset.
3. The 6.1% gain on CC is large, the improvement on the most recent and powerful baseline, CDC, is a more modest 1.1% average.
4. The paper repeatedly uses the term "overfitting"  to describe the model's bias toward high-density samples. This is a slightly imprecise use of the term.

### Questions
1. Related to weakness 1, could you provide a detailed analysis of the total wall-clock time per epoch, including the significant computational and memory overhead introduced by DSS? Specifically, what is the cost of calculating and storing prediction consistency histories for all $N$ samples at every epoch? The DACHI requires a k-nearest neighbor search within each initial cluster. How does this initialization step scale with very large datasets and high-dimensional features?
2. Related to weakness 2. Is there a more adaptive or heuristic-based method to set this threshold?
3. Have you investigated the interplay between your two modules? e.g. does the improved DACHI initialization lead to a larger or smaller set of samples being pruned by DSS later in training?

### Soundness
2

### Presentation
4

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
To address the problem that existing deep clustering methods treat all samples equally, resulting in excessive focus on simple redundant features in high-density areas and neglect of complex and diverse features in low-density areas, the paper proposes a plug-in module, which includes a density-aware clustering head initialization strategy (adaptively adjusting the contribution to the clustering prototype according to the local density of the sample to reduce bias in high-density areas) and a dynamic sample selection strategy (evaluating the sample learning status based on feature consistency and pseudo-label stability, and prioritizing resources to samples that have not been fully learned); this module can be seamlessly integrated into a variety of deep clustering architectures. Experiments on benchmark datasets show that clustering accuracy improves, and the performance gain in medium- and low-density areas (complex features) is more significant.

### Strengths
1. The method proposed in the article is implemented as a plug-in module. It can be seamlessly integrated into a variety of mainstream deep clustering architectures without requiring significant modifications to the original model's core structure. The integration process only requires replacing the initialization step (density-aware clustering head initialization) and the embedded training loop (dynamic sample selection), thereby significantly reducing the application cost in existing systems and providing strong adaptability.
2. Based on the local density of samples (k-nearest neighbor distance), the contribution weight of samples to the cluster prototype is adaptively adjusted to reduce the dominance of high-density samples on the prototype, retain the clustering structure of medium and low-density areas (complex and diverse features), and avoid bias in the initialization stage.

### Weaknesses
1. Density-aware clustering head initialization relies on features extracted by a pre-trained encoder. The document uses MoCo-v2 to pre-train ResNet-34 but does not discuss the impact of the pre-trained encoder's quality on the method. If the feature representations extracted by the pre-trained encoder are of poor quality (e.g., low discrimination for complex samples), subsequent density calculations and K-Means initial clustering based on these features will be biased, thereby affecting the accuracy of cluster prototype initialization.
2. In the early stages of training, the model has not yet fully learned, and the accuracy of pseudo-labels is low. At this time, screening samples based on pseudo-label stability may mistakenly judge "complex samples that really need to be learned" as "unstable and need to be retained", resulting in unreasonable allocation of early training resources.
3. The experiment only verifies the effectiveness of the method on the image dataset, and does not involve other modal data such as text, speech, and time series. The adaptability of the method to non-image modalities has not been verified, and the applicable scenarios are limited.

### Questions
1. One key technique is initializing the clustering head, but this is highly dependent on the model's pre-training. How to deal with non-pre-trained models?
2. While this method improves overall training efficiency, the dynamic sample selection process requires additional computation and tracking: generating weak/strong augmentation views for each sample, calculating prediction consistency (cosine similarity), and tracking changes in second-order differences and pseudo-labels over nearly three epochs. How can this additional overhead be quantified?
3. There are fluctuations in consistency in the early stages of training. How to judge the credibility of the consistency constraints?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a sample selection approach for deep clustering that addresses the problem that most clustering models treat all samples equally, even though samples in high-density regions are often redundant and simple, while those in low-density regions are complex and informative.
By relying more on samples in sparse regions (based on kNN distances), the tested models learn more effectively. The authors propose two main components:

1. Density-Aware Clustering Head Initialization (DACHI) – adjusts each sample’s weight when computing cluster prototypes, so that clusters are not dominated by redundant high-density samples.

2. Dynamic Sample Selection (DSS) – uses prediction consistency and pseudo-label stability to identify well-learned samples and temporarily remove them from training, letting the model focus on under-learned samples. 

The method is plug-and-play and can be integrated into existing deep clustering algorithms.

### Strengths
S1) Sampling strategies have not been researched enough on in the deep clustering area. The author's method can be combined with a lot of different deep clustering methods, potentially advancing the field broadly. 

S2) The methods are quite simple and intuitive. 

S3) By discarding some samples in each epoch the methods are even accelerated.

### Weaknesses
W1) Experimental evaluation could be in more depth: how does the improvement of results depend on 

a) the model complexity

b) the number / sizes of datasets

c) the number of concepts per class/cluster. 

I'm missing some synthetic experiments here. Also, it would be interesting what happens for datasets with consistent densities throughout the datasets, e.g. COIL 20. 
Furthermore, I would be interested in how well the methods works on, e.g., tabular data, as the paper focuses on image data.

W2) Sampling strategies are a major object of research in Active Learning. Setting the method into this context would improve the paper. 

W3) How much runtime do the computations for the sampling need? Is the runtime acceleration dependent on properties of the data that can be predicted? 

W4) Using k-Means as initlal clustering should be discussed more. It is not clear to me how the value of k for the initial k-Means clustering was chosen. What happens if the data does not follow typical assumptions fitting k-Means, e.g., for video data?

Especially the first three pages have quite some redundant text that could be shortened in order to tackle the above weaknesses or answer the questions below. 

Minor stuff: 
- Table 3 appears way before it is referred to in the text which hinders the reading flow.
- Type in line 90

### Questions
Q1) How does your method perform on data with consistent density, e.g. COIL20?

Q2) How does your method perform on tabular data?

Q3)  How is your sampling strategy related to Active Learning strategies? What could we learn from there, what are similarities?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes improvements to the self-labeling stage of deep clustering. First, it introduces a density-aware clustering head initialization that downweights redundant high-density samples and upweights rare/low-density samples when forming prototypes. Second, it proposes a dynamic sample selection strategy that prunes stable samples based on pseudo-label stability and consistency between weakly and strongly augmented views, allowing training to focus on unstable or underlearned samples. Overall, this leads to improved clustering results across several standard image clustering benchmarks and results in better training efficiency, since fewer samples are actively optimized in later epochs.

### Strengths
- The paper targets an important stage used in state of the art deep clustering algorithms: self-labeling. The motivation is clearly laid out and depicts a practical issue: overfitting to dense/easy samples while underfitting rare samples.
- The method improves both clustering accuracy and efficiency across several standard datasets and multiple baselines, suggesting it is broadly useful. The authors also provide a hyperparameter study on (
α,
k, 
ϵ) and show that the approach is generally robust to 
α and k.

### Weaknesses
- Training is always stopped after a fixed 100 epochs. Unsupervised stopping criteria remain an open problem in the self-labeling stage. The proposed framework (especially with dynamic pruning) could, in principle, offer a stopping signal, but this is not analyzed.
- The evaluation does not clearly report model selection details or variance across seeds. Because self-labeling inherently involves noise and instability (due to training on pseudo-labels), averaged results and standard deviations are important to properly interpret the reported gains.

### Questions
- How are models selected and tuned? What are the standard deviations across runs, e.g. for 5 to 10 seeds. How stable is the method overall?
- You run for 100 epochs, but given that you explicitly track stability and prune stable samples, could those same signals (e.g., the fraction of samples no longer changing) be used as a stopping heuristic?
- What happens if you continue training far beyond 100 epochs (e.g., 500+)? Do you eventually prune almost all samples, or does the model start overfitting the small subset of rare or ambiguous samples that never get pruned?
- At the moment a sample is pruned as stable, how often is it actually assigned the correct ground-truth class? Showing that would address the concern that you might be confidently pruning wrong assignments and locking in errors. That could, however, have the benefit of reducing error propagation by no longer learning on those errors.

### Soundness
2

### Presentation
3

### Contribution
3
