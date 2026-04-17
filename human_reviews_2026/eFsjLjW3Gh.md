# A Geometric Analysis of Logit Embeddings for Out-of-Distribution Detection

- Decision: Reject
- Scores: 2, 6, 0, 2

## Abstract
Out-of-distribution (OOD) data pose a significant challenge to deep learning (DL) classifiers, prompting extensive research into their effective detection methods.
Current state-of-the-art OOD detection methods usually employ a scoring technique designed to assign lower scores to OOD samples compared to in-distribution (ID) ones. 
Nevertheless, these approaches lack foresight into the configuration of OOD and ID data within the latent space. 
Instead, they make an implicit assumption about their inherent separation or force a separation post-training by utilizing selected OOD data.
As a result, most OOD detection methods result in complicated and hard-to-validate scoring techniques.
This study conducts a thorough analysis of the logit embedding landscape, revealing that the ID and OOD data exhibit a distinct spatial configuration.
Specifically, we empirically observe that the OOD data are drawn to the center of the logit space.
In contrast, ID data are repelled from the center, dispersing outward into distinct, class-wise clusters aligned along the orthogonal axes that span the logit space.
This study highlights the critical role of the DL vision-based classifier in differentiating between ID and OOD logits.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposes an analysis into the logit embeddings between in-distribution (ID) data and out-of-distribution (OOD) data. The key insight is that OOD data is drawn to the center of the logit space, and ID data are repelled from the center, dispersing outward into distinct and class-wise clusters. A lot of KDE plots are provided in this paper, and experiments cover a wide range of aspects.

### Strengths
1.	The proposed insight about the spatial distribution between ID and OOD data in the logit space is somehow beneficial.

2.	There are extensive intuitive KDE plots.

### Weaknesses
1.	Insufficient workload. The contribution of this work is merely the aforementioned spatial separability between ID and OOD data in the logit space. The workload is clearly insufficient. How can the separability help OOD detection by inspiring new effective detection scores? The authors are suggested to enrich this paper by proposing a relevant detection method with associate detection results instead of just showcasing a phenomenon.

2.	All the references are not in a correct form where the outermost parentheses are missing. Have the authors read the paper themselves? Do the authors feel uncomfortable when seeing so many in-text references without parentheses?

3.	From line 168 to line 174, the ID logits distribution after training is explained, which is an important conclusion in this work. However, there are too few contents and the key Proposition 2 is even left to the appendix. In contrast, there are so many contents from Line 138 to Line 167 talking about something widely-acknowledged. This part should be deeply revised.

4.	All the results are presented in KDE plots, which is not convincing. At least some numerical results should be presented to support the claim.

5.	The claimed spatial distribution of ID and OOD data in the logit space is also not strongly demonstrated in the submission. For example, the authors claim that ID data logits distribute across the orthogonal axes that span the logit space. To validate this, some PCA-based analysis should be provided, such as the reconstruction errors on ID and OOD data in the logit space. However, throughout the paper, there are only KDE plots on logit values and no any geometric empirical investigations. Such empirical results are insufficient and cannot provide a sound support on the claim.

### Questions
My questions correspond to the weaknesses above.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a comprehensive empirical study of In-Distribution (ID) and Out-of-Distribution (OOD) logit behavior in deep learning classifiers. The authors demonstrate that ID and OOD data exhibit distinct geometric patterns in the logit space: ID logits form clusters in positive regions aligned along class-specific orthogonal axes, whereas OOD logits remain centered near zero. The study shows that this logit configuration persists consistently across different architectures and datasets. Furthermore, the authors suggest leveraging ID-out logits as proxies for OOD detection, although the paper does not provide any methodology for practical deployment.

### Strengths
1.	The study covers a wide range of architectures and configurations, demonstrating the strong generalizability of the findings.
2.	The study integrates theoretical insights with experimental results, providing mutually consistent support for its conclusions.
3.	The geometric interpretation of the logit-space configuration could potentially serve as a foundation for developing simple OOD detectors.

### Weaknesses
1.	Although the empirical study is extensive, it primarily reinforces existing intuitions rather than offering a novel methodological or substantive theoretical contribution.
2.	The paper does not demonstrate that the reported findings can be effectively leveraged for designing an OOD detector.
3.	The text references numerous figures but detailed qualitative descriptions are limited

### Questions
1.	The before/after training analysis focuses exclusively on correctly classified ID samples, overlooking misclassified ID instances. However, in practice, the boundary between ID and OOD data often becomes ambiguous precisely because of such misclassified ID examples, which may exhibit logit distributions similar to OOD samples. Addressing this limitation would strengthen the empirical validity of the conclusions and provide a more realistic understanding of logit-space separability in practical OOD detection scenarios.

2.	The study covers a wide range of architectures and configurations, but it remains unclear which design choices most influence logit-space separability between ID and OOD samples. Providing guidelines or empirical insights on which architectures or hyperparameters favor clearer separation would significantly improve the paper’s practical utility.


3.	The paper proposes using ID-out logits as proxies for OOD, but it does not provide experimental validation. Presenting preliminary results for a binary classifier and clarifying the design of such a detector would strengthen the contribution.


4.	Although the paper includes a related work section, it does not clearly situate its new findings within the context of prior research. For example, are there existing OOD detectors that already exploit the observations made in this study? Which insights are genuinely novel compared to previous knowledge? Do any of the results contradict prior publications? Furthermore, the Neural Collapse (NeCo) phenomenon, extensively analyzed in several recent works, also describes the emergence of class-wise orthogonal clusters, though in the feature space rather than in the logit space. It would be valuable to discuss whether the observed logit-space configuration could be a manifestation or an extension of Neural Collapse. Such a connection could provide a theoretical grounding for the empirical patterns reported in this study.



This paper provides a thorough empirical analysis of ID and OOD logit behavior across a wide range of architectures and datasets. The experiments are well-executed, and the geometric patterns observed in logit space are consistent and generalizable, offering valuable insights for future OOD research. However, the study is primarily descriptive: it does not introduce a new method, lacks concrete guidance for OOD detector design, and does not fully situate its findings within existing theoretical frameworks such as Neural Collapse. Despite these limitations, the paper’s clarity, empirical rigor, and breadth of analysis make it a meaningful contribution that could inform and inspire follow-up work.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper studies the phenomenon that the magnitude of the logits for OOD samples are typically smaller than that of the ID samples. The mechanism behind this phenomenon is explained by analysing the loss. In the experiments, several networks are analyized by showing the difference in logit distributions of OOD/ID samples.

### Strengths
* In the experiments, many networks, including CNNs and vision transformers, are analyzed.
* The writing is easy to follow.

### Weaknesses
This phenomenon is not new, and it has been the reason for the design of the very first few OOD algorithms, such as MSP and max logit. Some later works [a] already explicitly designed a regularizaion to encourage the logit-smallness of OOD samples. The ID clustering structure is also used in [b] to desgin better OOD scores. Besides, the paper does not provide new OOD scores based on the gained insights. The experiments also lack quantitative results.


- [a] Training confidence-calibrated classifiers for detecting outof-distribution samples. ICLR 2018
- [b] MOS: Towards scaling out-ofdistribution detection for large semantic space. CVPR 2021

### Questions
/

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies learning from out-of-distribution data, and proposes a new perspective of geometric analysis. It is inspired by the limitation of existing OOD detection methods, which usually employs a scoring technique and may neglect the insight within the latent space. In this paper, the authors analyze the logit embedding distributions of ID and OOD data and reveal that the OOD data tends to cluster near the origin.

### Strengths
- The studied problem is meaningful. 
- The authors conduct extensive experiements to reveal the phenomenum.

### Weaknesses
- The contribution of this paper is limited. The main paper consists of many experimental results. However, there are no in-depth analysis regarding why the logit distributions of ID and OOD data exhibits such different phenomenum.
- After revealing the empirical discovery, there are no relevant algorithms proposed to further improve the OOD detection performance. This will also limit the contribution of this paper.
- There are no theoretical analysis provided.
- The experiments are only conducted on some general datasets (e.g., SVHN, CIFAR, ImageNet), which will limit the universality of the proposed method.

### Questions
- Is there any existing methods analyzing the OOD detection problem from the latent space?

### Soundness
2

### Presentation
2

### Contribution
1
