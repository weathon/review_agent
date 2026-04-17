# Efficient Patch Search in Whole Slide Images via Morphological Momentum Prototype Learning

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Digital histopathology images play a crucial role in cancer diagnosis, therapeutic response prediction, and identification of clinically relevant morphological features. However, processing Whole Slide Images (WSI) with gigapixel resolution introduces significant challenges in computer vision, exceeding the memory capacity of standard vision encoders. To address this, recent methods employ a multi-stage pipeline: dissecting the image into small patches, extracting patch-level features, and aggregating these features using global pooling through Multi-Instance Learning (MIL) to form a final slide-level representation. Despite achieving clinical-grade performance, this approach becomes increasingly complex with higher magnification due to the quadratic increase in patch numbers and the generation of numerous irrelevant or redundant patches. This complexity burdens the global pooling network, resulting in long inference times and excessive computational resources, while redundant patches introduce noise during the MIL process, limiting the model’s ability to utilize high-magnification features fully. To overcome these challenges, we propose Momentum Morphological Prototype Learning (MMPL), an efficient method that redefines WSI diagnosis as a searching process of relevant patch-level representations with a learned set of global prototypes. MMPL trains a fixed set of prototypes to retrieve the most informative patches, computing the diagnostic score using only the retrieved patches. Evaluated on WSI classification benchmarks, MMPL achieves state-of-the-art performance across various pathology tasks, including metastasis detection, tumor grading, and tumor subtyping.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work presents an efficient method for WSI analysis, called Momentum Morphological Prototype Learning (MMPL). MMPL trains a fixed set of prototypes to retrieve the most informative patches, and then computes the diagnostic score using only the retrieved patches, which can largely reduce the computational cost. The authors conducted experiments on three public datasets to evaluate the proposed method, along with a comparison with existing methods. Ablation study is also carried out to verify the contribution of the key components of the proposed method. Experimental results show that MMPL achieves the best performance on the CAMELYON16 and TCGA datasets for metastasis detection and tumor subtyping, but has inferior performance on the PANDA dataset for tumor grading.

### Strengths
The manuscript presents a new paradigm for WSI analysis. Different from previous works that utilized all patches of WSI for analysis, this work proposed MMPL that trains several prototypes to retrieve the most informative patches for WSI representation. In this manner, the computational cost is significantly reduced. For prototyping learning, the authors adopted an optimal transport (OT) formulation with uniform marginal constraints to achieve balanced representation of distinct morphological patterns while preventing prototype collapse and ensuring diverse information distribution. In summary, the proposed method is of some novelty.

### Weaknesses
1)	The proposed method involves multiple hyperparameters, such as the number of prototypes, k value for top-k sampling, and K value for the feature queue. Although the authors conducted experiments to investigate the effects of prototype number and k value on the classification performance on CAMELYON16, it is still unclear whether the proposed method is sensitive to those hyperparameters across different datasets.
2)	The computational cost of retrieval should be considered during training. Given the queue size set to 100,000, there is nontrivial computation per training iteration. Please elaborate on it.
3)	The proposed method did not achieve state-of-the-art performance on the PANDA dataset.

### Questions
Besides the weakness mentioned above, there are some concerns as follows:
1)	Please specify the number of WSIs of the PANDA and TCGA datasets in Section 4.1.
2)	Please briefly introduce the workflow of inference after the introduction of Section 3.4, which could give readers a better understanding of the test stage.
3)	The authors mentioned that their method MMPL used only 1.18% of the patch embeddings. It is better to provide the computation process in the supplementary file.
4)	For the visualization of the prototypes in Figure 3, I am surprised to see that the white background (non-tissue regions, i.e., prototype 14) is accounted for in this work. Normally, we exclude the white background in the data pre-processing stage. Please elaborate on it.
5)	The proposed method did not achieve state-of-the-art performance on the PANDA dataset, and there is no analysis of the reason. Besides, I am not clear about PANDA (R) and PANDA (K). For the prostate cancer grading tasks, did the authors compute the kappa score between the model predictions and the ground truth labels? Accuracy is also a commonly used evaluation metric.

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
3

### Summary
This paper proposes a novel framework named Momentum Morphological Prototype Learning (MMPL) for weakly supervised classification of Whole Slide Images (WSIs). The method addresses the computational inefficiency in multiple instance learning caused by an excessive number of instances by reformulating the problem as an efficient patch search and retrieval task. MMPL employs an optimal transport (OT) formulation to learn a set of prototypes, a design intended to prevent prototype collapse and ensure diversity in pattern representation. Additionally, the authors adopt a feature queue combined with exponential moving average (EMA) updates to mitigate cluster drift induced by rapid encoder parameter changes. Based on the similarity between prototypes and image patches, the method selects a subset from the WSI's patch collection and performs classification using only this subset. The authors demonstrate the effectiveness of their approach on multiple public histopathology datasets, and ablation studies confirm the contribution of each component within the proposed framework.

### Strengths
1. It raises a valuable question: how can we address the explosive growth in the number of patches during WSI tiling due to increased resolution, along with the consequent issues of computational efficiency and noise?
2. The proposed framework introduces an optimal transport (OT) formulation to learn prototypes in a self-supervised manner, and incorporates a retrieval mechanism to reduce computational cost during classification; furthermore, the entire framework is end-to-end trainable, demonstrating a certain degree of novelty.
3. The effectiveness of MMPL is validated on multiple datasets, with comparisons to various state-of-the-art methods, and ablation studies are conducted to verify the contribution of each component.

### Weaknesses
1. The authors' central claim—that their framework achieves high efficiency—has not been substantiated by sufficient experimental evidence or rigorous analysis.
2. The experimental comparisons exhibit anomalous results, which the authors fail to explain; these discrepancies may indicate incorrect implementation of the method or an unfair experimental setup.
3. Key techniques such as optimal transport for preventing prototype collapse, and the use of a feature queue combined with exponential moving average (EMA) to mitigate feature drift, have been previously proposed in influential works, yet the authors do not acknowledge or discuss these prior contributions.

### Questions
1. Could the authors provide a computational complexity analysis and performance metrics (e.g., inference latency, throughput) for the proposed framework when processing a single whole slide image (WSI) to completion of classification, and compare these metrics with those of other existing methods?
2. In Table 1, the Kappa score of IBMIL appears unusually low. Is this due to an implementation error? If not, could the authors provide an analysis explaining the underlying reasons?
3. Is it possible that end-to-end training, while effectively fitting the training task, might lead to overfitting and thus perform poorly on out-of-domain test sets or in real-world scenarios?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Momentum Morphological Prototype Learning (MMPL), a prototype-driven framework for efficient Whole Slide Image (WSI) classification. MMPL learns a fixed set of global morphological prototypes and uses an optimal transport assignment with the Sinkhorn-Knopp algorithm over a momentum feature queue to avoid prototype collapse and keep prototypes diverse, and retrieve only the most informative top-k patch features per slide. This retrieval reduces the number of patch vectors the aggregator must process and enables end-to-end training with lower memory use. The authors evaluate MMPL on standard WSI benchmarks (CAMELYON16, PANDA, TCGA), report state-of-the-art accuracy/AUC/Kappa in several settings, and present ablations showing benefits of the feature queue, OT-based prototypes, and dynamic top-k retrieval.

### Strengths
1 Reformulating WSI diagnosis as a multi-vector retrieval problem driven by learned prototypes and solving prototype collapse with an OT constraint is a fresh combination of ideas that hasn’t been widely applied in this context. Combining a momentum queue with Exponential Moving Average (EMA) for stable prototype learning is a reasonable adaptation and integration of ideas derived from self-supervised learning. 

2 Experiments across three widely-used WSI datasets (CAMELYON16, PANDA, TCGA) with multiple baselines (ABMIL, DSMIL, ZoomMIL, IBMIL, PANTHER, VIB) show consistent improvements. The ablation (queue/prototypes/OT vs K-means/uniform top-k) supports the mechanistic claims. 

3 The method is explained step-by-step with helpful figures showing architecture and a visualization of prototype patch retrieval. Loss terms and the joint training objective are clearly defined. 

4 WSI analysis is a high-impact application area; achieving similar or better performance while dramatically reducing processed patches addresses a real practical constraint (compute/memory) and can enable wider deployment and more frequent end-to-end training.

### Weaknesses
1 Efficiency claims need quantitative runtime/memory ablation. The paper states that only 1.18% of patches are used and claims reduced inference time, but provides no systematic runtime / GPU memory / FLOPs measurements vs baselines.

2 Sensitivity to key hyperparameters (M, queue size, top-k policy, τ, λ). The method depends on prototype count M, the queue size K, and the top-k allocation vector Ij. Only limited ablation is shown.

3 Theoretical justification or failure modes. While OT with uniform marginals mitigates collapse, the paper lacks a short formal discussion of when prototype assignments might still be suboptimal (e.g., extremely class-imbalanced slides) and how the method handles rare morphologies.

4 The MMPL architecture diagram is not aesthetically pleasing, except for the defect of misaligned box lines at first glance. Some symbols do not appear in the main text and are not explained in the captions

### Questions
1 Runtime & memory: Please report GPU memory usage and per-slide inference latency for MMPL and for at least tow strong baseline on the same hardware. 


2 Prototypes and M selection: How sensitive is performance to M? Despite OT uniform marginals, is there a risk that some prototypes remain unused? Provide the distribution of prototype assignments (counts per prototype) and show whether OT enforcement truly yields balanced semantic coverage. 

3 Provide clearer details and results comparing the two-step frozen-backbone training to full end-to-end training. Because the paper claims end-to-end is possible, thus conducting more ablation studies would make the argument more convincing. If possible, Could you provide a short expert evaluation or annotation to show prototypes align with meaningful histology concepts?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work introduces MMPL, a prototype-based method that searches whole-slide images for only the most informative patches to make a slide-level decision. MMPL works by learning a small set of global “morphological” prototypes with a momentum encoder and queue, assigning patch embeddings to prototypes via Sinkhorn-Knopp optimal transport, and then retrieving the top-k relevant patches per prototype to compute the prediction using just those patches. The proposed novelty is the combination of prototype learning with a memory queue for stable training and a dynamic allocation of retrieved patches per prototype. Experiments span CAMELYON16, PANDA (K/R), and TCGA tasks, comparing against MIL baselines such as ABMIL, DSMIL, CLAM, TransMIL, etc., and also evaluate different encoders including ResNet-50 and pathology foundation models (UNI, CONCH, and GigaPath).

### Strengths
- Lots of ablations: top-k selection, the effect of the queue and prototype components, uniform versus dynamic per-prototype selection, prototype assignment via Sinkhorn versus K-means, and backbone choices. 
- Presentation of the paper is good and math is sound. Method extends SK-OT for pathology and is overall sound (working both unsupervised and supervised).

### Weaknesses
- Very limited experiments and tasks. Only three datasets are evaluated (C16, PANDA, TCGA-Lung) and are only classification (no survival tasks). While experimental design on ablating performance of MMPL is sound, most of the tasks that MPPL are evaluated on are a bit simple. It would be interesting to evaluate on a greater range of tasks.
- Unclear if number of prototypes in MMPL are fixed versus adaptive, e.g. if the number of clusters can change. Why is the number of prototypes fixed to 15? Interpretability of MMPL clusters could also show deeper insights (comparison with PANTHER in learning quality of prototypes).
- Hard to understand where performance of supervised versus unsupervised MMPL is evaluated.
- Technically, this work addresses the problem of learning better prototypes for weakly-supervised classification tasks in pathology. The work generally extends the idea of SK-OT to the pathology domain, and though mostly applied, has good empirical experimentation of the technical components. For better impact, I would like to see more interpretability and evaluation on more diverse tasks as there are many MIL architectures that succeed on the evaluated tasks.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2
