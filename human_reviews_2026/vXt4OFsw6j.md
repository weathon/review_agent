# Representation Learning of Ancient Greek Letterforms across Time

- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Learning representations that remain robust across centuries of variation in handwriting is a key challenge in diachronic representation learning of ancient Greek manuscripts. We introduce three datasets of ancient Greek handwriting for diachronic representation learning: Hell-Char, a curated training set spanning the 3rd–1st centuries BCE, and two evaluation sets, PaLit-Char (1st–5th c. CE) and Med-Char (9th–14th c. CE). To address challenges of symbolic variation, scarce data, and systematic degradation, we propose two methodological innovations: a similarity-weighted supervised contrastive loss that biases embeddings by human-perceived confusability, and a lacuna-driven augmentation scheme that simulates realistic manuscript corruptions. Trained with these strategies, both a lightweight CNN and a pretrained ResNet achieve strong recognition performance and produce embeddings that more coherently separate character classes than PCA or generic pretrained models. These embeddings enable clustering, identification of stylistic subgroups, and construction of prototype images that visualize diachronic evolution and transitional letterforms. Our results demonstrate that incorporating expert priors and domain-specific corruptions yields robust, interpretable representations, offering a transferable paradigm for representation learning under scarce, temporally evolving, and noisy conditions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the challenge of diachronic representation learning for ancient Greek handwriting by introducing three datasets. Temporal generalization tests show strong performance on PaLit-Char (84% accuracy) but limited performance on Med-Char.

### Strengths
1.The three cross-temporal datasets fill a gap in ancient Greek handwriting research, providing a standardized benchmark for low-resource, diachronic visual recognition.

### Weaknesses
1.The core methodological components (supervised contrastive loss, data augmentation) are adaptations of existing frameworks (Khosla et al., 2020 for SCL; general image augmentation for corruption simulation) rather than novel theoretical constructs. The "similarity weighting" and "lacuna-driven" modifications are incremental tweaks to fit the ancient manuscript domain, without introducing new mathematical formulations, learning paradigms, or theoretical insights that advance the broader field of representation learning—this limits the work’s contribution beyond domain-specific application.

2.The sharp accuracy drop on Med-Char (45%) is attributed to letterform change but lacks exploration of solutions (e.g., cross-dataset fine-tuning).

3.The paper fails to specify key technical details: how the dynamic similarity matrix in the contrastive loss is computed (global vs. batch-wise) and update frequency.

4.The paper does not clarify parameters for lacuna augmentation (e.g., size, density, curvature of lacunae) or validate their impact on model performance.

5.It Only tests Swin Transformer among Transformer models, with no exploration of variants (e.g., ViT, DeiT) that may better capture local stroke features.

### Questions
1.The authors should explicitly clarify the theoretical novelty of the similarity-weighted supervised contrastive loss. How does it differ from existing SCL variants beyond adding a domain-specific weight term? Are there new theoretical insights derived from this modification?

2.How is the dynamic similarity matrix in the contrastive loss calculated? Is it based on the entire training set or individual batches?

3.What criteria determined the parameters (shape, size) of lacunae in the augmentation? Were they informed by paleographic studies of manuscript degradation?

4.Why was only Swin Transformer tested? Have you considered Transformer variants optimized for local features (e.g., ConvNeXt-V2) for better performance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses learning representations for ancient Greek letter forms as attested in ancient manuscripts across eras and handwriting styles. The paper curates a set of ancient Greek handwriting datasets, proposes modifications to a representation learning scheme to improve performance on ancient Greek handwriting, and shows results of paleographic analysis aided by these representations.

### Strengths
Computational paleography is an important and under-explored field, and it is encouraging to see modern representation learning methods being leveraged to aid expert analysis.

The paper curates data for studying ancient Greek paleography with computational methods from disparate sources, making them more accessible for future work.

It appears that the learned representations with the proposed method are more effective for paleographic analysis and yield valuable insights.

The anonymous code is appreciated for aiding transparency and reproducibility.

### Weaknesses
The technical contribution of the paper appears limited, both regarding methodology and data.

The components proposed as novel are (A) similarity-weighted supervised contrastive loss and (B) lacuna augmentation. However, neither of these are fully explained, and both have conceptual issues which are not addressed: 

* (A) Similarity-weighted supervised contrastive loss: The term S_{y_i, y_a} is given as “dynamically computed from embeddings”. It is not clear how it is computed, whether these embeddings are the representations actively being learned, or if these refer to fixed, pre-computed embeddings. Conceptually, the contrastive learning objective is already expected to model visual similarity between samples, so it is unclear what motivates adding an additional similarity term. In addition, L155 mentions an addition “standard cross-entropy” loss, but it is not clear if this refers to training the backbone on letter classification, and if so, whether the contrastive loss is applied to the output of the classification head or to intermediate activations.

* (B) Lacuna augmentation: While this is presented as one of the main contributions of the paper, it is never clearly defined. L150 mentions that this augments images by masking; if so, this is similar to standard image augmentation strategies for training vision models.

There also lacks a comparison to stronger representation learning baselines such as zero-shot or fine-tuned CLIP, DINO, or DIFT features, while comparing only to baselines like pretrained ResNet features which are expected to be weak on challenging images.

While the curation and aggregation of data is a valuable contribution, much of the data is sourced from existing datasets (Hell-Date, PaLit), and it is unclear how much new data is contributed. The abstract and intro do not mention this, suggesting that this is a new data contribution. It should be clarified whether new data is being contributed, or whether the main contribution is the curation or filtering of existing data.

Overall, the bulk of the paper is devoted to exploratory data analysis and paleographic insights from methods such as clustering. While these are valuable insights in general, it is not clear whether they are in the scope of this conference.

### Questions
Is the proposed method specific to ancient Greek? It seems like the proposed method could be applied more generally.

Is Med-Char sourced from an existing dataset, or new data being released? Similarly, as suggested on L333, do the other datasets being contributed contain new data?

When you discuss “PCA” or “Otsu+PCA” features (Sec 5.2), does this refer to PCA applied directly to pixel intensity values?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method for representation learning of ancient Greek letterforms across time, using three historical datasets (Hell-Char, PaLit-Char, and Med-Char). The method combines similarity-weighted supervised contrastive loss with a lacunae enhancement strategy to improve performance in classification, clustering, and prototype visualization. Experimental results show that the proposed method outperforms baseline approaches in terms of accuracy and clustering quality.

### Strengths
1. The method applies contrastive learning and data augmentation to the problem of historical text recognition, offering potential practical value.

2. The results show the effectiveness of the proposed method over baselines in classification and clustering tasks.

3. The dataset and methodology could be valuable for future historical document analysis tasks.

### Weaknesses
1.  The method builds on established techniques (contrastive learning, supervised contrastive loss, and data augmentation). The primary innovation lies in its application to historical text recognition rather than new methodological contributions.

2. The paper lacks detailed descriptions of how the similarity matrix is updated and how lacunae enhancement is applied.  

3. The approach uses circular or elliptical shapes for missing characters (lacunae), but the paper does not provide enough detail on how these are generated.

### Questions
1. Could you clarify the novel methodological contributions beyond the application of existing techniques to a new domain? This would help us better understand the innovation in your approach.

2. Could you provide more details on the implementation of the lacunae enhancement strategy?

3. How did you prevent data leakage, particularly ensuring that the same document or scribe's handwriting is not repeated in both the training and validation sets?

### Soundness
2

### Presentation
3

### Contribution
2
