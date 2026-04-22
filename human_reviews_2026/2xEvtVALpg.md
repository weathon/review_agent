# TiViT: Time Series Representations Lie Hidden in Pretrained Vision Transformers

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Time series classification is a fundamental task in healthcare and industry, yet the development of time series foundation models (TSFMs) remains limited by the scarcity of publicly available time series datasets. In this work, we propose **Ti**me **Vi**sion **T**ransformer (**TiViT**), a framework that converts time series into images to leverage the representational power of frozen Vision Transformers (ViTs) pretrained on large-scale image datasets. TiViT achieves state-of-the-art performance on time series classification and anomaly detection benchmarks by utilizing the hidden representations of large OpenCLIP models.
We explore the structure of TiViT representations and find that intermediate layers with high intrinsic dimension are the most effective for time series classification. Furthermore, we assess the alignment between TiViT and TSFM representation spaces and identify a strong complementarity, with additional performance gains achieved by combining their features. Finally, we provide theoretical and qualitative insights about the benefits of 2D patching for time series modeling with ViTs. Our findings reveal a new direction for reusing vision representations in a non-visual domain.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes TiViT (Time Vision Transformer), a framework that converts time series into two-dimensional images to leverage pretrained Vision Transformers (ViTs) for time series analysis. The authors show that frozen ViTs can outperform existing time-series foundation models on classification and anomaly detection benchmarks without any retraining or fine-tuning on time series data.

### Strengths
1. The paper explores transferring pretrained vision models to time series tasks, which is conceptually interesting and inspiring.
2. The experiments are comprehensive, covering multiple popular time series datasets and different pretrained ViT models.

### Weaknesses
1. The idea of using ViTs for time series modeling has already been widely explored. The proposed method of converting series into images via heatmaps, applying ViTs for classification, and freezing ViT parameters has been investigated before. The paper mainly reuses this framework with different pretrained backbones, lacking substantial methodological novelty.
2. As a “foundation model”, the paper only evaluates classification and anomaly detection, while omitting the more fundamental time series forecasting tasks, making the evaluation scope incomplete.
3. Pretrained ViTs are typically computationally expensive, yet the paper does not discuss time or computational complexity.
4. The time-series-to-image transformation does not introduce new information; it simply adapts the data to ViT’s input format by channel replication, but may result in redundant computation due to copy channel and interpolation-based upsampling.

### Questions
1. The authors claim that the model requires no training, yet it still includes a trainable linear classifier. Is the performance improvement mainly due to the pretrained ViT representations or the linear layer? Does the model truly support zero-shot inference?
2. Why do the authors choose $\sqrt{T}$ as the patch size? How does this differ from using periodicity-based patching strategies? Can the authors provide comparative results for different patch sizes?
3. The transformation from time series to images seems to add no semantic information and is mainly for input alignment with ViTs. Could a lighter transformation be used to reduce parameters and computational overhead?
4. During the image transformation, the authors replicate the same single channel three times instead of using multivariate time-series variables as separate channels. Would this design waste potential cross-variable information?

### Soundness
2

### Presentation
2

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
The paper introduces Time Vision Transformer (TiViT), a new approach to time series classification by leveraging pretrained Vision Transformers (ViTs). TiViT transforms time series data into images and utilizes the powerful representations from these pretrained models, achieving state-of-the-art performance in both time series classification and anomaly detection.

### Strengths
The approach transforms time series into 2D images and uses pretrained, frozen ViTs for feature extraction, without the need for training or fine-tuning on individual datasets. The method is straightforward and reusable across different downstream tasks.

### Weaknesses
1. This paper presents an innovative framework and demonstrates its superior performance through experiments. However, there is room for improvement in the comparison with existing methods and the depth of theoretical analysis. I recommend the authors further clarify TiViT's innovative points, explain the differences from related works (such as NuTime, VisionTS), and strengthen the discussion of the advantages of the 2D conversion method.

2.  The paper does not provide a detailed explanation of why the frozen image-based ViT is effective in extracting time series features such as periodicity, trends, and non-stationarity. What is the relationship between these different modalities (image and time series ) of input?

### Questions
Q1: While the idea of converting time series to images and using pretrained vision models (like ViT) for classification is interesting, it is not entirely novel. Many similar approaches, such as VisionTS, already exist. The authors should clarify TiViT's unique advantages and innovative aspects over existing methods.

Q2: The approach in TiViT seems very similar to VisionTS, except for using ViT instead of MAE. The authors should elaborate on how TiViT differs from VisionTS and explain why ViT is a better choice for this task.

Q3: The authors mention that “2D representations are more suitable than 1D for time series tasks” needs further exploration. I suggest the authors enhance the ablation studies in section 4.5, with a deeper comparison of 2D vs. 1D representations across different datasets.

Q4: The 2D conversion method is a key innovation. The authors should expand on the advantages of 2D conversion and compare it to 
other methods like line plots and Gramian angular fields to clearly highlight its benefits. This will help to clearly demonstrate the practical benefits of 2D conversion and its impact on model performance.

Q5: Reference update:  the paper cites VisionTS: Visual Masked Autoencoders, which has now been accepted by ICML 2025, and should be updated in the references lists.

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
3

### Summary
The paper titled “TiViT: Time Series Representations Lie Hidden in Pretrained Vision Transformers
 ”introduces Time Vision Transformer (TiViT), a framework that leverages the pre-trained frozen Vision Transformers (ViTs), like OpenCLIP, by converting time series data into images for classification and anomaly detection. TiViT significantly outperforms conventional Time Series Foundation Models (TSFMs) on time series classification benchmarks. The success is attributed to the inherent advantages of 2D patching for time series modeling, which is shown both theoretically and empirically to increase the proportion of "label-relevant tokens" and make learning more efficient than traditional 1D modeling.

### Strengths
The proposed model seems to be working well given that it has been shown to achieve state-of-the-art performance on time series classification and anomaly detection benchmarks.

Its surprising to see  that large, frozen Vision Transformers (ViTs) pre-trained solely on natural images can serve as universal feature extractors for non-visual domains like time series analysis.

### Weaknesses
There is not much in terms of novelty in the paper. The overall process of converting time series into images to leverage visual representations is acknowledged as an existing approach in the field. The other changes seem pretty straightforward.

The theoretical insight regarding the benefit of 2D patching relies on a framework originally introduced for shallow Vision Transformers on generic data. The theoretical proposition is constructive and based on specific assumptions about pattern placement, which may not hold generally.

While the authors compare against state-of-the-art TSFMs like Mantis and Moment, the broader comparison to the vast array of existing time series classification methods (especially modern, non-FM approaches trained from scratch) is limited to citing results from a previous work (Goswami et al. (2024)).

### Questions
na

### Soundness
2

### Presentation
3

### Contribution
1
