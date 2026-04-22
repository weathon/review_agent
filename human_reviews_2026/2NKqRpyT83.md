# Byte Pair Encoding for Efficient Time Series Forecasting

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
Existing time series tokenization methods predominantly encode a constant number of samples into individual tokens. This inflexible approach can generate excessive tokens for even simple patterns like extended constant values, resulting in  substantial computational overhead. Inspired by the success of byte pair encoding, we propose the first pattern-centric tokenization scheme for time series analysis. Based on a discrete vocabulary of frequent motifs, our method merges samples with underlying patterns into tokens, compressing time series adaptively. Exploiting our finite set of motifs and the continuous properties of time series, we further introduce conditional decoding as a lightweight yet powerful post-hoc optimization method, which requires no gradient computation and adds no computational overhead. On recent time series foundation models, our motif-based tokenization improves forecasting performance by 36% and boosts efficiency by 1990% on average. Conditional decoding further reduces MSE by up to 44%. In an extensive analysis, we demonstrate the adaptiveness of our tokenization to diverse temporal patterns, its generalization to unseen data, and its meaningful token representations capturing distinct time series properties, including statistical moments and trends.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a pattern-centric tokenization framework for time series inspired by Byte Pair Encoding (BPE) in NLP. It compresses frequently occurring motifs within time series into a single adaptive-length token. This addresses the inefficiencies and lack of adaptability inherent in existing sample-based or patch-based tokenization methods. During the decoding phase, the authors propose a lightweight post-processing method called conditional decoding, which minimizes quantization error without increasing computational overhead. Zero-shot experiments conducted on foundational time series models (such as Chronos) demonstrate that this approach substantially enhances efficiency while maintaining or improving predictive performance.

### Strengths
This paper introduces a novel tokenization paradigm for time series, shifting from fixed patching to adaptive pattern extraction. This represents a significant breakthrough over existing time series tokenization methods (sample-based or patch-based). Experimental results demonstrate that the proposed tokenization method achieves an average efficiency improvement of 1990%. This compression is crucial for processing lengthy sequences and reducing computational overhead in large foundational models. 

The proposed conditional decoding approach further enhances performance without requiring gradient computation or increasing computational overhead during inference. 

The paper provides a detailed analysis of how quantization granularity and the lower bound on keyword frequency influence compression rates and generalization capabilities.

### Weaknesses
The entire framework and experiments focus on univariate time series, yet many real-world tasks involve multivariate dependencies. It would be preferable for the paper to include some discussion or experimental validation in this regard.

The conditional decoding module relies only on first-order dependencies (conditioning on the previous token), which may be insufficient for sequences with long-term temporal structure. Exploring higher-order or context-aware decoding could better capture complex temporal patterns.

The dataset scope is somewhat limited to relatively stationary benchmarks (ETT, Weather, Electricity, Traffic). Adding more diverse datasets such as Solar and PEMS-BAY would better test the tokenizer’s robustness under non-stationary and spatially correlated settings.
The paper does not report variance or confidence intervals. Given the large reported gains, it remains unclear whether the improvements are consistent across random seeds. Reporting averaged results with standard deviations would make the claims more reliable.

### Questions
The proposed framework is presented and evaluated on univariate series. How would the motif merging and vocabulary construction generalize to multivariate time series where inter-variable dependencies are critical?

The method is primarily validated empirically. While the efficiency gains are clear, the paper lacks a formal justification for why the hierarchical motif merging approach, adapted from BPE, fundamentally improves generalization or reduces reconstruction error compared to fixed-size patching.

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
This paper addresses the computational efficiency challenges faced by Transformer models in time series forecasting by proposing an innovative pattern-centric word segmentation method. This method solves the problems of existing word segmentation strategies generating excessively long sequences, leading to high computational costs, and their inability to flexibly adapt to patterns of varying lengths and complexities in time series data.

### Strengths
A. This paper is the first to systematically introduce the highly successful adaptive word segmentation concept (BPE) from NLP into the time series domain, proposing the concept of "pattern tokens." This provides a novel and elegant perspective for solving the problem of representing variable-length patterns in time series.

B. In addition to proposing a word segmentation framework, it also introduces the ingenious optimization technique of "conditional decoding." This technique is computationally lightweight, requires no retraining, yet effectively compensates for the information loss caused by discretization, significantly improving the practicality of the method.

### Weaknesses
A. The paper emphasizes the efficiency improvements in the inference stage but doesn't discuss the computational cost of the word segmenter itself (i.e., the pattern vocabulary construction stage). While mentioning that it's "quite inexpensive," quantitative data on the time and memory required to build the vocabulary on large datasets is lacking.

B. Conditional decoding is based on the first-order Markov assumption. However, the authors don't mention that for complex sequences with long-term dependencies, higher-order dependencies might be ignored.

C. The vocabulary used in the paper is built on a large-scale hybrid dataset (Chronos) to pursue zero-shot generalization. Could this lead to learned patterns being too general and lacking capture of key features in certain domains?

D. The paper demonstrates a compression ratio of up to 128x. While conditional decoding can compensate for some quantization errors, could such extreme compression cause some subtle but crucial patterns (such as transient outliers or subtle morphological differences) to be lost during word segmentation, thus failing to be perceived by subsequent models? The paper lacks theoretical analysis or targeted experiments on the information loss boundary.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a pattern-centric, adaptive tokenization scheme for time series forecasting . The method quantizes time series samples into discrete symbols and then merges recurring, variable-length patterns (motifs) to build a vocabulary. Authors also introduce conditional decoding as a lightweight, post-hoc optimization to reduce discretization error. Empirical results demonstrate that this approach significantly outperforms existing methods in forecasting performance and efficiency, while adaptively handling diverse temporal patterns.

### Strengths
1. The paper presents the first pattern-centric time-series tokenization scheme. By drawing on Byte Pair Encoding from Natural Language Processing, it adaptively compresses repetitive patterns in time series into individual Tokens.

2. The proposed method exhibits excellent computational efficiency, with relatively low computational effort required.

### Weaknesses
1. The experiments are highly insufficient: Table 2 only includes 5 datasets, with only one setting considered per dataset. Table 9 selects overly outdated models, lacking models from the past two years—for instance, PatchTST, which adopts a fixed patch-partition approach, is not included. Additionally, the experimental results in Table 3 lack persuasiveness; as the authors themselves note, comparisons across different model architectures and pre-training datasets fail to demonstrate that the proposed Motif-based tokenization method outperforms Patch-based tokenization methods.

2. The paper is excessively brief in certain sections, specifically the Introduction and Experiment chapters. In contrast, the Appendices are disproportionately lengthy, and key experimental results are scattered across different sections—this significantly hinders readability. For small models, there are already quite a few studies on unfixed patches, yet the authors do not seem to have mentioned this.

### Questions
1. Line 73: "All these transformer architectures rely on two basic tokenization techniques: using every sample as a token or extracting fixed-length patches from a time series." In fact, there are already foundational models from related studies in recent literature.

2. Line 347: "Due to their fixed length, rigid patches are unable to capture these inter and intra series variations." Please explain this viewpoint more clearly and should include additional experiments on patch-based models to support this argument.

3. Line 363: The paragraph starting with "Compared to the MOMENT model..." cites many models, including PatchTST, but only provides verbal descriptions—yet I have not seen any experiments on the compression rates of these models.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel tokenization method for time series forecasting inspired by Byte Pair Encoding from natural language processing. The approach involves discretizing time series into symbols and then iteratively merging frequent adjacent patterns into a vocabulary of variable-length motifs, enabling adaptive compression based on pattern complexity. The authors also introduce conditional decoding, a post-hoc optimization technique that reduces quantization error by leveraging conditional distributions of continuous values. The paper claims significant improvements in forecasting accuracy and computational efficiency over existing methods like Chronos, MOMENT, and Moirai in zero-shot settings.

### Strengths
The adaptation of BPE to time series tokenization is creative and directly tackles the inflexibility of existing methods.
The study covers multiple model sizes, tokenizer configurations, and datasets, providing a broad empirical analysis.
The examination of token embeddings reveals meaningful correlations with statistical properties, adding depth to the methodology.

### Weaknesses
The outperformance of patch-based models is not isolated to tokenization, as differences in architecture and training data confound the results.
Compression ratios are used as a proxy for efficiency without reporting actual inference times or FLOPs, overlooking potential overheads from tokenization and detokenization.
The benefits of adaptive tokenization are not rigorously tested against a range of fixed-patch lengths, leaving open whether adaptivity is truly superior.
The paper lacks clear visualizations of what specific temporal patterns the motifs capture, reducing interpretability.

### Questions
1. Can you provide a controlled comparison where your BPE tokenizer is evaluated against fixed-patch tokenizers within the same model architecture and training framework to isolate the effect of adaptivity?
2. How do actual inference times compare between your method and baselines on standardized hardware, accounting for full tokenization and detokenization costs?
3. Could you visualize the top-K most frequent motifs as continuous-time shapes to clarify what temporal patterns they represent?
4. What practical guidelines do you recommend for selecting key hyperparameters like the number of bins and minimum motif occurrence on new datasets?

### Soundness
2

### Presentation
2

### Contribution
2
