# Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Irregular Multivariate Time Series (IMTS) are characterized by uneven intervals between consecutive timestamps, which carry sampling pattern information valuable and informative for learning temporal and variable dependencies.
In addition, IMTS often exhibit diverse dependencies across multiple time scales.
However, many existing multi-scale IMTS methods use resampling to obtain the coarse series, which can alter the original timestamps and disrupt the sampling pattern information.
To address the challenge, we propose ReIMTS, a **Re**cursive multi-scale modeling approach for **I**rregular **M**ultivariate **T**ime **S**eries forecasting.
Instead of resampling, ReIMTS keeps timestamps unchanged and recursively splits each sample into subsamples with progressively shorter time periods.
Based on the original sampling timestamps in these long-to-short subsamples, an irregularity-aware representation fusion mechanism is proposed to capture global-to-local dependencies for accurate forecasting.
Extensive experiments demonstrate an average performance improvement of 27.1\% in the forecasting task across different models and real-world datasets.
Our code is available at [https://github.com/Ladbaby/PyOmniTS](https://github.com/Ladbaby/PyOmniTS).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the valuable problem of learning multi-scale information from irregular time series. ReIMTS, a multi-scale method based on recursive splitting followed by concatenation, is proposed. Forecasting experiments demonstrate its competitive performance across settings. While the idea is overall well-motivated and novel, more explanatory experiments could further strengthen the claims.

### Strengths
1.The paper is well-motivated, and the proposed ReIMTS is an interesting and easily understandable solution to the problem.

2.Benchmark experiments are extensive for the IMTS forecasting task, and the corresponding codes are available.

### Weaknesses
1.Although Figure 1 illustrates a sampling pattern present in the dataset, the term "sampling pattern" is rarely used in existing works. More careful explanations of this concept are needed.

2.At line 187, the paper states that $L^n$ and $T^n$ are distinct, with the difference lying in the presence of time units. The notation causes some confusion during reading, raising questions about whether it is necessary to distinguish them using different symbols.

3.Although ReIMTS+GraFITi appears to have good efficiency compared to other multi-scale methods, its training time (66ms) is double that of GraFITi (33ms). This raises doubts about the efficiency of ReIMTS when applied to other backbones.


4.The paper is entitled “Learning... Representations...”, which implies that ReIMTS functions as a representation learning booster. While the benchmark experiments on the forecasting task are extensive, additional experiments on the classification task could make the claim more persuasive.

### Questions
1.As mentioned in Weakness 2, is it really necessary to distinguish between $ L^n $ and $ T^n $, two symbols that differ only in the presence of time units? For example, in line 203, why is $ e^1_{\text{time},0:T^1} $ using the symbol with time units, when it actually belongs to $ R^{L^1 \times d_{\text{model}}} $, which uses the symbol without time units?

2.As mentioned in weakness 3, how is the efficiency of ReIMTS when applied to other backbones, like PrimeNet[1] and mTAN[2]?

3.In Section 4.2, the paper introduces a representation fusion method that employs a scoring linear layer to assign weights to global representations. Could the authors provide further insights into this design to clarify how different representations are combined? Specifically, does ReIMTS tend to prioritize the use of local representations or global ones?

4.Representations learned in the classification task are plotted in Figure 8. However, benchmark comparisons have not included metrics from classification. How does ReIMTS perform under other downstream tasks, such as classification?


5.The paper's experimental settings follow those from HyperIMTS[3], while the HyperIMTS model is not included in the benchmark comparisons. Why is it not included? Can ReIMTS work with HyperIMTS?

[1] R. R. Chowdhury, J. Li, X. Zhang, D. Hong, R. K. Gupta, and J. Shang; “PrimeNet: Pre-training for Irregular Multivariate Time Series”; AAAI 2023.

[2] S. N. Shukla and B. Marlin; “Multi-Time Attention Networks for Irregularly Sampled Time Series”; ICLR 2021.

[3] B. Li, Y. Luo, Z. Liu, J. Zheng, J. Lv, and Q. Ma; “HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting”; ICML 2025.

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
This paper proposes ReIMTS, a recursive multi-scale modeling approach for irregular multivariate time series (IMTS) forecasting, which splits samples recursively and employs irregularity-aware representation fusion mechanism

### Strengths
1. The motivation is clear, and the method is novel.
2. Experimental evaluation is comprehensive.
3. The method is easily adaptable to different models.

### Weaknesses
1. The notation and equations are not professional and can be largely improved. This paper uses a lot of double subscripts/superscript, and very long subscripts/superscripts, especially in Sec 4.1, from Eq. 2 to Eq. 6. I suggest the author reduce the length of Sec 4.1, and move some interesting experimental results or useful method comparison (Fig. 6) from the supplementary to the main paper.
2. Potential comparison fairness issues:
 - Backbone models are modified when integrated into ReIMTS (layers reduced, dimensions changed - Appendix B), unclear if these modifications could account for some improvements
 - Is there a difference in hyperparameter settings used for ReIMTS variants vs. baselines?
3. (minor) This paper could benefit from more discussion on the results. Table 2 shows that the performance improvement achieved by using different backbones is quite different, 62.3% vs. 9.9%. This is somewhat unusual and could benefit from an in-depth analysis.
4. (minor) Increased memory usage. Figure 3 shows that the available memory has increased by more than 50%(0.390 vs 0.598).

### Questions
I am happy to raise my rating of the paper if the author can address my concerns about **1) notations/equations** and **2) fairness**.

### Soundness
3

### Presentation
2

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
This paper proposes ReIMTS, a recursive multi-scale modeling framework for Irregular Multivariate Time Series (IMTS) forecasting. Unlike prior approaches that rely on resampling to generate coarse-grained sequences, ReIMTS preserves the original timestamps and recursively splits each sample into subsamples with progressively shorter time spans. An irregularity-aware fusion mechanism is introduced to integrate information across scales, enabling the model to capture both global and local temporal dependencies without destroying sampling patterns.

### Strengths
1. The paper is clearly written and logically structured, providing background on multi-scale modeling for irregular time series.

2. The proposed irregularity-aware fusion is intuitive and aligns well with the challenges of IMTS.

3. Experimental evaluation is comprehensive, covering multiple datasets and models, and demonstrates strong empirical results.

### Weaknesses
1. The novelty of the approach is somewhat limited in that the recursive multi-scale design can be viewed as a restructured version of existing patch-based or hierarchical multi-scale strategies.

2. The motivation for using a recursive structure, as opposed to other multi-scale fusion designs, is not clearly justified.

### Questions
1. Does the recursive decomposition introduce cumulative error across scales, and if so, how is this controlled or regularized?

2. How does ReIMTS handle dependencies among different variables in IMTS?

3. Could the authors clarify the specific motivation for adopting recursion instead of parallel or hierarchical multi-scale fusion?

4. Since the recursive splitting resembles patch-based segmentation, what distinguishes ReIMTS from existing hierarchical or patch-based multi-scale models?

### Soundness
3

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
4

### Summary
The authors describe a new method to learn multi-scale representations for irregular time series data.

### Strengths
- original idea to learn multi-scale representations
- well written paper
- good results

### Weaknesses
- state-of-the-art comparison should be extended, especially in extending the discussions on strengths and weaknesses of previous work, clearly specifying the contributions and possible limitations of the new method proposed

### Questions
What are the limitations of your method, and for which dataset characteristics does it perform well, and for which not?

### Soundness
2

### Presentation
3

### Contribution
3
