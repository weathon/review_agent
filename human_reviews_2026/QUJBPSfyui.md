# Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Multivariate time series forecasting (MTSF) plays a vital role in numerous real-world applications, yet existing models remain constrained by their reliance on a limited historical context. This limitation prevents them from effectively capturing global periodic patterns that often span cycles significantly longer than the input horizon—despite such patterns carrying strong predictive signals. Naïve solutions, such as extending the historical window, lead to severe drawbacks, including overfitting, prohibitive computational costs, and redundant information processing. To address these challenges, we introduce the Global Temporal Retriever (GTR), a lightweight and plug-and-play module designed to extend any forecasting model’s temporal awareness beyond the immediate historical context. GTR maintains an adaptive global temporal embedding of the entire cycle and dynamically retrieves and aligns relevant global segments with the input sequence. By jointly modeling local and global dependencies through a 2D convolution and residual fusion, GTR effectively bridges short-term observations with long-term periodicity without altering the host model architecture. Extensive experiments on six real-world datasets demonstrate that GTR consistently delivers state-of-the-art performance across both short-term and long-term forecasting scenarios, while incurring minimal parameter and computational overhead. These results highlight GTR as an efficient and general solution for enhancing global periodicity modeling in MTSF tasks. Code is available at this repository: https://github.com/macovaseas/GTR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method that maintains a global temporal embedding and retrieves the corresponding temporal segment based on the absolute position of the input within the global cycle.

### Strengths
1. The core idea is sound and empirically effective. I particularly appreciate Figure 4, which provides a clear visualization that proves the practicality of the proposed approach.

2. The reported quantitative results are strong.

### Weaknesses
1. **Motivation with examples:**
The main motivation - namely, that certain global patterns cannot be captured within a limited lookback window - is reasonable but remains abstract. Providing a concrete conceptual example using an existing dataset would make this motivation more convincing and accessible.

2. **Global information guarantee of the global parameter matrix $Q$:**
The method introduces a learnable global parameter matrix $Q$, intended to encode global temporal patterns. However, it is not clear how or why this matrix is guaranteed to capture such global information. While Figure 4 provides empirical evidence, a more methodological justification or theoretical explanation would strengthen the paper.

3. **Weak ablation study:**
The ablation analysis is limited. It would be informative to include:

* Variants of the temporal pattern extraction module (e.g., replacing the 2D convolution).

* Experiments without instance normalization, to isolate its contribution.

4. **Limited comparison with related retrieval-based methods:**
The paper briefly mentions retrieval-based time-series forecasting approaches but does not include direct empirical comparisons. Including at least one retrieval-based baseline would help clarify the advantage of the proposed method.

### Questions
1. Provide why training this global parameter matrix $Q$ works, without any design consideration (W2).
2. Enhance experiments with more ablation study and retrieval-based baselines (W3, W4).
3. The conceptual example of scenario would strengthen the motivation (W1).

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
This paper tackles the limitation of restricted input windows in time series forecasting, which prevents models from capturing global periodic patterns. The authors propose a lightweight and plug-and-play module, **Global Temporal Retriever (GTR)**. GTR performs adaptive global temporal embedding over the entire cycle and dynamically retrieves and aligns globally relevant segments with the current input. This effectively bridges short-term observations with long-term periodicity. Experiments show that GTR achieves state-of-the-art performance on both short-term and long-term forecasting tasks with minimal computational overhead.

### Strengths
1. Simple and efficient design, easy to follow.
2. Clear motivation and well-organized paper structure.
3. Practical idea with strong feasibility and expected performance gains.

### Weaknesses
1. Experiments are mostly limited to input length 96, rather than the commonly used 336 setting in PatchTST/DLinear, which weakens the persuasiveness of the results.
2. The core idea shares similarities with models like **Cyclenet**, **TQNet** (which theoretically can learn long-term cycles from the entire training set), and **STiD**, i.e., serving as a plugin to capture long-term periodicity. This somewhat reduces novelty. The paper should compare more clearly and deeply with these methods to highlight differences.
3. Lacks comparison with other long-term periodic modeling techniques, such as timestamp-based long-cycle modeling approaches [1,2].
4. Baseline selection can be improved by including more recent strong models such as **SOFTS**, **TQNet**, etc.

**References**

[1] Variational Hierarchical N-BEATS Model for Long-term Time-series Forecasting.  
[2] Rethinking the Power of Timestamps for Robust Time Series Forecasting: A Global-Local Fusion Perspective.

### Questions
See weaknesses.

### Soundness
3

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
This paper proposes an adaptable and lightweight plug-and-play solution designed to mitigate the inherent lack of long-term temporal awareness in existing forecasting architectures. By enabling models to effectively access and utilize crucial global periodic patterns often missed in limited look-back windows, the proposed module significantly enhances the predictive ability, yielding superior and consistent performance gains in multivariate time series forecasting for both short-horizon and extended-horizon tasks.

### Strengths
1. The paper is well-written, clear, and logically structured. The authors provide a well-organized explanation of the technical specifics of the proposed plug-and-play module.

2. The experimental section is comprehensive. The authors have conducted extensive comparative and ablation studies on the multivariate time series forecasting task.

3. The proposed module is characterized by its lightweight and plug-and-play nature. This design choice significantly increases its generalizability and practical value when integrating with existing models, marking a valuable and impactful contribution to the field of temporal awareness modeling.

### Weaknesses
1. While the proposed module demonstrates effective performance improvements, its core mechanism is not fundamentally novel.

2. The clarity of the methodology are hindered by the lack of proper equation indexing, alongside several instances of ambiguous or missing variable dimensions within Section 3.

3. The theoretical analysis is confusing and unmatched with the proposed method, as the simplified linear assumptions in its derivation fail to establish a clear mechanistic link to the empirical success of the non-linear fusion and prediction architecture.

### Questions
1. Is the global cycle length ($L$) a predefined hyperparameter, or is its value learned by the network during the training process?

2. In the Eq.(1), what are the distinct advantages and disadvantages of the proposed positional embedding compared to conventional techniques (e.g., learnable embeddings) regarding periodicity awareness and noise robustness?

3. In page 5, line 246, is the output ($Z$) maintains the same feature dimension as the original input features?

4. In Table 3, can the authors explain whether these differential gains are primarily due to the inherent lack of temporal awareness in specific backbones or the unique periodic characteristics of the evaluated datasets?

5. Does the author assume a linear relationship between the embeddings, or that $z$ is linearly transformed from $x$? And is there any experimental evidence provided to directly support the claims made in Theorem 3.2?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposed a Global Temporal Retriever (GTR) module to model complex periodic dynamics and address the limitation that existing models have a fixed look-back window in multivariate time series forecasting. GTR overcomes this by defining a cycle index vector for cycle information alignment, which was used in retrieving temporal information from the global parameter matrix $Q$. Comprehensive experiments support the effectiveness of GTR, both when combined with a simple MLP backbone and when integrated into other architectures.

### Strengths
* The paper is very organized and well-written. The visualization (e.g., Pearson correlation matrix and Figure 2) clearly delivers the message and the design of GTR.
* The theoretical analysis and limitations are discussed in great detail, indicating the strengths and limitations of GTR that motivate future extensions. 
* While the GTR module is light-weight, the experimental results are strong.

### Weaknesses
* It would be better if the forecasting results included error bars, as the metrics have close values among different methods.
* The experimental setup could be further elaborated, e.g., how did the authors choose the hyperparameters for baseline methods? Please see the detailed questions below.

### Questions
* How did the authors decide the length of the time series segment (e.g., Figure 1)? Does it depend on the frequency of each dataset (as listed in Table 5)?
* How did the authors determine the global cycle length $L$ for each dataset in the experiments?
* Would this method work if the absolute time $t_0$ is shifted when pre-processing the dataset, e.g., standardizing the first observation to be 0?
* Does this method only produce point estimates, but not probabilistic forecasting?
* How did the authors choose the hyperparameters for baseline methods? Are these models trained from scratch, or do they reuse numbers from the literature? For example, in Table 3, the MLP-Layer (without GTR) is already competitive when compared with other state-of-the-art models (without GTR).

Typo:
In Appendix A.2, both metrics are defined with $\sum_0^T$, where $i=$ is missing and the index should start from $i=1$ (following the paper's notation).

### Soundness
3

### Presentation
4

### Contribution
4
