# TimesMR: Unlocking the Potential of MLP and RNNs for Multivariate Time Series Forecasting

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Multivariate time series forecasting is critical across various domains, requiring effective modeling of temporal dependencies and variable correlations. Existing multi-scale models, while effective for temporal dependency modeling, often rely on complex and computationally expensive feature extractors. Similarly, attention mechanisms, though powerful for capturing variable relationships, suffer from high complexity on high-dimensional datasets and introduce noise from weakly related variables, leading to performance degradation.
To address these challenges, we propose TimesMR, a novel model with two key innovations. First, we design multi-scale MLP modules, namely multi-patch MLP and multi-downsampling MLP, to enhance temporal dependency modeling with lightweight and efficient architectures. Second, we introduce grouped bidirectional RNNs for efficient variable correlation modeling, which reduce computational costs while preserving performance by grouping variables and capturing both intra- and inter-group correlations. 
Extensive experiments on sixteen datasets demonstrate that TimesMR achieves state-of-the-art performance, surpassing eighteen existing models. Our contributions include novel plug-and-play modules for temporal and variable modeling, offering an effective and efficient solution for multivariate time series forecasting. The code is available at https://anonymous.4open.science/r/TimesMR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors address two core challenges in MTSF: the high complexity of existing temporal models and the computational cost and noise-sensitivity of attention-based variable modeling in high-dimensional data.

To solve this, the authors propose **TimesMR**, an "effective and simple" architecture composed of two core modules:
1.  **Multi-scale MLP Module**: This module replaces complex temporal feature extractors with two lightweight variants (Multi-patch MLP and Multi-downsampling MLP) to efficiently capture multi-scale temporal patterns.
2.  **Grouped Bidirectional RNNs (Grouped BiRNNs)**: This module is proposed for efficient variable correlation modeling. The authors argue that the gating mechanism of RNNs (specifically GRU) helps suppress noise. To manage high dimensionality, variables are divided into $G$ groups (e.g., $G=\sqrt{C}$), processing "intra-group" correlations in parallel before an "inter-group" BiRNN exchanges information, reducing complexity to $\mathcal{O}(\sqrt{C})$.

The authors conduct comprehensive experiments on 16 datasets against 18 baselines, demonstrating that TimesMR achieves state-of-the-art (SOTA) or competitive performance in the vast majority of cases.

### Strengths
* **Strong Empirical Performance**:
    The most significant contribution of this paper is its SOTA results across an exceptionally broad set of benchmarks. TimesMR consistently outperforms 18 strong baselines (including iTransformer, Pathformer, and TimeMixer) on 16 datasets, making this a solid and convincing empirical contribution.

* **Pragmatic and Efficient Architectural Design**:
    The authors correctly identify the computational pain points of current SOTA models (like Transformers). The use of lightweight MLPs and RNNs as replacements is a very pragmatic design choice.
    The Grouped BiRNN module strikes a good balance between efficiency and performance. The analysis shows its computational complexity is $\mathcal{O}(\sqrt{C} \cdot D^2)$, which is superior to attention's $\mathcal{O}(C^2 \cdot D)$. Experiments confirm that TimesMR has a significantly faster training time on high-dimensional datasets (e.g., Traffic, PEMS07) than models like iTransformer.

* **Strong Component-wise Analysis**:
    The experimental design is thorough. Beyond SOTA comparisons, the paper validates the "plug-and-play" capability of its modules. Furthermore, it provides a direct comparison of Grouped BiRNN against Grouped Attention and a cost-benefit analysis for different group numbers, all of which greatly strengthen the paper's claims.

### Weaknesses
* **Architectural Redundancy**:
    The paper proposes two parallel Multi-scale MLP modules: Multi-patch and Multi-downsampling. However, the experimental results show their performance is nearly identical. Critically, the authors confirm in Appendix C.1 that combining them yields no performance benefit, attributing this to "inherent overlap" and "redundancy". This raises the question of whether it is reasonable to propose two highly redundant modules. The paper does not seem to describe any interaction between them (they are presented as an either/or choice in Fig. 3b), which makes the final model design feel less concise than it could be.

* **Incomplete Discussion of Grouping Strategy**:
    The paper's discussion of the Grouped BiRNN strategy is incomplete, focusing on efficiency while omitting key methodological details and trade-offs.
    * **Missing Methodology (The "How")**: The paper **completely omits *how* variables are assigned to groups**. Are they grouped sequentially (as they appear in the dataset), randomly, or based on some pre-calculated correlation? This is a critical missing detail. The formation of these local groups directly dictates which variables can interact within the "intra-group" BiRNN, and a poor grouping choice (e.g., splitting two highly correlated variables) could significantly hamper performance.
    * **Motivation (The "Why")**: The motivation appears to be **efficiency, not performance**. From an information-theoretic perspective, not grouping ($G=1$) is "lossless," while grouping ($G>1$) is a "lossy" approximation. The experimental results in Appendix B.4 seem to confirm this: **no grouping ($G=1$)** actually achieves the **best (lowest) MSE** on multiple datasets. This strongly suggests the strategy is an engineering trade-off that sacrifices accuracy for speed.

* **Unanalyzed Padding and Truncation Flow**:
    When the number of variables $C$ is not divisible by the group number $G$, the model must use Padding up to $C'$. This padded variable then participates in the "intra-group BiRNN" computation. At the end of the process, the model "truncates" the output, discarding the padded variable's representation. The paper **does not specify what padding method is used** and **provides no analysis** of whether this "compute-then-discard" process has an unintended impact on the vector representations of the *other real variables* within the same group.


* **Unclear Motivation for BiRNN Integration**: 
	The introduction of the BiRNN component lacks a clear and compelling design motivation, which raises concerns that the overall contribution may be perceived as a **progressive extension combining PatchTST and BiRNN** rather than a genuinely novel architectural advancement. The paper does not articulate: * **What** specific temporal dependencies BiRNN captures that patch-based MLPs or Transformers cannot; * **Why** this integration is essential instead of being an optional enhancement; * **How** BiRNN interacts with or complements the multi-patch representation in a principled way. Without addressing these questions, readers may question whether the incorporation of BiRNN is fundamentally necessary, potentially weakening the perceived scientific significance and originality of the proposed model.

### Questions
1. Given that the Multi-patch and Multi-downsampling modules are demonstrated to be functionally redundant, could the authors discuss the necessity of retaining both? Would one module suffice? Or, is there any latent interaction between these two modules that was not discussed?

2.  I have two questions about the group strategy：
    * Could the authors first clarify **what method** is used to assign variables to different groups? (e.g., sequential, random, correlation-based?) This is a critical, un-discussed detail.
    * Following from Figure 5, is it correct to conclude that grouping ($G>1$) is **purely** a trade-off for computational efficiency, and does not in itself offer a performance benefit over the "lossless" $G=1$ case?

3. I have two questions about the "pad-truncate" flow:
    * Could the authors first clarify what specific padding method (e.g., zero-padding, mean-padding, etc.) is used when the variable dimension $C$ is padded to $C'$?
    * Could the authors provide an analysis of the specific impact this "compute-then-discard" flow has on model accuracy? Does the padded variable negatively affect the representations of the real variables within its group during the intra-group computation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes TimesMR, a multivariate time-series forecasting model that combines (i) two lightweight multi-scale MLP temporal modules and (ii) a Grouped BiRNN variable-correlation module that processes variables to exchange intra- and inter-group information efficiently. Across 16 datasets and against 18 baselines, the authors report state-of-the-art average MSE/MAE, plus ablations showing each module’s contribution and efficiency analyses versus input length. Code is given as an anonymous link.

### Strengths
1. The proposed method achieves broad, consistent gains on 16 datasets, including high-dimensional traffic/PEMS benchmarks. Ablations show each module matters. Efficiency curves vs. input length are informative.
2. Architecture and data flow (patching/downsampling, RevIN, Grouped BiRNN intra/inter passes, fusion) are well illustrated.
3. Results indicate a compelling accuracy/efficiency trade-off for high-dimension settings where attention can be costly or noisy.

### Weaknesses
1. Temporal modules are close to prior multi-scale designs but implemented with MLPs; the variable module revisits RNNs with grouping. The paper’s contribution is engineering-oriented rather than theoretically or algorithmically deep.
2. Using dataset-level mean Pearson correlation (MCC) as a noise proxy is limited; correlation structure is often non-linear, time-varying, or lagged. Stronger diagnostics (e.g., partial correlations, Granger-style tests, lagged cross-correlations, spectral coherence) or synthetic studies would be helpful.
3. Given rising strong non-attention baselines and foundation-model-style forecasters, a brief comparison or discussion would be valuable even if only to argue scope differences.

### Questions
1. For Fig. 1, how was lookback length “searched”? Was the same protocol (search space, budget, early stopping) applied to all baselines? Please clarify the validation split and ensure no test leakage.
2. Could you report per-dataset standard deviations over multiple seeds and perform paired significance tests (e.g., Wilcoxon) against the strongest baseline(s)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes TimesMR, a forecasting model for multivariate time series. It combines: 1) Multi-scale MLP modules (multi-patch and multi-downsampling) to capture temporal patterns efficiently, and 2) Grouped BiRNNs to model relationships between variables while keeping computation manageable. Across extensive experiments, TimesMR achieves strong performance and scales better than attention-based models. Its components can also plug into other backbones and improve them.

### Strengths
1. The proposed Grouped BiRNN offers an effective and efficient way to model relationships among variables, reducing the computational burden of attention-based methods while preserving performance. 

2. The multi-scale MLP modules effectively capture temporal patterns with lightweight operations, demonstrating that complex architectures are not always necessary for strong results. 

3. Extensive experiments on 16 datasets, thorough ablations, and plug-and-play validations confirm both the effectiveness and generality of the proposed components. 

4. The paper is clearly written, well-organized, and convincingly shows that TimesMR achieves a strong balance between accuracy, efficiency, and scalability for multivariate time series forecasting.

### Weaknesses
1. The MLP-based multi-scale part feels like a lighter version of existing local time series embedding methods. The novelty mainly lies in simplifying.

2. The grouping method is more like heuristic. Variables are grouped by simple rules (√C or fixed G). There seems no discussion of how group splitting or variable order affects results.

3. The reason for using a BiRNN instead of other sequence-mixing methods in the Variable Correlation Module is not very clear to me. It seems that using attention or MLP layers could achieve similar effects: since the mixing occurs along the variable dimension, attention would not suffer from the quadratic complexity issue that arises from growing sequence length. Moreover, because the number of variables is fixed, an MLP could also perform this mixing effectively. More discussion is needed to justify this design choice and to better support the paper’s claim of “unlocking the potential of RNNs” in the title.

4. Although many recent works tend to use a short input length of 96 for simplicity, in my opinion, conducting experiments with longer input lengths would provide more practical and informative comparisons (for example, the default input length of 512 used in PatchTST). Since other sequence modeling fields, such as NLP, have already explored training with extremely long context windows, time series forecasting, as an area that often deals with relatively small datasets, could also benefit from revisiting such experimental settings. Moreover, because the searched lookback length is one of the key motivations of the proposed method, I believe it would likely perform even better with longer input lengths.

### Questions
1. Will the grouping method affect the final result significantly? Like using random, contiguous, or correlation-based grouping?

2. Why use an element-wise product for combining intra-group and inter-group features? It seems that additive would be more natural since the module can be seen as performing feature preprocessing? 

3. The sturcture of the predictor can be more detailed. A practical guidance on choosing the predictor, the patch and down variants would be helpful.

4. Notation: It seems that the RHS in Eq. 5 appears twice?

### Soundness
2

### Presentation
2

### Contribution
2
