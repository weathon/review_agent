# PatchCat: Rethinking Temporal Tokenization in Time Series Forecasting

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Temporal tokenization serves as a fundamental component in time series forecasting, transforming raw signals into token representations. Existing temporal tokenizers fall into three typical categories, mapping time series into tokens at the point-wise, patch-wise, or variable-wise level. Through a fair comparison, we observe that none of these paradigms simultaneously balance forecasting accuracy with computational efficiency. Motivated by the accuracy benefits of patch-wise tokenizers and the high efficiency of variable-wise tokenizers, we propose PatchCat, a competitive alternative. PatchCat segments the input time series into consecutive patches and concatenates these embeddings in chronological order. This workflow not only preserves local semantics and sequential information, but also enables univariate series to be compressed into a single token, achieving efficiency comparable to variable-wise methods. To further enhance representational capacity, we adopt a linearly increasing dimension allocation strategy and the variable-wise affine transformations. Experiments show that replacing the tokenizer in many existing methods with PatchCat can effectively improve prediction performance. To further leverage PatchCat's strengths, we develop PCMLP, a simple yet powerful model based on a multilayer perceptron. Extensive experiments across 13 challenging real-world datasets demonstrate that our approach achieves competitive performance compared to state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces PatchCat, a simple tokenization scheme that compresses each variable’s patch embeddings into a single token to balance accuracy and efficiency in time series forecasting. Using a lightweight MLP backbone (PCMLP), the method achieves strong results across multiple datasets with solid ablation and integration studies.

### Strengths
* The paper tackles a meaningful trade-off in time series forecasting, namely the accuracy of patch-wise tokenizers versus the efficiency of variable-wise ones. The motivation to balance these two aspects is sound.
* The proposed patch-then-concat strategy is easy to understand and implement. Compressing each variable’s patch information into a single token offers a clean bridge between existing tokenization paradigms.
* The experiments on multiple real-world datasets show that the proposed method achieves strong or even state-of-the-art results

### Weaknesses
* The name “PatchCat” and the method’s focus are very close to TimeCAT (ICLR 2025 submission), which also studies efficient, patch-based modeling for time series. The absence of any citation or comparison to this concurrent work makes the literature review feel incomplete.
* The paper claims that the “patch-then-concat” design preserves sequential information, but by collapsing all patch embeddings into a single token per variable, temporal relationships appear lost before reaching the backbone. In the PCMLP setting, an MLP cannot explicitly model temporal dependencies, making the approach resemble variable-wise tokenization rather than a genuine temporal one.
* The “linearly increasing dimension” rule is based on a simple “recent matters more” idea but lacks theoretical or empirical depth.

### Questions
* Please clarify the connection to TimeCAT and similar patch-based models. Given the clear overlap in motivation, a direct comparison or citation seems necessary.
* If all patches are compressed into a single token, how does PCMLP actually exploit sequential information? Can you provide evidence that temporal cues are retained and used effectively?
* Could you provide a stronger rationale or experimental support for the linear dimension allocation strategy? Why not use a nonlinear or learnable approach?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PatchCat, which divides input time series into consecutive segments and concatenates their embeddings in chronological order. This approach preserves local semantics and sequential information while compressing univariate series into a single token, achieving efficiency similar to variable-wise processing. Besides, PatchCat employs a strategy of linearly increasing dimension allocation and applies variable-specific affine transformations to enhance its representational power. Experiments show that replacing the tokenizers in existing methods with PatchCat improves prediction performance.

### Strengths
1. This work tackles key technical issues in current temporal tokenizers by first categorizing them into three types: point-wise, patch-wise, and variable-wise. It then introduces PatchCat, a new approach that uses a "patch-then-concat" strategy to address the common trade-off between accuracy and efficiency.
2. The proposed design choices are closely aligned with the nature of time series data. Concretely, allocating dimensions in a linearly increasing manner is shown to be more effective than uniform allocation, supporting the idea that "recent events matter more." Besides, variable-wise affine transformation improves the model's ability to distinguish between different variables during tokenization.
3. Extensive experiments show the effectiveness of the approach. PCMLP achieves better performance than 8 state-of-the-art models across 11 real-world datasets. Ablation studies confirm the contribution of each component.

### Weaknesses
1. Core ideas of PatchCat and PCMLP  are incremental improvements, lacking original tokenization paradigms or architectural designs. .
2. This paper also suffers from weak motivation and insufficient visualization. No intuitive visualizations to highlight existing tokenizers’ limitations, and theoretical basis for key designs (e.g., linear dimension allocation) is insufficient.

### Questions
1. What are the innovations of the PCMLP structure compared with other methods? It seems to be just a stack of relatively simple modules.  
2. In the results of Table 3, after integrating PatchCat, no significant performance improvement is achieved, which fails to accurately reflect the superiority of the proposed method.  
3. The idea of combining the advantages of two types of temporal tokenizers is somewhat straightforward and common. Is it possible to emphasize the innovation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes PatchCat, a method for tokenizing time series for forecasting models. Time series are split into patches and embedded with linearly increasing embedding dimension (recent patches have larger embedding size). The embeddings are then concatenated to construct a single token. Authors also proposed a complete model to validate this approach, which uses an MLP to map the token embedding into forecasts. Experiments have been conducted on 11 datasets showing improved results with the proposed approach.

### Strengths
- Authors explore a simple idea which works well, at least in the context of the results reported in this paper. Note: the empirical analysis has limitations, please see weaknesses for details.

### Weaknesses
- Although the paper investigates an interesting subproblem and demonstrates improved results, the main idea explored in the paper is not substantial enough for a main conference paper. As such, this work is better suited for a more focused venue (e.g., a time series workshop).
- The literature review only focuses on the relatively narrow long-term forecasting-style literature and makes no mention of time series foundation models, where the notion of tokenization has been explored from multiple angles and likely holds more significance than task-specific models. Relevant works (non-exhaustive list):
    - Ansari, Abdul Fatir, Stella, Lorenzo, et al. "Chronos: Learning the language of time series." arXiv preprint arXiv:2403.07815 (2024).
    - Woo, Gerald, et al. "Unified training of universal time series forecasting transformers." (2024): 53140.
    - Masserano, Luca, et al. "Enhancing foundation models for time series forecasting via Wavelet-based tokenization." arXiv preprint arXiv:2412.05244 (2024).
    - Talukder, Sabera, Yisong Yue, and Georgia Gkioxari. "Totem: Tokenized time series embeddings for general time series analysis." arXiv preprint arXiv:2402.16412 (2024).
    - Götz, Leon, et al. "Byte Pair Encoding for Efficient Time Series Forecasting." arXiv preprint arXiv:2505.14411 (2025).
    - Wang, Yihang, et al. "LightGTS: A Lightweight General Time Series Forecasting Model." arXiv preprint arXiv:2506.06005 (2025).
- The experiments mainly focus on scenarios with a fixed context length of 96. However, this is hardly the case in practice where one encounters varying series lengths. Moreover, for high frequency data (< 5min granularity), one would need context lengths of several thousand timesteps to correctly model seasonalities (e.g., weekly). In this case, the "token embedding" would grow dramatically which raises concerns about the generalizability of the method. I also have reservations considering this method as a type of tokenization for time series. Tokenization by its name implies something dynamic (a sequence), whereas this method just concatenates the embeddings into a single "token".
- The type of long-term forecasting benchmark considered in this work has been often criticized for its limitations. Please refer to the talk (and paper) from C. Bergmeir [1, 2] where he discusses the limitation of this benchmark and current evaluation practices. A recent position paper [3] also conducted a comprehensive evaluation of models on this benchmark showing that there's no obvious winner. Authors should consider using better benchmarks to demonstrate the effectiveness of their method. See, for example,
    - Chronos Benchmark II: This benchmark includes 27 datasets (42, if you include Benchmark I) providing a comprehensive coverage over domains, frequencies and other properties [4].
    - GIFT-Eval: This benchmark includes 90+ tasks across multiple datasets and domains. Please refer to https://github.com/SalesforceAIResearch/gift-eval.
    - The Monash Benchmark: https://forecastingdata.org/
    - or, the new released comprehensive fev-bench [5].

[1] https://neurips.cc/virtual/2024/workshop/84712#collapse108471            
[2] Hewamalage, Hansika, Klaus Ackermann, and Christoph Bergmeir. "Forecast evaluation for data scientists: common pitfalls and best practices." Data Mining and Knowledge Discovery 37.2 (2023): 788-832.            
[3] Brigato, Lorenzo, et al. "Position: There are no Champions in Long-Term Time Series Forecasting." arXiv preprint arXiv:2502.14045 (2025).            
[4] Ansari, Abdul Fatir, et al. "Chronos: Learning the language of time series." arXiv preprint arXiv:2403.07815 (2024).            
[5] Shchur, Oleksandr, et al. "fev-bench: A Realistic Benchmark for Time Series Forecasting." arXiv preprint arXiv:2509.26468 (2025).            

- Apart from the limitations of benchmark, there are also problems with the evaluation of baselines. Restricting context length to 96 does not ensure a "fair comparison". Some models work better with longer context lengths and, in practice, context length is not restricted and longer contexts are used whenever possible. A fair comparison would be to either provide longer contexts for model that work well with them or experiment with different context lengths and report the best results for each model.
- Authors integrate the proposed method into PatchTST but I don't understand what this means. PatchTST is a transformer-based sequence model. If the entire sequence is embedded into a single token, it does not remain a sequence model anymore.

Some discussion, which can be made more accurate.

> Previous patchwise tokenizers required overlap between neighbor patches to avoid semantic fragmentation caused

This is hardly the case. Most patch-based models (see the time series foundation model literature) use non-overlapping patches and it does not pose any problem.

> Notably, despite not using complex inter-variable relationship modeling modules like Attention or Transformer, PCMLP still excels in datasets with a large number of variables.

This just suggests that the benchmark datasets considered have limited multivariate structure to exploit, a phenomenon which has been discussed in the literature previously.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a new tokenization strategy for transformer-based time series forecasting models. There are 3 widely-used tokenization strategies: tokenizing every single timestep, tokenizing patches (initially used in PatchTST), and tokenizing the entire time series as a single patch (initially proposed in iTransformer), a.k.a the variable patching mechanism. This paper interpolates between the two extremes to propose PatchCat, where time series sub-sequences of equal length are mapped to hidden dimensions $d$, where $d$ increases as we move from the first to the last patch, linearly. All the embeddings are then concatenated. The authors demonstrate that their methods perform well on some widely used datasets.

### Strengths
I believe that the paper presents a intuitive and simple idea, which seems to work well on the compared datasets, against competitive baselines.

### Weaknesses
1. **Limited comparative scope.**
   The paper does not compare PatchCat against time series foundation models (e.g., Chronos-1 and 2, TimesFM, Tirex, MOMENT, etc.), which have explored a diversity of tokenization and patching strategies. Moreover, the authors state that:

> Direct comparison of the above three kinds of temporal tokenizers across existing works is confounded by the usage of heterogeneous backbones.

but do not provide any experiments to address this gap. Without such baselines and experiments, it is difficult to contextualize the contribution of PatchCat relative to the current state of the art. I would highly recommend that the authors (1) compare their methods to these state-of-the-art forecasting models, (2) compare a wider diversity of tokenization strategies (see [2] Table 9, for more details), and (3) compare their proposed patching strategy against a variety of backbones. 

2. **Dataset and evaluation limitations.**
   The datasets used (e.g., ETT, Exchange, Weather) are known to have limited diversity and issues. A few new benchmarks such as `GIFT-Eval` and `fev-bench` were released to partially address some of these gaps. I would highly recommend that the authors supplement their results with findings from these benchmarks. 

> Finally, we use PCMLP as a baseline to compare the performance of PatchCat vs. the variable-wise tokenizer.

Moreover, prior work such as [1] and Chronos-2 have also reported that multivariate modeling yields little or no benefit on these public datasets, which weakens the empirical claims around PatchCat’s improvements.

3. **Hyperparameter tuning transparency.**
   The paper lists search ranges for learning rate and hidden dimensions but does not specify the tuning procedure—e.g., whether tuning was performed on a held-out validation set or the test set. This raises concerns about potential overfitting or unfair comparisons.

4. **Interpretability of “semantic fragmentation.”**
   The paper states that previous patchwise tokenizers required overlapping patches “to avoid semantic fragmentation.” However, the notion of “semantic fragmentation” is not clearly defined or empirically demonstrated. It remains unclear how the proposed no-overlap design preserves or enhances temporal semantics.

5. **Marginal novelty of time encoding and dimension allocation strategies.**
   The introduced time encoding resembles mechanisms used in earlier architectures such as AutoFormer, and recent works have questioned the necessity of explicit time encoding. Similarly, the dimension allocation strategy is intriguing but not fully motivated or analyzed, particularly for periodic or long-range temporal dependencies. From Table 2, it is unclear if any of the differences are significant. 

### References
1. Żukowska, Nina, et al. "Towards long-context time series foundation models." arXiv preprint arXiv:2409.13530 (2024).
2. Goswami, Mononito, et al. "Moment: A family of open time-series foundation models." arXiv preprint arXiv:2402.03885 (2024).

### Questions
1. How exactly is *semantic fragmentation* defined and measured in this work? Can the authors clarify what empirical evidence supports the claim that overlap-free patching mitigates or avoids it?

2. How were hyperparameters tuned? Was there a distinct validation set, or was tuning performed directly on the test data? Clarifying this is critical for assessing experimental rigor.

3. Was the primary goal to isolate the effect of patch tokenization within a fixed base model (e.g., comparing PatchCat vs. variable-wise tokenization under identical architectures)? If so, it would be helpful to state this explicitly and explain why foundation model baselines were excluded.

4. Could the authors elaborate on why the proposed time encoding is expected to help, given that some recent studies (e.g., AutoFormer, TimesFM) have found temporal embeddings to have limited impact?

5. Do the authors anticipate that PatchCat would generalize beyond the benchmark datasets—especially to large-scale, real-world, irregularly sampled time series, or other modeling architectures, especially those used for foundation models?

### Soundness
2

### Presentation
3

### Contribution
3
