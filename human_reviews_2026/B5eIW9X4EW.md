# STT-LLM: Structural-Temporal Tokenization for Adapting LLMs to Longitudinal Profiles

- Decision: Reject
- Scores: 2, 2, 6

## Abstract
Large Language Models have shown strong generalization across natural language tasks but remain underexplored for longitudinal biomedical profiles. In sports, biological profiles are analyzed for doping, with particular emphasis on two key challenges for longitudinal data: (i) sequence prediction for early detection of prohibited substance use, and (ii) anomaly detection for identifying doping-related deviations. We propose STT-LLM, a structural-temporal tokenization framework that adapts LLMs to longitudinal analysis without modifying the backbone architecture. STT-LLM constructs joint embeddings that capture both temporal dynamics and biological pathway-based interactions, which are then transformed into LLM-compatible tokens through the specialized structural and temporal tokenizers. We evaluate our approach on real-world longitudinal steroid datasets from athletes, where STT-LLM consistently outperforms LLM baselines. In addition, we present a case study where STT-LLM provides contextual reasoning that aligns more closely with expert assessments compared to baseline models. These results highlight the effectiveness of embedding-guided tokenization for adapting LLMs to understand longitudinal biological data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces STT-LLM, a structural-temporal tokenization framework that adapts frozen large language models (with LoRA adapters) to longitudinal biomedical profiles by turning pathway-aware, time-dependent measurements into LLM-compatible tokens. It builds joint embeddings that capture metabolite interaction structure (via Laplacian eigenvectors) and temporal dynamics (via attention with positional encodings), then feeds these through dedicated structural and temporal tokenizers whose outputs are concatenated with the standard text prompt before LLM inference. The method targets two tasks in anti-doping analytics—sequence prediction for early detection and anomaly detection—and is evaluated on four real-world longitudinal steroid datasets (male/female and limited-sample variants). Compared to several 7–8B LLM baselines fine-tuned with their native tokenization, STT-LLM reports lower forecasting errors and higher anomaly-detection sensitivity/AUC in zero- and few-shot settings, with ablations indicating that both the structural and temporal tokenizers contribute materially. A small case study with expert-verified profiles further suggests improved contextual reasoning and faster inference under the same hardware budget.

### Strengths
The paper’s strengths are primarily methodological and integrative: it offers a clear, modular way to turn structured, time-varying biomedical profiles into tokens that a frozen LLM can use, combining pathway-aware structural embeddings with temporal encodings rather than forcing everything through plain text. On originality, the structural-temporal tokenization bridges graph-informed representations and sequence modeling inside an LLM framework with lightweight adapters, which is a nontrivial combination for longitudinal settings. Quality is supported by a reasonably principled construction (Laplacian-based structure, attention for time), an end-to-end pipeline that slots into off-the-shelf 7–8B models, and ablations indicating both structural and temporal components matter. Clarity is good: the data-to-token flow and how tokens are concatenated with the prompt are explained in a way that seems straightforward to reimplement. In terms of significance, demonstrating zero-/few-shot gains on real anti-doping datasets suggests practical promise for early detection and anomaly screening, and the design appears portable to other longitudinal biomedical profiles beyond the specific case study.

### Weaknesses
The evaluation is narrow and domain-specific, making it hard to judge generality: results are limited to four anti-doping datasets with closely related measurement spaces, few seeds, and primarily LLM baselines rather than strong time-series or graph-temporal models, so it’s unclear whether the gains come from the tokenization itself or from weaker comparators; adding competitive baselines (e.g., modern TS transformers and graph-temporal forecasters) and more seeds would strengthen claims. The construction choices need deeper justification and sensitivity: how many Laplacian eigenvectors are used, how pathway graphs are defined and updated over time, how token lengths scale with visit count, and how robust performance is to noisy or missing edges; reporting these ablations alongside compute/memory overhead would clarify practicality. Finally, clinical utility remains speculative: metrics tied to early-warning lead time, false-alarm rates, and cross-lab generalization, plus external validation and clear guidelines for privacy handling of longitudinal health data, would make the case more convincing.

### Questions
- Evaluation scope and baselines: could you broaden beyond only 7–8B LLM comparators to include strong time-series and graph-temporal baselines (e.g., purpose-built TS transformers and GNNs for irregular clinical series) and run more seeds with confidence intervals?

- Design choices and sensitivity: can you detail and ablate key tokenization decisions (how many Laplacian eigenvectors; how pathway graphs are defined/validated; how token length scales with visits; handling of missing/noisy edges), and report compute/memory overhead of S/T tokenizers and LoRA under increasing sequence lengths?

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
This paper proposes STT-LLM, a novel framework for adapting pre-trained Large Language Models (LLMs) to analyze longitudinal biomedical data, which is often numerical, irregularly sampled, and possesses complex underlying dependencies. The core contribution is a specialized tokenization strategy that processes structural and temporal information in parallel. A structural tokenizer encodes domain knowledge (e.g., metabolic pathways) from a feature interaction graph, while a temporal tokenizer captures the time-series dynamics. These specialized embeddings are then projected into the LLM's native token space, enabling a frozen LLM (fine-tuned with LoRA) to perform tasks on this data modality. The authors demonstrate their method on the specific use case of detecting doping in athletes' longitudinal steroid profiles, showing improved performance over standard LLM baselines in sequence prediction and anomaly detection.

### Strengths
Relevant Problem: The paper tackles the important and challenging problem of adapting the powerful capabilities of LLMs to complex, structured, and numerical time-series data, a domain where these models are not natively suited.

Intuitive Architecture: The proposed architecture is well-motivated and clearly presented. The idea of creating separate, specialized information streams for structural and temporal features before projecting them for the LLM is a reasonable and intuitive approach. The model diagram in Figure 1 is particularly effective.

### Weaknesses
Fundamentally Flawed Experimental Design: The paper's central claim that its tokenization strategy is superior is not scientifically validated by its experiments. The evaluation compares STT-LLM (using an unstated LLM backbone with the proposed tokenizer) against other LLMs (e.g., Llama-3) using their native text tokenizers. This is a confounded comparison (Backbone_A + Tokenizer_STT vs. Backbone_B + Tokenizer_Text). Any observed performance difference could be due to the choice of backbone model rather than the tokenization strategy. The paper fails to perform the essential apples-to-apples comparison needed to isolate the effect of its core contribution.

Critically Insufficient Baselines: The experimental evaluation lacks the necessary context to judge the method's practical value.

No Simple Baselines: The paper omits standard statistical (e.g., ARIMA) or simple ML baselines (e.g., linear models). Without these, it's impossible to know if the proposed complex LLM-based approach offers any real advantage over trivial or well-established methods.

No Specialized SOTA Baselines: For tasks like multivariate time-series forecasting and anomaly detection, there exist highly specialized and powerful models (e.g., graph-based Transformers). The paper avoids comparing against these likely superior models, justifying this by their incompatibility with "LLM inference pipelines," which is not a sufficient reason to exclude them from a performance benchmark.

No Relevant Foundation Model Baselines: The comparison overlooks powerful foundation models designed for numerical tabular data that have been successfully applied to time-series, such as TabPFN. This is a highly relevant baseline that operates on numerical data directly and would provide a much stronger point of comparison.

Lack of Transparency and Reproducibility:

The specific LLM backbone used for the proposed STT-LLM model is never stated, making the work impossible to reproduce or fairly evaluate.

The exact text serialization format used to feed the longitudinal profiles to the baseline LLMs is not provided, preventing an assessment of whether the baselines were configured fairly.

Several key results, including the main ablation study in Table 4 and the few-shot sequence prediction results in Table 2, are presented without error bars or standard deviations, making it impossible to assess the statistical significance of the reported gains.

Overstated Claims and Limited Scope:

The performance improvements on the sequence prediction task are marginal (often <1% in RMSE) and likely not statistically or clinically significant.

The paper makes broad claims about "longitudinal biomedical profiles," but the evaluation is confined to a single, niche application (doping detection). This lack of diversity in tasks and datasets fails to support the claims of general applicability.

### Questions
Experimental Design: Could you please clarify which LLM backbone was used for the STT-LLM model? To properly validate your core contribution, would it be possible to provide results from an apples-to-apples comparison, for instance, by evaluating Llama-3 with your STT tokenizer against Llama-3 with text flattening?

Reproducibility: Could you provide the exact text serialization format used to present the longitudinal data to the baseline LLMs? Furthermore, could you add standard deviations or error bars to the results in Tables 2 and 4 to allow for an assessment of significance?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The framework introduces a structural–temporal embedding pipeline combining spectral graph features from the normalized Laplacian with transformer-style attention and positional encodings, producing unified embeddings for longitudinal profiles. Two specialized tokenizers map these embeddings into the LLM's token space, enabling adaptation via LoRA while preserving the frozen backbone. The approach targets sequence prediction and anomaly detection (local sample-level and global profile-level) in anti-doping monitoring, with evaluation across four real-world steroid datasets. Results show consistent improvements over LLM baselines (Qwen-2.5, Falcon-3, Mistral, LLaMA-2/3.1, Phi-4, DeepSeek-R1) across RMSE/MAE/MAPE and detection metrics, supported by ablations demonstrating the necessity of both tokenizers and the embedding layer.

### Strengths
The paper clearly frames the core challenge - LLMs' discrete tokenization conflicts with multivariate, irregular, structurally linked longitudinal data - and positions tokenization as the adaptation mechanism rather than requiring architectural redesign. The modular combination of graph-based structural embeddings, attention-based temporal embeddings, dual tokenizers, and LoRA proves computationally efficient and compatible with general LLMs. 

Empirical evaluation spans zero-shot and few-shot performance across four datasets with realistic data scarcity and irregularity, covering both prediction and detection tasks while enforcing strict domain constraints on specificity. Comprehensive ablations and hyper-parameter studies validate design choices, while the expert case study and UMAP visualization (Fig 4) provide interpretability and practical relevance.

### Weaknesses
The baseline comparisons exclude strong non-LLM alternatives such as GNNs over metabolic graphs or purpose-built irregular time-series transformers and NeuralODE models, limiting the scope of performance claims to LLM-only comparisons. 

Several absolute metrics lack contextualization and details (Table 2) of how they are defined and computed. Local anomaly sensitivity remains low in zero-shot despite improvements (Table 3). 

The counterintuitive increase in error with more shots for many models suggests that the model struggles with more structural complexity and longer temporal dependencies (Table 2).

While the design is general, experiments focus exclusively on steroid modules; additional biomedical longitudinal domains would strengthen the claims.

### Questions
* Can you include comparisons to at least one non-LLM baselines (GNNs over metabolic graphs, NODE)?
* What are the units and scales of metabolites and targets? Can you provide normalization details and per-metabolite error decomposition?
* How are the few-shot examples selected?
* What token count do structural/temporal tokenizers add for typical profiles, and how does this scale with longer histories?
* Have you tested transfer to other longitudinal clinical datasets without redesigning graphs? How sensitive is performance to graph misspecification?

### Soundness
2

### Presentation
2

### Contribution
2
