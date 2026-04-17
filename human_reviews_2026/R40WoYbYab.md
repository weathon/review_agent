# Task-Aware Data Selection via Proxy-Label Enhanced Distribution Matching for LLM Finetuning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4, 4

## Abstract
Task-specific fine-tuning of foundation models is critically dependent on the quality and relevance of the instruction data. While prevailing data selection methods rely exclusively on instruction instances X to approximate the target distribution, we argue that selection should align with the joint distribution of instructions and task-specific labels (X,Y). However, task-specific labels Y are typically unavailable in practice. To address this, we reformulate the task-specific data selection problem and present a novel pipeline that leverages the reasoning capabilities of large language models (LLMs) to infer proxy labels, thereby facilitating joint distribution alignment. Our approach begins by propagating proxy labels from a small target set to a large, unlabeled source corpus. A two-stage filtering process then removes instances with label noise and refines the subset through distribution alignment. This strategy produces more semantically meaningful and task-aware selections than conventional similarity measures based on $X$ alone. Experimental results show that fine-tuning on a subset of only 10K samples, selected from a pool of 300K, achieves performance competitive or superior to state-of-the-art methods. Code is available at https://github.com/tmlr-group/TADS.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a task-aware data selection pipeline for fine-tuning llms. The approach focuses on aligning both input features and task-specific labels to improve the relevance and quality of selected instruction data. Since task-specific labels are often unavailable, the method uses LLMs to generate proxy labels for the target dataset, which are clustered and propagated to the source dataset. Then a two-stage selection process first filters out low-quality examples using llm-based scoring, and then matches the label distribution through incremental selection.

### Strengths
- The paper extends prior work by considering additional alignment dimensions during task-specific data selection, including task, topic, style, and audience.
- It offers an information-theoretic explanation that provides a principled understanding of the proposed data selection method and prior works.

### Weaknesses
- The method relies heavily on LLM-based judgment, but does not evaluate the robustness or reliability. It remains unclear how accurate the generated labels are and how consistent or calibrated the LLM-assigned quality scores are.
- The approach introduces several hyper-parameters and control knobs (e.g., k in k-means clustering, minimum score thresholds, label alignment choices) without providing clear guidance on how to tune them. According to the experiments, the results are sensitive to them.
- The paper does not provide any theoretical guarantees for the proposed distribution alignment algorithm. It is unclear whether this algorithm will converge or output a better match.
- The experimental results lack error bars, making it difficult to assess statistical significance or robustness of the reported improvements.
- Minor presentation issues: in Figure 2, the text and numbers are too small and overlap, which affects readability.

### Questions
- Why you choose 100 as the number of centroids in k-means clustering?
- In label propagation, how exactly are the source examples embedded? Are they also labelled by the llm using the same process as the target examples?
- In "Prompt Template for Scoring Source Samples", the connection between the scoring instructions and the provided labels is unclear. What purpose do the labels serve in this context?
- Can we jointly match multiple labels using this method?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles the problem of selecting high‐quality, task‐relevant instruction data for fine-tuning LLMs. The authors argue that existing data-selection methods focus only on aligning the input distribution X (i.e., instructions) with a target task, but neglect the joint distribution of (X,Y) where Y are task‐specific labels, which are often unavailable in practice. They propose a pipeline that uses an LLM to infer proxy labels for a large unlabeled source corpus, then applies a proxy-label enhanced distribution matching method: first filtering out noisy out-of-distribution samples, then aligning the remaining data to the target joint distribution (X,Y), and finally selecting a subset. Experiments show that fine-tuning on the selected subset can achieve performance competitive with or superior to using full dataset, thereby demonstrating task‐aware data selection is effective.

### Strengths
Novel viewpoint: Using proxy labels and distribution matching for task‐aware rather than input‐only data selection is an interesting insight.

Practical relevance: Demonstrating that a small subset of data can yield competitive fine‐tuning results addresses the real challenge of data efficiency in LLM tuning.

Clear presentation of the pipeline and motivation, making the method relatively easy to understand and adopt.

### Weaknesses
Proxy labels may introduce noise, and the paper gives limited analysis of how label quality affects downstream performance.

Transparency of cost/efficiency: While the claim of “smaller subset yields full‐data performance” is compelling, more detailed breakdowns (hardware, runtime, selection cost) would improve trust.

Risk of selection bias: Since the method selects based on proxy‐label generated metrics and distribution matching, it may inadvertently favour certain types of samples (e.g., easier ones, more model‐familiar) and perhaps neglect rare or hard tasks; the paper does not deeply analyse this risk.

Engineering complexity & scalability: Generating proxy labels, filtering, and distribution matching add overhead; discussion of how this scales or works in resource‐limited environments is limited.

### Questions
Can you report detailed metrics on proxy label quality: e.g., accuracy, noise rate, and how selection performance degrades (or improves) with differing label quality?

How robust is the method to different model architectures or sizes? If the fine-tune target model is quite different (size, family) from the one used to infer proxy labels, how does performance change?

Could you provide full cost breakdowns (selection cost + fine-tune cost + hardware) for your method and key baselines (input‐only selection, random sampling) under identical hardware?

Have you analysed the selected subset in terms of diversity: task types, difficulty levels, rare vs common categories, language styles? Is there any systematic bias in what gets selected vs discarded?

In truly low-data regimes (e.g., 1 K or 5 K samples) or for very niche tasks (domain‐specific), how does your method perform relative to full‐data or random sampling?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper introduces a proxy-label enhanced joint distribution matching approach for task-specific data selection in large language model fine-tuning. The key idea is to let the model generate task-related proxy labels so that both inputs and outputs are considered jointly when aligning distributions, rather than focusing only on input similarity. 

As a researcher specializing in human-computer interaction, this study clearly falls outside my area of expertise. Following the Area Chair’s instructions, I have selected “1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers” and submitted my review accordingly. Therefore, I will not be participating in the rebuttal stage for this manuscript. Thank you for your understanding.

### Strengths
1. The paper reformulates task-specific data selection as a joint distribution alignment problem, moving beyond traditional input-only approaches. The introduction of proxy labels adds a fresh perspective to modeling task relevance.
    
2. It proposes a complete and coherent pipeline, from proxy-label generation and clustering to noise filtering and incremental sampling, with clear logical flow and information-theoretic grounding.
    
3. Experimental results on multiple mainstream benchmarks, such as MMLU, TruthfulQA, and GSM8K, show stable or superior performance compared with SOTA methods like LESS and TSDS, especially under low-data conditions.

### Weaknesses
1. Although the paper uses LLMs to generate proxy labels, the analysis of their consistency, bias, and noise propagation is rather superficial and lacks quantitative evaluation or comparison with human annotations.
    
2. The multi-stage pipeline, involving annotation, clustering, filtering, and sampling, lacks detailed efficiency analysis on large-scale corpora. Its scalability and real-world deployment cost remain unclear.
    
3. The experiments are limited to English and general-purpose LLMs like LLaMA and Mistral. There is little discussion on adaptation to multilingual or multimodal tasks, and the explanation for performance drop on TyDiQA is vague.

### Questions
1. Can the authors provide an evaluation of the consistency or confidence of LLM-generated proxy labels compared with human annotations to verify label quality?
    
2. Can the proposed method generalize to cross-domain or cross-lingual scenarios, such as transferring from legal to medical tasks? Would new proxy labels be required in such cases?
    
3. Are the hyperparameters and target set sizes for LESS and TSDS exactly matched to those used in this paper? If not, this should be clearly stated to ensure fair comparison.
    
4. Have the authors analyzed the sensitivity of key parameters, such as the minimum score threshold or the number of clusters k? Without this, reproducibility and transferability could be limited.
    
5. The ablation only examines the combined effect of filtering and sampling. It would be helpful to further analyze how each stage contributes to different task types, such as reasoning, factual, or comprehension tasks.

### Soundness
2

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
4

### Summary
This paper proposes a proxy-label-based data selection method for instruction-tuning LLMs, aiming to select source data that best matches the target task by jointly considering instruction text and task-semantic proxy labels. The method targets limitations of prior work that only align input distributions without task semantics. Experiments across multiple benchmarks show improvements, though the gains vary by semantic field and threshold settings.

### Strengths
1. An interesting idea of jointly aligning instructional text and task-semantic proxy labels.
2. Comprehensive experimental coverage across multiple benchmarks and semantic dimensions, demonstrating systematic evaluation of the proposed approach.

### Weaknesses
1. The performance gains are inconsistent and not uniformly strong across benchmarks (Table 3). There is no single configuration that consistently outperforms others: for example, min-score ≥7 achieves two SOTA results, and min-score ≥6 also yields two SOTA results. Additionally, different semantic fields produce varying best configurations (e.g., TruthfulQA prefers “audience” under min-score ≥6 but “style” under ≥7), making it unclear how to select the semantic field and threshold in a principled manner. The authors also acknowledge inconsistent alignment effectiveness across fields (row 421), reinforcing this concern.
2. Important retrieval-style baselines such as representation-based RDS [1,2] and BM25 are missing, making it difficult to assess how much benefit comes from semantic distribution matching versus standard retrieval approaches.
3. The approach is modular and largely post-hoc rather than jointly optimized, which may limit conceptual novelty. The contribution appears to lie more in the combination of existing components.

[1] Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 586–595, 2018.

[2] Ivison, H., Zhang, M., Brahman, F., Koh, P. W., & Dasigi, P. (2025). *Large-Scale Data Selection for Instruction Tuning*. arXiv preprint arXiv:2503.01807.

### Questions
1. In Table 5, why does removing OOD filtering produce a very large drop on TruthfulQA.
2. Why is a separate OOD-filtering step required? Since Step 2 already computes similarity for anchor propagation, could OOD samples be filtered via a similarity threshold rather than a second LLM-based scoring step?
3. As a baseline or further exploration, what would happen if semantic-field information were integrated into existing data-selection approaches (e.g., adding semantic attributes to gradient-based LESS or representation-based RDS)? Would this mitigate the issue raised in rows 49-53 and unify the benefits without the need for proxy labeling and field-wise tuning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper reformulates task-specific data selection for LLM finetuning, arguing that prevailing methods, which only align the distribution of inputs X, are insufficient. The authors' central claim is that selection must instead align the joint distribution of inputs and labels $(X, Y)$ to capture true task relevance.

To achieve this, the paper proposes a novel four-stage pipeline that uses LLM-generated "proxy labels" since true labels are unavailable. 

Experiments show that finetuning a LLaMA-3.1-8B model on a 10K subset selected with this method achieves performance competitive with or superior to state-of-the-art baselines and a model trained on the full 300K-sample pool.

### Strengths
1. This paper argues that task-specific data selection should not be based on aligning inputs ($X$) alone, which is the common practice, but on aligning the joint distribution of inputs and labels ($X, Y$). This is a more accurate and semantically meaningful way to define task relevance.
2. The paper introduces a novel four-stage pipeline that operationalizes its new formulation. Since target labels ($Y$) are typically unavailable, it uses an LLM to generate structured "proxy labels" (Task, Topic, Style, Audience). This provides a concrete and practical solution to the challenge of joint distribution matching.

### Weaknesses
1. The proposed 4-step pipeline is highly complex. It requires two distinct LLM-based steps (proxy-label generation and OOD scoring), an embedding model, k-means clustering, and an incremental sampling algorithm. This complexity introduces numerous hyperparameters that are not thoroughly justified, such as the number of clusters ($k=100$), the OOD score threshold, and the choice of which label field to align (Task, Topic, Style, or Audience), suggesting the method requires extensive, task-specific tuning to work well.
2. The ablation study in Table 5 does not adequately isolate the core contribution of the paper. The paper's main claim is that aligning the joint distribution $P(X, Y)$ is superior to aligning the marginal distribution $P(X)$. However, the ablation study only compares the full pipeline against removing its own components (OOD filtering or incremental sampling). A crucial missing baseline would be to apply the exact same clustering and incremental sampling algorithm (Steps 2 and 4) directly to the input embeddings ($X$) instead of the proxy-label embeddings ($Y$). Without this direct comparison, it is unclear if the performance gains come from the novel $P(X, Y)$ alignment or simply from the clustering/sampling algorithm itself.

### Questions
1. Your results in Table 3 demonstrate that the choice of which proxy label to align (e.g., "Align_task", "Align_topic", "Align_style") is a critical hyperparameter, as the best-performing field changes for each benchmark. For a practitioner applying your method to a new task, how would you recommend they determine the optimal field to align? Does this not require them to run multiple full finetuning experiments for each field, which would undermine the method's goal of data efficiency?
2. Your core claim is that aligning the joint distribution $P(X, Y)$ is superior to aligning the marginal input distribution $P(X)$. However, your ablation study in Table 5 only compares your full pipeline against versions with its own components (OOD filtering or incremental sampling) removed. To truly isolate the benefit of using proxy labels ($Y$), could you provide results for a baseline that applies your exact same pipeline (clustering, OOD filtering, and incremental sampling) but operates directly on the input instruction embeddings ($X$) instead of the proxy-label embeddings ($Y$)?

### Soundness
2

### Presentation
3

### Contribution
2
