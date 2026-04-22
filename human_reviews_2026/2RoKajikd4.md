# RL Beats SFT while Mitigating Definition Bias in LLM-based Information Extraction

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
While large language models (LLMs) have been able to provide generally reasonable answers to complex information extraction (IE) tasks through prompt engineering and supervised fine-tuning (SFT), their performance and safety remain limited. We propose a novel fuzzy matching method to reveal that this is largely due to the definition bias between the model and the dataset. To mitigate this problem without human intervention, we use Reinforcement Learning with Verifiable Rewards (RLVR) to train the model, enabling it to independently learn the inherent definition of the task from the dataset. Specifically, we use Group Relative Policy Optimization (GRPO) to train LLMs of varying parameter sizes, rewarded with micro F1 scores, and achieve notably higher precision and recall than SFT across all models. We then apply fuzzy matching again to statistically demonstrate that this improvement is mainly primarily to the mitigation of the definition bias between the model and the dataset.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Information extraction used to be an important task for NLP. In this work, the authors claim that current LLM's IE solutions via prompt engineering / SFT is limited in terms of both performance and safety. They hence introduced a fuzzy match method to loose the constraints of exact match and demonstrated that using this metric, RLVR w/ GRPO outperforms SFT, with notable F1, precision & recall increase across all models.

While in the experiments, RLVR w/ GRPO does outperform SFT across different datasets and models, this conclusion is of no surprise. I think the authors' contribution: (1) propose a simple fuzzy matching method, i.e. allow category mismatch & allow 1-word-exception exact match and (2) RL > SFT is plain, normal and well-known, without sufficient technical ideas or insights.

### Strengths
1. The writing is good, easy to read.

### Weaknesses
1. While the authors claim that previous methods would cause privacy leakage or unnecessary information loss, their evidence is not convincing enough.
(1) privacy leakage: it should be avoided before the IE process,
(2) information loss: it can be reflected on the metrics, however, the authors do not include any baselines in their experiments. Further, the sota F1 reported in DWIE on NER with Joint + BERT is 87.7, higher compared to Qwen3-0.6B in this work (87.49). The lack of comparison with other methods would raise the concern about the effectiveness and contribution of the proposed method.
2. The contribution of this work can be summarized into two points: (1) propose a simple fuzzy matching method, i.e. allow category mismatch & allow 1-word-exception exact match, and (2) demonstrate RL > SFT, both of which, from my perspectives, are plain, normal and well-known, without sufficient technical ideas or insights.

### Questions
None

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper argues that SFT in LLM-based information extraction often inherits human-defined biases—rules and conventions embedded in prompts or annotations—causing models to miss the implicit definitions present in datasets. The authors term this gap definition bias and propose mitigating it through RLVR, using GRPO with micro-F1 rewards. A fuzzy-matching evaluation quantifies the bias and shows that RL consistently outperforms SFT across datasets, with most improvements stemming from reduced definition bias rather than memorization.

### Strengths
1. The paper has a clear problem definition and motivation. The concept of definition bias is intuitively introduced and visually explained (Figure 1). The paper articulates well why standard SFT cannot eliminate such bias because it introduces human-defined prompt bias rather than learning dataset-implied rules.
2. The experiments span multiple model sizes (0.6B–8B) and datasets (NER + EE).  The synthetic dataset experiment is particularly convincing. It clearly demonstrates that RL can learn implicit rules without human intervention.

### Weaknesses
1. The paper has Limited Novelty in Methodology.The paper primarily applies existing RLVR techniques (notably GRPO and micro-F1 rewards) to IE tasks, without introducing substantial algorithmic innovations or methodological insights. Similar RLVR IE frameworks have been explored in prior work [1][2], yet these are not sufficiently cited or contrasted. 
2. Although RL slightly improves F1 scores, the overall gains (≈2–5%) may not justify the heavy computational cost of RL compared to SFT. Moreover, the approach is confined to entity and event extraction, leaving it unclear whether the method extends to more complex reasoning or relation extraction scenarios.

[1]Li, Ran, et al. "Beyond path selection: Better LLMs for Scientific Information Extraction with MimicSFT and Relevance and Rule-induced (R $^ 2$) GRPO." 
[2]Dai, Runpeng, et al. "R1-re: Cross-domain relation extraction with rlvr."

### Questions
NA

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
4

### Summary
The paper studies “definition bias” in IE—mismatch between a model’s implicit task understanding and a dataset’s labeling conventions—and proposes (i) a fuzzy-matching evaluation to estimate the share of errors attributable to definition bias and (ii) an RL with verifiable rewards (RLVR) training that directly optimizes micro-F1 via GRPO. On DWIE/DocRED (plus smaller WikiNEuRal/DocEE and a synthetic variant), RL consistently improves precision/recall/F1 over SFT for multiple Qwen3 sizes, and a post-hoc fuzzy analysis attributes a substantial portion of the gains to reduced definition bias. The training stack includes a brief “format learning” SFT warm-up, then GRPO with group size G=8, and evaluation with micro-F1. Experiments run on up to 4×A100.

### Strengths
1) Clear problem framing: “definition bias” is a practical failure mode for IE; the fuzzy-matching lens is a useful way to quantify its impact. 

2) Simple, reproducible recipe: brief SFT for format control + GRPO with micro-F1 reward; the ablation on group size (G) is helpful. 

3) Consistent gains: RL improves F1 across Qwen3-0.6B/1.7B/8B on DWIE/DocRED; case studies illustrate fixes in class choice and span length. 

4) Attribution analysis: the second fuzzy-matching pass that explains where RL gains come from (relabeling/class-boundary/span tweaks) increases interpretability. 

5) Synthetic-rule test: shows RL can internalize implicit labeling rules better than SFT on several keyworded constraints.

### Weaknesses
1) Task scope: main story is document-level NER/entity extraction; relation extraction and more complex IE pipelines are left for future work, so broader generality remains uncertain. 

2) Reward design granularity: micro-F1 is assigned as a single scalar advantage per response; the paper notes token-specific credit assignment trials underperform, but deeper alternatives (span-level or structure-aware credit) are not explored. 

3) Fuzzy matching validity: while practical, the relaxation choices (class-permissiveness and ±k token span tolerance) embed assumptions; it would help to test multiple fuzzy schemes (or human audit) to bound over-attribution to “bias mitigation”. 

4) Limited methodological novelty: the core modeling advance is applying RLVR to IE. The fuzzy-matching component is valuable but primarily an evaluation/attribution tool rather than a modeling innovation.

### Questions
1) Can you report results on the canonical/full DWIE/DocRED splits (or provide calibration that trends hold when scaling sample sizes)? 

2) Any evidence that bias mitigation carries over to relation extraction or event arguments within the same corpora? 

3) For the fuzzy analysis, can you include a small human audit to bound over-attribution?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses "definition bias" in LLM-based information extraction. The authors propose using RL with Verifiable Rewards, specifically GRPO with micro F1 score as the reward, to enable models to learn extraction rules without human prompts. Authors introduce a novel "fuzzy matching" evaluation method to quantify definition bias by relaxing category and boundary constraints. Experiments on diverse Qwen3 models across DWIE and DocRED datasets show RL consistently outperforms SFT.

### Strengths
- The fuzzy matching approach is creative and provides an interpretable quantification of definition bias.
- RL improvements are consistent across all model sizes and datasets.
- The paper acknowledges negative results and discusses failure modes.

### Weaknesses
- The core contribution is applying existing RL techniques to IE tasks. While the application is novel, the technical contribution is incremental.
- The paper only compares RL vs. SFT. What about both SFT and RL?

### Questions
- A detailed training computational cost comparison should be provided.
- In Lines 480-481, the authors mention that "While we have also tried to assign token-specific advantages for finer granularity, the performance of the resulting model actually decreases". Could the authors provide more details on this part?

### Soundness
2

### Presentation
2

### Contribution
2
