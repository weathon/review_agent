# Path-Consistency with Prefix Enhancement for Efficient Inference in LLM

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
To enhance the reasoning capabilities of large language models (LLMs), self-consistency has become a popular approach, combining multiple samplings with majority voting. However, current methods are computationally expensive and time-consuming due to the need for numerous samplings. To address this, this paper introduces path-consistency, which leverages the confidence of earlier-generated answers to identify the most promising prefix and guide the generation of subsequent branches. By dynamically guiding the generation of subsequent branches based on this prefix, path-consistency mitigates both the errors and redundancies from random or less useful sampling in self-consistency. This approach reduces errors and redundancies from random sampling, significantly accelerating inference by minimizing token consumption. Our extensive empirical results demonstrate that path-consistency improves inference latency by up to 40.5\%, while maintaining task accuracy across various tasks, including mathematical reasoning, commonsense reasoning, and symbolic reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes path-consistency, an improved variant of self-consistency that aggregates reliable prefixes across a small batch of completions to reduce the token cost during sampling. The authors show that this method can improve performance and reduce inference latency on several datasets using Llama3-8B.

### Strengths
**[S1]** The idea is reasonable. By extracting reliable prefixes, the proposed method reduces token consumption during self-consistency sampling, thus lowering inference time.

**[S2]** According to the provided experiments, the method improves performance while decreasing latency.

### Weaknesses
**[W1]** The paper lacks comparison with recent works; most of the cited references are outdated and do not include any from 2025.

**[W2]** The main experiments are conducted only on Llama3-8B, which raises concerns about robustness. The authors should extend their evaluation to other model families and sizes.

**[W3]** The derivation in Section 4.2 relies on too many assumptions, making the theoretical justification unconvincing.

**[W4]** The reported self-consistency accuracy on GSM8K (64.1% in Table 2) is significantly lower than the official Llama3 result (~80%). This discrepancy questions the soundness of the experimental setup.

**[W5]** The paper does not evaluate on more challenging reasoning datasets such as MATH500 or AIME, which are commonly used and better reflect the method’s robustness on difficult multi-step reasoning tasks.

**[W6]** The claim around lines 324–325 lacks empirical support. The authors should include an ablation study on the sampling path parameter.

**[W7]** As shown in Tables 5 and 6, the proposed method shows limited advantage compared to related methods.

**[W8, minor]** The captions of Figures 1 and 2 exceed the bottom page margin.

### Questions
**[Q1]** Could the authors clarify how inference latency is measured? This is critical for assessing the efficiency of the proposed method.

**[Q2]** The authors do not report latency metrics for evaluations using DeepSeek-V3 with Llama-3.2-1B-Instruct. Could they provide corresponding results?

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
This paper presents path-consistency, with the aim to reduce the token consumption or improve the efficiency of standard self-consistency. The key idea is to extract prefixes from early branches and use them to guide the subsequent generations. Experiments are conducted on three domains of datasets. The results show that the proposed method could significantly improve the efficiency while preserve the accuracy in most cases.

### Strengths
- The paper is generally well-written.
- The proposed path-consistency is reasonable.
- Extensive experiments are conducted to show the effectiveness of the proposed method.

### Weaknesses
- In the experiments, the authors show that there exists a proper confidence threshold at which path-consistency simultaneously enhances both task performance and efficiency. However, how to determine this threshold is unclear.

- The authors claim that the proposed method integrates seamlessly with existing optimization methods, achieving even better acceleration performance. However, only self-consitency is evaluated in the experiments.

- Related to the previous point, only self-consistency is evaluated as baselines in the experiments. How about the direct generation methods as well as inference time techniques. An accuracy-efficiency comparison among more related techniques is appreciated.

### Questions
- The proposed method largely relies on the correctness of the prefixs of earlier generated paths. Can the authors analyze the results to check how the correctness of the prefixs affect the results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Paper proposes path-consistency with prefix enhancement for LLMs to solve the high computational cost of self-consistency. It extracts "prefixes" from early-generated paths using answer confidence to guide subsequent branch generation, reducing errors and token waste without extra computation, fine-tuning, or model changes. Evaluated on 10 datasets, it boosts inference speed and also outperforms baselines.

### Strengths
1.The paper demonstrates originality through a creative reformulation of self-consistency.  It does not merely prune paths but repurposes high-confidence segments to steer inference.
2. The work exhibits high methodological and empirical quality. Methodologically, PC is rigorously designed. Empirically, the evaluation is comprehensive.
3. The paper is well-structured and accessible. The "extract-and-sample" process is clearly illustrated through pseudocode and a running example, making the workflow easy to follow.
4. This work addresses the high computational cost of self-consistency in LLM deployment. By making inference drastically more efficient without accuracy loss, path-consistency has immediate practical significance for real-world applications requiring reliable reasoning.

### Weaknesses
1.The paper relies heavily on the beta confidence criterion but provides minimal analysis of how sensitive the results are to this specific choice.
2.The theoretical analysis assumes only one dominant incorrect answer, which oversimplifies real-world scenarios where multiple plausible but wrong answers compete.
3.While arithmetic/reasoning tasks show impressive gains, there's no evaluation on generation-intensive tasks (e.g., long-form QA, summarization) where prefix reuse might have different effects.

### Questions
1. Paper mentions extracting "shorter prefixes from optimal paths" when confidence exceeds a threshold, but it does not define what constitutes a "step" or how steps are segmented from raw model outputs. Is it rule-based or data-driven? Does step segmentation vary across tasks?
2. What is the computational overhead of PC’s prefix extraction and confidence calculation? 
3. PC performs iterative prefix selection, not one-time selection. Does iterative selection increase the risk of amplifying early minority errors (e.g., a wrong prefix selected at Level-1 leading to more wrong paths at Level-2)?
4."randomly sample from the extracted prefixes as part of the prompt." What is the sampling strategy？

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
3

### Summary
The authors propose Path-Consistency (PC), an inference-time method that accelerates Self-Consistency (SC) decoding for LLMs without additional training or model changes.
PC repeatedly (i) generates a small window of reasoning paths, (ii) estimates answer confidence, (iii) extracts the common prefix of high-confidence paths, and (iv) re-uses that prefix to seed the next window.
Iterating this “extract-and-sample” loop shortens subsequent generations, cutting token cost and wall-clock latency while preserving (or slightly improving) accuracy on ten reasoning benchmarks.
Extensive experiments with Llama-3-8B show up to 48 % speed-up and −58 % token consumption vs. vanilla SC, with comparable or better accuracy on math, commonsense, and symbolic tasks. The method is model-agnostic and orthogonal to other SC optimizations.

### Strengths
1. Significant efficiency gains: Demonstrates 20–48 % latency reduction and 20–60 % fewer tokens across diverse datasets, verified on a consumer GPU.
2. No training or architecture change: Pure inference algorithm—easy to plug into any decoder-only LLM and complementary to distillation, early-exit, or speculative schemes.
3. Careful ablation & analysis: Authors vary confidence thresholds, prefix levels, model sizes, and datasets; provide theoretical justification that PC does not amplify “truth-in-the-hands-of-a-few” problem when base SC already works.

### Weaknesses
1. Limited novelty: Core idea—early-commitment to promising partial solutions—closely resembles existing beam/pruning and adaptive-consistency techniques; incremental contribution.
2. Heavily engineered hyper-parameters: Window size, #branches, confidence metric, prefix level schedule, and threshold all require per-task tuning; no automatic or adaptive schedule offered.
3. Evaluation restricted to 20-path SC: All speed/accuracy numbers compare against a 20-sample baseline; unclear how PC behaves with larger budgets or stronger teacher models where SC accuracy saturates.

### Questions
1. How does PC compare when SC uses 40, 100, or adaptive samples—does the relative speed-up shrink or the error rate rise?
2. Fig. 2 shows 25–50 % tokens wasted on wrong branches; does aggressively lengthening the prefix (level-4/5) ever mis-guide later samples and hurt accuracy?
3. The proof assumes Pvote ≥ 0.5; what bound can you give for harder datasets where SC itself is below 50 %, and how should practitioners set Cthreshold in that regime?

### Soundness
3

### Presentation
2

### Contribution
2
