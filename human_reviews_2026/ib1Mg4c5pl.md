# Binary Diff Summarization using Large Language Models

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Security of software supply chains is necessary to ensure that software updates do not contain maliciously injected code or introduce vulnerabilities that may compromise the integrity of critical infrastructure.
Verifying the integrity of software updates involves binary differential analysis (binary diffing) to highlight the changes between two binary versions by incorporating binary analysis and reverse engineering. 
Large language models (LLMs) have been applied to binary analysis to augment traditional tools by producing natural language summaries that cybersecurity experts can grasp for further analysis. 
Combining LLM-based binary code summarization with binary diffing can improve the LLM's focus on critical changes and enable complex tasks such as automated malware detection.
To address this, we propose a novel framework for binary diff summarization using LLMs. We introduce a novel *functional sensitivity score* (FSS) that helps with automated triage of sensitive binary functions for downstream detection tasks. We create a *software supply chain security* benchmark by injecting 3 different malware into 6 open-source projects which generates 104 versions, 392 binary diffs, and 46,023 functions. On this, our framework achieves a precision of 0.95 and recall of 0.71 for malware detection, displaying high accuracy with low false positives. We outperform an existing industry-style rule-based baselines by $\approx 4\times$ higher recall on malware detection while maintaining high precision. Across malicious and benign functions, we achieve FSS separation of 3.0 points, confirming that FSS categorization can classify sensitive functions. We conduct a case study on the real-world XZ utils supply chain attack; our framework correctly detects the injected backdoor functions with high FSS.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the challenge of detecting malicious or vulnerable changes in software supply chain updates by proposing a binary diff summarization framework using LLMs. The framework combines traditional binary diffing with natural language summarization and introduces the Functional Sensitivity Score (FSS), a metric inspired by CVSS, to triage binary functions according to sensitivity. The paper creates a benchmark of six open-source projects injected with three types of malware, totaling 104 versions, 392 binary diffs, and over 46K functions. Experiment results show high precision and reasonable recall in malware detection, with clear separation between benign and malicious functions.

### Strengths
1. Introduces an integration of binary diffing, LLM summarization, and FSS scoring, enabling triage of sensitive code changes.
2. Builds a realistic dataset by injecting ransomware, RAT, and botnet malware into six widely used open-source projects.

### Weaknesses
1. The paper offers limited technical novelty.
2. The experiments are not comprehensive.

### Questions
This paper introduces an interesting pipeline, but in its current form it is not strong enough for acceptance.

1. The overall approach (diff → LLM summary → score) is primarily a pragmatic composition rather than a new technical contribution. Since FSS is directly adapted from CVSS with hand-tuned weights and the final malicious/benign decision is delegated to another LLM, the novelty is insufficient. 

2. There are many relevant baselines (e.g., those in Appendix A.1 such as DeepBinDiff). How does the proposed method compare against these in terms of accuracy and efficiency? In addition, how do different prompting strategies (e.g., zero-shot vs. chain-of-thought vs. few-shot) or reasoning-optimized/domain-specific models affect performance?

3. There are several existing datasets, such as those from DeepBinDiff or BinSimDB[1], etc. They could provide stronger external validation. Why were these not used, and how does the proposed dataset in this paper differ from or improve upon them? And how's the performance of the proposed framework on those datasets?

4. The framework does not clearly define a threshold for distinguishing malicious from benign using the FSS. For a deployable tool, a principled thresholding or calibration strategy seems necessary. 

5. The results show significant variance in recall across models and programs. Could the paper provide a detailed breakdown of failure cases to illustrate where and why the framework fails?

Reference:

[1] Zuo, Fei, Cody Tompkins, Qiang Zeng, Lannan Luo, Yung Ryn Choe, and Junghwan Rhee. "BinSimDB: Benchmark Dataset Construction for Fine-Grained Binary Code Similarity Analysis." In International Conference on Security and Privacy in Communication Systems, pp. 203-225. Cham: Springer Nature Switzerland, 2024.

### Soundness
1

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
2

### Summary
The author(s) propose a novel framework that combines LLM-based binary code summarization with binary diffing to enhance the model's ability to focus on critical code changes and support complex tasks such as automated malware detection.

### Strengths
They create a software supply chain security benchmark by injecting 3 different malware into 6 open-source projects which generate 104 binary versions, 392 binary diffs, and 46,023 functions.

### Weaknesses
1. The methodological justification requires more clarity. It would be beneficial if the authors could explain why the proposed combination of LLM summarization and binary diffing offers advantages over existing binary analysis or summarization approaches.
2. The experimental section needs more comparisons with related baselines.
3. The paper would benefit from a more detailed discussion of the hyperparameter settings and ablation study used during the framework and evaluation.

### Questions
see Weaknesses.

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
The paper proposes a framework for binary diff summarization using LLMs. Given two binary versions, the system uses Ghidriff to extract only the added, deleted, and modified functions, constructs a diff callgraph, and then processes functions in reverse BFS order so that each function's LLM prompt can include summaries of its callees. On top of the summaries, the paper introduces a Functional Sensitivity Score, inspired by CVSS, with five categories
$B,R,C,I,A \in \{\text{none},\text{low},\text{medium},\text{high}\},$
aggregated as
$S = 1 - (1 - B)(1 - R), \quad M = 1 - (1 - C)(1 - I)(1 - A),$
$\text{FSS} =
\begin{cases}
\text{roundup}(5.3S + 6.1M), & M > 0,\\
0, & \text{otherwise.}
\end{cases}$
Then the top-$k$ highest-FSS functions are fed to the LLM again to classify the whole diff as MALICIOUS or BENIGN. The authors also build a supply-chain benchmark (6 OSS projects, 104 versions, 392 diffs, 46023 functions, 3 malware families). With GPT-5 mini and $k=5$ + changelog, they report precision 0.98 and recall 0.64.

### Strengths
Originality: Instead of plain binary summarization, the method couples diffing, callgraph-aware prompting, and LLM-based triage (FSS), which is a novel combination for supply-chain inspection.

Quality: FSS is fully specified with explicit weights and formulas, and the two-prompt LLM pattern (summary $\to$ FSS) is clearly shown (Fig. 2).

Clarity: The pipeline (Fig.~1) is easy to follow and reverse-BFS ordering is well motivated.

Significance: The dataset is sizable and realistic for OSS supply-chain updates and the XZ case study connects the framework to a real incident.

### Weaknesses
Mid recall. The best setting gets precision 0.98 but recall only 0.64, meaning ~36% of malicious updates are missed. This is high precision, not high accuracy. A threshold/$k$ sweep is needed.

LLM brittleness. Some models do not output the requested labels, and the paper fixes this by using GPT-5 mini as the final predictor for all models (Table 3). This shows the method depends on a particular LLM.

FSS validation is indirect. FSS is evaluated against the very malware the authors injected (malicious vs. benign functions), not against human-labeled sensitivity or a trivial rule baseline.

Benchmark scope. All malwares are source-level, compiled with the same toolchain, and there is no obfuscation/packing, so the claim should be narrowed to "LLM-assisted analysis of cleanly injected updates.''

No FSS ablation. There is no experiment showing that FSS is actually better than, e.g., selecting functions by the presence of network/crypto/file ops.

### Questions
1- Can you report precision/recall for more $k$ values and/or FSS thresholds so we can see whether recall can be raised above $0.8$ at acceptable precision?

2- On what data were the FSS weights/tuning done, and were the same weights used on the XZ case? Please clarify to rule out test-set tuning.

3- In the XZ case, GPT-5 mini misclassifies its own summaries but GPT-5 succeeds. Does this mean the final predictor must be a high-capability LLM?

4- What happens when Ghidriff mismatches or fails to match a modified function? Is the downstream LLM step robust to an incomplete diff callgraph?

5- Will you release sanitized ghidriff outputs + prompts + LLM responses so others can test the FSS part without distributing malware binaries?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces an LLM-based binary diff summarization that can be used to detect maliciously injected code or vulnerabilities from software updates. This approach utilizes LLM to analyze binary code and diff. The paper introduces a software supply chain security benchmark with 3 malware injections into 6 open-source projects. The proposed LLM-based binary diff summarization technique is used to create summaries that a downstream detector (utilizing functional sensitivity scores) uses to detect the injected malware. The proposed techniques achieve a 98% precision and 64% recall on the introduced dataset.

### Strengths
+ Detecting malware is an important topic. Having a reliable automatic technique that can flag potential malicious changes would make an IT admin’s job much easier.
+ The precision is quite good without many false negatives.

### Weaknesses
- The paper did not discuss any potential baseline, nor did it compare to any baseline. It would be possible to use prior work in binary analysis to create a baseline.
- The recall is quite low at just under 60%. Having some quality analysis on modes of failure for those false negatives would give future researchers some insights into how to improve malicious diffs.
- The lack of an ablation study makes it hard to know how each component contributes to the overall effectiveness of the technique.

### Questions
1. What is the reason for not having a baseline? Can any of the prior binary analysis tools be used as a baseline?
2. Since FSS behaves similarly to CVSS3.1, can CVSS3.1 be used instead and still produce similar results?
3. Since using GPT5mini as a predictor while using a smaller model for summary produces very close results to the full GPT5mini stack. What are the differences in cost? Would a cost vs effectiveness analysis be done to show the cost vs gain?
4. What are the typical modes of failure for false negatives?

### Soundness
2

### Presentation
3

### Contribution
3
