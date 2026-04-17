# A Comprehensive Evaluation of Code Language Models for Security Patch Detection

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Detecting vulnerability-fixing commits (VFCs) is critical for timely security patch deployment, yet advisory databases lag patch releases by a median of 25 days and many fixes never receive advisories, driving interest in automated detection methods. We present a comprehensive evaluation of code language model (code LM) based VFC detection through a unified framework consolidating 20 fragmented datasets spanning more than \num{180,000} commits. Our analysis strenghtens existing observations through a systematic evaluation and reveals that high performance metrics mask fundamental limitations. Models achieve F1 scores of 0.9 when using full commits with messages, but drop to 0.6 on code alone while message-only models maintain close to original performance. This demonstrates reliance on textual patterns rather than semantic code understanding. We evaluate code LMs ranging from \num{125}M to \num{30}B parameters across multiple architectures, finding only marginal improvements with scale. Estimating out-of-distribution performance using repository-based splits exposes 10-11\% performance drops compared to temporal splits, revealing models learn project-specific patterns rather than security semantics. Even additional intra-procedural context fails to improve detection. Prompt-based classification with models up to 480B parameters also underperforms fine-tuned approaches, indicating limitations beyond model scale. High inter-model agreement rates indicate convergence on similar patterns rather than complementary understanding. Overall, our findings suggest that code LMs appear fundamentally limited for code-centric security patch detection. We release our unified framework and evaluation suite to enable future \spd\ research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a comprehensive evaluation of transformer-based VFC detection through a unified framework consolidating 20 fragmented datasets spanning more than 180,000 commits.
Its analysis reveals that high performance metrics mask fundamental limitations.

### Strengths
- This paper focuses on addressing an important AI4SE question
- This paper conducts a comprehensive experiments on VFC detection.

### Weaknesses
Actually, I really appreciate the authors' efforts in conducting an evaluation on existing VFC detection datasets and approaches. 

However, the main limitation is the insufficient study on related work [VulFixMiner, Sun et. al., Steenhoek et. al., ColeFunda, LLM4VFD], thus "re-finding" some existing findings.
- "Repository-based evaluation vs commonly-used temporal splits" please refer to [Steenhoek et. al.] Figure 5. Note that deep-learning-based vunlerability detection is highly-correlated with deep-learning-based vulnerability patch detection, this finding in this paper is not novel enough. Addtionally, VulFixMiner (ASE'21) also points out this issue and splits their training/validation/testing set in a repo-level.
- "the reliance on textual shortcuts" please refer to [Steenhoek et. al.]. That paper (also focusing on vulnerability patch identification) has already pointed out this fact in Table IV, i.e., commit message only performs better than code only, and including all codes is not a proper choice. 

"achieves 33× speedup over existing approaches". This speedup mainly comes from the poor efficiency of joern, which generates much unnecessary information. 

Model Selection. LLMs used in evaluation is a little outdated (the latest is StarCoder). Please consider more recent LLMs, which has shown substantially better performance in code-related tasks, e.g., Qwen3-Coder (the authors can also choose models with similar parameters)

#### Please add the following reference paper
[VulFixMiner] Zhou J, Pacheco M, Wan Z, et al. Finding a needle in a haystack: Automated mining of silent vulnerability fixes[C]//2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2021: 705-716.

[Sun et. al.] Jiamou Sun, Zhenchang Xing, Qinghua Lu, Xiwei Xu, Liming Zhu, Thong Hoang, and Dehai Zhao. 2023. Silent Vulnerable Dependency Alert Prediction with Vulnerability Key Aspect Explanation. In Proceedings of the 45th International Conference on Software Engineering (ICSE '23). IEEE Press, 970–982. https://doi.org/10.1109/ICSE48619.2023.00089

[Steenhoek et. al.] Steenhoek B, Rahman M M, Jiles R, et al. An empirical study of deep learning models for vulnerability detection[C]//2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE). IEEE, 2023: 2237-2248.

[ColeFunda] Zhou J, Pacheco M, Chen J, et al. Colefunda: Explainable silent vulnerability fix identification[C]//2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE). IEEE, 2023: 2565-2577.

[LLM4VFD] Xu Yang, Wenhan Zhu, Michael Pacheco, Jiayuan Zhou, Shaowei Wang, Xing Hu, and Kui Liu. 2025. Code Change Intention, Development Artifact, and History Vulnerability: Putting Them Together for Vulnerability Fix Detection by LLM. Proc. ACM Softw. Eng. 2, FSE, Article FSE023 (July 2025), 22 pages. https://doi.org/10.1145/3715738

[CompVPD] Chen T, Li L, Qian T, et al. CompVPD: Iteratively Identifying Vulnerability Patches Based on Human Validation Results with a Precise Context[J]. arXiv preprint arXiv:2310.02530, 2023.

### Questions
1. Please refer to the preceding weakness.
2. Note that existing LLMs are not trained with Gum-Tree's git diff formats, what about using raw git diff for LLM to comprehend?
3. What is the reference link of "[anonymous submission]" in the Section Data Availability?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a comprehensive and critical evaluation of transformer-based code language models (LMs) for the task of SPD. The authors construct a unified framework that consolidates 20 fragmented datasets, encompassing over 180,000 commits, to enable systematic comparison. The core finding is that the high performance reported in prior work is largely illusory, driven by models exploiting textual shortcuts in commit messages rather than learning the semantic patterns of security-relevant code changes. When restricted to code diffs alone, model performance drops precipitously (F1 ~0.6). The study further demonstrates that neither scaling model size (from 125M to 15.5B parameters) nor augmenting diffs with intra-procedural semantic context yields meaningful improvements. Evaluations using a rigorous repository-based split reveal significant performance drops (10-11%), indicating poor generalization and over-reliance on project-specific patterns.

### Strengths
•	Comprehensive Scope: The evaluation is unparalleled in its breadth across models, data, and experimental conditions.
	•	Rigorous and Realistic Evaluation: The use of repository-based splits and the PD-S metric provides a much more realistic and trustworthy assessment of model capabilities.
	•	Actionable Insights: The paper moves beyond simply reporting poor performance to diagnose the root causes: shortcut learning, lack of generalization, and architectural limitations.
	•	High Practical Utility: The release of the unified framework and preprocessing tools ensures high impact and promotes reproducibility and future research.

### Weaknesses
•	Label Quality Uncertainty: While acknowledged, the impact of noisy labels (e.g., in D3/D4 from ML/tool-based sources) isn't quantified via sensitivity analysis—could confound low code-only performance.
	•	Limited Scope in Evaluation: Focuses heavily on C/C++ (D1-D3); D4's multi-language inclusion is promising but underexplored (e.g., no cross-lang transfer results). Evaluation omits runtime metrics (e.g., inference speed) despite large models like StarCoder.
	•	Context Enrichment Limitations: Targets intra-procedural only; inter-procedural/cross-file dependencies (key in real patches) are unaddressed. Word-level diffs reduce size but may lose nuance in complex changes.
	•	Missing Baselines: Compares to graph methods (Wang et al., 2023) but not recent non-transformer approaches (e.g., dynamic analysis in Luo et al., 2024) or fine-tuned vision models on code graphs.
	•	Quantitative Gaps: Inter-model agreement is mentioned qualitatively; a table of pairwise agreements could strengthen claims of convergent (non-complementary) learning.
	•	Limited Exploration of the "Why": While the paper excellently documents what is not working (models don't understand code semantics), it offers less insight into why transformer-based architectures fail here. Is it a data-hunger issue, a fundamental limitation of the architecture for representing code changes, or a need for different pre-training objectives?
	•	Context Enrichment's Limited Scope: The context enrichment is strictly intra-procedural. The negative result, while valuable, leaves open the question of whether more costly but semantically richer inter-procedural or repository-level context (as in RepoCPG) could help.
	•	No Exploration of Very Recent LLMs: The model suite, while diverse, does not include the very latest generation of large language models specifically designed for code (e.g., CodeLlama, DeepSeek-Coder). Their performance might differ, though the fundamental limitations identified would likely persist.

### Questions
•	How sensitive are results to label noise? Did you perform robustness checks (e.g., dropping subsets with low manual verification) or estimate error rates via sampling?
	•	For context enrichment, what are the precision/recall of slicing at depths 1 vs. 2, and how does it handle non-C languages (e.g., Python's dynamic typing)?
	•	In repository-based splits, how did you select hold-out repos (e.g., size-matched to train?)? Any results on zero-shot transfer to unseen vulnerability classes?
	•	Given message reliance, have you analyzed specific textual patterns (e.g., keywords like "fix CVE") via interpretability tools, and do they correlate with false positives?
	•	The paper mentions releasing the framework—will it include pre-computed enriched diffs for all datasets, and what's the planned timeline/venue (e.g., GitHub)?
	•	Given the high model agreement rates, what do you hypothesize are the specific, but insufficient, "patterns" in the code diffs that all models are converging on? Have you analyzed the false positives/negatives to identify these common patterns?
	•	The results suggest a fundamental ceiling for current architectures. What, in your view, are the most promising alternative architectural approaches for this task? For example, should the community invest more in graph-based models, or models that explicitly reason over program semantics (e.g., using symbolic execution)?
	•	Beyond intra-procedural context, what other types of information or context do you believe are crucial for accurately identifying security patches from code alone? For instance, is cross-file data-flow, commit history, or a formal specification of the vulnerability necessary?

### Soundness
3

### Presentation
2

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
Core Strengths

Unified Dataset Framework: Addressing Fragmentation in the FieldExisting research on Vulnerability-Fixing Commit (VFC) detection is limited by fragmented datasets—different studies use small-scale datasets with independent annotations and inconsistent formats (e.g., PatchDB, Devign), making horizontal comparisons difficult. This paper is the first to integrate 20 cross-language, cross-vulnerability-type datasets (covering over 180,000 commits). Through standardized parsing, deduplication (three strategies: hash matching, semantic matching, and UniXcoder embedding-based deduplication), and filtering (supporting filtering by language and vulnerability type), it constructs a reusable unified framework. This contribution fills a gap in the field, provides a "fair comparison benchmark" for subsequent research, and can significantly reduce the verification cost of new methods after open-sourcing.

Revealing Core Limitations of Code LMs: Reliance on Textual Shortcuts Over Semantic UnderstandingPrevious studies (e.g., Tang et al. 2023, Wang et al. 2021b) only reported high F1 scores (~0.9) for "code + commit message" but failed to decompose the contributions of the two components. Through ablation experiments, this paper finds that:
When only code is used, the F1 score drops sharply to 0.6;
When only commit messages are used, the F1 score remains at 0.88 (close to the performance with full information).
This finding subverts the perception that "code LMs already possess security semantic understanding capabilities" and clearly indicates that existing models rely on textual keywords such as "security" and "fix" rather than truly understanding the security significance of code modifications, pointing out a core direction for improvement in the field.

Rigorous Generalization Evaluation: Exposing Cross-Repository Transfer DefectsMost existing studies adopt "temporal splits" (dividing training/test sets by commit time), which mask the generalization defects of models. This paper is the first to introduce "repository-based splits" (training and test sets from different projects) and finds that performance decreases by 10-11% (e.g., CodeBERT’s F1 score drops from 0.89 to 0.79). Additionally, graph models like GraphSPD experience a 30% F1 drop in cross-repository scenarios. This result proves that models learn "project-specific patterns" (e.g., the code style of a specific repository) rather than universal security semantics, providing a key warning for model application in real-world scenarios (e.g., cross-project dependency detection).

Lightweight Context Enrichment: Efficiency BreakthroughTo address the high preprocessing latency of existing graph models (e.g., GraphSPD, 57.36 seconds per sample), this paper proposes an "intra-procedural context enrichment method" based on data flow and control flow, achieving a 33x speedup (1.72 seconds per sample). Although it does not improve performance, it provides an efficient foundation for subsequent context optimization (e.g., reducing computational costs when extending to inter-procedural context).

### Strengths
Application Strengths

1. Open-Source Framework Empowers Industry and Academia

The paper open-sources the unified framework and evaluation suite, allowing researchers to quickly access 20 datasets and reproduce experimental results, significantly lowering the entry barrier for VFC detection research. For industry, the framework can be directly used for benchmark testing of internal vulnerability detection systems (e.g., evaluating the performance of different models on enterprise private projects), accelerating technology deployment.

2. Providing Key Warnings for Industrial Deployment

The core findings of the paper (reliance on messages, poor cross-repository generalization) can guide industry in risk avoidance:
If commit messages are non-standard in a project (e.g., lacking keywords like "fix"), the performance of existing models will drop significantly, requiring supplementary manual reviews;
In cross-project dependency detection (e.g., supply chain security), direct use of existing models should be avoided, and targeted optimization of generalization capabilities is necessary.

3. Lightweight Methods Adapt to Resource-Constrained Scenarios

Although the 33x speedup context enrichment method does not improve performance, its efficiency has potential value in resource-constrained scenarios (e.g., edge devices, real-time code review tools). Subsequent optimizations can build on this to enhance semantic capture capabilities while balancing speed and performance.

### Weaknesses
1. Lack of core algorithmic innovation, only staying at the "evaluation" level without exploring solutions;
2. Insufficient coverage of industry-critical dimensions such as multilingual support, silent patches, and engineering metrics, weakening the application guiding value.

### Questions
1. Explore "Code Semantics-Oriented" Model Optimization Solutions

- Data Augmentation: Randomly shuffle or mask security keywords (e.g., "security," "fix") in commit messages to force models to focus on code modifications;
- Semantic Supervision: Introduce code semantic signals (e.g., AST structures, data flow dependencies, vulnerability type labels) and build multi-task training (e.g., "VFC detection + vulnerability type classification") to enhance the model’s understanding of security semantics;
- Contrastive Learning: Design positive and negative sample pairs (e.g., "real vulnerability fixes" vs. "semantically similar non-fix code modifications") to enable models to distinguish between "surface text" and "deep security semantics."
2. Extend Context Enrichment to the "Inter-Procedural" Dimension
The existing intra-procedural context cannot cover vulnerability fixes involving multi-function collaboration (e.g., "missing input validation in function A leads to buffer overflow in function B"). Suggestions include:
- Construct inter-procedural control flow/data flow graphs (e.g., including function call chains) to expand the context scope;
- Evaluate the detection improvement of inter-procedural context for "complex vulnerability fixes" (e.g., logical vulnerabilities, multi-function dependent vulnerabilities).
3. Supplement Specialized Evaluations for Multilingual and "Silent Patches"

- Build specialized datasets for languages such as Python and Java to evaluate model performance differences across languages (e.g., dynamic vs. static languages);
- Screen "silent patches" (without advisories or security keywords) from the dataset, separately evaluate the model’s F1 and PD-S on such samples, and quantify the model’s ability to address real pain points.
4. Incorporate Engineering Metrics for Industrial Evaluation
Supplement engineering metrics such as model inference time (time per sample), memory usage, and the trade-off curve between parameter quantity and performance to provide industry with a "performance-cost" selection basis (e.g., recommending the 125M-parameter CodeBERT instead of the 15.5B-parameter StarCoder, as they have similar performance but lower deployment costs).
5. Deepen Failure Case Analysis
Classify samples where the model fails to detect (e.g., missed detections, false detections) and analyze:
- Which vulnerability types (buffer overflow, SQL injection, logical vulnerabilities) the model performs worst on;
- Whether failed samples share the commonality of "complex code modification semantics but no message prompts," providing specific directions for subsequent optimization.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper evaluates code language models (125M–15B parameters) for Security Patch Detection (SPD) across 20 datasets, covering C/C++ projects such as Linux, FFmpeg, and Chromium. The authors introduce a unified VFC (vulnerability-fixing commit) framework and attempt to enhance model capability by injecting intra-procedural semantic context (AST, CFG, program slices). Results show that transformer-based code LMs primarily rely on commit messages rather than true code semantics, fail to generalize across repositories, and benefit minimally from syntactic/semantic augmentation.

### Strengths
- Timely and security-relevant problem.
- Thorough empirical effort across many datasets and sizes.
- Highlights real and dangerous LM failure modes in SPD.
- Repository-based split addresses realistic deployment drift.
1. **Insufficient baseline variety** — no symbolic/static tool integration.
2. **Overclaims** (“architectural limitations”) without hybrid evaluations.
3. **Under-explored analytical depth**: no CWE-class breakdown, patch difficulty taxonomy, or qualitative failure cases.
4. **No modern LLM prompting or RAG security baselines**.
5. Only intra-procedural semantics — **no inter-procedural reasoning** or software supply-chain context.

### Weaknesses
1. **Insufficient baseline variety** — no symbolic/static tool integration.
2. **Overclaims** (“architectural limitations”) without hybrid evaluations.
3. **Under-explored analytical depth**: no CWE-class breakdown, patch difficulty taxonomy, or qualitative failure cases.
4. **No modern LLM prompting or RAG security baselines**.
5. Only intra-procedural semantics — **no inter-procedural reasoning** or software supply-chain context.

### Questions
1. Did you quantify dataset noise or perform label disagreement analysis?
2. How do results vary by vulnerability type (e.g., buffer overflow vs logic)?
3. Did you try prompting GPT-4/Claude with chain-of-thought or tool use?
4. Why no comparison with symbolic/static analysis or hybrid approaches?

### Soundness
3

### Presentation
3

### Contribution
2
