# Benchmarking and Enhancing LLM Agents in Localizing Linux Kernel Bugs

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 6, 2

## Abstract
The Linux kernel is a critical system, serving as the foundation for numerous systems. Bugs in the Linux kernel can cause serious consequences, affecting billions of users. Fault localization (FL), which aims at identifying the buggy code elements in software, plays an essential role in software quality assurance. While recent LLM agents have achieved promising accuracy in FL on recent benchmarks like SWE-bench, it remains unclear how well these methods perform in the Linux kernel, where FL is much more challenging due to the large-scale code base, limited observability, and diverse impact factors. In this paper, we introduce LinuxFLBench, a FL benchmark constructed from real-world Linux kernel bugs. We conduct an empirical study to assess the performance of state-of-the-art LLM agents on the Linux kernel. Our initial results reveal that existing agents struggle with this task, achieving a best top-1 accuracy of only 41.6\% at file level. To address this challenge, we propose LinuxFL$^+$, an enhancement framework designed to improve FL effectiveness of LLM agents for the Linux kernel. LinuxFL$^+$ substantially improves the FL accuracy of all studied agents (e.g., 7.2\% - 11.2\% accuracy increase) with minimal costs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces LinuxFLBench, a fault localization benchmark for Linux that includes 250 manually verified cases. Compared with SWE-Bench, LinuxFLBench presents more challenging Linux bug reports, primarily due to the large scale of the Linux codebase and the diversity of bug causes. In the evaluation section, the authors assess both traditional information retrieval–based methods and agent-based fault localization methods, finding that compared with SWE-Bench, these agents experience more than a 15% drop in recall on LinuxFLBench. Finally, the authors propose a new framework to mitigate these challenges.

### Strengths
- The paper is well-written and clearly presented.
- The proposed benchmark introduces a higher level of difficulty compared to SWE-Bench.
- This benchmark focuses on user-reported bugs rather than fuzzing-based ones, which makes the dataset more realistic.
- The experimental evaluation is fairly comprehensive, covering most mainstream fault localization approaches.

### Weaknesses
- When constructing the benchmark, the authors still define the ground-truth root cause based on the patch location. While simple, this approach can be imprecise since the patch location does not always correspond to the actual root cause.  
- The benchmark focuses solely on Linux bugs. Although Linux is one of the largest codebases, the motivation provided by the authors is not fully convincing:  
  - **Limited Observability**: the main challenge lies in the lack of runtime information. However, couldn’t one instead use bug reports from other large projects such as Redis or QEMU, filtering out those that include runtime information? Moreover, this challenge does not seem to be reflected in the benchmark’s evaluation results. The difficulties summarized in the evaluation are also unrelated to this point.  
  - **Diverse Impact**: while the diversity of root causes is indeed a major challenge in FL, is this a sufficient reason to choose only Linux? Couldn’t other large-scale projects (e.g., Redis, QEMU) also achieve this? Furthermore, does maximizing root-cause diversity conflict with the decision to focus solely on Linux cases?  
- In the evaluation, the authors only used GPT-4o, which is not a state-of-the-art (SOTA) LLM and lacks strong reasoning capabilities. Since the performance of agents is highly correlated with model capability, although the authors discuss this limitation, I would still encourage them to include comparisons with current SOTA models such as GPT-5 or Claude 4.5.

### Questions
- When evaluating these agents, did the authors provide them with an execution environment?  
- Were the agents’ tool invocation or command execution abilities (e.g., writing and running search scripts during iteration) considered in the experiments?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LinuxFLBench, a benchmark of 250 real-world Linux kernel fault-localization (FL) tasks constructed from user-reported bugs (Kernel.org Bugzilla), spanning 120 kernel versions and 66 components. Each task provides a bug report, the buggy codebase, and ground-truth buggy file/method(s) derived from patches. The paper evaluates state-of-the-art LLM agents (SWE-Agent, AutoCodeRover, Agentless) and classical IR baselines (BM25, BugLocator, BLUiR, Sentence-BERT). Results show that LLM agents outperform classical IR methods but still struggle on the Linux kernel compared to general software benchmarks like SWE-bench (e.g., best top-1 file-level recall is 41.6% vs. ~70% on SWE-bench). To address these gaps, the authors propose LinuxFL+, a post-hoc enhancement framework with (i) directory-aware expansion, (ii) potential-cause expansion via LLM hypothesis (direct and mail-augmented using LKML), and (iii) candidate integration with re-ranking. LinuxFL+ improves all agents with modest token cost (e.g., SWE-Agent Recall@10: 0.584 → 0.768; MRR: 0.476 → 0.610), and provides consistent gains even for a smaller open-source backbone (Qwen3-32B). Method-level FL remains challenging (Recall@1 < 0.14 even with LinuxFL+).

### Strengths
- **Originality**: New, domain-specific FL benchmark for Linux with real user-reported bugs at larger scale than SWE-bench(-lite). Novel and important observation of degraded agent performance on the kernel.
- **Quality**: Careful curation; strong baseline coverage (IR and agents); robust gains from LinuxFL+ across agents/backbones; statistical significance analysis; cost accounting shows practical overhead (~$0.04/task).
- **Clarity**: Clear motivation and high-level method; results/ablations are informative; limitations stated.
- **Significance**: Highlights challenges of applying LLM agents to large, low-observability, multi-factor systems. Likely practical impact for practitioners evaluating Linux bugs.

### Weaknesses
- **Benchmark bias via single-file constraint**: Excluding multi-file fixes may skew composition and agent failure modes. Please quantify exclusions, characterize them, and consider a multi-file track.
- **Leakage/provenance in mail-augmentation**: Despite pre-report restriction and filters, subtle leakage is possible. A formal audit (time-window sensitivity, near-duplicate detection, blinded checks) and a “clean” subset would strengthen claims.
- **Heuristic under-specification**: Directory-aware expansion (choice of k), merging and LLM re-ranking are lightly formalized. Compare against strong non-LLM re-rankers (BM25/SPLADE/ColBERT on path+report) and provide sensitivity studies.
- **Limited LLM/agent diversity**: Most results rely on GPT-4o with one open-source model. Explore more open-source backbones and alternative agent frameworks/policies to bolster generality.
- **Method-level FL still weak**: Although improved, recalls remain low. Decompose errors into file recall vs. intra-file ranking; explore alternative skeletonization (call-graph stubs, macro expansion hints).
- **Scope/portability**: C/Linux focus. Discuss portability to other large C/C++ systems and consider releasing scripts to replicate the construction elsewhere.

### Questions
1. **Leakage auditing**: Can you provide a formal audit for mail-augmentation (time windows, near-duplicate detection, blinded assessment that emails don’t directly reveal locations)?
2. **Single-file constraint**: What fraction of reports were excluded due to multi-file patches? How do agents and LinuxFL+ perform on a multi-file track?
3. **Sensitivity analyses**: Report sensitivity to directory expansion k, retrieval top-k, and re-ranking. Are gains robust across settings?
4. **Non-LLM re-ranking baselines**: How does LLM re-ranking compare to strong lexical/sparse/dense rankers over file path + report?
5. **Backbone diversity**: Beyond GPT-4o and Qwen3-32B, how do other open-source models (7B–72B) fare? Any scaling trend?
6. **Method-level bottlenecks**: Can you decompose failure into (i) file recall vs. (ii) method ranking? Would richer skeletons help?
7. **Generalization**: Any early results on other kernels/large C++ codebases? Will you release scripts for reproducing analogous benchmarks?
8. **Cost/latency**: Per-task wall-clock times with/without LinuxFL+ and guidance on batching/parallelization?

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
3

### Summary
The paper tackles the problem of fault localization for large scale software systems like the linux kernel. The authors first propose a well-defined FL benchmark created from bug reports/patches in the Linux Kernel. The authors compare baseline agent-based/and other prior software issue solving techniques and demonstrate significant drop in fault localization on the benchmark. Furthermore the authors propose LinuxFL to improve localization performance specific on the linux kernel through a combination of reranking and retrievel using linux mailing list.

### Strengths
- deals with a very important problem of fault localization in large software systems such as the Linux kernel
- construct a benchmark for this specific task that can be used by future research projects
- benchmark includes manual annotation which ensures high quality

### Weaknesses
- benchmark ignores more complex and difficult bugs/fault localization tasks:
	- The authors claims that "we kept only unambiguous cases where exactly one file was modified to ensured the reliability of the ground truth". This would ignore a lot of multi-file bugs which are extremely important to evaluate 
- baseline setups:
	- The paper compares against some prior baselines used for solving software development issues like the ones in SWE-bench. 
	- however, the task in this work is different (fault localization), its unclear how the authors modify an approach (e.g., SWE-agent) for a fault localization task?
- proposed approach are aimed specifically for Linux kernels and may not generalize to other large systems:
	- building on my prior point regarding the baseline setup, the prior tools work well as they can be utilized for other repos as well as different software engineering tasks (e.g., repair, feature implementation) 
	- On the other hand, the proposed LinuxFL+ is extremely tailored for Linux kernels with the authors even using Linux mailing lists

### Questions
1. How did the authors modify the prior agents to perform fault localization tasks instead of solving software issues
2. Please comment on if LinuxFL+ can be utilized for effective fault localization in other software systems. For example, have the authors applied similar approach for fault localization on the SWE-bench problems?
3. Using the Linux mailing list could lead to some data leakage (even excluding Bugzilla), as the developer may discuss patches that are very relevant for the bug you want to localize, how did the authors address this issue?
4. Can the authors also comment on if the constructed benchmark can be utilized for repair evaluation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces LinuxFLBench, a new benchmark for evaluating fault localization (FL) in the Linux kernel, and proposes LinuxFL+, an enhancement framework designed to improve existing LLM-based agents’ performance on this challenging domain.
The authors benchmark several leading LLM agents (SWE-Agent, AutoCodeRover, Agentless) and find that they perform significantly worse on Linux kernel bugs than on existing datasets like SWE-bench.

### Strengths
- The paper tackles an important and underexplored domain, namely fault localization in large-scale, low-level software like the Linux kernel.

- Provides a well-constructed benchmark (LinuxFLBench) with real-world bug reports and ground-truth patches.

- The enhancement framework (LinuxFL+) offers practical, empirically validated gains with low cost and good reproducibility.

### Weaknesses
1. **Limited methodological novelty in LinuxFL+**
Although LinuxFL+ improves empirical results, the method primarily combines straightforward retrieval expansion and RAG-style augmentation.
The two strategies( i.e., directory-aware expansion and potential-cause expansion) are intuitive extensions rather than novel algorithmic contributions.
No clear theoretical insight or principled reasoning is provided for why these expansions work or how they might generalize beyond this specific setup.

2. **Benchmark coverage and representativeness remain narrow**
LinuxFLBench contains only 250 bugs despite the vast number of real-world Linux kernel issues.
Moreover, the dataset filters only “single-file fix” bugs, which simplifies the problem but undermines realism.
As a result, the benchmark might not capture the complexity of multi-file interactions and cross-component dependencies typical of real kernel debugging.

3. **Insufficient Evaluation analysis**
The evaluation remains superficial, reporting results without diagnostic analysis of why LLM agents fail or how the solution addresses specific reasoning errors. The failure analysis in Section 4.3 is anecdotal, lacking a systematic categorization of error types. Furthermore, the reasons behind the limited method-level improvements

4. **Missing discussion on contamination and data leakage**
Since LinuxFLBench is built from public Bugzilla and mailing list data, and the models (especially GPT-4o) are trained on internet-scale corpora, there is a non-trivial risk of data contamination.
The paper does not describe any checks for overlapping content between pretraining data and evaluation tasks.

### Questions
Q1: Could you clarify what key conceptual or methodological innovation distinguishes LinuxFL+ from standard RAG or search expansion frameworks?

Q2: LinuxFLBench includes only 250 single-file bugs, which may limit its realism and generalizability. How do you plan to address the limited dataset scale and the exclusion of multi-file or cross-module bugs?

Q3: Can you provide a more systematic analysis of the types of failures or reasoning gaps observed in LLM agents, and evidence or procedures confirming that LinuxFLBench and the LKML data are not contaminated with model pretraining data?

### Soundness
1

### Presentation
3

### Contribution
2
