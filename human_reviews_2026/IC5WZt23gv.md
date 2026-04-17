# Ladders of Thought: A Self-Evolving Curriculum of Progressively Simplified Reasoning Traces

- Decision: Reject
- Scores: 2, 4, 6, 4, 4

## Abstract
Large language models (LLMs) excel at reasoning when scaled to hundreds of billions of parameters, but small- and mid-scale models remain brittle reasoners even with knowledge distillation (KD). We present Ladders-of-Thought (LoT), a framework that improves reasoning by combining progressive question rewrites with a self-evolving curriculum. LoT automatically generates semantically faithful but easier variants of reasoning problems, organizes them into difficulty buckets using step-based measures, and employs a self-evolving bandit scheduler to allocate training adaptively. Evaluated on two reasoning domains, math and multi-hop reasoning, across OPT-1.3B/2.7B and Pythia-1.4B/2.8B, LoT consistently improves over KD. It delivers large gains on arithmetic tasks (e.g., +32 percentage points on AddSub, +25pp on SVAMP), +2–8pp improvements on in-domain test splits, and strong though dataset-dependent benefits on multi-hop reasoning (e.g., +16pp on QASC, +25pp on StrategyQA). LoT also converges faster than staged curricula, highlighting the value of adaptive progression. These results show that progressive rewrites coupled with adaptive curricula provide a simple yet effective recipe for strengthening reasoning in smaller LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Ladders‑of‑Thought (LoT), a training framework for small–mid‑scale LLMs that combines (i) progressive rewrites, (ii) step‑based difficulty labeling, and (iii) a self‑evolving curriculum. Experiments on math and multi‑hop reasoning with OPT/Pythia show large OOD gains and consistent in‑domain improvements on EntailmentBank. The authors also provide ablations demonstrating optimal moderate rewrite depth and superiority of adaptive scheduling

### Strengths
S1: This paper introduces a clear and principled training recipe that pairs semantically faithful progressive simplification with an adaptive curriculum, avoiding brittle reliance on raw CoT length and enabling interpretable difficulty control via step counts. 

S2: This paper formalizes scheduling as a non‑stationary multi‑armed bandit and provides concrete update rules and policies, which make the "self‑evolving” curriculum operational and reproducible.

S3: This paper reports strong OOD gains on arithmetic datasets and non‑trivial improvements on EntailmentBank, demonstrating architecture‑agnostic benefits across OPT and Pythia.

### Weaknesses
W1: The core idea of progressively simplifying problems and training with difficulty-aware scheduling overlaps substantially with earlier work on task decomposition (e.g., Least-to-Most Prompting[1]) and iterative self-improvement[2,3]. The paper mainly combines these ingredients rather than introducing a clearly new algorithmic principle.

W2: This paper deviates from standard evaluation protocols by converting multiple‑choice tasks to free‑form generation and using a four‑stage verification pipeline with SBERT similarity and F1 thresholds, which may hinder comparability with prior work and inflate pass@5. 

W3: This paper’s step‑based difficulty relies on an LLM rewrite model to identify premises and steps, but the paper provides only spot‑checks rather than systematic fidelity audits.

W4: This paper does not quantify the additional training and data‑creation cost (rewrite generation, bandit evaluation) in tokens

W5: Several closely related reasoning‑oriented curricula and refinement methods at training or inference time are missing (e.g., [1]-[5]).

W6: Although the paper states a focus on small- to mid-scale LLMs, all reported models are 1.3B–2.8B and no 7–14B “mid-scale” results are provided. The paper should include one or two 7B–14B model to substantiate the claim. 

W7: Figures 1 and 2 are too low-fidelity and hurt readability, the authors are suggested to replace them with vectorized diagrams and enlarge key labels for accessibility. 



[1] Zhou, D. et al. Least‑to‑Most Prompting Enables Complex Reasoning in Large Language Models.  ICLR 2023

[2] Madaan, A. et al. Self‑Refine: Iterative Refinement with Self‑Feedback. NeurIPS 2023

[3] Shinn, N., Labash, B., & Gopinath, A. Reflexion: Language Agents with Verbal Self‑Reflection. NeurIPS 2023

[4] Chen, W. et al. Program of Thoughts Prompting: Disentangling Computation from Reasoning. TMLR 2023

[5] Gao, L. et al. PAL: Program‑Aided Language Models. ICML 2023

### Questions
Q1: This paper claims "minimal” steps after rewriting; how is minimality operationally verified beyond spot‑checks, and can the authors provide inter‑annotator agreement or automatic proofs that each rewrite reduces the necessity of one step?

Q2: This paper uses pass@5 with temperature/top‑p; how do results change under deterministic decoding (temperature 0) and single‑sample pass@1, and do trends hold?

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
This paper presents LoT, a recipe for strengthening the reasoning capabilities of small models. Rather than forcing students to imitate every step of a long expert chain, LoT employs a large-scale rewriter to progressively replace antecedents with their logically entailed conclusions, yielding a ladder of questions whose difficulty strictly decreases. A self-evolving, multi-armed-bandit curriculum then adaptively allocates training across these rungs. Evaluated on several datasets with 1–3 B parameter models, LoT achieves substantial gains over strong baselines.

### Strengths
1. LoT provides a fully automatic way to produce graded training examples without human difficulty labels. Manually labeling dataset difficulty is a massive undertaking.
2. The proposed progressive rewrite strategy leverages the dataset more effectively through difficulty bucketing and a multi-armed-bandit curriculum.
3. LoT demonstrates strong and consistent performance, substantially outperforming baseline methods across various datasets and model architectures.

### Weaknesses
1. Rewrite quality hinges on an external model, becoming a potential performance bottleneck. The rewriter must be both powerful and instruction-following; any semantic drift or erroneous intermediate conclusion will mislead the student with noisy supervision.
2. Theoretical analysis is missing. The method remains at the empirical level, offering no convergence or generalization bounds to explain why progressive simplification helps. Rewriter-generated simplifications are only spot-checked; no quantitative guarantee of correctness or semantic equivalence is provided.
3. Reasoning difficulty is not fully captured by execution step count. Difficulty is approximated solely by step count, ignoring numerical magnitude, lexical complexity, and other factors. Problems requiring the same number of steps can still differ vastly in information density, numerical size, and logical pattern.
4. Parameter scope is restricted to small models (1.3–2.8 B); scalability to 7/13/30 B remains untested. Tasks are confined to math word problems and English multi-hop QA, without more complex scenarios such as code synthesis, embodied instruction following, or cross-lingual mathematical reasoning.
5. Computational overhead and scalability are under-discussed. The rewrite stage requires rewriter calls for every training sample, incurring substantial pre-computation costs. Scaling to larger corpora or more complex data domains remains unaddressed.

### Questions
Please refer to Weaknesses.

### Soundness
2

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
Title: Ladders of Thought (LoT): A Self-Evolving Curriculum of Progressively Simplified Reasoning Traces Overall Recommendation: Strong Accept

This paper presents a novel and effective framework to tackle a well-known challenge: improving the reasoning capabilities of small to mid-scale (1-3B parameter) language models. The proposed "Ladders-of-Thought" (LoT) framework is built upon two core components.

Progressive Simplification: Instead of standard (question, answer) pairs, LoT automatically rewrites a complex, multi-step reasoning problem into a "ladder" of semantically faithful but progressively easier variants. As illustrated in Figure 2, this is achieved by injecting intermediate conclusions (e.g., C1, C2) into the problem's premises, replacing the original facts (e.g., P1, P2) used to derive them.

Self-Evolving Curriculum: LoT employs a Multi-Armed Bandit (MAB) scheduler rather than a fixed curriculum (e.g., easy-to-hard). This scheduler adaptively samples from different difficulty "buckets" (Easy, Medium, Hard) based on the model's learning progress on a validation set, thereby maximizing training efficiency.

### Strengths
1. Enhancing the reasoning abilities of small, efficient models is critical for practical and widespread deployment. This paper directly addresses the brittleness of small models (OPT 1.3B/2.7B, Pythia 1.4B/2.8B) in reasoning tasks

2. The "progressive simplification" approach is an innovative and intuitive mechanism. It differs from standard problem decomposition by preserving the integrity of the original task while providing variable levels of "scaffolding". This creates a natural and principled set of difficulty levels for curriculum learning.

### Weaknesses
The progressive simplification step relies on an external 'rewriting model' (R) , creating a dependency on this model's capabilities for the framework's success. To clarify this dependency: are the initial Chain-of-Thought (CoT) rationales (generated by 'teacher T' ) extracted from the same model that is used for the rewriting process (R)? The paper notes this is not required, but further discussion on the impact of this choice, and the potential performance gap between T and R, would be valuable.
2. The paper highlights a significant, dataset-dependent discrepancy in performance. For instance, while LoT achieves massive gains on out-of-distribution (OOD) arithmetic benchmarks like AddSub (+32.11pp for OPT-2.7B) and SVAMP (+25.09pp) , the improvement on the in-domain GSM8K test set is modest (+2.96pp). Could you elaborate on the underlying reason for this?

### Questions
check the weakness

### Soundness
3

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
This paper proposed Ladders-of-Thought (LoT), a curriculum-learning framework for small-sized LLMs that generates semantically faithful but progressively easier versions of each reasoning problem. A complex question is rewritten step-by-step into simpler variants, forming a “ladder” of decreasing difficulty. A multi-armed bandit scheduler then adaptively allocates training examples from these difficulty levels based on model progress. Compared to standard CoT distillation, LoT-trained students achieved higher accuracy and often converged faster. The experiment results show that LoT consistently improves reasoning performance. It yields large gains on arithmetic benchmarks and makes substantial improvements on multi-hop QA. Overall, the core contribution is showing that progressive question rewrites combined with an adaptive curriculum significantly strengthen reasoning in smaller models.

### Strengths
Originality and significance: LoT is a novel method that combines progressive problem simplification and adaptive curriculum scheduling for reasoning tasks. It provides a practical approach for improving reasoning in small language models.

Quality: The evaluation is thorough. It demonstrates consistent and large performance gains across multiple models (two sizes of OPT and two of Pythia) and tasks. They also include several ablations. Varying the rewrite depth shows that shallow-to-moderate ladders yield the best generalization, while excessive simplification can hurt, highlighting the importance of moderation in curriculum length. The authors also compare flat sampling, staged easy-to-hard, staged hard-to-easy, and the self-evolving bandit and show that dynamic curriculum is a key factor for success.

Clarity: The paper is well-organized and explains its methodology and findings clearly.

### Weaknesses
1. While large gains are shown for StrategyQA and QASC, the method shows severe performance degradation on MuSiQue (a regression of 19.6 for OPT-2.7B) and moderate regressions on OpenBookQA. The authors attribute this to differences between compositional reasoning and factual recall, but this lack of robustness raises questions about the framework's broad applicability.

2. LoT is only tested on 1–3B parameter models. Its applicability to larger LMs is untested, so it’s unclear if the same benefits hold as scale increases. Given that the paper's aim is to close the gap between small and large models, the lack of testing on models in the 7B-13B range leaves the question of practical scalability unanswered. Moreover, generating multiple rewrites per example and running a bandit curriculum adds significant computational overhead. The paper does not report training time or resource usage, leaving it unclear whether the accuracy gains justify the extra compute.

### Questions
1. LoT is only evaluated on 1–3B models. Can it scale to larger LLMs or more complex reasoning tasks? Clarifying expected performance or needed adaptations for bigger models would help assess general utility.

2. What is the additional training cost of LoT relative to standard distillation? It would help to see metrics (e.g. GPU-hours) to judge whether the accuracy gains justify the extra computation. Is the overhead dominated by rewrite generation or curriculum scheduling?

3. Can the authors provide a deeper analysis of the negative results on MuSiQue and OpenBookQA? Please identify why and how the proposed LoT could be used to improve these tasks.

4. How do authors ensure that rewritten questions remain semantically equivalent and at the intended difficulty level? Can the authors provide an analysis of the sensitivity of LoT's performance when using models of different sizes to generate the rewrites?

5. The paper notes that the "All" step depth is suboptimal because of "diluting the learning signal." Can the authors present an analysis of the MAB scheduler's distribution over time for the $\le3$ vs. "All" depth runs? Specifically, does the MAB actively avoid the 0-step bucket in the $\le3$ run, and what is its allocation to the 0-step bucket when "All" depths are available?

### Soundness
3

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
3

### Summary
The paper addresses a core limitation in reasoning for small- and mid-scale language models, which struggle to learn generalizable reasoning skills even with knowledge distillation from larger models. The authors propose a novel training framework that strengthens reasoning ability through progressive problem simplification and adaptive curriculum learning.

The proposed method automatically rewrites complex reasoning questions into a sequence of semantically equivalent but easier variants, forming a “ladder” of decreasing difficulty. Each problem’s difficulty is defined by the minimal number of reasoning steps required for a solution. These are grouped into difficulty buckets, and a self-evolving multi-armed bandit scheduler dynamically allocates training across these buckets based on learning progress.

### Strengths
1. The paper proposes an intuitive yet original framework.
2. It introduces a principled difficulty measure based on the minimal number of reasoning steps, avoiding noisy proxies like chain length.
3. Improvements are observed both in-domain and out-of-distribution, showing good generalization and robustness.

### Weaknesses
1. The authors verify their hypothesis on small base models (<3B), which makes their claim unsound. Because improving over the small model is relatively easier. 
2. Major experiments are conducted on two relative outdated model families (OPT and Pythia). Adding experiments on new models such as Qwen would strength the paper. 
3. Limited baselines are compared. For example, it remains unknown whether decomposing reasoning step is a better strategy compared to decomposing the original question (c.f. [1]). 

---
[1] Divide-or-Conquer? Which Part Should You Distill Your LLM?

### Questions
1. The proposed method is based on the hypothesis that every intermedia reasoning step can be combined with a premise to form a new reasoning step and the premise would be erased. However, I am curious if there exist a case that one premise would be used more than once. In such a case, would the proposed method still work?

### Soundness
1

### Presentation
2

### Contribution
2
