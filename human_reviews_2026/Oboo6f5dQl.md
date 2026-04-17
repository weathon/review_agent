# EDCO: Dynamic Curriculum Orchestration for Domain-specific Large Language Model Fine-tuning

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Domain-specific large language models (LLMs), typically developed by fine-tuning a pre-trained general-purpose LLM on specialized datasets, represent a significant advancement in applied AI. A common strategy in LLM fine-tuning is curriculum learning, which pre-orders training samples based on metrics like difficulty to improve learning efficiency compared to a random sampling strategy. However, most existing methods for LLM fine-tuning rely on a static curriculum, designed prior to training, which lacks adaptability to the model's evolving needs during fine-tuning. To address this, we propose EDCO, a novel framework based on two key concepts: inference entropy and dynamic curriculum orchestration. Inspired by recent findings that maintaining high answer entropy benefits long-term reasoning gains, EDCO prioritizes samples with high inference entropy in a continuously adapted curriculum. EDCO integrates three core components: an efficient entropy estimator that uses prefix tokens to approximate full-sequence entropy, an entropy-based curriculum generator that selects data points with the highest inference entropy, and an LLM trainer that optimizes the model on the selected curriculum. Comprehensive experiments in wireless and data communications domains demonstrate that EDCO outperforms common curriculum strategies for fine-tuning Qwen3-1.7B/4B models under supervised and reinforcement learning settings. Furthermore, our efficient entropy estimation reduces computational time by 83.5% while maintaining high accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes EDCO, a dynamic curriculum method for domain-specific LLM fine-tuning. Instead of a fixed, “easy-to-hard” order, EDCO repeatedly picks the highest–inference-entropy samples so the model is always challenged at its current frontier. It adds two efficiency tricks: Quick-Answer Prompting (ask the model to answer concisely so early tokens carry signal) and a Prefix Entropy Approximation (estimate entropy from the first L tokens rather than the full sequence).

### Strengths
- The paper makes a **simple, actionable loop** (filter → estimate prefix entropy → pick top-(N) → fine-tune → repeat) that others can readily implement. 
- The **Quick-Answer Prompting** and **Prefix Entropy Approximation** are pragmatic choices that lower cost while keeping a usable ranking signal. 
- Results show **consistent improvements** over length/answer-complexity/perplexity curricula in both SFT and RLFT settings.

### Weaknesses
- The evaluation is **narrow in domain and data provenance** (two communications domains with synthetic/curated data), so **generalization** to other areas (e.g., bio, law, open-domain QA) remains unclear. 
- Several **important baselines are missing**: there is no direct comparison to stronger **dynamic** curricula (e.g., learnable policies or uncertainty-aware selection beyond perplexity), which makes it hard to judge competitiveness at the state of the art. 
- The **related-work section** under-cites recent sentence/uncertainty-driven curricula and broader data-selection work, so the positioning could be better grounded in prior art.

### Questions
Can you compare against a **learned** or **bandit-style** curriculum policy (e.g., SEC-like) and a **confidence/variance-based** selector beyond perplexity? What about active-learning heuristics?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
EDCO proposes a dynamic per sample curriculum for domain specific LLM fine tuning. At each stage, the model’s uncertainty on every example is estimated efficiently by computing inference entropy over only the first L tokens of a quick and short answer prompt. The highest entropy samples are selected, the model is fine tuned on this subset, and the scores are refreshed for the next stage. This prefix entropy estimator correlates well with full-sequence entropy while avoiding the cost of full sequence entropy, enabling frequent re ranking. The paper evaluates EDCO on two communications domains Wireless and Data Communication using supervised fine tuning with Qwen-3 4B and reinforcement learning fine tuning with Qwen-3 1.7B. EDCO consistently improves accuracy over heuristic curricula such as Random, Length, Answer Complexity, and Perplexity, yields competitive results in reinforcement learning fine tuning, and achieves substantial per sample speedups in entropy estimation.

### Strengths
1. Firstly, the motivation of trying to find a dynamic approach that works in practice is valuable. Moreover, the paper introduces a good approach by combining a quick answer prompting step with prefix only entropy, effectively reducing the computational load of selection during training phases. This approach addresses a limitation in dynamic curricula, enabling more frequent re ranking and thereby enhancing their practical applicability in standard training pipelines, both in supervised fine tuning and reinforcement learning based fine tuning scenarios.
2. The empirical quality is ok as the method shows consistent improvements over established static curricula on Wireless and Data Communication, and the training process analyses together with the efficiency study reporting an approximately 83.5 percent per sample reduction in entropy estimation time provides a clean explanation of why the approach works and why it is practical.
3. The presentation is clear, and the select then fine tune then refresh loop is easy to follow, and key choices such as the quick answer prompt, prefix length, and refresh cadence are described with enough detail that new researchers could implement and tune the method without much effort.

### Weaknesses
1. The main contributions are engineering improvements using known parts like uncertainty-based selection, quick-answer prompting, and prefix truncation rather than a new objective or learning principle.
2. Results are limited to two communication domains with synthetic datasets, with 12k high-quality train samples and only 230 test samples per domain. The evidence for generalization to other domains and standard benchmarks is missing.
3. Most evaluations compare against static curricula rather than dynamic curricula; the paper should at least apply these low-cost static methods periodically and report the resulting accuracy to strengthen the claims.
4. The paper reports an 83.5% per sample speedup for prefix entropy in Table 1 but dynamic curricula also incur periodic re ranking passes, the paper itself notes this added overhead, yet end to end wall clock and total token budgets aren’t reported.
5. Another minor error to note in the paper is in the paper Fig 3 (B) the caption is written as “Ablation study on the prefix length. To better visualize data trends, the sample indices are arranged in ascending order of the entropy estimated with the 128-token prefix” which doesn’t match the figure and seems like it was really the caption for Figure 4 (B).

### Questions
1. The paper positions it as a practical/efficient domain-specific dynamic curriculum, but the paper does not report end to end wall clock time or total tokens under matched budgets, if possible, please do add that because it will give a complete idea on the accuracy compute trade off which is very useful in a practical setting.
2. Please add dynamic versions of your low cost scoring rules at the same refresh times as EDCO and report accuracy and total time or tokens under the same budget. For example, recompute each rule with the current model at refresh time and select the top m for the next round. This shows whether EDCO's gains come from its entropy scoring. Candidate rules can include perplexity, length, answer complexity, or diversity based scores. If full parity is too expensive, you can refresh these baselines fewer times but keep the overall budget matched.

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
The paper proposes a novel framework that dynamically adjusts the training curriculum of large language models (LLMs) during fine-tuning based on inference entropy. Unlike traditional static curricula, EDCO continuously selects high-entropy samples, those that the model is most uncertain about, to maintain exploration and prevent premature convergence. The approach integrates three components: an LLM-driven quality filter to remove low-quality samples, a dynamic curriculum generator that ranks samples by estimated inference entropy, and an efficient entropy estimator that approximates full-sequence entropy using prefix tokens and “quick-answer” prompting. Experiments in wireless and data communication domains demonstrate that EDCO improves fine-tuning performance for Qwen3-1.7B/4B under both supervised and reinforcement learning paradigms, while reducing entropy-estimation cost.
.

### Strengths
Strengths:

1.	The issue that this paper addressed is a well-known issue of entropy collapse in RL-based fine-tuning. 

2.	This paper provides clear details about the implementation. For example, the integration of prefix-based entropy approximation and quick-answer prompting is both innovative and practical, yielding large computational. 

3.	The experiments are carefully designed, demonstrating consistent improvements across supervised and RL settings. The dynamic of entropy and number of new samples during training clearly validate the. 

Overall, the paper provides a clear, reproducible, and domain-relevant advance that bridges entropy-driven learning theory with efficient, real-world LLM fine-tuning

### Weaknesses
To me, there are two main limitations:

First of all, to me the method is both heuristic and incremental relative to existing curriculum and entropy-based sampling approaches.
Given this, comprehensive experiments are usually necessary. 

However, the experiments are narrow in scope, limited to Qwen models and communication domains, which makes it unclear if the method generalizes beyond this setting. If would be appreciate that if the authors could provide experiments using other models, and experiments on some common dataset beyond communication.

### Questions
See weaknesses

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
3

### Summary
This paper proposes EDCO, a dynamic curriculum learning framework that adaptively selects high-entropy samples during domain-specific LLM fine-tuning. Rather than using static curricula or “easy-to-hard” schedules, EDCO prioritizes samples with the largest inference uncertainty to maintain exploration and avoid early entropy collapse. The approach introduces two key techniques for scalable entropy estimation: (1) Quick-Answer Prompting, prompting the model to produce answers quickly so early logits reflect difficulty, and (2) Prefix Entropy Approximation, estimating entropy using only prefix tokens instead of full outputs. Experiments on wireless and datacom domains (Qwen-1.7B/4B) show improved accuracy over static curricula and random sampling in both SFT and RL fine-tuning settings, with 83.5% lower computation cost for entropy measurement. The paper also provides ablations and entropy-dynamics analysis.

### Strengths
- Introduces a reverse-curriculum strategy, focusing on hard, high-entropy samples, to efficiently specialize pretrained LLMs, countering “easy-to-hard” convention. It is also motivated well by the entropy collapse phenomena in RL-trained LLMs, aligning with recent works on exploration-preserving training. 
- Prefix-Entropy Approximation is a practical innovation enabling dynamic curricula at scale. 
- Strong empirical improvements. 
- Overall the problem is interesting and addresses a pressing issue: domain LLM specialization under limited high-value data.

### Weaknesses
- The method is tested only on telecom & wireless engineering tasks. I would like to see evaluation results on a qualitatively different domain to strengthen generality claims, like medical reasoning/legal tasks/math/code etc. 
- There is a potential Over-focus on High-Entropy Outliers. This method selects top-entropy samples but high entropy may arise from nonsensical edge cases or OOD errors, and could lead to overfitting to pathological difficulty zones. Maybe the authors can test threshold-based vs top-k selection or include a “moderate-entropy window” experiment.
- I have some compute scaling concerns. Prefix entropy still requires evaluating the full dataset at intervals. Maybe the authors can discuss adaptive update strategies (e.g., stop updating entropy once stabilized, or partial-dataset entropy sweeps).

### Questions
See Weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3
