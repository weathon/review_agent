# When Two is Enough: CoT–PoT Ensembling for Efficient Self-Consistency in LLM Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 2

## Abstract
Self-consistency (SC) is a popular technique for improving the reasoning accuracy of large language models by aggregating multiple sampled outputs, but it comes at a high computational cost due to extensive sampling. We introduce a  hybrid ensembling approach that leverages the complementary strengths of two distinct modes of reasoning: Chain-of-Thought (CoT) and Program-of-Thought (PoT). We describe a general framework for combining these two forms of reasoning in self-consistency, as well as particular strategies for both full sampling and early-stopping. We show that CoT-PoT ensembling not only improves overall accuracy, but also drastically reduces the number of samples required in comparison with the most efficient SC method. In particular, the majority of tasks can be addressed with *only two* samples, which has not been possible with any prior SC methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- Paper proposes **cross-modal self-consistency**: sample CoT and PoT, aggregate under full budget (CPMaj/CPMax/CPAgr) or **early-stop** using a **Bayesian agreement** model; also gives simple $a_2=1$ heuristics (CPAA/CPFA/CPFF).
- Empirically improves over CoT-only/PoT-only self-consistency on math/tabular QA while **reducing samples** (often to two: one CoT + one PoT).

### Strengths
- **Clear practical goal:** leverage complementary error profiles of CoT vs PoT.
- **Simple deployable heuristics:** $a_2{=}1$ rules are easy to implement and often efficient.
- **Reasonable empirical coverage:** multiple datasets/models; ablations touch intra- vs cross-modal agreement.

### Weaknesses
- **Novelty:** Section 3.1 largely amounts to ensembling CoT and PoT with voting variants; the main gain appears to come from cross-modal diversity, which is conceptually incremental.
- **Narrow task scope:** all benchmarks reduce to numeric/program-executable answers; unclear applicability to open-ended reasoning.
- **Decorative theory:** Section 2.2.1’s Bayesian agreement model seems ornamental; best-performing procedures reduce to ad-hoc cross-modal agreement checks (CPAA/CPFA/CPFF with $a_2=1$).
- **Fairness:** Comparisons mix tool-using PoT vs tool-free CoT, no baseline with tool-augmented CoT under identical stopping.
- **Sensitivity & tuning:** Early-stopping hinges on thresholds / priors (e.g., $ρ$, $a_2$); robustness is not convincingly demonstrated.
- **Scope & metrics:** Only executable numeric tasks; no latency/token-cost reporting; limited statistical rigor (no CIs/significance).

### Questions
- **Necessity of Section 2.2.1:** What does the Bayesian agreement model add beyond the simple cross-modal agreement heuristics actually used (CPAA/CPFA/CPFF)? Can you show any setting where the Bayesian estimator changes decisions and improves over those heuristics?
- **Two-sample claim robustness:** For the “two is enough” claim, provide per-dataset/model histograms of stop counts and accuracy conditioned on early stop vs continued sampling. How stable is the ~2-sample regime under domain shift or weaker models?
- **Generality:** any results on non-executable, free-form reasoning (e.g., multi-hop QA, scientific explanations)?
- **Fair baselines:** calculator-augmented CoT and PoT-only early-stopping under identical thresholds?
- **Sensitivity:** curves for accuracy/efficiency vs $a_2$, $\rho$, temperature, and max budget; robustness when $a_2<1$?
- **Cost & latency:** token counts and wall-clock (including interpreter overhead) per dataset/model.
- **Failure modes:** when CoT and PoT agree yet are wrong, what patterns dominate, and can agreement be qualified (e.g., sanity checks)?

### Soundness
2

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
4

### Summary
This paper studies how to improve self-consistency (SC) reasoning in large language models by combining two distinct reasoning modalities: Chain-of-Thought (CoT) and Program-of-Thought (PoT). The authors argue that CoT and PoT exhibit complementary error patterns—CoT being more flexible but error-prone in arithmetic, while PoT being computationally precise but symbolically brittle. They formalize the cross-modal agreement between these reasoning modalities through a Bayesian framework and propose both full-sampling and early-stopping strategies based on this formulation. Extensive experiments on multiple reasoning benchmarks and LLMs show that the approach achieves comparable or higher accuracy with far fewer samples, often requiring only two (one CoT and one PoT).

### Strengths
**Intuitive complementarity between reasoning modalities:**
Combining a natural-language stepwise reasoning modality (CoT) with a symbolic/programmatic modality (PoT) aligns with an intuitive notion of complementary error modes and increased diversity of reasoning traces. The framework captures this complementarity succinctly and leverages it for more efficient ensemble decisions.

**Potentially general framework for multimodal or heterogeneous reasoning:**
The paper presents a coherent probabilistic (Bayesian) framework that operationalizes how cross-modal agreement can be interpreted as a confidence signal. Although the experiments focus on CoT and PoT, the proposed Bayesian agreement mechanism could, in principle, be extended to any cross-modality reasoning setting, where agreement between different reasoning forms (e.g., textual, symbolic, or formal) can serve as a confidence signal for ensemble consistency.

**Empirical performance:**
 The method achieves remarkable efficiency improvements, solving the majority of tasks with only two samples while maintaining high accuracy.

### Weaknesses
**Assumption about program decomposability / limits of PoT usage.**
The approach leverages PoT outputs as one reasoning modality and implicitly assumes that tasks can be represented or approximated programmatically. While the Case Study discusses weaker PoT capabilities in smaller models and introduces a self-induced PoT variant, it does not address tasks fundamentally unsuited to programmatic reasoning (e.g., tasks that do not decompose naturally into executable programs or symbolic forms), leaving the framework’s broader applicability unclear.

**Missing comparison with advanced reasoning frameworks:**
The paper does not discuss or compare against other recent multi-step reasoning paradigms such as ReAct, Reflexion, or tool-augmented reasoning, which might also address efficiency and correctness issues at inference time.

**Limited generalization analysis:**
The Bayesian model depends on seed probabilities $(c, a_1, a_2)$ estimated from data. For the data-independent variant, the method assumes $a_2 \approx 1$ (i.e., agreement almost guarantees correctness). It is not demonstrated whether these parameters—or this assumption—generalize across diverse tasks or models.

### Questions
**1. Applicability:**
For problem classes that are not easily representable as executable programs, what is the expected behavior of the CoT–PoT ensembling framework? Do you observe failure modes where PoT is systematically inapplicable or misleading, and how should practitioners detect or mitigate such cases?

**2. Generalization of the Bayesian model:**
The Bayesian scheme uses seed probabilities estimated from held-out data. Can these learned parameters $(c, a_1, a_2)$ be transferred between datasets, tasks, or model families? If not, how sensitive is method performance to mismatches between training-held-out data and test distributions, and do you have practical recommendations for re-estimating or adapting these priors in low-data regimes?

**3. Consistent error reduction:**
Have you evaluated whether cross-modal agreement reduces the incidence of self-consistent but incorrect outputs (i.e., situations where multiple samples agree on a wrong answer)? In particular, can you relate your empirical findings to the phenomenon discussed in “Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs”—does cross-modal ensembling mitigate or merely shift such failure modes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a method that combines CoT and PoT reasoning through majority voting. Experimental results demonstrate that this approach enhances reasoning accuracy and outperforms traditional majority voting techniques.

### Strengths
1. The experimental setup is well-designed and yields strong results.

2. The paper is clearly written and easy to follow.

### Weaknesses
1. The idea presented in the paper is not particularly novel. Prior work from two years ago has already explored combining CoT and PoT [1]. Moreover, the combination of only CoT and PoT offers limited contribution, as there are several other reasoning methods, such as ToT [2], that could be considered. It would be more valuable to investigate the integration of a broader range of reasoning approaches.


2. The PoT method is relatively outdated, and numerous recent approaches have advanced the use of code in reasoning—particularly with the emergence of reinforcement learning techniques, such as ReTool [3], as well as several deep research efforts. Given these developments, combining two older reasoning methods like CoT and PoT offers limited relevance and impact in the current landscape.

[1] Automatic Model Selection with Large Language Models for Reasoning. EMNLP 2023

[2] Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023

[3] ReTool: Reinforcement Learning for Strategic Tool Use in LLMs. Arxiv 2025

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes to combine chain-of-thought and program-of-thought as self-consistency measure. Experiments indicate that ensembling CoT-PoT improves accuracy, and is a more efficient approach.

### Strengths
N/A The paper writing is clear, experiments are in general thorough.

### Weaknesses
Unfortunately, I believe this paper is currently below the acceptance bar for the conference. Rather than continuing to refine it for submission, I would suggest that the authors consider discontinuing this project or substantially rethinking its core idea.

First, the combination of CoT and PoT reasoning is a rather straightforward extension, and similar attempts have been explored several years ago. Conceptually, PoT does not provide a complementary signal to CoT. That is to say, if a problem cannot be solved by PoT, it is unlikely that CoT alone would succeed either.
Second, the experimental results show marginal improvements from combining CoT and PoT compared with existing approaches, validating what my understanding of this project.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2
