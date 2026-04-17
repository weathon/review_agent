# Can LLMs Reliably Evaluate Themselves? A Probabilistic VC Framework

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
As Large Language Models (LLMs) are increasingly deployed in autonomous reasoning tasks, the capacity to reliably evaluate their own outputs becomes paramount. We address this challenge by establishing a formal framework grounded in statistical learning theory. By operationalizing self-evaluation as a property of the hypothesis class induced by prompting strategies and stochastic decoding, we extend the classical Vapnik-Chervonenkis (VC) dimension to the probabilistic setting. We introduce two novel complexity measures: the Probabilistic VC (PVC) dimension, which quantifies the discriminative expressiveness of self-assessment, and the Calibration-aware PVC (C-PVC) dimension, which imposes a strict alignment constraint between confidence and correctness. In contrast to isolated calibration metrics, our unified framework provides integrated complexity measurements with provable generalization guarantees. A systematic evaluation of eleven 7--8B models across mathematical, factual, and commonsense domains highlights a fundamental trade-off: enhanced discriminative capacity systematically incurs a degradation in calibration quality. This structural tension suggests that current reasoning optimization paradigms do not implicitly resolve, and may exacerbate, miscalibration. Our framework offers the necessary diagnostic tools to quantify these risks, laying the groundwork for the development of trustworthy autonomous systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The abstract of this paper is excessively lengthy and fails to concisely communicate the key insights of the work. Consequently, the overall presentation does not meet the standards I expect.

This paper explores the ability of Large Language Models (LLMs) to reliably evaluate their own reasoning, specifically by distinguishing correct solutions from incorrect ones with well-calibrated confidence. To address this, the authors propose two novel theoretical frameworks: PVC and C-PVC.

### Strengths
The proposed theoretical framework is novel, however, its practical applications require further clarification. The paper would benefit from additional details and examples illustrating how PVC and C-PVC can be effectively utilized in real-world scenarios.

### Weaknesses
1. The paper lacks a comprehensive discussion of related work and baselines concerning the calibration of LLM reasoning. For practical methods, ASC [1] introduces a self-calibration approach across various benchmarks, including mathematics, counting, and intelligence tests. On the theoretical side, RPC [2] proposes a framework for analyzing LLM reasoning performance from the perspective of confidence estimation. The authors should thoroughly review and discuss these closely related works, either in the technical section or the related work section, to provide a more complete context for their research.
2. It is confusing why the authors focus on dataset categories in the experiments, which seems to be of limited relevance to practical applications. Therefore, it is important for the authors to clarify how the proposed theoretical framework addresses real-world problems and applications, and how it extends to different LLMs and reasoning paradigms.
3. The presentation of this paper requires significant improvement. As discussed in the `Summary` section, the abstract is overly long and fails to effectively highlight the main contributions. Moreover, there are typos, such as in Line 210, where a verb is missing after `Select_f(q)`.
4. As mentioned at the end of the abstract and contribution section, the paper claims to be critical for building reliable autonomous systems. However, I have not seen any related experiments to support this claim. In other words, how does this paper substantiate its importance for building reliable autonomous systems?

**Reference**

[1] Efficient Test-Time Scaling via Self-Calibration. Arxiv 2025.

[2] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning. NeurIPS 2025.

### Questions
Please refer to the `Weaknesses` section.

### Soundness
2

### Presentation
1

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
The paper introduces new complexity metrics based on Vapnik-Chervonenkis (VC) dimension theory suitable for Large Language Models (LLMs), which encode the ability of LLMs to classify its own answers correctly above a given confidence, and to be calibrated when doing so. Moreover,  the work establishes connections to existing metrics and a generalisation result based on the newly introduced ones, and empirically estimates there metrics for several LLMs.

### Strengths
- originality: although I am not closely familiar with VC theory and statistical learning theory, it seems to me that the introduced concepts are novel and insightful
- quality: extensive experimental setup, connection with other existing metrics, and large body of additional results in appendix
- significance: being able to bound generalisation for LLMs’ ability to assess their own answers is important; therefore, the aim of the paper is commendable

### Weaknesses
What mostly affects my scores negatively is the lack of clarity in some aspects. While I understand that this is a complex area of mathematics and I don’t possess full familiarity with the field (thus, my comments may be obvious to people more familiar with it), I found quite hard to understand some of the theoretical parts of the paper as well as their connection with the empirical part: 

- line 101: “with high probability”: it may be worth specifying what this means. Is the probability over the sample from a given distribution?
- the meaning of the “probability” term $\mathbb P$: in line 146, it seems that the probability is what $f$ assigns while, in line 159, it comes from the data distribution.
- the definition in line 159 conditions on $f(x)$ assigning confidence $p$, but $p$ is a continuous value. Are we not conditioning on an event with probability 0? How is this treated?
- I don’t understand $\hat p$ in Definition 3: why is this additional quantity needed? How is it linked to $f$? Shouldn’t the C-PVC dimension also depend on the way in which $\hat p$ is obtained from $f$?
- I don’t get how the VUS quantities are useful, as it seems they do not appear in the generalisation result
- I also don’t understand how the PVC and C-PVC are estimated from the datasets: how is the estimation procedure described in Sec 3.3 linked to the actual definitions? Those dimension metrics are defined as a max over set sizes for which there exists at least one set satisfying some property, and this seems hard to estimate in practice as, naively, we’d need to check all possible sets of a given size?
- Putting aside my confusion regarding the estimation of PVC and C-PVC, I am not sure what the experimental setup is aiming to do: is it attempting to validate the theoretical bounds between the PVC and C-PVC dimensions and generalisation? Or is it exploring for unrelated trends between those quantities and ECE/AE? If so, why is this interesting? I am asking as it seems to me that the PVC and C-PVC are not interesting by themselves, but rather they are only interesting as they appear in the generalisation bounds.

Minor comments:

- it seems to me that the abstract is longer than the traditional length for AI conferences; a shorter version may be more suitable to quickly let readers understand the goal and scope of the paper.
- line 116 uses $\epsilon$, which was already used in Sec 2.1 with an apparently different meaning. Moreovoer, line 117 includes $\epsilon_t$ which was not introduced
- Line 205-206: “External judge LLMs determine the objectively superior solution” this seems debatable. Why this reliance on a judge and not on checks to automatically determine solution correctness?

### Questions
- I am intrigued by the lack of anything similar to the PVC dimension before this paper: while I see this is needed for LLMs, LLMs are not special in producing probability distributions. Do the authors have any idea for this lack? Or, is there a 1-1 connection with the fat-shattering dimension? If so, was this obvious with other ML models, so that people did not feel the need to introduce something similar to the PVC?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates whether large language models (LLMs) can reliably evaluate their own outputs through a novel theoretical and empirical framework grounded in statistical learning theory. The authors propose two new measures: Probabilistic VC (PVC) and Calibration-aware PVC (C-PVC) to quantify a model’s self-evaluation expressiveness and calibration reliability. They provide probabilistic generalization bounds and validate the framework across 11 open-source 7–8B models on mathematical, factual, and commonsense reasoning benchmarks. The experiments reveal a trade-off between self-evaluation expressiveness and calibration quality, suggesting that stronger introspection often comes at the cost of confidence misalignment. Overall, the paper contributes an original theoretical perspective on model self-evaluation, offering tools for understanding and improving introspective reliability in future LLMs.

### Strengths
This paper presents an innovative and theoretically grounded framework for assessing the self-evaluation reliability of large language models (LLMs). The introduction of Probabilistic VC (PVC) and Calibration-aware PVC (C-PVC) extends classical VC theory to probabilistic predictors, offering a rigorous approach to quantify self-assessment expressiveness and calibration simultaneously. The combination of formal theoretical derivations, generalization bounds, and extensive empirical validation across 11 models and three reasoning domains makes this study both ambitious and timely. The motivation—understanding when models “know they are wrong”—aligns closely with the broader agenda of trustworthy and introspective AI systems.

### Weaknesses
1. The PVC and C-PVC definitions are conceptually appealing but insufficiently formalized; key probabilistic assumptions and measurable function spaces are not clearly defined.

2. The C-PVC bound lacks a complete derivation or closed-form upper bound, relying on intuition rather than proof.

3. The VUS (Volume Under Surface) metric is introduced without mathematical rigor—its integration domain and theoretical justification remain vague.

4. There is no statistical significance testing or confidence interval reporting, making it difficult to judge robustness.

5. The ground-truth evaluation relies solely on other LLMs (Claude, Nova, DeepSeek-R1) instead of human annotation, which weakens objectivity.

6. Prompt formats and sampling parameters differ across models, introducing uncontrolled variance.

7. Only 7–8B scale models are tested, limiting claims about scalability or generalization trends.

8. Ablation and scaling analyses are missing.

9. The paper contains a small typo (“sysem card” instead of “system card”).

### Questions
How does the proposed PVC metric behave under stochastic sampling noise—does it remain stable across decoding seeds?

Could the authors clarify whether PVC ≥ C-PVC is theoretically guaranteed or merely empirically observed?

What is the computational cost of estimating VUS and calibration surfaces, and how does it scale with model size or dataset size?

Have the authors tested the sensitivity of self-evaluation accuracy to prompt templates or confidence calibration methods?

Can the PVC framework be extended to multi-turn reasoning or tool-augmented LLMs where the “self” boundary is ambiguous?

Is there a plan to validate the framework with human-judged correctness or cross-model consensus instead of single-model baselines?

### Soundness
4

### Presentation
4

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
This paper studies whether LLMs can reliably evaluate their own outputs by proposing a theoretical framework based on statistical learning theory. The authors proposes the Probabilistic VC (PVC) and Calibration-aware PVC (C-PVC) dimensions to jointly quantify a model’s discriminative self-assessment ability and calibration fidelity. They derive generalization and sample-complexity bounds analogous to classical VC theory. Empirically, the study tests several 7/8B model, where each model must select between two of its own answers and rate its confidence. Authors show a consistent trade-off in results where models with higher discriminative expressiveness tend to be less calibrated.

### Strengths
- Introduces the Probabilistic VC (PVC) and Calibration-aware PVC (C-PVC) dimensions
- Connects discriminative self-assessment ability and calibration quality, with provable generalization and sample-complexity bounds
- Evaluates 11 open-source 7–8B models

### Weaknesses
Model choice of s1.1-7B seems questionable. s1K-1.1 is essentially a small and difficult SFT dataset targeting math reasoning, which is very effective for training large models (32B or larger) but causes significant performance degrade on small models without careful tuning and processing. Authors of s1 have also suggested using s1.1-32B instead of the 7B version (https://huggingface.co/simplescaling/s1.1-7B). The model's output traces could be excessively long and its accuracy (avg@k) could be low on benchmarks. 
- Besides the metrics presented in this paper, can the authors also present actual benchmark performance of each model? If the accuracy is too low, I don't think the metrics in the paper are enough to make the conclusion.

Benchmark choice is another problem. Math-360 seems to be a dataset constructed by the authors, and reading from Table 7, the questions seem very simple and synthetic. More standard benchmarks such as MATH, minerva math and AIME should be included. For the other two selected benchmarks (TruthfulQA and CommonsenseQA), it's also unclear how authors processed these data just by reading the sentence "we grouped each of the latter two datasets into 10 broad categories and sampled 240 questions per benchmark". More standard and transparent evaluation protocol is necessary here.

### Questions
n/a

### Soundness
2

### Presentation
3

### Contribution
2
