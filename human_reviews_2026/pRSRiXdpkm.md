# CORE: Concept-Oriented Reinforcement for Bridging the Definition–Application Gap in Mathematical Reasoning

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Large language models (LLMs) often solve challenging math exercises yet fail to apply the concept right when the problem requires genuine understanding. Popular Reinforcement Learning with Verifiable Rewards (RLVR) pipelines reinforce final answers but provide little fine-grained conceptual signal, so models improve at pattern reuse rather than conceptual applications. We introduce $\textit{CORE}$ (Concept-Oriented REinforcement), an RL training framework that turns explicit concepts into a controllable supervision signal. Starting from a high-quality, low-contamination textbook resource that links verifiable exercises to concise concept descriptions, we run a sanity probe showing LLMs can restate definitions but fail concept-linked quizzes, quantifying the conceptual reasoning gap. $\textit{CORE}$ then (i) synthesizes concept-aligned quizzes, (ii) injects brief concept snippets during rollouts to elicit concept-primed trajectories, and (iii) reinforces conceptual reasoning via trajectory replacement after group failures, a lightweight forward-KL constraint that aligns unguided with concept-primed policies, or standard GRPO directly on concept-aligned quizzes. Across several models, $\textit{CORE}$ delivers consistent gains over vanilla and SFT baselines on both in-domain concept--exercise suites and diverse out-of-domain math benchmarks. $\textit{CORE}$ unifies direct training on concept-aligned quizzes and concept-injected rollouts under outcome regularization. It provides fine-grained conceptual supervision that bridges problem-solving competence and genuine conceptual reasoning, while remaining algorithm- and verifier-agnostic.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed the make a new math dataset that, in addition to question and answer pairs, augments related concepts by extracting and translating content from a Chinese math book. The authors then compared 3 GRPO based methods to show the effectiveness of their proposal.

Although the paper claims in the abstract that CORE is "an algorithm-agnostic training framework that turns explicit concepts ...", the fact that its algorithm relies on the $N$ generation failures suggests it is a derivation of GRPO, which invalidates the claim. In addition, I have 2 major concerns.

### Strengths
* The paper is tackling an important problem: 
* While some details are hard to get (e.g., Sec 3.4), the paper is generally well-written.

### Weaknesses
* Unclear technical novelty of the RL method vs. the training data. While the paper emphasizes the goal of bridging the 'definition-application gap', its primary technical contribution (core-cr and core-kl) appears to be an application of existing RL strategies rather than a novel method. The core mechanism involves dynamically intervening with expert guidance when the policy fails. This is methodologically similar to prior work on "teacher" or "expert-anchored" RL. The paper's novelty therefore seems to be rest on the content of the intervention rather than the mechanism. But are the performance gains attributable to the unique properties of using a declarative concept, or would any high-quality reasoning trace produce the same benefit when used within this dynamic-intervention framework? The paper lacks an ablation study to disentangle the value of the content from the value of the method.
* Generalizability and conflation of methodological contribution with data curation. The method section describes a non-trivial, multi-stage curation pipeline to create the data, involving generations by Qwen2.5-72B-Instruct, vadilation and filtering by GPT4o, cross-model validation strategy, etc. This rigorous process suggests that the effectiveness of the proposed framework is linked to the quality fo the dataset. Actually, the core-base model, trained on this data, already shows significant improvement over the baselines. When comparing the performance boost after applying core-cr and core-kl, only TB and CM showed meaningful improvements. This thus questions if a large portion of the gains come from the data itself rather than the specific CORE-CR or CORE-KL intervention recipes.

### Questions
See details in the weaknesses section.

### Soundness
2

### Presentation
3

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
This paper introduces CORE, an RL training framework that turns explicit math concepts into a controllable supervision signal. The proposed approach comprises three key components: dataset curation, gap diagnostics, and a concept reinforcement recipe. The framework is evaluated on two 7B models across multiple mathematical reasoning benchmarks, demonstrating clear effectiveness under the given experimental setup.

### Strengths
1. The paper targets an important research problem: LLMs often solve math problems by pattern matching rather than genuine conceptual understanding

2. The proposed approach, including the curated dataset and the CORE-KL framework, demonstrates clear value and effectiveness.

### Weaknesses
1. My main concern lies in the experimental setup. Although the proposed method shows effectiveness under the reported setting, the evaluations are primarily conducted on relatively simple or medium-difficulty math benchmarks. Moreover, the baseline results (Table 2) are inconsistent with those reported in the corresponding technical reports. For example, the performance of Qwen2-Math-7B on GSM8K and MATH does not match, and it is unclear why the model performs significantly worse on the simpler GSM8K compared to the more challenging MATH.

2. The RL training set is synthesized and evaluated by LLMs, raising concerns about data quality and reliability. It remains unclear how the authors ensure the correctness of the training data and mitigate potential issues such as reward hacking when using LLMs as judges.

### Questions
See the weaknesses section

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
This paper presents concept-oriented approach to create synthetic data based on math concepts. The initial diagnostic evaluation results show that base models, while perform well in standard evaluation, they are prone to error when tested with variation inputs (that permutes parameters). The paper then presents an RL based approach to train models based on guidance from concepts in events of reasoning divergence. The results show that Core RL approaches can generalize beyond the synthetic puzzles.

### Strengths
- puzzle dataset construction based on math text book
- diagnostic analysis about issues of why current models could fail in truly understand math concepts.
- RL frameworks to train models to be coherent with concepts with different reward models
- evaluation shows out of domain generalization

### Weaknesses
- difference in RL methods CR / KL have some improvement over the baseline RL-base, it is not clear how these methods truly improve on top of baseline RL approach.
- (neutral comment) another simple baseline of enhancing models with variation data (e.g., with permuted params) could also be used to understand the wether concept reasoning is important, or it is just a matter of data augmentation is needed.

### Questions
Please elaborate comparison of three RL approaches, and how they differs in contribution to performance gains. It's good to understand importance of augmented data vs new RL methods.

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
This paper focuses on the definition–application gap in large language models (LLMs) for mathematical reasoning, ie, models can accurately recite formal definitions but often fail to select and apply the correct concepts when solving problems.

To address this issue, the authors curate a structured concept–exercise corpus from a linear algebra textbook, containing 236 concept definitions, 703 examples, and 140 exercises. They further construct a Concept Probes dataset of 1,110 multiple-choice quizzes by generating items with Qwen2.5-72B and filtering them through GPT-4o for quality assurance. This dataset serves both diagnostic and training purposes.

Building on GRPO reinforcement learning, the authors propose CORE (Concept-Oriented REinforcement) through three variants:

1. CORE-Base: standard RL training directly on the concept-probe dataset.

2. CORE-CR: when all sampled trajectories fail, failed ones are partially replaced with concept-guided rollouts and assigned a reward bonus.

3. CORE-KL: upon failure, the model minimizes the forward KL divergence between the concept-guided and unguided policies, effectively distilling the reasoning distribution that emerges when a concept is provided.

Experiments on Qwen2-Math-7B and Llama-3-8B-Instruct show that all CORE variants consistently improve both in-domain textbook exercises and multiple out-of-domain mathematical reasoning benchmarks, demonstrating enhanced concept selection and application.

### Strengths
The paper introduces concept texts as a controllable and fine-grained supervision signal in reinforcement learning, offering richer and more directed feedback compared to traditional scalar rewards based solely on final answer correctness.


Experiments are conducted on both base and instruction-tuned models across multiple, stylistically diverse mathematical benchmarks, demonstrating the generality and robustness of the proposed approach.


Through small-scale manual case analysis and perturbation tests with irrelevant concepts, the authors provide preliminary evidence that the performance gains stem from improved concept selection and application rather than superficial heuristic matching.

### Weaknesses
The comparison with relevant baselines is insufficient: the paper does not include fair experimental comparisons with conceptually similar methods such as BREAD (which also employs failure-branch replacement), nor with other related process-supervision or verifier-guided RL approaches. As a result, the marginal contribution and uniqueness of CORE remain unclear.

Most reported performance improvements are relatively small and lack statistical significance.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
2
