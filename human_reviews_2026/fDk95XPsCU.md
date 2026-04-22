# Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
While reinforcement learning with verifiable rewards (RLVR) is effective to improve the reasoning ability of large language models (LLMs), its reliance on human-annotated labels leads to the scaling up dilemma, especially for complex tasks. Recent self-rewarding methods investigate a label-free alternative to unlock the reasoning capabilities of LLMs, yet they frequently encounter the non-negligible training collapse issue, as the single-view supervision signal easily forms the self-consistent illusion, yielding the reward hacking. Inspired by the success of self-supervised learning, we propose \textit{Co-rewarding}, a novel self-supervised RL framework that improves training stability by seeking complementary supervision from another views. Specifically, we instantiate Co-rewarding in two ways: (1) \textit{Co-rewarding-I} is a data-side instantiation that derives reward signals from contrastive agreement across semantically analogous questions; and (2) \textit{Co-rewarding-II} is a model-side instantiation that maintains a slowly-updated reference teacher with pseudo labels to realize self-distillation. Intuitively, such instantiations introduce different levels of discrepancy to increase the difficulty of training collapse on trivial reasoning solutions. We also explore their orthogonally combined version to further boost the performance. Empirically, Co-rewarding exhibits stable training across various setups, and outperforms other self-rewarding baselines by $+3.31\%$ improvements on average on multiple mathematical reasoning benchmarks, especially by $+7.49\%$ on Llama-3.2-3B-Instruct. Notably, Co-rewarding reaches or even surpasses RLVR with ground-truth (GT) label in several cases, such as a Pass@$1$ of $94.01\%$ on GSM8K with Qwen3-8B-Base remarkably higher than GT. Our code is released at~\url{https://github.com/tmlr-group/Co-rewarding}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Co-rewarding, a self-supervised RL framework for eliciting LLM reasoning that aims to avoid the training collapse often seen in label-free/self-reward approaches. The core idea is to decouple the reward signal from the current policy's single-view outputs by enforcing invariance views across data-side and model-side. 

Co-rewarding-I (data-side) obtains contrastive agreement between semantically analogous (rephrased) questions and the original, using majority-vote pseudo-labels across rollouts to shape a GRPO-style objective $J_{\text{Co-rewarding-I}}$ with cross-refereed advantages (Eqs. (6)–(8)). Co-rewarding-II (model-side) reuses the GRPO reference model as a slowly updated teacher via EMA (cosine-scheduled $\alpha(k)$) to generate rollouts and majority-vote pseudo-labels, thus temporally decoupling supervision from the online policy (Eqs. (9)–(11)).

Across Qwen and Llama backbones, Co-rewarding improves stability and accuracy over recent self-reward baselines (Self-Certainty, Entropy, Majority Voting).

### Strengths
1. The instability of single-view self-reward (reward hacking -> collapse) is articulated and empirically supported, with Co-rewarding showing steadier progress and avoided collapse.
2. The idea is interesting, differs from (i) internal score rewards (entropy/certainty) and (ii) single-view self-consistency.
3. The method is simple to implement and seems to be quite practical, it reuses GRPO reference as teacher, and no extra LLM is introduced (beyond rephrasing for co-rewarding-I).
4. Good experimental results, achieves 94.01% GSM8K Pass@1 on DAPO-14k with Qwen3-8B-Base.

### Weaknesses
1. The paper shows co-rewarding-I collapses on DAPO-14k but II does not; the causes are hypothesized (rephrasing efficacy) rather than directly measured.
2. The use of Qwen3-32B to generate rephrases may raise a concern of equivalence/label leakage beyond examples; although instructions are provided and case studies look reasonable.
3. On code/IFEval/MMLU-Pro, improvements are mixed/modest, and there is little task-level analysis (e.g., error modes, solution length vs. correctness).
4. The mechanisms behind beating GT-reward is not dissected (e.g., improved exploration vs. label noise in GT-reward?).

### Questions
1. For I, can you quantify features (lexical diversity, named-entity variation, length) that predict stability across MATH vs DAPO-14k, validating the "rich background descriptions" hypothesis?
2. For AMC/CRUX/IFEval/MMLU-Pro, where gains are small, what failure modes are most common? Would task-aware rephrasing (e.g., code refactorings or spec restatements) strengthen I/II?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper defines two methods for unsupervised RL finetuning of LLMs for reasoning tasks. The first uses a similarity-based objective across semantic augmentations to train problems; the second uses majority voting across rollouts from a reference policy that periodically tracks the RL policy with an EMA update. The results show that these unsupervised reward signals stabilise training relative to other ground-truth-free baselines and can in some cases outperform using ground truth reward.

### Strengths
The paper sets out to define a stable and performant approach to RL finetuning for reasoning without use of ground truth labels and the results demonstrate the effectiveness of the proposed method. Three models are used across in-domain and out-of-domain benchmarks and two separate training datasets are considered. Combined with suitable ablations and analysis of training dynamics (length, reward) this makes for a comprehensive results section.

### Weaknesses
The training datasets used are ones for which ground truth labels are available. It seems important to validate the method in a setting that is better motivated by self-supervised methods (i.e., those without the availability of verifiable rewards during training). 

The written communication of the paper could be improved. There are grammatical errors throughout.

The related work currently features in the appendix and should be in the main paper text.

The paper primarily reports pass@1 results, which leaves the reliability of the results uncertain.

### Questions
Why not experiment with a combination of Co-rewarding-I and Co-rewarding-II?

Why do you think Co-rewarding sometimes outperforms using GT reward? This seems unintuitive and is not currently adequately discussed, only stated.

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
4

### Summary
This paper tackles the training collapse issue in self-rewarding RL for LLM reasoning. The authors propose Co-rewarding, a self-supervised RL framework that seeks complementary supervision for stability. It is a two-stage process where Co-rewarding-I uses "contrastive agreement" via cross-reference between original and rephrased questions and Co-rewarding-II employs self-distillation from a slowly-updated EMA "teacher model" to provide stable pseudo-labels, decoupling the reward signal from the online policy.

Experiments show Co-rewarding achieves stable training, avoiding the collapse seen in baselines like Entropy and Majority-Voting. The method outperforms these baselines and, notably, even surpasses training with ground-truth labels in several cases, achieving a 94.01% Pass@1 on GSM8K.

### Strengths
1.The paper's claims are supported by comprehensive experiments. The authors validate their method's effectiveness across a diverse range of models (Qwen series, Llama-3.2-3B-Instruct)and multiple training datasets (MATH, DAPO-14k, OpenRS). The evaluation is similarly thorough, spanning not only in-domain mathematical reasoning but also out-of-domain tasks like code generation and general abilities (MMLU-Pro, IFEval) . This extensive validation strongly supports the paper's conclusions.
2. A key strength of this work is the principled and general nature of the proposed Co-rewarding framework. The two instantiations are not ad-hoc fixes but are well-grounded in established, sound concepts: Co-rewarding-I draws its intuition from contrastive learning (analogy-invariance) , while Co-rewarding-II effectively implements a form of self-distillation (temporal invariance).

### Weaknesses
1. Although I like the idea of Co-rewarding-I, it seems not that effective. It relies on additional stronger model to revise the training data but offers marginal improvement in across both training datasets. Sometimes may harm the performance. The author tries to show the effectiveness of rewarding-I with a ablation study but the ablation is not fair in my view point, the fair comparasion should be a model trained on the union of orignal and rephrased instead of training on them seperately since the rewarding-I use both of the datasets.
2. The ablation study in Table 3 is confusing as it appears to conflate results from different training sets. Specifically, the Co-rewarding-I ablation data (e.g., "Majority-Voting w/ Original") matches the MATH-trained results from Table 1 , while the Co-rewarding-II ablation data ("w/o Updating Reference") matches the DAPO-14k-trained results from Table 2. This is not explicitly stated in the table's caption or the main text, making a direct comparison difficult.

### Questions
1. The evaluation omits high-difficulty, competition-level benchmarks (e.g., AIME 24/25). Given that the training set includes data from DAPO-Math-17k, why wasn't a benchmark like AIME used to more rigorously validate the model's advanced reasoning capabilities?
2. The design of Co-rewarding-II, which uses a dynamic, EMA-updated 'teacher' as the reference for the KL constraint , seems to violate the fundamental purpose of this term. Traditionally, the KL divergence penalizes deviation from a fixed reference policy, which acts as a stable anchor to prevent catastrophic forgetting . By using a 'moving anchor' that slowly follows the student policy, doesn't this design remove the primary safeguard against significant policy drift?
3. The paper presents Co-rewarding-I (data-side) and Co-rewarding-II (model-side) as two 'orthogonal' instantiations of the core framework. Given their complementary nature, was a combined 'Co-rewarding-I+II' method ever explored?

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
2

### Summary
The paper proposes Co-rewarding, a self-supervised reinforcement learning framework designed to improve the reasoning ability of large language models without relying on ground-truth labels. The method aims to prevent the collapse and reward-hacking issues common in existing self-rewarding approaches by introducing complementary supervision from two perspectives: (1) Co-rewarding-I, which derives reward signals through contrastive agreement across semantically rephrased questions, and (2) Co-rewarding-II, which employs a slowly updated teacher model to provide temporally decoupled pseudo-labels. Experiments across several reasoning benchmarks (GSM8K, MATH, AMC, etc.) show that Co-rewarding achieves more stable training and in some cases matches or surpasses RL with ground-truth rewards, especially when using Qwen-3 and Llama-3 models.

### Strengths
+ Introduces a clear and well-motivated framework that targets a real weakness in current self-rewarding RL methods: instability and collapse caused by single-view reward signals.

+ Extensive empirical results across multiple models, datasets, and baselines, including ablations that isolate the contribution of each component. 

+ The paper is overall well-written, with clear mathematical formulation and structured presentation.

### Weaknesses
+ The reliance on high-quality paraphrasing for Co-rewarding-I is insufficiently examined. The framework may degrade when rephrasing quality is low or domain-specific, yet no robustness analysis is provided. Although the paper claims that rephrased questions should yield similar reasoning outcomes, in practice the reasoning trace can vary a lot depending on how the question is phrased. Thus, how do the authors ensure the meta-transferability of both reasoning and final answers during rephrasing? Moreover, when paraphrasing leads to divergent results, how do the authors obtain the updated groundtruth?
 
+ Although positioned as scalable and label-free, the method still requires additional compute from rephrasing models and dual rollouts, but the paper does not report cost or latency analysis w.r.t. the baslines.

### Questions
1. The paper states that the two co-rewarding techniques stem from the same core insight but offer complementary perspectives for cross-supervision. However, it remains unclear whether combining them actually yields better performance than using either one alone. If yes, what is the underlying mechanism that makes the combination more effective than a single method?

2. The paper would benefit from additional case studies and quantitative evaluation of the rephrasing models, particularly under low-quality paraphrasing which may change the reasoning results. Since the method claims to stabilize RL, one would expect robustness to varying data-augmentation conditions. Can the authors analyze training stability when rephrasing degrades, and report how performance varies with different levels of paraphrasing noise?

### Soundness
2

### Presentation
3

### Contribution
2
