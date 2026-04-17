# When Names Disappear: Revealing What LLMs Actually Understand About Code

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Large Language Models (LLMs) achieve strong results on code tasks, but how they derive program meaning remains unclear. We argue that code communicates through two channels: structural semantics, which define formal behavior, and human-interpretable naming, which conveys intent. Removing the naming channel severely degrades intent-level tasks such as summarization, where models regress to line-by-line descriptions. Surprisingly, we also observe consistent reductions on execution tasks that should depend only on structure, revealing that current benchmarks reward memorization of naming patterns rather than genuine semantic reasoning. To disentangle these effects, we introduce a suite of semantics-preserving obfuscations and show that they expose identifier leakage across both summarization and execution. Building on these insights, we release ClassEval-Obf, an obfuscation-enhanced benchmark that systematically suppresses naming cues while preserving behavior. Our results demonstrate that ClassEval-Obf reduces inflated performance gaps, weakens memorization shortcuts, and provides a more reliable basis for assessing LLMs’ code understanding and generalization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper separately evaluates LLMs on code understanding with two channels: the formal structure and the human-readable names (intent). To achieve this, the paper applies various semantic-preserving obfuscations, such as renaming variables with meaningless or misleading terms, and evaluates LLMs on two tasks: code summarization and execution prediction.

The results show that removing meaningful names leads to notable performance drop on both tasks, highlighting that LLMs often rely on identifier names to understand code, instead of performing actual semantic reasoning, and this issue cannot be revealed with current benchmarks. Therefore, the paper releases its benchmark with obfuscation as a more robust way to evaluate LLMs' capabilities on code understanding.

### Strengths
- **Methodology:** The idea of two-channel evaluation makes sense, as ideally we want an LLM that can robustly understand code behaviors without the reliance on human-readable identifier names which could be misleading and ambiguous. This paper tries to quantify such robustness.
- **Experimental Design:** The evaluation includes code understanding at two different granularity levels, the high-level intent and the low-level execution. The evaluation samples also cover two categories, one is about competitive programming and the other is not, showing differences between them.
- **Empirical Results:** The paper presents several empirical results, including: (1) obfuscation leads to worse performance across various settings, (2) larger models are more robust. These results raise notable concerns about the true reasoning capability of LLMs.

### Weaknesses
- **Limited Novelty:** The community already acknowledges the lack of deeper code semantics understanding of current code LLMs, and using obfuscation to stress-test code LLMs is not new (e.g. [NIKIEMA et al. 2025](), [Fang et al. 2024](https://www.usenix.org/system/files/usenixsecurity24-fang.pdf)). Previous works also explore obfuscation beyond renaming variable names, such as refactoring control flows. All these papers show similar results, and I do not see enough new insights in this paper.
- **Insufficient Evaluation Efforts:** There is no evaluation on SOTA reasoning models, which can use more CoT steps to conduct more complicated analysis.
- **No Practical Guidance on Improvements:** This paper does not propose potential approaches to improve the robustness of code semantics understanding. Actually, the relatively small performance gap of code summarization on LiveCodeBench implies that a simple data-driven approach is likely to address the problem. That is, if we add more training data with obfuscated variable names, then the model will be more robust on such obfuscation-based evaluations. In fact, there are already some works proving this ([Paul et al. 2025](https://openreview.net/forum?id=VYvxrD7aS0)). Therefore, the evaluation proposed here does not point out a novel and challenging enough problem that the community has difficulty with, and provides limited new practical guidance on how to build better code LLMs for real-world coding tasks.

### Questions
Please see the section of weaknesses above, where the concerns are stated.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper examines what LLMs truly “understand” about code by separating a structural semantic channel from a human naturalness channel. The authors evaluate models with a suite of semantics preserving obfuscations that progressively suppress identifiers and prose while keeping behavior unchanged. Across intent summarization and execution prediction, and across real world and competitive programming data, they find a consistent graded pattern: intent level performance declines sharply as naturalness is removed, while behavior level metrics remain largely stable except when names carry semantic content. The results offer converging empirical support for a two channel account of code understanding and motivate reporting before and after obfuscation deltas alongside human aligned intent metrics.

### Strengths
1. The paper’s starting point is reasonable. Constructing harder settings to test model robustness and generalization is necessary and can, to some extent, reflect a model’s generalization and resilience to misleading cues.
2. The work compares against cutting edge models, and the experimental results support the conclusions.

### Weaknesses
1. Does not assess whether the new evaluation correlates with scaling as model size increases.
2. The new evaluation introduces challenges, but can these be addressed quickly and easily with synthetic data? If a bit of synthetic data can solve it, the significance of this setting would be greatly reduced.
3. How to demonstrate that this evaluation helps real world coding tasks? After all, the real world does not replace variable names wholesale with meaningless symbols.

### Questions
please refer to weakness

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
This paper investigates the extent to which Large Language Models (LLMs) rely on human-interpretable identifier names versus structural semantics to understand code. The authors argue that code communicates through 2 distinct channels:

(1) A formal, structural channel that entails formal semantics

(2) A "naturalness" channel that conveys intent through naming and other linguistic cues.

To disentangle these two channels, the authors introduce a suite of semantics-preserving obfuscation techniques that systematically degrade or remove identifier names while leaving the program’s underlying logic intact. They evaluate the performance of several state-of-the-art LLMs on 2 complementary tasks: code summarization and code execution prediction, using both their original and obfuscated versions. The results show that on intent-rich code, summarization performance collapses when names are removed, with models regressing to simple line-by-line descriptions. More surprisingly, the authors also observe a consistent performance drop on execution prediction tasks, which should theoretically only depend on the code’s structure. This suggests that current benchmarks may inadvertently reward memorization of naming patterns rather than genuine semantic reasoning.

### Strengths
## Novelty

**1. Comprehensive Obfuscation Suite**

While the core technique of identifier obfuscation is not new, this paper stands out for its systematic and multi-faceted approach. The introduction of a *suite* of semantics-preserving obfuscations, ranging from simple alpha-renaming to the use of ambiguous identifiers, cross-domain terms, and misleading semantics, provides a more granular and insightful analysis than previous studies that have typically focused on a single type of perturbation.

## Soundness

**1. Complementary Dataset Choices**

The authors' choice to evaluate models on both intent-rich, real-world code (ClassEval) and algorithmic, competitive-programming code (LiveCodeBench) is a key strength of the study. This allows them to demonstrate that the impact of naming is not uniform across all types of code, but rather depends on the extent to which the code relies on the "naturalness" channel for conveying intent.

### Weaknesses
## Novelty

**1. Insufficient Differentiation from Extensive Prior Work**

The paper presents variable name obfuscation as a novel framework for evaluating LLM code understanding, but this approach has been extensively explored in prior work. Gao et al. (2023) introduced the CREAM framework, which uses counterfactual reasoning to separate helpful from misleading identifier signals and explicitly employs identifier renaming to test model robustness [1]. Wang et al. (2022) developed ReCode, a comprehensive robustness evaluation benchmark that includes multiple schemes of variable renaming as semantics-preserving transformations [2]. Lam et al. (2025) proposed CodeCrash, which applies systematic renaming among other logic-preserving perturbations to expose over-reliance on natural language cues [3]. Even earlier, Rabin et al. (2021) studied the generalizability of neural program models with respect to semantic-preserving transformations, including identifier renaming [4]. The core technique of using identifier obfuscation to test semantic understanding is well-established in the literature. While the paper mentions CREAM and CodeCrash, it does not provide a detailed comparison to clearly delineate what is genuinely novel beyond applying an existing technique to a slightly different set of tasks and models.

**Suggestion:** It would be better to include a dedicated subsection that explicitly compares the proposed approach with CREAM, ReCode, CodeCrash, and other prior work on semantic-preserving transformations. The authors should clearly articulate what specific methodological innovations or insights their work provides beyond the existing literature, rather than presenting identifier obfuscation as if it were a new concept.

## Soundness

**1. Fundamental Flaw in Summarization Task**

The paper's use of code summarization as a proxy for "intent-level understanding" suffers from a critical methodological flaw that undermines the validity of the conclusions. The ground truth summaries almost certainly contain information that can only be inferred from variable names, not from program structure alone. For example, as illustrated in Figure 1, after name obfuscation, there is no way to determine that the original program implements a "Minesweeper game" rather than any other grid-based application with identical control flow—it could equally represent a spreadsheet calculator, a Conway's Game of Life implementation, or a warehouse inventory system. Similarly, a simple multiplication function `result = a * b` could represent price × quantity, discount × price, or units × count—all implemented identically but with entirely different semantic meanings conveyed solely through naming. When the ground truth summary includes domain-specific concepts like "Minesweeper" or "pricing calculation," the model is being evaluated on information that has been deliberately removed from the input. This creates an inherently unsolvable task after obfuscation, making the performance drop inevitable and uninformative about the model's structural reasoning capabilities. The evaluation would only be sound if the ground truth summaries were verified to be abstract enough that they contain no information implied by variable naming—a condition the paper does not establish or enforce.

**Suggestion:** The authors should either (1) create a new set of ground truth summaries that are provably derivable from structure alone, excluding any domain-specific or application-level concepts that require naming cues, or (2) acknowledge this as a fundamental limitation and reframe the summarization results as measuring "naming-dependent intent recovery" rather than "structural understanding." Additionally, a human evaluation study could assess whether the obfuscated summaries produced by models are actually incorrect or simply more abstract than the naming-rich ground truth, which would help clarify whether the performance drop reflects a genuine failure or an artifact of the evaluation design.


**2. Incomplete Analysis of Execution Prediction Results**

While the paper shows that execution prediction accuracy drops under obfuscation, this finding does not conclusively demonstrate a fundamental flaw in the models' understanding of program semantics. An alternative and equally plausible interpretation is that descriptive variable names simply provide shortcuts that make correct predictions easier to generate on the first attempt, without implying that the model lacks the capability to reason through the obfuscated version given more attempts. If the authors had evaluated Pass@5, Pass@10, or higher values of k instead of only Pass@1 and Pass@3, it is quite possible that the accuracy on obfuscated programs would converge to that of original programs. Such convergence would suggest that naming provides efficiency gains (reducing the number of attempts needed to find the correct answer) rather than revealing a fundamental reasoning deficit. The current experimental design does not rule out this possibility, which significantly weakens the claim that models are "memorizing naming patterns rather than genuinely reasoning about semantics." Without exploring higher values of k or analyzing the distribution of correct answers across multiple samples, the interpretation remains ambiguous and the conclusions overstated.

**Suggestion:** It would be worthwhile to extend the evaluation to Pass@5, Pass@10, or even Pass@20 to investigate whether performance on obfuscated code converges to that of the original code. Additionally, analyzing the variance and consistency of predictions across multiple samples could help distinguish between "naming as a shortcut" and "naming as a prerequisite for reasoning." If convergence is observed, the authors should revise their claims to acknowledge that the results may reflect efficiency differences rather than fundamental capability gaps. This would provide a more nuanced and accurate understanding of the role of identifiers in execution prediction.

**3. Limited Model Diversity Undermines Generalizability**

The paper evaluates only four models (GPT-4o, Qwen3-Coder 480B, DeepSeek V3, Llama 4 Maverick), which is insufficient to support broad claims about how LLMs understand code. To establish that the observed phenomena are not artifacts of specific training procedures or architectures, the evaluation should include systematic variation across multiple dimensions. Specifically, the study lacks: (1) multiple models from the same family at different scales (e.g., 7B, 13B, 70B versions of Llama or Qwen) to assess whether the reliance on naming is scale-dependent, and (2) explicit comparison between reasoning models versus standard models (Qwen 3 in thinking mode v.s. non-thinking mode) from the same family to determine whether reasoning capabilities mitigate the naming dependency. Without this systematic exploration, it remains unclear whether the findings generalize across the broader landscape of code LLMs or are specific to the particular models chosen.



## Significance

**1. Underexplored Practical Implications**

The paper's findings have important implications for the development and evaluation of code LLMs, but the discussion of these implications remains somewhat abstract. The paper concludes that its new benchmark will provide a more reliable basis for assessment, but it could offer more specific, actionable recommendations. For example, how should developers change their coding practices in light of these findings? Should they prioritize more descriptive names, or should they focus on writing code with clearer structure? How can LLM developers use these insights to build more robust models—should they augment training data with obfuscated code, modify the training objective, or employ different architectural choices? The lack of concrete guidance limits the practical impact of the work.

**Suggestion:** It might be worthwhile to add a discussion section that explores the practical implications of the findings in more detail. This could include concrete recommendations for developers on how to write code that is more amenable to LLM analysis (or conversely, how to recognize when LLMs may struggle) and for researchers on specific training strategies or architectural modifications that could help LLMs become more robust to identifier obfuscation while maintaining high performance on naturally-named code.


## Effectiveness

**1. Unvalidated Real-World Utility of the Proposed Benchmark**

The paper convincingly shows that CLASSEVAL-OBF can mitigate the problem of identifier leakage and provide a more accurate measure of a model's reasoning ability in a controlled experimental setting. However, the ultimate goal of a benchmark is to predict a model's performance on real-world tasks. The paper does not provide any evidence to show that performance on CLASSEVAL-OBF is a better predictor of downstream task performance (such as bug fixing, code generation in real-world software projects, or code review) than existing benchmarks. Without this validation, it remains unclear whether optimizing for CLASSEVAL-OBF performance would actually lead to models that are more useful in practice, or whether it might inadvertently optimize for a different set of capabilities that are less relevant to real-world applications.

**Suggestion:** It would be better to include an experiment that investigates the correlation between performance on CLASSEVAL-OBF and performance on downstream tasks such as bug fixing, code generation in real-world software projects, or repository-level code understanding. This would provide stronger evidence for the practical utility of the proposed benchmark and help establish that it measures capabilities that matter for real-world applications.



## References

[1] [Two sides of the same coin: Exploiting the impact of identifiers in neural code comprehension (Gao et al., 2023)](https://ieeexplore.ieee.org/document/10172869/)

[2] [ReCode: Robustness Evaluation of Code Generation Models (Wang et al., 2022)](https://arxiv.org/abs/2212.10264)

[3] [CodeCrash: Stress Testing LLM Reasoning under Structural and Semantic Perturbations (Lam et al., 2025)](https://arxiv.org/abs/2504.14119)

[4] [On the generalizability of Neural Program Models with respect to semantic-preserving program transformations (Rabin et al., 2021)](https://www.sciencedirect.com/science/article/pii/S0950584921000379)

### Questions
## Clarification Questions

1. In Section 5.1, you describe four obfuscation strategies. Could you provide more detail on how the "cross-domain terms" and "misleading semantics" were generated? Was this done manually or automatically? If automatically, what was the process?

2. The LLM-as-a-judge evaluation for summarization is based on a rubric from the BigGenBench framework. Could you elaborate on the specific criteria used in this rubric and how they were weighted to produce the final scores?


## Discussion Questions

1. Given the fundamental tension between obfuscating names and evaluating against naming-dependent ground truth summaries, do you believe the summarization task can be salvaged (e.g., by creating structure-only ground truth), or should it be replaced with a different task that is provably solvable from structure alone?

2. If Pass@10 or Pass@20 on obfuscated code converges to the performance on original code, would you still interpret the results as indicating a fundamental flaw in LLM reasoning, or would you agree that naming primarily provides efficiency shortcuts?

3. Your work focuses on analyzing the understanding of existing code. How do you think these findings might apply to the task of code *generation*? Would a model that is less reliant on naming cues be better at generating novel code that solves new problems, or might it struggle to produce code that is readable and maintainable?

4. What specific training strategies or architectural modifications do you think could help LLMs become more robust to identifier obfuscation while maintaining high performance on naturally-named code? For example, should training data be augmented with obfuscated examples, or should the model architecture be modified to better capture structural semantics?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a code understanding benchmark to probe the semantic understanding of Code LMs by subjecting it to summarization and output prediction tasks on obfuscated code generated from well-known benchmarks.

### Strengths
1. The authors have explored a taxonomy of obfuscation types
2. The obfuscation codes are sourced from code snippets that are solutions to relevant benchmarks that are not too easy and also commonly used for evaluating code LMs.

### Weaknesses
1. Some interesting ablations for gauging the effect of obfuscation on Code LMs is missing, which could enhance the paper:

1.1 The authors could have explored the effect of obfuscating various types of identifiers separately e.g. class names, function names, variables, etc
1.2 The authors should look into the effect of obfuscating varing proportions of identifiers. It would be interesting to note at what points the degradtions of code understanding set in.

2. Some very relevant existing work on training model's to be robust to these perturbations [1,2,3] is missing from the literature and discussions.

[1]	Marie-Anne Lachaux, Baptiste Rozière, Marc Szafraniec, Guillaume Lample: DOBF: A Deobfuscation Pre-Training Objective for Programming Languages. NeurIPS 2021: 14967-14979

[2] 	Indraneil Paul, Haoyi Yang, Goran Glavas, Kristian Kersting, Iryna Gurevych: ObscuraCoder: Powering Efficient Code LM Pre-Training Via Obfuscation Grounding. ICLR 2025

[3].   Seyedreza Mohseni, Seyedali Mohammadi, Deepa Tilwani, Yash Saxena, Gerald Ketu Ndawula, Sriram Vema, Edward Raff, Manas Gaur: Can LLMs Obfuscate Code? A Systematic Analysis of Large Language Models into Assembly Code Obfuscation. AAAI 2025: 24893-24901

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
3
