# Rewriting Pre-Training Data Boosts LLM Performance in Math and Code

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
The performance of large language models (LLMs) in program synthesis and mathematical reasoning is fundamentally limited by the quality of their pre-training corpora.  
We introduce two openly licensed pre-training datasets, released under the Llama 3.3 Community License, that significantly enhance LLM performance by systematically rewriting public data. SwallowCode ($\approx$16.1 billion tokens) refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Unlike prior methods that rely on exclusionary filtering or limited transformations, our transform-and-retain approach refines low-quality code, maximizing data utility.
SwallowMath ($\approx$2.3 billion tokens) enhances Finemath-4+ by removing boilerplate, restoring context, and reformatting solutions into concise, step-by-step explanations. Within a fixed 50 billion token training budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +16.1 on HumanEval+ compared to Stack-Edu, surpassing the baseline model's code generation capabilities. Similarly, substituting SwallowMath yields +12.4 accuracy on GSM8K and +7.6 on MATH. Ablation studies confirm that each pipeline stage contributes incrementally, with rewriting yielding the largest gains.
By releasing datasets, prompts, checkpoints, and pipeline code, we ensure reproducibility and provide a transferable transform-and-retain methodology that can be adapted to other base models and LLM rewriting setups.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Experiments show that continued pretraining using the resulting dataset improves performance of LLMs on several benchmarks.

### Strengths
1. The writing is clear and easy to follow
2. They provided the code for the data processing pipeline, improving the reproducibility of the work

### Weaknesses
1. The novelty seems limited. Using LLMs to rewrite data for continued pretraining has been adopted by previous works such as Qwen2.5-Math. The proposed method seems more like minor engineering tricks than major algorithmic novelty.
2. The proposed method was only compared to coarse pretraining datasets such as Stack v1 and Stack v2. Comparison between SwallowMath and other more carefully filtered and processed datasets such as OpenWebMath [1], Lemma [2], and MathCoder2 [3] should be considered as well.

3. The code only contains Python snippets. It is unclear whether it can improve the models’ coding abilities of other programming languages.

[1] Paster, Keiran, et al. "Openwebmath: An open dataset of high-quality mathematical web text." arXiv preprint arXiv:2310.06786 (2023).

[2] Azerbayev, Zhangir, et al. "Llemma: An open language model for mathematics." arXiv preprint arXiv:2310.10631 (2023).

[3] Lu, Zimu, et al. "Mathcoder2: Better math reasoning from continued pretraining on model-translated mathematical code." arXiv preprint arXiv:2410.08196 (2024).

### Questions
See weaknesses.

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
3

### Summary
Transform-and-retain rewriting of public corpora using a two-stage LLM pipeline, rather than exclusionary filtering. For code: Style-Guided Code Rewriting (SGCR) followed by Self-Contained Optimization Rewriting (SCOR). For math: rewrite Finemath-4+ to remove boilerplate, restore context, and produce concise step-by-step solutions.

### Strengths
- Thoughtful analysis of why synthetic-from-scratch may underperform due to diversity issues.
- Decontamination checks and cross-model validation (Qwen2-7B) improve credibility.
- Open release of data, prompts, and checkpoints enhances community value and reproducibility.

### Weaknesses
- generality claims are suggestive but not demonstrated across languages or larger scales.
- no analysis of downstream generalization outside HumanEval/+.
- No quantitative quality checks on rewritten outputs (e.g., compile/run rate, test pass rate, semantic drift)—risk of introducing hallucinated correctness.

### Questions
- What proportion of rewritten code compiles and runs? Any automated test execution stats on a held-out suite?
- For SCOR, how often does algorithmic “optimization” reduce correctness (e.g., edge cases)? Any spot-audit or unit-test sampling?
- Any evidence the gains persist or improve at larger pre-training budgets?

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
2

### Summary
This paper introduces SwallowCode and SwallowMath, two openly licensed datasets created by systematically rewriting existing public corpora using a multi-stage LLM-driven pipeline. The approach combines filtering (syntax and linter-based) with two-stage rewriting (SGCR for style, SCOR for self-containment and optimization) to enhance data quality. Continual pre-training of models like Llama-3.1-8B and Qwen2-7B on these datasets shows significant gains on code and math benchmarks, outperforming existing datasets like Stack-Edu and Finemath-4+.

### Strengths
* This paper releases two open source datasets under permissive licenses, supporting community reuse and extension.


* The ablation studies isolate the impact of each pipeline stage, demonstrating clear and interpretable improvements.

### Weaknesses
* The success and ceiling of this method are fundamentally limited by the capabilities and potential biases of the powerful LLM. Rewriting relies on Llama-3.3-70B-Instruct, which may introduce its own stylistic or semantic biases.



* While the paper claims rewriting enforces "algorithmic efficiency," it is not clear how the paper quantifies or measures this specific gain. Providing concrete metrics or examples of complexity improvements would strengthen this claim.

### Questions
The SwallowCode pipeline utilizes a two-stage LLM rewriting process (one for style, one for efficiency/self-containment). Why was this split necessary? Did the authors attempt a comparative study using a single-stage LLM prompt designed to achieve all stylistic and functional improvements simultaneously, and if so, what was the performance degradation compared to the more complex two-stage approach?

### Soundness
2

### Presentation
2

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
The manuscript introduces two novel openly licensed datasets — SwallowCode and SwallowMath — designed to enhance LLM performance in code synthesis and mathematical reasoning. Both datasets apply a transform-and-retain methodology: instead of filtering out low-quality samples, the authors employ LLM-driven rewriting to refine existing corpora. Extensive ablations show significant gains on HumanEval, HumanEval+, GSM8K, and MATH benchmarks.

### Strengths
The method of rewriting is well-motivated with sufficient ablation studies of different data-filtering methods.

The improvements are substantial and consistent across multiple benchmarks (HumanEval, HumanEval+, GSM8K, MATH), with comprehensive ablation studies demonstrating that each pipeline stage contributes incrementally.

The paper is well-written and easy to follow.

### Weaknesses
The novelty is somewhat limited. The paper reads more as an engineering work in improving pre-training performance with careful experimental design rather than introducing new ideas. The author should try to clarify the unique contribution and insight of this paper to academic community.

### Questions
Are there other LLM rewriting techniques to compare with the proposed method? The comparison could include quality and training results (if applicable).

Another question is whether the proposed rewriting method is limited by the rewriting LLM, meaning the dataset would need frequent updates.

Could the method be applied iteratively?

### Soundness
3

### Presentation
3

### Contribution
3
