# DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Diffusion Language Models (DLMs) present a compelling alternative to autoregressive models, offering flexible, any-order infilling without specialized prompting design. However, their practical utility is blocked by a critical limitation: the requirement of a fixed-length masked sequence for generation. This constraint severely degrades code infilling performance when the predefined mask size mismatches the ideal completion length. 
To address this, we propose DreamOn, a novel diffusion framework that enables dynamic, variable-length generation. DreamOn augments the diffusion process with two length control states, allowing the model to autonomously expand or contract the output length based solely on its own predictions. We integrate this mechanism into existing DLMs with minimal modifications to the training objective and no architectural changes. 
Built upon Dream-Coder-7B and DiffuCoder-7B, DreamOn achieves infilling performance on par with state-of-the-art autoregressive models on HumanEval-Infilling and SantaCoder-FIM and matches oracle performance achieved with ground-truth length.
Our work removes a fundamental barrier to the practical deployment of DLMs, significantly advancing their flexibility and applicability for variable-length generation. 
Our code is available at https://github.com/DreamLM/DreamOn.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper solves the fixed-length mask limitation of code infilling diffusion models by introducing [expand] and [delete] tokens. The training process is adapted to first (synthetically) introduce these tokens in the sequence before embedding and predicting each token. Experiments show that diffusion models with these tokens perform significantly better than the original models, and even perform almost on par with auto-regressive models with infilling objective (pass @ 1).

### Strengths
- Controlling the size of the generation is a fundamental challenge in discrete diffusion models for text. Approaches that do not require (large) architectural changes can leverage pre-trained models more effectively.
- The ablations focus on relevant design decisions, such as not using either of the tokens.
- The paper is properly structured, well-written and clear.

### Weaknesses
- The paper mentions but does not compare against other approaches. FlexMDM and DDOT are both applicable to infilling and can be directly compared with, and their code is available.
- The paper considers only pass@1. CodeFusion [1—missing reference] shows that code diffusion models excel at higher pass@k rates, because of the model's increased diversity: compiler and execution feedback can be used to select candidates.

[1] Singh, M., Cambronero, J., Gulwani, S., Le, V., Negreanu, C. S., & Verbruggen, G. (2023, November). Codefusion: A pre-trained diffusion model for code generation. In The 2023 Conference on Empirical Methods in Natural Language Processing.

### Questions
- How does the performance evolve in function of the length of the final generation? I'm supposing that longer masks, which require more [insert], will perform worse?
- If the above holds, would extending [insert] into [insert1], [insert2], ..., [insertX] make the process converge in fewer iterations? In table 2, the multi-line performance has a big jump to the oracle performance, indicating that requiring many insertions is a bottleneck for longer code snippets.

### Soundness
3

### Presentation
3

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
This paper identifies the fixed-length mask limitation of diffusion language models for code infilling and proposes DREAMON, a lightweight augmentation that enables dynamic, variable-length generation. DREAMON trains diffusion models to predict two special sentinel states, expand and delete, so the model can autonomously lengthen or shorten masked spans at inference without architectural changes. The authors demonstrate large gains on code-infilling benchmarks, showing near-oracle accuracy and much faster, more robust generation.

### Strengths
1. The paper proposes a clear mechanism (expand/delete sentinel states) that gives masked diffusion models native variable-length control without changing model architecture. 

2. Empirical results show strong improvements on code-infilling benchmarks, achieving near-oracle accuracy and robust generation compared to prior diffusion and autoregressive baselines.

### Weaknesses
1. I feel the baselines are weak — simply plugging this into a DLM will almost certainly yield improvements. You should compare against other methods applied to DLMs to see whether DREAMON still has an advantage. Or are you the first to implement dynamic-length generation on diffusion language models?

2. Claiming in Contribution 1 and 2 that you’ve solved the fixed-length bottleneck of DLMs feels overstated, because the validation is only on code infilling. If you want to make that claim, you need to validate on a more diverse set of tasks.

3. As you note in lines 466–469, your advantages over concurrent work are low training cost and no architectural changes; however, the DAEDAL method you cite in lines 464–465 is training-free. I know that is concurrent work, but the advantage is not accurately stated.

### Questions
I didn’t have time to read all the details carefully.
1. In a single inference run, is the length that `[expand]` produces a fixed value? If so, what happens if fewer tokens are actually needed at that point, and how sensitive are results to this parameter?

2. Is it possible for `[expand]` to expand into `[delete]` (i.e., can the expansion produce delete sentinels)?

3. Have the 7B DLMs already been SFT-trained on OpenCoder, and are the training settings the same when you add DREAMON? The single-line improvement looks unusually large — how should that be explained?

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
4

### Summary
The paper proposes DreamOn, a simple extension to diffusion language models that enables variable-length generation by introducing [expand] and [delete] control tokens. It eliminates the need for fixed mask lengths, significantly improving infilling performance on code benchmarks and matching oracle-length results.

### Strengths
- Addresses a critical and practical limitation of diffusion language models—fixed-length generation—with a minimal, plug-and-play solution.
- Achieves strong empirical results, closing the performance gap with autoregressive models and matching oracle-length performance on standard infilling benchmarks. Table 1's performance is impressive.
- Requires no architectural changes and integrates seamlessly into existing DLMs with only minor modifications to the training objective.

### Weaknesses
- Clarity of Figure 2: The current illustration of the diffusion process in Figure 2 is hard to follow. It would be significantly clearer if the forward and backward processes were presented separately, with explicit depiction of how [expand] and [delete] tokens interact with the sequence during denoising.
- Inconsistent or Missing Baselines: The paper mentions evaluating LLaDA (line 305) but does not include it in any experimental results (e.g., Table 1 or 2), making it difficult to assess comparative performance. Additionally, there appears to be an inconsistency between Table 1 and Table 2: Table 1 claims all experiments use an initial mask length of 64, yet the performance reported for mask length = 64 in Table 2 does not align with the corresponding numbers in Table 1.
- Lack of Analysis on Length Control Stability: While [expand] and [delete] tokens enable dynamic length adjustment, the paper does not discuss mechanisms to prevent unstable behaviors, such as oscillation (e.g., expanding a segment only to delete it in the next step) or unbounded expansion.

### Questions
- Could the authors clarify the discrepancy between Table 1 and Table 2 regarding the mask length = 64 setting?
- Is there any safeguard (e.g., termination condition, token-level constraints, or training regularization) to avoid infinite loops or erratic length changes caused by conflicting [expand]/[delete] predictions?
- The method is evaluated only on infilling tasks. Can DreamOn be extended to general unconditional or prefix-guided generation tasks, where target length is also unknown a priori?
- Would it be possible to report the actual FLOPs or computational cost in Table 1 to better assess the efficiency trade-offs of dynamic length adaptation?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Diffusion language models have been a promising direction, especially in code generation because of flexible code in-filling capabilities. However, traditional diffusion language models use a fixed size canvas and diffuse language tokens within those. This paper introduces a novel way to remove the constraints of a fixed size canvas with two special control tokens, [expand] and [delete], allowing the model to autonomously adjust the output length during generation without architectural changes. Experiments show this method removes the fixed-length bottleneck, allowing DLMs to achieve performance on par with state-of-the-art autoregressive models and nearly matching oracle performance where the correct length is known in advance.

### Strengths
1. This paper  addresses the fixed-length mask, a major practical limitation for diffusion language models with `[expand]` and `[delete]` token solution is elegant, effective, and requires no architectural changes.

1. The paper has very nice presentation, with illustrative examples of the control tokens, and their effects. 

1. The paper clearly validates the necessity of both the expansion and deletion mechanisms through detailed experiments and ablations.

1. The proposed method allows diffusion language models to match or exceed the performance of state-of-the-art autoregressive models in code infilling of the same compute level.

### Weaknesses
- The method is presented as a general solution for Diffusion Language Models, but it is only evaluated on Python code infilling. It is unclear if the `[expand]` and `[delete]` logic would translate well to more fluid and creative natural language tasks.

- The deletion broadcasting is an example of a "training-free" efficiency, but it seems like it would be better suited if this were learned in a data-driven manner.

### Questions
- For deletion broadcasting, were there any experiments done to make the model learn the start and ends of spans for deletions instead of the heuristic of deleting all mask tokens?

- I'm curious how your method would behave on non-structured language modeling tasks, would it be fair to say that using the expansions and deletions would be a more significant challenge when it's not code?

### Soundness
3

### Presentation
3

### Contribution
3
