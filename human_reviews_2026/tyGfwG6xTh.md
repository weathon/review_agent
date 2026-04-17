# Reasoning Models Can be Accurately Pruned Via Chain-of-Thought Reconstruction

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Reasoning language models such as DeepSeek-R1 produce long chain-of-thought traces during inference time which make them costly to deploy at scale. We show that using compression techniques such as neural network pruning produces greater performance loss than in typical language modeling tasks, and in some cases can make the model slower since they cause the model to produce more thinking tokens but with worse performance. We show that this is partly due to the fact that standard LLM pruning methods often focus on input reconstruction, whereas reasoning is a decode-dominated task.  We introduce a simple, drop-in fix: during pruning we jointly reconstruct activations from the input and the model’s on-policy chain-of-thought traces. This “Reasoning-Aware Compression” (RAC) integrates seamlessly into existing pruning workflows such as SparseGPT, and boosts their performance significantly. Anonymized code can be found at: https://github.com/RyanLucas3/Reasoning-Aware-Compression

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a pruning strategy for reasoning models. Specifically, it uses the fact that there are more decoding tokens than input tokens for reasoning models, so it collects activations for those and applies existing pruning methods.

### Strengths
- The paper is easy to read and follow.
- The solution is simple and straightforward.
- The results show that it preserves accuracy and improves efficiency.

### Weaknesses
- Isn’t the self-calibration a somewhat similar mechanism [1]? Please correct me if I’m wrong.
- Missing comparison with efficient reasoning on the token side.
    - What is the pruning cost compared to the fine-tuning cost of training the model to perform concise reasoning?
    - How is model pruning more effective compared to token reduction? Reducing decoding tokens might reduce the cost more. Also, since both can be applied, I wonder what would happen if we applied both.
- Runtime minutes appear reversed. It seems like the dense model is the most efficient, and pruning makes inference slower according to the tables.
- Evaluation sets for math are limited to MATH500, and the models already perform very well on it (90%+ for 7B+ models). It is not surprising that pruning preserves high accuracy. I would like to see results on AIME as well.

[1] Williams et al. “Self-calibration for Language Model Quantization and Pruning”, NACCL 2025

### Questions
- Are the runtime minutes reversed, or am I misunderstanding something?
- As in the weaknesses, comparing with efficient fine-tuning for short CoT would make the paper more valuable.
- Why do you think naive pruning leads to more tokens in the output?

### Soundness
3

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
3

### Summary
This paper identifies that pruning reasoning models (e.g., DeepSeek-R1) with standard methods fails, causing accuracy drops and slower inference. The authors blame a calibration mismatch: methods use prompt activations, but reasoning is decode-dominated. The fix, Reasoning-Aware Compression (RAC), adds on-policy Chain-of-Thought (CoT) activations to the calibration set. Experiments show RAC significantly boosts accuracy (e.g., 90.0% vs. 74.4% at 50% sparsity for a 7B model) and reduces the "rambling" CoTs that slow down inference.

### Strengths
1. Clear Problem & Diagnosis: Clearly identifies and diagnoses a non-obvious problem: pruning reasoning models can slow them down due to a prompt/decode activation mismatch (Fig. 1).

2. Simple & Practical Solution: The proposed solution (RAC) is simple and practical: a "drop-in" fix for the calibration data, not a complex new algorithm.

3. Strong Validation (on DeepSeek): Provides strong empirical validation on the DeepSeek-R1 family, showing consistent gains on math/code and a clear reason for success (lower decode error in Fig. 2).

### Weaknesses
1. Generalizability Concerns: I think this is the primary weakness. The claims are broad, but results are confined only to the DeepSeek-R1 (GRPO-trained) family. There is no evidence this works on standard models (Llama, Mistral) using CoT prompting or other reasoning models like Qwen3, QwQ. This severely limits the contribution's impact.

2. Incomplete Method Comparison: The comparison of pruning methods is incomplete (Table 6). To validate RAC as algorithm-agnostic, the baseline (C4) results for ALPS and Wanda are needed to show relative lift, not just absolute performance.

3. Unquantified Overhead: The paper notes RAC adds overhead but never quantifies the wall-clock time for calibration versus the baseline, making its practicality hard to assess.

### Questions
1. Will RAC generalize to standard models (Llama, Mistral) using CoT prompting or other reasoning models like Qwen3, QwQ, or is the benefit specific to GRPO-trained models like DeepSeek-R1? And I think I may improve my score if some of the experiments can be done.

2. Please quantify the calibration overhead (wall-clock time) of RAC versus the 1M C4 baseline.

3. Table 4 shows on-policy traces are best. Is matching a model's idiosyncratic activation patterns more important than using 'cleaner' traces from a larger model?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This is a method to compress models for reasoning. The key idea is that reasoning involves long generated sequences relatively to the prompts, compared to standard tasks / benchmark where the generated part is short or nonexistent for multi-choice. Hence, using the prompts themselves as "calibrating data" to prune is misguided and performance can be improved by using generated sequences as calibrating data. Experiments show that it works very well.

### Strengths
The method is simple, straightforward and comparisons to other methods from that domain demonstrate superior performance.

### Weaknesses
The proposed method is too simple and the contribution too limited. It boils down to saying "use the data of your task to calibrate pruning for your task".

Now, model compression is not a domain I know well and it is easy to have an illusion of triviality post-hoc.

About the form of the paper, some sections are just loading pages with equations which are in my opinion needless and give an impression of gratuitous filling to look impressive.

### Questions
I do not have question, but I am curious to see if other reviewers who know the field better confirm that this method is not already well known and used in practice.

### Soundness
4

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
3

### Summary
This paper introduces Reasoning-Aware Compression, a pruning method for reasoning language models that maintains high accuracy while improving efficiency. Unlike standard pruning, which calibrates using only prompt signals, RAC aligns pruning with decoding behavior by incorporating internal signals from both the prompt and the model’s own chain of thought. This alignment ensures pruning reflects how the model actually reasons during generation. Experiments show that RAC-pruned models produce shorter, cleaner reasoning chains and retain accuracy close to the dense baseline- a 7B model at 50% sparsity maintains about 90% accuracy, compared to 74.4% with standard calibration.

### Strengths
1. Aligns pruning with chain-of-thought activations, effectively preserving reasoning quality. The input column to be considered for pruning considers not just the provided input prompts tokens but aslo the LLM policy's self generated reasoning CoT tokens, which aligns the model with what it would do during inference time.

2. Achieves up to 95% dense-model accuracy at 50% sparsity with no retraining.

3. Simple, plug-and-play method compatible with existing SparseGPT or WanDA approaches.

### Weaknesses
1. How do you pre-calculate $\mathcal{D}_M$ which tells howmany times you would self-generate tokens from the model?
2. While the paper has good rigor the presentation and notations can be well-defined, especially the core part in page 4 and 5. Eg. what is $N_P$?

### Questions
1. Is there any study on how the pruning varies with the CoT quality. With long reasoning traces, many times we observe the model has a long approach then backtracks and then corrects itself etc to get the output. Did you observe any specific behaviors there?
2. How would this method generalize to other tasks?

### Soundness
4

### Presentation
3

### Contribution
3
