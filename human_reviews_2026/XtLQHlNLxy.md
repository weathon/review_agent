# Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Diffusion Large Language Models (DLLMs) have emerged as a compelling alternative to Autoregressive models, designed for fast parallel generation. However, existing DLLMs are plagued by a severe quality-speed trade-off, where faster parallel decoding leads to significant performance degradation. We attribute this to the irreversibility of standard decoding in DLLMs, which is easily polarized into the wrong decoding direction along with early error context accumulation. To resolve this, we introduce Wide-In, Narrow-Out (WINO), a training-free decoding algorithm that enables revokable decoding in DLLMs. WINO employs a parallel draft-and-verify mechanism, aggressively drafting multiple tokens while simultaneously using the model's bidirectional context to verify and re-mask suspicious ones for refinement. Verified in open-source DLLMs like LLaDA and MMaDA, WINO is shown to decisively improve the quality-speed trade-off. For instance, on the GSM8K math benchmark, it accelerates inference by 6 while improving accuracy by 2.58%; on Flickr30K captioning, it achieves a 10 speedup with higher performance. More comprehensive experiments are conducted to demonstrate the superiority and provide an in-depth understanding of WINO.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Identified problem: DLLMs have to deal with quality speed trade off and it is attributed to irreversiblity of standard DLLM decoding.

The paper introduces WINO (wide in, narrow out), a training free algorithm that enables revocable decoding.

WINO's parallel draft-and verify strategy includes:
- Drafting (Wide in): aggressively draft multiple new tokens based on a lenient confidence threshold.
- Verification (Narrow out): A verification module that re-evalutes all unmasked tokens with a bit stricter verification check, remasking the ones that fail the check.

Results show good efficiency and quality gains on GSM8k and Flickr30k.

### Strengths
- Revokable Decoding: The paper’s main idea is to make the decoding process "revokable." This means the model can undo an early mistake and fix it right away, instead of being stuck with it (which is what happens in standard DLLMs).

- No Extra Training Needed: The WINO method is a simple algorithm change, meaning it works immediately with existing DLLMs like LLaDA and MMaDA without having to retrain them.

Improved Speed and Quality: WINO solves the trade-off between speed and accuracy. It makes the models much faster while also making them more accurate. The method provides huge gains in real-world tasks: GSM8K and Flickr30K.

### Weaknesses
- Please include some comparison on latency instead of decoding steps. 
- Additionally add a comparison of the number of parameters in the baseline model vs combined number of params in the drafter and verifier.
- It is a bit unclear, but IIUC, the baselines do not use remasking based on low confidence, which is pretty standard and also mentioned in the related work section. Do your baselines use remasking?

### Questions
see weakness section

### Soundness
3

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
The paper proposes Wide-In, Narrow-Out (WINO), a decoding method for accelerating inference in DLLMs. WINO appends a shadow block to the end of the context so that, after a single forward update, the model can write new tokens in parallel at several positions within the current block and also verify whether tokens written earlier in that block are reasonable, re-masking those deemed incorrect. Two hyperparameters, a drafting threshold and a verification threshold, allow a balance between throughput and accuracy. Because WINO is training-free and plug-and-play, it can be adopted by other DLLMs. The method is validated across multiple benchmarks and baselines.

### Strengths
1. The work introduces a novel draft-verify procedure that replaces the naive write-many-at-once update used in prior DLLM decoding. Compared with earlier approaches, WINO maintains high speedups without sacrificing performance and in many benchmarks even improves accuracy. 
2. The study evaluates both text-only and vision-language settings over 14 tasks, which strengthens the credibility of the conclusions. 
3. WINO does not require retraining and is easy to deploy as a plug-and-play component in other DLLMs.

### Weaknesses
1. As described in Section 4.2, during verification each position in y_shad conditions on the corresponding position in y_curr from the previous step. It is unclear whether this one-step lag negatively affects performance. 
2. The paper lacks detailed hardware information for the experiments, especially which GPU models and how many were used to obtain the main TPS results.

### Questions
1. Why did the authors choose to append an additional auxiliary block at the end of the context to run the verification step on the freshly drafted block? Were alternative designs explored to achieve the same goal, for example adding trainable layers or parameters inside the model? 
2. The draft-verify workflow can be interpreted as explicit self-refinement during DLLM generation, where the decision to re-mask a position is the refinement step. Under this view, allowing more reflection steps during inference could trade speed for higher accuracy. However, in the threshold-tuning ablations (Fig. 4), accuracy often rises at first and then declines as the number of steps increases. Do the authors have a plausible explanation for this pattern?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies the "irreversibility" of standard decoding in dLLMs as a primary cause for their poor quality-speed trade-off. To address this, the authors propose a training-free decoding algorithm named WINO: Wide-In, Narrow-Out. WINO introduces a revokable generation process through a parallel draft-and-verify mechanism. At each step, it uses a lenient threshold to aggressively draft multiple tokens ("Wide-In") and, in the same forward pass, leverages an auxiliary "shadow block" to re-evaluate previously generated tokens. Tokens that fail a stricter verification check are re-masked for later refinement ("Narrow-Out"). The method is evaluated on the LLaDA and MMaDA models, where it is shown to significantly reduce the number of decoding steps while often improving task performance. 


While I am not against accepting the paper, I have a some concerns regarding the empirical evaluation and a lack of detail on the practical overheads. My primary reservations is on baselines. The paper's central claim of superiority is substantially weakened by the lack of comparison to other sophisticated, recently proposed dLLM acceleration techniques(but maybe they are concurrent work? depending on the definition of ICLR on this term). The experiments primarily compare against "standard" (one token at a time) and "naive parallel" decoding, which are weak baselines. Without benchmarking against other dynamic samplers, it is difficult to assess the true contribution of WINO over the state-of-the-art.

### Strengths
- **Clear and Intuitive Motivation:** The identification of "irreversibility" as a core problem is a strong conceptual contribution. The proposed draft-and-verify solution is a logical and elegant response to this problem. However, this seems to be aligning with many recent work, which might be worth including to related work, and the idea is also not super novel in AR LLMs community. Being honest with that will not hurt the novelty of the paper but establishes a great baseline. 

- **Novel Mechanism:** The "shadow block" trick for enabling verification within a single forward pass is a clever and efficient design.

- **Promising Initial Results:** The ability to simultaneously reduce decoding steps by a large margin and improve accuracy on challenging benchmarks like GSM8K and ARC is a noteworthy and promising result. (but I have another concern that GSM8K seems to be a very contaminated dataset...)

- **Training-Free and Model-Agnostic:** The method's applicability to existing, pre-trained dLLMs without any fine-tuning is a major practical advantage.

### Weaknesses
- **Missing Key Baseline Comparisons:** This is the most significant weakness. The paper compares WINO to generating 1 token/step or a fixed M tokens/step. However, the related work section itself mentions other advanced samplers like Fast-dLLM-parallel and the entropy-bounded (EB) sampler, which also perform dynamic, confidence-based parallel decoding. To convincingly demonstrate the superiority of the WINO mechanism (specifically, the benefit of *revocation*), it is crucial to compare against these state-of-the-art dynamic samplers that perform aggressive drafting but lack a verification/revocation step. Without this, the reported gains might stem from simply being a better dynamic sampler, rather than from the novel revocation mechanism.

- **Lack of Analysis on Computational and Memory Overhead:**
     -  **Latency:** The speedup is reported in terms of step reduction and TPS. However, WINO increases the sequence length by adding a shadow block and requires a non-standard attention mask. This could slow down the per-step computation, especially if it prevents the use of highly optimized kernels like FlashAttention. A direct wall-clock time ablation comparing a forward pass with and without the WINO modifications (on sequences of the same effective length) is needed to understand the true per-step cost.

    - **Memory:** The paper claims the memory cost is "marginal." While likely true for the tested block length of 128, this cost is `(batch_size * Lb * d_model)`, which is non-trivial and scales with the block length `Lb`. A more rigorous analysis of how this memory overhead scales and its implications for longer-context decoding would strengthen the paper.

### Questions
- My most critical point is the comparison to other dynamic samplers. Can you provide results comparing WINO against Fast-dLLM-parallel and/or an entropy-bounded sampler on key benchmarks like GSM8K and Flickr30k? This would be essential to isolate the benefit of your novel verification/revocation mechanism.

- Could you provide a more detailed breakdown of the practical overheads? Specifically, what is the wall-clock time for a single forward pass with WINO compared to a standard forward pass on a sequence of equivalent length (i.e., `L_orig + Lb`)? This would clarify the true per-step latency cost of the custom attention mechanism.

- Additionally,  a formal pseudocode algorithm for the WINO decoding loop would be highly beneficial for clarity and reproducibility.

### Soundness
3

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
This paper introduces Wide-In, Narrow-Out (WINO), a novel training-free decoding algorithm for Diffusion Large Language Models (DLLMs) that enables revokable decoding. WINO's core contribution is a parallel draft-and-verify mechanism. It first aggressively generates a large set of candidate tokens, and then uses the model's bidirectional context to verify all currently unmasked tokens (including historical ones), re-masking any that are identified as low-quality. The experiments show that WINO improves the quality-speed trade-off, reducing inference steps while improving accuracy.

### Strengths
1. The proposed method is training-free, allowing it to function as a plug-and-play module for existing pre-trained DLLMs.
2. The algorithm demonstrates strong performance gains while accelerating the inference.

### Weaknesses
1. The proposed solution to prevent information leakage during verification seems insufficient. The paper states that a token in the Y_shad cannot attend to its corresponding token in Y_cur. However, it can attend to other tokens within Y_cur. Since full self-attention is applied within Y_cur, these other tokens already contain contextual information about the token being verified. Therefore, information leakage still exists. It causes doubt on the mechanism's claimed effectiveness, as the model is not re-evaluating the token with truly independent context.

2. The applicability of WINO to the full class of diffusion language models is unclear. It is not obvious how this method would apply to or benefit other diffusion frameworks, such as SEDD[1], which utilize different sampling schemes (e.g., τ-leaping) that are inherently iterative and corrective.

3. The decoding algorithm appears to be deterministic. The paper does not address how WINO could be adapted for stochastic sampling to produce diverse outputs for a single prompt.

[1] Lou et al., Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
3
