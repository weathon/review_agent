# Blockwise SFT for Diffusion Language Models: Reconciling Bidirectional Attention and Autoregressive Decoding

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Discrete diffusion language models have shown strong potential for text generation, yet standard supervised fine-tuning (SFT) misaligns with their semi-autoregressive inference: training randomly masks tokens across the entire response, while inference generates fixed-size blocks sequentially. This mismatch introduces noisy prefixes and leaky suffixes, biasing gradients away from the desired blockwise likelihood. We propose Blockwise SFT, which partitions responses into fixed-size blocks, selects one active block per step for stochastic masking, freezes all preceding tokens, and fully hides future ones. Loss is computed only over the active block, directly mirroring the blockwise decoding process. Experiments on GSM8K, MATH, and MetaMathQA show consistent gains over classical SFT under equal compute or token budgets. Block size consistency studies and ablations confirm that improvements stem from faithful training–inference alignment rather than incidental masking effects. Our results highlight the importance of matching supervision granularity to the decoding procedure in diffusion-based language models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a simple yet effective strategy for supervised fine-tuning of diffusion-based large language models such as LLaDA and Dream. Specifically, the authors introduce a blockwise SFT approach where, during training, a single block is selected while masking information from future blocks and preserving information from preceding blocks without masking. The authors argue that this training strategy better aligns with the blockwise inference process inherent to these models. Experimental results demonstrate that the proposed blockwise SFT yields notable improvements over classical SFT methods.

### Strengths
1. The writing is clear and easy to follow. 
2. The ablation study is comprehensive and addresses several of my concerns.

### Weaknesses
1. I believe this method offers limited advantages compared to block diffusion models. Block diffusion models, while requiring twice the number of training tokens, enable training across all blocks and kv cache in the inference process. In contrast, blockwise SFT only trains one block at a time, which restricts its training coverage. Consequently, I find the contribution of this work to be limited. 
2. Blockwise inference is not strictly necessary for LLaDA. During LLaDA's training, EOS tokens are padded to enable variable-length generation, which can lead to premature EOS generation during inference, resulting in truncated responses. However, several strategies can mitigate this issue without requiring blockwise inference. For instance, suppressing EOS token probability in early inference stages (as demonstrated in https://arxiv.org/pdf/2509.23924) can effectively address this problem. This undermines the necessity of the paper's core motivation.

### Questions
1. What specific advantages does blockwise SFT offer compared to block diffusion models? 
2. When employing alternative sampling strategies—such as reducing EOS generation probability without blockwise inference—does blockwise SFT still demonstrate gains over classical SFT?

### Soundness
2

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
This paper proposes Blockwise SFT, a supervised fine-tuning method for discrete diffusion language models that aligns training with semi-autoregressive, blockwise decoding used at inference. Classical SFT applies random full-sequence masking and bidirectional attention,  Blockwise SFT resolves this by freezing clean prefixes, masking only one active block per step, and fully hiding future tokens, optimizing loss solely within that block. The authors provide theoretical guarantees—showing that the new objective gives a variational upper bound on the true blockwise likelihood and yields unbiased gradient estimates—and demonstrate consistent empirical gains on GSM8K, MATH, and MetaMathQA benchmarks.

### Strengths
The paper is clearly written and well-organized. The methodology and theoretical sections are presented with both formal rigor and intuitive explanations. The algorithm and the diagrams make the proposed training recipe easy to follow, which enhances readability.

### Weaknesses
- Limited novelty: The proposed Blockwise SFT can be viewed as a straightforward adaptation of existing ideas for aligning training objectives with blockwise decoding. While the paper frames the problem clearly, the methodological innovation appears incremental.

- Training inefficiency: Since each training step only supervises one active block, the number of effective supervised tokens per update is significantly lower than in standard SFT under the same FLOP budget.

- Missing baselines: The experimental comparisons lack a direct baseline using block diffusion.

### Questions
I am not fully convinced about the necessity of Blockwise SFT. If blockwise generation is already established, wouldn’t directly training with a block diffusion objective be more natural and efficient—while also enabling the use of KV caching at inference?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces **Blockwise SFT**, a training-time objective designed to align **diffusion language models (DLMs)** with their **semi-autoregressive, blockwise decoding** behavior.  
Classical SFT randomly masks tokens across full responses, leading to a mismatch: noisy prefixes, suffix leakage, and token-level supervision that misaligns with block-level generation.  
Blockwise SFT resolves this by training only on a **single active block per step** — freezing all preceding tokens (clean prefix), fully hiding future ones (no leakage), and computing loss only within the active block.  
The authors theoretically derive it as a **variational upper bound on blockwise likelihood**, prove unbiased gradient estimation, and implement it without architectural change.  
Empirical results on **GSM8K**, **MATH**, and **MetaMathQA** show consistent gains: +5.2 pts on GSM8K and +1.6 pts on MATH over the strongest baselines, validating that aligning supervision granularity with decoding yields measurable improvements.

### Strengths
- **Clear motivation & strong alignment insight:** The paper precisely diagnoses the training–inference mismatch in diffusion LMs and proposes a clean fix grounded in the decoding procedure.  
- **Theory–practice coherence:** Provides formal bias analysis, variational bound, and unbiased gradient theorem; yet remains simple to implement (mask-only change).  
- **Empirical rigor:** Demonstrates consistent improvements across reasoning datasets, with ablations (block size, prefix noise, suffix leakage) directly supporting the central hypothesis.

### Weaknesses
## Major
- **Limited architectural coverage.** All experiments use only **LLaDA-8B-Instruct** with LoRA fine-tuning. The claim of being “architecture-agnostic” is unsupported without testing on other diffusion backbones such as **BlockDiffusion**, **APD**, or **RDM**.  
- **Scope restricted to reasoning tasks.** Evaluation is limited to GSM8K, MATH, and MetaMathQA, leaving uncertainty about performance on open-ended or dialogue data.  
- **Lack of scaling analysis.** The paper does not study behavior under larger models (e.g., 30B+) or longer block sizes (e.g., 512+), limiting insight into scalability.  
- **Theoretical framing stops short of efficiency guarantees.** While unbiasedness is proven, practical convergence or compute-efficiency trade-offs are not quantified.

## Minor
- **Compute fairness.** EQUAL-FLOPS and EQUAL-TOKENS protocols are well explained but omit wall-clock runtime comparisons.  
- **Notation heaviness.** Some derivations (Eq. 7–8) obscure intuition for non-diffusion audiences.  
- **Ablation breadth.** Although prefix/suffix studies are thorough, broader tests on other modalities or instruction-following data would strengthen the empirical story.

### Questions
1. How does Blockwise SFT perform when combined with RLHF or preference-based fine-tuning ?
2. Is the performance gain mostly due to reduced prefix noise, or does strict hidden-suffix causality contribute more?  
3. Can adaptive block sizes during training (e.g., uncertainty-driven scheduling) outperform fixed ones?  
4. What is the actual training-time overhead of sampling one active block per step versus full-sequence masking?  
5. Does the variational bound (Theorem 3.2) still hold under non-uniform or variable-length block sampling?

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
5

### Summary
This paper identifies a training–inference mismatch for discrete diffusion language models (DLMs) that are decoded semi‑autoregressively in fixed‑size blocks but are trained with bidirectional, full‑sequence random masking. The authors propose Blockwise SFT, a drop‑in training objective that (i) partitions responses into blocks, (ii) samples one active block per step, (iii) keeps the prefix clean (no noise), (iv) fully hides the future, and (v) computes loss only within the active block, thereby aligning supervision with the deployed decoding regime. They analyze classical SFT’s bias under this mismatch (Theorem 3.1), derive a variational upper bound on the blockwise negative log‑likelihood (Theorem 3.2), and show an unbiased stochastic gradient estimator when sampling a block and diffusion step (Theorem 3.3). Experiments fine‑tuning on MetaMathQA and evaluating on GSM8K / MATH report consistent Pass@1 gains over classical SFT and several diffusion‑SFT variants under matched FLOPs and matched supervised‑token budgets; block‑size consistency studies find peak accuracy when training and inference block sizes match, and prefix/suffix ablations indicate that preserving a clean prefix and strictly hiding future tokens are key. The work is simple, architecture‑agnostic, and argues that matching supervision granularity to blockwise decoding materially improves diffusion LMs.

### Strengths
- The paper nails a real but often-overlooked problem: standard SFT trains on full sequences with bidirectional context, but blockwise inference sees only a clean prefix and hides the future. The proposed Blockwise SFT fixes this by supervising only the active block with a clean prefix and hidden suffix. The theoretical contributions are solid—gradient-bias analysis (Theorem 3.1), a variational upper bound matching the inference factorization (Theorem 3.2), and an unbiased single-block estimator (Theorem 3.3).

- The experiments are well-designed with proper controls: EQUAL-FLOPS and EQUAL-TOKENS setups isolate the effect of the objective itself. Results are consistently better (GSM8K: 76.0 vs. 70.8, MATH: 34.2 vs. 32.6) with smoother training. The diagnostics are thorough—the block-size consistency grid shows performance peaks when training and inference block sizes match, and ablations confirm that prefix cleanliness and strict future masking are both critical. Algorithm 1 is simple, requires no architecture changes.

- The writing is clear and efficient. Figures communicate the core idea instantly, theorems come with intuition, and experimental protocols include concrete examples that make FLOPs/token accounting transparent.

- This work matters because blockwise decoding is now standard in discrete diffusion LMs. A drop-in training fix that mirrors deployment behavior has immediate practical value and points toward a broader principle: match your supervision structure to your decoding structure. The simplicity of the method plus strong empirical results make it likely to be adopted.

### Weaknesses
1. Loss inside the active block (Eq. (5))

   The paper defines

   $$
   \tilde{\mathcal{L}}_t(\theta; \mathbf{x}, a) = -\sum_{i\in\mathcal{I}_a} \log p_\theta(x_i \mid \mathbf{z}_t, t),
   $$

   yet Algorithm 1 samples an intra-block mask ($m_i\sim\mathrm{Bernoulli}(\pi)$) and states that loss is "only on the active block." In discrete diffusion / masked-LM style training, loss is normally computed only on masked positions; otherwise, unmasked tokens create an identity path and can yield degenerate gradients. The indicator ($\mathbf{1}[m_i=1]$) that appears in Eq. (1) is missing in Eq. (5), and Eq. (6) omits the expectation over the mask distribution.

2. Conditioning on context

   Eq. (5) conditions $p_\theta(\cdot \mid \mathbf{z}_t,t)$ but omits the explicit context (clean prefix, hidden suffix), whereas Eq. (9) does condition on $\text{context}^{(a)}$.

3. In Thm. 3.1, the bound scales with the probability that at least one prefix token is corrupted, $1-(1-\pi)^{|\mathcal{I}_{\text{prefix}}|}$, and the probability that at least one suffix token leaks, $1-\pi^{|\mathcal{I}_{\text{suffix}}|}$. This is a very loose surrogate for the actual perturbation magnitude, which typically scales with the expected number of corrupted tokens, e.g., $\pi\cdot|\mathcal{I}_{\text{prefix}}|$ and $(1-\pi)\cdot|\mathcal{I}_{\text{suffix}}|$. Moreover, the theorem does not quantify the third mismatch ("diluted supervision"), even though the text highlights it as critical. Finally, gradients being "$L_{\text{pre}}$- and $L_{\text{suf}}$-Lipschitz" with respect to discrete masking patterns is a strong, nonstandard assumption that needs justification.

4. In Thm. 3.2, it is correct that a diffusion ELBO can upper-bound the conditional block NLL. However, the proof text for Thm. 3.3 asserts that the inner gradient can be "identified with that of the $t{=}0$ blockwise NLL … hence the estimator targets $\nabla_\theta \mathcal{R}_{\text{block}}$ up to a scalar." This is not implied by the ELBO inequality: minimizing a surrogate bound does not make its gradient equal (or proportional) to the true NLL gradient.

5. In Thm. 3.3, the estimator multiplies by $\sum_s \\omega_s$ when sampling $t\sim\\tilde{\\omega}\propto\\omega$, which is fine, but the theorem statement should be explicit that unbiasedness is for the bound in Eq. (6) (not for the true NLL) and that the expectation is also over $\\mathbf{m}$ when masks are stochastic.

6. §3.4 states "In practice we use sequence length $L=128$," while Appendix A.2 says the max length is 256, and the running example uses $L_c=32$, $L_r=96$ ($L=128$). It is unclear which length was actually used in reported results.

7. Claims of generality are supported only on math datasets. Blockwise alignment may behave differently on open-ended generation (coding, dialogue).

### Questions
- What about baselines like the following?
    - (i) Causal SFT: classical SFT but with a causal mask (clean prefix, hidden suffix) over the *entire* response
    - (ii) Blockwise-but-bidirectional: blockwise loss but allowing future visibility
    - (iii) Span-corruption within a causal window (T5-style) matched to block size

-  The $(B_{\text{train}})$ vs $(B_{\text{infer}})$ grid is promising; what are the results for $B=1$ (degenerating to token-wise) and very large $B$ (near fully parallel)?

- Are there any failure modes for Blockwise SFT? When does Blockwise SFT hurt (e.g., very short responses, high-entropy next blocks)?

### Soundness
2

### Presentation
3

### Contribution
2
