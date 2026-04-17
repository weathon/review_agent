---
job_id: 1e9104ad-1bc3-4d80-9553-c79834906e61
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: rajioNWfRs.pdf
paper: TNT: Improving Chunkwise Training for Test-Time Memorization
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a new training framework for deep recurrent memory architectures aimed at efficient long-context sequence modeling, clearly within core ICLR topics (representation learning, sequence modeling, optimization / systems aspects).

## Minimum Quality
Pass ✅.  
The paper is complete (Abstract, Introduction, related work in Appendix, method, experiments, results, conclusion), written in English, technically coherent, and backed by substantial experiments on language modeling and reasoning benchmarks. No fatal methodological or statistical flaws are evident.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts to instruct or manipulate automated reviewing systems or other hidden prompts; the content appears to be a standard research manuscript.

---

# Expected Review Outcome:

## Summary

The paper introduces TNT, a two-stage training framework for deep test-time memorization modules such as Titans and TTT. Stage 1 uses a hierarchical memory architecture with a global memory traversing the full sequence at large chunk size and multiple local memories with periodic resets, enabling massive context parallelism; it also introduces a Q–K projection mechanism to reconcile a mismatch between compression (keys → values) and retrieval (queries). Stage 2 performs a short fine-tuning with smaller local chunk sizes to adapt the model for high-resolution inference, decoupling efficient pre-training from inference-time performance. Experiments on 150M-parameter Titans-based models show up to 17× faster training to a target loss compared to the most accurate Titans configuration, while also improving perplexity and downstream accuracy relative to RNN baselines and standard Transformers of similar size.

## Strengths

1. **Clear identification of concrete bottlenecks in deep memory modules.**  
   Section 3 articulates three specific challenges: inefficient training throughput due to small chunk sizes, a domain mismatch between compression and retrieval (keys vs. queries), and strong sensitivity to pre-training chunk size. Figure 2, which shows perplexity vs. inference chunk size for a Titans model trained at \(C=64\), visually substantiates Challenge 3: the perplexity curve has a sharp minimum at 64 and degrades both when chunks are smaller and when they are larger. This kind of diagnosis is valuable because it frames TNT as targeting clear, empirically demonstrated pathologies rather than vaguely “improving efficiency”.

2. **Hierarchical memory with periodic resets is a simple but effective way to get context parallelism for non-linear RNNs.**  
   The proposed update rules in Equations (5) and (6) introduce a global sequential memory \(V\) with large chunk size \(C_G\) and sharded local memories \(W_t\) that are periodically reset every \(S_L\) tokens to a learnable \(W_{\text{init}}\). Figure 1 usefully illustrates how the global memory runs across the entire sequence while each local memory operates on windows with resets, clarifying the parallelization pattern. This addresses the long-standing difficulty of parallelizing non-linear recurrences along the sequence without relying on linear scan tricks and seems practically impactful for large-batch training.

3. **Q–K projection is a conceptually clean fix for the compression–retrieval mismatch.**  
   Equation (7) introduces the projection \(\sum_{\tau=\xi(t,C_L)}^{t} \frac{k_\tau k_\tau^\top}{\|k_\tau\|^2} q_t\) so that the local memory is queried in the same key subspace it is trained on. Appendix B/C elaborates that this projection can be maintained via a running \(d\times d\) matrix updated in chunkwise parallel fashion, preserving efficiency. In Table 3, ablating Q–K projection (“w/o Q-K projection”) raises language-model perplexity from 21.04 to 22.01 and drops common-sense accuracy from 40.6% to 36.4%, providing strong evidence that this mechanism materially contributes to performance rather than being a cosmetic tweak.

4. **Stage-2 small-chunk fine-tuning is an effective, cheap way to decouple training efficiency from inference resolution.**  
   Section 4.2 and Table 2 show that after efficient Stage-1 training at relatively large chunks, a very short Stage-2 fine-tune with smaller local chunk sizes (e.g., \(C_L' = \{2,4,8,16\}\)) brings average perplexity from 23.13 down to 23.09 and improves average accuracy from 40.6% to 40.9%, using only 5% extra pre-training compute (Table 4). This concretely addresses Challenge 3 and directly demonstrates that models can be trained with hardware-friendly chunks yet exploited with fine-grained chunks at inference.

5. **Compelling runtime and time-to-quality evidence, with appropriate baselines.**  
   - Figure 4 and Figure 5 show per-step runtime versus sequence length. TNT curves (especially with \(C_L = \{64,128\}\)) grow roughly linearly and eventually cross both JAX Titans and JAX attention curves. For instance, at 32K tokens TNT with \(C_L=\{128\}\) is faster than FlashAttention despite being implemented in plain JAX.  
   - Table 1 is particularly persuasive: to reach a target training loss of 3.20, TNT with \(C_L=\{64\}\) takes 1.12 hours, yielding a 17.37× speedup over Titans with \(C=8\) (19.48 hours) while slightly improving the final quality (by earlier comparison to performance table). Even with identical small chunk (C=8), TNT is 7.68× faster than Titans, suggesting the gain is due to the architecture and parallelism rather than only operating at larger chunks.

6. **Quality of empirical evaluation and ablation depth (for a 150M scale study).**  
   - Table 2 benchmarks TNT against Transformers, DeltaNet / GatedDeltaNet, Titans, and TTT on multiple language modeling datasets (C4, FineWeb, PG19) and four common-sense reasoning benchmarks, which is a nontrivial breadth for a systems-heavy paper. TNT Stage 1 with four local memories \([4,8,16,32]\) and Stage 2 fine-tuning yields better average perplexity than Titans and TTT, and comparable accuracy to state-of-the-art gated Transformers.  
   - Ablations in Table 3–8 systematically explore: number of local memories \(N\), presence/absence of global memory, Q–K projection, global chunk size \(C_G\), local chunk size \(C_L\), and heterogeneous vs homogeneous local hierarchies. For instance, Table 7 shows a clear trend that smaller \(C_L\) reduces perplexity (up to an optimum), supporting the claim that fine-grain local chunks help, while Table 5 shows that increasing local depth up to 9 modules continues to bring modest gains. This level of hyperparameter analysis is above average.

7. **Good clarity and accessibility of the main ideas.**  
   The paper is generally well written and the core mechanisms are easy to follow. Figure 3, which diagrams Stage 1 with the single global and multiple local memories, shows how compression and retrieval integrate into the overall block, and how global and local outputs are combined before the feed-forward. Definitions of \(\xi(\cdot,\cdot)\), \(C_G\), \(C_L\), and shard length \(S_L\) are given early; appendices provide enough detail to understand Q–K projection’s efficient implementation. While there are some notational rough spots (see weaknesses), the high-level picture is clear.

## Weaknesses

1. **Theoretical underpinning of the hierarchical + reset architecture is thin.**  
   The core architectural changes in Equations (5) and (6) are justified mostly by intuition and empirical gains. There is no analysis of what information is lost by periodic resets of \(W_t\), how global and local memories jointly approximate the original fully sequential Titans dynamics, or any bounds on approximation error as a function of shard size \(S_L\) and local chunk size \(C_L\). Given that the pitch is “massive context parallelism for non-linear deep memory modules”, at least a qualitative analysis (e.g., showing equivalence in some limiting regimes, or characterizing what classes of dependencies survive the reset) would strengthen the scientific value beyond an engineering recipe.

2. **Equation (6) and the chunkwise update logic are somewhat opaque and potentially confusing.**  
   The local update is stated as  
   \[
   W_{t} \leftarrow 
   \begin{cases}
   W_{\text{init}} &\text{if } 0 \equiv t \pmod{S_L}\\[2mm]
   W_{t-1} - \sum_{\tau=\xi(t,C_L)}^{t} \eta_\tau \nabla_W \mathcal L\left(f(W_{\xi(t,C_L)},k_\tau), v_\tau\right) &\text{otherwise.}
   \end{cases}
   \]  
   This mixes a tokenwise recurrence \(W_{t-1} \to W_t\) with a gradient sum from \(\xi(t,C_L)\) to \(t\) all evaluated at \(W_{\xi(t,C_L)}\). If literally applied per-token, it appears to repeatedly subtract cumulative gradient terms overlapping with previous steps, which is not the same as Eq. (3)’s chunkwise rule where \(W_t\) is set directly from the chunk start state by a cumulative sum. The text asserts that the hierarchical structure “follows a standard chunkwise formulation”, but the exact implementation corresponding to Eq. (6) is not clearly spelled out: is \(W_t\) computed via a scan over tokens within the chunk (as in Eq. (3)), or via a re-used prefix-sum that avoids double-counting? Some explanation of how Eq. (6) is realized in code (e.g., by parallel prefix + reindexing) is needed to ensure there is no silent mathematical inconsistency.

3. **Scalability claims are limited to one model size and one architecture family.**  
   All main results are on 150M-parameter models trained on 10B tokens with Titans-style memory. While this is not trivial, the paper repeatedly suggests TNT as a “general training paradigm applicable to any deep memory module” and hints at its importance for future large-scale RNNs. Without any experiment on a larger model (e.g., 1B) or another non-linear RNN (e.g., RNNs from recent fast-weight or implicit LM work) the generalization claim remains speculative. Even a partial result, like a smaller-scale experiment on a different deep memory architecture, would reduce the risk that TNT’s benefits are tightly coupled to design quirks of Titans.

4. **Runtime comparisons, while suggestive, are not entirely apples-to-apples and omit memory / capacity effects.**  
   - In Table 1, TNT is contrasted with Titans, vanilla Transformers, and Gated Transformers using either JAX or FlashAttention kernels. TNT in plain JAX nearly matches or even slightly surpasses FlashAttention+Gating at fixed 150M scale. However, the paper also notes that TNT lacks specialized kernels. This is fair, but the comparison remains somewhat unstable: with a custom kernel for TNT, results might look significantly better or worse, and the current wall-clock results may reflect framework-level differences rather than architectural fundamentals.  
   - Memory consumption and peak activation footprint are not analyzed. The Q–K projection maintains a \(d\times d\) matrix per shard, and multiple local memories accumulate such matrices; yet there is no discussion of the resulting activation and parameter memory overhead relative to Titans or attention baselines. Given that these models are pitched as efficient for long contexts, some reporting of GPU/TPU memory usage at 32K context length (e.g., for Figure 4 and 5) would be quite informative.

5. **Q–K projection’s capacity and numerical behavior are not fully examined.**  
   While Table 3 shows a strong ablation signal, there is little discussion of how the projection matrix \(\mathcal M_t = \sum k_\tau k_\tau^\top / \|k_\tau\|^2\) behaves in practice. For instance:
   - Does \(\mathcal M_t\) become ill-conditioned or dominated by a few directions over long shards, effectively collapsing the query space to a low-rank subspace?  
   - How does the choice of normalization (dividing by \(\|k_\tau\|^2\) vs. normalized keys) impact stability?  
   - Are there any mitigation strategies (e.g., decay, clipping, low-rank approximation) that were tried?  
   Appendix C sketches efficient computation via prefix sums, but not these numerical aspects. Without this, practitioners may struggle to know when Q–K projection might fail or require tuning.

6. **Stage 2 design choices and costs are under-analyzed.**  
   Section 4.2 and Table 4 show that fine-tuning with smaller local chunks is cheap and beneficial, but several important design choices remain underspecified or unexplored:
   - Table 2 includes only a handful of Stage-2 configurations (e.g., \([1], [2,4], [2,4,8], [2,4,8,16]\)), and there is no analysis of how many fine-tuning steps are actually needed to “adapt” from large training chunks to \(C_L'=1\). Is 5% compute overkill or close to the minimum?  
   - At inference, the ideal regime is said to be \(C_L'=1\). However, Stage 2’s best perplexity is for \([2,4,8,16]\) which does not include 1, and the configuration with only \([1]\) is weaker. This suggests that multi-resolution local memories help even at inference, but the paper does not comment on this subtlety or its computational implications for deployment.

7. **Positioning relative to closely related recent work could be stronger.**  
   The paper cites high-level test-time memorization and linear-RNN work, but some immediately relevant recent works are missing or under-discussed:
   - *Memory Caching: RNNs with Growing Memory* (Li et al., 2026) proposes a structured way of expanding RNN memory over time, conceptually related to TNT’s hierarchical memory in that both aim to manage memory capacity at multiple timescales. A comparison in Section 2.1 or 4.1, at least conceptually, would clarify differences: TNT uses resets plus a global memory, whereas caching uses growing persistent memory.  
   - *Nested Learning: The Illusion of Deep Learning Architectures* (Behrouz et al., 2025) and *It’s All Connected: A Journey Through Test-Time Memorization…* (Behrouz et al., 2026 version) both analyze the relationship between online optimization, deep architectures, and test-time memorization. Since TNT also reinterprets Titans-style updates as online optimization with multiple scales, a more explicit connection in the discussion or conclusion would situate TNT better within that framework.  
   - *Synthetic Text Generation for Training LLMs via Gradient Matching* (Nguyen et al., 2025) is less directly about architectures but focuses on efficient pre-training methods; mentioning it in the context of “enabling affordable experimentation with deep memory models” in Section 5 could broaden the narrative around efficiency solutions.

   The omissions do not invalidate the technical work, but they do give the impression that TNT is somewhat more isolated than it actually is in the emerging “test-time optimization + memory” literature.

8. **Some notation and terminology are inconsistent or slightly sloppy.**  
   Examples:  
   - In Eq. (5) the index \(t\) runs from \(kC_G\) to \((k+1)C_G\) but the outer loop also uses \(k \in \{0,\ldots,L//C_G\}\); it would help to specify clearly whether the last chunk is truncated or padded.  
   - In Section 4.1.2 the text says “The projection matrix, \(\sum_{\tau=1}^{t} \frac{k_\tau k_\tau^\top}{\|k_\tau\|^2}\), can be maintained as a running sum” but the retrieval rule in Eq. (7) uses the sum from \(\xi(t,C_L)\) to \(t\). This discrepancy matters: are we maintaining per-shard matrices \(\mathcal M_t\) that reset, or a full-prefix one and subtracting older terms? Appendix C eventually clarifies a shard-level reset, but the main text could be more self-contained.  
   These issues do not prevent understanding but slightly detract from polish.

Overall, the paper’s central contributions are technically sound and empirically supported, but the work leans heavily practical and would benefit from deeper analysis of dynamics and numerical behavior, as well as slightly stronger positioning within closely related memorization / growing-memory work.

## Potentially Missing Related Work

1. **Li, Z., Behrouz, A., Deng, Y., “Memory Caching: RNNs with Growing Memory,” 2026.**  
   - Directly related: proposes a structured mechanism for increasing memory capacity of RNNs over time, also targeting long-context efficiency and memory management.  
   - How/where to add: Section 2.1 (Deep Memory Modules) and/or Section 3 (Challenges) should briefly contrast TNT’s hierarchical global+local reset scheme with growing, persistent caches, clarifying when each is preferable and whether they can be combined.

2. **Behrouz, A., Razaviyayn, M., Zhong, P., “It’s All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization,” 2026 version.**  
   - Directly related: provides a unifying view of test-time memorization and online optimization that underlies Titans and related architectures; TNT is built exactly on such deep memory modules.  
   - How/where to add: Section 2.1 and the introductory discussion of test-time memorization could cite and briefly connect TNT to this broader analytical framework.

3. **Behrouz, A., Razaviyayn, M., Zhong, P., “Nested Learning: The Illusion of Deep Learning Architectures,” 2025.**  
   - Directly related: studies hierarchical / nested learning structures that may conceptually overlap with TNT’s multi-layered memory hierarchy.  
   - How/where to add: In Section 4.1 or Appendix E when discussing generalized TNT with many local modules, it would be helpful to compare this to nested learning, highlighting differences in how hierarchy is used (temporal vs architectural).

4. **Nguyen, D., Li, Z., Bateni, M., “Synthetic Text Generation for Training Large Language Models via Gradient Matching,” 2025.**  
   - Related: not about memory modules per se, but about improving pre-training efficiency for language models. TNT’s stated motivation is to make expressive RNNs affordable to train; including this work in Section 5.1 or the conclusion as a complementary line (data-side efficiency versus architecture-side efficiency) would broaden the context.

## Questions

1. **Clarification of Equation (6) and actual update implementation.**  
   Could you detail how Eq. (6) is implemented in practice for local memories? Is \(W_t\) computed via a parallel prefix-sum per chunk similar to Eq. (3), or is there an incremental recurrence that avoids double-counting gradients from earlier positions in the same chunk? A small pseudo-code sketch would help ensure there is no mismatch between the stated rule and the efficient implementation.

2. **Behavior and regularization of the Q–K projection matrix.**  
   Have you observed any numerical instability or degenerate behavior in \(\mathcal M_t = \sum k_\tau k_\tau^\top / \|k_\tau\|^2\) for long shard lengths \(S_L\)? Did you try any regularization such as exponential decay, spectral clipping, or low-rank approximation? Providing statistics such as condition numbers or spectral norms for a trained model would increase confidence in the robustness of this component.

3. **Generality beyond Titans and 150M scale.**  
   Do you have any preliminary experiments applying TNT to another deep memory architecture (e.g., TTT, Atlas, or a recent non-linear implicit LM), or to a larger model size? Even a brief qualitative summary or small-scale result during rebuttal would help assess whether TNT is likely to transfer beyond the particular Titans-based setup.

4. **Inference-time configuration and cost.**  
   For deployment with \(C_L' = 1\), do you maintain multiple local modules (e.g., \([2,4,8,16]\) from Table 2) or collapse to a single effective memory? What is the wall-clock and memory cost of decoding versus a standard Transformer of similar size for long-context generation? Some numbers, even approximate ones, would make the “practical for deployment” claim more concrete.

5. **Effect of shard length \(S_L\).**  
   The experiments fix \(S_L = 2048\) or 4096. How sensitive are training speed and final performance to \(S_L\)? Larger \(S_L\) increases parallel shard length but also the reset period, which might hurt long-range modeling. Any ablation or at least intuition-backed guidance here would be valuable for practitioners.

Author responses that clarify Eq. (6) and the implementation details, provide some insight into Q–K projection stability, and give initial evidence of generality beyond Titans would significantly increase my confidence and could raise my score.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The empirical methodology is solid, equations are mostly consistent, and key claims (speedup and quality improvements over Titans) are well supported. Some mathematical details of the local update and Q–K projection dynamics could be clearer and more analyzed, but there are no obvious fatal flaws.

## Presentation Rating

3: good.  
The core ideas and architecture are clearly presented, figures and tables (e.g., Figures 1–5, Tables 1–8) are well chosen, and the writing is generally lucid. Minor notational inconsistencies and somewhat underspecified implementation details prevent an “excellent” rating.

## Contribution Rating

3: good.  
The paper offers a useful combination of ideas (hierarchical global/local memory with resets, Q–K projection, and two-stage chunk-size adaptation) that materially advances the practicality of deep memory RNNs, plus strong empirical validation at a non-trivial scale. The absence of deeper theory and broader architectural demonstrations keeps the overall contribution shy of “excellent”.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The work is technically sound and provides a concrete, empirically validated recipe for making non-linear test-time memorization models much more trainable, with clear speedups and quality gains over strong RNN baselines. The contribution is primarily architectural / systems rather than theoretical, and the evaluation is confined to a single model family and scale, but within that scope the results are convincing and likely to be valuable to researchers working on RNN-based alternatives to Transformers.

## Reviewer Confidence

4: confident.  
I am familiar with the literature on efficient sequence models and test-time optimization, have checked the main equations and ablations carefully, and feel confident in the assessment, though I would welcome implementation clarifications from the authors.