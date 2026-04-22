# Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Large Language Models (LLMs) often hallucinate, generating content inconsistent with the input. Retrieval-Augmented Generation (RAG) and Reinforcement Learning with Human Feedback (RLHF) can mitigate hallucinations but require resource-intensive retrieval or large-scale fine-tuning. Decoding-based methods are lighter yet lack explicit hallucination control. To address this, we present \textbf{Token-Guard}, a token-level hallucination control method based on self-checking decoding. Token-Guard performs internal verification at each reasoning step to detect hallucinated tokens before they propagate. Candidate fragments are further evaluated in a latent space with explicit hallucination risk scoring, while iterative pruning and regeneration dynamically correct detected errors. Experiments on HALU datasets show Token-Guard substantially reduces hallucinations and improves generation accuracy, offering a scalable, lightweight solution for reliable LLM outputs. Our code is publicly available\footnote{\url{https://github.com/rhq945/Token-Guard}}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper's Token-Guard method uses self-checking decoding to fix LLM token-level hallucinations. It targets the big problems with current decoding methods: no token-level checks, unclear hallucination risk, and weak dynamic correction. So it built a three-layer control: first check single tokens, then score segment coherence, finally adjust the reasoning chain globally.
Experiments are solid enough. On HALU datasets (like FinanceBench, DROP), testing with Llama and Qwen models, its average F1 beats CoT, Tree-of-Thought—for example, Llama’s base model only gets 28.29, but Token-Guard hits 51.03. Case studies (like picking the 61-yard score in DROP_nfl) also show it’s more fact-accurate. Unlike RAG or RLHF that need lots of resources, it only tweaks the decoding step—good for scenes with limited resources.

### Strengths
1. Interesting to perform fine-grained token-level hallucination control in decoding. Uses latent space scoring to filter bad tokens early, so errors don’t spread.
2. No need for external retrieval or big human feedback datasets and just need adjusting decoding. Works on different model sizes (like 3B Llama-3.2) too.
3. The experiments are quite rigorous. Ablation tests (Table 2) prove key parts (token scoring, global iteration) matter; cross-dataset tests show it’s stable.

### Weaknesses
1. Efficiency concerns: Its three-layer processing adds extra work. For long texts or tasks that need fast responses, how fast is it (like tokens per second) compared to light baselines like CoT? Can params like $N_{max}$, $M_{max}$ change automatically to balance accuracy and speed?  
2. RLHF/RAG comparison: The paper says it’s lighter than RLHF/RAG, but no direct numbers—like how much hallucination each cuts, or how much compute they use per sample. What if we combine it with RLHF? Does that work better?  
3. Edge cases: If inputs are messy and fragmented, does segment scoring fail? For LLMs that lack key knowledge (like rare medical terms in PubMedQA), hallucinations come from missing info, not bad decoding—can Token-Guard fix that?
4. Probably this paper is not the "first" token-level hallucination detection work, check this out:

[1] Sehyun Choi, Tianqing Fang, Zhaowei Wang, Yangqiu Song, KCTS: Knowledge-Constrained Tree Search Decoding with Token-Level Hallucination Detection, EMNLP 2023

### Questions
see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Token-Guard, a token-level hallucination control framework for large language models (LLMs). Unlike retrieval-based or fine-tuning methods (e.g., RAG, RLHF), Token-Guard introduces a lightweight self-checking decoding process that dynamically detects, prunes, and corrects hallucinated tokens during generation.

### Strengths
1. The paper proposes a clear and practical framework for controlling hallucinations at the token level using self-checking decoding, segment-level verification, and global iteration. This multi-stage design offers a novel angle on decoding reliability.  
2. The experiments cover six benchmark datasets and two LLM backbones, demonstrating consistent and measurable improvements in both factual accuracy and fluency.  
3. The ablation analysis is detailed and informative, helping isolate the contribution of each module and confirming that all components are necessary.  
4. The latent-space hallucination scoring method and hierarchical structure are intuitively motivated and well integrated into the decoding process.  
5. The approach is model-agnostic and easily deployable without retraining, increasing its practical value.

### Weaknesses
1. The method increases computation time and output length, sometimes approaching or exceeding heavy decoding frameworks such as Tree-of-Thought. The claim of being lightweight should be more carefully qualified.  
2. The method relies on many fixed hyperparameters, but the paper provides no sensitivity or stability analysis to show robustness across different values.  
3. The definition of latent hallucination scoring is intuitive but not empirically verified; it is unclear whether cosine similarity between hidden states best correlates with factual accuracy.  
4. The benchmarks are mostly QA-style hallucination datasets, so it remains uncertain how well the method generalizes to open-ended or long-form generation.

### Questions
1. Add a detailed efficiency comparison using wall-clock time and memory consumption, not only token counts.  
2. Include experiments showing sensitivity to threshold and weighting parameters to demonstrate stability of the method.  
3. Expand to tasks beyond hallucination benchmarks, such as reasoning or summarization, to show broader applicability.  
4. Visualize token-level and segment-level hallucination scores to clarify how the model filters unreliable fragments.  
5. Explore alternative similarity or scoring metrics for the latent-space hallucination risk.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors study decoding frameworks of Autoregressive LLMs in order to mitigate hallucinations in generated responses. To do so, the paper proposes Token-Guard, a self-checking three-stage scoring and detection pipeline at the level of individual tokens, candidate segments of tokens and finally a global iteration and correction step, in a sequential manner. Furthermore, TokenGuard is seen to achieve consistent improvements in accuracy over standard datasets, when compared to baseline decoding strategies.

### Strengths
1) The paper tackles a pertinent research avenue in autoregressive LLMs, namely that of controlled decoding to mitigate hallucinations in generated responses. While the proposed method is a little complex, the paper is well-written and presents each individual component in a fairly clear and lucid manner.

2) TokenGuard is seen to achieve significant improvements in ExactMatch and F1 scores over several decoding methods including Chain-of-Thought and Tree-of-thought, and is demonstrated using standard LLM models such as Llama-3.1-8B-Instruct and Qwen3-8b over diverse datasets for benchmarking.

3) The paper presents a detailed ablative study over several components of TokenGuard, and helps establish the relative importance of the different stages, namely prompt initialization, token level scoring, segment-level scoring, and finally global iteration. 

4) Furthermore, while Token-Guard does impose a large computational overhead when comparing with CoT and Tree-of-Thought, the overheads are only marginal when compared to other well-used decoding methods such as Guided Decoding and Predictive Decoding, while simultaneously achieving notable improvements in response accuracy.

### Weaknesses
1) While TokenGuard is seen to be quite effective, the overall method is considerably complex, and involves a large set of hyperparameter choices. This suggests that the methods may not be very practical in several realistic settings. This also raises the question of how these hyperparameters can be set - for instance in Appendix G, some sets are shown, but could the authors clarify if the F1 scores shown are for the final test data, or a hold-out validation set? Furthermore, could the authors clarify if the same exact hyperparameters are used for all evaluation settings over all datasets (and plausibly models as well)? 

2) In Stage 2, if a segment that occurs fairly early in the generation process is detected to be modified, would the scores of every later segment necessarily change? For instance, the hidden states of segments that occur after the modified segment would need to be recomputed, since the autoregressive context available is modified as well. Could the authors kindly clarify this specific aspect, since it was not entirely clear, particularly after referencing the overall algorithm presented in Appendix C. 

3) While the paper clearly presents the run-time and output token overheads in Table-3, the memory overheads necessitated by TokenGuard are less clear. For instance, the token-level score relies upon the averaged hidden state upto that token, which with a running average still requires a doubling of the hidden-state memory. For the second stage segment score and third stage global iteration, this overhead appears to be potentially even higher since cosine similarities are computed over hidden representations corresponding to these segments. For instance, in Stage 3, the chain closest to each cluster centroid is retained, but the memory overhead that occurs prior to this choice is not very clear at all!


4) In Line 259, the paper mentions the use of an external verification score E_k in order to compute the factual consistency score. Could the authors kindly clarify if an external score is actually utilized for the Stage 3 global iteration? This is an extremely pertinent aspect, since the baseline comparisons would be very different once external verification tools can be incorporated. Additional details on the computation of the sim_ctx score which quantifies semantic alignment is also needed. 

5) Several scoring methods introduced seem to be very poorly motivated, for instance the convex combination of the cosine similarity of hidden embeddings and the final probability of a given output token such as in Eq 5. This does not seem well principled apriori, and could be motivated further. For instance, more clarity is needed on which layer is chosen, and how this potentially influences the proposed score in practice. Prior works such as INSIDE [1] use similar components like consistency of representations, but are much more principled and well-motivated!

6) For the final stage, with global iteration and corrections, could the authors clarify how many chains are used? Why not just use the original order of the segments?

7) Given the correction and verification steps, comparisons with prior works such as [2], [3], [4], [5] would be highly relevant baselines to compare and contrast with. 




[1] INSIDE: Llms’ internal states retain the power of hallucination detection, Chen et al., ICLR-2024

[2] SELF-REFINE: Iterative Refinement with Self-Feedback, Madaan et al., NeurIPS 2023

[3] Towards Mitigating Hallucination in Large Language Models via Self-Reflection, EMNLP 2023

[4] Generating Sequences by Learning to Self-Correct, Welleck et al., ICLR 2023

[5] Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection, Asai et al., ICLR 2024

### Questions
1) Kindly refer to the questions mentioned in the weaknesses section above. I would be happy to raise my score further if these could be adequately addressed.
2) Line 189-194: The difference between the hidden state “h_t^(i)” and “H_t^(i)” which denotes the representation vector of candidate token a(i) was not apparent, could the authors kindly clarify this?

Minor Typos:

Line 151: “Full details are provided in the Appendixe A.1”

Line 196: “where Ht^(i) denotes” -> “​​where H_t^(i) denotes” (the “t” should be a subscript to keep notation consistent)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Token-Guard, a decoding method to mitigate the hallucination through token level self-checking and iterative refinement. Token-Guard integrates three key components: (1) token-level hallucination scoring using latent-space similarity and probability weighting (2) segment-level scoring combining local consistency and global alignment and (3) global iteration with factual and logical verification across reasoning chains. Token-Guard prunes low-confidence tokens and segments during generation, aiming to prevent error propagation. Experiments across six benchmarks demonstrate improved factuality and consistency over four decoding baselines.

### Strengths
1. The paper defines a well-structured three-stage pipeline, incorporating token-level, segment-level, and global iteration.
2. The experiment spans six benchmarks, demonstrating remarkable performance compared with other baselines.
3. The method solves a meaningful task. It detects and mitigates the hallucination at the token-level, segment-level, and globally, preventing hallucination propagation.
4. The case study in Table 4 explicitly demonstrates token-wise correction, supporting the claim of local self-repair.

### Weaknesses
1. The paper claims that current decoding methods lack token-level hallucination checking mechanism. But layer contrasting method (e.g. DoLa, Contrastive decoding) already mitigate hallucinations at the token level and achieve strong results. Relevant baselines are not discussed or compared.
2. In the latent token environment initialization, the method mentioned it requires initializing the accepted tokens a_j, but no details are provided.
3. Some datasets (e.g., RAGTruth, PubMedQA) show near-zero EM but high F1 (Table 1), suggesting mismatched evaluation or tokenization inconsistencies, but no explanation was found.
4. The experiment was conducted on limited LLM architectures (llama3.1 and Qwen 3), and model scale(3B and 8B).
5. The paper could benefit from providing a thorough examination of failure modes, especially those failed cases after re-generation.

Typo: Ht^{(i)} in line 194 should be H_t^{(i)}

### Questions
1. The method selects tokens that are similar to the previously accepted tokens. Could this reduce generation diversity or lead to over-constrained outputs?
2. Line 323, the author set the softmax temperature to 0.3, and the sampling temperature to 0.4. Is there any specific reason to set such low temperatures?
3. Table 3 lists Time and output token comparison but did not provide a normalized comparison(e.g., token/sec). Providing such results would be more straightforward.
4. Some tables and figures are far from their textual references. Repositioning them near relevant sections would improve readability.

### Soundness
2

### Presentation
2

### Contribution
2
