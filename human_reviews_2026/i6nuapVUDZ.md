# Fast and Expressive Multi-Token Prediction with Probabilistic Circuits

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Multi-token prediction (MTP) is a prominent strategy to significantly speed up generation in large language models (LLMs), including byte-level LLMs, which are tokeniser-free but prohibitively slow. However, existing MTP methods often sacrifice expressiveness by assuming _independence_ between future tokens. In this work, we investigate the trade-off between expressiveness and latency in MTP within the framework of probabilistic circuits (PCs). Our framework, named MTPC, allows one to explore different ways to encode the _joint_ distributions over future tokens by selecting different circuit architectures, generalising classical models such as (hierarchical) mixture models, hidden Markov models and tensor networks. We show the efficacy of MTPC by retrofitting existing byte-level LLMs, such as EvaByte. Our experiments show that, when combined with speculative decoding, MTPC significantly speeds up generation compared to MTP with independence assumptions, while guaranteeing to retain the performance of the original verifier LLM. We also rigorously elucidate the optimal trade-off between expressiveness and latency when exploring the possible parameterisations of MTPC, such as PC architectures and partial layer sharing between verifier and draft LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MTPC, a probabilistic-circuit-based framework for multi-token prediction (MTP) in large language models. Unlike traditional MTP approaches that assume independence among future tokens, MTPC flexibly encodes joint token distributions through diverse probabilistic circuit architectures, generalizing models such as mixture models, HMMs, and tensor networks. Applied to byte-level LLMs like EvaByte, MTPC, combined with speculative decoding, substantially accelerates generation while preserving the verifier model’s performance. The study systematically explores the trade-off between expressiveness and latency, showing that appropriate architectural and parameter-sharing choices yield efficient, expressive, and consistent multi-token generation.

### Strengths
1. Theoretical novelty: The paper provides an interesting and elegant theoretical formulation of multi-token prediction using probabilistic circuits, enriching understanding of the expressiveness–latency trade-off.

2. Significant performance gains: Experimental results demonstrate strong acceleration and efficiency improvements while maintaining model quality, highlighting the practical impact of the proposed approach.

### Weaknesses
1. Lack of comparison with related methods: The paper does not clearly distinguish MTPC from tree-based speculative decoding approaches, nor does it discuss relevant prior work, which weakens its positioning in the broader MTP literature. [1]https://arxiv.org/abs/2402.12374 [2] https://arxiv.org/abs/2305.09781 [3] https://arxiv.org/abs/2401.10774

2. Limited model applicability: The experiments focus mainly on byte-level LLMs, with no evaluation or discussion on generalizing MTPC to mainstream models such as LLaMA, Qwen, or DeepSeek, leaving its scalability and universality uncertain.

### Questions
In the weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes MTPC, a family of multi-token prediction (MTP) heads built from probabilistic circuits (FF, CP, HMM, BTree) that model joint distributions over future tokens and plug into a shared-backbone self-speculative decoding setup. The framework lets one trade off expressiveness (acceptance rate) vs latency by (i) choosing the PC architecture and (ii) selecting how many layers the draft/verifier shares or separates. On EvaByte (byte-level LLM), MTPC improves throughput over AR and over fully factorized MTP while guaranteeing AR quality under speculative decoding.

### Strengths
1. The paper introduces MTPC, a multi-token prediction framework built on probabilistic circuits, which overcomes the independence assumptions of prior MTP methods. This allows MTPC to model joint token dependencies more effectively than factorized or tensor-decomposition-based approaches.

2. The paper rigorously studies the trade-offs between acceptance rate and generation latency across different PC architectures and different levels of layer sharing. This provides a clear and interpretable design space for controlling speed–quality trade-offs.

3. The framework is evaluated on EvaByte, where MTPC demonstrates substantial throughput improvements, for example, ×5.47 over autoregressive decoding and ×1.22 over MTP models with independence assumptions, while maintaining output quality. The experiment highlights practical deployment viability in real LLM inference settings.

### Weaknesses
1. Experiments focus on a single 6.5B byte-level model (EvaByte) and one SFT mixture (Tülu-3). It would strengthen claims to show transfer to a subword LLM (to decouple gains from byte vocabularies) and to other additional datasets/domains.

2. While the loss and discounting are described, ablations on optimization sensitivity (γ, window overlap, head depth/width) are limited. Providing more ablation studies would strengthen the paper.

3. The paper emphasizes latency but gives fewer numbers on memory vs n and r for different PCs (esp. BTree with higher ranks). Besides, this paper introduces a verifier that consumes an additional memory footprint. Therefore, a detailed memory footprint plot will help the audience understand the memory consumption of this paper.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MTPC, a framework for MTP in LLMs based on probabilistic circuits. Existing MTP methods assume independence between future tokens, sacrificing expressiveness and leading to implausible outputs. MTPC addresses this by parameterizing joint distributions over token windows using PC architectures that encode hierarchical mixture models. The framework encompasses fully factorized models (FF), canonical polyadic decompositions (CP), and introduces novel hidden Markov model (HMM) and binary tree (BTree) factorizations for MTP. Combined with speculative decoding, MTPC guarantees retention of the original autoregressive LLM's quality. The authors identify two key trade-offs: (1) PC architecture choice affecting expressiveness vs. latency, and (2) number of LoRA layers shared between draft and verifier models. Experiments retrofitting EvaByte (a 6.5B byte-level LLM) demonstrate 5.47× speedup over autoregressive generation and 1.22× speedup over independence-based MTP, with BTree achieving optimal throughput for n=16 tokens and 2 LoRA layers.

### Strengths
1. MTPC provides a unified probabilistic circuit framework that systematically navigates MTP design space, introducing novel HMM and BTree architectures with BTree achieving optimal throughput by parallelizing latent sampling while maintaining high acceptance rates.
2. The paper rigorously examines trade-offs across PC architecture selection (FF/CP/HMM/BTree) and partial layer sharing via LoRA (0-4 layers), revealing device-specific optimal configurations through systematic ablations across mixture components, window sizes, and GPU types.
3. MTPC uses speculative decoding to provably match autoregressive quality while achieving 5.47x speedups, outperforming provided baselines.

### Weaknesses
1. All experiments focus exclusively on EvaByte (6.5B byte-level model with v=320), without validation on subword-level LLMs where vocabularies are 300× larger or across different model families/sizes, limiting claims about scalability.
2. Key design decisions including inhomogeneous HMMs, identity matrix initialization, and why BTree outperforms CP lack theoretical justification beyond empirical validation, with no analysis of when specific architectures excel for different prompt characteristics.
3. The paper omits comparisons with recent MTP methods like Hydra and Eagle that introduce sequential dependencies, dismisses Basharin's KL loss without thorough evaluation, and lacks validation on standard speculative decoding benchmarks.

### Questions
1. Have authors evaluated MTPC on subword-level LLMs with large vocabularies (v≥100k), and how do the memory/computational costs of CP/HMM scale compared to FF in such settings?
2. Can authors provide theoretical or empirical guidelines for when to choose BTree vs. HMM vs. CP based on prompt characteristics, sequence lengths, or task requirements beyond throughput measurements?

### Soundness
3

### Presentation
4

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
This paper introduces multi-token PCs (MTPCs). The main idea is to movie beyond fully-factorized and simple mixture models in the context of multi-token prediction for speculative decoding. The authors evaluate MTPC on EvaByte, a byte-level LLM and observe that MTPC increases the throughput of EvaByte by 1.22x compared to the less expressive MTP speculative decoding.

### Strengths
- I found the paper to overall be well-written, aside from a few nitpicks that I've highlighted in my questions below.

- The paper offers a general, principled framework that encompasses several of the previous works.

- By exploiting connections to previous work, the authors manage to increase the expressiveness of the drafters while minimizing the latency for an overall improved throughput of 1.22x

### Weaknesses
- The paper deals with byte-level LLMs which in my opinion greatly limits its scope as it's hard to draw strong conclusion about its performance on sub-word LLMs that are a lot more commonly used by the community.

- The paper details the requirement to train the MTPC which by the authors' description is a very arduous process, and could therefore
limit adoptability of the proposed approach.

### Questions
- The authors mention that "MTPC guarantees that they match the quality of an AR LLM via speculative decoding, exactly for greedy decoding, or in expectation for sampling". Are the authors making the claim that the output of MTPC follows the AR LLM distribution? If so, isn't that a standard assumption in speculative decoding approaches? Is the "in expectation for sampling" a weakening of that assumption?

- The authors mention "repurposing" and/or "retrofitting" EvaByte, but my understanding is that the language modeling component is largely left unchanged?

- I find it a bit confusing how *speculative decoding* is separated from the *fully-factorized* and *canonical polyadic factorization* in section 2, since it is my understanding that the latter two are a means to realizing the former.

- I believe the parameterization of the PC with an LLM bears great resemblance to [1], which should be mentioned.

- I am a bit confused by paragraph 293-303. Is the implication that the model being used, EvaByte, is used with n=1 to recover a STP model? Are all the experimental results reported using greedy decoding with EvaByte? If so, it would've been useful to expand more upon the greedy speculative decoding paper by Stern et. al to show how one can guarantee argmax consistency with speculative decoding (which is not specific to MTPC)

- Referencing your conclusion, similar to the work of Zhang et. al regarding integrating constraints during generation, [2] offers a way to do so without training an HMM, which might integrate nicely with your framework.

References:

[1] Kareem Ahmed, Stefano Teso, Kai-Wei Chang, Guy Van den Broeck, & Antonio Vergari. Semantic Probabilistic Layers for Neuro-Symbolic Learning. NeurIPS 2022.
[2] Kareem Ahmed, Kai-Wei Chang, Guy Van den Broeck. Controllable Generation via Locally Constrained Resampling. In ICLR 2025.

### Soundness
3

### Presentation
3

### Contribution
3
