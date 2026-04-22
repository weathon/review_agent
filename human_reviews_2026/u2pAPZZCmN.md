# DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
As large language models (LLMs) scale up, accuracy improves, but the autoregressive (AR) nature of decoding increases latency since each token requires a serial forward pass. Speculative decoding addresses this by employing a fast drafter to propose multi-token drafts, which are then verified in parallel by the target model. However, many deployments still rely on AR drafters, where sequential passes limit wall-clock gains. We revisit the drafting stage and present **DiffuSpec**, a training-free drop-in framework that uses a pretrained diffusion language model (DLM) to produce multi-token drafts in a single forward pass, while remaining compatible with standard AR verifiers. Because DLM drafts are generated under bidirectional conditioning, parallel per-position candidates form a token lattice in which the locally highest-probability token at each position need not form a causal left-to-right path. Moreover, DLM drafting requires pre-specifying a draft length, inducing a speed-quality trade-off.  To address these challenges, we introduce two practical components: (i) a causal-consistency path search (CPS) over this lattice that extracts a left-to-right path aligned with AR verification; and (ii) an adaptive draft-length (ADL) controller that adjusts next proposal size based on recent acceptance feedback and realized generated length. Across benchmarks, DiffuSpec yields up to $3\times$ wall-clock speedup, establishing diffusion-based drafting as a robust alternative to autoregressive drafters for speculative decoding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces DiffuSpec, a training-free framework designed to accelerate large language model (LLM) inference by improving speculative decoding. Instead of a traditional autoregressive drafter, DiffuSpec employs a pretrained diffusion language model (DLM) to propose multi-token drafts in a single forward pass, significantly reducing drafting latency. To overcome challenges unique to DLMs, such as their bidirectionally generated outputs and the need for a pre-specified draft length, the authors introduce two key components. The first is a causal-consistency path search (CPS), which extracts a coherent left-to-right sequence from the DLM's token lattice to align with the verifier model. The second is an adaptive draft-length (ADL) controller that dynamically adjusts the number of proposed tokens based on recent acceptance rates, optimizing the trade-off between speed and quality. Through experiments, DiffuSpec is shown to achieve up to a 3x wall-clock speedup, outperforming other training-free baselines and demonstrating that diffusion-based drafting is a viable and effective alternative for accelerating LLM generation.

### Strengths
1. The authors identified a critical problem in diffusion-based drafters -- inconsistent causality among simultaneously generated tokens, which is addressed by a small n-gram LM in this paper.

2. The writing is smooth and easy to follow.

### Weaknesses
1. The paper claims that DiffuSpec is training-free; however, its applicability appears limited. The experiments are conducted on Qwen-2.5 32B and Dream-8B, where Dream is continually trained on Qwen-2.5 8B, thus sharing the same tokenizer and vocabulary. This raises concerns about generalizability—specifically, how can DiffuSpec be applied to other autoregressive models (ARMs)?
Most ARMs are released as families of models across multiple scales, making standard speculative decoding naturally applicable. But it's hard to find a diffusion counterpart for a specific ARM. 

2. The comparison with training-based methods is not entirely convincing. The paper cites results on Vicuna-33B rather than Qwen-2.5 32B, introducing potential inconsistencies. It is recommended that the authors either (a) personally train or re-evaluate those methods on Qwen-2.5 32B for fairness, or (b) evaluate DiffuSpec on Vicuna to ensure a consistent comparison. However, as noted in Weakness 1, the latter may not be feasible.

3. As shown in Table 2, the proposed adaptive draft length (ADL) is much less effective than causal-consistency path search (CPS). I think the design of ADL is just too trival. A possible enhancement would be to feed the diffusion drafter with a longer sequence and extract a small number of tokens (e.g., 20–30) as the proposed draft. This could effectively mitigate premature EOS generation.

4. The authors should include a direct comparison between DiffuSpec and SpecDiff [1] under identical model configurations, as both methods share conceptual similarities in leveraging diffusion-based drafting for speculative decoding.

[1]. Christopher J K, Bartoldson B R, Ben-Nun T, et al. Speculative diffusion decoding: Accelerating language generation through diffusion[J]. arXiv preprint arXiv:2408.05636, 2024.

### Questions
1. How to apply DiffuSpec to other auto-regressive models without any additional training ?

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
5

### Summary
This paper introduces DiffuSpec, a technique that leverages an off-the-shelf DLLM as the draft model for an auto-regressive model. This method successfully achieved a 3x speedup on the Qwen2.5-32B model. Furthermore, to address the two primary challenges associated with using a DLLM for drafting—specifically, the non-causal nature of the DLM's proposals and the requirement to predetermine the draft length—the authors implemented two enhancements: the causal-consistency path search (CPS) and the adaptive draft-length (ADL) mechanism.

### Strengths
1. The approach of leveraging the existing DLLM as a draft model for the auto-regressive model represents a valuable and worthwhile endeavor.
2. The proposed CPS and ADL mechanisms are demonstrated to be both sound and effective.
3. Experimental results indicate that DiffuSpec achieves a substantial speedup.

### Weaknesses
1. The author claims that DiffuSpec is a training-free method, but I would argue that DiffuSpec is fundamentally a training-based approach because it relies on pre-trained draft models from the community. Specifically, if we aim to accelerate an auto-regressive model but lack a suitable DLLM to serve as the drafter, DiffuSpec still necessitates training a new draft model. This necessity arises in scenarios such as: existing DLLMs being of an inappropriate scale, or existing DLLMs not sharing a common vocabulary with the target model.
2. The author in the text refers to [1] as "concurrent work." However, since this work was first publicly released in August 2024, I do not consider it to be truly concurrent with DiffuSpec. Given its significance, it is necessary to include it in the comparison.
3. Even though the authors stressed that EAGLE-3 did not provide a draft model checkpoint for Qwen2.5-32B, I still consider EAGLE-3 a critical baseline, and its omission from the comparison is inappropriate. I strongly recommend that the authors train an EAGLE-3 draft model themselves for the comparison, given that their training code is open-source.
4. Regarding methods such as Medusa, Hydra, and EAGLE/EAGLE2, the authors directly compared their speedup ratios achieved on Vicuna 33B with that of DiffuSpec on Qwen2.5-32B. I believe this approach is highly inappropriate. All comparisons must be conducted using the identical target model. This is because prior research has already demonstrated that the acceleration achieved by the same method varies across different target models, even when the models are of the same size.
5. Regarding the SPS baseline, I do not believe that using Qwen2.5-7B as the draft model constitutes a fair comparison. This is because Dream-7B can generate all draft tokens in a single forward pass, whereas Qwen2.5-7B, when used as a drafter, necessitates multiple forward passes. Consequently, the overhead (or computational cost) of Qwen2.5-7B as the drafter would be several times that of Dream-7B. Simply matching the draft model size therefore does not ensure an equitable comparison. To comprehensively demonstrate the efficacy of DiffuSpec, I suggest that the authors also present experimental results for the SPS baseline using smaller draft models, such as Qwen2.5-3B/1.5B/0.5B.

[1] Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to address the high latency issue associated with the autoregressive decoding of Large Language Models (LLMs). The authors propose a speculative decoding framework named DiffuSpec, whose core innovation is the use of a pretrained Diffusion Language Model (DLM) as the drafter. Since a DLM can generate multiple tokens in parallel within a single forward pass, it addresses the bottleneck of serial generation in traditional autoregressive drafters. To tackle the challenges introduced by the DLM, the paper introduces two key components: 1) a Causal-Consistency Path Search (CPS) to extract a causally valid path for autoregressive verification from the non-causal token lattice generated by the DLM; and 2) an Adaptive Draft-Length (ADL) controller to dynamically adjust the length of subsequent drafts based on verification feedback. Experimental results, conducted on a specific combination of models (Qwen2.5-32B as the verifier and Dream-7B as the drafter), show that the method achieves up to a 3x wall-clock speedup.

### Strengths
1.  Novel Problem Formulation: The paper accurately identifies the serial draft generation as a core bottleneck in existing speculative decoding methods and creatively proposes using an inherently parallel diffusion language model to address it. This is a valuable and forward-looking approach.
2.  Elegant Core Algorithm (CPS): To address the paradigm mismatch between the bidirectional dependency of DLMs and the causal dependency of autoregressive models, the proposed Causal-Consistency Path Search (CPS) is an elegant, training-free solution that effectively bridges the gap between the two model types.
3.  Clarity of Presentation: The paper is well-written and well-organized, allowing readers to easily understand its complex ideas and methodology.

### Weaknesses
1.  "General Framework" Claim is Unsubstantiated and Relies on a Specialized Model Pair: This is the most critical weakness. The paper's claim of being a general "drop-in framework" is undermined by a severe lack of experimental validation.
First, all experiments are confined to only a single model family (Qwen), without testing on other families like LLaMA. This experimental scope is too singular for a paper claiming to be a general framework (and does not yet involve cross-combinations).
Second, the paper provides no evidence of cross-family generalization. It fails to address the common scenario of pairing models with different vocabularies (e.g., a LLaMA verifier and a Dream drafter), which is a critical test for a "general" framework.
Third, the paper does not demonstrate intra-family generalization. For instance, it provides no solution for a user wishing to accelerate a Qwen-7B model, as there is no corresponding smaller, pre-trained, and aligned Dream drafter available for it.
    These limitations mean the paper presents a highly specialized case study, not a general framework.

2.  Invalid Comparison to Training-Based Methods: The paper's comparison to training-based methods (Medusa, EAGLE, etc.) is fundamentally flawed. As stated in Section 5, the authors report results for their method on Qwen2.5-32B but compare them against authors' official results on Vicuna-33B. This is an invalid comparison across different models, hardware, and experimental stacks. All claims of approaching the performance of training-based methods (including in the abstract and conclusion) are based on this unsound comparison and must be disregarded.

3.  Limited Novelty and Contribution of the ADL Controller: The paper's second main contribution, ADL, is overly simplistic in its design, essentially amounting to a trivial heuristic feedback rule. The idea of dynamically adjusting draft length through heuristic methods has been explored in prior work (e.g., Minions (Wu et al., 2024)), yet the authors do not cite or compare their approach to these existing methods. Furthermore, the paper's own ablation study (Table 2) shows that ADL's contribution to performance (+0.04x speedup) is far smaller than that of CPS (+0.29x speedup). Elevating ADL to one of the two core contributions overstates its importance.

References
Wu, Z., et al. (2024). Minions: Accelerating Large Language Model Inference with Adaptive and Collective Speculative Decoding. arXiv preprint arXiv:2402.15678.

### Questions
1.  Regarding Generalizability, Constraints, and Availability: This is the most critical set of questions.
    a) Given these constraints, how can the authors defend the claim that this is a "general drop-in framework" rather than a specialized solution for the Qwen-32B/Dream-7B pair?
    b) How does the framework handle the general case where the drafter and verifier do not share the same vocabulary (e.g., pairing LLaMA-3 with Dream-7B)?
    c) What is the proposed solution for target models (even within the Qwen family, like Qwen-7B) for which no pre-trained, aligned DLM drafter currently exists?
2.  Regarding the Invalid SOTA Comparison: The paper's claims of approaching the performance of training-based methods are currently based on an invalid comparison (Qwen-32B vs. Vicuna-33B). To provide a valid, direct comparison, have the authors considered implementing and training methods like Medusa or EAGLE on the same Qwen2.5-32B target model? A direct comparison on the identical model stack is necessary to substantiate these claims.

3.  Regarding ADL Novelty: Can the authors compare ADL with prior heuristic-based adaptive length strategies, such as in Minions (Wu et al., 2024), and clarify its specific novel contributions beyond being a simple feedback loop?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents DiffuSpec, which attempts to use diffusion transformers as the draft model in speculative decoding. While the verification stage remains the same, the drafting stage is adapted to resolve two main issues in dLLM: 1) dLLM outputs are non-causal 2) dLLM uses a fixed draft length. The author proposed CPS, which utilizes a causal proxy to search for a causal sequence in the generated token lattice, in addition, the author uses ADL to dynamically change the draft length by tracking the length signals. The author used Qwen2.5-32B as the target model and Dream-7B as the draft model, achieving an average 3.08x speedup over 6 tasks, surpassing the training-required and training-free baselines.

### Strengths
- The motivation of the paper is clear
- The paper is generally easy to follow, with the methods well explained.
- CPS and ADL are well-motivated and lightweight to implement.
- The variety of evaluation tasks is sufficient for speculative decoding benchmarks

### Weaknesses
- The paper only evaluates the results on Qwen2.5-32B + Dream-7B, which makes it not convincing enough that the observed performance gains can be generalized.
- The evaluation is not clear enough as it seems not to ensure the comparison conditions are equal for all methods, i listed some of my questions below

### Questions
- As speculative decoding's performance is dependent on the draft model's approximation of the target model's distribution, using Dream-7B seems to give an unfair advantages as evidences in the highest MAT. With the highest MAT, it seems reasonable that DiffuSpec can achieve the highest speedup. However, an ablation is required to break down the performance gains, e.g. how much gain is contributed by DiffuSpec and a good dLLM draft model separately.
- Eagle/Eagle2 only use 1 layer of decoding layer as the draft model, why is a 7B dLLM faster than a 1 layer decoding layer?
- Nowadays, many popular models no longer use another well-trained LLM as the draft model, instead, they have MTP/EAGLE model trained either during training or after training. thus, picking an existing LLM as the draft model seems not a popular choice any more since it is difficult to happen to have such a model. Thus, it seems to me that the motivation for this paper is not well aligned with the current trend.

### Soundness
2

### Presentation
2

### Contribution
2
