# POME: Post Optimization Model Edit via Matrix Orthogonalization

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
We introduce Post-Optimization Model Edit (POME), a new algorithm that enhances the performance of fine-tuned large language models using only their pretrained and fine-tuned checkpoints, without requiring extra data or further optimization. The core idea is to apply a muon-style projection to $\Delta W$, the difference between the fine-tuned and pretrained weights. This projection uses truncated singular value decomposition (SVD) to equalize the influence of dominant update directions and prune small singular values, which often represent noise. As a simple post-processing step, POME is completely decoupled from the training pipeline. It requires zero modifications and imposes no overhead, making it universally compatible with any optimizer or distributed framework. POME delivers consistent gains, boosting average performance by +2.5\% on GSM8K and +1.0\% on code generation. Its broad applicability—from 7B foundation models to 72B RLHF-instructed models—establishes it as a practical, zero-cost enhancement for any fine-tuning pipeline.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the challenge of improving a fine-tuned large language model (LLM) post-training, using only the pretrained (W_pre) and fine-tuned (W_ft) checkpoints, without additional data or further optimization.

### Strengths
- The paper demonstrates strong originality by reimagining matrix orthogonalization (typically a per-step operation in optimizers like Muon) as a one-shot, training-free post-processing edit on accumulated weight deltas.
- Good experiments but with small models
- It offers an enhancement that could integrate into any LLM fine-tuning workflow. Raises the question, if we need to apply this, maybe we are not training models correctly? Maybe we need more regularization for post-training so we dont need to also apply this step?

### Weaknesses
The paper does a poor job at formalizing and conveying its main objective. 

I believe the research question they are trying to address is (correct me if I misinterpreted it please): ``Can you take an already fine-tuned large language model and make it perform better after training is complete, using only the pretrained checkpoint (W_pre) and the fine-tuned checkpoint (W_ft), without any extra data, additional training steps, or modifications to the original training pipeline?'' This should be conveyed more effectively.

Discuss relation to other research works such as:

'LoRA: Low-Rank Adaptation of Large Language Models'
'Asymmetry in Low-Rank Adapters of Foundation Models'
'Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment'
'LoRA Fine-Tuning Without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs'

Lora is a concurrent line of research and different from this paper, but it would help contextualize the contribution.

### Questions
Can the authors come up with any generalization theoretical results similar to:
'Asymmetry in Low-Rank Adapters of Foundation Models' ?

Can this scale to larger models?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the POME (Post Optimization Model Edit) technique. Inspired by Muon, POME can serve as a post fine-tuning technique, which optimizes the delta weight updates from the fine-tuning stage. It proposes the key insight that `the benefits of orthogonalization in Muon do not fundamentally require per-step enforcement'. POME demonstrates consistent performance gains on different model sizes and different post-training stages (fine-tuning to RLHF).

### Strengths
1. This paper proposes and validates a great insight: Muon's benefits can be achieved without requiring per-step enhancement. This can resolve some distributed training issues introduced by the Muon optimizer, while also improving fine-tuning performance.
2. Thorough experiments validate POME's advantage over vanilla fine-tuning on different datasets and settings.

### Weaknesses
1. Although I recognize this paper's contribution, the main intuition and most of the method details are adopted from Muon, which limits this work's contribution.
2. No direct comparison (both theoretically and experimentally) between POME and Muon-trained models' performance, readers cannot fully understand the trade-off between training efficiency/flexibility and performance.
3. The improvements are marginal compared with Adam/NEFTune. Additionally, the performance appears to be highly sensitive to the chosen layer and the retention rank ratio, which limits this method's usability.
4. Flawed presentation: e.g., the two methods in Table 7 are both 'Dr. DRPO'.

### Questions
1. Could the authors theoretically and experimentally (efficiency, performance) compare POME with Muon?
2. Could the authors provide more ablation experiments on the applied layers and the sensitivity to hyperparameters like learning rate?

### Soundness
3

### Presentation
2

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
The paper introduces POME, a training-free and data-free procedure that improves a fine-tuned LLM after training, using only the pretrained checkpoint​ and the fine-tuned checkpoint. Let \Delta be the weight difference of the two checkpoints. POME computes a layer-wise truncated SVD of the difference weight of \Delta, keeps its action on its top-k singular space identical (optionally rescaled by) and zero-out its action on the complement space, which gives rise to the edited delta​. The final edited model is given by this edited delta added to the pertained checkpoint. In authors' words, this transfers the “orthogonalization/equalization” idea popularized by Muon-style optimizers from per-step updates to a one-shot post-hoc edit of the accumulated update. The method is data-free, adds no training-time overhead, so is easy to deploy.

### Strengths
1, The proposed method of truncated SVD is a simple yet non-trivial way of linking pre-trained ckpt and sft ckpt, and appears broadly applicable and easy to implement
2, The method is data free and requires no training. The method itself seems broadly applicable
3, Paper is easy to follow; starting with a clear motivation, a clean box of algorithm, detailed tables, substantial SFT experiments spanning across various domains.

### Weaknesses
1, The choice of k and \alpha in the main algorithm needs better guidance and especially one needs a more principled rule and a better understanding of the sensitivity to the choices of those hyperparams. Moreover, one would expect different layers and matrices to prefer different levels of truncation, which seems to be under-studied.
2, The claim of “linear layer to benefit the most from subspace shaping” seems to be backed up by math domains experiments only. Would it make sense to extend this to other domains?
3, The handling of token embedding matrices (which usually consist of a large proportion of parameters) is unclear. 
4, The method assumes a dense architecture, and doesn’t discuss MoE, where subspace dynamics for routers and each expert’s FFN could be very different.
5, Whether POME is able to scale up (to 70B or beyond) is also unclear.

### Questions
1, Can the authors provide a more principled approach or rule-of-thumb for selecting the truncation rank k and scaling factor \alpha? How sensitive is POME to these hyperparameters across different architectures, datasets, and fine-tuning regimes? Would different layers or matrix types benefit from different truncation levels?
2, The claim that linear/FFN layers benefit most from subspace shaping is primarily supported by math-domain results. Do similar trends hold for other domains such as code, commonsense reasoning, or multilingual tasks? Could the authors share ablations on at least one non-math domain?
3, How does POME handle token embedding (and output) matrices, which typically constitute a large fraction of total parameters and have different functional roles than FFNs and attention projections? Are SVD based methods effective on those matrices too, or would they require to be handled differently?  
4, The method assumes dense layers, so how would POME generalize to MoE models where expert FFNs and router networks have distinct subspace behavior? Are router parameters edited, and if so, does this impact routing stability, and would such edits be compatible with expert choices?
5, How does the method scale up to larger models? Can the authors provide wall-clock runtime, memory usage, and GPU parallelism strategies for applying POME to large models (70B+ parameters)?

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
3

### Summary
This paper introduces a training-free post-processing step that takes only the pre-trained ($W_{pre}$) and fine-tuned ($W_{ft}$) checkpoints and improves the model by orthogonalising the weight delta $ΔW = W_{ft} – W_{pre}$.  The core idea is borrowed from the Muon optimizer: equalise the contribution of each principal update direction via truncated SVD and spectrum equalisation.

### Strengths
-  Post-hoc re-weighting of deltas is not new, but casting it as a *MuON-style orthogonalisation* executed *after* training is a fresh twist.  The paper cleanly separates the geometric benefit of Muon from its per-step scalability burden.
- The method is derived from a constrained optimisation problem with a closed-form solution; the empirical protocol is careful (grid-search on rank ratio and scale α, ablation of truncation vs. equalisation, comparison with EMA/NEFTune). 
- A two-line call to `torch.svd` that reliably boosts LLM performance with *zero* training cost is clearly valuable to practitioners.

### Weaknesses
- All benchmarks are either maths word problems or short coding puzzles.  No evidence on long-context reasoning, dialogue safety, or knowledge-heavy QA where weight interference may behave differently.
-  Only FFN up-projection layers are edited because they “work best”. No principled criterion is offered; the community would benefit from a predictor of which layers benefit from orthogonalisation.
- Fixing $k = 0.5·rank(ΔW)$ is empirical; Figure 1 shows this knee but does not explain why it appears across architectures.  A data-driven way to set $k$ (e.g., based on spectral gap or validation perplexity) would strengthen practical adoption.

### Questions
- Is there a risk of catastrophic forgetting on out-of-domain prompts?  A simple evaluation on the out-of-domain benchmark datasets before/after POME would reassure readers that broad knowledge is not harmed.
- Does the gain vanish when the fine-tuning already uses a matrix-aware optimiser (e.g., Muon, Shampoo, SOAP)?  An experiment that fine-tunes with Muon and *then* applies POME would clarify uniqueness.
- How does performance change if you orthogonalise *attention* deltas or the *entire* weight matrix?  The restriction to FFN seems ad-hoc; authors could report a layer-type ablation table.
- What happens when ΔW is extremely low rank (e.g., LoRA rank 16)?  POME could over-truncate; please supply results on low-rank adapters.

### Soundness
3

### Presentation
3

### Contribution
2
