# Not All Experts and Tokens Matter: Selective Token-guided Expert Pruning for MoE

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Mixture-of-Experts (MoE) architectures achieve exceptional scalability for large language models but present significant deployment challenges due to substantial expert parameter overhead.  Existing expert pruning approaches rely on token-agnostic heuristics, such as routing frequency or similar statistical metrics. These methods dilute critical signals from important tokens, conflate statistical presence with functional importance, and completely discard pruned experts' knowledge. To address these limitations, we introduce ***STEP*** (Selective Token-guided Expert Pruning), a novel compression framework driven by three key innovations: (i) **Token-aware expert evaluation** that prioritizes informative tokens for context-sensitive expert assessment; (ii) **Loss-impact scoring** that quantifies expert importance through direct loss contribution rather than statistical proxy metrics; (iii) **Expert-to-bias conversion** that preserves domain knowledge via compact adaptive vectors, transforming pruning from a "discard-and-forget" to a "compress-and-preserve" paradigm. Extensive experiments demonstrate ***STEP***'s superiority across model scales and MoE architectures. At 50\% expert sparsity of the 30B Qwen model, our pruning method achieves nearly a 50\% reduction in memory usage with minimal performance degradation. The method is accompanied by a 1.5$\times$ throughput acceleration, and the entire process of pruning and converting the model completes within just 10 minutes. This enables efficient and scalable deployment of MoE models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents STEP (Selective Token-Guided Expert Pruning), a post-training compression framework for Mixture-of-Experts (MoE) models. The method selects informative tokens via attention statistics, scores experts by their loss impact, and converts pruned experts into lightweight bias vectors to preserve knowledge. Experiments on OLMoE, Moonlight, and Qwen3 models show up to 50% expert sparsity, approximately 50% memory reduction, and about 1.5× inference speedup without retraining.

### Strengths
- The motivation of connecting token importance with expert pruning is reasonable and supported by empirical attention analysis.

- The proposed expert-to-bias conversion is intuitive and practical for preserving pruned knowledge.

- The experimental setup is comprehensive and includes multiple architectures, ablations, and efficiency analyses.

### Weaknesses
- The overall idea combines existing principles of token importance and expert pruning. While the integration is clean, it does not introduce a fundamentally new compression concept.

- The analysis of token importance relies on qualitative attention statistics. There is little formal justification that attention-based token selection consistently improves expert evaluation.

- The benchmarks are mainly reasoning and classification datasets. The lack of generation or long-context experiments leaves uncertainty about broader applicability.

- Many similar frameworks already explore token-guided pruning or knowledge-preserving expert compression.

### Questions
Please refer to the weakness.

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
4

### Summary
This paper studies expert pruning for MoE models. The authors review prior work and argue that existing methods fail to account for token importance. Since the top 20% of tokens contribute roughly 80% of attention, using these tokens as pruning indicators is more reasonable. Moreover, existing approaches remove experts outright without preserving information. The proposed STEP addresses these issues from three angles: (1) identify information-rich tokens via attention patterns, (2) guide pruning based on loss increases induced by removing experts, and (3) convert pruned experts into compact biases to retain knowledge. Experiments on three MoE architectures show STEP outperforms prior methods.

### Strengths
1. The proposed method is highly-efficient and is deployment-friendly.

2. The scope of evaluated MOE models is broad and reasonable.

3. The paper is well written and easy to follow.

### Weaknesses
I have concerns corresponding to the three main components of the method:

1. While it is reasonable that the top 20% of tokens contribute ~80% of attention, where these tokens are distributed is essentially random from sample to sample. Calibrating with only 128 samples seems unjustified.

2. Using the dot product of frequency and feature norm to assess expert importance is reasonable, but it is essentially how the MoE forward pass already works (Eq. (7)). I do not find this particularly novel: the authors decompose the MoE forward into two parts and then multiply them back together—this reads more like a “frequency & feature” framing than a substantive algorithmic contribution.

3. Converting pruned experts into compact “bias vectors” to retain knowledge lacks theoretical support. Relying on a small amount of calibration data to encode biases raises serious concerns about transfer and generalization. Modern MoE models are trained on massive pretraining data; I doubt that a 10-minute pruning procedure can preserve the information from removed experts. On the contrary, such biases may negatively affect performance in real-world scenarios.

Other Concerns

1. There are factual/detail errors. For example, in Table 1, at a 50% pruning rate, 128 should correspond to 64 experts, but the paper labels it as 96. Please carefully proofread the current version.

2. The evaluation datasets are too simple to reflect whether the model truly preserves its original capabilities in practical settings—especially for the third component of the method. I recommend adding results on datasets like GSM8K and HumanEval.

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The research focus of this paper is expert pruning in Mixture-of-Experts (MoE) models. Existing expert pruning methods rely on token-agnostic heuristic strategies, which attenuate critical signals from important tokens. This paper proposes the STEP (Selective Token-guided Expert Pruning) pruning method, incorporating the following innovations: (1) Token-aware expert evaluation; (2) Loss-impact expert scoring; (3) Expert-to-bias conversion. Experiments demonstrate that STEP exhibits superiority across different model scales and MoE architectures.

### Strengths
*Methods*
1. Token-aware expert evaluation that prioritizes important tokens for context-sensitive expert assessment.
2. Loss-impact expert scoring that quantifies expert importance through direct loss contribution rather than statistical proxy metrics.
3. Expert-to-bias conversion that preserves domain knowledge via compact adaptive vectors, transforming pruning from a “discard-and-forget” to a “compress-and-preserve” paradigm.

*Experiment*
1. The comparative experiment and ablation experiment are sufficient, and the experimental analysis is complete.

### Weaknesses
1. There is a typographical error in Table 1: for Qwen3-30A3B with a 50% pruning ratio, the value should be 64, but it was incorrectly listed as 96 by the authors.
2. I would be interested in the performance of the proposed pruning method compared to existing works without fine-tuning. If feasible, could this be provided?
3. Is it reasonable to use the masked attention map for the selection of important tokens?
4. As noted by the authors in the limitations section, pruning experiments could be conducted on large-scale MoE models. Given that only the router and bias are updated during fine-tuning, experimental validation on larger-parameter models should be relatively straightforward.

### Questions
Refer to Weaknesses

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
4

### Summary
This paper proposes STEP, a three-stage pruning framework for MoE: (1) attention-guided token selection (keep only “important” tokens per layer), (2) expert importance scoring using a dual factor (activation frequency × feature-norm), and (3) expert-to-bias conversion so pruned experts’ outputs are replaced by cached bias vectors and lightly calibrated.

### Strengths
* Clear intuition, formalizes token-selection as boosting SNR for expert evaluation; implements importance via last-token attention. 

* Simple, plug-and-play scoring: expert score = frequency * feature-norm; no router retraining required.

### Weaknesses
* The method leans on the last token’s attention to rank tokens each layer. That’s intuitive, but attention isn’t a calibrated importance measure and can be brittle across prompts or objectives. The paper shows an attention heatmap and a plot that “important-first” token guidance helps frequency-based pruning, but I’d like to see more direct evidence that the selected tokens truly matter for routing/pruning decisions (e.g., visualizations of which tokens are kept, or correlation with gradient-based saliency). 
 
* In table 3, without all the three components, is it frequency-based dropping? This baseline looks quite strong. It would help to surface this baseline in the main results table and explain why other pruning/merging methods degrade so much on the same setups.
 
* The method performs a short calibration (“bias updating”) and shows that 1 epoch suffices, but it’s unclear whether competing methods received equivalently effective adaptation budgets (or whether they benefit from similar lightweight tuning). If STEP’s success hinges on a small, targeted calibration phase, comparable post-processing should be granted to other baselines.
 
* Using the final position’s attention to score importance presumes next-token prediction is the dominant objective. For multi-token generation or tasks with long-range reasoning, other positions may matter. More justification on this choice would strengthen the story.

### Questions
* After pruning/biased experts, how do routing change?

* Without that small amount of additional re-training, how will the benchmark results look like?

### Soundness
3

### Presentation
3

### Contribution
2
