# SparseSSM: Efficient Selective Structured State Space Models Can Be Pruned in One-Shot

- Decision: Reject
- Scores: 4, 6, 6, 2, 6

## Abstract
State-space language models such as Mamba match Transformer quality while permitting linear complexity inference, yet still comprise billions of parameters that hinder deployment. Existing one-shot pruning methods are tailored to attention blocks and fail to account for the time-shared and discretized state-transition matrix at the heart of the selective state-space module (SSM). In this paper, we introduce SparseSSM, the first training-free pruning framework that extends the classic optimal brain surgeon (OBS) framework to state space architectures. Our layer-wise algorithm (i) derives an approximate second-order saliency score that aggregates Hessian-trace information across time steps, (ii) incorporates a component sensitivity analysis to guide feed-forward network (FFN) pruning, which also sheds light on where redundancy resides in mamba architecture, (iii) can be easily extended to semi-structured and structured sparsity. Empirically, we prune 50% of SSM weights without fine-tuning and observe no zero-shot accuracy loss, achieving the current state-of-the-art pruning algorithm for Mamba-based LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes SparseSSM that introduces structured sparsity to reduce computation and memory overhead specifically for the SSM recursive dynamics. The method leverages Hessian-based sensitivity analysis to identify redundant parameters within the state transition matrix $A_{\log}$, combined with a time-weighted aggregation scheme that accounts for the temporal dependency of SSM dynamics. This approach enables one-shot, training-free pruning, eliminating the need for fine-tuning while preserving accuracy. SparseSSM further extends to structured and N:M sparsity patterns, demonstrating flexibility across various pruning regimes. Experiments on multiple Mamba-based language models (Mamba1 and Mamba2) show competitive or improved performance compared to dense and existing pruning baselines, with significant speedups and memory savings.

### Strengths
The paper addresses the under-explored challenge of pruning the state transition matrix within SSMs. To the best of my knowledge, the derivation of SSM Hessian Matrix is novel. The proposed method achieves strong empirical results without requiring retraining, which is particularly valuable for SSM-based models. The experimental analysis is thorough, covering various sparsity settings, ablation studies, and practical speedup evaluations.

### Weaknesses
### Major Concerns
- In SSM-based models, most parameters typically reside in the input and output projection layers, which also tend to be the primary latency bottlenecks during generation. The paper focuses on pruning the state transition matrix A, but the motivation and practical impact of pruning this component are not entirely clear. 
- The proposed hierarchical aggregation protocol and sensitivity-aware pruning appear to be primarily heuristic, and their theoretical justification is limited. It would strengthen the work to provide more formal grounding or analytical discussion on why these mechanisms improve pruning stability or accuracy preservation.
- The reported latency improvement for the parallel scan appears limited, only about 1.05× speedup on the Mamba-2.8B model.
- The paper does not report model sizes (in terms of parameters or memory size) in any of the experimental tables. Including these details would significantly enhance the clarity of the results and allow readers to better quantify the trade-offs between accuracy, sparsity, and model compactness.

### Minor Concerns
- While the paper includes several pruning baselines, the magnitude pruning (MP) baseline from Han et al. (2015) [1] is somewhat outdated.

[1] Learning both weights and connections for efficient neural network

### Questions
- Would the authors provide more insight into why pruning the state transition matrix A is particularly important or beneficial compared to pruning the input and output projection layers, which typically dominate both the parameter count and runtime cost in SSM-based architectures?
- It would be helpful if the authors could report the parameter and model size reduction achieved after applying the proposed pruning method to the SSM module. This information would clarify the practical compression benefits.
- Would the authors profile and compare the end-to-end latency? The paper would be strengthened by including end-to-end latency profiling and corresponding speedup measurements after pruning the SSM module, ideally across both CPU and GPU platforms.
- Would the authors provide the results on Mamba2-8B? Including results on larger-scale models, such as Mamba-2-8B, would provide a more convincing evaluation on large models.

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
4

### Summary
This paper introduces SparseSSM, a training-free pruning framework that extends the Optimal Brain Surgeon (OBS) methodology to Mamba-based state-space models (SSMs). The authors derive an approximate Hessian for the time-shared, discretized state-transition matrix 𝐴 log , propose a power-law temporal aggregation to merge per-step importance scores, and add a component-wise sensitivity analysis to guide FFN pruning. Experiments on multiple Mamba variants (130 M – 2.8 B params) show ≈ 50 % sparsity with minimal degradation on language-modeling and zero-shot tasks, with limited but measurable speedups.

### Strengths
1. Novel technical adaptation – Thought OBS is known; however, this work extends second-order (OBS) pruning to time-shared, discretized SSM parameters, a non-trivial advance beyond Transformer-focused methods.

2. Mathematical rigor – Theorem 1 and Appendix A give a clear Hessian-trace derivation consistent with the Gauss–Newton assumption.

3. Comprehensive experiments – Multiple model sizes, datasets, and sparsity regimes, including unstructured/structured variants.

4. Practical value – Training-free, minimal compute overhead, and modest speed/memory savings useful for deployment.

5. Clear exposition – Strong alignment between problem motivation and architectural properties of Mamba.

### Weaknesses
1. Temporal aggregation lacks theoretical grounding – The power-law choice is ad hoc; no comparison with exponential or learned weighting.

2. Scalability and robustness – Performance drops at larger model scales; no analysis of failure modes or eigenvalue stability.

3. Limited comparative context – No direct cost-benefit discussion versus iterative training-based pruning, achieving higher compression.

4. Modest hardware validation – Reported 1.05–1.09 × parallel speedup lacks wall-clock analysis on target devices.

5. Missing benchmarks – No Long Range Arena or robustness tests; would strengthen claims for long-sequence modeling.
.

### Questions
1. What theoretical or empirical reasoning supports the power-law aggregation form? Have exponential or learned schemes been tested?

2. Why does performance degrade for large (2.8 B +) models—fundamental to one-shot OBS or fixable with design changes? 

3.  To the best of my understanding, "Efficient Unstructured Pruning of Mamba SSMs,(arXiv 2505.08299) showed better results on WIKItext for the 50% case? Would it be possible to at least pick wikitext and one more dataset from that to show how your work stands with this paper? This is an accepted paper from EMNLP 25. 

4. Have you examined eigenvalue distributions pre-/post-pruning to confirm stability of the selective SSM dynamics?

.5. Do you plan to include Long Range Arena or other long-context benchmarks to highlight SSM advantages? It is not mendatory but I would appreciate.  

6. What value of the exponent p is used, and how sensitive are the results to its tuning?

If my questions/ concerns/limitations are properly addressed, I am happy to increase my score. I appreciate the authors for their wonderful work!

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
This paper presents \modelname, a training-free, one-shot pruning technique for Mamba-based state-space language models. The method adapts the Optimal Brain Surgeon framework to accommodate the time-shared and discretized nature of the state-transition matrix in selective state-space modules, and incorporates a sensitivity-based approach for pruning feed-forward network components. The authors report that their algorithm can prune up to 50% of SSM weights without fine-tuning and with no observed drop in zero-shot accuracy, outperforming several baseline pruning strategies. The work offers a modest contribution to efficient compression of state-space models as an extension to an existing method.

### Strengths
Originality: The paper introduces an extension to the OBS pruning framework tailored to Mamba-based state-space models, addressing architectural challenges—like time-shared and discretized parameters—not considered in prior work. 

Quality: The approach has rigorous theoretical derivations and an empirical evaluation spanning different model sizes, pruning granularities, and both language modeling and reasoning benchmarks. The inclusion of ablation studies and sensitivity analyses validates the method’s effectiveness and robustness.

Clarity: The presentation is clear and organized, offering step-by-step explanations and visual aids that facilitate understanding of both the technical challenges and solutions. Key concepts and distinctions from prior work are well-articulated.

Significance: Efficient pruning of SSMs is practically important as these models see wider use. Achieving high sparsity without accuracy loss makes deployment more feasible and compression methods are important in this emerging area.

### Weaknesses
1. Limited Baseline Comparison: The experimental evaluation primarily contrasts the proposed method with magnitude pruning, SparseGPT, and Mamba-Shedder. However, recent pruning approaches such as Wanda are not included. Further evaluation would clarify whether the paper’s architectural adaptations are essential and show real empirical advantages.

2. Experimental Scope and Generalization: Although multiple scales of Mamba models are evaluated, the experiments are limited to Mamba architecture alone. The claim that OBS-style pruning generalizes to state-space models would be much stronger if demonstrated on alternative SSMs (e.g., S4 or S5) and hybrid models (e.g., JAMBA, Zamba). Additionally, assessing the robustness of the pruning method under longer context lengths or distributional shifts—key strengths of SSMs—would provide more convincing evidence of practical utility.

3. Evaluation Metrics and Practical Impact: The focus is primarily on perplexity and zero-shot accuracy. The benefits of pruning for deployment (such as latency, memory savings, or throughput improvements) are asserted but not quantified. Structured pruning is said to improve efficiency, but actual hardware speedup, resource consumption, or inference time should be reported across diverse sparsity settings, ideally with comparisons to pruned transformer models.

4. Theoretical Clarification: The adaptation of Hessian-based saliency scores to time-shared and discretized SSM parameters is reasonable, but the discussion could more thoroughly explain why generic second-order pruning (such as SparseGPT) fails or underperforms in this context, both mathematically and experimentally. Demonstrating fundamental limitations or failure cases of alternative methods would strengthen the motivation for the approach.

### Questions
Please address the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a method to perform training-free pruning of the SSM component of Mamba-1 and Mamba-2 LM models. The pruning framework extends the OBS idea to the SSM portion of a Mamba model, along with some well motivated ideas on how to prune other parts of the network based on sensitivity (this reminded me of how a work like LLM.int8() handles outlier weights during quantization).

While the paper has certain promising ideas, it also contains several flaws in soundness and practicality in the current form that make me lean towards rejection.

### Strengths
- The idea is clear and is executed well. The authors also point out the limitations/challenges of applying it more broadly to the non-SSM portions of the architecture.

- The method is training-free, making it tractable to apply to pretrained models in an inexpensive way.

- The chosen baselines and experiments make sense. The results are generally good when compared to the chosen baselines on the tested datasets. The paper is also technically clear and easy to read and understand.

- The work clearly shows a good understanding of the Mamba architecture, and contains detailed results for runtime efficiency in both parallel scan and recurrent modes.

### Weaknesses
- This paper (the way it is written) has a fundamental flaw. The core motivation mentions that Mamba(1 and 2) models have billions of parameters which hinders deployment. However, > 95% of the parameter count lies in the in_proj and out_proj layers that are just fully connected projection layers. The paper instead focuses on the SSM layers instead, which (especially in Mamba2) have much lesser impact on the overall efficiency (in both parallel scan and recurrent inference modes). At the very least, the writing should be made clearer to reflect this so that this does not mislead the eventual reader.

- The authors seem aware of the above limitation and did try to investigate pruning the in_proj, out_proj, and other layers (in section 3.4 and in B.2.3), however the results incur a significant performance drop.

- When I compare the dense Mamba-1.4B model to the pruned Mamba-2.8B model, I see similar or worse performance (for both downstream evals and runtime efficiency) for the pruned larger model. For the method to be practically applicable, I would expect to see better performance for the pruned larger model (otherwise why would a user not just use the dense model directly)?

- The paper focuses more on Mamba-1, whereas Mamba-2 has objectively been shown to be more practically relevant (for general LM tasks and also long-context associative recall etc) in the Mamba-2 paper. While some of the results like the speedup (e.g. the numbers in B.2.4 and Table 8) make sense for Mamba-1, I wonder how much they generalize to Mamba-2 (given the different structure of the A tensor).

- In Mamba2, the A tensor has a negligible number of parameters (often a tiny number like 32 corresponding to the number of heads). I feel like this paper overindexes on the token mixer portion of the whole architecture, whereas the largest gains in efficiency (both speedup and parameter reduction) are to be had in the other bulkier parts of the model

- The results in Tables 11 - 13 are extremely suspicious. Does SparseGPT really get extremely poor numbers on the first 3 tasks, and then is competitive on the rest of the tasks and ends up with a competitive average score? It would be great if you can check/verify this.

- While the authors do evaluate on some of the standard downstream LM datasets, I feel like the overall evaluation (and ablation studies) in the paper could have been more rigorous. This is especially heightened by the fact that the proposed method is training-free and requires minimal changes to be able to run inference on datasets. Specific ideas here that could have been tried include running the same process on a model like Vision Mamba (to show the applicability to another modality), showing validation loss trends on the pretraining dataset, additional datasets typically used in LM eval harness (e.g. the ones reported in the Mamba-2 paper), and perhaps even doing it on a pretrained hybrid Mamba-Transformer model. Strong results on a wider variety of tasks would significantly strengthen my evaluation of the paper.

### Questions
- Why did we not evaluate Mamba 2 at the other parameter ranges (especially at 1.4B and 2.8B) in Table 1? The other baseline methods seem more competitive on Mamba2, so it seems surprising to not evaluate a broader range of parameters here.

- We get extremely bad performance for the baselines on Mamba-1, but very good performance on Mamba-2. What is the reason behind this? Is there a potential problem in the evaluation setting?

- A useful and well calibrated experiment here could be to run inference on the validation set of a dataset like FineWeb (analogous to the perplexities reported on the first three datasets). This is a strong proxy for downstream LM performance, and any deltas here would be good to see between the baselines and the proposed method.

- Could you please elaborate about the methodology used to produce Tables 8 and 9? E.g. what data, batch size, and any other information about the inference setting?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SparseSSM, the first training-free, one-shot pruning framework designed specifically for Mamba-based state-space models.

### Strengths
This paper clearly presents its method and provides comprehensive numerical results across multiple Mamba checkpoints, demonstrating consistent performance retention under high sparsity levels. The proposed SparseSSM method is conceptually sound, extending the Optimal Brain Surgeon framework to selective state-space models in a theoretically grounded way. The layer-wise Hessian-based importance estimation and time-weighted aggregation strategy are well-motivated and novel in the context of SSMs. Furthermore, the empirical results convincingly show that SparseSSM can prune up to 50% of weights without fine-tuning or noticeable accuracy degradation, which is a strong practical advantage. The method’s ability to generalize across both Mamba-1 and Mamba-2 architectures also highlights its robustness and general applicability.

### Weaknesses
While the proposed SparseSSM method is well-motivated and empirically effective, several important aspects remain insufficiently addressed. First, since Mamba and other state-space models already benefit from efficient recurrent inference, it is unclear whether pruning further improves real-world inference speed or hardware efficiency. The paper does not provide detailed runtime, memory, or energy analyses to substantiate the claimed computational benefits. Second, evaluating performance solely on language modeling and zero-shot benchmarks may not fully capture the potential trade-offs introduced by sparsity—especially regarding numerical stability and robustness under low-precision or long-sequence settings. A more comprehensive discussion of how sparsity affects runtime performance, stability, and generalization beyond standard benchmarks would greatly strengthen the paper’s empirical rigor and practical relevance.

### Questions
This paper proposes SparseSSM for one-shot pruning of Mamba-based models. While the results demonstrate negligible performance degradation under various sparsity levels, the paper does not clearly quantify the corresponding computational savings. It would be valuable to report metrics such as inference latency, energy efficiency, or throughput speedups to substantiate the claimed efficiency benefits.

### Soundness
2

### Presentation
3

### Contribution
2
