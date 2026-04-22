# PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Low-Rank Adaptation is a widely used finetuning method for large models. Its small memory footprint allows practitioners to adapt large models to specific tasks at a fraction of the cost of full finetuning.  Different modifications have been proposed to enhance its efficiency by, for example, setting the learning rate, the rank, and the initialization. Another improvement axis is adapter placement strategy: when using LoRA, practitioners usually pick \emph{module types} to adapt with LoRA, such as Query and Key modules. Few works have studied the problem of adapter placement, with nonconclusive results: original LoRA paper suggested placing adapters in attention modules, while other works suggested placing them in the MLP modules. Through an intuitive theoretical analysis, we introduce PLoP (Precise LoRA Placement), a lightweight method that allows automatic identification of module types where LoRA adapters should be placed, given a pretrained model and a finetuning task. We demonstrate that PLoP consistently outperforms, and in the worst case competes, with commonly used placement strategies through comprehensive experiments on supervised finetuning and reinforcement learning for reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Precise LoRA Placement, a lightweight method to determine where to place LoRA adapters. Specifically, this paper proposes a alignment metric based on feature norm that identifies the modules requiring fine-tuning by performing inference on a small number of samples from downstream datasets. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. The proposed metric appears to be lightweight and efficient.

2. The proposed method has demonstrated its effectiveness through extensive experiments.

3. The availability of code enhances the reproducibility of the work.

### Weaknesses
1. The experimental section only compares the proposed method with the original LoRA, without including other LoRA variants or approaches that also explore efficient LoRA adaptations.

2. The definition of the Normalized Feature Norm is confusing. The motivation for introducing a random variable here, rather than directly normalizing by the product of the norms of two vectors (or matrices), is not clearly explained.

3. The results in Figure 4 show that $v_proj$ consistently exhibits lower alignment score across different models, which is an interesting observation. Providing an explanation for this phenomenon could deepen the understanding of the proposed metric and enhance the overall depth of the paper.

4. Directly reporting the computation time and memory usage required for the proposed metric would provide a clearer and more intuitive demonstration of the method’s efficiency.

### Questions
1. According to the definition of the proposed metric, it is possible to directly compute the score for each module in every layer. Why not adopt a more fine-grained localization strategy, instead of grouping modules such as Q, K, and V together for unified localization?

2. The results presented in Figure 5 appear to contradict the intuition behind the proposed method. The paper suggests that modules with lower alignment are more suitable for fine-tuning, yet Figure 5 shows that, for task-specific models, the largest score improvements (compared to general model) occur in the modules that were initially the highest, rather than in the low-alignment ones.

3. It would be helpful to show the dynamic changes of the proposed metric during the fine-tuning process.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The article under review proposes PLOP, an inexpensive, task-aware way to place LoRA adapters by ranking module types with a simple alignment score and inserting LoRA where alignment is lowest.  The authors use a normalized feature norm (nfn) that compares a module’s feature magnitude on real inputs to a randomized-input baseline:
$$
\mathrm{NFN}(W,x)=\frac{\lVert W z_{\text{in}}(x)\rVert}{\lVert W \tilde z_{\text{in}}(x)\rVert}
\approx
\frac{\lVert W z_{\text{in}}(x)\rVert}{\lVert W\rVert_F,\lVert z_{\text{in}}(x)\rVert}
$$
$(\tilde z_{\text{in}})$ shares the norm of $(z_{\text{in}})$ but has i.i.d. gaussian coordinates. higher nfn means stronger module–data alignment; scores near 1 mean weak alignment. plop averages nfn per module type and places LoRA adapters in the lowest-scoring types. Some theoretical justifications behind the method are also provided.

### Strengths
1. The paper reframes adapter placement as a task-aware alignment problem and proposes a simple, gradient-free proxy (normalized feature norm) to rank module types. this is a clean definition that removes scale effects via a randomized baseline, making it broadly applicable across architectures and tasks. the theory–metric–procedure chain is coherent and distinct from gradient/sensitivity-based selection, which is heavier and less LoRA-friendly.

2. Theoretical pieces motivate why feature norms should grow with alignment and width, then connect that to a practical nfn estimator; even if idealized, the analysis yields testable predictions that the experiments largely reflect. the empirical study spans sft classification, sft math generation, and rl (grpo) with careful parameter-matching, showing consistent wins or parity against attention-only and mlp-only baselines, sometimes beating “all modules” at lower budget. compute claims are credible and supported by explicit cost discussions and a one-batch nfn recipe.

3. The mechanism is easy to follow: compute nfn per type, pick the lowest, insert lora. the paper repeatedly reminds the reader what the type codes mean (q, k, v, o, u, g, d), provides nfns as maps and per-type bar charts, and summarizes training configs in appendices. figures and tables tie back to claims (e.g., llama3.2 parity with mlp explained by low mlp nfn; qwen math gains with d–o–v). overall presentation is minimal and readable.

4. Adapter placement is a common pain point with real budget constraints; a method that costs roughly one forward pass and often matches or improves over prevailing heuristics is practically valuable.

### Weaknesses
1. The finetuning tasks are confined to anli and math (metamathqa to gsm8k) for sft and grpo; there are no non-math generative, instruction-following, multilingual, or long-context evaluations. while fig. 4–5 report nfn patterns for code/history/logic, there are no finetuning experiments on those domains, so generality remains untested.

2. NFN is computed from a single forward pass using hooks, but the paper does not report estimator variance, sensitivity to batch/sequence length, or across-batch stability of type rankings. without confidence intervals or stability curves, it is hard to know how often plop would pick the same types on a new sample. 

3. The authors note that layer-level placement using the nfn-map gave “inconsistent results” and defer it, yet per-layer or per-head selection is exactly where a cheap ranking could shine; not probing this systematically is a weakness.

### Questions
1. How stable are nfn rankings in practice? Please report bootstrap confidence intervals and a stability curve (kendall-τ vs number of tokens/batches) for per-type nfn, plus any sensitivity to seq length. 

2. Can you give a targeted layer-level ablation using the nfn-map? Choose the bottom-k individual layers by nfn and compare against type-level plop under the same parameter budget; report when layer-level helps or hurts, and whether an nfn-gap threshold or per-type cap stabilizes results. this would clarify the “inconsistent results” remark and indicate how to operationalize finer placement.

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
3

### Summary
The paper proposes a method called PLoP for dynamically selecting LoRA training modules through precomputed metrics. Specifically, PLoP first defines an indicator, namely NFN, to measure the alignment degree between model modules and training data, and selects the modules for inserting LoRA adapters based on this indicator. The NFN score of each module is obtained by dividing the output from the module with the original data features by the output from the module with random features.

### Strengths
1) The proposed method exhibits strong generalization capabilities. It presents a unified training framework that can be generalized to any large model training approach. By simply providing the model and dataset, computing the NFN metrics, and then freezing modules based on these metrics, the method can be applied universally.

2) The theoretical analysis is comprehensive. The design motivation of NFN is supported by the theoretical basis of "Feature Norm Growth in Linear Networks", making it theoretically reasonable.

### Weaknesses
1) The most unacceptable issue with this work is that the experimental results are not significant. In the main experiments, there is no substantial difference between PLOP and fine-tuning other modules, which may indicate that researching which module is more worthy of fine-tuning has little value.

2) As a method for saving computational resources, what I expect from PLoP is its efficiency in the efficient fine-tuning of large-parameter models. However, the experimental part only uses small models below 7B.
3) The datasets used are relatively single. Consistent with the first point, since NFN calculation depends on specific datasets and models, the generalization ability of PLoP may lack verification on multiple datasets.
4) It is recommended to optimize the typesetting of Figure 8's position.
5) Figure 2 and Figure 8 are duplicated. Should it be corrected to Figure 2 in Line 191 ?

### Questions
1) PLOP is equivalent to fine-tuning one of the modules via LoRA. Why didn't the authors indicate which module corresponds to PLOP in the experiments?

### Soundness
3

### Presentation
1

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
This paper introduces PLoP, a method for dynamically assigning LoRA adapters to different module types given pretrained models and fine-tuning tasks based on the normalized feature norm (NFN) score. The authors provide theoretical and empirical justification for the NFN score as a measure of module-data alignment. PLoP is derived from the NFN score to identify modules that benefit most from SFT. Computing MFN is done via a few forward passes (without gradients etc). The authors provide results to prove the benefit of PLoP on LoRA fine tuning for classification, text generation, and mathematical reasoning using accuracy as a metric.

### Strengths
Authors outline the importance of adapter placement, and show a computationally tractable and architecture-agnostic way of doing so. 

Theoretical justification of NFN scores is compelling. 

Experiments shed light on the variance of MFN scores between layers, modules, and architectures (with and without reasoning), indicating that there is a need for a method that takes that variance into account.

The proposed framework and contributions are novel to my knowledge. 

Parameter-efficient finetuning methods are very timely (especially for discussions of continual learning).

### Weaknesses
In Text:
- Figure 1 needs more clarification on inputs and outputs, baseline feature norm. 
- In section 3, first motivate and introduce random baseline feature norm, then introduce NFN formulation to improve clarity.
- The second and third paragraph in the Introduction can be tied together and streamlined. They also cover almost exactly the same content as related works.
- Please elaborate more on the baseline presented in section 3/figure 4 and 5
- Authors need to improve formatting in Appendix.

Empirical support for PLoP as a framework is lacking: 
- It is not fully clear to me why the specialized Qwen Math model has higher NFN scores on a Math dataset; more focus on this could strengthen empirical justification
- PLoP^{-1} serves as a weak baseline, I would expect to see current SOTA selection methods (if existing), random selection, other scoring heuristics, or oracle module placement as baselines (this could be found by running a grid search over the modules of only one layer (for simplicity) of a 1B model, and comparing to the best selection).
- The improvements offered by PLoP in SFT for text generation are not super clear in section 4 (the authors are ablating over multiple parameters in the same table, unclear which numbers to compare); it took me quite a bit of time to understand the benefit PLoP offers over the baseline; could use better highlighting or illustrations to emphasize improvements.
- Baselines that are lacking: full finetuning, base model performance, gradient-based selection methods.  
- Currently, the experiments section is hard to follow. Details on experimental setup and motivation are intertwined with the results, which is distracting.
- Please provide more details on computational complexity of PLoP and how it compares to methods that take gradients.

### Questions
I'd like to better understand the rationale behind the chosen baselines.

### Soundness
3

### Presentation
2

### Contribution
3
