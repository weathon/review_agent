# Holdout-Loss-Based Data Selection for LLM Finetuning via In-Context Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Fine-tuning large pretrained language models is a common approach for aligning them with human preferences, but noisy or off-target examples can dilute supervision. While small, well-chosen datasets often match the performance of much larger ones, systematic and efficient ways to identify high-value training data remain underexplored. Many current methods rely on heuristics or expensive retraining. We present a principled, resource-efficient framework for data selection and reweighting. At its core is an In-Context Approximation (ICA) that estimates the holdout loss a model would incur after training on a candidate example by conditioning on a small, curated holdout set in context. ICA requires no reference model and no additional finetuning. 
We define the resulting estimate as the ICA score, and  derive per-example weights that dynamically reweight gradient updates as model parameters evolve. Across SFT, DPO, and SimPO, and over diverse backbones and datasets, ICA-based reweighting consistently improves model alignment with minimal overhead. We analyze sensitivity to score update frequency and the number of in-context holdout examples. We also discuss 
limitations in rapidly drifting on-policy settings, highlighting directions for future work.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a framework to select and re-weight training examples during LLM fine-tuning based on In Context Approximation (ICA) of holdout loss. The ICA score measures (by proxy) a training example's contribution to minimizing the holdout loss for one step. The authors improve the algorithm by sampling the top k most similar holdout examples (to a training example) and updating scores at intervals. Their results show improvement in win rate (alignment performance) across SFT, DPO and SimPO fine-tuning settings.

### Strengths
1. This algorithm uses in-context learning as a data re-weighting tool to improve alignment-based finetuning - which I find very interesting and seems practically efficient (only ~1.5% overhead reported). 
2. Experiments with SFT, DPO and SimPO suggest generalizability across different paradigms. Their LLM-as-a-judge metric shows a clear empirical signal to support the utility of the in-context approximation. Particularly liked the experiment in Figure 1 -  the analysis furthers support for their work and has far-reaching implications in domain adaptation. 
3. Smart optimizations backed by ablation experiments further the case for in-context approximation to select high-valued samples for LLM alignment.

### Weaknesses
1. This paper needs to be compared with other off-policy re-weighting baselines like GREATS [1]. 
2. Reweighting stability across different runs is unclear. Another ablation experiment that could support the paper would be if this method aids in faster convergence. 


[1] GREATS: Online Selection of High-Quality Data for LLM Training in Every Iteration

### Questions
1. While the paper focuses on selecting samples for alignment, it would be interesting if the method affects overall LLM performance.
2. Did you consider comparison with other domain adaptation / alignment baselines?
3. This paper will be strengthened with a re-weighting sensitivity analysis to determine over multiple runs (at least a couple runs in case of compute restrictions) that the same set of examples are highly valued (generally measured by the standard deviation of data value across runs).

### Soundness
2

### Presentation
2

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
This paper proposes a new framework for data selection during the fine-tuning of LLMs. The core problem addressed is the negative impact of noisy or off-target data often present in large fine-tuning datasets. The proposed solution aims to identify and prioritize high-value training examples based on their potential to improve the model's performance on a small, high-quality holdout set. The key innovation is the In-Context Approximation (ICA) method, which leverages the LLM's in-context learning (ICL) ability to efficiently estimate the holdout loss change associated with training on a candidate example, without requiring expensive retraining or auxiliary models. These ICA scores are then used to dynamically reweight gradient updates during fine-tuning across various paradigms like SFT, DPO, and SimPO. The authors present experimental results suggesting that this method consistently improves model alignment compared to standard training.

### Strengths
1. The core idea of using ICL to approximate the impact of a training sample on holdout loss (the ICA score) is innovative and computationally efficient. It cleverly avoids the high costs associated with traditional data valuation methods that require retraining or reference models, potentially making holdout-loss-based selection more practical.


2. The method is motivated by theoretical considerations, building upon the holdout loss approximation framework established in prior work (e.g., RHO-Loss). Linking the ICA score to a first-order update towards the holdout optimum provides a principled justification for its use.




3. The framework demonstrates versatility by showing positive results across multiple fine-tuning techniques (SFT, DPO, SimPO) , different model families and sizes (LLaMA-3, Qwen-3; 3B/8B) , and varied data selection scenarios (quality improvement, domain relevance).

### Weaknesses
1. The theoretical justification for the core In-Context Approximation (ICA) itself appears weak. While the goal (approximating holdout loss impact) is theoretically grounded, the paper does not provide a rigorous analysis or proof showing why or how accurately a single forward pass with k-shot ICL (Eq. 6) approximates the complex model state change ($\theta^{*}(\mathcal{D}_{t}\cup\mathcal{D}_{ho})$ in Eq. 4) that involves actual gradient updates. The reliance on the general observation that ICL performs "implicit fine-tuning" feels insufficient to bridge this gap, weakening the claim of a fully "theoretically grounded" method.

2. The empirical evaluation relies heavily on a potentially unreliable and limited metric, hindering convincing validation. The primary evaluation metric is the "win rate" determined by GPT-4o judging pairwise comparisons. This methodology is susceptible to the biases and limitations of the judge model, costly, hard to reproduce, and lacks granularity. The absence of standard, objective task-specific metrics (e.g., accuracy on downstream tasks, perplexity, benchmark scores like MT-Bench/AlpacaEval) makes it difficult to assess the actual impact on model capabilities beyond alignment style as perceived by one specific LLM.

3. The experimental scope feels somewhat narrow, potentially limiting the generalizability of the findings. While multiple models and methods are tested, the experiments primarily use only two main dataset pairs (Alpaca/Alpaca-cleaned for quality, Yahoo/SHP for domain). Validation on a wider range of dataset types, sizes, and noise levels would be necessary to confirm the robustness and broad applicability claimed. Furthermore, the comparison is limited to RHO-Loss and One-Shot learning; comparing against simpler heuristics (e.g., loss-based filtering, perplexity) or more recent data selection methods could provide better context.

4. This paper would be strengthened by a comparison of this method to other recent frameworks for data selection and robust optimization during fine-tuning [1, 2, 3].

[1] Provably Robust DPO: Aligning Language Models with Noisy Feedback. ICML 2024.

[2] ROPO: Robust Preference Optimization for Large Language Models. ICML 2025.

[3] LESS: Selecting influential data for targeted instruction tuning. ICML 2024.

### Questions
1. The ablation study shows that using k=3 for ICL examples performs best, with performance degrading for k=5 and k=10. This seems counter-intuitive, as ICL performance often benefits from more examples. Could the authors provide some intuition or analysis for this observation?

2. How sensitive is the method to the choice of the k holdout examples selected via kNN? Does the variance introduced by different kNN selections significantly affect the stability of the ICA scores and the final performance?

3. There seems to be an inconsistency in model naming: the text mentions using LLaMA-3-3B-Instruct and Qwen-3-4B , but Table 3 refers to "Qwen 3B". Please clarify if a 3B Qwen model was used or if this is a typo for 4B.

### Soundness
2

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
3

### Summary
The paper proposes In‑Context Approximation (ICA) to estimate, per training example, how much the holdout loss would decrease if the example were used for training.  It defines an ICA score as the difference between unconditional and conditional losses and uses these scores to reweight per-example gradients. The method targets SFT, DPO, and SimPO and is evaluated on LLaMA-3 (3B/8B) and Qwen-3 (4B/8B) across Alpaca/Alpaca‑cleaned, Yahoo Answers Topics (domain shift), UltraFeedback‑binarized, and SHP‑2. Results are reported as GPT‑4o judged win rates of  vs. standard training (no reweighting),  RHO‑Loss, and One‑Shot Learning.

### Strengths
- Simple, general mechanism: Framing per-example value via an in‑context estimate of “training on the holdout set” is elegant and easy to drop into standard training loops (Eq. 6–9; Alg. 1). 

- Broad applicability: Demonstrated on SFT, DPO, SimPO and across two model families/sizes. Tables 1–3 show “Ours vs. w/o reweighting” win rates generally >60% and often much higher (e.g., LLaMA‑8B SFT Yahoo: 85.10%), supporting cross‑paradigm usefulness. 

- Low overhead claim with ablations: The method adds ~1.5% runtime (Table 10), and ablations on k (few in‑context demos), R (score recomputations), weighting vs. filtering, and embedding choice are included (Sec. 5.3; Tables 6–9), which helps practitioners calibrate cost/performance trade‑offs. 

- Clear exposition of pipeline: Pseudocode (Alg. 1; Alg. 2) and prompts (Table 12) make the approach easy to reimplement; training/eval config is summarized (Table 5)

### Weaknesses
1. Apparent holdout leakage in multiple setups: In Yahoo Answers Topics, the paper states: “The holdout and test sets each contain 3,000 Sports examples. The training set consists of 1,000 examples from each of the remaining domains, together with the holdout examples, totaling 12,000 training examples.” (Appendix B.1, emphasis added). This means the holdout set is inside the training set, which undermines the notion of “holdout loss,” confounds the ICA scoring (which conditions on $D_{ho}$), and risks overstating gains. A similar pattern occurs for SHP‑2 and UltraFeedback‑binarized (“From the training set, a holdout set … is selected… All examples in the training set are used for training”). These choices likely cause data leakage between “scorer” (holdout) and “trainer,” conflating reweighting with simply learning from the curated subset directly. 
This affects both internal validity (does ICA help when the holdout is truly held out?) and comparability to RHO‑Loss/One‑Shot under proper splits.

2. Evaluation design relies on single‑seed training and model‑judge only: The paper explicitly trains and evaluates each model only once due to compute constraints (Sec. 5.1). Variability is assessed only by repeating GPT‑based evaluation (Table 4), not by rerunning training. This leaves uncertainty about training variance and statistical significance of reported win‑rates. Moreover, the main metric is pairwise win rate judged by GPT‑4o (temperature 0, seed None; Table 5/“Evaluation Parameters”), with no human evaluation or confidence intervals. 


3. Theoretical grounding for ICA is informal The abstract claims: “Under a local linearization, ICA is equivalent to a first‑order update toward the holdout optimum” (p. 1), but the paper does not present a derivation or bounds for this equivalence. Appendix A reproduces the RHO‑Loss Bayesian derivation but does not analyze the error of the ICA approximation or its dependence on 𝑘, context formatting, or model scale. This weakens the “theoretically grounded” claim. 

4. Comparisons and reporting: Absolute performance is not reported; only win rates vs. baselines. Without absolute metrics (e.g., accuracy, reward, BLEU, or task‑specific scores), it is hard to gauge practical effect size. Baselines exclude several strong data‑selection methods (e.g., influence‑function variants, Grad‑Match, importance resampling) beyond citations. Given the paper’s emphasis on efficiency, a compute‑matched subset of these would strengthen claims. 

5. Dynamic reweighting is emphasized, yet the default setting computes scores once at t=0 (R=1; Sec. 5.1), which effectively turns the method into a static ranking for most experiments; the ablation (increasing R) shows only modest gains (Table 7).

### Questions
1. Holdout leakage: Why are holdout examples included in the training set (Appendix B.1, Yahoo/SHP‑2/UltraFeedback)? Please re‑run the core experiments with strictly disjoint train/holdout/test splits and report results, especially for Yahoo‑Sports and UltraFeedback. 

2. Theory gap: Can you provide the promised derivation (or at least a formal sketch with assumptions) showing the local‑linearization equivalence between ICA and a first‑order update toward the holdout optimum? What conditions on model smoothness / context length / 
k are required? 

3. Variance and significance: Please run ≥3 training seeds for at least one representative setting per paradigm (SFT/DPO/SimPO) and report mean ± std and binomial confidence intervals for win rates. Also include absolute scores on standard benchmarks where available. 

4. Dynamic vs. static: Since the default is R=1, how much do results change for R>1 under a fixed compute budget (i.e., reducing training steps to keep total GPU hours constant)? 

5. Prompt sensitivity: How sensitive is ICA to demonstration formatting and ordering? Can you include a robustness table varying the in‑context template / example order?
 
## Suggestion:

- Ablate proper splits: Re‑report Tables 1–3 with strictly disjoint holdout (for scoring) and training datasets, and add a variant where you train on the holdout alone to show that ICA’s gains are not due simply to including curated data. 

- Absolute metrics: In addition to judge‑based win rates, include task‑specific metrics or reward m- odel scores to contextualize effect sizes. 

- Compute‑fair baseline suite: Add at least one influence‑based (or Grad‑Match / importance‑resampling) baseline under matched compute, even on a reduced scale, to strengthen comparative claims. 

- Report overhead vs. R: Extend Table 10 to show overhead for R=3/5/9 to back the claim of efficiency when more frequent updates are used (Table 7).  Also see https://openreview.net/pdf?id=p0KTYl2B9T for overhead scaling.

- Clarify claims: Either include the local‑linearization derivation or soften language from “equivalent” to “motivated by,” and discuss limitations (e.g., prompt length constraints and ICA’s off‑policy nature mentioned in Sec. 6)

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
This paper introduces In-Context Approximation (ICA), a resource-efficient framework for data selection and reweighting when fine-tuning Large Language Models. At its core, the method estimates the value of a training example by measuring how it would reduce the loss on a small, curated holdout set. Crucially, this estimation is performed using the model's in-context learning capabilities. By dynamically reweighting gradient updates based on these ICA scores, the method prioritizes high-value data, leading to improved model alignment with minimal computational overhead.

### Strengths
+ A major advantage of the ICA method is its ability to approximate the holdout loss (similar to RHO-Loss) without the significant overhead of training a separate reference model on the holdout set. This makes it a more practical and scalable solution.

+ The method demonstrates consistently strong results across various models, datasets, and alignment tasks (SFT, DPO, SimPO). Its performance is shown to be comparable to the more resource-intensive RHO-Loss baseline, validating its effectiveness.

### Weaknesses
- The framework's success is dependent on having a high-quality, demonstration holdout set to act as a proxy for the desired test distribution. The performance could degrade if the holdout set is noisy, small, or unrepresentative of the target domain.

- The experiments focus primarily on instruction-following and preference alignment tasks. The paper does not evaluate the method on more complex reasoning domains like mathematics or science, where data quality and logical consistency are critical.

- The method introduces new hyperparameters that may require careful tuning. The paper shows that k=3 (the number of in-context examples) works well for their experiments. However, the optimal k could be highly dependent on the model's context window size, the complexity of the task, and the nature of the domain. This adds an extra layer of tuning complexity for practitioners looking to adopt the method.

- The entire method is built on the assumption that a model's behavior when conditioned on a few examples in-context (ICL) is a faithful approximation of how its parameters would change after a gradient-based update on those same examples (fine-tuning). However, for large SFT training, I am not sure if ICL can mimic the benefit from SFT.

### Questions
+ In the description of the demonstration set C (line 114), what does the function s(...) represent? It appears to be a formatting function but is not explicitly defined.
+ Could a direct comparison of the ICA scores and the resulting data rankings be provided against those computed by the RHO-Loss method? This would offer deeper insight into how closely ICA's efficient approximation mirrors the baseline's more expensive calculation.
+ How sensitive is the method's performance to the quality of the holdout examples used for in-context demonstration? It would be insightful to see an ablation study where demonstrations are drawn from random training examples instead of a curated, high-quality set, especially in scenarios where the training and test data share a similar distribution.
+ Could this data selection method be effectively applied to fine-tuning tasks that require complex reasoning, such as solving math problems or answering scientific questions?
+ The derivation for the holdout loss approximation in Equation 3 relies on a conditional independence assumption. This seems to imply that adding a single example (x, y) does not significantly alter the model's posterior. How valid is this assumption in practice, especially since adding more examples to the context is known to change the model's output?

### Soundness
2

### Presentation
3

### Contribution
2
