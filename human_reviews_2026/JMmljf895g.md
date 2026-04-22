# Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs

- Avg Score: 3.00
- Decision: Accept (Poster)
- Scores: 4, 2, 2, 4

## Abstract
Speculative decoding is widely used in accelerating large language model (LLM) inference. In this work, we focus on the online draft model selection problem in speculative decoding.  We design an algorithm that provably competes with the best draft model in hindsight for each query in terms of either the token acceptance probability or expected acceptance length. In particular, we show that we can accurately evaluate all draft models, instead of only the chosen model without incurring additional queries to the target model, 
which allows us to improve exponentially over the existing bandit-based approach as the number of draft models increases.
Our approach is generically applicable with any speculative decoding methods (single draft, multi-drafts and draft-trees). Moreover, we design system-efficient versions of online learners and demonstrate that the overhead in computation and latency can be substantially reduced. We conduct extensive experiments on open-source LLMs and diverse datasets,
demonstrating that our methods substantially outperform the state-of-the-art EAGLE3 and the BanditSpec baseline in a variety of domains where specialized domain-expert drafters are available,
especially when long reasoning chains are required.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers draft model selection in speculative decoding. It designs an online learning algorithm, HedgeSpec, which chooses the draft model in an online manner with full-information feedback, i.e., all the draft models can be evaluated without consuming much additional time. Theoretical guarantees for the token acceptance probability and expected acceptance length are provided, compared to the best fixed draft model in hindsight. Experiments are carried out with Llama-3.1-8B and Qwen-3-8B as the target models, and Eagle 3 series as the draft models.

### Strengths
1. The idea of using the verified trajectory to evaluate all the draft models is novel. It circumvents the bandit feedback setup in the literature and provides an alternative full-information with delayed feedback approach.
2. It provides some theoretical guarantees on the performance of HedgeSpec for both acceptance probability and expected acceptance length objectives.
3. Empirical experiments with Eagle 3 are conducted to demonstrate the effectiveness of the proposed approach.

### Weaknesses
1. **More clarity:** Some definitions of the notations are unclear. For example, do the two bounds in Theorem 4 hold at the population level or the prompt level? Are the probabilities conditional on $x_{<t}$ or on the entire input prompt? Is $T_{\text{token}}$ fixed, or is it a stopping time? If it is fixed, what happens when the generated sequence terminates before $T_{\text{token}}$? If it is a stopping time, how does the expected result look?
    
2. **Lack of interpretation:** More interpretation of the regret results would be helpful. In particular, for the result on accepted length (Line 299), while _AcceptLength_ is positively correlated with the quality of the draft model, its practical meaning is not well elaborated. For instance, the algorithm initiates speculative decoding only on a subsequence of the token sequence, and this subsequence is random.
    
3. **Limited draft methods:** The experiments are conducted exclusively with the EAGLE-3 series, providing limited evidence for the claim in Line 66 that “HedgeSpec could work with any given set of drafters on any speculative decoding method.”
    
4. **Mismatch between theory and experiments:** The theoretical results are derived under an adversarial setting, whereas the empirical studies are conducted under an i.i.d. assumption (Line 318). While parameter tuning in experiments is understandable, the shift from adversarial to i.i.d. settings may be too large. Moreover, the motivation for adopting the i.i.d. assumption is unclear, since tokens in autoregressive generation depend on all previous tokens.

**Minors (mostly about writing)**:
- I suggest that the authors carefully review the notations throughout the manuscript to ensure consistency and correctness, which would significantly improve readability. A few examples I noticed are listed below:
    
    1. Inconsistent notation: Accept_Length, $T_{Token}$.
        
    2. Line 199: an extra “=” in $q_i$.
        
    3. Line 209: the conditional probability expression appears problematic.
        
    4. The use of $\gamma_{i,j}$.
        
    5. Line 760: $h$ should be $t$.
        
    
- For self-consistency, it would be better to include the algorithms used in the appendix—for example, Algorithms 1 and 2 from Joulani et al.—as the _BASE_ algorithm in Theorem 4 does not appear within this manuscript.
    
- Since the paper claims that drafter selection can be performed in a full-information setting, it would be useful to explicitly discuss the corresponding losses and gains from both theoretical and empirical perspectives. There is likely a trade-off between faster learning and computational cost, as evaluating more draft models inevitably increases time and resource consumption.

### Questions
1. The proof of Theorem 4 invokes Theorem 1 from Joulani et al. (2013), which requires the delays to be independent of the forecaster’s prediction. However, this assumption seems violated in the setup of this paper, as the delays depend on the generated tokens of the chosen drafter model. Could the authors clarify this point?    
2. Could the authors elaborate on Line 52: “Hedge or NormHedge to identify the best drafter _exponentially_ faster in N than bandit-based approaches”? It would be helpful to specify in what sense the improvement is exponential.
3. According to Figure 2, does this imply that the statistics at one position are reused multiple times? Can the authors further clarify on how these statistics are shared or updated across positions?
4. As claimed in Appendix B.5, evaluating more draft models slightly increases latency in the inference system. How does this latency scale with the number of draft models N? If the increase becomes non-negligible for large N, would a semi-bandit feedback mode—evaluating only a subset of the draft models per step (between bandit and full-information mode)—be a more practical strategy?
5. If one relaxes the acceptance criterion on Line 96, does the proposed method still work? How to adapt the method to this scenario? 

Please also refer to the weaknesses above.

I favor the delicate way to evaluate all draft models and would appreciate it if the authors can resolve my concerns.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses adaptive drafter selection in speculative decoding for large language models. Instead of relying on a fixed small drafter, the authors propose HedgeSpec, an online learning framework that dynamically selects among multiple drafters during inference.

The key insight is that speculative decoding inherently provides full-information feedback—the target model’s verified trajectory allows computation of all drafters’ performance without additional target calls. This reframes the problem from a bandit to a full-information setting, enabling no-regret guarantees via standard Hedge algorithms.

### Strengths
1. The paper presents a clean and insightful reframing of drafter selection as a full-information online learning problem. This perspective simplifies speculative decoding analysis and allows provable no-regret guarantees without additional target calls。

2. The proposed HedgeSpec framework is efficiently realized and demonstrates consistent throughput and acceptance gains across multiple domains and model families (LLaMA-3.1-8B and Qwen-3-8B).

### Weaknesses
- Although the evaluation phase introduces minimal overhead per drafter, it assumes access to sufficient computational resources for parallel evaluation. On resource-constrained systems, this could become a bottleneck. Since the results are only reported on 8 A100 GPUs, it would be helpful to include experiments on different hardware configurations and a breakdown of evaluation time and memory usage as the number of drafters N increases.
    
- The theoretical analysis relies on the assumption that feedback for all drafters can be computed without bias from the verified trajectory. Any deviation from this assumption may affect empirical optimality, and the error may accumulate as decoding process proceeds. An ablation across different numerical precisions would strengthen this claim.
    
- The evaluation scope is somewhat limited. (1) The method is described as training-free (Line 1043), but the draft models are pre-trained on specific datasets; and (2) the experiments primarily reuse these datasets. Beyond GSM8K and HumanEval, a broader evaluation on benchmarks such as SpecBench would be valuable.
    
- The experiments use Llama-3.1-8B-Instruct and Qwen-3-8B as target models. Demonstrating results on larger models would make the findings more convincing.
    
- Including experiments with batched prompts would provide insight into the method’s effectiveness in realistic, production-like settings.
    
- It would be interesting to discuss or demonstrate how the proposed approach integrates with other speculative decoding methods such as PLD, Medusa, and SAM.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper argues that multi-drafter selection for speculative decoding is not a bandit problem: by leveraging the single verified trajectory produced during standard speculative decoding, one can compute full-information feedback for all candidate drafters at no extra target-model calls. 

Building on this, the authors introduce HedgeSpec, which applies experts/hedging algorithms (e.g., Hedge/NormalHedge) with no-regret guarantees under two objectives—ETAP (expected token acceptance probability) and EAL (expected acceptance length). The paper formalizes ETAP/EAL, derives an unbiased off-policy estimator for EAL, and handles delayed/censored feedback. Empirically,  Llama-8B and Qwen 8B targets and pools of domain drafters are evaluated.

### Strengths
- Uses the verified trajectory to evaluate all drafters, enabling no-regret experts methods without extra target calls.

- Clear metrics  and formal treatment of delayed/censored feedback.

- MAT and tokens/s improve versus EAGLE and bandit baselines across domains, but not domain specialized recent specdec are considered.

- Scales with drafter pools for regret and MAT vs N are reported.

### Weaknesses
1. Related-work positioning is incomplete. The paper should explicitly acknowledge that the one other same online drafter-selection problem was formulated as a bandit [A], and compare assumptions/guarantees head-to-head (feedback availability, regret constants, robustness to partial/censored logs). It should also cite that motivates domain-specialized drafter pools [B].

[A] A Unified Framework for Speculative Decoding with Multiple Drafters as a Bandit, Kim et al.

[B] Towards Fast Multilingual LLM Inference: Speculative Decoding with Specialized Drafters, Yi et al.

2. Estimator quality not fully characterized. Unbiasedness is shown, but variance under realistic  D (long contexts, domain shift) is not analyzed; large variance could slow adaptation or misrank experts early. 

3. Systems analysis is latency-centric. Deployment-relevant throughput (under concurrency), HBM/KV-cache footprints, and a roofline view (operational intensity vs. bandwidth ceilings for drafter prefill vs. target verify) are missing. Current overhead tables don’t fully address multi-tenant contention. 


4. Limited breadth of targets/drafters. Results center on 8B-class targets and EAGLE-style domain specialized drafters; broader speculative schemes (REST/Medusa/Spectr) and larger targets would strengthen generality. 


5. The method optimizes EAL/ETAP but reports MAT/tokens-per-second; a direct analysis of objective mismatch is absent.

### Questions
1. Theory vs. Kim et al. (bandit). Under what logging or feedback constraints does your full-information assumption fail so that the bandit view is preferable? Please summarize regret/variance trade-offs and practical implications.

2. Specialized drafter pools. Following Yi et al. (multilingual/specialized drafters), how does HedgeSpec perform when the pool contains varaints of EAGLE specialists per domain/language? How quickly is mis-specialization corrected under shift?

3. Estimator robustness. Do you report empirical variance/CI for EAL across domains and K? Any stabilization (clipping, shrinkage, delayed-feedback smoothing)? 


4. Throughput & concurrency. With N drafters, what are tokens/s under realistic concurrent loads, and how do you cap parallel drafter evaluations to avoid HBM pressure/KV thrash? Please report GPU util, queueing delays, and prefill-verify overlap.

5. Roofline & memory. What are per-drafter HBM and KV-cache footprints at target context lengths, and where do the kernels sit on a roofline plot for (i) multi-drafter prefills and (ii) target verify?

### Soundness
2

### Presentation
2

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
The paper introduces speculative decoding with multiple specialized drafters. Contrary to previous bandit-based approaches where one picks single drafter and get information from them, author developed counter-factual estimator which can utilize mutliple drafter's accepntance rate, thereby converting bandit problem to online learning scenario, which proves to be more effective in EAGLE-3 based experiments.

### Strengths
* Converting bandit problem to the online learning setting with theoretical guarantee (one-step estimator in Theorem 3) seems novel and interesting.  
* Experiment is conducted on one of the SOTA method (EAGLE-3 [1]). 
* Proper ablations are conducted with sound presentations.

### Weaknesses
* **Drafter overhead** : While authors show utilizing multiple drafters in every SD step takes little overhead, this could be only hold for EAGLE-3 like model (where drafter size is way smaller than the target model). While EAGLE-3 shows SOTA performance, other method utilize relatively larger drafter (Medusa [2] for instance). Moreover, in real serving scenario, there might be SD framework with larger drafter. This could increase the cost of the drafter overhead linearly or reduce the merits of the online algorithm performance by increased drafter overhead. Discussion reagrding this scenario would be beneficial.

* **Hyper-parameter for online algorithms** : The algorithm leverages (dealyed) Hedge algorithm which is standard algorithm in online learning. However, the regret bound is proved with the learning rate $\eta$ which depends on the known horizon $T$ or current round $t$ (for anytime regret) but it might not be optimal for parcitcal performance. It would be good that author can clarify about the base algorithm they used. 

* **Non-stationary scenario** : In real-world scenario, a single query can be non-stationary especially for long context or contained in multiple scenario. In this case, Hedge algorithm might not be optimal.

### Questions
Questions 

* Can authors provide more experimental details about fine-tuning the drafters in 4.2? 

* Can you test the HedgeSPEC on batched inference scneario and prove the thorughput improvement?

* While author cited the algorithm of the paper, it would be good to clearly state the overall algorithm (with base algorithm) is actually used in the experiment for clearer presentation.

* (Minor) It seems like bandit-based approaches with multi-drafter scenario is first proposed by MetaSD [3], which is not properly reflected in the current manuscript.

[1] (Li et al.) EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test

[2] (Cai et al.) Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads

[3] (Kim et al.) A Unified Framework for Speculative Decoding with Multiple Drafters as a Bandit

### Soundness
3

### Presentation
2

### Contribution
3
