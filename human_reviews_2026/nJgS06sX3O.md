# DefensiveKV: Taming the Fragility of KV Cache Eviction in LLM Inference

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6, 6

## Abstract
Large language models have revolutionized natural language processing, yet their deployment remains hampered by the substantial memory and runtime overhead of the transformer’s Key-Value cache. To mitigate this, recent methods employ a scoring-aggregation framework to evict unimportant cache entries, based on the "stability assumption"—that a fixed subset of entries remains consistently important during generation. However, prior work has largely focused on refining importance indicators for scoring, while defaulting to mean aggregation due to a faithful trust in the stability assumption. In this work, we argue that this underlying assumption is inherently fragile, making mean aggregation highly vulnerable in extreme cases. To counter this, we propose a simple yet elegant defensive aggregation strategy: a two-step, linear-time approach that controls worst-case risk, thereby defending against extreme cases with negligible computational overhead. Embodying this strategy, we propose a novel cache eviction method, DefensiveKV and its extension, Layer-DefensiveKV, which incorporates layer-wise budget allocation. Across seven task domains (18 datasets), our methods reduce generation quality loss by 2.3× and 4.3× respectively, versus the strongest baseline under a 20\% cache size. These results set new performance benchmarks and pioneer a promising direction for optimizing cache eviction against underlying fragility through worst-case risk management.Our code is available at https://github.com/FFY0/DefensiveKV .

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work identifies a critical fragility in the common stability assumption underlying KV cache eviction methods for LLMs. It introduces a defensive aggregation strategy that manages worst-case risk by estimating peak importance and applying adaptive correction, moving beyond vulnerable mean aggregation. The resulting methods, DefensiveKV and Layer-DefensiveKV, achieve substantially lower generation quality loss under strong cache size constraints. Comprehensive evaluations across 18 datasets show these methods outperform prior art, reducing quality loss compared with baseline.

### Strengths
* S1) The concept of Defensive Aggregation presents a novel approach to KV cache eviction, and its demonstrated effectiveness is supported by empirical evidence across benchmarks including LongBench and NIAH. 
* S2) The paper is written in a clear way, making it easy to follow the authors' ideas and understand their results.

### Weaknesses
* W1) The primary limitation of this work is its exclusive validation on dense and GQA-based model architectures. The proposed cache eviction method remains unverified for the increasingly prevalent Mixture-of-Experts (MoE) models. Since MoE architectures inherently reduce KV cache usage through selective parameter activation, the fundamental dynamics of token importance may differ. Consequently, the applicability and effectiveness of the defensive aggregation strategy in this critical and distinct setting are currently unknown. 

* W2) The authors mention in their setup that "the context is compressed independently before the question is introduced" is beneficial for multi-turn dialogue, but no experimental validation was conducted on real multi-turn dialogue datasets. 

* W3) While the paper demonstrates strong performance at cache retention rates of 20% and above, it does not systematically evaluate the proposed method under more extreme compression scenarios. The performance trajectory of DefensiveKV when cache size drops below 10% remains unclear. For example, HeadKV evaluates its performance when retaining an average of 128 to 1024 tokens on LongBench. 

* W4) The authors' experiments primarily focus on long-text datasets. I think it would be better to test the performance of the proposed method on mathematical and reasoning tasks.

* W5) There is a lack of theoretical analysis or mathematical proof to demonstrate why defensive aggregation outperforms mean aggregation.

* W6) It would be helpful if the authors could provide additional information to support reproducibility.

### Questions
Please refer to Weaknesses.

### Soundness
2

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
3

### Summary
The paper “Taming the Fragility of KV Cache Eviction in LLM Inference” argues that existing KV cache eviction methods rely on a fragile stability assumption—that a fixed subset of cache entries stays important throughout generation. This makes the common mean aggregation approach unreliable under instability. The authors propose Defensive Aggregation, a linear-time method that manages worst-case risk instead of average importance. Building on this, they introduce DefensiveKV and Layer-DefensiveKV, which achieve over 2×–4× lower quality loss than state-of-the-art baselines (e.g., CriticalKV) at small cache budgets, with negligible computational overhead. The work pioneers a “worst-case risk-aware” perspective for robust and efficient LLM cache eviction.

### Strengths
1. strong experiments comparing with many previous baselines. improvement is significant.
2. New idea of adding online adapatation.
3. Clearly identifies and formalizes a previously overlooked fragility in the “stability assumption” behind KV cache eviction and proposes a simple, elegant, and computationally cheap fix (Defensive Aggregation).

### Weaknesses
1. The “worst-case” heuristic lacks formal guarantees or ablation beyond empirical trends. Need to show some results on why this worst case is guaranteed besides it works empirically.
2. No discussion on interaction with quantization, retrieval, or offloading beyond brief appendix notes.

### Questions
Can we have some more formal guarantee or experiments to show the worst case scenario is prevented?
Can we give some discussion on how this online process can be potentially integrated into inference?

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
This paper first argues that most of the existing methods (such as snapkv, criticalkv) are committed to improving the scoring step (i.e., looking for better importance indicators). They use simple average aggregation by default because they rely on a stability assumption: a small group of key cache entries will remain important throughout the generation process. However, the importance of cache entries may suddenly decline sharply at some time, resulting in extreme cases or outliers. The paper shows that the worst-case risk should be addressed. To do this,  this paper proposes a new defensive aggregation strategy to control the worst-case risk, which combines worst-case risk estimation and adaptive prior-risk correction. A DefensiveKV, which is a basic version, and a Layer-DefensiveKV are proposed, which integrate a layer-wise budget allocation strategy, respectively.

### Strengths
1. This new aggregation strategy has the same linear time complexity as the average aggregation strategy, and the additional computational overhead is negligible.

2. Significant performance improvement

3. Good writing and illustrations.

### Weaknesses
1. Is it possible that Adaptive Prior-Risk Correction will lead to: In a head with high risk, it may wrongly raise the score of really unimportant items; In a head with generally low risk, it cannot discover those rare key items that do not show high risk in the limited observation window.


2. It seems that in the code dataset, most SOTA methods provide elegant results. Could you provide some discussion about this?

3. In cirticalKV, the score is evaluated by using the attention weights and norm of V*W^O, which shows an excellent performance. This paper also adopts the same strategy. I also want to see some discussion about this: How many output differences does this paper's method reduce?

4. Algorithm 1 is conducted in layer-wise or head-wise?

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper revisits the standard scoring–aggregation pipeline for KV cache eviction in LLM inference and argues that the usual mean aggregation is fragile because the importance of cache entries can vary abruptly during generation. Based on this, this paper introduces Defensive Aggregation, a linear-time, two-step rule that first estimates a worst-case risk per entry via a maximum over historical-token scores and then applies an adaptive prior-risk correction, yielding two methods, DefensiveKV and Layer-DefensiveKV with layer-wise budget allocation. Across LongBench and Needle-in-a-Haystack, the introduced methods achieve better performance versus baselines at various cache budgets.

### Strengths
* The introduced method is well-motivated by convincing observations about the fragility of importance stability.

* Thorough experiments and analyses are presented in the paper.

### Weaknesses
* Could you elaborate on the rationale behind fixing the historical window size at 32 tokens? Are there any experiments to evaluate the sensitivity to this choice?

* The cache budget settings could be more varied. 20% cache is not a strictly small budget. Would it be possible to provide experimental results under small budgets such as 1% and 2% to demonstrate how the method behaves under extreme compression conditions?

* Missing discussion on non-scoring-aggregation pipelines in the paper such as “Lacache: ladder-shaped kv caching for efficient long-context modeling of large language models (ICML2025)," which would help situate DefensiveKV more clearly within the broader space of KV cache compression methods.
 
* Why on needle-in-a-haystack tasks, DefensiveKV outperforms Layer-DefensiveKV sometimes? Could you please provide any insight into under what circumstances DefensiveKV/Layer-DefensiveKV will perform better? A short discussion on this might help readers better understand their trade-offs.

### Questions
Will the code be open-sourced? I think it would be valuable for the community.

### Soundness
3

### Presentation
3

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
This paper identifies a key issue in KV-cache eviction: prior methods assume that importance scores derived from historical tokens remain stable. However, as generation progresses, these scores can shift significantly, making that assumption unreliable. To address this, the authors propose a worst-case aggregation strategy that accounts for score fluctuation and improves eviction effectiveness.

### Strengths
1. This paper identifies a new problem in KV cache eviction methods: the aggregation step ignores the worst-case risk.
2. Comprehensive experiments evaluated the effectiveness of the proposed method.
3. Simple but effective method for optimizing KV eviction methods further.
4. Good writing, easy to follow, and well motivated.

### Weaknesses
1. Experiments:

+ The proposed method is orthogonal to many existing KV-cache eviction strategies. To ensure fair comparison and demonstrate the generalizability of your approach, it would be beneficial to include an additional table showing how your method can be combined with representative prior techniques. For example, selecting one model and one benchmark to report results for “baseline methods” versus “baseline + your method” would more clearly highlight the benefits.

      | Methods       | Task 1 | Task2 |
      | ------------- | ------ | ----- |
      | e.g. SnapKV   | ...    | ...   |
      | + DefensiveKV | ...    | ...   |
      | e.g. CAKE     | ...    | ...   |
      | + DefensiveKV | ...    | ...   |

2. Ablation Study:

+ In the default setting, the historical window size is set to 32. According to the paper’s analysis, this hyperparameter plays a crucial role in evaluation, as it affects both scoring and aggregation behaviors. Therefore, it would be valuable to include an ablation study on the historical window size to better understand its impact.

3. Typo:
+ Figure 3 (a) 16st --> 16th

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
