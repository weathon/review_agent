# MC-SJD : Maximal Coupling Speculative Jacobi Decoding for Autoregressive Visual Generation Acceleration

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
While autoregressive (AR) modeling has recently emerged as a new paradigm in visual generation, its practical adoption is severely constrained by the slow inference speed of per-token generation, which often requires thousands of steps to produce a single sample. To address this challenge, we propose MC-SJD, a training-free, lossless parallel decoding framework designed to accelerate AR visual generation by extending the recently introduced Speculative Jacobi Decoding (SJD). Although SJD shows strong potential for accelerating AR generation, we demonstrate that token instability across iterations significantly reduces the acceptance rate, a limitation that primarily arises from the independent sampling process used during draft token generation. To overcome this, we introduce MC-SJD, an information-theoretic approach based on coupling, which substantially accelerates standard SJD by maximizing the probability of sampling identical draft tokens across consecutive iterations, all while preserving its lossless property. Remarkably, this method requires only a single-line modification to the existing algorithm, yet achieves substantial performance gains, delivering up to a ~3.8x speedup in image generation and ~10x speedup in video generation compared to standard AR decoding, without any degradation in output quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the slow inference of autoregressive (AR) visual models by improving Speculative Jacobi Decoding (SJD). The authors identify that SJD's performance is bottlenecked by token instability, where independent sampling creates dissimilar draft tokens even when underlying probability distributions are close. The proposed solution, MC-SJD, is a lossless and training-free framework that uses "Coupling" to maximize the token similarity between iterations, thereby stabilizing convergence. The method demonstrates impressive speedups, achieving up to ~$3.8\times$ in image and ~$10\times$ in video generation without any degradation in output quality.

### Strengths
* Well-written and easy to follow.
* Provides a novel problem and method in SJD based on the theoretically sound principle of coupling.
* Demonstrates significant empirical speedups over baselines. The inclusion of experiments on both image and video generation tasks is a positive aspect of the evaluation.

### Weaknesses
1. **Limited Empirical Support for Motivation**
	* The correlation in Fig. 2, which is used to empirically validate Observation 1, is not fully convincing. The Y-axis is restricted to an extremely narrow range (59.0–61.0), which makes the "strong correlation" difficult to confirm. The argument would be far more compelling if SJD, GS-SJD, and MC-SJD were all plotted on the same graph. This direct comparison would clearly visualize whether the proposed methods truly shift performance toward lower token difference and lower NFE, which is central to the paper's hypothesis.
	
2. **Apparent Discrepancy Between Theory and Results**
	* An interesting discrepancy arises in Table 1 ($L=32$), where the theoretically suboptimal $\pi_{GS}$ (Gumbel Coupling) achieves a better NFE than the "optimal" $\pi_{MC}$. The paper fails to provide any analysis for this critical discrepancy, undermining its own central argument.
	
3. **Incomplete Experimental Reporting**
	* Tables 2 and 3 report results for an ambiguous "Ours" metric, failing to specify whether $\pi_{MC}$ or $\pi_{GS}$ was used. As established in Weakness 2 (and Table 1), the performance trade-off between $\pi_{MC}$ and $\pi_{GS}$ seems inconsistent and setup-dependent. Therefore, it would be necessary to report results for both methods to provide a complete and transparent comparison.
	* Table 3 (Janus-Pro) omits wall-clock latency. This is a critical metric for evaluating the method's practical benefit, especially given the computational overhead of $\pi_{MC}$ and $\pi_{GS}$ observed in Table 1.
	* The validation for each task is confined to a single dataset (MSCOCO2017 for images, real-state-10k for video), which limits the assessment of the method's generalizability.

4. **Lack of Limitation**
	* The paper does not provide any discussions regarding limitations.

### Questions
1. The paper speculates that Gumbel Coupling ($\pi_{GS}$) promotes "long-range stabilization". Could the authors elaborate on this concept? Specifically, what is the mechanism of this stabilization (e.g., how does sharing the same Gumbel noise across all iterations achieve this effect), and what is its concrete role in the decoding process? Does it, for example, help the entire sequence trajectory converge to a stable state more quickly?

2. Following the first question and weakness 2, Table 1 shows that at $L=32$, $\pi_{GS}$ achieves a better NFE than the theoretically optimal $\pi_{MC}$. Does this finding suggest that the "greedy" optimization of $\pi_{MC}$ (maximizing immediate $t$ vs. $t-1$ collision) is not always the globally optimal strategy for minimizing total NFE? Does it imply that, depending on the setup (like window size L, model, or task, etc), the "long-range" approach of $\pi_{GS}$ can actually be the superior strategy?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors show that the bottleneck in Speculative Jacobi Decoding (SJD), low acceptance due to independent draft sampling, can be removed by coupling the draft distributions across iterations. They propose two couplers: Maximal Coupling (with acceptance equal to 1−TV(p,q)) and a cheaper Gumbel-Coupling (shared Gumbel noise) with a provable lower bound on collision probability; both plug into SJD with a one-line change in the drafting step. Theoretical analysis connects acceptance to total-variation distance and shows that independent sampling yields low collision probability bounded by Rényi-2 entropy, especially flat in visual AR; coupled sampling markedly raises acceptance trajectories βₜ (Fig. 3–4).

### Strengths
* Theoretical observation is very interesting. The paper replaces independent drafting by a principled coupling with formal guarantees (Theorem 2/3). The relationship between the acceptance and the total-variation is clear.

* Drop‑in practicality. The “single‑line” modification to SJD is attractive for practitioners and retains lossless correctness of speculative decoding. 

* Strong speedups across both image and video domains, scaling favorably with larger SJD windows where vanilla SJD saturates.

### Weaknesses
I'm not an expert in this area, but I have some concerns about the paper based on my understanding.

1. I have some doubt on compute/memory cost of coupling. What is the runtime and memory overhead for maximal vs. Gumbel coupling (per step, per window L)? Any GPU kernel implications vs. independent sampling?

2. Is there any failure modes observed empirically? For example, when p and q are both flat (I think it would be common in AR images), TV is small but entropy is high. Does coupling still help, or do we hit the Rényi‑2 bound and stagnate? Please show acceptance curves for extreme‑flat logits.

3. CFG interaction: you note speed dips at higher guidance (sharper logits). Can you quantify the acceptance–guidance trade‑off and provide an adaptive lambda schedule for maximum throughput?

### Questions
Please refer to the weaknesses.

### Soundness
4

### Presentation
4

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
The manuscript proposes MC-SJD, a training-free modification of Speculative Jacobi Decoding (SJD) for autoregressive (AR) visual generation. The manuscript analyzes that the acceptance rate is negatively proportional to the total variation of the drafter and verifier distributions, and argues that independent drafting in SJD yields very low collision under flat vision logits. Aware of this, they propose a key idea to couple the draft sampling across consecutive Jacobi iterations via maximal coupling (using the same MRS routine as SD verification) or a cheaper Gumbel noise–sharing variant to increase token collisions between drafts, thereby boosting acceptance rates while remaining lossless. Empirically, they report up to ~3.8× image speedup and ~10× video speedup without quality degradation.

### Strengths
1. Clean and intuitive problem statement - points out an overlooked but crucial detriment of SJD.
2. MC-SJD can be employed with minimal implementation change.

### Weaknesses
1. Baseline coverage is a bit limited. While SJD and one baseline (GSD) are included, there’s no comparison with other accepted speculative decoding baselines, such as EAGLE-3.
2. Latency analyses w.r.t. the batch size are absent.
3. Memory overhead analysis of the cached probabilities is needed, especially w.r.t. the batch size and window length.
4. Latency breakdown or microbenchmark for the sampling process would benefit the paper.

### Questions
1. When using top-k sampling (as in the experiments section) or using CFG, how is the lossless-ness guaranteed? How does the sampling process go?

### Soundness
3

### Presentation
3

### Contribution
3
