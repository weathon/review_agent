# Guided Speculative Inference for Efficient Test-Time Alignment of LLMs

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6

## Abstract
We propose Guided Speculative Inference (GSI), a novel algorithm for efficient reward-guided decoding in large language models.
GSI combines soft best-of-$n$ test-time scaling with a reward model $r(x,y)$ and speculative samples from a small auxiliary model $\pi_S(y\mid x)$. We provably approximate both the optimal tilted policy
$\pi_{\beta,B}(y\mid x) \propto \pi_B(y\mid x)\exp(\beta\,r(x,y))$ of soft best-of-$n$ under the base model $\pi_B$, as well as the expected reward under the optimal policy. In experiments on reasoning benchmarks (MATH500, OlympiadBench, Minerva Math, MMLU-STEM, GSM8K) and across different model families, our method achieves higher accuracy than standard soft best-of-$n$ with $\pi_S$ and reward-guided speculative decoding (Liao et al., 2025), and in certain settings even outperforms soft best-of-$n$ with $\pi_B$, while reducing end-to-end latency by up to 28%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces Guided Speculative Inference (GSI), an efficient decoding algorithm that integrates reward-guided generation with speculative sampling from a smaller auxiliary model. GSI provably approximates the optimal soft best-of-n policy and achieves higher accuracy on reasoning benchmarks than both standard soft best-of-n and prior reward-guided speculative decoding methods.

### Strengths
1. The paper is well written and easy to follow.
2. The proposed approach is theoretically grounded, and the theoretic analysis holds under more realistic conditions compared to SPECS.
3. GSI empirically achieves better performance than other reward-guided speculation approaches, when evaluated on a wide range of math benchmarks.

### Weaknesses
The authors evaluate on only a single pair of LLMs and a single PRM. This makes the practical generalizability of their findings questionable. Also, the evaluation is heavily focused on math benchmarks.

### Questions
Could you provide some more results on diverse setup, including different LLMs, PRMs, and non-mathematic benchmarks?

### Soundness
2

### Presentation
3

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
This paper proposes Guided Speculative Inference (GSI), a test-time algorithm that mixes (i) soft best of n (S-BoN), ii) a process reward model PRM $r$ and iii) speculative drafts from a small model $\pi_s$ to approximate the reward tilted distibution of a larger base/target model $\pi_B$. The method computes the tilted rewards as a linear combination of the rewards of generated seequences and loglikelihood ratios of those sequences between the target (big) model and the draft (small) model. The paper claims theoretical guarantees (KL bound to $\pi_{\beta,B}$ which is their approximation (GSI) algorithm, and empirical gains on math-reasoning benchmarks (MATH500, OlympiadBench, Minerva, MMLU-STEM, GSM8K) against RSD and S-BoN (with small model) and states that rarely GSI even outperforms S-BoN with big model.

### Strengths
* The algorithmic description with pseudocode in Algorithm 1 is clear.
* Reasoning-trace visualizations (Fig. 3) aid intuition about reject/accept decisions.

### Weaknesses
* The biggest problem is that the paper claims efficiency but the presented algorithm is not efficient at all on hardware utilization. Paper serves three models on separate H100s; no FLOPs/GPU-seconds/tokens-per-second or memory footprint comparisons vs S-BoN($\pi_B$) is provided. It seems that GSI uses more memory footprint compared to  S-BoN($\pi_B$)  (almost 30-50% more given that for reasoning traces the KV-Cache dominates model parameters hence small model and large model usage are comparable). Nonetheless, GSI fails to outperform S-BoN($\pi_B$) on most benchmarks, thus there is no or little incentive to use GSI.

* Aside from higher burden on the hardware, GSI could claim they are efficient because of being faster. However there is no clear demonstration of GSI scaling better with the time spent for answering problems against the accuracy in those problems. The x-axis in Figure 1, Figure 4 and Figure 5 have $n$, but this is expected and do not convey much information. Nonetheless, the question arises: why would it be desirable to run GSI when someone can run S-BoN($\pi_B$) with fewer resources and get better results?
* Theorem 1 bound is practically vacuous at $\beta=20$ with $r \in [0,1]$. No empirical check on $\chi^2(\pi_B||\pi_S)$ or the coverage constant $C_\infty$. The guarantee does not meaningfully inform realistic $n$.

Overall, I do not think this paper will be practically relevant or desirable to implement, and theory seems even further away from being useful.

### Questions
* Report end-to-end metrics: GPU-seconds, FLOPs, tokens/sec, peak VRAM, and number of served models per method. 
* Include curves of accuracy vs end-to-end latency under a fixed hardware budget; compare to S-BoN($\pi_B$) and best-of-n baselines.
* Can you quantify PRM calibration and correlation with ground truth correctness? 
* Please provide the time-breakdown of different components in the algorithm: how much time is spent on big model and how much time is spent on small model generation. 
* Compare all the frameworks here against just doing speculative decoding with the same Big and Small models and doing S-BoN. 
* Please provide empirical estimates or upper bounds for $\chi^2(\pi_B||\pi_S)$ (or $C_\infty$) on your tasks; discuss practicality at $\beta=20$.If the bound is loose, what guidance does it give practitioners?

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
- The paper presents an interesting approach for efficient reasoning by combining speculative decoding with reward guidance.
- Demonstrates improved accuracy over state-of-the-art baselines, though at the cost of higher compute.
- The paper is well-written, and the proofs are well-justified.

### Strengths
- Novel integration of speculation and reward guidance for reasoning tasks.
- Strong empirical results showing accuracy improvements.
- Theoretical analysis and proofs are clear and rigorous.
- Writing quality is high, making the paper easy to follow.

### Weaknesses
- The scaling factor $\beta$ (to balance reward and log-likelihood ratio) requires exhaustive search, which reduces practical efficiency.
- Higher computational cost compared to standard RSD methods.
- Limited discussion on scalability and efficiency trade-offs.

### Questions
1. The draft model appears to perform strongly based on Table 1, particularly as $n$ increases. Could the authors provide an analysis of the proposed method’s effectiveness when using weaker draft models?
2. For draft models with higher $n$, performance seems competitive. Could the authors report the accuracy for the s-BON draft configuration with $n=16$?
3. How is the activation point for the PRM model determined? or how many tokens are generated before calling the PRM. What heuristic is this based on? For reference in speculative decoding, the optimal draft length is downstream task dependent. 
4. The new reward formulation incorporates the logarithm of the acceptance criterion. Could the authors elaborate on any theoretical connections between this formulation and standard speculative decoding?
5. The experiments utilize Qwen2.5, while newer and stronger models such as Qwen3 are available. Could the authors clarify why Qwen3 was not considered?
6. The PRM model is reported to be the same size as the base model. Were any ablation studies conducted with smaller PRM models to assess trade-offs between performance and computational cost?
7. In Theorem 2, expression (b), is there a missing term involving $\tilde{\pi}_{\text{GSI}}(Y\geq)$​ ?
8. In Figure 6, each curve appears to correspond to a different value of β\betaβ. Do the circular points on the curves represent specific β\betaβ values? If so, please clarify.
9. The paper mentions that Table 2 compares GSI with SPECS, yet there is no row for SPECS in the table. Could the authors explain this discrepancy?

### Soundness
3

### Presentation
3

### Contribution
3
