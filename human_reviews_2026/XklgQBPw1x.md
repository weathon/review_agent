# Latent Guided Sampling for Combinatorial Optimization

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Combinatorial Optimization problems are widespread in domains such as logistics, manufacturing, and drug discovery, yet their NP-hard nature makes them computationally challenging. Recent Neural Combinatorial Optimization (NCO) methods leverage deep learning to learn policies for constructing solutions, trained via Supervised or Reinforcement Learning. While promising, these approaches often rely on task-specific augmentations, perform poorly on out-of-distribution instances, and lack robust inference mechanisms. Moreover, existing latent space models either require labeled data or use an instance-independent latent distribution. In this work, we propose LGS-Net, a novel latent space model that conditions on problem instances, and introduce an efficient inference method, Latent Guided Sampling (LGS), based on Markov Chain Monte Carlo and Stochastic Approximation. We show that the iterations of our method form a time-inhomogeneous Markov Chain and provide rigorous theoretical convergence guarantees. Empirical results on benchmark routing tasks show that our method achieves state-of-the-art performance among NCO baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes LGS-Net, a latent-space model for Neural Combinatorial Optimization (NCO). The approach trains an encoder to map a problem instance into a distribution for a latent variable.  Based on a sample of this distribution, the decoder generates a distribution over the solution space which then is sampled to obtain solutions. At inference time, the authors introduce Latent Guided Sampling (LGS) — an MCMC-based method augmented with stochastic approximation updates to the decoder parameters, allowing test-time adaptation.
Theoretical analysis establishes convergence guarantees for both fixed and adaptive parameter settings. Experiments on TSP and CVRP benchmarks show that LGS-Net achieves state-of-the-art or near-optimal performance, sometimes improving upon strong baselines such as COMPASS and EAS by small margins.

### Strengths
- The paper addresses recognized limitations of prior NCO approaches (e.g., data requirements, lack of inference-time optimization) and positions the work within a clear conceptual framework.

- Principled combination of latent modeling and RL. The unification of stochastic approximation with MCMC inference in a learned latent space is elegant and theoretically grounded.

- Theoretical soundness. The convergence proofs are rigorous and build on established MCMC theory, adapted to a time-inhomogeneous setting.

- Comprehensive experimental comparison. The authors benchmark against numerous recent NCO baselines and demonstrate consistent improvements, confirming the competitiveness of the approach.

### Weaknesses
- Unclear role of the concept of the latent space: During inference this method employs a local search based on MCMC updates and weight updates of the decoder. This iterative search evaluates the solution quality of intermediate samples in solution space. Hence this method can hardly be considered a latent CO method since it does not learn to solve CO problems via latent space dynamics but relies on generating samples in solution space and evaluating the cost function for them. More generally, the role of the stochastic latent variable is not clear.

- Incremental novelty. The method extends earlier latent-space NCO works such as CVAE-Opt and COMPASS. The architectural and training components are similar, with the main novelty residing in the inference mechanism.

- Limited empirical diversity. All experiments are on Euclidean routing problems (TSP/CVRP ≤ 150 nodes). The approach’s generality to other combinatorial settings (e.g., knapsack, scheduling) remains untested.

- Small empirical margins. Performance gains over prior SOTA are modest (typically < 0.2 % gap). It is unclear whether these differences are statistically significant and practically meaningful given the added complexity of MCMC + SA inference.

### Questions
- Why is the latent variable needed?
- Why not use only a decoder model that takes as input the problem instance x instead of the latent z(x)?
- Why is the latent variable stochastic and why couldn't a deterministic representation work as well?
- Can you show the benefit of having an encoder and the stochastic variable?

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces LGS-Net, a latent-space neural model for combinatorial optimization (e.g., TSP, CVRP). Similar to [Chalumeau et al., 2023], it learns instance-conditioned latent representations using reinforcement learning, yet performs inference through a new scheme based on MCMC called Latent Guided Sampling (LGS).

Contributions:
- A new latent-space NCO model (LGS-Net) that conditions on problem instances.
- A provably convergent inference method (LGS) integrating MCMC and stochastic approximation (SA).
- SOTA results on TSP and CVRP benchmarks (in distribution and slightly out of distribution), outperforming prior methods like POMO, COMPASS, and EAS, with strong generalization to unseen problem sizes.

### Strengths
The paper is overall well-written, gives good credit to related works, and performs decent experiments and ablations.
- Strong theoretical grounding regarding convergence.
- Standard experiments on TSP and CVRP: n in [100, 125, 150].
- SOTA results on these benchmarks, even beaten the solver for CVRP 100-125.

### Weaknesses
The paper and method suffer from a few weaknesses.
- Additional complexity of the inference scheme: MCMC and SA must bring a significant overhead.
- Limited experimental scope: I assume this would transfer well to other combinatorial problems like job-shop scheduling or graph problems, but the paper only tests on TSP and CVRP.

### Questions
1. How would you think about applying such a training and inference scheme outside of this NCO setup?

2. How realistic with respect to real problems are the assumptions made in the convergence analysis?

### Soundness
4

### Presentation
3

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
This paper introduces a new inference strategy for neural solvers applied to combinatorial optimization, with a primary focus on routing problems but applicability beyond them. The approach is based on latent-space models coupled with a subsequent latent-space search. It includes a training phase that anticipates the inference-time search budget and presents a coupled training–inference procedure. The method establishes close connections with prior inference strategies in the literature, namely EAS, COMPASS, and CVAE-Opt. The paper provides both mathematical guarantees and empirical results on two widely studied combinatorial optimization problems (TSP & CVRP) evaluated in- and out-of-distribution.

### Strengths
Significance: The use of inference-time strategies for combinatorial optimization (CO) problems is highly relevant. The proposed method achieves state-of-the-art performance with statistically significant improvements on reference benchmarks, evaluated on two well-studied problems (TSP and CVRP), both in- and out-of-distribution. I appreciate that the paper provides not only SOTA results but also mathematical justification for the proposed approach.

Originality: Most underlying concepts already exist, and the general idea of latent-space search is not new (which the authors acknowledge clearly), but the combination of latent-space modeling, MCMC sampling, and gradient-based updates is novel and supported by solid mathematical analysis. While there is no major paradigm shift, the proposed approach presents an interesting and meaningful contribution.

Clarity: The paper makes a commendable effort to explain the method, and it presents the mathematical components and proofs clearly. The main ideas are easy to follow, and the inclusion of figures, algorithms, and plots effectively supports the reader’s understanding.

Quality: The paper engages well with the existing literature, being transparent about its close connections to COMPASS, EAS, and CVAE-Opt. It presents the mathematical aspects rigorously and includes results and extended materials in the appendix. Overall, the paper demonstrates good quality, though there remains room for improvement (see weaknesses).

### Weaknesses
**W1. Weak motivation and unclear justification of contributions.**

The main motivation of the work remains somewhat weak. Lines 51–54 and 57–60 present the limitations of prior work (EAS and COMPASS) that LGS is supposed to address (lines 60–62). This section is crucial, as it motivates the entire paper, yet it is not fully convincing.
(i) The authors claim that EAS “fine-tunes” the policy, which poses computational challenges, but LGS also performs fine-tuning.
 (ii) The statement that “COMPASS enforces independence between the problem instance and the latent space structure” would benefit from clarification, specifically, how limiting this assumption truly is, and why relaxing it is expected to yield substantial performance gains. 

Moreover, the results and ablation studies do not provide evidence that the “instance dependence” introduced in LGS is responsible for the observed performance improvements. The only ablation reported in the main paper concerns the inference-time search/sampling component. One could therefore wonder whether applying the LGS sampling and parameter-update mechanism to a COMPASS checkpoint would suffice. Overall, the paper does not convincingly establish that the proposed modifications address the stated weaknesses of prior methods. Clearer intuitions about why the proposed approach constitutes a conceptual improvement, and corresponding empirical validation through targeted ablations, would strengthen the work considerably.

Another motivation mentioned in the introduction is the removal of reliance on augmentation tricks. However, it is not clear how LGS achieves this, as other methods such as COMPASS and EAS do not fundamentally rely on augmentation either. They may use it, but they function well without it. The authors should clarify what makes LGS different in this regard; if there is no clear distinction, this point should not be used as a central motivation.

**W2. Missing ablation studies.**

Related to the previous point, some important ablations are missing. The only ablation presented concerns the sampling method, whereas one of the main motivations, conditioning on the instance, is not ablated. Ideally, this should be tested explicitly. Alternatively, the authors could revise their narrative to avoid presenting this aspect as a central feature of the method without empirical support.

**W3. Questionable timing results.**

The timing results reported in Tables 1, 4, and 5 appear inconsistent, which raises concerns about the validity of the experimental study. The times reported for POMO, EAS, and COMPASS do not align with expectations, and the reported runtime for LGS is also surprising.

First, the times for POMO and COMPASS should be roughly identical, as both share the same bottlenecks, the forward pass and environment rollouts. Their only difference is the CMA-ES update, which is negligible (see the original COMPASS paper). Therefore, one should expect similar runtimes across the benchmark, which is not the case (e.g., 10M vs. 20M on TSP100; 1h30 vs. 40M on TSP150).

Second, the reported time comparison between POMO and the other methods lacks coherence, as POMO is sometimes faster and sometimes slower, although it should consistently be equal or faster.

Third, EAS and COMPASS are reported to take the same time, which is implausible, since EAS involves backpropagation steps that should make it slower.

Finally, LGS-Net is reported to have the same runtime as COMPASS, even though it includes regular parameter updates. It should therefore be slower (perhaps only slightly if updates are infrequent), but this needs discussion if the times are indeed identical.

Overall, I am confident that there are issues with the reported timings. The authors should carefully verify and correct these results, and provide more detailed information on the time performance of their algorithm. Beyond hindering fair comparison between COMPASS and LGS-Net, these inconsistencies undermine confidence in the experimental section as a whole.

**W4. Clarity and presentation.**

The clarity of the paper could be improved. Figure 1 could include more detail to illustrate key components. It is somewhat confusing how the encoding interacts with the existing encoding that AM and POMO architectures are already using; the authors should be explicit about what differs in LGS, and how the latent space relates (or not) to the original embedding space. Additionally, certain elements, such as the “proposal distribution” (used in Algorithm 1 but not discussed in Section 4), should be better introduced.

Minor suggestions:
- “To learn solution strategies” (line 14) sounds awkward.
- It is acceptable to use “NCO” without specifying “learning-based NCO,” as the term already implies learning; this would simplify some sentences (e.g., lines 24 and 39).
- Line 52: stating that EAS is a SOTA search method is inaccurate, as it is outperformed on several benchmarks by COMPASS, PolyNet, and MEMENTO. The sentence should be softened to “among the leading methods for search-based RL.” MEMENTO [1] (NeurIPS 2025, Spotlight) could also be added to the related work.
- Line 62: “rely on the augmentation trick” is misleading, most methods are agnostic to it, even if some experiments make use of it.
- Line 113: COMPASS does not learn the latent space; it uses a fixed prior, and the policy is trained to exploit it effectively.
- Line 116: “removes the need for a pre-trained policy” should be clarified, why does this hold?
- Line 251. Small suggestion to swap theta and phi to follow the order in which encoder and decoder are mentioned.
- Line 382. I believe the usual dataset used in the NCO literature for TSP100 has 10000 instances, and TSP125 and TSP150 have 1000 instances. 

[1] Memory-Enhanced Neural Solvers for Routing Problems, Neurips 2025

### Questions
It would be valuable to provide stronger motivation for the design choices introduced in the proposed method. For instance, why was the POPPY loss from COMPASS removed? Why was a latent-space encoder introduced? And why are all decoder parameters updated if the learned latent space is already expected to capture much of the variability? I can anticipate some of the rationale behind these decisions, but I would really appreciate hearing the authors’ explicit perspective. Clarifying these points, ideally supported by additional ablation studies, could substantially strengthen the manuscript.

In Section 4.1, it is also worth noting that the baseline AM/POMO architecture already includes an encoder. It would therefore help to clarify what the new encoder adds, and what type of information is expected to be captured in the latent space that is not already represented in the existing embedding.

It would also be helpful to report the dimensionality of the latent space used in the experiments.

Regarding the reported timings: could the authors confirm whether these values are accurate? Given the inconsistencies mentioned earlier, I am concerned about the validity of the experimental results.

Finally, on line 116, the paper states that LGS “removes the need for a pre-trained policy.” Could the authors elaborate on why this is the case?

Overall, I find the work promising and I am genuinely keen to support it. However, several concerns remain that need to be addressed to lean toward acceptance. I sincerely hope that the authors will be able to provide clarifications and additional results during the rebuttal phase to help advance this discussion.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
LGS-Net is an RL-based latent variable neural combinatorial optimization model that requires no labels. It uses instance-conditioned latent embeddings and trains via a cost-weighted, entropy-regularized objective function. The key innovation is the Latent Guided Sampling (LGS) inference procedure, which updates decoder parameters through stochastic approximation while running interacting MCMC chains in latent space. It provides convergence guarantees for both fixed and adaptive parameters. The model achieves learning-based SOTA on TSP and CVRP benchmarks, and even outperforms LKH3 on some CVRP settings.

### Strengths
The instance-conditioned latent model eliminates the need for pre-computed solutions or pre-trained policies unlike CVAE-Opt and COMPASS, and demonstrates clear performance improvements. LGS inference combines interacting MCMC with real-time parameter updates, consistently outperforming all alternatives including DE, CMA-ES, active search, and gradient-based finetuning. The convergence proof for time-inhomogeneous Markov chains is a non-trivial theoretical contribution for this problem class. The experiments include multiple datasets, comparisons with and without augmentation, runtime reports, and detailed hyperparameters, demonstrating excellent reproducibility.

### Weaknesses
The adaptive chain convergence relies on Assumption 4, but the paper only provides high-level justification. There is a lack of sufficient conditions linking Algorithm 1's specific step-size choices and gradient-variance conditions, creating a gap between theory and implementation. 
The empirical study is limited to Euclidean routing (TSP/CVRP), restricting the demonstration of generality. Experiments on non-Euclidean problems or other combinatorial domains would help establish the versatility of the latent formulation. 
Although negative gaps versus LKH3 on CVRP are reported, verification via exact solvers or optimality certificates is absent, making the evaluation protocol ambiguous.

### Questions
What is the sensitivity to the choice of latent dimension d_z and bounded diameter R? The ablations only vary K and update schedules, not d_z or R.

### Soundness
3

### Presentation
4

### Contribution
3
