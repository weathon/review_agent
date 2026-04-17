# Geometric-Mean Policy Optimization

- Decision: Accept (Poster)
- Scores: 2, 8, 6, 4

## Abstract
Group Relative Policy Optimization (GRPO) has significantly enhanced the reasoning capability of large language models by optimizing the arithmetic mean of token-level rewards. Unfortunately, GRPO is observed to suffer from unstable policy updates when facing tokens with outlier importance-weighted rewards, which manifest as extreme importance sampling ratios during training. In this study, we propose Geometric-Mean Policy Optimization (GMPO), with the aim to improve the stability of GRPO through suppressing token reward outliers. GMPO is plug-and-play—simply replacing GRPO's arithmetic mean with the geometric mean of token-level rewards, as the latter is inherently less sensitive to outliers. GMPO is theoretically plausible—analysis reveals that both GMPO and GRPO are weighted forms of the policy gradient while the former enjoys more stable weights, which consequently benefits policy optimization and performance. Experiments on multiple mathematical reasoning benchmarks show that \Ours-7B improves the average Pass@1 of GRPO by up to 4.1%, outperforming many state-of-the-art approaches. Code is available at https://github.com/callsys/GMPO and verl.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Geometric-Mean Policy Optimization (GMPO), a simple and effective variant of GRPO for stabilizing RL-based fine-tuning of LLMs. While GRPO maximizes the arithmetic mean of token-level importance-weighted rewards, GMPO instead maximizes its geometric mean, which is theoretically less sensitive to outliers.

The paper provides:

- A clear analysis showing that GMPO has a narrower objective range and smaller variance in importance sampling ratios.
- A gradient-based justification showing geometric averaging provides smoother updates.
- Extensive experiments on reasoning benchmarks (AIME24, AMC, MATH500, Minerva, OlympiadBench, Geometry3K, and MoE settings).
GMPO improves average Pass@1 by 4.1% on GRPO-7B and shows improved entropy stability and lower KL divergence during training.

The paper is clearly written and supported by sufficient experiments. However, I have several reservations about its theoretical depth and novelty (see the weaknesses section). In particular, the connection between the proposed theory and the empirical behavior of the model remains somewhat unclear. Additionally, the idea is closely related to **Group Sequence Policy Optimization (GSPO)**, which seems to share a similar motivation and formulation. Depending on how ICLR treats concurrent submissions and the arXiv paper, this overlap may further reduce the perceived novelty. Given these concerns, I lean toward a negative recommendation at this stage.

### Strengths
1. **Simple but intuitive idea.**
    
    Replacing arithmetic with geometric mean provides a clear and theoretically motivated way to suppress reward outliers, which is an elegant modification of GRPO with minimal implementation overhead.
    
2. **Extensive experiments.**
    
    Results across multiple reasoning and multimodal benchmarks (including MoE settings) are consistent and demonstrate improvement without introducing extra hyperparameters or architectural changes
    
3. **Comprehensive ablation.**
    
    The paper carefully studies token-level vs. sequence-level clipping, clipping range, normalization, and entropy behavior, which strengthens empirical credibility.
    
4. **Readable and well-structured.**
    
    The presentation is clear, the pseudo-code and figures are well-designed, and the comparisons with Dr.GRPO, DAPO, and GRPO variants are complete.

### Weaknesses
1. **Limited conceptual novelty.**
    
    While the geometric mean idea is intuitive and effective, it is arguably a *minor variation* of GRPO rather than a fundamentally new algorithmic contribution. The theoretical part mainly formalizes this intuition but lacks deeper insight into the connection to existing stability mechanisms (e.g., adaptive learning rate, variance regularization). The existence of GSPO further weakens the novelty of this paper.
    
2. **The analysis is not quite clear to some extent.** The main results of analyzing the gradient space of GMPO and GRPO lie in equations 5 and 6, where we see that the main difference is the reweighting coefficient of the gradient to log-pi. Then, how do the results relate to the model’s improved behaviors in:

    - The more stabilized dynamics are shown in Figure 3. One potential explanation is the insensitivity to the outliers of the importance sampling ratio; then, would vanilla GRPO+filter on this ratio solve the problem? If not, where are the benefits of applying the geometric mean here?

    - Figure 4 provides many good signs that GMPO indeed stabilizes the model’s training. Could the authors provide more analysis on why GMPO can make them happen, e.g., why the entropy is not decaying, and why the gradient norm is more stable? I believe that only claiming to the insensitive to the outlier is not enough.

### Questions
1. Importance sampling, not important sampling.

2. This is a major concern. In Section 3 (line 262), the authors state that GMPO performs clipping at the token level, which is different from GRPO. However, when comparing Equations (4) and (1), the clipping operations appear to be applied inside the summation (or product) over the token index t. Furthermore, by examining the gradients in Equations (5) and (6), we can see that GRPO’s rho depends explicitly on t, whereas GMPO’s weight is normalized across all t.
    
    In other words, GRPO effectively operates in a token-level manner, while GMPO behaves more like a sequence-level method, which contradicts the authors’ claim that GMPO’s token-level behavior.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes Geometric-Mean Policy Optimization (GMPO), a modification of Group Relative Policy Optimization (GRPO) designed to improve training stability for large language models. The core issue identified with GRPO is its use of the arithmetic mean of token-level rewards, which makes it sensitive to outliers in importance sampling (IS) ratios and leads to unstable policy updates. GMPO addresses this by replacing the arithmetic mean with the geometric mean, which is inherently more robust to outliers. The authors provide theoretical analysis showing that the GMPO objective has a narrower value range and results in more balanced gradient updates. Empirically, through experiments on several mathematical and multimodal reasoning benchmarks, the paper demonstrates that GMPO consistently outperforms GRPO, achieving significant performance gains (e.g., an average Pass@1 improvement of up to 4.1%). The paper includes extensive ablation studies that validate the key design choices of GMPO, such as token-level clipping and the normalization factor in the geometric mean.

### Strengths
1.  The core idea of replacing the arithmetic mean with the geometric mean is simple, elegant, and well-motivated. It directly targets a plausible source of instability in GRPO (sensitivity to outlier rewards) with a classic statistical tool known for its robustness. The "plug-and-play" nature of the modification makes it highly practical.
2.  The empirical evaluation is comprehensive and convincing. The authors demonstrate consistent improvements over GRPO across multiple model sizes (1.5B, 7B, 32B MoE), different model architectures (dense and MoE), and different domains (language-only and multimodal reasoning). The performance gains are substantial and statistically significant.
3.  The ablation studies presented in Section 4.3 are excellent. They systematically dissect the contributions of each component of the proposed method: the geometric mean itself, the token-level clipping strategy, the clipping range, and the normalization term. This analysis provides strong evidence for the authors' design choices and adds depth to the paper's claims.
4.  The paper is generally well-written and clearly structured. The motivation for the work is established effectively in the introduction, and the figures, particularly Figure 1 and Figure 4, provide strong visual support for the claims regarding improved stability (tighter IS ratio range, higher entropy, lower KL divergence) and its connection to final performance.

### Weaknesses
1.  The manuscript has numerous small but distracting typographical and formatting errors.
    *   The text is missing a character at the beginning of lines 190 and 202.
    *   The formulas in the derivation on lines 182-189 lack eq number.
    *   Several plots in Figure 4 are missing x-axis labels (e.g., plots c, e, g), which should be labeled "Training steps" for clarity.
2.  The related work section, while comprehensive, reads like a long list of recent methods without sufficient structure or synthesis. It does not adequately position GMPO with respect to other methods that also aim to improve stability, such as those using adaptive reward normalization (BNPO) or variance reduction techniques (OPO). The paper would be stronger if it discussed the conceptual differences and potential synergies with these alternative approaches, rather than solely focusing on the direct comparison with GRPO.

### Questions
1.  In line 266, it says "clipping at the token-level, as shown in Eq 3". I didn't see the clipping operation in formula 3; should this refer to Eq 4?
2. What is the intended value of $sgn(\hat A_i)$ when $\hat A_i=0$? The current code treats it as negative; the math does not specify it. This edge case can occur because the group based advantage in GRPO is standardized. Please clarify. 
3. Can you report GRPO with the exact same per token clipping in log space and the same wide range $(e^{-0.4},e^{0.4})$ so that the only difference is arithmetic vs geometric aggregation? That would isolate the effect of the geometric mean from the effect of wider exploration.
4. In Table 3 you compare against several external methods such as PRIME Zero and OpenReasoner Zero. Were those numbers reproduced under your pipeline or copied from their papers which might have used different data or evaluation protocols?

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
4

### Summary
The paper proposed GMPO, which replaces the arithmetic mean in GRPO with the geometric mean to achieve better  stability of training.

### Strengths
The method is easy to follow, and the paper writing is clear.

### Weaknesses
* Judging from Table 4, it seems that the proposed method's gain is quite marginal. The token-wise clip also seems not so effective.
* The major technical change from GRPO seems to be token-level clipping + geometric mean vs group arithmetic mean. The technical novelty is low.
* Although discussion is present, comparisons is missing against the method that controls training stability by carefully setting clipping ratio (e.g. DAPO)

### Questions
See weakness

### Soundness
3

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
This paper addresses the critical challenge of training instability in GRPO. The authors identify the root cause of this instability as the GRPO objective's reliance on the arithmetic mean of token-level rewards, which is highly sensitive to outlier importance sampling ratios. To remedy this, the paper proposes Geometric-Mean Policy Optimization (GMPO), which replaces the arithmetic mean with the geometric mean. The central hypothesis is that the geometric mean, being inherently more robust to extreme values, will effectively dampen the influence of outlier tokens.

### Strengths
- The problem is clearly defined and motivated. The authors narrow their focus to a specific, plausible mechanism: the sensitivity of the arithmetic mean aggregator in the GRPO objective to these outlier IS ratios. 

- The method is practical and easy to implement. The loss is a small rewrite of GRPO. The authors enhance this practicality by providing clear pseudo-code in Algorithm 1, which details the implementation in log-space for numerical stability.1 This transparency is crucial for reproducibility and allows the community to build upon the work easily.

- The paper substantiates its claims with extensive empirical evidence across a variety of benchmarks and model scales.

### Weaknesses
- The core novelty of the paper lies in the application of the geometric mean to the GRPO objective. While this application is new in this specific context, the underlying idea of using a robust statistical estimator to handle outliers is a foundational concept in statistics and data analysis. This weakness is compounded by a failure to justify the choice of the geometric mean over other standard robust estimators that designed to be less sensitive to outliers and could plausibly offer similar or even superior stabilization benefits.

- The theoretical analysis presented in Section 3 and Appendix A does not provide formal convergence guarantees, nor does it connect the method to more rigorous optimization frameworks that are commonly used to analyze the stability of policy gradient methods.

- A key design choice highlighted in the paper is the use of clipping range for the importance sampling ratio, which is substantially larger than the typical range used in GRPO. The authors justify this choice empirically, showing in their ablation study (Table 5) that this specific range yields the best performance among the options tested. However, the paper fails to provide a deeper explanation for why this is the case. 

- The evaluation can be further enhanced. There are several alternatives to improve GRPO stability, such as GSPO, DAPO, or Dr.GRPO. Given how close GMPO is to sequence-ratio weighting and clip-widening, this is critical. Also, the evaluation is mostly math. It will be more convincing if the authors can demonstrate code tasks, tool-use, or larger open-ended reasoning. The improvement seems marginal.

### Questions
- Why does the geometric mean objective enable a wider clipping range to be effective without causing instability? Is there a principled way to determine the optimal clipping range in conjunction with GMPO, or is it simply another sensitive hyperparameter that requires careful tuning? 

- Have the authors analyzed the computational and memory overhead of this operation compared to the simple summation in GRPO, particularly for very long reasoning chains (e.g., >2000 tokens)? Is there a point at which this overhead becomes a significant practical concern? A complete analysis would include profiling the computational time and memory usage of the GMPO loss function versus the GRPO loss function for varying sequence lengths to quantify any potential overhead.

### Soundness
2

### Presentation
3

### Contribution
2
