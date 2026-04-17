# HATSolver: Learning Gröbner Bases with Hierarchical Attention Transformers

- Decision: Accept (Oral)
- Scores: 4, 6, 4

## Abstract
At NeurIPS 2024, Kera (2311.12904) introduced the use of transformers for computing Groebner bases, a central object in computer algebra with numerous practical applications. In this paper, we improve this approach by applying Hierarchical Attention Transformers (HATs) to solve systems of multivariate polynomial equations via Groebner bases computation. The HAT architecture incorporates a tree-structured inductive bias that enables the modeling of hierarchical relationships present in the data and thus achieves significant computational savings compared to conventional flat attention models. We generalize to arbitrary depths and include a detailed computational cost analysis. Combined with curriculum learning, our method solves instances that are much larger than those in Kera (2311.12904).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes HATSolver, that uses a transformer with hierarchical attention mechanism to capture the hierarchy of he natural tree structure of polynomial systems (tokens  $\rightarrow$ terms $\rightarrow$ polynomials $\rightarrow$ system). The model performs a bottom-up local self-attention followed by a top-down refinement via cross-level attention, and the authors provide a complexity analysis showing how hierarchical factorization reduces the quadratic attention bottleneck as lengths scale. The approach is trained on “backward-generated” datasets that pair non-Gröbner input systems with their reduced lexicographic Gröbner bases in shape position. The paper further adds a curriculum learning schedule to improve scalability. Empirically, the authors show that curriculum learning alone scales the baseline transformer beyond the n≤5 limitation, and demonstrate that their model converge faster and reach higher accuracy than the baseline under equal training budgets.

### Strengths
I think adapting hierarchical attention to polynomial systems is a thoughtful inductive bias. Moreover, the results are fairly explorative and include a larger-scale experiment at $n=13$ with a broad density sweep and baselines. On the clarity side, I can say that the two-phase hierarchical mechanism is described clearly with consistent notation. I also believe that scaling and even a partial success in neural Gröbner basis (GB) computation from n≤5 to n=13 is a meaningful step, especially since GB computation is a backbone in many symbolic tasks.

### Weaknesses
1. I think the actual success in this task is the exact string equality to the ground-truth reduced lex basis from backward generation. While reduced lex GB is unique, the paper does not report algebraic verification (e.g., reduce every ($f\in F$) by predicted ($\widehat{G}$), check ($\mathrm{lt}(\langle\widehat{G}\rangle)=\mathrm{lt}(I)$), minimality/monicity) as a post-hoc correctness test. The study grades ``exact string equality'', near-misses caused by non-monic scaling, ordering, or tails not fully reduced will be counted as failures, even if the ideal and leading-term ideal are correct. A CAS pass would quantify how many such string failures are algebraically valid solutions. Please add a CAS-based verifier in evaluation. 
2. I think the test set is somewhat biased. The scope is restricted to shape position, 0-dimensional radical ideals over ($\mathbb{F}_7$). That matches prior work, but the actual algorithmic alignment requires tests on other fields (incl. non-prime), other monomial orders, and non-shape zero-dimensional ideals. The paper acknowledges this as future work; however, even small OOD tests ($F(5)/F({11}$), random orderings, non-shape samples) would not only strengthen claims, but also would calibrate sensitivity to the data generator and support the beyond ($\mathbb{F}_7$) claim. 
3. Since PoSSo and Gröbner basis computation are NP-hard in the worst case, it would strengthen the paper’s soundness and positioning to discuss connections to Neural Algorithmic Reasoning as a neural alignment work (see below). What I really want to see is a comprehensive discussion on what algorithmic primitives (e.g., S-polynomial selection/reduction, Buchberger’s criteria, elimination structure) does your architecture implicitly learn, and how does the hierarchical attention help align with those primitives? A brief discussion would clarify whether HATSolver is merely pattern-matching the training distribution or actually capturing algorithmic invariants that might transfer OOD (e.g., to different fields/orders/densities). Relevant references are:

- Veličković et al. (ICLR 2020), Neural execution of graph algorithms.
- Georgiev et al. (LoG 2023), Neural algorithmic reasoning for combinatorial optimisation.
- Dudzik et al. (LoG 2024), Asynchronous algorithmic alignment with cocycles.
- Hashemi et al. (NeurIPS 2025), Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms.

### Questions
1. Questions discussed in Weaknesses.  
2. In Table 2, it’s unclear to me if those numbers are per-instance, per-batch, or total, and for how many instances. Please clarify them.
3. What pooling function you ultimately used (mean? learned? top-k?)
4. Table 4 reports observed max degree 11, while Table 6 lists “Max degree 20” for the training hyperparameters. Clarify the degree cutoffs effective during training vs. evaluation. 
5. Please add multiple seeds runs error bars to the reported numbers.
6. Since HATSolver-3 suffers from heavy padding, did you try bucketing by length to mitigate doubled token counts? If so, what was the effect on throughput/accuracy?

### Soundness
2

### Presentation
3

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
This study presents a new attention-based architecture for computing polynomial systems, particularly Gröbner bases. The prior study, Kera et al. (2024) used a standard architecture and suffered from prohibitively long input sequences and attention memory cost for large polynomial systems. The proposed study introduced a hierarchical attention mechanism that applies attention within terms, polynomials, and systems, accordingly. The experiments show a drastic training cost reduction and extension to systems with a larger number of variables than in the prior study.

### Strengths
- A clear focus on scaling up Transformer models for polynomial system computation. 
- New hierarchical attention module, which is particularly advantageous for the case where input sequence length surpasses the embedding dimension. 
- Experimental results show Transformer models with the proposed attention module scale up the problem size to the size where mathematical algorithms require long computation, which was not the case with the prior work.

### Weaknesses
I generally appreciate the achievement of this work - scaling up the problem size of learning-based Gröbner basis computation. The major weaknesses are two-fold: separation from prior studies and clarity/depth of experiments. I suppose that these are manageable weaknesses and expect the rebuttal process will resolve this. 

---
**Vs. Prior Studies** 

While the proposed concept is clear and aligns with processing polynomial systems, hierarchical attention mechanism has been used in a wide range of applications, as the authors summarize in the Related Work section. It is unclear from the authors' description whether the proposed module is technically novel. For example, MEGABYTE [1], SpaceByte [2], and Block Transformer [3] are general hierarchical attention models. None of them are cited in the paper. 

I'd like the authors to present the technical novelty of the proposed method over such works. I understand that the focus of this work is more specialized for polynomial systems. In this context, the proposed method can be regarded as a generalization of monomial embedding [4]. The main text should clarify this as well. 

[1] Yu et al., "MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers," NeurIPS'23

[2] Sagel, "SpaceByte: Towards deleting tokenization from large language modeling," NeurIPS'24

[3] Ho et al., "Block transformer: Global-to-local language modeling for fast inference," arXiv'24

[4] Kera & Pelleriti et al., "Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms," arXiv'24

---
**Unclear Experimental Setup**

The training setup is somewhat unclear to me. 
- In Section 4.1, why was the 4/4 model used rather than the 6/6 one as in the prior study? 
- According to the paper, the curriculum learning used 1M samples for each ($n$, $\rho$) combination. The range and step size of $\rho$ are not provided, but the number of training samples seems significant. 
- The caption of Table 1 mentions the out-of-distribution evaluations. However, no detailed description can be found. Why do colored texts suggest out-of-distribution? Is that because the corresponding values of $\rho$ were not used in training? Again, the range and step size of $\rho$ are not clarified, but I suspect the "out-of-distribution" $\rho$ here is still interpolation rather than extrapolation. 
- I feel the same metric concept is referred to by different names, which made me confused in reading. Specifically, what is the difference between "Accuracy" in Table 1, "Sequence Accuracy" in Figure 2, and "Success" in Table 2? 

---

**Experiment Depth**

I suggest a few more experiments that might strengthen the value of the proposed model further. 
- [4] proposes a new backward transform method that is more general than the one used in the experiments (adapted from Kera et al.'24). It would also be interesting to evaluate the impact on this setup. 
- The visualization of the embedding space of terms, polynomials, and systems would be valuable. For example, are "similar" systems (e.g., having the same leading term set, or Gröbner basis) mapped close together? 
- The experiments target polynomial system solving, so the experiments only tested HAT-2 and HAT-3. Setting up an artificial problem/dataset and testing HAT-n with n > 3 could demonstrate the broader utility of this work.  

---

I feel the potential practical impact of this work is great, but the machine learning (and algebraic) technical novelty seems limited. My score is temporal, and if the authors can address particularly the first concern (**Vs. Prior Studies**), I'm happy to increase the score.

### Questions
Besides the comments in the Weaknesses, I'd like the authors to clarify several points. 
- **Dimension design.** [l.263-265] suggests several designs of $(d_i)_i$. Are there any ablation studies and insights into this point?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on the recent line of work on learning to compute Gröbner bases with Transformers and proposes to replace flat attention with a hierarchical attention transformer (HAT) that is tailored to the tree shape structure of multivariate polynomial equations. The starting point is the limitation in Kera et al. that the usual Transformers could handle only up to about 5 variables since, in part, the tokenized sequences become extremely long and flat attention is quadratic in sequence length.  The authors propose a HAT layer that is based on hierarchal attention, similar in spirit to U-net based architectures.  The idea is a 2 phase computation where first attention is computed at each layer, pooling to reduce the dimensionality, and in the second phase, information is propagated in a top-down manner via cross addition or residual adds.

### Strengths
The motivation compared to Kera et al. is well stated.  The example given to illustrate how their hierarchical model works e.g., in fig 1 is pretty clear and the authors convincingly show that the hierarchical formulation is natural for this data.  They do go beyond the earlier baseline quoted from Kera et al. They claim to reach 13 variables, degree 11, whereas Kera et al. stopped earlier. If those numbers are correct and all other factors are equal, then this is a sizable improvement.  

At the same time I should note that I am not a domain expert in learned Gröbner basis computation, so I cannot fully judge how substantial this improvement is from the perspective of the Gröbner basis community, or whether this is mainly a better serialization and positional scheme applied to the same underlying idea.

### Weaknesses
The main part of the argument in Section 3 of this paper is that the problem is really the explosion in sequence length and that we should reflect the inherent tree structure in the model. This is reasonable, but it is also very close to what people have been doing with structured tokenization and with position functions that respect the underlying structure in the data, for example with hard ALiBi-like schemes for organized inputs (see eg [1]). The paper does not make it clear why this must be a new hierarchical attention block rather than a careful tokenization and positional encoding strategy that bakes in this inductive bias from the outset.

Second, I am not clear about which part is doing the heavy lifting for the improved performance over Kera et al.  As far as I can tell, there isn't any ablations reported in the paper of whether the performance boost is due to the hierarchical attention, more careful curriculum, different serialization, or training the model longer.

Given how hierarchical the data is, I would have expected at least one baseline that uses grouping of tokens into fixed length blocks with attention only inside blocks plus a cross block layer. Right now the paper is arguing that the custom HAT is needed without showing that a simpler approach fails. This makes the contribution somewhat difficult to judge.  However, I am willing to defer to other reviewers on this point.

Finally, some of the writing is quite informal, particularly in the introduction (such as the first paragraph).  I did not penalize the authors for this.

[1] https://arxiv.org/abs/2402.01032

### Questions
(See above)

### Soundness
3

### Presentation
2

### Contribution
2
