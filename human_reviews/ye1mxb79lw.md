# BILBO: BILevel Bayesian Optimization

- Decision: Reject
- Scores: 6, 6, 6, 6, 3

## Abstract
Bilevel optimization, characterized by a two-level hierarchical optimization structure, is prevalent in real-world problems but poses significant challenges, especially in noisy, constrained, and derivative-free settings. To tackle these challenges, we present a novel algorithm for BILevel Bayesian Optimization (BILBO) that optimizes both upper- and lower-level problems jointly in a sample-efficient manner by using confidence bounds to construct trusted sets of feasible and lower-level optimal solutions. We show that sampling from our trusted sets guarantees points with instantaneous regret bounds. Moreover, BILBO selects only one function query per iteration, facilitating its use in decoupled settings where upper- and lower-level function evaluations may come from different simulators or experiments. We also show that this function query selection strategy leads to an instantaneous regret bound for the query point. The performance of BILBO is theoretically guaranteed with a sublinear regret bound and is empirically evaluated on several synthetic and real-world problems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the black-box bilevel optimization problem based on the Gaussian process (GP) model.
The author proposes an algorithm that leverages the confidence set of the underlying objective function $f$ under the Bayesian assumption (i.e., $f$ is the sample path of GP.)
The proposed algorithm achieves $O(\sqrt{\gamma_T T})$ cumulative regret upper bound whose 
instantaneous regrets are defined as the extension of existing work [1] for the formulation of the bilevel optimization. The numerical experiments are conducted, including two problems motivated by real-world problems.

[1] Nguyen, Quoc Phong, et al. "Optimistic Bayesian Optimization with Unknown Constraints." The Twelfth International Conference on Learning Representations. 2023.

### Strengths
- Overall, the writing of this paper is clear, well-structured, and easy to follow. 
- Comprehensive proofs are provided, and the proofs seem correct, as far as I can see.
- As far as I know, the proposed algorithm is the first theoretically guaranteed approach for bilevel Bayesian optimization. 
- The practical behavior of the algorithm is well-described in Figures 1 and 2, which will be helpful for practitioners.

### Weaknesses
My main concern is the lack of novelty. The confidence bound-based approach for 
complex structured optimization problems have already been extensively studied 
in the BO field (e.g., robust BO [1,2,3], constrained BO [4,5], and composite objectives [6, 7], etc.).
Specifically, it seems that the proposed algorithm can be constructed by combining the extended regret definition of [4] and the sampling for potential solutions based on a confidence set.  As far as I see, the non-trivial points of the proposed algorithm construction are the following:

1. The construction of $\mathcal{P}_t$ with $\overline{z}$ (Eq. 3.4, 3.5). Specifically, the natural construction of the potential solution set $\mathcal{P}\_t$ is based on maximum $\max l\_{f,t}(x, z)$; however, we cannot bound $r\_{f}$ with such construction.
2. Reassignment of $z_t$ based on Lemma 3.6.

I believe that the other parts of the algorithm construction and analysis are not novel for the reader who studies related BO fields (e.g., robust BO, constrained BO, etc.) since they only use the basic, well-known result of the existing algorithm and are naturally derived. 
Furthermore, uncertainty-based input selection by fixing one input, such as the procedures of point 2, is also commonly leveraged to upper bound the instantaneous regret in the BO field (e.g., [2,3]). 

For the above reasons, I slightly lean my score toward rejection.

[1] Bogunovic, Ilija, et al. "Adversarially robust optimization with Gaussian processes." Advances in neural information processing systems 31 (2018).

[2] Kirschner, Johannes, et al. "Distributionally robust Bayesian optimization." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

[3] Iwazaki, Shogo, Yu Inatsu, and Ichiro Takeuchi. "Mean-variance analysis in Bayesian optimization under uncertainty." International Conference on Artificial Intelligence and Statistics. PMLR, 2021.

[4] Nguyen, Quoc Phong, et al. "Optimistic Bayesian Optimization with Unknown Constraints." The Twelfth International Conference on Learning Representations. 2023.

[5] Xu, Wenjie, et al. "Constrained efficient global optimization of expensive black-box functions." International Conference on Machine Learning. PMLR, 2023.

[6] Li, Zihan, and Jonathan Scarlett. "Regret bounds for noise-free cascaded kernelized bandits." arXiv preprint arXiv:2211.05430 (2022).

[7] Xu, Wenjie, et al. "Bayesian optimization of expensive nested grey-box functions." arXiv preprint arXiv:2306.05150 (2023).

(Minor)
- I believe that there is further room to enhance the paper's quality. For example:
    - The existing work [1] studies the particular case of the problem setting of this paper. The comparison with [1] will make the position and novelty of this paper clearer.
    - Finiteness assumptions of $\mathcal{X}$ and $\mathcal{Z}$ should be explicitly described in Section 2.
    - The references should be appropriately capitalized (e.g., bayesian -> Bayesian).
    - The notation $|\mathcal{X}|$ usually denotes the cardinality of the set, not the dimension. I recommend using another notation.
    - Definition of the maximum information gain and its explicit upper bound for several commonly used kernels (SE or Mat\'ern) is beneficial for the reader who are not familiar with theory.
    - The vector graphics figures (such as .pdf, .svg) are favorable.
    - The result of Section.4 seems yet unstable in $5$ trials.
- (typo) Corollary 3.2: $[u\_{h,t}(x, z), l\_{h,t}(x, z)]$ -> $[l\_{h,t}(x, z), u\_{h,t}(x, z)]$
- (typo) Line 107: The definition of the posterior mean misses the prior mean $m_h(\cdot)$.
- (typo) Line 108: \sigma^2 I ->  \sigma^2 \bm{I}

### Questions
- Are there any problems with extending the proposed method for an infinite input set? I believe that we can extend it by relying on the discretization arguments (as in [1]) under the four-times differentiability assumption of the kernel and the compactness of the input set.
- Why does the author focus on the cumulative regret in Theorem 3.9? As with the existing works (e.g.,[2,3])
the upper bound of the simple regret (or the stopping time upper bound for finding $\epsilon$-accurate solution) seems to be derived based on the pessimistic estimated solution. I believe that simple regret is a more suitable performance measure from the optimization perspective.

[1] Srinivas, Niranjan, et al. "Gaussian process optimization in the bandit setting: No regret and experimental design." arXiv preprint arXiv:0912.3995 (2009).

[2] Bogunovic, Ilija, et al. "Adversarially robust optimization with Gaussian processes." Advances in neural information processing systems 31 (2018).

[3] Kirschner, Johannes, et al. "Distributionally robust Bayesian optimization." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

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
3

### Summary
This paper introduces a new algorithm designed to solve bi-level constrained Bayesian optimization problems. It demonstrates that the algorithm achieves a sublinear regret bound and provides experimental results on both synthetic and real-world datasets.

### Strengths
1. The problem addressed in this work is significant, and the proposed algorithm introduces a novel approach. Most existing bi-level Bayesian optimization algorithms tackle the problem using a nested framework, where each upper-level query requires separately optimizing the lower-level problem to convergence. In contrast, the proposed algorithm optimizes both levels concurrently, greatly enhancing sample efficiency, as shown in the experiments.
2. An infeasibility declaration method is also included.
3. Additionally, the algorithm achieves a theoretical sublinear regret bound, extending the classic bound from single-level to bi-level problems.

### Weaknesses
This approach involves keeping track of a set of nearly optimal solutions for the lower-level problem. In low-dimensional cases, this can be managed through discretization. However, it likely becomes difficult to scale effectively to even moderately high-dimensional problems.

### Questions
Have the authors considered a hybrid setting, where some parts of the bi-level problem are black-box while other parts are white-box with explicit expressions? I guess such problems also arise widely in practice.

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
This paper introduces BILBO, a novel bilevel Bayesian optimization framework that simultaneously optimizes functions at both upper and lower levels. The algorithm employs trusted sets based on confidence bounds at each level. The authors establish a sublinear cumulative regret bound in a decoupled setting and validate BILBO through experiments on simulated and real-world problems, using TrustedRand and Nested policies as baselines.

### Strengths
- The paper proposes a new algorithm, BILBO, specifically tailored for general bilevel optimization problems, where simultaneous optimization at both levels is essential.
- The authors derive and prove a sublinear cumulative regret bound for the decoupled setting, enhancing the theoretical understanding of bilevel optimization.
- Experiments are conducted on both synthetic and interesting real-world scenarios, demonstrating the practical applicability of BILBO.
- The writing is generally clear and well-structured, making the paper accessible to readers.

### Weaknesses
- While the algorithm is well-explained, the motivations for certain design choices, particularly the decoupled setting and the specific function query mechanisms ($\mathcal{F}$ and $h_t$), are not sufficiently discussed. A deeper exploration of these choices would help readers understand the broader implications and advantages of BILBO’s design.
- The paper’s discussion of related work is somewhat limited. A more detailed comparison with existing bilevel optimization methods, particularly those offering theoretical guarantees or practical performance benchmarks, would provide a clearer picture of BILBO’s relative strengths and weaknesses.
- The baselines used in the experiments—TrustedRandom and Nested—are not the strongest available. Including comparisons with state-of-the-art bilevel optimization algorithms, such as those from Kieffer et al. (2017) or Dogan & Prestwich (2023), would offer a more rigorous evaluation of BILBO’s performance.
- No code is provided alongside the paper. This omission hampers the reproducibility of the results and limits the ability of the research community to build upon or validate the findings presented.

### Questions
- Can you explain more about the motivations for studying the decoupled setting? And the reasons to consider $\mathcal{F}$ and function query $h_t$ is the algorithm design and regret definition? For example, can we only focus on the upper and lower-level functions in the regret definition? 
- For regret bound in Theorem 3.9, the authors discussed the relationship to constrained Bayesian optimization in Nguyen et al. (2023), is there any existing bilevel BO regret result that can be compared with? 
- In experiments, BILBO is compared with TrustedRandom and Nested. As described in Appendix C.1, the Nested baseline used a nested approach with SLSDP. But the related works mentioned, e.g.  Kieffer et al. (2017); Dogan & Prestwich (2023) are not included. Can you compare it with the associated works directly? If not, can you explain the reasons that BILBO cannot compared with the related work? 
- In Figure 1a, can you explain why BILBO has a sudden drop at around 150 queries? And what is the regret of BILBO after 150 queries (cannot observe the line afterwards)? The uncertainty level looks huge as well, I am wondering what would happen if you increase the number of runs (or change random seeds/initialisations, etc). 
- In experimental results, we observe Nested is generally similar to or worse than TrustedRand (which is only a random sampling), can you explain the possible reason? Does it suggest Nested is not a strong baseline (as it is from a previous work)?

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
4

### Summary
This submission proposes a novel algorithm for bilevel Bayesian optimization that optimizes both upper and lower level problems jointly in a sample-efficient manner. The key idea of this algorithm is using confidence bounds to construct trusted sets of feasible and lower level optimal solutions. Theoretical studies show that trusted sets guarantee points with instantaneous regret bounds and sublinear regret bound is proven. Empirical studies are also provided.

### Strengths
1.	Bilevel Bayesian optimization is an important problem with many real-world applications in machine learning, economics, energy management, and investment.
2.	The proposed algorithm also works for the decoupled setting where both level functions may come from different simulators or experiments.
3.	Both theoretical and empirical studies are provided. Real-world experiments are reported to positively show the effectiveness of proposed algorithm.
4.	Overall the writing is good and easy to follow.

### Weaknesses
1.	I have concerns on novelty of this submission since all techniques in this paper are pretty standard. First, functions in both levels are modelled by Gaussian process and then trusted regions are defined using confidence bounds. Then theoretical results are derived from Srinivas et al., 2010. I cannot find challenges that are unique to the bilevel Bayesian optimization, or I might have missed something. Please clarify this point.
2.	In Theorem 3.9, theoretical results are given based on $|\mathcal{F}|,|\mathcal{X}|,|\mathcal{Z}|$, so all of them must be finite, which put more restrictions on problem setting.

### Questions
Optimization from two constructed sets can be very hard; see Step 7 of Algorithm 1. How do you implement this step while maintaining two sets in Step 17 efficiently? I guess only approximated solutions can be given.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper investigates Bayesian Bilevel Optimization through a sampling-based approach. For general bilevel optimization problems, it provides a bound on the regret of the proposed algorithm and demonstrates its effectiveness across various synthetic and real-world problems.

### Strengths
- The writing is in general good and clear. I liked the motivation of Bilevel optimization. 

- The paper performed extensive experiments on various datasets including some real-world examples. This is clearly a strength, though this could have been much more convincing if the paper is positioned to claim contributions on the empirical side, rather than the theoretical side.

### Weaknesses
The paper's introduction is somewhat misleading. Unlike typical optimization studies focused on the computational complexity of local-search algorithms, this paper addresses statistical complexity, assuming exhaustive search over a function hypothesis space. Consequently, it diverges from the usual challenges associated with noisy, constrained, and derivative-free settings in continuous stochastic optimization, making direct comparisons with gradient-based methods inappropriate.

Below are a few detailed comments:

- Line 59-60: What is "information flow"? This is not a well-defined term. 

- Line 68-69: why functions are modelled using Gaussian Process? And this assumption appears abrupt. 

- Gaussian process: notation is messy and discussion is hard to follow.

- At the beginning of Section 3, it is not clear what you exactly get from querying points.

- Line 155 "trusted sets": in my first read, this was very confusing. I think the paper should have been much clearer about the interaction protocol, the fact that exhaustive search over hypothesis class is okay, and it is not going to be about convergence of local-search methods. 

- Line 301-302: What is "information gain"? Define it precisely.

While the extensive experiments in Section 4 are impressive, it remains unclear how to evaluate this section if the main contribution is claimed on methodological rather than real-world applicability.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
