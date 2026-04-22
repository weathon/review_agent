# Bayesian Combinatorial Lottery Ticket Machine: Bayes Meets Extremal Combinatorics

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 6, 2, 6

## Abstract
This paper proposes a fundamentally new framework for Bayesian machine learning (BML) by introducing redundant objects inspired by \emph{extremal combinatorics}.} In general, BML offers the advantage of quantifying uncertainty across all possible models (hypotheses) that explain data (observable phenomena in nature and society), making it particularly useful in applications that require transparency in data analysis, such as biomedical health and finance. However, its practical use is hindered by challenges in (1) model construction (often involving measure theory), (2) inference algorithm design (such as Markov chain Monte Carlo (MCMC) tailored to individual case studies), and (3) theoretical analysis (such as the mixing time of MCMC). This paper aims to mitigate the aforementioned obstacles, particularly for BML that infers combinatorial structures (permutations, partitions, binary sequences, binary trees, acyclic directed graphs, Cambrian trees, rectangular partitions, etc.) including consensus ranking, classification, factor analysis, hierarchical clustering, causal inference, phylogenetic analysis, and relational data analysis. Our key insight is to apply redundant universal objects, inspired by extremal combinatorics, as general-purpose generative probabilistic models independent of specific application scenarios. Our contributions include (1) a unified model construction method that expresses random target objects by randomly selecting a substructure from the universal objects, (2) a simple MCMC algorithm for Bayesian inference that is naturally derived from this model representation, and (3) a clue to the theoretical upper bound on the MCMC mixing time.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces a new approach for Bayesian inference with discrete structures. The authors propose to utilize ideas from extremal combinatorics for model construction, propose a generic MCMC inference algorithm and analyze convergence of this algorithm. Furthermore, the paper discusses the application of the framework to, among other problems, the Bayesian Traveling salesperson problem, consensus ranking, hierarchical clustering and relational data analysis.

### Strengths
The idea of the paper seems extremely interesting, yet well founded and very natural. I agree with the authors that their idea could be highly relevant for future applications as well as future research. 

The paper is methodologically **very** thorough and detailed and the conceptual and theoretical advancements are well supported by the discussed applications. Some experiments on real-world data confirm that the proposed method is sound and actually works in practice. 

The paper is also well written, very detailed and has nice supplementary material. I also found that the paper provides a nice introduction to the necessary concepts from extremal combinatorics

### Weaknesses
The paper (blissfully) claims to revolutionize all of Bayesian ML, while not even once mentioning that it only considers discrete structures. (This is not a big deal, but making this restriction very clear would certainly help readers from the generic Prob. ML direction)

While the contribution of the paper is indeed nice, I do have the feeling that selling it as a remedy to "all obstacles faced by current BML" is not supported by the evidence provided in the paper. To support such a claim, even more analyses and experiments would be needed. 

I see one big issues with the presented method: Prior specification seems to be a big issue because it needs to be done wrt. the super-structure which might not be a "natural" or easy-to-work with object for the relevant problem. However, priors are of course very important for Bayesian inference

Finally, given that the proposed method is quite different from a lot of other methods, I would very much like to see a more detailed discussion regarding (a) weaknesses and (b) potential future directions regarding applications and extensions.

### Questions
Could the authors please provide more reasons for this statement: "Our current standpoint is that we do not need much delicate control over a prior model, and that it is sufficient if the universality of the model is preserved."?

Please also feel free to respond to the weaknesses mentioned above.

### Soundness
4

### Presentation
4

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
The paper proposes a unifying Bayesian modeling and inference scheme—“Super Bayes (SB)”—that imports redundancy-centric ideas from the Lottery Ticket Hypothesis and extremal combinatorics into Bayesian machine learning. The core recipe is proposed as representing latent combinatorial objects (e.g., permutations, partitions, trees) by extracting a constrained subsequence/substructure from a “universal object” (e.g., super-permutation), enforced via a partition matroid. This yields a common target distribution over index sets, enabling a single, simple Gibbs-style MCMC (Algorithm 1) that swaps one index at a time. The also authors provide an explicit upper bound on the mixing time (Theorem 2.5), separating a “price of redundancy” term from a problem-dependent deviation-from-modularity term, and discuss a quantum-inspired acceleration (Algorithm 2). Empirically, the paper instantiates SB for several tasks, consensus ranking, hierarchical clustering, and relational data, showing results comparable or slightly better than bespoke baselines. The quantum variant (SB-q) often edges out the classical one.

### Strengths
Originality. The “universal object + partition matroid” formulation is a creative bridge between extremal combinatorics and Bayesian modeling. Framing many Bayesian combinatorial models as probability measures over matroid-constrained subsets yields a common inference mechanism (Algorithm 1). The mixing-time analysis provides transparency often missing in bespoke Bayesian samplers, with an interpretable decomposition into redundancy and likelihood-dependent pieces. The quantum-inspired parallelization, while exploratory, is a novel angle within Bayesian inference for combinatorial structures.

Quality. The SB reformulation is worked out concretely for permutations via a random permutation concatenation (RPC) super-permutation and a partition-matroid constraint ensuring subsequences are permutations. Equivalence to Bayesian TSP under a uniform prior is made explicit (Proposition 2.3). The mixing-time upper bound (Theorem 2.5) is nontrivial and helps pinpoint where model/likelihood design hurts or helps convergence. Multi-domain experiments (ranking, trees, rectangulations) show the unified sampler is competitive with tailored baselines, and SB-q often improves further.

Clarity. The pipeline (S1–S5) is clearly enumerated, and Figure 1 usefully situates universal objects, matroid constraints, substructures, and target structures. Algorithm 1 is simple and easy to implement; the narrative repeatedly connects back to the “family of subsets” view to orient the reader.

Significance. If robust, the approach could reduce engineering churn in Bayesian inference for new combinatorial models by reusing a single sampler and a common analysis template. The analysis highlights a path to principled mixing-time statements in practical Bayesian pipelines, which is a rare contribution in this space.

### Weaknesses
Prior control is indirect and limited. The framework “bakes in” prior mass via how often a target structure appears as a valid substructure of the universal object. Fine-grained priors are hard to encode, as acknowledged by the authors in Limitations. It may be helpful to provide recipes to shape priors beyond uniformity: e.g., weighted partition matroids, non-uniform RPC (with provable lack of bias), temperature-controlled counts, or auxiliary potentials over $S$ that map to interpretable priors over $T$.

Fixing a single RPC sample may introduce variance/bias. The paper argues that one RPC is “already highly redundant,” but this is just an assumption. For finite $n$, the realized coverage and multiplicities of target structures under a single draw could distort posterior estimates. The authors may consider adding an ablation comparing: (i) one RPC (fixed), (ii) multiple RPCs with Rao-Blackwellization over $\rho$, (iii) resampling RPC during burn-in. It is valuable to quantify effects on calibration, variance, and convergence.

Scalability and memory footprint is questionable. Time complexity $\mathcal{O}(n)$ per update is good, but space requirement $\mathcal{O}(n^2)$ (storing RPC and indices) may be a bottleneck for large $n$ or high-dimensional generalizations.

Theoretical dependence on $\zeta$-likelihood lacks concrete guidance. The bound depends exponentially on a deviation-from-modularity term. Readers need intuition for realistic magnitudes and how to design likelihoods with a small $\zeta$.

Quantum MCMC section is promising but underspecified. The SB-q description is brief; the performance lift is intriguing but the mechanism and costs (e.g., Trotter steps, coupling parameters) lack ablations. The authors may consider adding a controlled study varying the Trotter number, coupling strengths, and wall-clock vs. ESS/sec comparisons against SB-c and baselines. Clarify computational overhead and when SB-q is worth it.

Experimental scope is limited. Datasets are relatively modest. It’s unclear if gains persist at scale or under distributional shift. The stopping criterion “match empirical convergence of baselines” is subjective. The authors are suggested to report effective sample size (ESS), R-hat, and wall-clock/ESS to standardize comparisons, and to include larger datasets (e.g., $>50$ items in ranking; $>10k$ nodes in networks) and stress tests (noisy rankings and imbalanced matrices).

Generality beyond permutations requires more detail. While the paper outlines mappings $U2T$ from permutations to partitions/trees/rectangulations, the construction details (including ensuring lack of induced prior bias and tractable $\Pi$) deserve more explicit, worked examples. For at least one non-permutation target (e.g., binary trees), the authors are suggested to include a full construction of $\rho$, $C$, $U2T$, demonstrate uniformity/non-uniformity of the induced prior, and discuss computational/storage complexity.

### Questions
1.	Can you propose a principled way to incorporate informative priors (e.g., smooth permutations; sparse/balanced partitions) within the SB framework without losing the unified MCMC? A short addendum with a weighted-matroid or auxiliary-potential construction would be valuable.
2.	What is the empirical sensitivity to the particular RPC draw? Could you report variance across multiple $\rho$ samples and whether marginalizing over $\rho$ materially changes predictions or credible intervals?
3.	For each presented $\zeta$-likelihood, can you compute or bound $\zeta$ on the used datasets, and show how it correlates with observed mixing (R-hat/ESS)? This would strengthen the theoretical-empirical bridge.
4.	Please include ablations on the Suzuki–Trotter resolution, coupling parameters, and their effect on ESS/sec and solution quality. What is the additional compute cost vs. SB-c?
5.	Can you add experiments for (a) $n\ge 50$ in ranking with Mallows, (b) larger graphs/matrices in relational tasks, and (c) trees with >1k leaves, reporting memory and runtime?
6.	In Proposition 3.1, uniformity holds for TSP/permutations. For trees/rectangulations/partitions, what priors are induced by your concrete universal objects? Are there tasks where this induced prior is clearly undesirable?
7.	Could you standardize comparisons using ESS, R-hat, autocorrelation time, and wall-clock? “Match empirical convergence” is of qualitative nature and hard to reproduce.
8.	How does SB’s unified sampler compare to state-of-the-art black-box variational inference or generic MCMC within modern probabilistic programming languages on the same tasks (both accuracy and compute)?

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
4

### Summary
The paper proposes a framework for Bayesian machine learning that represents target combinatorial structures (permutations, trees, partitions, etc.) as random substructures of "universal objects" from extremal combinatorics (e.g., superpermutations). 
That is, instead of defining probability distributions directly over combinatorial objects (permutations, trees, partitions), they represent them as random subsequences of "universal objects". E.g., for permutations of length n, use a "superpermutation" of length n^2 that contains all permutations as subsequences. This converts the problem to sampling subsets of a large fixed structure with partition matroid constraints.
They provide a unified MCMC algorithm with mixing time bounds and demonstrate applications to consensus ranking, hierarchical clustering, and relational data analysis.

### Strengths
- The unified MCMC framework that applies across multiple combinatorial structures (permutations, trees, partitions, rectangulations) is *conceptually* elegant.
- The explicit mixing time upper bound for the proposed MCMC are informative and links a redundancy term and problem-dependent term.
- The paper includes experiments that span diverse applications (ranking, clustering, relational analysis). The authors show the method performs comparably to established baselines.

### Weaknesses
- The connection to the Lottery Ticket Hypothesis (LTH) is quite superficial and misleading. LTH concerns learned sparse subnetworks emerging from training, while this work simply uses pre-existing universal mathematical constructions. The analogy reduces to "universal objects are universal" and serves primarily as marketing rather than providing genuine conceptual insight or technical contribution to extremal combinatorics or LTH.
- While the authors show that one can do this, the claimed advantages are not convincingly demonstrated. 
- The O(n^2) memory usage appears highly significant for all but trivial problem sizes.
- Contributions to extremal combinatorics or LTH-related deep learning are lacking.
- Contribution to Bayesian ML is minor. The unified MCMC is nice in principle, but apparently paying O(n^2) memory cost for the redundancy is not. They also leverage existing theory on strongly Rayleigh measures to get mixing time.
- Experiments show only "comparable" or marginally better performance than baselines. 
- Despite the universality claims, The mixing time bound still contains an uncontrolled problem-dependent term that they can't control, undermining universality.
- The paper's scope and motivation are unclear. It attempts to cover an eclectic mix of topics, including extremal combinatorics, Bayesian modeling, Lottery Ticket Hypothesis, Traveling Salesperson, and quantum systems / computing without coherent motivation or obvious links. 
- Section 2.5 on quantum systems and quantum MCMC is marked "optional" yet it was included in the paper.
- The obstacles O1-O3 are stated at such a high level that it's impossible to verify whether they were actually overcome.
- The writing quality obscures the contributions. The abstract oversells the work with vague claims about "new breakthroughs." 
- The paper (specifically also title and abstract) are dramatically overselling the results.
- The introduction immediately dives into technical preliminaries without motivation. 
- Terms like "infinite description length" and statements like "we may need deep knowledge of measure-theoretic probability theory" use inflated language without clear definitions or justification for a broader ML audience.
- The experimental evaluation is weak. Results show the method is "comparable", not superior (hence no real breakthrough), raising questions about practical value given the added complexity and memory costs. 
- No ablation studies examine the impact of the redundancy or universal object.

### Questions
- Could you maybe introduce the motivation behind your work already early on? 
- Can you clarify O1-3 so that we clearly link your results to the obstacles?
- Could you discuss and expand on the problem-specific term in your bound?
- What evidence supports that the redundancy overhead is worthwhile given this limitation?
- Remove or significantly de-emphasize the LTH connection unless you can establish a deeper technical relationship. 
- I'd propose focusing the paper content a bit, so that your main message is clearer. Concepts like LTH and quantum systems make it harder to see a common thread.

### Soundness
3

### Presentation
1

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
This paper introduces a new formal framework for Bayesian Machine Learning (BML), balled Super Bayes (SB). SB allowsone  to express random target objects by randomly selecting a substructure from a universal object. This model representation naturally permits a simple MCMC algorithm for Bayesian inference, as well as a theoretical upper bound on the MCMC mixing time.

### Strengths
- The paper proposes a unified framework for BML than can accommodate a number of problems, while also coming with an MCMC algorithm for Bayesian inference as well as a theoretical upper bound on the MCMC mixing time.
- The paper looks technically sound and quantum MCMC seems an interesting contribution (even though I wasn't able to follow all details).
- The proposed SB framework can be readily applied to many BML Application cases, including mixture models, factor models, tree models, ancestral graphs, and relational models.
- The authors experimentally assess the framework on several important BML Applications, seemingly achieving higher performance than the competitors.

### Weaknesses
- To me, it is not entirely clear whether the proposed framework has additional benefits, besides introducing a unified formalism. The MCMC algorithm is standard, and my understanding is that the main benefit of this framework lies in the unified language for expressing various BML problems.
- Even though the SB framework can accommodate several BML applications of interest (e.g., mixture models, factor models, tree models, ancestral graphs, and relational models), it still relies on identifying the right universal object. The authors explained in detail how the SB framework covers the aforementioned scenarios, but there is no general formula. I may well be the case that some BML applications cannot fall under the SB framework.
- The paper is extremely verbose, and this can slow down the reading process. I also feel that some claims were potentially confusing. For example, I personally failed to see how the lucky ticket hypothesis is of any relevance to this framework. The lucky ticket hypothesis in deep learning is more about the fact that a big model with many redundancies can contain a good submodel within it; furthermore, it is precisely these redundancies that facilitate learning this submodel. In SB, the global object can obviously contain all possible sub-configurations, but redundancies are not what makes learning possible. Unless I misunderstood the connection the authors are making, I feel there is a confusion there about what the lucky ticket hypothesis is really about.
- The upper bound contains a problem-dependent term, which possibly renders it less useful.

### Questions
- What is the main contribution of this work? The introduction of a new formalism? The fact that this formalism permits MCMC and some theory on the MCMC mixing time? 
- What about BML applications not covered by the SB framework? Have the authors considered such applications?
- Is the upper bound useful in practice, given the problem-dependent term?

### Soundness
3

### Presentation
2

### Contribution
2
