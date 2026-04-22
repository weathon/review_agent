# Contextual Causal Bayesian Optimisation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
We introduce a unified framework 
for contextual and causal Bayesian optimisation, 
which aims to design intervention policies 
maximising the expectation of a target variable. 
Our approach leverages both observed contextual 
information and known causal graph structures 
to guide the search. Within this framework, we 
propose a novel algorithm that jointly optimises 
over policies and the sets of variables on which 
these policies are defined. This thereby extends and 
unifies two previously distinct approaches: 
Causal Bayesian Optimisation and Contextual 
Bayesian Optimisation, while also addressing 
their limitations in scenarios that yield 
suboptimal results. We derive worst-case and 
instance-dependent high-probability regret bounds 
for our algorithm. We report experimental results 
across diverse environments, corroborating that 
our approach achieves sublinear regret and reduces 
sample complexity in high-dimensional settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors argue that existing Causal Bayesian Optimization and Contextual Bayesian Optimization are not suitable for problems where both causal and contextual information is present. They propose a method which combines those two methods in order to construct a method which can handle such problems.

### Strengths
- The paper illustrates well (albeit with a toy example) that using only context or only causal information can lead to unnecessarily large regrets.
- Despite "simply" combining Causal BO and Contextual BO, this blend is non-trivial enough for the contributions of this paper to be relevant.
- Both experimental and theoretical evidence is presented in defense of the proposed method.

### Weaknesses
- The introduction does not motivate the problem (where does it emerge? Any specific cases?), and the first paragraph seems disconnected from the rest, mentioning POMISs, which are not used anywhere in the text.
- Not enough is said about CaBO and CoBO for context, and for comparison with the method proposed in this paper. It should be very clear where exactly this method differs from those two.
- It is not explained why the problem/task of the agent (lines 120-127) is relevant to study in the first place, even in principle. This is my most serious problem with this paper. See questions 1-3 below about this.

### Questions
1. Can the authors provide an example of a situation modeled by the mathematical problem described in the box at lines 120-128?
2. Why would the practitioner/agent not use all of the context it has access to? Why should it choose any scope other than the maximal (and still informative) one at every step? Is it because mixed policies would be harder to learn or is there something else to it?
3. The past observations (line 122) do not include the elements of $\mathbf{v}_{l}$ not in $\mathbf{c}_{l}$. Is this sensible? I understand that the mixed policy only makes use of the $\mathbf{c}_{l}$, but why not make use of the observed $\mathbf{v}_{l} \setminus \mathbf{c}_{l}$ in future steps?
4. Your definition of POMP drops the requirement of them being NROs from (Lee & Barenboim 2020). What is the justification for this? This should be justified in the paper.
5. In line 133, why $\mu^{*}_{S} > \mu^{*}_{S'}$ instead of $\mu^{*}_{S} \ge \mu^{*}_{S'}$? As it stands, won't we eliminate MPS $S$ that are indeed optimal for some SCM, but it just so happens that there is another MPS $S'$ which leads to the same average reward? It seems to me that both $S$ and $S'$ should be taken as POMPS in that case, not both eliminated.
6. In line 138 you state that the regret would increase linearly. Why?
7. Isn't the MAB too "complex" to actually converge to something useful? I'll explain: the MAB has POMPS as arms, but for each POMP $\mathcal{S}$ that it may choose, there are multiple (possibly uncountable) policies $\pi_{\mathcal{S}}$ that can be chosen; hence the $\mu_{S}$ that the MAB is trying to maximize is an expected value over a complicated distribution combining the different $\pi_{\mathcal{S}}$ and the probabilities of each of them being chosen, which heavily depends on the details of the chosen BO method. Another way of viewing this, is that each arm of the MAB does not simply have one reward distribution, but multiple, and the choice of which to sample from is done by the BO. Could you explain why this is not an issue?
8. Related to the previous question: from what I gather, what we want to choose is not the best $\mathcal{S}$, but the best pair $(\mathcal{S}^{*}, \pi_{\mathbf{S}}^{*})$. It seems to me that it could easily happen that the best such pair would not be found by the proposed method. It may happen that $A$ suggests $\mathcal{S}^{*}$ a few times but a very bad $\pi_{\mathcal{S}}$ is suggested by the BO, and the MAB "concludes" that $\mu_{\mathcal{S}^{*}}$ is low and starts picking other MPSs. Thus, I do not see the algorithm being trustworthy for reasonable (i.e. non-astronomical) numbers of $T$. Could you explain why this is not an issue?
9. In the paragraph starting at line 237, the authors justified why CaBO's selection method is unsuitable for the case with context. But why is CoCa-BO's method suitable? This is not explained in the main text/in a high-level way. Could you also include a discussion of your method and why it is suitable in the main text?

### Other comments

- It should be clearer that the definition of mixed policy scope comes from (Lee & Bareinboim 2020), for example by citing in the definition's title. Same comment holds for other concepts taken from that or other sources.
- Re-defining $G_S$ in the first footnote can cause confusion. I would stick would the formally correct definition only.
- "the nature generates..." (line 123/124) is strangely phrased.
- typo line 132: ...*is* a possibly-optimal...
- In line 136, you retroactively changed the protocol in the box. This can make it confusing to understand the exact problem this paper is trying to solve.
- typo line 919: equivalence
- Defining the MAB as a mapping seems to hide the actual complexity of that step. I would suggest explaining in detail what is the MAB problem here (see also question above).
- line 209: I do not see how the details of the MAB update step are provided by Algorithm 2. Is this a typo?

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
4

### Summary
The paper proposes CoCa-BO, a framework that unifies causal and contextual Bayesian optimisation by integrating the concept of Mixed Policy Scopes (MPSs) from causal inference with Bayesian optimisation (BO) methods. The algorithm operates in two layers: (i) a multi-armed bandit (MAB) mechanism that adaptively selects among possibly-optimal mixed policy scopes (POMPSs), and (ii) a Gaussian-process-based BO routine that optimises interventions within each chosen scope. Theoretical analysis establishes high-probability sublinear regret bounds for this hierarchical process. Empirical studies on synthetic causal models demonstrate that CoCa-BO avoids linear regret suffered by standard CaBO and CoBO baselines in regimes where either causality or context alone is insufficient.
Strengths

### Strengths
Summary
The paper proposes CoCa-BO, a framework that unifies causal and contextual Bayesian optimisation by integrating the concept of Mixed Policy Scopes (MPSs) from causal inference with Bayesian optimisation (BO) methods. The algorithm operates in two layers: (i) a multi-armed bandit (MAB) mechanism that adaptively selects among possibly-optimal mixed policy scopes (POMPSs), and (ii) a Gaussian-process-based BO routine that optimises interventions within each chosen scope. Theoretical analysis establishes high-probability sublinear regret bounds for this hierarchical process. Empirical studies on synthetic causal models demonstrate that CoCa-BO avoids linear regret suffered by standard CaBO and CoBO baselines in regimes where either causality or context alone is insufficient.

### Weaknesses
Weaknesses
Limited empirical validation

Experiments are mostly synthetic or semi-realistic; no real-world dataset or ablation studies.

The runtime overhead of managing multiple scopes is not quantified.

Independence across scopes

Each POMPS maintains a separate GP, which may underutilise shared structure among overlapping interventions.

Discussion on potential information sharing (e.g., multitask GPs or shared hyperpriors) would be valuable.

Algorithmic detail gaps

Algorithm 1 is schematic; explicit update formulas for Opts.update() and A.update() are missing.

Schedules for \beta_t,\rho_t and constants in the bounds are not provided, hindering reproducibility.

Clarity
Generally clear and well-written, though dense. A figure illustrating the two-level UCB mechanism (scope-level vs. within-scope) would enhance intuition. Captions could be more self-contained. Also the authors could reintroduce POMPS and MPS for better readability. 
Relation to Prior Work
Missing citations include:

Aglietti et al. (2020) Multi-task Causal Learning with Gaussian Processes (very relevant in my opinion)
Lattimore et al. (2016) Causal Bandits: Learning good interventions via Causal Inference

Chowdhury & Gopalan (2017)  On Kernelised multi armed bandits 

Adding these would help situate the work fully.

### Questions
see above

### Soundness
3

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
3

### Summary
The authors introduce a unified framework for contextual and causal Bayesian optimization (CoCa-BO) that aims to design intervention policies maximizing the expected value of a target variable. The proposed approach integrates both observed contextual information and known causal graph structures to effectively guide the search process. Theoretical contributions include worst-case and instance-dependent high-probability regret bounds for the algorithm. Empirical evaluations across diverse environments further demonstrate that the proposed approach achieves sublinear regret and significantly reduces sample complexity in high-dimensional settings.

### Strengths
The paper presents an extensive theoretical analysis of the proposed method. In addition, the authors use real-world examples to illustrate and support the proposed method.

### Weaknesses
Although the paper provides thorough theoretical results on the regret bounds, several important aspects require further clarification and experimental validation:

(1) Missing Baselines.

The experimental evaluation lacks comparisons with several relevant baselines, such as existing causal Bayesian optimization (CaBO) methods — e.g., CBO (Aglietti et al., 2020) and MCBO (Sussex et al., 2023). Even though the proposed framework targets a specific problem setting, existing CBO methods could potentially be applied and should therefore be included for a fair comparison. Moreover, standard Bayesian optimization (BO) baselines such as UCB-based methods should be evaluated as well, since in some scenarios UCB can outperform CBO approaches.

(2) Limited Discussion of Related Work.

The paper should provide a more comprehensive discussion of prior work on Contextual Bayesian Optimization (CoBO) and include corresponding experimental comparisons. While the authors compare CoCa-BO against CoBO, they do not discuss other existing CoBO methods in depth, leaving the relationship between these approaches unclear.

(3) Unclear Distinction from Existing CBO Frameworks.


The conceptual and methodological differences between existing CaBO and the proposed CoCa-BO remain insufficiently discussed. If existing CaBO methods are capable of incorporating contextual variables, the motivation for introducing CoBO and CoCa-BO becomes less convincing. For instance, in the illustrative example shown in Figure 1, the context variable C could also be integrated within most CaBO frameworks. 
To strengthen the paper’s contribution, the authors should include additional experiments comparing CoCa-BO with CaBO models that explicitly incorporate contextual variables.

### Questions
(a) The advantages  of CoCa-BO over CaBO with contextual modeling remains unclear. The authors should provide additional experimental evidence to explicitly demonstrate the advantages of CoCa-BO over CaBO when context variables are considered.

(b) The experimental evaluation would benefit from including more baselines and diverse datasets to better substantiate the generalizability of the proposed method.

(c) Minor issue: the spelling should be standardized to “Bayesian optimization” (instead of optimisation).

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
The paper introduces CoCa-BO, a unified framework that optimises intervention policies while jointly selecting the policy scope (which variables to intervene on and which contexts to condition on) using causal structure and observed context. It generalises and connects Causal BO (CaBO), which uses a known DAG but ignores context, Contextual BO (CoBO) which uses context but fixes scope showing that either alone can suffer linear regret in plausible settings. CoCa-BO treats each possibly-optimal mixed policy scope (POMPS) as an arm in a bandit over scopes, and within a chosen scope runs contextual GP-BO (HEBO) to select the intervention values. The authors provide worst-case and instance-dependent high-probability regret bounds and experiments demonstrating sublinear regret and lower sample complexity in high-dimensional problems.

### Strengths
- The paper presents a clean framework that truly unifies causal scope selection with contextual action choice and identifies precise failure modes of CaBO/CoBO. 
- The authors also provide sound theory with practical knobs. That is, there are clear regret guarantees with interpretable dependence on info gain; the approach also uses well-known GP-UCB machinery. 
- Scope-as-arms with HEBO inside each arm is sensible and parallelisable; 
- The paper is careful about the acquisition mismatch in contextual settings. 
- There are also compelling empirical examples which show when each baseline fails and how CoCa-BO succeeds;  The work also includes a large dimensional environment in A.3 to stress sample efficiency.

### Weaknesses
- The approach requires known causal structure; while standard in CaBO, many real tasks need robustness to graph misspecification or partial knowledge. (They defer discovery to future work.)
- Computing the POMPS set can be exponential in |V| (albeit parallelisable). For very large graphs, this may become the bottleneck.
-The MAB over scopes uses historical acquisition values; bandit feedback is influenced by the within-scope BO’s learning speed and noise, which may cause slow scope identification when arms differ mainly via context distributions.
- While reasonable, results depend on a single contextual BO backend; it would strengthen claims to show backend-agnostic performance (e.g., GPflow/BoTorch comparisons)
- Robustness is explored but limited; at very high noise or with heavy-tailed outcomes, the GP assumptions might need adaptation.
- Most importantly, the approach  treats context as an observable input variable but does not explicitly model how context affects the causal mechanisms or intervention effects (e.g., through structural equations). This limits causal interpretability since the GP simply learns correlations between c_t and outcomes rather than a causal modulation of intervention efficiency.
- Each POMPS scope maintains its own contextual GP, trained on samples observed when that scope was active. This means data from different scopes, which might share the same contexts, is not shared across arms. In practice, this can lead to data inefficiency: if two scopes operate under overlapping context distributions, each must relearn context-outcome relationships independently.

### Questions
1. In large-dimensional settings, how does the model avoid the curse of dimensionality when embedding context directly into the GP input? Were any dimensionality-reduction or feature-selection methods considered?

2. For graphs with many nodes, enumerating POMPS can be exponential. Could you discuss heuristic or approximate methods for pruning scopes without losing theoretical guarantees?

3. Can you show a task where the same interventions yield different optima under distinct contexts, demonstrating that context truly modulates causal effects?

4. Have you compared against meta-learning or multi-task BO baselines that can share context information across tasks, to isolate the benefit of causal reasoning?

5. What happens if you remove context from the GP input but keep the same scope structure? Does performance drop substantially, confirming that context contributes non-trivially?

### Soundness
3

### Presentation
3

### Contribution
2
