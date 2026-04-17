# Adapting, Fast and Slow: Transportable Circuits for Few-Shot Learning

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Generalization across the domains is not possible without asserting a structure that constrains the unseen target domain w.r.t. the source domain. Building on causal transportability theory, we design an algorithm for zero-shot compositional generalization which relies on access to qualitative domain knowledge in form of a causal graph for intra-domain structure and discrepancies oracle for inter-domain mechanism sharing. Circuit-TR learns a collection of modules (i.e., local predictors) from the source data, and transport/compose them to obtain a circuit for prediction in the target domain if the causal structure licenses. Furthermore, circuit transportability enables us to design a supervised domain adaptation scheme that operates without access to an explicit causal structure, and instead uses limited target data. Our theoretical results characterize classes of few-shot learnable tasks in terms of graphical circuit transportability criteria, and connects few-shot generalizability with the established notion of circuit size complexity; controlled simulations corroborate our theoretical results.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the problem of domain adaptation (and domain generalization) from a perspective of causal structure transfer. Specifically, there are three main parts:
1. Transferring statistical information to a target domain when the causal structure of all domains is fully known and the target domain has a similar form to one of the source domains
2. Transferring information to a target domain when the target function can be constructed by composing the source domain functions, and details of each domain and their relationships are known
3. Using a small number of samples in the target domain to conduct an exhaustive search over a space of causal structures in order to conduct the procedure in part 2 when there is less domain information available
4. Approximating the exhaustive search with gradient-based optimization

### Strengths
1. The introduction does a good job of providing context, motivating the problem, and describing the contributions of the paper.
2. The theoretical results are relevant to the paper
3. The paper is generally clear about the assumptions made in each section
4. Corollary 3.5 is very interesting; perhaps this facet of the paper is worth exploring more.

### Weaknesses
1. The first methods require many assumptions to hold (in the form of extremely specific domain knowledge). While the method in section 3 resolves most of this, it is computationally intractable and it still requires the user to somehow determine $T^*$. The experiments with the gradient-based approximation are useful for resolving the first issue, but they are not entirely convincing; more on this in the next point.
2. The experiments could use several improvements:
    - Timing results, to see the cost of the proposed method compared to the baselines
    - More baselines (specifically, one that trains solely on the target data)
    - More tests in settings with varying amount of source-target causal transferability
    - More runs of each experiment, to verify reliability of the results
    - More explanation of the result in Figure 4b, where the method in the paper is significantly outperformed by the baseline (and doesn't seem to display any learning at all)
3. The writing (and math notation) got continually harder to follow throughout the paper. While the brief sentences to explain various formulas/algorithms are appreciated, the formulas and algorithms themselves are often very difficult to understand (especially in section 4). I suspect that the issue is that there is simply far too much notation going on (and many terms have both superscripts and subscripts); this gets difficult to parse very quickly, especially when different indices on a single symbol range over different types of objects.
4. The module-TR and circuit-TR algorithms seem trivial. The basic idea is to simply search over the possible causal "similarities" between the source domains and the target domain, identify and restructure the relevant data, and train the model. However, although the idea is simple, the algorithms and description are highly complex; it would take me a very, very long time to get this concept out of the pseudocode in algorithms 1 and 2. Simplicity and clarity are important goals in exposition.
5. Putting together equations (6) and (7), and comparing to equation (3) in the case where transport is not possible, makes it unclear whether the proposed circuit-AD method would ever actually perform well in a practical scenario; the additional term introduced by (7) seems like it would typically be very large, and it only scales with $n$ (target domain data) rather than $N$ (source domain data). The paper would significantly benefit from a discussion of this topic: under what realistic conditions is circuit-AD expected to perform better than simply training on the target domain data? Note that, in the small synthetic example 2.4, the value of $T^*=3|\mathcal{V}|$ would already be very large.
    - Relatedly, based on this, it seems that the sentence in the introduction (point 2, adaptation, on page 2) that gives the error rate is not correct; this error rate is in addition to the base error rate of circuit-TR, but that detail is not mentioned in the introduction.
    - The sentence "[...] since $n = \Omega(|\mathcal{V}|^3)$ is needed for a constant excess risk; in fact, the regular ERM on target yields similar error rate" seems to indicate that circuit-AD is, in fact, generally not better than simply training on the target data.

### Questions
1. On page 4, it is mentioned again that there is a positivity assumption, which (in the context of the sentence I am referring to) means that any function the system is trying to learn must be noisy. What are the ramifications of this requirement?
2. Section 4 mentions, in passing, another paper (Nichani et al. 2024) that studies the way that transformers learn causal structure. If existing architectures automatically learn causal structures, why is the gradient-based method proposed in the current paper necessary?

### Soundness
3

### Presentation
2

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
This paper extends causal transportability theory to address zero-shot and few-shot domain adaptation. The authors introduce module-transportability and circuit-transportability defining formal criteria for when modules (local causal mechanisms) or compositions of such modules into higher-level circuits can be transferred from source to target domains. The authors present two algorithms: 1. Circuit-TR, which computes predictors for the target domain by composing transportable modules given explicit causal structure and a discrepancy oracle, 2. Circuit-AD that performs supervised domain adaptation without the causal structure by generating multiple candidate circuits and choosing the one that best fits a small amount of target data. The authors show synthetic experiments consistent with the theory.

### Strengths
- The link between causal transportability theory and circuit complexity is novel. The authors map few-shot adaptation rates to circuit size complexity and reframe sample-efficiency in terms of structural complexity.
- The work is theoretically sound: the authors prove strong control of excess risk for both structure-informed (Circuit-TR, Theorem 2.7) agnostic and agnostic adaptation (Circuit-AD, Theorem 3.2). 
- The proposed gradient-based surrogate for Circuit-AD borrows attention-like components and is computationally viable. The demonstration in Fig.3 that the pre-training phase implicitly learns causal adjacency and discrepancy indicators is a particularly convincing empirical validation.

### Weaknesses
- Empirical validation is only on synthetic arithmetic sequences with a small number of observed variables ($T = 10$) and a small vocabulary size.  The gap between these experiments and real domain adaptation challenges is huge. While real-world experiments are not necessary, more empirical evaluation on synthetic sequences of varying $T$ and $\mathcal{V}$ would help understand the computational viability of the proposed algorithms, especially since Circuit-AD is exponential in $T$ (if I understood correctly).
- Can you offer some insights into how relying on known or perfectly recoverable causal graphs and strict positivity under discrete finite vocabularies translates into practice-- exactly what kind of datasets do these admit? What kind of real-world domain adaption data would it be possible to run the proposed algorithms on?
- In case of a failure of full transport (Appendix D), under what conditions could partial transport still yield some estimable/identifiable quantities?
- It would help to position this paper within the broader landscape of modular and compositional learning work  to understand its contributions beyond transportability.

I am happy to increase the score if the authors can address my concerns and questions.

### Questions
- Does the gradient-based surrogate provably approximate Circuit-AD in the limit of infinite data?
- The analysis is discrete and fully-observed. Can the framework extend to continuous or partially-observed systems using, e.g., additive noise models or variational causal discovery?
- The paper's setup of learning from multiple source domains to quickly adapt to targets resembles meta-learning. Could you add some explanation on the relationship between circuit transportability and gradient-based meta-learning algorithms like MAML?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The draft is attempting to model the few-shot learning, or domain adaption, using ``causal'' graph as without assumption on the structures of variables in the problem we can not achieve transferable machine learning models. The main issue of the draft is that it is very likely the results are not based on causal graph, and instead, it is still based on correlation. The reason is that let's consider a causal system where X is observed data and Y is the label, the contents in X, may not be simply considered as the parent of Y and they can often be child if Y in a causal graph. Furthermore, even if X only contains the parent of Y, it may happen some contents of X are also the child of those part that are parents of Y, in this case, in general it may not be possible of recover the parent of Y from X without further assumption. 

Furthermore, even if for the example case in Figure 2, if we only have X and Y, if may not be possible to recover V3, V4 and V5 from the observed X and Y. In this case, I believe the paper need a major improvement before publication.

### Strengths
There are just too many mistakes in the draft and thus I did not see any strengths of the paper.

### Weaknesses
1. Some of the theory part is wrong.

2. The claim of ``causal'' graph is wrong. 

3. The experiment is weak and only simple simulation is considered and from the current model presentation I can hardly believe that the proposed model would work for a real-world setting.

### Questions
1. How do you justify the graph structure in the draft? I can hardly believe that in a real-world setting a graph would have similar structure. 
2. Are all the V variables observable? If they are, then I believe classic approaches such as Lingam would work, if not, I believe the model is not identifiable in general.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The work extends classical causal transportability theory to compositional settings and formalizes mechanism‑level reuse across domains: if local mechanisms are shared, one can transport them and compose a target circuit even when the target function wasn't used in any of the sources as long as its decomposition was. Algorithms in no-confounder settings are introduced depending on the complexity of the target domain mechanisms and availability of the oracle. Gurantees in excess risk are given in zero- and few-shot learning cases. Finally a NN-based method is proposed for estimating the causal graphs and indicator function of shared mechanisms, with gurantees attached to it.

### Strengths
1. The work appears to be technically solid
2. The motivation for studying compositionality at circuit/mechanism level is clear

### Weaknesses
1. Broader context is unclear: the paper does not convincingly show how the proposed framework could be useful in real applications or what kinds of problems would benefit from it. It reads primarily as a technically detailed but narrowly scoped work rather than a conceptual contribution
2. Empirical results are weak both in breadth and depth; they appear to be sanity checks as opposed to an in-depth stress-tests of the proposals
3. Missing related work: the idea of reusing and composing operators is well-established in modular/meta-learning, the contribution appears to be a formalization inside causal transport.

### Questions
1. Is there a chance of testing the proposals in more realistic scenarios?
2. How does the approach relate empirically to modular/meta-learning architectures that already reuse mechanisms? A concrete example would be [1]
   
 
[1] Parascandolo, Giambattista, Niki Kilbertus, Mateo Rojas-Carulla, and Bernhard Schölkopf. “Learning Independent Causal Mechanisms.” arXiv:1712.00961. Preprint, arXiv, September 8, 2018. [https://doi.org/10.48550/arXiv.1712.00961](https://doi.org/10.48550/arXiv.1712.00961).

### Soundness
3

### Presentation
1

### Contribution
2
