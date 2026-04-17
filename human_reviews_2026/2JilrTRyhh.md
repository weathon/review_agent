# Transformers as Measure-Theoretic Associative Memory: A Statistical Perspective and Minimax Optimality

- Decision: Accept (Poster)
- Scores: 8, 4, 2

## Abstract
Transformers excel through content-addressable retrieval and the ability to exploit contexts of, in principle, unbounded length.
We recast associative memory at the level of probability measures, treating a context as a distribution over tokens and viewing attention as an integral operator on measures. 
Concretely, for mixture contexts $\nu = I^{-1} \sum_{i=1}^I \mu^{(i)}$ and a query $x_{\mathrm{q}}(i^\*)$, the task decomposes into (i) recall of the relevant component $\mu^{(i^\*)}$ and (ii) prediction from $(\mu_{i^\*},x_{\mathrm{q}})$.
We study learned softmax attention (not a frozen kernel) trained by empirical risk minimization and show that a shallow measure-theoretic Transformer composed with an MLP learns the recall-and-predict map under a spectral assumption on the input densities. We further establish a matching minimax lower bound with the same rate exponent (up to multiplicative constants), proving sharpness of the convergence order. The framework offers a principled recipe for designing and analyzing Transformers that recall from arbitrarily long, distributional contexts with provable generalization guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In the paper, the authors cast associative memory in transformers as a problem in retrieving the correct component from a mixture distribution via querying. The authors show that a (measure-theoretic) transformer trained on empirical risk minimization is able to implement via an approximately one-hot softmax the optimal target function with sub-polynomial rate.

### Strengths
The authors carry out a sophisticated theoretical analysis of the problem. Particularly nice is the tight characterization of the excess-risk rate. Despite the complex nature of the work, the paper is relatively well written and organized. I especially like that the authors reserved some space to give a quick proof sketch for theorems in the main body. The proofs seem reasonable, even though I could not check the appendix in detail.

### Weaknesses
The limit of the work is obviously the lack of clear practical implications, being a purely theoretical work. The lack of experimental results is also a consequence of the same fact. Nonetheless, the results are interesting and a worthy theoretical contribution.

On the minor side:
1. Lines 176-177 are repeated at lines 178-179;
2. The definition at line 211 is confusing: there is no F in the argument;
3. I would take a little space to explain what RKHS in Sec. 3.1.1;
4. There is an extra "that" at line 406.

### Questions
1.Do you think that this work could be relevant and practically useful for the community in any way? More specifically, why do you think that an infinite-dimensional context in the form of a mixture distribution can be meaningful in practice?
2. The assumptions are not discussed extensively and they seem quite strict. In the conclusion you mention the possibility to extend your results to broader regimes such as polynomial instead of exponential regimes. What is currently the main difficulty in carrying out such a generalization?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper develops a measure-theoretic view of Transformers for 
associative recall.
Instead of treating a context as a finite token sequence, the input is a 
mixture of probability measures over tokens; a query identifies the 
relevant component measure (recall), and the predictor maps that 
recalled measure (plus the query) to an output (predict).
Technically, attention is lifted to an integral operator on measures, so 
sums over tokens become integrals.
Under a spectral assumption, i.e. Mercer eigenvalues of the RKHS
kernel decay exponentially, the authors construct a shallow (depth-2) 
Transformer+MLP and prove a sub-polynomial excess-risk bound of the 
form
$\exp\{-\Theta((\log n)^{\alpha/(\alpha+1)})\}$ for ERM, and provide a 
matching minimax lower bound (same exponent, up to constants).
They argue learned softmax attention can implement near one-hot, 
query-adaptive recall that isolates the correct component measure and 
aggregates a small set of its Mercer coefficients for prediction.

### Strengths
To the extent of my understanding, which is limited, the modeling of contexts as probability distributions seems an interesting line of research, with several other recent papers treating it.

### Weaknesses
Clarity:
I found the paper extremely hard to parse (which may be my fault, and not a weakness, I let the area chair judge). For example:
- line 73: it is not clear what is meant by "context"
- line 74: it is not clear on which space do the measuere nu, mu^(i) live in
- line 75: it is not clear which space the query x_q lives in
- line 76: what is the "target map". Is it some sort of ground-truth function? This should be defined clearly
- etc...
While stated as "informal", I find this introductory section quite confusing.
The paragraph in line 149-167 is not illuminating either on this, and one needs to reach line 169 to finally know what are the objects we are speaking about.
While the paragraph starting at line 169 provides definition, they are hard to understand: I would suggest providing examples here!

More generally, important intuitions (effective dimension $(\log n)^{1/(1+\alpha)}$, role of the second attention block, sensitivity to query–key geometry) are 
scattered across sections and appendices, hampering readability for a 
broader audience.

Model:
The query explicitly contains the relevant tag $v^{(i^\*)}$ and the tags 
are well-separated (inner products $\le 0$), making recall potentially substantially 
easier than in natural setting.
 
Lack of any empirical validation:
The work is purely theoretical. Even small 
synthetic experiments would help test the sharpness/spikiness of learned 
softmax and test the assumptions (e.g., performance deterioration 
when spectra are polynomial).

### Questions
Polynomial spectra. Can you extend (even heuristically) the analysis to 
kernels with polynomial eigen-decay? Which proof steps break and what 
rates would you expect?

Query robustness. What happens if $x_q$ does explicitly contain 
$v^{(i^\*)}$ or if the tags are weakly separated/noisy? Can the softmax 
gate still become sufficiently spiky under ERM?

Empirics. Could you add synthetic experiments varying $\alpha$ and 
$I$, and compare learned softmax vs. linear attention / frozen kernels, to 
illustrate the theory?

typos:
- line 179 repeats line 176
- line 180: \Chi_0 not defined
- line 211: inside the argmin should be F? Same in line 214

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper attempts to study the associative memory problem theoretically. It formulates the "context" to a transformer as a mixture of measures.  The goal of the transformer is, on reading a query corresponding to one of these constituent measures, to identify the associated measure.

### Strengths
The paper attempts a theoretical analysis of a practically relevant setup in transformers, which is always welcome and appreciated.  It presents matching upper and lower bounds to the risk of a statistical estimation problem that they pose as a proxy to the associative memory problem.

### Weaknesses
The biggest issue of this paper is that the mathematical notation is (a) far too dense for a conference such as ICLR, and (b) just poor, objectively (there are too many to list, but see Questions for a few).  At a high level, the paper reads as follows to me: the paper spend 7 pages (!) setting up notation and making several assumptions (some of which, admittedly, do seem reasonable, but, e.g., the eigenvalue decay rate, just show up out of nowhere) and then obtains some results that likely do not hold if any of the assumptions are violated even slightly (e.g., if the eigenvalue decay rate is different).  

Is there really a need for the entire paper to be measure-theoretic?  What would be different if we just assumed these were just a linear combinations "discrete distributions" of some finite set and then went with it?  The insistence on measure theory and functional-analytic analysis makes the paper very specialized, so if it could be simplified at all, it would be greatly appreciated (I understand that often we do actually require this level of sophistication to analyze some setups, but this setup feels like a more interesting analysis could be made with simpler mathematics).

What exactly is the statistical estimation problem introduced in Section 3?  As far as I understand, it is an abstraction of the memory recall problem (I might be wrong, but this is what I understood from the paper).  In this case, what exactly is the value of such an abstraction and analysis?  I do not think there is value in knowing at exactly what rate it decays, given that the problem itself is a "fake" problem invented only a a proxy.  I think there would be value if the paper were able to provide insights/explain some curious phenomena on the general problem of associate memory and recall, but I do not see any such insights.

### Questions
1. Why this choice of eigenvalue decay (lambda_j = exp(-c j^alpha))?  This seems rather arbitrary and I do not see any justification/explanation for it.  More importantly, this alpha is crucial and shows up in the final results, making them very fragile (not robust to this assumption).

2. Example 1 lists some examples satisfying the assumptions, but it is not clear if these examples satisfy all or some subset of the assumptions, and whether these examples are interesting for the setup being considered (e.g., one could list any arbitrary assumption and say that there are functions satisfying that assumption, but whether it makes sense to consider that example for that setup is important).  Comments?

Minor comments:
1. Equation at line 211: should be F, not F^\star
2.  Section 3.2, one of the expressions for either SAttn or MSAttn must be wrong: both have a V^h x at the end, and MSAttn does not have a W^h  (guessing that SAttn is right and MSAttn has W instead of V)?
3. Definition 5, the equation: what is nu_1, did you mean mu_1?

### Soundness
2

### Presentation
1

### Contribution
1
