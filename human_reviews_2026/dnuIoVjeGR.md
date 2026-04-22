# Unified Neural Scaling Laws

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
We present a functional form (that we refer to as a Unified Neural Scaling Law (UNSL)) that accurately models and extrapolates the scaling behaviors of deep neural networks as multiple dimensions all vary simultaneously (i.e. how the evaluation metric of interest varies as one simultaneously varies the number of model parameters, training dataset size, number of training steps, and various hyperparameters) for various architectures and for each of various tasks within a varied set of upstream and downstream tasks. When compared to other functional forms for neural scaling, this functional form yields extrapolations of scaling behavior that are considerably more accurate on this set.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a functional form for scaling laws across domains and architectures. The scaling law obeys a number of intuitive desiderata and generalizes a number of prior observed scaling behaviors. Empirically, the scaling law predicts performance very precisely across models and datasets.

### Strengths
The biggest strength in my view is the very strong predictive power of UNSL. The experiments in this paper are extensive and convincing. The authors don't compare with many baselines (primarily only DC/CF)- but frankly, this is fine given the very substantial improvement in predictive power that UNSL provides relative to the baselines here. It is hard to imagine a prior proposed scaling law with this strong fit to data. I overall feel that finding strong functional forms of scaling laws is a very important research direction, so I think this paper has the potential to be an excellent contribution to the field.

Also, the purple box in section 4 is a nice touch!

### Weaknesses
My main concern with this paper is that the proposed scaling law is incredibly expressive- in fact, it's so incredibly expressive that it's hard to imagine it couldn't fit the data. Its comparable to if performance were predicted by an expressive neural network that took in the $x_i$ as input.

Now, this would be ok if either 1) UNSL were theoretically motivated or 2) UNSL were shown to be the minimal scaling law explaining the empirical data. Regarding 1, UNSL obeys some nice desiderata but is not strictly theoretically motivated like some other common scaling laws in the literature. Perhaps the paper would be more convincing if the authors could prove that UNSL is the unique scaling law obeying desiderata 1-8. Regarding 2, the the authors do attempt to show that UNSL is minimal with the ablations A1-A3. Unfortunately, given the large number of moving parts in UNSL, I believe more ablations are necessary to justify each and every component of UNSL. It's hard for me to imagine that there isn't another scaling law with similar complexity to UNSL that also achieves strong predictive performance.

Minor comments
- Please avoid vertical lines in tables (use booktabs style)
- Typically, the appendix is placed after references
- The text in some of the figures is so small that it is unreadable (e.g. Figure 10)

### Questions
- Can UNSL be proven to be the unique scaling law obeying the authors' desiderata?
- Can the authors provide additional ablations beyond A1-A3?
- See minor comments above

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new neural scaling law formula which can be fit to empirical data. This formula contains multivariate broken neural scaling laws as a special case but also incoporates overfitting effects for multi-epoch training. They show that their fits can extrapolate accurately on computer vision, language and sparse parity tasks.

### Strengths
I think this paper attempts to provide a more general and flexible framework for empirical neural scaling law fits. The authors also included a heroic number of experiments attempting to characterize the quality of their model's extrapolation.

### Weaknesses
**Theoretical Motivation** Some of the new terms that the authors introduce are not immediately intuitive, especially the "oppositional force from hyperparameters." Is there a minimal toy model that motivates these terms? In general, I would appreciate any minimal models that motivate inclusion of each term (for instance it appears double descent motivates the broken scaling law in data + model params if I understand correctly?). 

**More complex can be less interpretable** While this model has more predictive power, I am concerned that we could risk developing increasingly complex functions to approximate the scaling laws which makes design decisions (eg compute optimal joint scaling of train time and params) more complex and less interpretable. Chinchilla scaling laws are nice in this regard. As a possibly unfair reductio, at what point do we use a non-parametric or deep network to fit the scaling law itself? 

That said, I do appreciate any improvements in predictive accuracy. I would also be especially interested if the **gaps in explainability** between different scaling law models can be attributed to phenomena that we can invest time understanding scientifically. This is why I am curious about minimal models that capture failure modes of classic scaling laws.

### Questions
1. How do you decide how many breakpoints to include in the fit? What would happen if the breakpoint count were much larger than the true number of breaks?
2. Can you easily deduce compute optimal scaling strategies ($\min_{N,T} \mathcal L(T,N) \text{s.t.} TN = C$) from this unified scaling law?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a highly general functional form meant to describe how neural network performance scales jointly with model size, dataset size, training steps, and hyperparameters. It aims to go beyond the standard two-variable scaling laws (like Kaplan or Chinchilla) toward a unified predictive framework.
Overall Impression
The work is technically ambitious and very thorough, but the motivation and clarity of purpose could be stronger. I appreciate the attempt to unify the many partial scaling laws into one consistent form, but at first read it’s not clear why this particular functional structure is the right way to do it. The introduction lays out the goal of a “unified law,” but the number of nested definitions and the complexity of Equations (1–4) make it hard to see the intuition behind them or to grasp what each part captures in terms of actual network behavior.
As one colleague put it after a quick look: “I couldn’t find the smoking-gun. We already have scaling laws that extrapolate accurately over 10⁴× changes in compute — so what exactly does this improve?” I share that reaction to some extent. The authors demonstrate clear empirical accuracy, but it’s not obvious whether UNSL provides qualitatively new predictive power or simply adds flexibility to fit more phenomena (like overfitting and poor hyperparameter choices) that existing forms treat as noise.

### Strengths
The authors collect an unusually broad suite of benchmarks: multiple architectures (ViT, BiT, Mixer, Transformer), both vision and language domains, and even a sparse-parity setup that tests generalization. That level of coverage is commendable.  Across these tasks, UNSL tends to outperform simpler baselines (CF, DC, A1–A3) in RMSLE, often by a wide margin. The experimental design is careful, and the visualizations of extrapolation (e.g., Fig. 2) are clear and fair. The paper explicitly models phenomena like overfitting and learning-rate effects, which most scaling laws ignore. That’s a valuable step toward realism.

### Weaknesses
Clarity and motivation. The mathematical form is extremely intricate — nested sums, products, and “oppositional forces.” Even with the appendices, it’s hard to understand what behavior each term is meant to represent. A few schematic diagrams or concrete toy examples would help a lot. Predictive novelty. It’s not clear whether UNSL truly extends the predictive range beyond existing laws. The extrapolation plots look accurate, but they don’t seem to cover very wide ranges (e.g., the 10^4-fold compute spans shown in Kaplan et al. 2020 or Hoffmann et al. 2022). Demonstrating a case where UNSL predicts something current methods fail to capture would strengthen the contribution. Interpretability vs. overfitting. Because the form includes many adjustable constants, it’s possible that it simply absorbs “parasitic” phenomena like sub-optimal hyperparameters rather than isolating fundamental scaling behavior. The fitting process (choosing n, S, λ, and seed selection) adds further degrees of freedom. Some sensitivity analysis or cross-validation results would help rule out over-fitting to small datasets. Relation to existing scaling theories. There’s an active line of work tying scaling exponents to the intrinsic dimensionality of the data manifold. It would be useful to know whether UNSL is compatible with or contradicts those ideas. Right now, the connection is left unexplored.

### Questions
The paper would benefit from showing some practical payoff. For example: can this form be used to predict optimal hyperparameters (learning rate, batch size) that minimize loss at a given compute budget? Or to forecast the point of diminishing returns? Without such demonstrations, it risks being an elegant but abstract curve-fitting tool.  The breakpoints (“hyperbreaks”) look numerous and somewhat local; it’s not clear whether UNSL can extrapolate far beyond observed data or whether each region must be densely sampled. The claim that one must fit “sufficiently close” to each hyperbreak raises concern about general utility. The authors position UNSL as an extension of “broken” scaling laws (Caballero et al., 2023), and that’s reasonable. But some comparison with μP (maximal update parameterization) and CompleteP frameworks would be useful. Those approaches explicitly normalize hyperparameters across scale, effectively removing one source of non-scaling behavior. If UNSL primarily captures those sub-optimal regimes, then its “extra accuracy” may just reflect modeling of effects that better parameterizations already avoid.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Unified Neural Scaling Laws (UNSL) — a general composable functional form that provides a lot of flexibility to which kinds of empirical relations between predictors (model size, training steps, ...) and observables (loss, downstream evaluations, ...) can be fitted with UNSL. UNSL allows for capturing a wide array of behaviors: breaks between different power-law trends (as in broken neural scaling laws); various bottlenecks; "oppositional force" of overfitting or suboptimal choice of hyperparameters; misperformance limits.    

The authors define eight desiderata (Section 2.2) to set the desired properties of scaling laws. Then, they demonstrate in Appendix 13, with varying degree of rigor and detail (Appendix 13), that UNSL satisfies all 8 disiderata. Lastly, in Appendix 14, the authors provide a brief empirical evidence for each disiderata. 

In the experimental part, the authors take various open datasets of (predictors, evaluation metric) pairs from previous scaling studies and fit them with UNSL, obtaining lower extrapolation error. These open sets are complemented by additional experiment with learning sparse parity task with MLP. The UNSL fitting process is described in Appendix 10, and amounts to 20000 steps and 20 random seeds of KFAC-JAX second order optimization method to be run on a set of UNSL hyperparameters (such as the number of breaks) to determine the best ones for the final fit.

### Strengths
The authors pursue an ambitious goal of unifying various scaling behaviours in a single functional form. To this end, they provide a suitable high-level structure: 
- formulate a set of desired properties or scaling behaviors (through 8 desiderata points)
- empirically shows the need for these properties 
- connect the set of properties to the scaling law functional law (through showing that UNSL satisfies 8 desiderata)

### Weaknesses
My main concern is the excessive structural complexity of the proposed UNSL. This has several downsides
- *Paper focus.* Almost a third of the paper is taken up just by writing down and providing comments on UNSL form. As a result, many conceptually important steps, like empirical justification of 8 proposed disiderata points, and connecting them to UNSL are fully moved to the appendix. Likewise, experimental validation of the utility of UNSL takes a lot of space just to plainly state the settings and results. Even then, the whole scope of UNSL feels barely covered by the presented experiments.  
- *Analysis depth*. Most of the aspects feel to be discussed/analyzed very briefly (consequence of UNSL complexity and scope), making it hard to convince a reader of the paper's claims. 
- *Switch from modeling actual model behavior to brute force fitting with an expressive functional form.* As the authors rightly mention in the introduction, adding more predicting factors or fitting parameters reduces the conditional entropy of evaluation metrics to be fitted by a scaling law. Naturally, UNSL with many hyperparameters fits the data very well due to lots of parameters (vs <5 for classical chinchilla scaling law). Then, it is unclear whether the good fit comes from the model in the experiment really exhibiting the 8 disiderata behaviors encoded in UNSL, or from the sheer expressive power of UNSL with many parameters. 
- *Practical usefulness.* An important aspect of classical scaling laws is simplicity. This makes them robust and easily verifiable (e.g. a single power-law can be visually verified by plotting data in log scale). UNSL feels more like a complex black box fitted with a second-order optimization method. Then it is very hard to get an intuition about the behaviour of the fitted model; draw conclusions, like compute optimal allocations; and scary to use UNSL for extrapolation to expensive large-scale regimes.     

At least from my perspective, focusing on a few novel aspects of UNSL (i.e. the respective desiderata points) and showing that they are essential for explaining the behaviour of a certain set of deep learning systems would be much more interesting and useful work. 

Also, the writing of the manuscript can be greatly improved - currently it reads as something between a paper and a working journal/draft

### Questions
- Can disiderata points be more rigorously formalized, so that satisfaction of these points by UNSL could be formulated as a theorem? 
- Are there some constraints on UNSL parameters? For example, it is reasonable to assume that $a_i$ are positive, but it is not mentioned in the text.
- What are the principles/practical guidelines behind choosing the sets of predictors $U_r$? The straightforward search over all subsets of $\{1,\ldots,m\}$ might be hard due to the combinatorial explosion of all the possibilities. 
- In the review, I was assuming that all the experiments, except for learning sparse parity task, are taken from prior work and open sources. In some cases, this is explicitly mentioned in the paper, but for other cases, I could not find an explicit comment. Please point out if I mistakenly confuse some of the original experiments of this work as borrowed from the literature.

### Soundness
2

### Presentation
2

### Contribution
2
