# Behavior Learning (BL)

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Inspired by behavioral science, we propose Behavior Learning (BL), a novel general-purpose machine learning framework that learns interpretable 
and identifiable optimization structures from data, ranging from single optimization problems to hierarchical compositions. It unifies predictive performance, intrinsic interpretability, and identifiability, with broad applicability to scientific domains involving optimization.  BL parameterizes a compositional utility function built from intrinsically interpretable modular blocks, which induces a data distribution for prediction and generation. Each block represents and can be written in symbolic form as a utility maximization problem (UMP), a foundational paradigm in behavioral science and a universal framework of optimization. BL supports architectures ranging from a single UMP to hierarchical compositions, the latter modeling hierarchical optimization structures that offer both expressiveness and structural transparency. Its smooth and monotone variant (IBL) guarantees identifiability under mild conditions. Theoretically, we establish the universal approximation property of both BL and IBL, and analyze the M-estimation properties of IBL. Empirically, BL demonstrates strong predictive performance, intrinsic interpretability and scalability to high-dimensional data. Code: https://github.com/MoonYLiang/Behavior-Learning; installable via pip install blnetwork.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles the central challenge of building interpretable machine learning models without sacrificing accuracy. It proposes Behavior Learning, which models predictions as solutions to stacked utility maximization problems, yielding latent structures that are interpretable by design. The authors also introduce Identifiable Behavior Learning, an architecture with identifiability guarantees for recovering ground truth parameters under certain assumptions. Across predictive and causal inference benchmarks, the method delivers performance on par with strong baselines while providing intrinsic interpretability and demonstrating scalability to high-dimensional data.

### Strengths
1. The problem of providing more interpretable ML systems without sacrificing performance is a well motivated problem. The proposed algorithm is also quite interesting, relating the predictive methodology of their approach to solving a system of behavioral science motivated MDPs.
2. The theoretical results are well presented and all proofs look sound.
3. BL outperforms competitive methods across multiple prediction and causal inference problems, while still providing interpretability. The Boston Housing case study provides an interesting and necessary demonstration for how to extract the human interpretable structure from the model.

### Weaknesses
1. It seems like the work needs some moderate restructuring. The IBL identifiability result hinges on Assumption D.1, but it only appears in the appendix. Related work coverage in the main text is also sparse. The paper ends abruptly without a final discussion of their contributions.
2. While the Boston Housing case nicely illustrates BL(Single) and individual blocks, there’s no intuitive instruction for how to interpet layers of blocks as in BL(Deep). It is unclear how if the model becomes seeming more difficult as more layers, how this approach would still claim to be providing intrinsic interpretability.
3. The implementation uses bounded-degree polynomials, however many real systems have non-polynomial structure and complex interactions.

### Questions
1. For BL(Deep), what is the procedure to generate scientific explanation from the network of composed blocks?
2. Could BL be easily extending to consider non polynomial feature maps?

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
2

### Summary
The paper introduces Behavior Learning (BL), a machine learning framework for prediction (primary focus) and generation, designed to be intrinsically interpretable. BL models are built from modular computation blocks, each mathematically equivalent to a utility maximization problem (UMP). This structure enables direct interpretability: the model’s output can be understood as the result of maximizing a utility function subject to constraints.

The paper establishes theoretical contributions for BL. First, it defines a subclass called Identifiable Behavior Learning (IBL), where penalty functions are smooth and monotone. It shows that the model parameters in this class are identifiable up to an equivalence class, ensuring a unique and consistent interpretation of the learned behavior. Second, both BL and IBL possess universal approximation capabilities, demonstrating that interpretability does not come at the cost of predictive performance. For IBL, the authors additionally prove desirable statistical properties, including consistency, universal consistency, asymptotic normality, and asymptotic efficiency.
Finally, the paper empirically validates BL across a range of prediction tasks, including those with high-dimensional inputs, and shows that BL can accurately estimate causal effects.

### Strengths
**S1**. The framework is novel in that it leverages utility maximization problems (UMPs) to construct models with intrinsic interpretability, allowing outputs to be directly understood as utility-maximizing decisions under constraints.

**S2**. The theoretical analysis is thorough and solid, including universal approximation guarantees, identifiability results for IBL, and other properties of IBL. 

**S3** The experiments are comprehensive, evaluated on diverse datasets with strong baselines, and demonstrate that BL performs well not only on prediction tasks but also on causal effect estimation.

### Weaknesses
**W1**. Although BL is intrinsically interpretable at the block level, the utility and constraint functions are fully learned during training, making it difficult to connect the resulting functions to real-world meaning. For instance, in the interpretability case study (Fig. 2), the learned utility $\tanh((1-P)(1+P-RM) + R_u)$ lacks a clear semantic interpretation. As BL becomes deeper, it introduces more learned symbolic functions that may reduce the practical interpretability of the model.

**W2**. The paper lacks sufficient case studies that demonstrate interpretability in practice. Despite interpretability being a key claimed advantage of BL, the experiments focus mainly on prediction accuracy, with only limited analysis of interpretability (e.g., Section 3.3). The case study for deeper BL architectures (Fig. 10) remains high-level and does not include a concrete real-world interpretability example.

**W3**. The causal inference section (CausalBL) is confusing in definition and terminology. Section 3.2 and Appendix E describe the setup, method, and experimental results, but the treatment of potential outcomes and propensity scores is inconsistent. Lines 1860–1861 suggest the goal is to estimate $p(t \mid x), y_0, y_1$, implying an ATE formulation. However, lines 1944–1955 describe a setup where y becomes input to the propensity score and the target becomes ITE, not ATE. The paper should use causal inference terminology more carefully to avoid confusion, even if ATE estimation can be derived from ITE in practice.

### Questions
**Q1**. In the shallow and deep BL architectures, multiple UMP blocks (Eq. (4)) are computed in parallel and then combined through an affine transformation. After this aggregation step, is the entire composed model still guaranteed to be equivalent to a single utility maximization problem (UMP)? If this is not the case, in what sense can we still interpret each individual block as performing a UMP, especially when the affine weights may change the sign or scale of the penalty/utility terms?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes _Behavior Learning (BL)_, a specialization of Energy-Based Models which provides a natural decomposition of an EBM into constrained utility optimisation problems ("blocks") each of which can be interpreted as a utility term minus  loss terms which penalise solutions that do not satisfy constraints.
Specifically, it parameterises a joint distribution $p_{\tau}(x,y)$  energy $BL_\theta(x,y)$, where the energy is built by composing “utility–minus–constraint-penalty” blocks.
Each block encodes a *utility* term $U$ and *constraint* terms $C$ (inequality) and $T$ (equality), passed through fixed activations and linearly combined into a single scalar compositional utility. 
Training uses cross-entropy for discrete $y$ and denoising score matching (DSM) for continuous $y$, thereby avoiding explicit partition-function estimation.
Inference for fixed $x^0$ uses the Gibbs form $p_{\tau}(y\mid x^0) \propto \exp(BL_\theta(x^0,y)/\tau)$.
A sub-type (IBL) restricts the activations to smooth families and gains identifiability of parameters to shape-constrained (monotone/smooth) forms to obtain guarantees of identifiability of parameters.
As such BL is perhaps best understood as a collection of  sub-families of EBMs which guaranteeing utility/constraint semantics.
Experiments (classification/regression/causal tasks; some higher-dimensional inputs) aim to show competitive accuracy with stronger interpretability and sometimes improved OOD behaviour.

### Strengths
### Originality

The architectural idea is, it seems, novel, even though it is closely connected to other attempts at the same goal. Reparameterizing an energy function as a composition of relaxed constrained utility maximisation programs is neat. Universal approximation theorems this reassure us that a BL program the capacity to well-approximate both any conditional density and rather general problem constraints, albeit relaxed via penalties. We can view this as guaranteeing interpretable behavioral model for random distributions.

**Identifiable variant (IBL)**: In the IBL sub-family, not only does this interpretation exist, but we can read it off the learned parameters under reasonable assumptions. 

This synthesis of constrained optimisation theory seems simple in hindsight but I cannot find it in the literature. As such: cool insight.

### Quality

The model has a strong  Mechanistic narrative: “utility minus penalties” + Gibbs distribution = stochastic bounded-rational behavior. The existence of such a disentangling is satisfying, and they show that in some cases it can even be recovered.
The Gibbs form makes the model a bona fide conditional density; the $\tau\to 0$ limit recovers deterministic best response.

We have other networks that naturally encompass constrained solutions; for example Gould's Deep Declarative networks: compared with inverse RL/bi-level pipelines that differentiate through argmax/KKT by the implicit function theorem, BL’s training objective stays first-order and stable, using standard autodiff on $BL(x,y)$.
The theory for IBL** seems promising:  identifiability and consistency and asymptotic efficiency are unusual strengths for high-capacity models.

### Clarity

Presentation is generally clear (although see below for some notes regarding the Deep BL and CausalBL variants). 
Examples solid. Prose style is clear.

### Significance

Definitely a first step in an interesting direction.
The notion here of “interpretable EBMs” is appealing, and the framework seems flexible enough to accommodate a variety of domain theories via the choice of block heads and compositions.
Causal extensions and the promise of interpretability are high value in high-risk or sensitive or legally constrained domains, or where the natural way to incorporate prior knowledge is via constraints.

### Weaknesses
There is a lot crammed into this paper (identifiability results, convergence results, universal approximation bounds...) and the authors have made some judgment calls about what to include in the body of the paper, which makes it difficult to evaluate the paper in terms of the body text claims.
Ideally it would have been two papers; there is enough material here for two.

IMO the places where this is toughest are 

1. Causal BL: Impressive, but it is entirely relegated to the appendix, as well as the considerable machinery that it requires. I understand that this result is super cool, but we can only evaluate it base on the main text. If this is to remain in the paper something else should be cut to make room for a fuller treatment. 
2. Deep BL: This feels to me like the least intuitive part of the paper, and the treatment in the main text is light. I have many questions about the mechanics of this setting (*see below) 

**Interpretability trade-offs**

The model runs straight at the performance-interpretability tradeoff challenge, but there are some open questions about how well it navigates this tradeoff in practice.

- If I understand correctly composition of BL block results in different behaviour in later layers than in earlier ones; it seems there are bottlenecks in the layers and that practical high capacity networks throw out many of the interpretability guarantees; we probably want to know more about this tradeoff surface.

AFAICT, cost of using monomial bases for $U,C,T$ can explode combinatorially with input dimension and degree. The paper partly sidesteps this (affine heads in deep/high-dim settings), but then the identifiability and symbolic clarity are less compelling. The trade-off deserves a tighter discussion and ablation.

At finite temperature, Gibbs does not enforce constraints exactly; samples can violate $C\le 0, T=0$. This weakens the normative “optimizer” story unless $\tau$ is very small or penalties are very large; then training/sampling may suffer from stiffness. We probably need to know more about this,

### Questions
### How does Deep BL work in practice?

I'm trying to map my understanding of the composed DeepBL to standard deep nets, and have gotten stuck on the paper. Can the authors clarify for me what happens in various cases? 
I have questions both about how the deep network works, and what it means.

In the default case (around eq (5)) each block outputs a single scalar, so at layer $\ell$ we have a state vector of size $d_{\ell}$.
That is, later layers operate on a low-dimensional vector of utilities, not on rich features.
Is that correct? This is a serious disanalogy with blocks in the first layer which actually observe the input features and labels.

We learn that introducing $(x,y)$ as inputs ("skip-connections") to later blocks restores capacity but it looks like weakens the “everything is a UMP” interpretation, since later blocks are no longer functions purely of upstream utilities.

Anyway, my question is: have I got that right? I got lost in the details of the Deep BL section and would appreciate some clarifications.

Also, why are Deep and Shallow BL treated differently? They are both the same but $>2$ right? I'm not sure I understand the choice to spend previous column inches on introducing more subfamilies than absolutely necessary in this crammed paper.

### Interpretability-performance tradeoffs

I don't think the authors claim to have eliminated the performance-interpretability tradeoff but I think the paper makes a claim to have implicitly shifted the pareto frontier of this tradeoff downwards.
Have they quantified this?

Not that I am proposing new experiments, but I think the paper needs to be clearer about the tradeoffs it is making, and to what extent the interpretability gains come at a cost in accuracy or efficiency.

The BL formalism seems to include a lot of design choices that affect this tradeoff (Skip connections, affine versus polynomial bases etc), and I think the paper needs to be clearer about what happens at scale to the interpretability guarantees.

For example to isolate the value of the BL parameterization, the paper could compare against capacity-matched EBM/Deep declarative network/ symbolic regression/Variational bayes baselines, and see how complex the hypothesis space can grow while still retaining "interpretability" in some sense, for each of these.
This is probably out of scope for the current paper, but what is the closest approximation to this that could be done in the current body of work?

### Constraints

**Constraint satisfaction under noise.**  
*Q:* How large must penalty weights or how small must $\tau$ be to achieve near-hard constraint satisfaction in practice?  
*Request:* Provide plots of feasibility violation vs $\tau$/penalty scale and discuss numerical stability.

It would be super nice if you for example imposed a physics residual constraint (e.g. conservation of mass/energy) and showed how well it is satisfied in practice as a function of $\tau$ or penalty scale in  a high dimensional problem.

### CausalBL

I have many questions about the CausalBL variant; is there any chance you can squeeze an actual description of it in to the body text?
If not, should it even be in the paper? 
These results seem powerful, and potentially very significant, but also barely explained. Maybe there is a whole other paper there?

### Calibration

Since the model is probabilistic, how well-calibrated are the predictive distributions?  
Any chance we could get proper scoring metrics (e.g. ECE/NLL/Brier)  vs EBM/MLP baselines?

### Hybrid approaches for partial explainability

It seems that this model gives us some desirable explainability qualities under simple settings (essentially the IBL setting: polynomial bases, if the network is deep then block communicate only by utility signals etc) but in larger practical problems the authors relax these simplifications (using affine bases over large input spaces, skip connections and so on). This suggests that maybe "partial explainability" is potentially within reach, selecting a subset of model space in which IBL blocks are used, but then allowing "non-explainable/identifiable" blocks to participate in inference, for example if we do not need explainability over all model interactions, but desire it for a small subset. Potentially we could absorb arbitrary black box interactions that way. Have the authors considered this?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes to learn data distribution P(x | y) by using a specific form of parametrized probability families, motivated from the Lagrangian reformulation of constraint programs. The paper moves on to develop relevant theories for this parametrized probability class, including approximation results, uniqueness of optima of the loss constructed from this class, and certain estimation results including consistency.

### Strengths
1. The paper is very well written, and compares with many benchmarks. In appendix there are abundant introduction of the dataset used and the competing methods.  
2. The proposed parametrization has utility maximization interpretation. Learning the model is uncovering agents' constraint, utility function at the same time.

### Weaknesses
1. The statistical results part is standard. A more interesting theory would be to show, the parametrization is effecient if the data distribution is indeed generated from some utility maximization process.

### Questions
1. In Eq 4, what are the exact forms of phi, rho, and psi? Are the the same as those in Theorem 2.1? Are they hyper-parameters to be chosen? If so how to choose them? 
2. In section 3.2, do the authors consider endogeneity in the dataset, or do they assume exogeneity? If latter, then the problem is just simple regression problem.
3. In all the numerical comparison, are the number of parameters similar? Using similarly complex models ensures a fair comparison. 
4. In the numerical part, does BL recover the relevant utility and constraint functions?

### Soundness
3

### Presentation
4

### Contribution
2
