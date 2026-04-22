# Disentangled Representation Learning for Parametric Partial Differential Equations

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
Neural operators (NOs) excel at learning mappings between function spaces, serving as efficient forward solution approximators for PDE-governed systems. However, as black-box solvers, they offer limited insight into the underlying physical mechanism, due to the lack of interpretable representations of the physical parameters that drive the system. To tackle this challenge, we propose a new paradigm for learning disentangled representations from NO parameters, thereby effectively solving an inverse problem. Specifically, we introduce DisentangO, a novel hyper-neural operator architecture designed to unveil and disentangle latent physical factors of variation embedded within the black-box neural operator parameters. At the core of DisentangO is a multi-task NO architecture that distills the varying parameters of the governing PDE through a task-wise adaptive layer, alongside a variational autoencoder that disentangles these variations into identifiable latent factors. By learning these disentangled representations, DisentangO not only enhances physical interpretability but also enables more robust generalization across diverse systems. Empirical evaluations across supervised, semi-supervised, and unsupervised learning contexts show that DisentangO effectively extracts meaningful and interpretable latent features, bridging the gap between predictive performance and physical understanding in neural operator frameworks. Our code and data accompanying this paper are available at \url{https://github.com/ningliu-iga/DisentangO}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes DisentangO, a neural operator designed to disentangle physically meaningful latent factors within PDE-governed systems. The model aims to unify forward and inverse PDE learning, enhance interpretability, and achieve latent identifiability under varying supervision levels. It introduces a variational hypernetwork formulation where the latent variables encode distinct physical mechanisms, supported by theoretical identifiability proofs. Experiments across three settings (supervised, semi-supervised, and unsupervised) demonstrate that DisentangO can recover structured latent spaces and achieve competitive prediction performance on forward and inverse problems.

### Strengths
The paper explores building interpretable neural operators capable of discovering physically meaningful latent factors from data. This is an important and promising research area. The experimental design is well-structured, progressing from fully supervised to semi- and unsupervised scenarios, effectively illustrating the model’s generality and stability. Notably, the unsupervised material learning results are engaging: the latent traversal visualizations show that each latent dimension corresponds to interpretable material properties. The paper provides a conceptually coherent attempt to connect neural operator modeling with disentangled representation learning.

### Weaknesses
(1) While the experimental structure is good, the experiment setups remain relatively low-dimensional and simplified. The generalizability of DisentangO to complex systems remains uncertain.

(2) The paper provides only limited quantitative analysis. The evaluation mainly relies on qualitative visual evidence, with few metrics or diagnostic studies to support the claimed physical disentanglement. It would strengthen the work if the paper could include more quantitative or property-based analyses to better assess the model’s physical interpretability and internal behavior.

### Questions
The paper claims that DisentangO enables physically interpretable latent variables. Could the authors elaborate more concretely on how each latent dimension corresponds to specific physical quantities or mechanisms, and whether these mappings can be verified quantitatively?

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
5

### Summary
Researchers propose DisentangO, a novel neural operator architecture that extracts interpretable, disentangled physical parameters from a black-box model, bridging the gap between predictive accuracy and physical understanding in PDE solutions. While neural operators are powerful for solving PDEs, their black-box nature limits physical insight. To address this, the authors introduce DisentangO, a hyper-neural operator designed to solve the inverse problem of identifying the underlying physical parameters. It combines a multi-task neural operator with a variational autoencoder to disentangle latent physical factors from the model's own parameters. This approach not only improves interpretability but also enhances generalization, as demonstrated in supervised, semi-supervised, and unsupervised learning scenarios.

### Strengths
It combines the predictive accuracy and physical interpretability in a unified ML-based modelling framework, improving the automation level and usage of data in this area.

### Weaknesses
This is a straightforward application of VAE and Neural Operators for solving inverse problems. The presentation can be improved.

### Questions
1. The selected baselines are VAE or MetaNO, but there are many baselines are not compared, e.g., classical methods (not ML-based) for solving inverse problem, "adaptive operator learning for infinite-dimensional Bayesian inverse problems" [Gao, et al, 2024], "Deciphering and integrating invariants for neural operator learning with various physical mechanisms"[Zhang, et al, 2024], etc. 
2. Questions related to Figure 2, the derived loss function and the experiment: 
-What is the first term in the loss function in page 1? Is it "f(x)" in Figure 1? It seems that f(x) is the input of the encoder, but it doesn't appear in the Figure 1.
-Which term in the loss function in page 7 related to the true value of b^{\eta}(x) in the supervised learning?
-If b^{\eta}(x) is given, is it also the input of MetaNO?
- Could you explain what is the physical meaning of b^{\eta}(x) and \theta in its related PDE in the experiment in supervised learning?
3. There is format issue in page 2. 
4. In the key contributions, you mention robust generalization. What does "robust" refer to?
5. The title can be improved. The key application seems to be the inverse problems in PDE.

### Soundness
2

### Presentation
2

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
This paper proposes DisentangO, a novel variational hyper-neural operator architecture designed to learn disentangled, interpretable representations of physical factors from parametric PDE systems. The core idea is to perform disentanglement not on the observational data (e.g., the solution field $u(x)$), but on the parameters of the neural operator itself. Specifically, it uses a hierarchical variational autoencoder (HVAE) to encode and decode the task-specific lifting-layer parameters ($\theta_P^\eta$) of a meta-learned Implicit Fourier Neural Operator (IFNO). The authors claim this framework can simultaneously solve the forward problem (physics prediction) and the inverse problem (physics discovery). They provide theoretical analysis to guarantee component-wise identifiability of the latent factors and demonstrate results across supervised, semi-supervised, and unsupervised learning tasks.

### Strengths
The paper's primary strength is its originality. It shifts the focus of disentanglement from the data space ($u, f$) to the model's parameter space ($\theta_P^\eta$). This is a creative approach to embedding an inverse solver inside the forward solver's parameterization, using the MetaNO lifting layer as the information bottleneck.

The inclusion of a theoretical analysis on identifiability (Section 3.2, Theorems 1 and 2) is a major strength. It provides a formal basis for the claim that the latent factors $z$ can, in principle, recover the true generative factors. This is a level of rigor often missing in empirical machine learning papers. The empirical validation of the assumptions (Appendix C.4) further strengthens this.

The paper's "physics discovery" claims are strongly supported by its qualitative visualizations. The latent traversals in Figure 5 (unsupervised) and Figure 6 (semi-supervised) are clear and compelling. They show that the learned latent dimensions correspond directly to meaningful and distinct physical variations (e.g., fiber orientation, digit morphology), providing strong intuitive evidence that the disentanglement was successful.

### Weaknesses
The paper's performance claims are unsubstantiated due to improper baselines in the semi-supervised and unsupervised experiments (Exp 2 & 3). The models are benchmarked against standard VAEs that are not designed for the task and fail (e.g., 61.10% error). To be taken seriously as a neural operator, the method must be benchmarked against other SOTA NOs (like FNO, FUSE, etc.), not just its own backbone.

The paper's core hypothesis is that disentangling from parameters is the right approach. It never validates this against the simpler, more obvious alternative: (1) training a MetaNO for prediction and (2) training a VAE on the data ($u, f$) for interpretation. Without this comparison, there is no evidence that the proposed complex HVAE architecture is justified.

The method pays a high price for interpretability, with the forward prediction error nearly doubling in the unsupervised case. This severe trade-off is never discussed. It is unclear for what practical scientific or engineering application this level of accuracy degradation would be acceptable.

### Questions
Can the authors justify the omission of SOTA neural operators (FNO, FUSE, etc.) as baselines in Experiments 2 and 3? Furthermore, could they provide results for the more direct baseline: a MetaNO for prediction combined with a VAE trained on the data ($u,f$) for discovery?

The results (e.g., 2.67% error for MetaNO vs. 5.28% for DisentangO in Table 3) show a significant drop in predictive accuracy. Can the authors comment on this trade-off? Is this an inherent cost of the method, and in what practical applications would this be an acceptable exchange?

Table 4 shows a dramatic performance drop when using UFNO, which contradicts the "model-agnostic" claim. Can the authors clarify this? Does the method's success (both for prediction and discovery) depend heavily on the specific structure of the IFNO lifting layer?

What is the primary motivation for assuming the lifting-layer parameters $\theta_P^\eta$ are a better representation for disentanglement than the physical data fields ($u, f$) themselves, especially given that using $\theta_P^\eta$ seems to damage the model's forward predictive capabilities?

### Soundness
2

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
2

### Summary
This paper introduces a methodology for learning solution maps of both forward and inverse problems simultaneously with a single neural operator architecture. This is accomplished with a hypernetwork structure. The type of physical problem considered is a family of partial differential equations with forcing and coefficients. The authors use a neural operator to map forcings to solutions for fixed coefficient (supervised). The trained neural operator for this map will depend on the coefficient. So, the authors then train a hypernetwork to map the trained neural operator's parameters (which depend on the coefficient) back to a representation of the coefficient itself (unsupervised). This is the inverse problem part. The proposed method is applied to various benchmark problems in computational mechanics and its accuracy and interpretability are assessed are compared to alternative methods.

### Strengths
The paper is original. The literature review is good. I like the main idea of using hyper-networks to solve forward and inverse problems in an end-to-end fashion. Such methods have potential to significantly advance the field of SciML if scalable enough. The theory part seems adequate and the assumptions and their limitations are discussed well. While not mathematically sophisticated, the results seem interesting and important for the problem considered in the paper. 
The paper considers several numerical experiments and benchmarks against many other related methods. The numerical comparisons seem well done.

### Weaknesses
This paper makes a scientific machine learning (SciML) contribution. As such, it should be accessible to SciML readership. Unfortunately, I do not think this is the case. There is a lot of language and conventions that are unfamiliar to SciML researchers. It seems the paper is mainly written for a core CS/Ml audience. Overall, this paper was very hard to read. I kept going back and re-reading paragraphs. There is a lot of text, but there are often only vague claims of addressing physical interpretability. The paper would benefit from more precise writing and less filler text in Secs 1-3.
The quality is low at times. Notation is not defined before being used, causing much confusion. For example, all of a sudden $\theta^\eta_P$ is used in line 221, where the meaning of P is not discussed.

Some claims seem to good to be true. For example, lines 168--169 says "... we propose a novel neural architecture to alleviate the
curse of ill-posedness without requiring prior knowledge." This seems impossible because the ill-posedness is inherent to the inverse map being learned and cannot be removed by an architecture alone, unless the architecture incorporates prior knowledge (AKA, regularization). While the VAE does regularize the training, it is not clear if it regularizes the ill-posedness inherent in the data and the physical problem.

### Questions
Questions:
1. I find Figure 1 hard to understand. Why are all three input fields $ f $ in the figure the same? What is "multitask" about that? What is the function that is being input into the Fourier neural operator layers? Why does the figure discuss loss functions? An "architecture" should be completely independent of any loss or training approach.
		
1. Is it crucial that $ n_{train} $ is fixed to be the same number for all $ S $ datasets?
		
1. Line 200: it is not clear yet why the hypernetwork map from parameters to coefficient needs to be unsupervised. Do you assume knowledge of the form of the PDE itself? I thought the physics was supposed to be "hidden"? It seems line 330 considers a fully supervised version.
1. Lines 205--215 are confusing. What is $\theta$? If it is the trainable network parameters, why do you mention that $\theta$ is related to Eqns 3.5--3.6, which only involve the data and are independent of any choice of architecture? It is likely that this is a notation issue.
1. What is the role of VAE here compared to an standard autoencoder or deterministic map? It is not clear why the probabilistic method is chosen over a deterministic one.
1. Lines 243--245: what are the different tasks? these should be stated clearly in terms of the PDE model.
1. Sec 4.1, I don't think NIO is applicable to this test problem because it only is defined for empirical-distribution-to-function inverse maps. What is the observed data for the inverse Holzapfel–Gasser–Ogden (HGO) model?

Typos and notation:
1. The authors severely shrink the space between paragraphs, sections, and figure captions to make the page limit. The result is not visually appealing.
1. line 146-147: should the noise depend on $ x\in\Omega $? E.g., $ \zeta=\zeta(x) $ here and elsewhere in the paper. Lines 184 is not well defined because adding scalar Gaussian noise to function values $u(x)$ does not even define a function $ u $. Maybe they mean to define a spatially correlated Gaussian noise process instead?
1. Eqn 3.7 $\mathcal{G}$ should depend on $ \eta $ (or $ z $)?
1. Does $ E $ mean expectation? This is never defined. $\mathbb{E}$ is much more common.
1. Line 206-207: What is $\mathcal{D}$, is this $\mathcal{D}_{tr}$ or a different dataset? Also, the authors should say that the max is taken over $\theta$ instead of being ambiguous.

### Soundness
3

### Presentation
1

### Contribution
2
