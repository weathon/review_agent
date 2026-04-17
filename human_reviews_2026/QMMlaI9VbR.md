# Geometric Autoencoder Priors for Bayesian Inversion: Learn First Observe Later

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Uncertainty Quantification (UQ) is paramount for inference in engineering. A common inference task is to recover full-field information of physical systems from a small number of noisy observations, a usually highly ill-posed problem. Sharing information from multiple distinct yet related physical systems can alleviate this ill-posedness. Critically, engineering systems often have complicated variable geometries prohibiting the use of standard multi-system Bayesian UQ. In this work, we introduce Geometric Autoencoders for Bayesian Inversion (GABI), a framework for learning geometry-aware generative models of physical responses that serve as highly informative geometry-conditioned priors for Bayesian inversion. Following a ''learn first, observe later'' paradigm, GABI distills information from large datasets of systems with varying geometries, without requiring knowledge of governing PDEs, boundary conditions, or observation processes, into a rich latent prior. At inference time, this prior is seamlessly combined with the likelihood of a specific observation process, yielding a geometry-adapted posterior distribution. Our proposed framework is architecture-agnostic. A creative use of Approximate Bayesian Computation (ABC) sampling yields an efficient implementation that utilizes modern GPU hardware. We test our method on: steady-state heat over rectangular domains; Reynolds-Averaged Navier-Stokes (RANS) flow around airfoils; Helmholtz resonance and source localization on 3D car bodies; RANS airflow over terrain. We find: the predictive accuracy to be comparable to deterministic supervised learning approaches in the restricted setting where supervised learning is applicable; UQ to be well calibrated and robust on challenging problems with complex geometries.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposed an autoencoder-based scheme of constructing priors for inverse Bayesian problems in PDE. These priors are build from data of similar tasks with the same underlying physics but different meshes.

### Strengths
The strength in this paper lie in strong experimental evaluation. The results of the experiments seem good and improve upon the state of the art by up to an order of magnitude. The Experimental range is quite wide, starting from the simple heat equation to more complex, non-linear systems of PDEs in varying geometries. This is the main reason for a good contribution score.

### Weaknesses
The main weakness lies int he presentation of the paper. I had quite a hard time of following it and even at this stage I do not get the details of the approach.
 - Some problems lie in formalizing the notation. See the questions below. I stress that these questions are just for the main paper, the appendix (which I just flew over) was equally problematic. I want to clarify that for many of my questions bwloe it is alluded what the notation means, but not made formal.
 - The paper solves a relevant problem. However, it took me quite some time to understand (and then only even vaguely) which problem is being solved. My main problem was that it took some time to understand what was given and what was not given.
 - Experiments compare to machine learning methods only, and not to methods from the inverse problems literature. (I know that it is hard to do experiments against all potential competitors, so I am not dwelling on this point for so long, as some comparisons are being made.)
 - I did not find any description of computational time in this paper. For such computational approaches, I think this is a critical point.
 - l. 256ff "The proposed methodology is agnostic to any particular architecture, however we need the following characteristics: Geometry aware architecture and data-structure; Mapping to and from fixed-dimensional vectors; Non-locality. We expand on these points in B." This makes the paper (without appendix) far from being standalone.
 - Lemma 2.1 is relatively simple. However, it is stated very technically, without giving an intuition. 
These weaknesses explain my soundness and presentation scores.

### Questions
l. 96: which space is $u_o$ from?
l. 97: What kind of Operator is $H_o$? Linear? Bilinear? 
l. 97: $R^o$ is the output of $H_o$. Why/how does it select nodes?
l. 198: How do you compute the argmin? (SGD does not compute argmin!)
l. 207: what is $H_o$? What is $D_o$?
l. 211-215: what are the various $p$'s introduced here?
l. 255ff: I do not even know which part of the methodology is trained using neural networks. Encoder? Decoder? $H_o$? All of them?

### Soundness
1

### Presentation
1

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
This paper combine autoencoder (AE) and approximate Bayesian computation (ABC) for inverse problems. AE is trained on geometry of the physics problems to obtain common features in the latent space, which is independent of the governing equations (PDE). The observation process is only used in the sampling stage with ABC. The proposed method, hence named Geometric Autoencoders for Bayesian Inversion (GABI), enables efficient Bayesian inversion in a `learn first, observe later` fashion.

### Strengths
The idea is straightforward, clearly presented, and efficiently implemented (via multi-GPUs). The demonstration on high-dimensional real engineering problems (car body acoustic vibration) is definitely a plus.

### Weaknesses
It is not clear the trade-off between the size of geometric AE and the precision of the downstream ABC. And it is not clear how it compares with transfer learning method.

### Questions
1. How large the geometric AE is required to ensure sufficient precision in ABC? Is there any trade-off between the size of AE and accepted sample size $N_a$? Can you do some ablation study on it?

2. Since the AE is trained on the geometry of different but similar physics problems, I wonder how GABI relates to or compares with transfer learning methods?

Minor things:

Line 062: remove semicolon.
Line 098: I assume you meant for $H_o: R^{d_u} \to R^{d_o}$?
Line 119, 143: period is missing after the reference.
Line 422: `Figure` 4.
After Equation (6), maybe emphasize that the likelihood is not computed but ABC is used.

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
This is an interesting work applying Bayesian methods (autoencoders) to Geometric (prediction) problems. This will be applied when a new shape is designed, and people want to know how the field (temperature, pressure) will change. The authors use a simple solution that just encode the shapes (M) into a latent space during learning. When given a new shape, suppose we have enough trained shapes - the model will sample in the latent space N(0, 1), and produce related vector fields from the decoder, and match with the available observations. From four experimental study, the performance is verified.

### Strengths
I like this paper. It proposed a very valid method using autoencoders, and potentially solves a pretty difficult and highly practical problem. 

1. The presentation is very nice. The written style, mathematical notation, experimental figures are all well presented. 

2. The idea is clean, and nicely described. The algorithm is simple and easy to understand. 

3. The experimental study nicely demonstrates the advantage of this method.

### Weaknesses
The theory part looks strong, but may have very limited theoretical impact. Although everything is well documented in theory, but it could create problems for readers. And an important critique from the reviewer is that - the theory seems to be rooted on two things (i) such latent space exists and (ii) the decoder (from this latent space) can be expressive enough, and it is able to nicely extrapolate data. These are all likely to be true, yet unfortunately, these assumptions are potentially hidden in the problem statement. 

Please see questions below as well.

### Questions
1. In 1.1 setup, would the authors define U? In Lemma 2.1 or 1.3, would the authors use a line or two to describe what z normally is (e.g. the latent space)? 

2. In Figure 2, the reviewer would like to ask (i) why is this figure so small.. and (ii) how should we interpret (b)? 

3. The reviewer is very curious if the computation can be accelerated without incorporating the Bayesian framework. It seems that the Bayesian framework is important, but might not be that essential (if one wants to have the fastest algorithm). 

4. How would GABI perform for OOD shapes? The reviewer believes that OOD it will perform poorly as the decoder can have random behavior during extrapolation. A comment here would be great (don't need to change paper).

5. Would the authors add some words about the implicit assumption in the main paper? As the latent space z might not actually exist. Or - the latent space z might be able to represent U/M under a bounded area/domain. With better/larger collected data, this problem can be resolved. 

The reviewer is likely to improve the rating if some of the comments above being resolved.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces GABI (Geometric Autoencoder Priors for Bayesian Inversion), a framework for solving inverse problems, such as reconstructing temperature, pressure, or flow fields from limited noisy measurements, when system geometries vary. Instead of assuming known physics or PDEs, GABI learns a geometry-aware prior via an autoencoder that compresses physical fields across diverse geometries into a shared latent space. During inference, it performs Bayesian inversion in this latent space using Approximate Bayesian Computation (ABC) sampling, which scales efficiently on GPUs and outputs well-calibrated uncertainty. The approach enables a “train once, use anywhere” model that generalizes to unseen geometries and observation setups. Experiments across heat transfer, airfoil flow, car acoustics, and large-scale terrain flow show GABI matches supervised baselines in accuracy while offering full Bayesian uncertainty and far greater flexibility.

### Strengths
I think what makes this paper stand out isn’t just that it “does Bayesian inversion with deep learning”, but how it redefines the workflow. Most prior works either fix geometry and bake the PDE into the model, or treat inference as a supervised regression from observations to fields. Both approaches that crumble when geometry or observation setup changes. GABI breaks this dependency entirely: it learns a geometry-conditioned latent prior that can be reused across arbitrary domains, making it like an attempt for a foundation model for physical inversion. The theoretical guarantee that latent-space inference is equivalent to inference in the original field space is elegant and nontrivial, anchoring the method in solid Bayesian footing, rather than just heuristics.

Another secondary (yet distinctive) strength is the combination of MMD-based autoencoding and ABC sampling, which replaces the usual VAE–MCMC pairing with something GPU-parallelizable and architecture-agnostic, making large-scale problems (like the multi-GPU terrain flow case) actually tractable. 

Finally, GABI’s decoupling of learning from the observation process is a rare and important contribution (it may have been proposed before but I am not 100% sure, other reviewers could correct me if I am wrong), most real engineering problems involve messy, variable sensors and noise models, and this paper’s “learn once, observe later” philosophy directly addresses that reality instead of hand-waving around it.

### Weaknesses
The paper seems to lean heavily on the assumption that a single latent space can meaningfully represent diverse geometries, but offers little analysis of when this assumption holds or how latent alignment behaves as geometry complexity scales, especially in high-frequency, discontinuous regimes (e.g., shocks or turbulence).

### Questions
How robust is the learned latent geometry field representation when the target system exhibits physical behaviors outside the training manifold, such as topological changes, discontinuous boundaries, or new forcing regimes? And can GABI’s prior adapt online without retraining to preserve its Bayesian consistency?

### Soundness
3

### Presentation
3

### Contribution
3
