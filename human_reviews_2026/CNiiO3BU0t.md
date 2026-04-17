# When do World Models Successfully Learn Dynamical Systems?

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
In this work, we explore the use of compact latent representations with learned time dynamics ('World Models') to simulate physical systems. Drawing on concepts from control theory, we propose a theoretical framework that explains why projecting time slices into a low-dimensional space and then concatenating to form a history ('Tokenization') is effective at learning dynamical systems, and characterise when the underlying dynamics admit a reconstruction mapping from the history of previous tokenized frames to the next, with the key novel insight being observability. To validate these claims, we develop a sequence of models with increasing complexity, starting with least-squares regression and progressing through simple linear layers, shallow adversarial learners, and ultimately full-scale generative adversarial networks (GANs). We evaluate these models on a variety of datasets, including modified forms of the heat and wave equations, the chaotic regime 2D Kuramoto-Sivashinsky equation, and a challenging computational fluid dynamics (CFD) dataset of a 2D Kármán vortex street around a fixed cylinder, where our model is successfully able to recreate the flow. Comparisons to FNO and DeepONet show comparable short- and improved long-term accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes using control theory concepts, specifically observability, to understand when world models can successfully learn physical dynamical systems. The authors develop a theoretical framework connecting tokenization to observability, then validate it with experiments ranging from simple linear PDEs through chaotic Kuramoto-Sivashinsky equations to turbulent cylinder flow simulations.


The core idea is interesting: they argue that world models work when the underlying system is "observable" - meaning you can reconstruct the full state from a history of tokenized observations. They prove some PAC learning results and demonstrate this with GANs of increasing complexity.

### Strengths
- Connecting observability from control theory to world model learnability is genuinely original. I haven't seen this angle before in the ML for physics literature, and it's a useful lens for thinking about why some systems are easier to learn than others. The intuition that you need sufficient observability to learn dynamics makes sense.

- I appreciate the methodical approach - starting with least squares, moving to linear layers, then shallow adversarial networks, finally full GANs. This helps isolate what's driving performance. The choice of numerical methods is also sound (ETDRK4 for KS equation, LES for turbulence).

- The analysis of linear systems using Kalman observability matrices (Theorem 3) was well-executed. The results about the history length needing to exceed the compression factor make sense and are properly validated.

### Weaknesses
- Here's my biggest concern: In Theorem 6, the authors prove that standard heat and wave equations with constant coefficients are NOT observable under their patch-averaging tokenisation scheme. The solution? Modify the Laplacian to have position-dependent stochastic coefficients ∇·(a(x)∇u).

If I'm not mistaken, this is an issue. Heat equations ARE observable in the standard control theory sense - you can reconstruct the state from boundary measurements or interior point sensors. The issue is that their specific tokenization loses high-frequency information.

All the linear PDE experiments use these modified equations. This isn't validation - it's engineering problems to fit the theory. 

It is interesting in itself, but the paper should discuss those limiations.

- The paper cites FNO and DeepONet extensively in the introduction and related work. But there are ZERO quantitative comparisons in the results. Not one number comparing performance.

- You only use 28 training initial conditions for a chaotic PDE. Isn't it a bit too small?

- I'm less sure about that, but your theoretical framework assumes deterministic reconstruction (continuous map G: Y^k → X). But your experiments use stochastic GANs that learn distributions. GANs optimise distribution matching, not pointwise reconstruction. They can have mode collapse, generate diverse samples, etc.
There's a fundamental mismatch between your deterministic theory and your stochastic experiments. The GAN's success doesn't necessarily validate your observability framework?

### Questions
- What happens if you test on unmodified heat equations?
- How does your method compare quantitatively to FNO and DeepONet? 
- How do you resolve the mismatch between deterministic theory and stochastic GANs?

### Soundness
3

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
This paper provides a theoretical framework to study "when do `world models` successfully simulate physical systems", using approaches from control theory. It defined the notion of "observability" of a system, and ties it to its learnability-- namely the ability to recover the full state c from a history of low-dimensional "tokenized" observations. Experiments on linear PDEs, chaotic equations, and turbulent flows validate this, showing the model outperforms neural operators like FNO and DeepONet in long-term stability.

### Strengths
- **Originality**: The paper's main strength is in merging concepts from diverse fields -- linking concepts from control theory, "observability" to generative "world models" using , and further framing the latter as operator learning. In doing so it provides a framework that allows deeper insight on the expected performance of world models.

- **Quality**: The work is of good quality, providing rigorous theoretical grounding alongside empirical validation. The theory is formally presented with clear definitions and theorems, with proofs available in the appendix. The experiments span diverse linear and non-linear systems. The quantitative evaluation support claims on performance and robustness.

- **Clarity**: The paper is well-written and logically structured. Providing the necessary background, definitions and theoretical foundation, towards the experimental evaluation. 

- **Significance**: The impact of this work is in providing a  theoretical handle to analyze world models-- guidance when would generative models for physical simulation be accurate.

### Weaknesses
- **Assumptions / scope of theory**: The main analysis relies on "global observability" and the existence of a continuous inverse 
map $G$. A strong condition; which as indicated by the authors in nonlinear PDEs it is rarely checkable, following this the PAC theorems are qualitative. 
- **Circular validation**: Following the previous comment, the authors specifically state that observability for the full turbulent flow is unknown and thus rely on the model's success as "experimental evidence". This creates a circular loop--the theory is supposed to explain success, but success is used to infer back the theory (the observability).
- **Broader Impact / Applications**: While the general appeal of the approach is clear the paper is lacking a discussion or presentation of possible applications or use cases beyond accuracy measurements.

### Questions
The following questions are wrt to the above weaknesses:
- **Assessing observability**: The key contribution is the link between observability and model performance. However, proving observability for complex non-linear systems is often intractable. To avoid the "circular" validation would it be possible to derive diagnostics to estimate whether a system is "observable enough"?
- **Broader impact**:  
(1) Beyond the provided metrics, showcasing improved performance, could you elaborate on specific, practical applications where the framework could provide a unique advantage / contribution.
(2) Could you elaborate more broadly on the contribution and significance of this work.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors investigate when world models, i.e. latent generative models with autoregressive temporal dynamics, can successfully learn the behavior of dynamical systems. They frame these models within a control-theoretic context, so that observability determines whether the true system dynamics can be recovered from tokenized latent representations. Using both a novel PAC theoretical apparatus and a series of experiments, they show that a finite-history autoregressive map exists and can be learned only if the system is observable from its latent outputs. Empirical tests on linear and nonlinear PDEs, including the heat, wave, and Kuramoto–Sivashinsky equations, demonstrate that increasing the autoregressive history length improves prediction accuracy up to the point where observability is achieved. The authors further show that their world-model framework, implemented using a hierarchy of increasingly complex architectures culminating in video GAN models, produces long-horizon stable simulations of fluid and chaotic systems and outperforms neural operator baselines such as FNO and DeepONet in stability and efficiency.

### Strengths
1. To my knowledge, the theoretical PAC formulation of these latent, autoegressive models is new contribution to data-driven dynamical systems. The authors proceed very systematically and the rigor and clarity of their theoretical framework and experimental results is well-received. The empirical predictions the authors make about when models will be easily learned or not are strong and elegant. 

2. Benchmarking against other methods in terms of performance and complexity is very thorough.

### Weaknesses
1. The principal weakness of this paper is its rhetorical presentation. In particular, introduction makes it very hard to understand what problem the authors want to solve and how their approach differs from other methods. After reading, it becomes clear that they seek to address the lack of applications of auto-regressive, latent-space models (i.e. world models) to physical systems rather than video generation, particularly with rigorous guarantees of performance based on observability. Furthermore, the authors refer mid-way through to "our model", but it is unclear at this point what the "model" is since all architectural details are relegated to the (extremely detailed) appendix. It seems as if what the authors refer to as their "model" is indeed the autoregressive approach itself, independent of the implementing architecture, which varies by application. If this is indeed their contribution, it must be stated much more clearly. It took me a while to realize in retrospect that, indeed, most forecasting models for dynamical systems take a one-step approach, predicting x_t+1 from x_t with a detail through latent space (Champion et al., 2019 PNAS; Vlachas et al. 2022 NMI ). Again, this is a rhetorical weakness, but it has the result of destabilizing and disorienting the reader. The motivation and material contribution must be stated exactly. 

2. Comparison to previous world learning/autoregressive models for physical systems is weak. The authors mention work by Skorokhodov and Klemmer and say that "they neither analyse system-theoretic foundations nor compare with operator-learning approaches". But why is that a fatal flaw? In that sense, are you arguing that your main contribution is the PAC theoretical work and operator learning benchmarking? Or do you offer a performance boost as well? Also, from some brief googling, I found: (1) https://www.nature.com/articles/s41598-024-68944-0? and (2) https://eurasip.org/Proceedings/Eusipco/Eusipco2022/pdfs/0002216.pdf?. These both seem like autoregressive models for physical systems. I'm not saying you have to compare directly to these, but rather that I, as a reader, was left a bit confused as to what was novel between your approach and these earlier methods. 

In summary, this is a technical tour de force, but its concrete novelty and "problematic" must be emphasized.

### Questions
Summarizing based on the weakness section:

1. What are your concrete contributions and what problem are you solving? 

2. Assuming those contributions are not 100% novel, how do they differ from previous approaches? For example, you cite a lot of interesting work in the Related Work section, but nowhere do you say how those approaches differ from or are weaker than what you propose. This makes the paper seem like a very dry, technical report as oppose to a novel contribution.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes to study when world models can successfully learn dynamical systems. The authors claim a system-theoretic and PAC-learning framework linking observability to learnability and present numerical experiments on simple PDEs (heat, wave, KSE) and a CFD case (vortex shedding).

### Strengths
The paper raises an interesting conceptual question connecting world models and operator learning.

The experiments cover several standard PDE benchmarks and attempt to bridge control theory and data-driven modeling.

### Weaknesses
1. The paper does not answer “when world models succeed” in any rigorous sense. The so-called theoretical results restate standard observability conditions and a generic PAC-learning existence theorem, without offering new criteria or quantitative conditions. The main theorems are textbook results in nonlinear systems (Kalman, Hermann–Krener) and elementary PAC learnability statements. 

2. It is not clear why the models used are “world models.” They are standard autoregressive video predictors applied to PDE snapshots,

3. The “observability” discussion is never quantitatively linked to training outcomes; the perturbation of coefficients in the heat equation cannot be interpreted as testing observability. The theoretical part is largely decorative.

4. The paper does not clarify what fundamentally distinguishes its proposed model from standard neural operator methods such as FNO or DeepONet. From the reviewer’s perspective, there are at most two apparent differences: 1. It applies an autoencoder to learn latent dynamics. 2. It includes memory by feeding the entire trajectory history instead of the final frame.
However, both aspects can be trivially incorporated into baseline operator-learning models, and the paper does not explain why these would yield fundamentally different behavior. Moreover, the implementation details of FNO and DeepONet are not described, and the extent of ablation studies is unclear. Hence, it is difficult to assess whether the observed improvements are due to architectural choices or merely training conditions.

5. The numerical comparisons with FNO and DeepONet are limited to table-level metrics without controlling for architecture size, data resolution, or rollout time. Claims of “superior long-term stability” lack statistical justification, and no variance or ablation results are provided.

### Questions
1. The presentation of Definition 1 is unclear. Specifically, the function \( y \) appears to depend only on \( t \) in equation (2), whereas in Definition 1 it is defined as a function of both \( t \) and \( x \). This discrepancy should be clarified to avoid confusion regarding the role of each variable.

2. The definition of G is missing. Since G appears to play a role in the formulation or analysis, its precise meaning and mathematical structure should be clearly stated to ensure the reader can follow the development and verify the results.

3. The manuscript should demonstrate that the set of defined observable pairs is nonempty. Without such a justification, it remains unclear whether the proposed framework admits any valid instances.

4. The study is presented as theoretical but is mainly empirical; it does not develop new algorithms or theory, nor does it analyze failure modes beyond trivial observability remarks.

### Soundness
2

### Presentation
2

### Contribution
2
