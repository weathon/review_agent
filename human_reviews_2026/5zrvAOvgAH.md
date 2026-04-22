# Governing Equation Discovery from Data Based on Differential Invariants

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
The explicit governing equation is one of the simplest and most intuitive forms for characterizing physical laws. However, directly discovering partial differential equations (PDEs) from data poses significant challenges, primarily in determining relevant terms from a vast search space. Symmetry, as a crucial prior knowledge in scientific fields, has been widely applied in tasks such as designing equivariant networks and guiding neural PDE solvers. In this paper, we propose a pipeline for governing equation discovery based on differential invariants, which can losslessly reduce the search space of existing equation discovery methods while strictly adhering to symmetry. Specifically, we compute the set of differential invariants corresponding to the infinitesimal generators of the symmetry group and select them as the relevant terms for equation discovery. Taking DI-SINDy (SINDy based on Differential Invariants) as an example, we demonstrate that its success rate and accuracy in PDE discovery surpass those of other symmetry-informed governing equation discovery methods across a series of PDEs. Additional results further indicate that our method exhibits strong robustness to data and symmetry noise, as well as significant potential for solving high-dimensional dynamic systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a differential-invariant-based technique (DI-) to leverage symmetry for PDE discovery. The method can be plug-and-play for various PDE discovery approaches. SINDy is taken as the major baseline for the proposed DI-SINDy. The method works by replacing the standard derivative terms in the PDE dictionary with the differential invariants derived from a known/discovered Lie group of symmetries for the PDE. The authors provide theoretical justification that this transformation to an invariant dictionary is lossless. Experiments on several classic PDE discovery datasets demonstrate superiority of the proposed method.

### Strengths
1. The method is well-grounded in the established mathematical theory of Lie groups and differential equations, providing a strong theoretical justification for its design.
2. A pseudo-algorithm and a figure are provided to make the overall pipeline clear to readers.
3. The experimental evaluation is systematic, and the results are well-organized and easy to interpret.

### Weaknesses
1. The paper is hard to read. The paper's theory part relies heavily on the textbook by Olver. However, the paper doesn't summarize the idea/definition clearly enough. I have to refer to the textbook to get a clear understanding for the general flow. I believe it is crucial that the author provides sufficient intuition at the beginning of the theory section for a non-expert audience. There are also typos which hinder understanding. Equation 5 from my understanding is only combining $\eta$ and $pr^{(n)}v$. However you refer to Equation 5 as related to infinisimal criterion. Then we should have $pr^{(n)}v [\eta] = 0$ instead of the original equation. 
2. The paper's main idea is a direct application of Olver's theory. However, the soundness of the method is based on a priori knowledge of the PDE's Lie group of symmetries. In realistic scenarios, these symmetries are unknown or only approximate. The authors briefly mention combining with symmetry-discovery methods to tackle this issue, but no such integration or robustness analysis is provided.

### Questions
1. Given that the "known symmetry" assumption is the primary limitation, would it be possible that some experiments on datasets with unknown symmetries be provided? Furthermore, the calculation of differential invariants is also a time-consuming/hard procedure for those problems with unknown symmetry. An analysis of the complete pipeline, including the computational cost of finding invariants, is needed to evaluate the method's true effectiveness. I would be more than happy to raise my score if this question is addressed. 
2. How does the DI-SINDy framework scale to high-dimensional PDEs, such as the 2D NS equations? The examples presented are relatively low-dimensional. The complexity of finding symmetries and constructing a complete basis of differential invariants seems likely to grow significantly with the number of variables.
3. The appendix provides experiments on noisy data, which is good. As an extension of Question 1, could the authors attempt to perform symmetry discovery directly on the noisy data and then feed the resulting generators into DI-SINDy? This would test the robustness of the entire end-to-end pipeline in a challenging, realistic setting.
4. The author also mention in the paper that compare to EquivSINDy, DI-SINDy can handle nonlinear cases. Could the authors please clarify where this advantage is demonstrated in the experiments? A more direct comparison highlighting this specific capability would be beneficial.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This is actually a very nice paper that integrates concepts of invariants, in particular differential invariants, into account in a model discovery architecture.  

Importantly, the authors are dealing with an important issue of leveraging symmetry ideas into a better approach for model discovery.

My biggest criticism is simple (and universal in my reviews):  How well does this hold up with real data and noise?  The examples are on what appears to be clear numerical data.  However, real mode discovery is always embedded with noise and there are a great number of architectures that completely fail with even a little noise.  This is not a trivial criticism.... .it is quite important that the algorithm hold up under not just weak noise, but modest levels of noise or it is more of a mathematical interest than anything one would deploy in practice.

### Strengths
A strong concept for model discovery if it holds up under more realistic conditions.

### Weaknesses
There seems to be no noise study on the method a the method itself really does have to be evaluated on this idea.  More broadly, there are works that are pretty similar, even tough this is a new architecture, but perhaps for a conference like ICLR a higher level of distinction of novelty is required in comparison with other methods.

### Questions
My question is repeated again:  When does this fail under noise?  It needs to hold up for at least a modest amount in order for it to be relevant.

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
The paper proposes a partial differential equation(PDE) discovery process based on differential invariants, aiming at expressing the equation term search space without losses through a symmetrical prior. The main idea is to use infinitesimal generators of the symmetry group to calculate the prolonged term and differential invariants as the skeleton of the equation terms. With the differential invariants basis, the algorithm takes advantage of SINDy to accomplish sparse regression. The DI-SINDy proposed by the authors has shown advantages of a high success rate, strong robustness, long-term forecast stability through several numerical experiments, such as KdV, KS, Burgers, and nKdV PDEs.

### Strengths
1. The algorithm systematically introduces the theory of differential invariants into data-driven equation discovery, demonstrating significant theoretical depth and methodological innovation.
2. The method is built upon a solid mathematical foundation, ensuring strict adherence to symmetry constraints.
3. It features a plug-and-play nature that allows for integration with existing discovery methods, offering strong practicality and exhibiting strong robustness against noise and symmetry deviations.

### Weaknesses
1. The method's performance is highly dependent on the accuracy of the symmetry information. For complex systems where the symmetries are unknown or difficult to discover, the practical applicability of the method may be limited.
2. The discussion is confined to Lie point symmetries. Compared to a broader perspective on symmetry in machine learning (e.g., as explored in works like "Machine-learning hidden symmetries" by Ziming Liu and Max Tegmark, which deals with Translation invariance, Lie invariance, equivariance, Hamiltonian structure, etc.), the scope of symmetries considered here is narrow.
3. The experimental comparisons are primarily conducted against SINDy-based models. A more comprehensive evaluation is needed, including comparisons with variants of Physics-Informed Neural Networks (PINNs) and other neural-based symbolic regression methods.
4. The DI-SINDy approach assumes that the final governing equation can be expressed as a linear combination of the identified differential invariants. This may limit its expressiveness and generalization ability for systems governed by more complex, non-linear relationships between the invariants. There is a lack of theoretical justification for this linearity assumption, and supporting experiments on equations requiring intricate non-linear combinations of invariants are missing.
5. The paper contains insufficient machine learning content and is recommended for submission to mathematics-focused conferences/journals specializing in PDE research.

### Questions
1. Could the framework be extended to incorporate a broader range of symmetries beyond Lie point symmetries, such as those explored in works like "Machine-learning hidden symmetries" (e.g., Hamiltonian structure, gauge invariance)?
2. How does the proposed method compare against other AI-driven paradigms for equation discovery, such as various Physics-Informed Neural Network (PINN) variants or other neural symbolic regression approaches?
3. Can the method's effectiveness be demonstrated on PDEs with stronger, less linear-like non-linearity, or be generalized to the inference of equations in higher-dimensional, coupled systems?
4. Have the authors considered modernizing the core SINDy module itself, for instance, by replacing it with a more expressive symbolic regression model like a Transformer or genetic programming to discover complex combinations of the differential invariants?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a symmetry-guided framework for governing equation discovery based on differential invariants. The key idea is to embed known (or discovered) symmetries into the equation discovery process by constructing invariant feature libraries derived from the symmetry group’s infinitesimal generators. This approach effectively reduces the search space without sacrificing expressiveness or correctness. The authors formalize this through a completeness proposition and implement the framework within SINDy (resulting in DI-SINDy). Empirical results on several canonical PDEs (e.g., KdV, Burgers, KS, nKdV) show improved identification accuracy, success rates, and long-term prediction stability compared to baseline methods such as SINDy and EquivSINDy.

### Strengths
This paper offers a theoretically sound and elegant contribution to the field of equation discovery. By leveraging differential invariants derived from known or learned symmetries, the authors present a clear and general framework that can be applied to improve many existing methods such as SINDy. The theoretical foundation, grounded in classical Lie symmetry analysis, is both rigorous and well-integrated into the machine learning context.
The conceptual innovation of embedding symmetry directly into the function library through invariant construction represents a meaningful advance over prior symmetry-regularized approaches. Importantly, the proposition ensuring lossless expressivity gives the method strong theoretical credibility. From a presentation standpoint, the paper is well-written and visually clear. The introduction and related work are accessible, providing an intuitive motivation for readers from both applied mathematics and ML backgrounds. Figures, particularly the main pipeline diagram, effectively communicate the overall workflow. The results section is also strong: quantitative tables and rollout plots convincingly demonstrate that DI-SINDy outperforms baselines across multiple PDE benchmarks.

### Weaknesses
Despite its theoretical depth, the paper’s empirical scope is somewhat narrow. The experiments are limited to one-dimensional canonical PDEs such as Burgers, KdV, and KS, leaving open questions about scalability to higher-dimensional or real-world systems. The claim of computational efficiency is also not empirically validated, while the method does reduce the search space, the additional overhead of computing differential invariants is not benchmarked. A runtime or complexity analysis would help substantiate this claim.
Furthermore, the mathematical exposition can be dense, particularly in the sections involving infinitesimal generators and prolongations. Readers without a background in Lie group theory may find these derivations difficult to follow. Including a small, concrete example would greatly improve accessibility. Finally, the presentation of results tables could be clearer: key tables are introduced without sufficient context, requiring readers to read ahead before fully understanding the metrics. Overall, while the theoretical foundation is excellent, the paper would benefit from more diverse experiments, clearer runtime validation, and slightly improved accessibility in the methodological sections.

### Questions
1) How does DI-SINDy perform when the input symmetries are incomplete or slightly mis-specified?

2) Have you evaluated the runtime trade-offs between computing differential invariants and the reduction in search-space size?

3) Can this framework be extended to approximate or data-driven symmetries, where invariance holds only approximately?

4) How does noise in the symmetry generators or dataset affect performance, and can robustness results from the appendix be summarized in the main text?

### Soundness
3

### Presentation
3

### Contribution
4
