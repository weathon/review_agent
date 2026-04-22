# Causal Score Conditioning for Multi-Resolution Latent Systems

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Complex causal systems with interdependent variables require inference from heterogeneous observations that vary in spatial resolution, temporal frequency, and noise characteristics due to data acquisition constraints. Existing multi-modal fusion approaches assume uniform data quality or complete observability -- assumptions often violated in real-world applications. Current methods face three limitations: they treat causally-related variables independently, failing to exploit causal relationships; they cannot integrate multi-resolution observations effectively; and they lack theoretical frameworks for cascaded approximation errors. We introduce the Score-based Variational Graphical Diffusion Model (SVGDM), which integrates score-based diffusion within causal graphical structures for inference under heterogeneous incomplete observations. SVGDM introduces causal score decomposition enabling information propagation across causally-connected variables while preserving original observation characteristics. Diffusion provides a natural way to model scale-dependent sensing noise, which is common in remote-sensing, climate, and physical measurement systems, while the causal graph encodes well-established mechanistic dependencies between latent processes. We provide theoretical analysis and demonstrate superior performance on both synthetic and real-world datasets compared to relevant baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes the use of causally structured diffusion priors to integrate multi-resolution data. They call the method SVGD, which involves a graphical factorization in latent space, augmenting the drift term of the SDE to encourage multi-resolution integration, and variational inference for inference. An error analysis and experimental results are provided on real and synthetic datasets.

### Strengths
- Use of latent space for multi-resolution observations is sound and seems natural. The way that the multiple frequencies encourage similar latent trajectories seems novel.
- Embedding a graphical model within the diffusion model is an interesting idea.

### Weaknesses
- The motivation behind the methods proposed in this paper are unclear to me. Why are latent graphical models useful for data integration? Why is diffusion the right choice?

- With respect to graphical structure, the authors repeatedly list benefits such as "respecting graphical structure", but there is no mention of how the graphical structure arises. 

- Scalability is given a few times as a benefit of score-based diffusion, but given that the diffusion model is over latent space it doesn't seem like it would be a big issue. 

- The use of PGMs is not inherently causal and thus I find the title and abstract (e.g., relating to causal inference) to be misleading. 

- The paper is very difficult to follow. At many times it is unclear what the overall goal is and what problems are being solved precisely.

### Questions
- Is the end goal of the methods proposed in this paper to infer the shared latents under integrated multi resolution observations? If so why are there no comparisons to other multi-modal methods, e.g., multi-view VAEs?

- The actual model structure is unclear. Are the $z_{p_l}$ variables in Section 3 referring to $z$ at $t=1$? Presumably $t=0$ indicates pure noise. But how can $p_0(z)$ factorise as the causal graph in Theorem 1? Please clarify. 

- Do we need to know the graphical structure a priori? How is this determined for the latent scale?

- Do we need to infer the forward and backward maps $\phi$? Do we run into identifiability issues in doing so?

- In Theorem 1, I'm not sure an SDE with drift only influenced by parents preserves causal structure at all time points. Consider an ODE (i.e., degenerate SDE) and the graph $z_1 \to z_2 \to z_3$ with independent initial condition $p_0(z)$. Then, as we propogate dynamics, even though $z_1$ is not a parent of $z_3$, the indirect effect gives a dependence and the factorisation will be $p_t(z) = p_t(z_1)p_t(z_2 | z_1) p_t(z_3| z_2, z_1)$ which is not the original graph structure. This can be shown with change-of-variables, i.e., the Jacobian of the implied flow has a non-zero term $J_{1,3}$. Could the authors double check this? 

- The authors write that (l99) PGMs fail to model variables with different resolutions or noise levels. I don't see why this has to be the case? PGMs only give a formalism for factorising the joint distribution, it doesn't specify the actual model.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the Score-based Variational Graphical Diffusion Model (SVGD), a novel framework for causal inference on latent systems observed through heterogeneous and multi-resolution observations. The authors address key limitations of existing methods by exploiting the causal structure and multi-resolution data. SVGD integrates score-based diffusion processes with a known causal DAG. The core technical innovation is the causal score decomposition, which leverages the Markov blanket property to enable joint inference and information propagation across causally-connected variables. The framework provides theoretical analysis, including the existence of node-wise causal SDEs and an error cascade analysis for the locally Gaussian approximation used in the causal consistency term. Empirical results on synthetic systems and real-world disaster estimation tasks (earthquakes and wildfires) demonstrate the performance.

### Strengths
Clarity: The paper is generally well-written with a clear structure, though there are some parts not clear to me (please see my questions). 

Originality: The SVGD framework, built around the causal score decomposition, is a original approach to handle systems where data quality is heterogeneous (e.g. varying resolution). The problem itself is highly significant for critical applications like climate modeling and disaster assessment.

Theoretical Rigor: The paper includes a substantial theoretical backbone, proving the existence of the SDE system, showing the approximate preservation of the causal blanket under diffusion, and providing an explicit error analysis.

### Weaknesses
Assumption of Known Causal Structure: The most significant limitation is the reliance on a known causal DAG. In many of the complex systems discussed (e.g., Earth systems), the structure especially involving latent variables is unknown and itself a challenging problem.
As a comparison, some causal representation learning methods can learn the structure among latent variables. This drastically limits the significance of the proposed method.

Limited Synthetic Experiment Complexity: The synthetic experiments, while demonstrating the proof-of-concept, are conducted on a very limited setting (i.e., only three latent variables). The paper also notes that computational complexity scales with the number of causal dependencies. More complex synthetic experiments (e.g., more than 10 latent variables with different sparse and dense graphs) are needed to convincingly validate the proposed method and the computational scalability. Given that the method already assumes the structure is given, a synthetic experiment with only three latent variables looks far from sufficient.

### Questions
1. Table 2 "SVGDM" should be "SVGD"? Plus, it looks like $\phi$ in the problem formulation part (line 135)  has different meaning with $\phi$ in eq 2? If yes please clarify.

2. Given that the assumption of a known causal structure is the primary weakness, could the authors propose a concrete future direction for jointly learning the causal DAG G alongside the diffusion model parameters? 

3. The observation model assumes additive Gaussian noise. How does the method perform empirically when this assumption is violated? Some empirical results would help.

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
2

### Summary
This paper proposes SVGD, a framework that combines diffusion process and causal graphical models. Techinically, this paper (i) formulates the node-wise forward and reverse process with causal relations, (ii) introduce a causal score decomposition based on "causal blanket", (iii) estimate the marginal distribution with continuous‑time DSM and causal term via a locally Gaussian conditional model. Experiments on synthetic and real-world data report consistent gains over baseline models.

### Strengths
1. The motivation is clear and well-demonstrated. The idea to model the diffusion process with modularity is simple and intuitive. The combination of graphical‑model locality with score‑based diffusion through a causal score decomposition is, to my knowledge, novel and significant.
2. The loss functions are closely related to the theorem. The learning recipe is concrete. Besides, a cascade error bound and plus convergence under error control are provided.
3. The exeperiments on both synthertic and real-world applications achieves a better performance.
4. Codes are provided.

### Weaknesses
1. Some ablation studies and component analysis can better attribute gains, i.e., if some of the loss terms are ignored.
2. Appenix C provides a "COMPUTATIONAL SCALABILITY ANALYSIS" with an O(Nd) complexity. Please provide measured runtimes vs. N and d , and GPU memory scaling, to substantiate these claims.
3. The model requires a known causal graph. In the real world experiments, where does the prior come from? Moreover, what will happen if the causal graph is partially wrong? Can you conduct an experiment to analyse the sensitivity?
4. Theoretical validity (Theorem 3) requires local log‑concavity and bounded Tweedie error. In strongly non‑Gaussian regimes (heavy tails/multimodality), how robust is it?
5. The abbreviation is the same as Stein Variational Gradient Descent [1].

[1] Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm

### Questions
In Sec. 3.3, the author claims "reverse SDE defines a normalizing flow from noise to data", which introduce a deterministic flow. However, in the appendix, the author uses "reverse SDE with predictor–corrector" and "DDIM-style update", which is not determinstic. Could the author explain this confliction?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposed the Score-based Variational Graphical Diffusion Model (SVGD), to address inference challenges in complex systems under heterogeneous and incomplete observations. By embedding score-based diffusion processes within causal graphical models, SVGD leverages a causal score decomposition mechanism to facilitate efficient information propagation across causally interdependent variables while maintaining the fidelity of the original observational data. Theoretical analysis and empirical evaluations on synthetic and real-world datasets demonstrate its superior performance compared to state-of-the-art baseline approaches.

### Strengths
1. The paper proposes the Score-based Variational Graphical Diffusion Model (SVGD), which addresses the gap in inference under heterogeneous and incomplete observations in existing approaches.
2. The authors validated the reliability and effectiveness of the model through theoretical analysis.

### Weaknesses
1. The problem addressed in the paper is based on a key assumption: "the causal DAG is known," and the task is "to perform inference based on the known causal structure," rather than "learning the causal structure." However, the title only mentions "Causal Inference," without explicitly stating "based on a known causal structure," which may lead readers to mistakenly believe that the research problem is about "learning causal structures from data and performing inference."
2. The model assumes that the causal DAG is completely known, but in most real-world scenarios, the causal structure often needs to be inferred first. If the causal structure is unknown, the applicability of the model may be limited, as it relies on a predefined causal DAG. 
3. Although the authors provide theoretical analysis, it relies on assumptions such as "log-concavity of conditional distributions" and "additive Gaussian observational noise." If these assumptions are not satisfied, the proposed method in the paper would not be applicable.
4. Some notations in the paper lack definitions. For example, in Eq. (1), what does $W_i$ mean?

### Questions
See the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2
