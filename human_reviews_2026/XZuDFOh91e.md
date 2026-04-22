# Reward-Guided Flow Merging via Implicit Density Operators

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 8, 2

## Abstract
Unprecedented progress in large-scale flow and diffusion modeling for scientific discovery recently raised two fundamental challenges: $(i)$ reward-guided adaptation of pre-trained flows, and $(ii)$ integration of multiple models, i.e., model merging. While current approaches address them separately, we introduce a unifying probability-space framework that subsumes both as limit cases, and enables reward-guided flow merging. This captures generative optimization tasks requiring information from multiple pre-trained flows, as well as task-aware flow merging (e.g., for maximization of drug-discovery utilities). Our formulation renders possible to express a rich family of implicit operators over generative models densities, including intersection (e.g., to enforce safety), union (e.g., to compose diverse models) and interpolation (e.g., for discovery in data-scarce regions). Moreover, it allows to compute complex logic expressions via generative circuits. Next, we introduce Reward-Guided Flow Merging (RFM), a theory-backed mirror-descent scheme that reduces reward-guided flow merging to a sequential fine-tuning problem that can be tackled via scalable, established methods. Then, we provide first-of-their-kind theoretical guarantees for reward-guided and pure flow merging via RFM. Ultimately, we showcase the capabilities of the proposed method on illustrative settings providing visually interpretable insights, and apply our method to high-dimensional de-novo molecular design and low-energy conformer generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
My expertise is in AI for Science, not core AI theory. Therefore, this review primarily assesses the paper's practical application, experimental validation, and its claimed contribution to scientific tasks. I have focused less on the fundamental methodological novelty in the pure AI domain, deferring that judgment to other reviewers.

This paper presents a novel, theoretically-grounded framework for unifying reward-guided optimization with flow model merging. The concept is well-motivated and clearly demonstrated on multiple well-structured toy examples.

However, the empirical validation on the high-dimensional task is limited. There is only one application task. Moreover, the core reward-based optimization (RFM-RB) fails, performing significantly worse than a simpler baseline in both key metrics (energy and validity).
These results restrict the method's practical utility.

In summary, while the paper introduces a promising and theoretically sound framework, its practical effectiveness is not yet thoroughly validated due to limited experimental results on high-dimensional tasks.

---
**The usage of LLM**: I wrote the entire review myself and only used the LLM to correct the grammar and improve readability.

### Strengths
1. **Unified Framework:** A primary strength is the novel formulation that unifies reward-guided optimization and flow merging into a single objective function. This framework is supported by a solid theoretical background and demonstrated with diverse operators and toy examples (e.g., intersection, union, interpolation).
2. **Clear Motivation and Methodology:** The paper presents a clear motivation and a well-articulated methodology. This is effectively explained in Sections 3 and 4 and well-illustrated through several intuitive toy examples.
3. **Demonstrated Feasibility on High-Dimensional Data:** The authors successfully demonstrate the feasibility of their flow merging approach on a high-dimensional task (molecular generation) that involves both discrete (molecular graph) and continuous (3D coordinates) data.

### Weaknesses
The paper effectively demonstrates its motivation and capabilities through several well-designed toy examples. However, its validation on high-dimensional tasks appears insufficient.

**1. Significant performance degradation on optimization setting**

A core claim of the paper is to _simultaneously_ optimize a reward function while merging multiple flows.
> **Page 1, Line 46**: *Can we fine-tune a pretrained flow model to optimize a given reward function while integrating information from (i.e., merge) multiple pre-trained flows.*

However, the empirical evidence for the reward-guided optimization component is weak and concerning.
- The primary optimization experiment (RFM-RB) is not presented in the main manuscript and is only available in Appendix F.2.
- The results in Table 1 show that RFM-RB fails to outperform the PRE-2 baseline (which was trained with standard Adjoint Matching). In fact, RFM-RB performs significantly worse: it achieves a worse mean total energy (−12.47 Ha) compared to the baseline while suffering a significant drop in molecular validity (from 76% to 33%).

The authors attribute this performance degradation to the multi-objective nature of molecular design.
However, this explanation is insufficient, as the adjoint matching (PRE-2) demonstrates a better trade-off, improving energy significantly (−14.76 Ha) with a minor validity drop (from 76% to 68%).

This poor performance undermines the paper's central claim.
The authors must provide additional evidence to validate that RFM can practically solve reward-based optimization problems beyond its theoretical guarantees.

**2. Limited Application Study**

The paper is limited to a single application: a molecular discovery task.
This contrasts with other RL-based finetuning methods, which are often validated on well-established tasks (e.g., image generation) with diverse baselines[1] or across multiple domains[2,3].


**3. Ambiguity of experiment name**

There is a significant ambiguity in the naming of the high-dimensional experiment that is likely to mislead domain experts.
The authors repeatedly describe the task as "conformer generation" including in the abstract.

However, this task is **3D molecular generation** (i.e., jointly generating the molecular graph and 3D coordinates).
This is fundamentally different from **molecular conformer generation**, which involves generating 3D coordinates for _given_ molecule(s).

This misnaming caused significant confusion. I strongly recommend the authors correct this terminology throughout the manuscript to accurately reflect the experimental task.
- **3D molecular generation:** Joint design of molecular graph (categorical) and its 3D conformer (continuous). (e.g., FlowMol[4], SemlaFlow[5], CGFlow[6])
- **Molecular conformer generation:** 3D conformer generation of a _given_ molecule. (e.g., Torsional Diffusion[7], ETFlow[8])

---
**Reference**
1. Domingo-Enrich, Carles, et al. "Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control." _arXiv preprint arXiv:2409.08861_(2024).
2. Venkatraman, Siddarth, et al. "Amortizing intractable inference in diffusion models for vision, language, and control." _Advances in neural information processing systems_ 37 (2024): 76080-76114.
3. Venkatraman, Siddarth, et al. "Outsourced diffusion sampling: Efficient posterior inference in latent spaces of generative models." _arXiv preprint arXiv:2502.06999_ (2025).
4. Dunn, Ian, and David Ryan Koes. "Mixed continuous and categorical flow matching for 3d de novo molecule generation." _ArXiv_ (2024): arXiv-2404.
5. Irwin, Ross, et al. "SemlaFlow--Efficient 3D Molecular Generation with Latent Attention and Equivariant Flow Matching." _arXiv preprint arXiv:2406.07266_ (2024).
6. Shen, Tony, et al. "Compositional Flows for 3D Molecule and Synthesis Pathway Co-design." _arXiv preprint arXiv:2504.08051_ (2025).
7. Jing, Bowen, et al. "Torsional diffusion for molecular conformer generation." _Advances in neural information processing systems_ 35 (2022): 24240-24253.
8. Hassan, Majdi, et al. "Et-flow: Equivariant flow-matching for molecular conformer generation." _Advances in Neural Information Processing Systems_ 37 (2024): 128798-128824.

### Questions
- What do the authors perceive as the primary limitations of the proposed RFM approach?
    
- Questions related to the molecular design task:
    - **Figure 3 and Table 1**: What do `RFM-B` and `RFM-UB` stand for? I assume those mean "balanced" and "unbalanced," which are mentioned in the text, but this is not explicitly defined in the figure/table captions.
    - **Page 23 Line 1231**: What reward function was used to obtain the PRE-2 model via AM? Was the exact same reward function used for the RFM-RB experiment?
    - What was the initial model ($\pi_\text{init}$) used for the RFM-B, RFM-UB, and RFM-RB experiments? Was it PRE-1 or PRE-2?
        
- Minor Typos/Formatting:
	- **Page 2, Line 82**: Could you change the citation format from "$p_{data}\text{Lipman et al. ...}$" to "$p_{data}~(\text{Lipman et al. ...})$"?
	- **Page 5, Line 236**: The hyperlink for the equation 8 appears to be missing.
    - **Page 22, Line 1163 (in Algorithm 3, step 7)**: The objective function is referred to as '??'. MOreover, the equation number is missing.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces Reward-Guided Flow Merging (RFM), a unified framework that jointly addresses reward-guided adaptation of pre-trained flows and integration of multiple models. The method formulates merging as optimization over diverse implicit density operators, such as intersection, union, and interpolation. RFM employs a mirror-descent scheme that converts complex merging tasks into sequential fine-tuning problems. The paper presents theoretical proofs as well as experiments on the drug design task.

### Strengths
The paper presents a unified theoretical framework that generalizes two difficult problems in AI for science, reward-guided fine-tuning and model merging. This framework will benefit the design of many scientific models. The paper also provides rigorous proofs of theoretical guarantees for the proposed Reward-Guided Flow Merging algorithm, providing a reliable foundation for the framework.

### Weaknesses
The paper only evaluates its framework on the molecular design task. This limits the demonstration of the claimed wide real-world applications. More experiments in future works would strengthen the generality and practical impact of the algorithm.

### Questions
The paper provides an effective framework for merging pre-trained models. Could the authors discuss more about how model performance and efficiency scale as the number of merged models increases?

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
This work proposes a generative optimisation framework that advocates task-aware reward-guided adaptation of multiple pretrained flow models. The proposed formulation entails implicit density operators (union, intersection, interpolation, and their combinations) over generative model densities. The authors also implement a mirror-descent scheme to approximate the objective in terms of a sequence of reward-guided fine-tuning problems. 

nprecedented progress in large-scale flow and diffusion modeling for scientific discovery recently raised two fundamental challenges:  reward-guided adaptation of pre-trained flows, and integration of multiple models, i.e., model merging. While current approaches address them separately, we introduce a unifying probability-space framework that subsumes both as limit cases, and enables reward-guided flow merging. This captures generative optimization tasks requiring information from multiple pre-trained flows, as well as task-aware flow merging (e.g., for maximization of drug-discovery utilities). Our formulation renders possible to express a rich family of implicit operators over generative models densities, including intersection (e.g., to enforce safety), union (e.g., to compose diverse models) and interpolation (e.g., for discovery in data-scarce regions). Moreover, it allows to compute complex logic expressions via generative circuits. Next, we introduce Reward-Guided Flow Merging (RFM), a theory-backed mirror-descent scheme that reduces reward-guided flow merging to a sequential fine-tuning problem that can be tackled via scalable, established methods. Then, we provide first-of-their-kind theoretical guarantees for reward-guided and pure flow merging via RFM. Ultimately, we showcase the capabilities of the proposed method on illustrative settings providing visually interpretable insights, and on a high-dimensional drug design task generating low-energy molecular conformers.

### Strengths
--- The manuscript is generally easy-to-read. 

--- Model composition and fine-tuning are clearly, and justifiably, extremely active areas of research, so the work deals with a topical subject.

### Weaknesses
I apologise in advance for what might come across as a rather disappointing/critical review for the authors, but I put in extensive effort on reviewing this paper to the best of my ability and knowledge of the field. 

— What the authors call 'merging' is typically referred to as composition in the literature. An entire body of work has been dedicated to composition  (including under constraints and/or using density operators including based on logical And-Or-Not Operators, Ito operators, Feynman-Kac etc.) that has been completely sidestepped in the current work. See, e.g., references [1, 2, 4, 5, 6, 7, 8, 9]. Not only should the work have been contextualised in terms of similarities and differences with this prior work, but comprehensive empirical benefits over them should have been shown. 

There are other works such as [3] which being contemporaneous are not subject to this criticism.  

-- For a technical/optimisation perspective, "reward guidance" is not central to the work and comes across as rather peripheral/contrived since the solution to the formulation only cares about the convexity/concavity of the problem and availability of the function gradient (including, or without, the reward). Similarly, the generality of "merging" in the current context is overstated. As the formulation is restricted to an affine/convex combination of divergences, it cannot implement important operations such as negation and contrast. As soon as the overall concavity/convexity is violated, no global convergence guarantee holds as the classical stochastic approximation theory (Robbins-Monro style updates) can only guarantee a local solution. 

--- The entire framework is essentially a straightforward amalgamation of existing ideas, and it's unclear how this work advances the field at all. In particular, heavily derives from/relies on key notions and machinery already tackled in the literature on RL under general utilities and stochastic optimal control [10, 11, 12, 13].  

--- Experiments are also underwhelming, with essentially no meaningful comparisons included against state-of-the-art baselines on composition and fine-tuning. 

[1] Khalafi et al. Constrained Diffusion Models via Dual Training. NeurIPS 2024.

[2] Giannone et al. Aligning Optimization Trajectories with Diffusion Models for Constrained Design Generation. NeurIPS 2023.

[3] Khalafi et al. Composition and Alignment of Diffusion Models using Constrained Learning. arXiv 2025.

[4] Garipov et al. Compositional Sculpting of Iterative Generative Processes. NeurIPS 2023. 

[5] Thornton et al.  Composition and Control with Distilled Energy Diffusion Models and Sequential Monte Carlo. AISTATS 2025. 

[6] Skreta et al. The superposition of Diffusion Models using the Ito Density Estimator. ICLR 2025. 

[7] Skreta et al. Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts. ICML 2025. 

[8] Karczewski et al. Devil is in the Details: Density Guidance for Detail-Aware Generation with Flow Models. ICML 2025. 

[9] Shih et al. Long Horizon Temperature Scaling. ICML 2023. 

[10] Zhang et al. Variational Policy Gradient Method for Reinforcement Learning with General Utilities. NeurIPS 2020. 

[11] Domingo-Enrich et al.  Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control. arXiv 2024.

[12] Han et al. Stochastic Control for Fine-tuning Diffusion Models: Optimality, Regularity, and Convergence. ICML 2025.

[13] Uehara et al. Fine-tuning of continuous-time diffusion models as entropy-regularized control. arXiv 2024.

### Questions
Could you please address my concerns detailed in the weaknesses section? In addition, wondering 

(1) what the computational cost of the entire procedure is (noting that a sequence of fine-tuning steps needs to be solved)? 

(2) what the effect of inexact updates is in practice in terms of discrepancy from the solution arrived through exact updates?

### Soundness
1

### Presentation
3

### Contribution
1
