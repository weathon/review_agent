# Extending Fourier Neural Operators for Modeling Parameterized and Coupled PDEs

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Parameterized and coupled partial differential equations (PDEs) are central to modeling phenomena in science and engineering, yet neural operator methods that address both aspects remain limited. We extend Fourier neural operators (FNOs) with minimal architectural modifications along two directions. For parameterized dynamics, we propose a hypernetwork-based modulation that conditions the operator on physical parameters. For coupled systems, we conduct a systematic exploration of architectural choices, examining how operator components can be adapted to balance shared structure with cross-variable interactions while retaining the efficiency of standard FNOs. Evaluations on benchmark PDEs, including the one-dimensional capacitively coupled plasma equations and the Gray–Scott system, show that our methods achieve up to 55~72% lower errors than strong baselines, demonstrating the effectiveness of principled modulation and systematic design exploration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes two architectural extensions to Fourier Neural Operators (FNOs) to better handle two important classes of PDEs: (i) parameterized dynamics with fixed initial conditions, and (ii) systems of coupled PDEs with interacting physical fields. For parameterized dynamics, the authors introduce a lightweight hypernetwork that modulates FNO layers based on physical parameters. For coupled PDEs, they perform a systematic architectural design study to determine where and how to introduce cross-variable coupling, ultimately proposing a Fourier-space interaction scheme. The final model, FNOx (and its parametric variants pFNOx, hpFNOx), achieves significant gains over strong baselines — including coupled multiwavelet neural operators — on both a newly proposed 1D capacitively coupled plasma benchmark and the standard Gray–Scott system.

### Strengths
* **Addresses underexplored settings**: The paper targets two highly practical yet understudied scenarios in neural operator learning — parameterized dynamics with fixed initial conditions, and multiple interacting PDE fields. Both are common in engineering and physics simulations.

* **Effective, minimal extensions**: The proposed improvements are simple but well-motivated. The hypernetwork-based modulation (hpFNOx) conditions model behavior across parameters without blowing up model size, and the spectral-domain coupling design elegantly leverages FNO’s strengths.

* **Systematic architectural study**: The paper carefully explores different configurations (shared vs separate layers, coupling placements) and justifies its final FNOx architecture based on empirical and structural considerations.

* **Strong empirical results**: On both benchmarks, the proposed models achieve large performance gains — often reducing error by 50–70% relative to the best baseline — while maintaining similar runtime and parameter count.

* **New benchmark dataset**: The 1D plasma physics setup with tunable parameters offers a meaningful and realistic testbed for studying parametric and coupled operator learning.

* **Clarity and reproducibility**: The paper is generally well-written, with detailed experimental setups and promised code release. It adheres to good reproducibility practices.

### Weaknesses
* **Incremental novelty**: The core ideas — hypernetwork modulation and coupling in neural operators — are grounded in existing techniques. HyperFNO and related works already explored hypernetwork-based parameter conditioning, and Fourier-space coupling is a natural extension within FNOs.

* **Narrow scope**: The benchmarks, while appropriate, are relatively small-scale and low-dimensional (1D/2D). It’s unclear how well the approach generalizes to larger or more complex systems (e.g. 3D Navier–Stokes, multiple coupled fields).

* **Limited analysis of generalization**: There’s little discussion on how well hpFNOx handles out-of-distribution parameters or extrapolation, which is a practical concern in real applications of parameterized PDE solvers.

* **Brief treatment of second benchmark**: The Gray–Scott results are reported only briefly in the main paper, which weakens the case for generality.

### Questions
1. **How does the hypernetwork-modulated FNO behave on out-of-distribution (OOD) physical parameters?** Would performance degrade sharply or smoothly? Have you tested its extrapolation limits?

2. **Why is the coupling introduced only in the Fourier domain?** Have you explored combining it with local (real-space) cross-variable interactions, and how would that affect performance or efficiency?

3. **Can your proposed architecture scale to PDE systems with more than two coupled fields?** If not, what limitations would arise — architectural, memory, or training-related?

4. **How would the design choices in FNOx (e.g., Q2c, L2, G) transfer to more challenging PDE domains like turbulent 2D/3D fluid flow or real-world inverse problems?** Do you expect the same configurations to hold?

5. **To what extent does the performance improvement come from architectural tuning versus true modeling of coupled/parameterized dynamics?** Could stronger baselines (e.g. HyperFNO, CoDA-NO) close the gap?

6. **In the case of high-dimensional parameter spaces (e.g., 10–20 physical parameters), would the hypernetwork remain effective without overfitting or becoming too large?** How would you scale it?

7. **How sensitive is the performance to the size and architecture of the hypernetwork?** Could a different modulation scheme (e.g., multiplicative gating) outperform simple bias shifts?

### Soundness
3

### Presentation
3

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
This paper extends Fourier Neural Operators (FNOs) to handle parameterized and coupled PDEs. The authors propose two lightweight modifications: (1) conditioning FNOs on physical parameters through either input concatenation (pFNO) or a small hypernetwork modulation (hpFNO), and (2) enabling coupling between multiple PDE variables by performing cross-variable mixing in Fourier space. The resulting framework (FNOx, pFNOx, hpFNOx) retains the efficiency of standard FNOs while improving accuracy for systems with varying parameters or interacting fields. The authors introduce a new benchmark based on a 1-D capacitively coupled plasma model and also evaluate on the Gray–Scott equations. The proposed hpFNO achieves up to 55–72 % error reduction over strong baselines.

### Strengths
- Addresses a timely and important gap: FNOs are increasingly popular for PDE surrogates, yet parameterization and coupling remain under-explored.

- The architecture modifications are simple and principled—hypernetwork modulation and Fourier-space coupling can be integrated into existing FNOs with minimal effort.

- The approach is general and modular, applicable to multidimensional parameter fields and different neural-operator families.

- Demonstrates comparisons across multiple strong baselines (FNO, CFNO, MWT, DeepONet, U-Net) with solid quantitative gains.

- Introduces a new plasma-physics benchmark that could be useful to the community.

### Weaknesses
1. Writing is meandering and often unfocused; core ideas and notation could be stated more directly.

2. Organization: the Related-Work section appears late; it would be clearer to position it before the Methods.

3. The Gray–Scott example feels secondary, lacking detailed analysis or ablation—appears appended to strengthen results. In fact, the experiments are limited and cannot demonstrate general applicability. The paper should be enhanced with more experiments from different PDEs.

4. The claimed generality for “parameterized and coupled PDEs” is convincing for 1-D cases but not well demonstrated on higher-dimensional or more complex systems.

5. Limited qualitative insight—plots focus on error metrics. The authors should delve deeper into the interpretation of learned coupling or modulation effects.

6. typos like "learable". should have numbered the equations.

### Questions
1. How sensitive is the hpFNO performance to the design of the hypernetwork (depth, parameterization, modulation type)?

2. Could the same effect be achieved by allowing \( W_\ell(\mu) \) (parameter-dependent local maps) instead of additive \( s_\ell(x, \mu) \) biases?  

3. Are there stability or extrapolation limitations when \( \mu \) lies outside the training range?  

4. How does coupling only in Fourier space compare to coupling in both spatial and spectral domains?

5. Can the authors comment on scalability to 2-D/3-D PDEs and memory growth with the number of coupled variables?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper extends the Fourier Neural Operators (FNOs) to handle coupled and parameterized PDEs by introducing new architectures. The authors propose designs that model parametrized dynamics and coupled systems and evaluate it on a newly developed 1D capacitively coupled plasma and Gray-Scott dataset. Experimental results demonstrate that the proposed hpFNO models achieve higher predictive accuracy compared to existing FNO variants.

### Strengths
* The introduction clearly motivates the need for efficient operator learning on coupled and parametrized PDEs.
* The proposed pFNO, and hpFNO architectures aim to improve representations for coupled and parametrized PDEs, and the results on the benchmarks are promising.
* The ablation study thoughtfully analyzes the effects of design components, showing careful empirical work.

### Weaknesses
1. Lack of architectural visualization : The paper introduces multiple new and extended operators, but there is no schematic figure summarizing the overall architecture.
2. Unclear explanation for “best 5 of 10 runs” in Table 2 : Why is this criterion used for $T_{in} = 2$ and $T_{in} = 1$?
3. Sparse content in Section 4.3 : This section contains only a brief summary and refers the reader to the appendix. Including a compact summary table would be better for understanding cross-domain generalization.

### Questions
1. What exactly does “acc” in Figure 3 represent? It seems the lower is better, so 'acc' might be misleading.
2. How many CCP benchmark data were generated, and how were they split into training/test sets?

### Soundness
4

### Presentation
2

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
1. The paper is well-written, easy to follow, and clearly motivated.

2. The experimental results are impressive, demonstrating a 55-72% reduction in error over strong baselines and confirming the value of the principled design.

3. I am not an expert in this area and will defer to the other reviewers for their expert opinions on the technical contribution

### Strengths
1. The paper is well-written, easy to follow, and clearly motivated.

2. The experimental results are impressive, demonstrating a 55-72% reduction in error over strong baselines and confirming the value of the principled design.

### Weaknesses
The manuscript would benefit from additional visualizations, such as a diagram illustrating the model architecture, to help readers better understand the proposed method.

### Questions
What is the key difference between your method and the baselines, and why does your approach work?

### Soundness
3

### Presentation
3

### Contribution
3
