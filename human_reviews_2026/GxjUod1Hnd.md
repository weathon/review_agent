# PDE-SHARP: PDE Solver Hybrids Through Analysis & Refinement Passes

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Current LLM-driven approaches using test-time computing to generate PDE solvers execute a large number of solver samples to identify high-accuracy solvers. 
These paradigms are especially costly for complex PDEs requiring substantial computational resources for numerical evaluation.
We introduce PDE-SHARP, a framework to reduce computational costs by replacing expensive scientific computation by cheaper LLM inference that achieves superior solver accuracy with 60-75\% fewer computational evaluations.
PDE-SHARP employs three stages: $\textbf{(1) Analysis}$: mathematical chain-of-thought analysis including PDE classification, solution type detection, and stability analysis; $\textbf{(2) Genesis}$: solver generation based on mathematical insights from the previous stage; and $\textbf{(3) Synthesis}$: collaborative selection-hybridization tournaments in which LLM judges iteratively refine implementations through flexible performance feedback.
To generate high-quality solvers, PDE-SHARP requires fewer than 13 solver evaluations on average compared to 30+ for baseline methods, improving accuracy uniformly across tested PDEs by $4\times$ on average,
and demonstrates robust performance across LLM architectures, from general-purpose to specialized reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PDE-SHARP (PDE Solver Hybrids through Analysis & Refinement Passes), an LLM-driven framework that generates numerical PDE solvers dramatically more efficiently than brute-force sampling methods11. It utilizes a three-stage approach: Analysis (mathematical planning), Genesis (code generation), and Synthesis (collaborative LLM judges and refinement tournaments)2. PDE-SHARP achieves superior solution accuracy, with an average $4\times$ improvement (geometric mean), while cutting computationally expensive solver evaluations by $60\%-75\%$ (requiring $<13$ evaluations vs. $30+$ for baselines).

### Strengths
1. PDE-SHARP successfully validates its core hypothesis by substantially reducing the number of expensive solver executions (average $13.2$ evaluations) while delivering higher solution quality than exhaustive sampling baselines4444. This is critical for complex PDEs where simulation costs dominate5.
2. The structured Analysis stage, encompassing PDE classification and stability checks, injects deep domain knowledge before code generation. This leads to high-quality initial code, evident in the system's successful, immediate discovery of superior hybrid analytical-numerical solutions for the Reaction-Diffusion PDE, a strategy missed by baselines
3. The Selection-Hybridization Tournaments enable efficient refinement using performance feedback. This mechanism proves robust to the underlying LLM choice for code generation and demonstrated impressive adaptation by auto-correcting the Advection solver from an analytical method to a finite-volume scheme that better matched the training data reference solution.

### Weaknesses
1. High LLM API Cost: While computationally efficient in terms of GPU time, the framework incurs a significant LLM API overhead, with the total cost ($4.01 using GPT-4o) being $5\times$ higher than CodePDE ($0.75) and $7\times$ higher than OptiLLM-CoT ($0.58)12. This high LLM dependency limits immediate practical adoption for users relying on costly API servic.

2. Accuracy/Metric Alignment: The paper notes that achieving the full accuracy gain is sensitive to the alignment between the feedback metric and the final evaluation metric. Suboptimal performance results if the feedback doesn't match the target evaluation

### Questions
1. Cost Justification: Given the high LLM cost, quantify the strategic overhead: How many hours of GPU time (for a computationally complex PDE like Navier-Stokes) must be saved to make the $4.01 API cost for PDE-SHARP economically superior to the brute-force baselines?
2. Convergence Order and Quality: Why do the final solvers for complex PDEs like Navier-Stokes and Darcy Flow still show a predominance of first-order convergence (around $60\%-75\%$ of cases) across all methods, including PDE-SHARP? Does the stability analysis prioritize robustness (e.g., first-order upwinding) over formal accuracy order (e.g., second-order finite volume)?
3. I wonder whether the paper has tested on any 3D NS problems which would be more challenging and interesting. Please try it and I will change my rating accordingly with your feedback.

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
3

### Summary
This paper proposes PDE-SHARP, an LLM-based framework for solving PDEs. It consists of three stages: an Analysis stage for PDE classification, a Genesis stage for generating different solvers based on the PDE category, and a Synthesis stage for iteratively fine-tuning the solvers. Experiments show that PDE-SHARP significantly improves accuracy across different LLM backbones compared to other methods. An ablation study effectively demonstrates the necessity of all three stages.

Overall, I believe this work is sufficiently innovative and supported by comprehensive experimental analysis, so I am inclined to accept it.

### Strengths
- To my knowledge, PDE-SHARP is the first to introduce a PDE classification module. Its detailed analysis phase enables the construction of tailored solvers for different cases, offering strong scalability.

- The experimental section evaluates the performance using various LLM backbones, comprehensively demonstrating the robustness of the method.

- The ablation study effectively validates the necessity of the three stages.

### Weaknesses
- The description of the method is relatively brief. It would be better to add an algorithm block to clearly illustrate the complete process, supplemented by concrete examples to aid understanding.

- Although PDE-SHARP requires fewer iterations, it appears to have significant time and API overhead in some cases, as shown in Figure 7b and Table 3.

### Questions
- What are the specific differences between the different analysis results shown in Figure 2?

- Can you show some examples of PDE solver code generated by Genesis stage?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces PDE-SHARP, a three-stage framework for generating numerical solvers for Partial Differential Equations (PDEs) using Large Language Models (LLMs). The core idea is to replace expensive, brute-force evaluation of many solver candidates with a more intelligent process that leverages cheaper LLM inference. The framework consists of: (1) an **Analysis** stage, where an LLM performs a mathematical chain-of-thought analysis of the PDE, including classification and symbolic stability analysis, to create a solver plan; (2) a **Genesis** stage, which generates initial solver candidates based on this plan; and (3) a **Synthesis** stage, which uses collaborative "tournaments" of LLM judges to iteratively refine and select the best solver through performance-informed feedback.

The authors demonstrate that PDE-SHARP can generate high-quality solvers with 60-75% fewer computational evaluations compared to baseline methods like `CodePDE`. Across five representative PDE tasks, their method achieves an average accuracy improvement of 4x and shows robustness to the choice of the underlying generator LLM.

### Strengths
1.  **Originality & Significance:** The central thesis—swapping expensive scientific computation with cheaper, structured LLM inference—is highly significant and timely. The framework moves beyond the naive "generate-and-test" paradigm. The **Analysis** stage, in particular, is a novel and powerful concept. Tasking the LLM to perform a symbolic, mathematical pre-analysis (e.g., deriving stability bounds like the CFL condition) before writing any code is a major step towards integrating genuine mathematical reasoning into automated scientific discovery. This has the potential to make LLM-driven science far more computationally tractable.

2.  **Quality & Clarity:** The paper is exceptionally well-written and clearly structured. The PDE-SHARP framework is thoughtfully designed, with each stage serving a distinct and logical purpose. The **Synthesis** stage, with its collaborative tournaments and flexible feedback mechanisms (nRMSE, PDE residual, etc.), is a sophisticated and well-conceived approach to iterative refinement. The ablation studies (e.g., Figure 6) effectively demonstrate the individual contributions of the analysis and synthesis components, adding to the quality and rigor of the work *within its defined scope*.

3.  **Robustness:** The paper convincingly shows that the tournament-based synthesis stage mitigates the weaknesses of individual generator LLMs, leading to consistently strong performance regardless of the base model (Table 2). This is a valuable finding, suggesting the framework's architecture is more important than the specific LLM used for the initial code generation.

### Weaknesses
Despite its clever design, the paper's evaluation suffers from a fundamental methodological flaw that undermines its central claims about solver quality.

1.  **Fundamentally Misguided Evaluation Paradigm:** The paper's primary weakness is its evaluation context. It demonstrates that PDE-SHARP is superior to other LLM frameworks that also generate Python solvers *from scratch*. However, this is not the relevant benchmark for practical scientific computing. The state-of-the-art for solving PDEs is not to write low-level finite difference loops in Python, but to orchestrate highly-optimized, mature numerical libraries (e.g., **FEniCS**, **PETSc**, **deal.II**, or JAX-based toolkits like **JAX-CFD**).
    *   A human expert, whom the paper implicitly claims to compete with, would almost never write a production solver from scratch in Python for the problems tested. They would write a short script to call a powerful library.
    *   By focusing on generating code from scratch, the paper is optimizing a suboptimal workflow. The impressive "Analysis" stage is spent reasoning about the stability of a simple, from-scratch finite difference scheme, rather than reasoning about how to best use a powerful, pre-existing tool. This limits the practical significance of the results.

2.  **Insufficient Baselines and Metrics:** The comparison is made against other LLM-driven methods (`CodePDE`, `OptiLLM`) but not against a truly strong, non-LLM baseline. While the paper improves upon `CodePDE`, it inherits the same evaluation weakness: comparing against a method that itself was not benchmarked against the practical state-of-the-art. The primary metric is final `nRMSE` on a fixed grid, which is insufficient for evaluating numerical algorithms. A rigorous comparison requires **convergence analysis**, showing how error decreases with grid refinement and computational cost (wall-clock time). Without this, it is impossible to assess the true order of accuracy and efficiency of the generated solvers.

3.  **Limited Problem Complexity:** The chosen PDE test cases (1D advection, 1D Burgers, etc.) are standard but relatively simple. They feature simple geometries (1D/2D boxes) and boundary conditions (periodic). These are scenarios where even basic, from-scratch methods can perform adequately. The framework's utility is not tested on problems where the choice of numerical method is truly critical, such as those with complex geometries (requiring FEM), multi-scale phenomena, or severe stiffness, which are ubiquitous in science and engineering. It is unclear if the "from-scratch" paradigm would scale to these challenges at all.

4.  The paper fails to benchmark against readily available, state-of-the-art general-purpose code agents (e.g., Cursor, the Claude code). These tools excel at the practical task of generating high-level scripts that leverage powerful, pre-existing numerical libraries. My own experiments suggest that for the problems discussed, these agents can produce correct, library-calling scripts with a success rate exceeding 95%, where the solution's accuracy is guaranteed by the underlying traditional algorithm. By not comparing against this highly effective and simpler workflow, the paper misses the most relevant baseline.

5.  The quality of several figures is poor, with noticeable distortion and low resolution. For example, Figure 5 is blurry and difficult to read. The authors should revise all figures to ensure they are high-resolution and legible for the final version.

### Questions
1.  The paper's core premise is that generating low-level solver code from scratch is a valuable goal. However, the scientific computing community has invested decades in building robust, high-performance libraries to abstract away these low-level details. Could the authors justify why generating from-scratch solvers is a more promising direction than prompting an LLM to act as an "expert user" that writes high-level scripts to **call libraries like FEniCS or PETSc**? How would the performance, and more importantly, the scalability of PDE-SHARP to complex, real-world problems, compare to this alternative library-centric paradigm?

2.  The evaluation relies heavily on the final `nRMSE` value. For numerical methods, this can be misleading. Could the authors provide **convergence plots** (e.g., log-log plots of error vs. grid spacing $\Delta x$, and error vs. wall-clock time) for at least one or two of the PDE tasks? These plots should compare the best solver from PDE-SHARP against a canonical, well-implemented solver from a standard library (e.g., a second-order finite volume method for advection, or a spectral method for Burgers' equation). This would provide a much more meaningful assessment of the generated solver's accuracy and efficiency. My assessment of the paper would improve substantially if such a methodologically sound comparison were presented.

3.  The mathematical `Analysis` stage is a key strength. However, it appears to focus on deriving stability for explicit finite difference or simple hybrid schemes. How would this stage generalize to problems where more advanced discretizations are required, such as the Discontinuous Galerkin (DG) method for conservation laws or spectral methods for smooth problems? Does the LLM possess the deep mathematical knowledge to perform a symbolic stability analysis for these more complex schemes, or is the framework's intelligence implicitly limited to a narrow class of numerical methods?

4.  There are many mature general-purpose code agents on the market, but this paper does not compare them with those such as Cursor and Claude Code. Based on my personal testing, for the problems discussed in this paper, having a general-purpose code agent call a commonly used PDE library can achieve a success rate of over 95%, and the solution accuracy can be guaranteed by traditional algorithms. Based on this, I would like to ask the author: 1. Could you add a comparison with these general-purpose code agents? 2. Compared to them, what is the applicable scope of the algorithm presented in this paper, and what are its advantages? Please demonstrate this with experiments.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
PDE-SHARP introduces a three-stage LLM-driven framework for generating PDE solvers that reduces computational evaluations by 60-75% while improving accuracy by 4× on average. Evaluated on five PDEs from PDEBench, PDE-SHARP requires fewer than 13 solver evaluations versus 30+ for baselines, demonstrates robustness across diverse LLM architectures.

### Strengths
1.	Comprehensive experiments across five diverse PDEs demonstrate consistent improvements over multiple competitive baselines.
2.	The Analysis-Genesis-Synthesis decomposition is intuitive and effective—mathematical reasoning precedes code generation, and collaborative tournaments refine implementation.
3.	The paper provides detail across 51 pages including extensive appendices covering implementation specifics, complete prompt templates, etc.

### Weaknesses
1. The evaluation covers only five toy-level PDEs—four 1D problems (Advection, Burgers, Reaction-Diffusion, Navier-Stokes) and one 2D steady-state problem (Darcy Flow)—which are textbook examples with well-established numerical methods, failing to demonstrate capability on challenging real-world scenarios such as 3D turbulent flows, coupled multi-physics systems, the NS equations, the Maxwell Equations etc.

2. The figures resemble simple process diagrams without conveying the holistic design, making it difficult to grasp how components integrate and why this specific architecture addresses limitations of existing methods.
3. Section 3 provides only informal prose descriptions without formal problem formulations, algorithmic pseudocode, or mathematical definitions

### Questions
1.	Are the three judge LLMs different models, or the same model with different prompts/temperatures? Does judge diversity (e.g., mixing reasoning models like o3 with general-purpose models like GPT-4o) improve tournament outcomes?
2.	How do PDE-SHARP-generated solvers compare in accuracy and efficiency to hand-tuned implementations using established libraries (FEniCS, PETSc) by domain experts? Is the goal to match expert performance or provide accessible automation for non-experts?
3.	Section 3 describes Genesis in only two sentences—how exactly does the stage translate Analysis outputs (PDE classification, stability bounds, decomposition decisions) into concrete solver code? 
4.	The paper uses "n" for both initial solver candidates and top-n/2 selections

### Soundness
2

### Presentation
3

### Contribution
2
