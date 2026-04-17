# Refine Drugs, Don’t Complete Them: Uniform-Source Discrete Flows for Fragment-Based Drug Discovery

- Decision: Accept (Poster)
- Scores: 2, 6, 8, 4

## Abstract
We introduce InVirtuoGen, a discrete flow generative model for fragmented SMILES for de novo and fragment-constrained generation, and target-property/lead optimization of small molecules. The model learns to transform a uniform source over all possible tokens into the data distribution. Unlike masked models, its training loss accounts for predictions on all sequence positions at every denoising step, shifting the generation paradigm from completion to refinement, and decoupling the number of sampling steps from the sequence length. For \textit{de novo} generation, InVirtuoGen achieves a stronger quality-diversity pareto frontier than prior fragment-based models and competitive performance on fragment-constrained tasks. For property and lead optimization, we propose a hybrid scheme that combines a genetic algorithm with a Proximal Property Optimization fine-tuning strategy adapted to discrete flows. Our approach sets a new state-of-the-art on the Practical Molecular Optimization benchmark, measured by top-10 AUC across tasks, and yields higher docking scores in lead optimization than previous baselines. InVirtuoGen thus establishes a versatile generative foundation for drug discovery, from early hit finding to multi-objective lead optimization. We further contribute to open science by releasing pretrained checkpoints and code, making our results fully reproducible.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Through this paper, the authors propose InVirtuoGen, a discrete flow generative model for fragmented SMILES for de novo and fragment-constrained generation, and target-property/lead optimization of small molecules. Specifically, InVirtuoGen learns to transform a uniform source over all possible tokens into the data distribution by refining the sequence.

### Strengths
- The authors provided the codebase.

### Weaknesses
I will combine the *Weaknesses* section and the *Questions* section. My concerns are as follows:
- The main weakness of this paper is its weak novelty. The method section is completely missing. The proposed InVirtuoGen framework is a combination of a discrete flow model (Section 2.1) and the fragmented SMILES notation (Section 2.2). Only the latter corresponds to the invention of this work, but its contribution appears to be very minor.  The new notation is very similar to the SAFE notation [1], and its advantages over the SAFE notation is not described. The sentence “Our notation preserves fragment integrity while providing explicit attachment point numbering” is included in the caption of Figure 2, but this also applies for the SAFE notation. Overall, I am not convinced that this work provides a significant contribution from an ML perspective compared to previous methods in the domain.
- The proposed InVirtuoGen framework shows very low molecular validity in the fragment-constrained generation tasks (Table 1).

---

**References:**

[1] Noutahi et al., Gotta be safe: A new framework for molecular design, ArXiv, 2023.

### Questions
Please see the *Weaknesses* section for my main concerns.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents InVirtuoGen, a discrete flow–based molecular generator that refines fragmented SMILES strings rather than completing them autoregressively. The model starts from a uniform source and learns a refinement process that updates all tokens simultaneously. It integrates a genetic algorithm for initial candidate recombination and a PPO-based fine-tuning scheme for property optimization, achieving state-of-the-art results on PMO and competitive lead optimization performance.

### Strengths
1. Proposes a novel refinement-based discrete flow paradigm that departs from completion-style generation.

2. Strong empirical results across de novo, fragment-constrained, and property optimization tasks.

3. The GA + PPO hybrid optimization design is elegant and practically effective for sample-efficient molecular search.

### Weaknesses
1. The sampling modification (Eq. 6) lacks theoretical justification and deviates from formal flow definitions.

2. Fragment-constrained generation is not well-aligned with the model’s refinement philosophy, showing lower validity.

3. Limited analysis of training stability and efficiency vs. on-policy/off-policy baselines (e.g., GFlowNets, diffusion models).

### Questions
1. How does the proposed refinement training compare in sample efficiency to off-policy methods like GFlowNets?

2. Does the PPO fine-tuning use off-policy data reuse (e.g., replay) or purely on-policy updates, and how does that affect stability?

3. Could the refinement framework be extended to stochastic sampling or energy-based objectives for better exploration?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This submission investigates the problem of drug discovery, more specifically fragment-based drug discovery. The authors present a method called InVirtuoGen, which adopts the discrete flow model framework of Gat et al., 2024 and a slightly modified version of the SAFE framework of Noutahi et al., 2023 for molecule representation. The main contribution of the submission is the proposed method for Target Property Optimization on top of the Discrete Flow Model. This method combines a genetic algorithm for faster convergence with PPO-based Reinforcement Learning for gradient-based optimization. Experiments were performed on benchmarks for de novo (unconstrained) generation, fragment-constrained generation, target property optimization, and  lead optimization. Experimental results are supportive.

### Strengths
The submission focuses on the practically important tasks of molecule optimization instead of de novo molecule generation. 

InVirtuoGen proposes a discrete flow model that transforms a uniform source distribution over all tokens into molecular data. The simultaneous generation of the whole molecule is beneficial for optimization tasks.

It enables longer generation trajectories, where the quality improves with more refinement steps.

InVirtuoGen integrates Genetic Algorithm + PPO for target-conditioned and goal-directed optimization, a novel hybrid scheme that combines the strengths of both ingredients.

Strong experimental results on high-diversity generation benchmarks, in particular for property optimization and lead optimization.

### Weaknesses
The number of baselines compared with experimentally is small (2 and 3, respectively), and the credibility of the experimental results would be significantly strengthened by comparing with more SOTA methods.

It remains unclear whether the experimental results for the baselines copied from the original publications are comparable with the experimental results obtained for InVirtuoGen by the authors.

Limited intuition for why validity is lower than GenMol or SAFE-GPT, even though quality metrics (valid, unique, QED ≥ 0.6) are higher.

Appears better suited for de novo or goal-directed generation, rather than fragment-constrained generation — GenMol’s masked diffusion approach respects constraints more effectively.

### Questions
InVirtuoGen proposes a discrete flow model that transforms a uniform source distribution over all tokens into molecular data. How is this different from starting with uniform prior, rather than a masked prior commonly done (https://arxiv.org/pdf/2402.04997)?

Most experimental results for the baselines were copied from the original publications. How did you ensure that the experimental designs are identical and the experimental results comparable?

Please perform experiments with other SOTA methods!

### Soundness
3

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
4

### Summary
This paper introduces InVirtuoGen, a novel generative model for fragment-based drug discovery. The core contribution is a paradigm shift from "auto-regressive" used by SAFE-GPT or "completion" used by GenMol to "refinement." Authors investigate the model performance on four drug discovery tasks: de-novo generation, fragment-constrained generation, property optimization, and lead optimization. The method achieves a superior quality-diversity pareto frontier in de novo generation and sets a new state-of-the-art (SOTA) on the Practical Molecular Optimization (PMO) benchmark under both 260k and 10k oracle calls.

However, there are some concerns, and I would like to increase the score depending on the response.

---
**Usage of LLM:** I wrote the entire review myself and only used the LLM to correct the grammar and improve readability.

### Strengths
- **Novel & Versatile Generation Paradigm**: Authors proposed a novel paradigm for molecular generation. In particular, it is different previous auto-regressive approach or unmasking approach.

- **Commitment to Fair Benchmarking**: A major strength is the paper's transparent and fair evaluation. For the PMO benchmark, the authors correctly identify that prior SOTA models (GenMol, f-RAG) used a 250,000-call pre-screening step without fully accounting for it. This paper reports results in both settings (with and without pre-screening), providing a much fairer comparison against all baselines (e.g., Genetic GFN) and restoring integrity to the benchmark.

- **Transparency in Reporting Failures**: The authors are commendably transparent about the model's full capabilities, rather than cherry-picking only successful tasks. They include a detailed analysis of fragment-constrained generation, a task where the model performs poorly and its validity drops significantly. This transparency provides valuable insights into the model's "philosophy" and its operational boundaries.

### Weaknesses
**1. Lack of Conditional Generative Capabilities**

The paper successfully employs a PPO-based fine-tuning strategy for property optimization. However, unlike baseline model SAFE-GPT, authors do not investigate conditional generative capabilities proposed by Scaffold-GGM[1] and BBAR[2]. The ability to control key molecular properties directly (e.g., "generate a molecule with LogP = 2") is an important task. Demonstrating this capability would have significantly strengthened the paper's claim of being a "versatile generative foundation."

**2. Misleading Paper Structure**

The paper's structure is misleading. The fragment-constrained generation task is presented early (Section 3.2, Table 1), suggesting it is a core capability. However, this task yields extremely low molecular validity (e.g., 28.6% for superstructure design) and, as the authors rightly admit, fundamentally "conflicts with the refinement philosophy." This prominent placement of a failed task creates significant doubt about the model's reliability. I suggest moving this section to the last section of manuscript or appendix in a camera-ready version.

**3. Lack of validity metrics**

Related to W2, authors did not report independent validity metrics for its other, more successful tasks (de novo generation and property optimization). In the de novo task (Sec 3.1), validity is conflated with other metrics inside the "Quality" score, making it impossible to assess the base generative success rate. In the optimization tasks (Sec 3.3, 3.4), only the final scores of successful molecules are reported, giving no insight into the efficiency of the process (i.e., how many invalid molecules were generated and discarded to achieve those scores). This omission is significant, as it prevents a full assessment of the model's generative reliability in its primary use cases.

---
**References:**
1. Lim, Jaechang, et al. "Scaffold-based molecular design with a graph generative model." Chemical science 11.4 (2020): 1153-1164.
2. Seo, Seonghwan, Jaechang Lim, and Woo Youn Kim. "Molecular generative model via retrosynthetically prepared chemical building block assembly." Advanced Science 10.8 (2023): 2206674.

### Questions
- Line 153: How is `c1ccccc1` tokenized? `c1` or `c`&`1`?
- Line 283: As I know, GenMol cut an arbitrary single bond in molecules to create offspring. Since this cutting can be conducted with a reaction template, it does not restrict the chemical space.

### Soundness
2

### Presentation
1

### Contribution
3
