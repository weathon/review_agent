# Learning Collective Variables from BioEmu with Time-Lagged Generation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Molecular dynamics is crucial for understanding molecular systems but its applicability is often limited by the vast timescales of rare events like protein folding. Enhanced sampling techniques overcome this by accelerating the simulation along key reaction pathways, which are defined by collective variables (CVs). However, identifying effective CVs that capture the slow, macroscopic dynamics of a system remains a major bottleneck. This work proposes a novel framework coined BioEmu-CV that learns these essential CVs automatically from BioEmu, a recently proposed foundation model for generating protein equilibrium samples. In particular, we re-purpose BioEmu to learn time-lagged generation conditioned on the learned CV, i.e., predict the distribution of molecular states after a certain amount of time. This training process promotes the CV to encode only the slow, long-term information while disregarding fast, random fluctuations. We validate our learned CV on fast-folding proteins with two key applications: (1) estimating free energy differences using on-the-fly probability enhanced sampling and (2) sampling transition paths with steered molecular dynamics. Our empirical study also serves as a new systematic and comprehensive benchmark for MLCVs on fast-folding proteins larger than Alanine Dipeptide.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BIOEMU-CV, a novel framework for automatically learning collective variables (CVs) for molecular dynamics (MD) simulations. The authors identify that existing machine-learning-based CVs (MLCVs) often struggle to scale from small toy systems to more complex proteins.

The proposed method cleverly leverages a large, pre-trained protein foundation model, BioEmu, as a "scaffolding" to train a new, lightweight CV encoder. The core idea is to train this encoder to find a low-dimensional CV ($c_t$) from a protein's current state ($x_t$) that is maximally predictive of a future, time-lagged state ($x_{t+\tau}$). This is achieved by freezing the weights of the powerful BioEmu model and training only the small encoder and an adapter to "condition" BioEmu to generate $x_{t+\tau}$ based on $c_t$. This time-lagged objective theoretically forces the CV to capture only the slow, persistent dynamics (e.g., folding) while discarding fast, random fluctuations.

The method is benchmarked against three other self-supervised MLCVs (DeepTICA, TAE, VDE) on three fast-folding proteins (Chignolin, Trp-cage, BBA). The evaluation is performed on two key downstream tasks: (1) free energy difference estimation using On-the-fly Probability Enhanced Sampling (OPES) and (2) transition path sampling using Steered Molecular Dynamics (SMD).

### Strengths
1. The paper's most significant contribution is demonstrating a clear failure of popular baseline methods (VDE and TAE) to scale beyond the 10-residue Chignolin protein. As shown in Table 3 and Figure 5, VDE is completely incapable of separating the folded and unfolded states for Trp-cage and BBA. BIOEMU-CV, by contrast, scales robustly and provides excellent state discrimination for all three proteins. This finding alone reframes the problem, suggesting that foundation-model approaches may be a necessary solution.

2. The method demonstrates superior qualitative performance. The CV provides a clear, unambiguous separation between the folded and unfolded basins (Table 3), which is the primary prerequisite for any useful CV. Furthermore, Figure 6 shows it successfully assigns distinct values to different secondary structures ($\alpha$-helix vs. $\beta$-sheet) in BBA, a task where VDE and TAE fail.

3. The SMD results (Table 2) provide the strongest quantitative evidence for the method's utility. BIOEMU-CV achieves a 93.8% target hit percentage (THP) for BBA, compared to just 18.8% for DeepTICA (the only other baseline to even run the task). This is a decisive result and shows the CV is highly effective as a reaction coordinate for guiding folding pathways.

4. The approach of re-purposing a large, frozen foundation model is computationally appealing. It avoids the cost of training a massive new model from scratch and instead produces a small, fast, and practical CV encoder that can be easily plugged into existing MD simulation packages.

### Weaknesses
1. This is the most significant weakness in the paper's evaluation: The OPES simulations (Table 1) are demonstrably not converged. The reported standard deviations for the free energy difference ($\Delta F$) are often larger than the mean values themselves (e.g., $1.36 \pm 7.92$ kJ/mol for Trp-cage and $0.82 \pm 16.57$ kJ/mol for BBA). With such massive error bars, it is statistically impossible to compare the methods. The authors' claims of "winning" based on the mean value are unsupportable. These results should be treated as inconclusive.

2. Even within the inconclusive OPES results, the "win" is not uniform. For Chignolin (the most well-behaved system), DeepTICA achieves a notably better PMF-MAE (2.64) than BIOEMU-CV (3.07).

3. The target state $x_{t+\tau}$ is a synthetic construct (BioEmu C$\alpha$ + hpacker + relaxation), not a pure MD frame. This introduces a significant potential for bias, where the CV may be learning the artifacts of the hpacker algorithm rather than the true, all-atom slow dynamics. This concern is not addressed with an ablation study (this point is expanded in Question 1).

4. The method is benchmarked on three "fast-folding proteins". While this is an improvement over systems like alanine dipeptide, these are still relatively small proteins (10-28 residues). Its scalability to much larger, more complex proteins (e.g., 100+ residues) is not demonstrated.

5. The study is limited to a 1-dimensional CV ($d=1$) for simplicity. While this is readable, it is likely limiting for multi-pathway kinetics or larger proteins; the paper doesn’t probe $d>1$ (this point is expanded in Question 3).

### Questions
1. The target state $x_{t+\tau}$ is not a pure simulation frame but a synthetic construct (BioEmu C$\alpha$ + hpacker side chains + OpenMM relaxation). This pipeline may introduce its own biases. How can the authors be sure that the learned CV isn't just learning to predict the artifacts of the hpacker algorithm rather than the true underlying slow dynamics of the protein? An ablation study training on "native" MD frames vs. the synthetic targets would seem necessary to resolve this.

2. Given the massive standard deviations in Table 1, the 1µs OPES simulations are clearly unconverged. Can the authors provide longer-run data or, failing that, justify why any conclusions are being drawn from this specific task? As it stands, the $\Delta F$ results are statistically inconclusive and weaken the paper's overall argument.

3. The study is limited to a 1-dimensional CV ($d=1$) for simplicity. However, complex folding events often require more than one coordinate. How does the method perform when tasked with finding 2 or 3 CVs? Does the time-lagged generation approach successfully separate multiple slow modes?

4. The steering variable for the SMD task is a combination of the learned CV and C$\alpha$-RMSD. This is a significant confounding variable. Can the authors please provide results for the SMD task using only the learned CV as the steering variable? This is necessary to prove that BIOEMU-CV itself, and not the C$\alpha$-RMSD helper, is responsible for the high target hit percentage.

5. The paper does not include "simple baseline" controls. How does the learned CV's performance (e.g., in SMD THP) compare to a simple, non-learned CV, such as a linear PCA or TICA component derived from either the input coordinates or the internal latents of the frozen BioEmu model? This is needed to demonstrate that the complex, time-lagged training is necessary.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Enhanced sampling uses collective variables (CVs) for sampling from long timescales which can represent rare events. However, finding good CVs is not trivial. This paper proposes to learn these from BioEmu. Concretely, an encoder is trained to embed the input structure to a low-dimensional vector, which represents the CVs. This vector is added to the single representation in BioEmu so that the time-lagged conformation is generated.

To evaluate the model, it looks at the free energy difference estimation and transition path sampling, which requires correct long timescale encoding. Comparisons to the baselines show comparable/better performance of the proposed model.

### Strengths
* originality 
 Learning latent representations from a fixed pretrained model is not new but this paper adapts it for learning CVs for slow dynamics which is a novel application.  
 * quality 
 The model's predictions are compared against state-of-the-arts quantitatively while it would have been better if more that 3 proteins were tested.
 * clarity
 It is mostly clear to understand except for some typos and sentences.
 * significance
 Running MD simulations to investigate slow dynamics of proteins, e.g. folding, is challenging due to the required computation resource and error accumulation. Enhanced sampling methods can tackle this challenge but it needs good VCs. This paper introduces an effective way of learning them by wrapping around an existing protein ensemble generation model, BioEmu. The results are competitive and extensive comparisons were conducted. The folding path sampling results show a good performance of the model.

### Weaknesses
* To see if the model works well on a variety of proteins, it would make the paper stronger if more than 3 proteins were tested.

### Questions
* There are two downstream tasks used: free energy difference estimation and transition path sampling. What could be other metrics to used to measure the effective encoding of slow dynamics?
* The proposed model is compared against self-supervised models. Are there any supervised models for the task?
* Line309: 'not meaningless'->'meaningless'?
* line317: 'through out'->'throughout'
* Fig1 caption: 'earn'->'learn'
* Fig4, the circle represents the folded state. Why is the red convex hull not around it for chignolin, while it is for others? Also, would it be better to draw lines for transition paths somehow, to clearly see the paths?
* Fig3, it would be interesting to also show the transitions of samples from compared models.
* Have you tried using other models than BioEmu, e.g. Boltz, AlphaFlow? I guess it should be straightforward to do it and should be interesting to compare the results.

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
3

### Summary
This work proposes a novel framework for learning collective variables. The model trains an encoder to extract CVs by enabling the frozen generative model BioEMU to generate molecular conformations under time-lagged conditions. The paper benchmarked MLCVs for two slow degree freedom tasks and demonstrated the model's effectiveness.

### Strengths
1. The idea of capturing collective variables through a conditional generative task is novel. 
2. The paper is well-structured, clearly presented, and supported by thorough experiments.

### Weaknesses
1. The method heavily relies on the model capabilities of BioEmu. The captured CV may contain model bias.
2. The paper a bit overstates the scope of its CVs, especially in the title. It should make clear that the method focuses on CVs for enhanced sampling, rather than general CV learning.
3. The dataset is quite limited, and the selected proteins are all very small. The method should be tested on a broader set of proteins with varying sizes to better demonstrate its generality and robustness.

### Questions
1. How physically interpretable are the learned CVs?
2. If BioEmu’s modeling capabilities are insufficiently accurate, will the learned CV still reflect the system's true slow degrees of freedom? How can this issue be mitigated?

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
3

### Summary
The paper proposes a method for efficient molecular dynamics by learning collective variables (CV) from the BioEmu model. The paper adds a small encoder on top of a frozen BioEmu diffusion model to learn a one-dimensional collective variable (CV) using a time-lagged objective. The authors evaluate the learned CV on three fast-folding proteins using OPES (for free-energy and PMF error) and SMD (for target hitting and maximum energy along paths). All CV’s are restricted to 1 dimension.

### Strengths
- The integration is simple and practical: keep BioEmu frozen and train only a small conditioning head. This is an attractive engineering path if it works broadly.


- On SMD tasks, the method often reports better target-hitting probability and lower maximum energy along the path, which suggests the CV is useful for guiding transitions.


- The paper provides reasonably clear setup details for OPES and SMD, which helps with reproducibility.

### Weaknesses
- The paper needs more ablation. It is unclear which components matter most. For example, comparisons for time-lagged vs. non-time-lagged training, frozen BioEmu vs. partial fine-tuning, different encoder sizes and placements, and SMD runs without mixing the learned CV with RMSD. Without these tests, it is hard to attribute the reported gains to the proposed choices, and hard to evaluate the novelty of the proposed method vs. the reported baseline methods.
- Restricting the CV to one dimension and forcing a fixed range can hide multi-modal kinetics or make methods look closer or farther apart depending on alignment. While reported CV’s don’t need to be that high dimensional, understanding if the method scales to multi-dimensional CV’s at all is important for future practitioners.


- Several results have large standard deviations. In Tables 1 and 2, many method averages fall within each other’s uncertainty ranges. This weakens the ranking of methods and makes performance differences difficult to trust.

### Questions
- How much of the gain comes from time-lagging itself versus using BioEmu features?


- Does partially fine-tuning BioEmu help or hurt compared to freezing it?


- How sensitive are results to encoder size and to the choice of where conditioning is attached within the model?


- Does the approach generalize to larger proteins or multi-path systems, and to higher-dimensional CVs?

### Soundness
3

### Presentation
3

### Contribution
2
