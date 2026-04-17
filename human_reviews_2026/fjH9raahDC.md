# Less is More: Improving Molecular Force Fields with Minimal Temporal Information

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Accurate prediction of energy and forces for 3D molecular systems is one of fundamental challenges at the core of AI for Science applications. Many powerful and data-efficient neural networks predict molecular energies and forces from single atomic configurations. However, one crucial aspect of the data generation process is rarely considered while learning these models i.e. Molecular Dynamics (MD) simulation.
Molecular Dynamics (MD) simulations generate time-ordered trajectories of atomic
positions that fluctuate in energy and explore regions of the potential energy surface
(e.g., under standard NVE/NVT ensembles), rather than being constructed to steadily lower
the potential energy toward a minimum as in geometry relaxations.
This work explores a novel way to leverage molecular dynamics (MD) data, when available, to improve the performance of such predictors. We introduce a novel training strategy called FRAMES, that use an auxiliary loss function for exploiting the temporal relationships within MD trajectories. 
Counter-intuitively, on two atomistic benchmarks and a synthetic system we
observe that minimal temporal information, captured by pairs of just two consecutive
frames, is often sufficient to obtain the best performance, while adding longer
trajectory sequences can introduce redundancy and degrade performance.
On the widely used MD17 and ISO17 benchmarks, FRAMES significantly outperforms its Equiformer baseline, achieving highly competitive results in both energy and force accuracy. Our work not only presents a novel training strategy which improves the accuracy of the model, but also provides evidence that for distilling physical priors of atomic systems, more temporal data is not always better.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose FRAMES, a training strategy for machine-learned interatomic potentials (MLIPs) where trajectory data is learned at the same time as energies and forces, thereby increasing the amount of geometrical and physical information available to the model and improving prediction accuracies for the MLIP. The method is employed to improve performance on the MD17 and ISO17 datasets.

### Strengths
Perhaps needless to say, both MLIPs and trajectory predictor models are known in the literature. Using them together to improve MLIP accuracies is however an original and interesting idea. The quality of the investigation is high, all design choices and experiments are reasonable and justified, and the clarity of the presentation is excellent.

### Weaknesses
In my opinion, the work has a single major weakness, which unfortunately prevents it from having a wide significance in the field. Modern datasets for MLIPs are not generally obtained from molecular dynamics simulations. While this dataset construction technique was popular around 2017 (although, even at that time, it was never the one primarily used by actual practitioners in chemistry), it is now at best used as one of the many splits that go into a well-crafted dataset. More often than not, it is not used at all. Popular alternatives include sampling from equilibrium databases, rattling of equilibrium structures, cutting (to obtain surfaces), random structure searches, random defect creation, random element substitutions, cell distortions and so on. Maybe for this reason, the authors limit their investigation to two relatively old datasets (MD17, ISO17) which would definitely not be used to train state-of-the-art models today. As such, I believe the evaluation is lacking, but I do not know if improving it is possible, given that recent datasets cannot easily be trained with the proposed strategy. A similar recent idea, denoising non-equilibrium structures (DeNS, which is incidentally not referenced by the authors), is struggling to be used consistently despite its applicability to arbitrary (including modern) datasets. This is due to its modest gains in accuracy (which are comparable to those shown here), compared to a moderately tedious implementation (also comparable to the architecture developed in this work). As such, I do not realistically see many people in the field benefitting from FRAMES.

Other weaknesses:
- In the abstract, the sentence "MD generates trajectories of atomic positions of molecular systems moving from higher energy states to lower energy stable/equilibrium states" is incorrect. This is not geometry optimization. If this were true, MD would eventually have to lose all its energy and freeze. In short, in molecular dynamics, it is not the positions that are updated with something proportional to the forces, but the momenta. This creates a much more complex dynamical system.
- The literature review section is lacking in many aspects. Recent evidence points to the fact of "incorporating E(3)-equivariance" not being "crucial" (line 84) for MLIPs. On the contrary, unconstrained models can achieve similar accuracies and at least equal computational efficiency compared to equivariant models. In the same section, something called "TEMPO" (line 87) is mentioned. I can only suppose this was an earlier name for FRAMES. As mentioned earlier, DeNS deserves a mention as a related piece of work, trying to achieve a similar effect to the current work. FlashMD, a modern variant of MDNet, provides a good analysis to what the authors correctly identify as the need for only two structures to predict the positions of the new frame. Finally, recent literature on architectures for MLIPs in chemistry and materials is omitted, although I can totally understand if this is due to the page limit.
- Comparison to state-of-the-art architectures other than Equiformer is very limited, and, once again, this is probably due to the fact that only MD-derived datasets are usable for this method. Nonetheless, the authors could try their method to improve a different architecture for completeness.

### Questions
I have no questions for the authors at this stage. If they see fit, they can elaborate on the points above and we can discuss them more in detail.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors suggest adding an auxiliary prediction head to MLIPs during training only, that takes in embeddings from multiple geometries and predicts the atomic displacement $\Delta p = p_{t+1} - p_t$.
They test the idea using EquiformerV1 on ISO17, MD17, and a toy harmonic oscillator.

### Strengths
- The idea is simple to implement and cheap
- The suggested auxiliary loss improves forces, and especially the energies, on ISO17

### Weaknesses
- Weak results on MD17 (table 1). The loss only improved the accuracy over the baseline in 9/16 cases, each improvement being less than 5%.
- I am not sure the small gains have anything to do with temporality. The auxiliary loss seems like a reconstruction loss.

### Questions
- Can you add ablations with (a) predicting the atomic displacement to a small random perturbation instead of $p_{t+1}$ (predicting $\Delta p = p_t^{noised} - p_t$) (b) just reconstructing the current input geometry p_t (only taking in a single geometry, predicting $p = p_t$)? These "noisy node" auxillary losses are known in the literature to help model performance and are easier to implement. Thus the proposed new method should beat this baseline
- To improve the generality of the empirical results, can you show results for direct-force prediction, e.g. with the otherwise similar EquiformerV2?
- I think the synthetic toy benchmark of the harmonic oscillator is missing the MSE of predicting the force without using the auxiliary loss as a baseline
- Figure 2 needs to be revised with axis labels, clearer x-tick labels, and larger fonts
- “MD generates trajectories of atomic positions of molecular systems moving from higher energy states to lower energy stable/equilibrium states.” I think you are referring to “relaxations”. The usual microcanonical NVT-ensemble MD conserves total energy.

### Soundness
1

### Presentation
2

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
This paper proposes FRAMES, a new training strategy for machine-learning interatomic potentials (MLIPs) on molecular dynamics (MD) trajectories: In addition to the usual approach of predicting energies and forces for each frame independently, an additional head is tasked with predicting the positions at the next step, improving the quality of internal representations and therefore accuracy. The authors further investigate how much temporal context is beneficial and observe that using two frames improves accuracy, while three frames degrade performance. Results are presented on MD17, ISO17, and a toy spring–mass system.

Note: An LLM (ChatGPT 5) was used to expand the review from notes to the full text. The model did not suggest useful feedback, and therefore did not contribute beyond this task.

### Strengths
- A simple and appealing idea, clearly explained and easy to implement, with practical relevance for MLIP training.
- The method is model-agnostic, requires no additional inference-time cost, and appears practically useful.
- Experimental results generally support the claim that using minimal temporal information can improve performance.
- Error bars are included for some results, which is appreciated.
- The spring–mass example illustrates the intended intuition.

### Weaknesses
- Statistical signal is weak in several experiments and some cases with error bars show high variance. This makes the "less is more" conclusion, and the overall claim of FRAMES providing a decisive advantage, feel somewhat premature.
- The use of original MD17, rather than revMD17 (https://archive.materialscloud.org/records/pfffs-fff86; DOI:10.1088/2632-2153/abba6f) introduces known noise issues; the conclusions may be confounded by simulation artefacts rather than true learning dynamics.
- The paper occasionally overstates its claims. Observing that three frames perform worse than two does not necessarily imply that more temporal information is inherently harmful in general.
- The method is limited to deterministic, fixed–time step MD trajectories (NVE). It would not apply to data from stochastic thermostats or irregular sampling. In practice, due to their inherent correlations, data from MD is not often used directly for MLIP training.
- Only a single architecture is tested. This weakens claims of generality.

### Questions
- Did you consider using non-adjacent frames (e.g., larger temporal spacing)? This would isolate long-range temporal cues from redundancy effects and help test the stated hypothesis more cleanly.
- I would suggest providing a more comprehensive discussion a potential mechanistic explanation for why more temporal data is worse.
- I strongly suggest including at least one experiment on revMD17 or an ablation that explicitly examines the role of simulation noise. This would reduce concerns that the observed degradation with $T=3$ is partly due to noise accumulation, and likely increase the signal in the reported results.
- Consider increasing sample counts or reporting more seeds, especially for Table 4 / Figure 2. The variability observed makes interpretation difficult.
- Please phrase some claims more cautiously. It may be more accurate to say that "for short-range deterministic MD data, two frames suffice and three may introduce harmful redundancy" rather than the broader "less is more" framing.
- Make explicit that the method assumes deterministic MD with fixed time-step and no stochastic thermostat; otherwise readers may overgeneralize. Please comment on practical relevance given that MD trajectories are typically sub-sampled to avoid strongly correlated training data.
- Results with a second architecture (e.g., EGNN, NequIP, MACE, PET) would strengthen the argument for generality, but I do not view this as essential for acceptance.
- Using $\mathbf{p}$ for positions is an unconventional choice for MD, as it is commonly used for momenta. $\mathbf{r}$ is the usual choice. I would suggest changing this notation.
- It may be interesting to consider a different/additional toy system that has simple governing equations, but chaotic dynamics, for example a double pendulum.

### Soundness
2

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
This paper introduces FRAMES, a novel training strategy for improving neural network predictions of molecular dynamics (MD). It proposes a novel training approach that uses an auxiliary loss function during training to leverage temporal information from MD trajectories. The backbone FRAMES model is the Equiformer (equivariant GNN). Through an additional auxiliary head on top of the traditional prediction head, it regularize the model training via atom position changes across frames $\Delta{p_t}$ from concatenated frame embeddings. In the experiments, the author demonstrates an interesting finding that minimal temporal information (2 consecutive frames) is optimal
Using 3+ frames introduces data redundancy and degrades performance - "Less is More". Across two real-world MD benchmark and one synthetic datasets, FRAMES(T=1) achieves better performance than Euiformer and (T>1) variants.

### Strengths
1. The paper introduces an effective auxiliary loss that predicts the atomic displacement across time frames. As a result, it significantly improves the result on ISO17, especially on energy prediction. 

2. Problem formulation and literature review is crystally clear and includes most of the relevant work.

### Weaknesses
1. The performance improvement is only measured in one of the SOTA model - Equiformer. The generalization of FRAMES optimization is not demonstrated. 

2. The novelty might be limited as the only innovation is on introducing an auxiliary loss, which is common in neural network molecular predcitions.

### Questions
1. When T>1, $\mathcal{L}_\text{aux}$ calculates the displacement across non-adjacent frames, I am wondering what if $\mathcal{L}_\text{aux}(T=2)$ sums $\mathcal{L}_\text{aux}(T=1)$ at T and T-1. Does it perform better? 

2. Do you have any insightful or theoretical explaination on why T=1 works better? 

3. What does the X-axis in Figure 2 represent? I didn't understand its correlations with Table 4.

### Soundness
3

### Presentation
3

### Contribution
2
