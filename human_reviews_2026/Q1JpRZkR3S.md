# Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein Dynamics

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Molecular dynamics (MD) simulations remain the gold standard for studying protein dynamics, but their computational cost limits access to biologically relevant timescales. Recent generative models have shown promise in accelerating simulations, yet they struggle with long-horizon generation due to architectural constraints, error accumulation, and inadequate modeling of spatiotemporal dynamics. We present STAR-MD (Spatio-Temporal Autoregressive Rollout for Molecular Dynamics), a scalable SE(3)-equivariant diffusion model that generates physically plausible protein trajectories over microsecond timescales. Our key innovation is a causal diffusion transformer with joint spatiotemporal attention that efficiently captures complex space-time dependencies while avoiding the memory bottlenecks of existing methods. On the standard ATLAS benchmark, STAR-MD achieves state-of-the-art performance across all metrics--substantially improving conformational coverage, structural validity, and dynamic fidelity compared to previous methods. STAR-MD successfully extrapolates to generate stable microsecond-scale trajectories where baseline methods fail catastrophically, maintaining high structural quality throughout the extended rollout. Our comprehensive evaluation reveals severe limitations in current models for long-horizon generation, while demonstrating that STAR-MD's joint spatiotemporal modeling enables robust dynamics simulation at biologically relevant timescales, paving the way for accelerated exploration of protein function. Project page: https://bytedance-seed.github.io/ConfRover/starmd

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes STAR-MD, an autoregressive model for MD trajectories of protein systems. The key innovation is a causal diffusion transformer to predict the conditioning at a given frame using spatio-temporal attention in an efficient manner.  The model samples per-residue SE(3)-frames at each timestep via a diffusion process on SO(3), conditional on this conditioning vector. Experiments are performed on the ATLAS dataset, showing improved coverage of the conformational space as compared to previous methods, with significantly reduced memory requirements.

### Strengths
The model removes the Pairformer block present in existing protein models to save on memory. Instead, the model utilizes attention over the single-residue features across all frames and all residues. By clever training tricks such as feeding the clean and noisy frames together with a special attention mask, the training is parallelized over all frames, improving efficiency.
The experiments on long-horizon trajectory generation are good to see, and the model performs well in this challenging setting.

### Weaknesses
The empirical results in this paper are indeed impressive, but I would have liked to see a discussion and evaluation in identifying functional motions in proteins (eg. transitioning between active/inactive states, for example as represented by DFG in/out motifs in kinases. See https://www.rcsb.org/structure/6XR6 vs https://www.rcsb.org/structure/6XR7 for an example with Abl kinase.) Indeed, that is the main point of performing long MD simulations. ATLAS itself has very few functional motions, and tICA itself is not an ideal metric to extract these motions. That being said, I think these comments are more about the problem (trajectory generation) rather than this specific approach.

For the model itself, I would have liked to see generalization experiments to unseen lengths or sequences with low similarity to the training set, since that is a very common setting in practice.

### Questions
* Can you explain the structural noise aspect in the section “Contextual Noise Perturbation for Robust Rollouts”? Is it that you sample $τ ∼U[0.0, 0.1]$ and the corresponding noisy frame $x_\tau$ instead of keeping the clean history?
* Is the main reason for improved efficiency due to significantly lesser memory? The authors point out $O(N^2 L^2)$ for STAR-MD vs $O(N^3 L)$ for previous methods, but in the large $L$ limit, I would expect STAR-MD to become more inefficient?
* Please mention how the ATLAS test set was created (even if referring to prior work) since it helps put generalization results into context. Do you find that the model is able to generalize to new proteins with low sequence similarity?
* Please mention the inference costs of all models to enable a comparison to the ground truth MD.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces STAR-MD, an autoregressive SE(3) diffusion model designed to simulate long-horizon protein dynamics. The authors identify two key bottlenecks in existing generative models: 1) poor computational/memory scaling (due to $\mathcal{O}(N^3)$ or $\mathcal{O}(N^2)$ pair-based operations) and 2) error accumulation that leads to instability in long rollouts.

STAR-MD's methodology tackles both problems directly. For Scalability, it uses a novel joint Spatio-Temporal (S$\times$T) attention that operates only on single-residue features, avoiding the expensive pair-feature bottleneck of competitors like ConfRover. For Stability, it employs a "contextual noise perturbation" (a form of diffusion forcing) that makes the model robust to its own small errors, preventing them from compounding over time.

The results are very strong. On the 100 ns ATLAS benchmark, STAR-MD achieves the best combination of structural validity and conformational coverage. More importantly, it is the first model to demonstrate stable, physically plausible generation out to the 1 microsecond timescale, a task where all baseline models (MDGen, ConfRover-W, AlphaFolding) fail catastrophically due to either memory limits or error accumulation.

### Strengths
1. The primary strength is the joint S$\times$T attention on singles-only features. This is a very clever architectural trade-off. It avoids the $\mathcal{O}(N^3)$ or $\mathcal{O}(N^2 L)$ complexity of competitors (AlphaFolding, ConfRover), which is the main barrier to long-horizon simulation. The KV cache analysis (Fig. 5) proves this is a massive practical win (e.g., 6.6MB vs 1.3GB per layer).

2. The "contextual noise perturbation" is the second key contribution. Autoregressive models are notoriously unstable in long rollouts. This method demonstrably fixes the problem. The stability plots (Fig. 3) and long-horizon results (Table 2) are direct proof that STAR-MD is robust to compounding errors.

3. The results are unambiguous. On the 100ns benchmark (Table 1), STAR-MD has the best balance of coverage and validity. On the 1µs benchmark (Table 2), it's the only model that works, maintaining 80% validity while competitors are unusable (e.g., MDGen 23% validity, AlphaFolding 0.06%).

4. The paper provides strong evidence that this is a trajectory simulator, not just an "ensemble generator." The long-horizon stability (Fig. 3) and kinetic fidelity metrics (tICA and RMSD-vs-lag) show that the model generates a temporally coherent and physically plausible "movie," not just a shuffled bag of good-looking frames.

### Weaknesses
1. The paper's greatest strength is also its biggest unproven assumption. The model trades physical explicitness (i.e., operating on pair features) for computational speed. It bets that its S$\times$T attention is powerful enough to implicitly learn all the complex, long-range pairwise interactions (like allostery) just from single-residue features. While this seems to work for the ATLAS proteins, it is a major leap of faith that this will hold for more complex systems defined by subtle, cooperative, long-range motions.

2. The "Dyn." metrics are a bit superficial. The tICA score is just a single number, and the paper's own Table 1 shows that the "AlphaFolding" baseline (which is 99.89% invalid) can get a "perfect" 0.17 tICA score. This "AlphaFolding Paradox" proves that the tICA metric alone is an insufficient measure of a good simulator. The RMSD/autocorrelation plots are good, but the paper would be much stronger if it included a more rigorous kinetic analysis (e.g., implied timescales from an MSM, VAMP-2 scores, or transition rate calculations).

3. For the long-horizon tests (Table 2), the ConfRover baseline had to be modified to a windowed variant (ConfRover-W) to manage its high memory requirements . While this strongly supports STAR-MD's superior scalability, it is not a direct comparison of kinetic fidelity, as the baseline is architecturally constrained by a limited temporal window. The AlphaFolding baseline appears to be a non-competitive comparator. With a reported structural validity of only 0.11% (Table 1), this model is fundamentally unsuited for this task . Its "perfect" tICA score (0.17) serves primarily to demonstrate the insufficiency of the tICA metric in isolation, rather than providing a meaningful performance benchmark.

### Questions
1. Can the authors please comment on the "singles-only" trade-off? How confident are they that this architecture can capture complex, cooperative phenomena like allostery, which are fundamentally defined by long-range pairwise correlations? Have they considered testing STAR-MD on a known allosteric benchmark protein to validate this?

2. The validation of conformational coverage (JSD/Recall) is done in the PCA space of the reference trajectory. This metric is a good measure of ensemble overlap but does not validate the kinetic pathways or transition probabilities. A model could, in theory, match this distribution without being a good simulator. Could the authors comment on this choice and why a more kinetically-focused metric (like implied timescales) was not also used?

3. The tICA metric details are sparse. Given the "AlphaFolding Paradox" (where a 0.11% valid model gets a perfect tICA score), this metric needs more justification. What featurization, lag time, and number of components were used?

4. Related to the AlphaFolding baseline: the 0.11% validity is shockingly low. Can the authors confirm their setup for this baseline? It's hard to believe this is a representative result for that model, and it makes the comparison less meaningful.

### Soundness
4

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
This paper introduces STAR-MD, a new approach to modeling temporal protein dynamics, focusing on long-term consistency, using generative diffusion models. To this end, the authors use a scalable residue frame representation, following previous work. However, this corresponds to a coarse-grained representation compared to the fully atomistic molecular dynamics simulations that are used to train the model. Based on this observation, it is possible to show with Mori-Zwanzig theory that an accurate modeling of the temporal dynamics in the coarse representation requires the incorporation of memory (opposed to a fully Markovian approach of some prior works) as well as complex spatial-temporal correlations. Therefore, the authors introduce an architecture that not only leverages a single previous frame but a context window of several frames, thereby capturing the system's history as required by Mori-Zwanzig theory. Moreover, they use explicit joint temporal-spatial attention layers to also capture the necessary complex correlations. That aside, the authors leverage established architecture modules, like invariant point attention and pair representations. Conceptually, the approach can be considered related to methods in the video generation literature. The authors then validate their method and outperform prior baselines. The simulated protein dynamics trajectories are scored in terms of structural quality, conformational coverage and dynamic fidelity. In particular for long roll-outs the method performs better than baselines, avoiding major drift and error accumulation.

### Strengths
- **Clarity**: Overall the paper is easy to follow (however, I believe the motivation for the approach could be explained better, see weaknesses below).
- **Originality:** The paper makes a novel and original contribution in the area of protein dynamics by explicitly modeling long context windows and using joint spatio-temporal attention with conditioning noise to achieve long-term temporal generative modeling of molecular dynamics trajectories. However, the different components can be found elsewhere in the literature, e.g. in video generation, and the approach is therefore not overly creative from an architecture perspective. This is not a major concern, though, as the results seem mostly strong. On the other hand, the coarse-graining and Mori-Zwanzig formalism perspective is novel in the context of generative method design and interesting (but this could be presented better, see comments below).
- **Experimental results:** Overall, the paper shows satisfactory experimental results, outperforming baselines. In particular the long-horizon results are strong.
- **Significance**: Accelerating and emulating molecular dynamics simulation via generative methods is an important research topic with impact in biology, physics and chemistry. The paper makes a non-negligible step in this direction, and therefore the contribution can be considered significant.

### Weaknesses
- **Presentation of motivation:** In the main text, the authors argue without much explanation that long context and complex spatio-temporal modeling is necessary to accurately model molecular dynamics trajectories. Naively, it will be difficult to understand for most readers why that is, given that the original atomistic molecular dynamics data is essentially Markovian, corresponding to a simple integration of equations of motion (possibly with a thermostat/barostat). The reasoning for this is only explained in more depth in the appendix, when looking at the problem through the lens of coarse graining and Mori Zwanzig theory, which many readers won't know. Looking at the problem of generative modeling of molecular dynamics from this perspective is very interesting and novel, yet this is only "hidden" in the appendix. I would suggest the authors to restructure the paper and explain more of this in the main paper. Meanwhile, the diffusion formalism and other basics are completely standard and could be moved to the appendix. This would improve the paper, I believe.
- **Results in Figure 2:** In Figure 2, right, the MD trajectory of STAR-MD covers more phase space than the baselines, but it still never explores the mode on the right hand side (high PC1, low PC2 values). While the method is evidently better than the baselines, it is still unclear how well STAR-MD explores the configurational state space.
- **Configurational sampling and baselines:** More generally, long-horizon molecular dynamics, the core problem tackled in this paper, is meant to explore the full phase space. I believe the authors should study this in more depth and compare to other methods that sample molecular conformations. For example, how does STAR-MD compare to BioEmu (https://www.science.org/doi/10.1126/science.adv9817) in that regard, one of the most well-known and prominent methods to sample similar molecular systems (proteins)? Can we have full phase space diagrams for different test proteins and compare (e.g. the fast folding test proteins used in BioEmu)? The calculated coverage metrics are nice, but not very interpretable. I would expect the method to show conformational coverage similar to MD to be useful in practice -- this is a critical aspect.
- **Free Energy Calculations:** Also related to the above, does the method enable free energy calculations? For instance of conformational transitions? Free energy calculation is one of the most important tasks of molecular dynamics. Hence, it would be good to know if STAR-MD can be used for such tasks, and it would be great to add results on this.

### Questions
In section 4.4, the paper generates trajectories with different strides. How far can this be pushed? The model is trained with a stride length up to 10ns (section 3.3). What happens if we generated data with this maximum stride length and use ~850 frames/steps, as in the other experiments? What simulation time could we reach without error accumulation? Shouldn't this improve coverage and phase space exploration?

That aside, I don't have further questions, but some typos:
- line 036: integrate -> integrates
- line 107: embeddings -> embedding's?
- line 124: blocks -> block

And a suggestion: In Figure 2, the two MD reference curves are almost exactly on top of each other - I initially thought a curve is missing. I would strongly suggest the authors to address this and fix the visualization so it's clear there are two curves on top of each other.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces STAR-MD, a spatio-temporal SE(3)-equivariant diffusion model designed for learning protein molecular dynamics directly from Cartesian coordinate trajectories. The model represents each residue as a rigid frame (translation + rotation) and predicts its time evolution conditioned on physical stride Δt, trained via diffusion-style denoising. It is evaluated on the ATLAS dataset and compared against baselines including MDGen, ConfRover, and AlphaFolding-4D. The authors report that STAR-MD reproduces MD-like short-horizon dynamics (RMSD vs lagtime, tICA correlations), achieves high backbone validity, and maintains stability over extended rollouts (up to 1 μs). Ablation studies test variants without diffusion noise and without joint spatio-temporal attention.

This is a technically solid and promising paper whose model is likely to influence future ML-based coarse-grained MD approaches. However, the evaluation would be much stronger with explicit clarification of (i) representational scale differences, (ii) alignment and Δt normalization procedures, and (iii) statistical variability across rollouts.
The current comparisons convincingly show stability, but not necessarily accuracy, and the role of coarse-graining deserves explicit discussion.

### Strengths
1. Clear technical contribution: Combines SE(3)-equivariant frame modeling and continuous-time conditioning in a diffusion framework which conceptually elegant and well-motivated.
2. Strong empirical results: STAR-MD substantially outperforms prior coordinate-based models in conformational coverage, dynamic fidelity, and stability over long horizons.
3. Method clarity (core model): The paper’s architectural description and use of per-residue SE(3) frames are well explained.
4. Long-horizon evaluation: Extending rollouts to hundreds of nanoseconds or microseconds is ambitious and valuable for assessing numerical stability.
5. Well-integrated metrics: Use of RMSD, tICA, and validity metrics ties the work to established MD analysis practices.

### Weaknesses
1. Comparative clarity. It’s difficult to realize from the text that the baselines use very different geometric representations: AlphaFolding-4D is all-atom, MDGen and ConfRover are backbone-level, and STAR-MD is Cα-only.
Because this difference likely affects both stability and metric scale, the paper should state it explicitly and discuss how Cα-level evaluation influences comparisons.

2. Alignment and Δt normalization. The paper does not describe how trajectories are aligned prior to RMSD/PCA/tICA computation, nor how variable Δt conditioning was reconciled with baselines’ fixed strides.

3. Single-trajectory visuals. Figures 2–3 show single rollouts; ensemble variability or error bars would make coverage and stability claims more convincing.

4. Unexplained shading in Fig. 3. The caption omits what the shaded bands represent (presumably inter-protein variation).

5. Minimal ablation scope. Only “w/o noise” and “w/ separate attention” are tested. These yield small metric changes, so it remains unclear which components truly drive the reported improvements.

### Questions
1. Role of representation. The results suggest that representational scale (Cα vs. backbone vs. all-atom) may contribute more to stability than architectural details. Do the authors have evidence or experiments that could separate representational effects from architectural ones?
2. Alignment. Were trajectories aligned (e.g., via Cα superposition) before computing RMSD and PCA/tICA metrics, and if so how?
3. Δt comparability. How were lag times normalized when comparing models with variable vs. fixed strides?
4. Shading in Fig. 3: What statistic do the shaded bands represent? Standard deviation, percentile range, or something else?
4. Long-horizon ablations. Do the “w/o noise” and “w/ sep attn” variants remain stable in 240 ns – 1 µs rollouts?

### Soundness
4

### Presentation
3

### Contribution
4
