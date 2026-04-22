# Panda: A pretrained forecast model for chaotic dynamics

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 2, 10

## Abstract
Chaotic systems are intrinsically sensitive to small errors, challenging efforts to construct predictive data-driven models of real-world dynamical systems such as fluid flows or neuronal activity. Prior efforts comprise either specialized models trained on individual time series, or foundation models trained on vast time series databases with little underlying dynamical structure. Motivated by dynamical systems theory, we present Panda, Patched Attention for Nonlinear Dynamics. We train Panda on a novel synthetic, extensible dataset of 20,000 chaotic dynamical systems that we discover using an evolutionary algorithm. Trained purely on simulated data, Panda exhibits emergent properties: zero-shot forecasting of unseen chaotic systems preserving both short-term accuracy and distributional measures, nonlinear resonance patterns in attention heads, and effective prediction of real-world experimental time series. Despite having been trained only on low-dimensional ordinary differential equations, Panda spontaneously develops the ability to predict partial differential equations without retraining. We also demonstrate a neural scaling law for differential equations, underscoring the potential of pretrained models for probing abstract mathematical domains like nonlinear dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces  PANDA (Patched Attention for Nonlinear DynAmics), a pretrained transformer model designed to forecast chaotic dynamical systems. The model provides the results of the model to forecast unseen chaotic systems and experiment data snippets in the zero-shot manner; the paper explores the instrinsic properties of Panda in Scaling law, long term statistical property, and some ability in PDE forecasting.

### Strengths
The manuscript shows the following strengths worth credits:

1. the idea of new chaotic trajectories dataset creation is novel;
2. the emprical visualization of the scaling law Panda Performance;
3. key indicators in long-term properties preserving and patch-attention maps are included

### Weaknesses
1. Buzz words cause overclaims, such as 'a universal model ...'. The model does not specify whether the model learns from one modality in the system and is able to forecast other modalities in the real-world systems and PDEs.
2. The emergent PDE performance is less solid, how is the long term statistical property evaluated?
3. Though the results presentation is extensive, the model layers, param and inference computing details are not found.

### Questions
1. As the authors claimed 'a framework to create the dataset of 2e4 ODEs' in the contribution 1, the framework would be a great contribution if the authors can clarify a) what algorithmic evolutionary method is applied; b) how to demonstrate the recombined thing is equivalent to physic solvers and simulators? c) What the phyics meanings of the recombined systems and how the community can access and benefit?
2. The emergent PDE performance is baseline-limited, how the performance is comparable specific surrogate models?
3. The emergent PDE performance is case-limited, e.g., how the performance is in classic benchmarks of 2D/3D Kolmogrov Flows?
4. Why Panda uses this set of attention design? What can be expected if other attention types (e.g Fourier Attention, fast attention, attention as gating filters) are applied?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Panda, a pretrained transformer model for forecasting chaotic dynamical systems. The key contributions are: (1) an evolutionary algorithm that discovers ~20,000 novel chaotic ODE systems through mutation and recombination of 129 base systems; (2) a multivariate patch-based architecture incorporating dynamics-inspired features. in particular, channel attention; (3) demonstration of zero-shot forecasting capabilities on held-out chaotic systems, real-world experimental data, and emergent PDE forecasting; (4) a scaling law showing that diversity of dynamical systems, rather than data volume, drives generalization performance.

### Strengths
1. **Synthetic dataset generation**: The evolutionary discovery framework is interesting and well-executed. I think this is a valuable contribution.
2. **Scaling law for dynamics**: The finding that diversity of unique systems matters more than total data volume is very interesting scientifically.
3. **Architectural choices**: Channel attention for coupled systems and dynamics embeddings are good.
4. **Experimental results**: The evaluations are comprehensive and the results strong compared to baselines.
5. **Emergent PDE forecasting**: The zero-shot ability to forecast PDEs after training only on low-dimensional ODEs is impressive.
6. **Computational efficiency**: Table 8 shows significant speedup over Chronos, the nearest baseline.

### Weaknesses
1. **Clarity**: While the paper is generally very well presented, there are a few places where clarity could be improved. For example, the fact that training uses only 3 randomly sampled channels while evaluating on arbitrary dimensions is buried in Appendix B, but for me this an important detail. In addition, the paper doesn't explicitly state how the univariate baselines are applied to multivariate data—do they process channels independently and concatenate forecasts? How does Chronos-SFT handle multivariate data during fine-tuning?
2. **Mean regression problem**: Section 5.4 reveals that Panda regresses to the mean on long rollouts, yet the paper claims superior distributional metrics compared to Chronos, which "parrots" the input. How can mean regression preserve attractor statistics at all? I think this issues deserves closer study. The claims about attractor statistics would be strengthened if, for example, the model was able to reproduce Lyapunov spectra.
3. **Training details**: While Table 8 reports inference time showing favorable comparison to Chronos, the paper provides no analysis of training setup or timing.

### Questions
1. Please explicitly describe the inference procedure for univariate baselines on multivariate data. Do they process each channel independently and concatenate forecasts? How does Chronos-SFT handle multivariate data during fine-tuning?
2. Why does MLM pretraining help at training horizon but hurt rollout? Do you have some intuition here?
3. How can mean regression achieve better distributional metrics than "parroting"?
4. What are the training costs?
5. Have you tried computing Lyapunov spectra?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents Panda, an attention-based foundation model (FM) dedicated to zero-shot forecasting chaotic dynamics. Using a novel synthetic dataset constructed from a library of known chaotic systems using principles from dynamical systems theory (DST), Panda is trained to forecast trajectories based on limited dynamical context. The model demonstrates robust generalization, successfully providing zero-shot forecasts for entirely novel chaotic systems, diverse real-world datasets, and even PDE problems. Panda is compared to other existing time series FMs in the field and demonstrates superior performance.

### Strengths
1. Panda shows great scaling behavior both in the amount of data as well as number of parameters.
2. Great to see DST motivated architectural design and the focus on setting a suiting inductive bias, which other FMs in the field lack.
3. The manuscript is well written and easy to follow.
4. Generating new chaotic systems using skew-product coupling and evolutionary search is a great idea and powerful method to generate vast amounts of valid data for this class of FMs. I think this greatly benefits research in the entire field.

### Weaknesses
My two main concerns are the following:
1. The paper misses relevant literature which introduces another DS FM model: DynaMix [1]. Both methods train on a synthetic dataset comprised of low-d chaotic DS, both methods have the aim of zero-shot forecasting and both question the efficacy of existing TS foundation models. I think it would be highly valuable to the SciML community if Panda is compared to such a similarly specialized model, which seems to perform much better than context parroting architectures such as Chronos.
2. “Panda exhibits emergent properties: zero-shot forecasting of unseen chaotic systems preserving both short-term accuracy and long-term statistics.” I find the claim that Panda preserves long-term statistics not sufficiently backed up by evidence. The main metrics employed to verify this statement are comparison of the fairly short prediction lengths (up to 512) w.r.t. the ground truth attractor (4096 time steps) in Section 5.4. On the same note, the authors identify regression-to-the-mean (or convergence to fixed point dynamics) as a failure mode of Panda. Proper evaluation of long-term dynamics would consist in performing *longer autoregressive roll-outs* in which the model, ideally, converges to the “climate” hinted by the context (see e.g. [1]). In this case, according to the argumentation of the authors that Chronos exhibits the more suiting inductive bias for climate forecasting (parroting), Chronos should actually perform better in both Hellinger distance as well as KL measure. The way these measures are currently used only strengthens the point that Panda is superior in preserving e.g. geometric features only in the *short-term*, also indicated by the consistently rising mean measure values with prediction length (see Table 1, 2, 10). A SciML motivated architecture should in the end aim to preserve the physics of the underlying system in the long-term, i.e. $T \rightarrow \infty$.

Other concerns:
1. Since the emergent zero-shot PDE inference is highlighted as a major contribution of the paper, the authors should include comparisons to baselines given by domain specific models trained on the context window (e.g. Neural Operator based architectures such as DeepONets [2] or FNOs [3])
2. (Appendix E / Figure 15) ground-truth + completions to ground truth using the correlation dimension lacks baselines: As a reader I can not quantify how well naive imputation methods would perform in estimating the correlation dimension.

Minor comments:
- Fig. 2 misses labels A, B, C mentioned in the caption
- Abbrv. “MLM” is first used in Section 5.1 but hasn’t been introduced in the main text before
- Double citation in References: “Jonas Mikhaeil, Zahra Monfared, and Daniel Durstewitz. On the difficulty of learning chaotic dynamics with rnns. Advances in neural information processing systems, 35:11297–11312, 2022a.”  and “Jonas M. Mikhaeil, Zahra Monfared, and Daniel Durstewitz. On the difficulty of learning chaotic dynamics with rnns. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022b. Curran Associates Inc. ISBN 9781713871088.”

**References**:

[1] Hemmer, Christoph Jürgen, and Daniel Durstewitz. "True zero-shot inference of dynamical systems preserving long-term statistics." arXiv preprint arXiv:2505.13192 (2025).

[2] Lu, Lu, et al. "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators." Nature Machine Intelligence, vol. 3, no. 3, Mar. 2021. https://doi.org/10.1038/s42256-021-00302-5

[3] Liu-Schiaffini, Miguel, et al. "Neural Operators with Localized Integral and Differential Kernels." International Conference on Machine Learning. PMLR, 2024.

### Questions
1. What is the rationale for comparing context + ground truth to context + prediction for several experiments and measures throughout the paper? Why not simply ground truth vs. prediction? Doesn’t inclusion of the context heavily skew the outcome of empirical estimation of the Lyapunov exponent, for example? Especially in cases where $L_{pred} = 128$, where the estimated value is dominated by the context which is $4 \times L_{pred} =  512$?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper introduces Panda, which is a large foundation model for chaotic time series data. To generate the training dataset, the authors introduce a novel, genetic algorithm-like method for discovering new chaotic systems. The model architecture is an extension of the PatchTST model, utilizing patching along the time dimension, feature augmentation with random polynomial and Fourier features, and the use of both temporal attention and channel attention. Comparing the trained model to previous time series foundation models of similar size, the authors find that Pandas gives more accurate predictions, both in the short-term exact trajectory prediction and in the long term predictions of the statistical properties of the attractor. Furthermore, Pandas was able to perform zero-shot forecasts for both experimental data (which contains imperfections not presented in the training dataset) and for PDE systems (which were not seen by the model since it is only trained on ODEs). The paper contains lots of additional results, such as the scaling law with respect to the diversity of the training dataset, and analyses of the attention maps learned by the model.

### Strengths
1.	The proposed model shows excellent performance in both short term trajectory predictions and long term statistical property predictions. It is carefully designed to predict chaotic systems in particular and contain lots of architectural considerations that will be useful to researchers working on similar problems.

2. The automatized dataset generation process is also novel, and the suite of criteria used to automatically sift for chaotic systems seems quite useful. The final dataset of 20000 new chaotic ODEs is also, to my knowledge, first of its kind. Such large scale dataset will also help the further development of foundational models for chaos.

3. Lots of experiments were performed to demonstrate the model's performance. The out-of-distribution results for experimental data and the PDE is quite impressive too, and suggests that indeed the model is able to properly forecast chaos, instead of just having memorized motifs in the training dataset.

### Weaknesses
1.	This is a very well written paper, and there seem to be no significant weaknesses.

### Questions
1.	I am curious about the additive skew-product coupling used in the recombination step for the data generation process. From equations 1 and 2, it seems that recombination will lead to a ODE of larger dimensionality than its parent, since given ODEs for x and y, recombination gives a ODE for the concatenated state [x, y]. If so, I would imagine that the dimensions of the ODEs in the dataset will grow over iterations due to the repeated recombination process. Is this indeed the case?

2. The authors mention that the integration horizon and the granularity (which I assume indicates the sampling period of the trajectories) is standardized using the dominant timescale. How was this timescale determined? Are the authors looking at the dominant frequency in the frequency spectrum, or using the inverse of the maximum Lyapunov exponent?

3. Looking the generated chaotic systems in A.1 and A.2, it seems that loosely there are families of related systems (single, double, triple scrolls, etc). I wonder if it would be possible to do a low dimensional embedding of the generated dataset (or if it is too expensive, some subset of it) to create a 2D plot that shows how different systems in the dataset are positioned (in the chaotic behavior space) with respect to one another, as done in Figure 1 of [1].

4. How long does it take to train the Panda model? How does this compare to the competitors considered such as Chronos and Time MOE?

5. On a similar note, how long does the data generation process take? 

[1] W. Gilpin. Chaos as an interpretable benchmark for forecasting and data-driven modelling. NeurIPS (2021).

### Soundness
4

### Presentation
4

### Contribution
4
