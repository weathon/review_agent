# SIMPL: Scalable and hassle-free optimisation of neural representations from behaviour

- Decision: Accept (Poster)
- Scores: 6, 5, 8, 6

## Abstract
Neural activity in the brain is known to encode low-dimensional, time-evolving, behaviour-related variables. A long-standing goal of neural data analysis has been to identify these variables and their mapping to neural activity. A productive and canonical approach has been to simply visualise neural "tuning curves" as a function of behaviour. However, significant discrepancies between behaviour and the true latent variables -- such as an agent thinking of position Y whilst located at position X -- distort and blur the tuning curves, decreasing their interpretability. To address this, latent variable models propose to learn the latent variable from data; these are typically expensive, hard to tune, or scale poorly, complicating their adoption. Here we propose SIMPL (Scalable Iterative Maximization of Population-coded Latents), an EM-style algorithm which iteratively optimises latent variables and tuning curves. SIMPL is fast, scalable and exploits behaviour as an initial condition to further improve convergence and identifiability. It can accurately recover latent variables in spatial and non-spatial tasks. When applied to a large hippocampal dataset SIMPL converges on smaller, more numerous, and more uniformly sized place fields than those based on behaviour, suggesting the brain may encode space with greater resolution than previously thought.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new method, SIMPL for estimating neural representation regarding behaviors. The method is based on the Expectation and Maximization method where during E-step, a model produces estimates of latent trajectories using Kalman smoothing and during M-step, the model uses KDE for fitting the intensity function.

### Strengths
* The paper is well-written. Especially, the references in the introduction can be a good guide to readers who are not familiar with neuroscience, especially, the hippocampus and MEC although the authors simply listed 10 papers in the actually related work section.

* The figure is generally helpful in understanding the method and the results.

* The proposed method is a very simple E-M-based algorithm but it is more efficient and has better performance than CEBRA in the grid cell simulation with the RatsInABox simulator.

### Weaknesses
1. The author claims general but limited to specific regions of brain place cells and grid cells. I highly recommend authors compare with various datasets such as the macaque dataset and the mouse visual cortex datasets used in Schneider et al. (2023).

2. The authors misuse big-O notation. For example, the authors mentioned O(1 hour, 200 neurons, 10^6 spikes) ~ O(1 min) in L 120 and O(hours) in L219. This causes confusion, especially to computer scientists who have been trained in big-O notation for measuring time or space complexity. If the authors want to use big-O notations, please change them to match the function of neurons/spikes and so on. However, I believe, just changing to plain text might be better.

3. Except for Section 4.2, there are no comparisons with existing methods or ablation studies. Although the results appear promising, it remains challenging to discern whether this is due to the simplicity of the task or the efficacy of the proposed method. Moreover, I am not sure whether CEBRA is the only method to compare with.
The authors should provide more details on the training and testing splits. L305–306 and 416–417 lack information: are the splits based on spatial segmentation (using specific parts of the box for training) or trial separation?

4. Although experiments with Tanni et al. (2022) represent the main result in this paper, the authors scarcely describe the task and the significance of each plot (particularly in Figure 6c), simply referring back to the original paper.

5. The authors compare SIMPL with CEBRA, a machine learning-based method running on CPUs. If they provide a runtime comparison showing that SIMPL on a CPU is faster than CEBRA on a GPU, it would further support their efficiency claims.

6. Lines 477–479 are ambiguous as to whether the authors discuss place cell remapping or another concept. If remapping is the intended topic, I recommend explicitly stating this and including a citation to aid readers from the machine learning community who may be unfamiliar with this concept.

7. The writing can be improved a lot:

7.1 It is possible that the readers do not know what the “turning curve” is since it is used in neuroscience literature and ICLR is generally a machine learning community. It is good to define what it is in the introduction.

7.2 It would be better to move the Related Work section to Before Method (standard ML conference styles) or Discussion (some ICML styles). Currently, it disconnects methods and results.

7.3 Please choose between British English and American English. Currently, it is used together (e.g., In L477, optimization, behaviour).

7.4 Consider that readers may not be familiar with “tuning curve,” a term more common in neuroscience literature. Since ICLR has a general machine learning audience, defining it in the introduction would be helpful.

7.5 Moving the Related Work section to a position before the Method section (per standard ML conference styles) or to the Discussion (in line with some ICML styles) may improve flow, as it currently separates the methods and results sections.

7.6 Maintain consistency in English dialect, as the paper currently alternates between British and American English (e.g., "optimization" vs. "behaviour" in line 477).

7.7 In line 131, specify “Appendix A” instead of “Appendix,” even if there is only one section.
7.8 Ensure the font for v is consistent in lines 142 and 143.

7.9 Since space permits, it may be preferable to adjust the placement of Figure 5 so that it doesn’t span pages 7 and 8, avoiding the empty space near lines 399–405.

7.10 IN L816, 1, …, P => \{ 1, …, P \}.

7.11 Correct citation formatting:
* In line 116, “Tanni et al. (2022)” should use citep.
* In lines 486, 493, and 790, change citep to citet.
* In line 522, switch citet to citep.

7.12 It would also be beneficial to unify the reference formatting:
* Journals like Nature and Nature Neuroscience do not list volume numbers, while JMLR and most eLife papers do (except in lines 583–585). Additionally, both “elife” and “Elife” are used inconsistently, as are “Nature” and “nature.”
* Clarify “cite” with a red background in L608.
* While “ICLR” is abbreviated, other conferences like NeurIPS and ICML are listed in full; consider unifying this approach.

### Questions
* What the arrow of epoch 1 -> 10 in Figure 3 is different from others? Does it have a specific meaning?

* What does “data now shown in L466 mean? Does it mean that the authors intentionally did not include the results since it is insignificant? If so, please include it in the appendix.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose a new latent variable model for neuroscience that uses kernel density estimation to learn tuning curves from an unobserved latent variable for population spiking data. The model is fit using a closed form EM algorithm and is scalable to large datasets. The authors validate the model on simulated and real neural data.

### Strengths
The validation on synthetic and real neural data is a nice addition. The simplicity of the approach has potentially some appeal to experimentalists who want to avoid some of the more bespoke or complex latent variable models for the neural spiking data.

### Weaknesses
The authors provide a very brief overview a very broad field of latent variable models used in neuroscience to motivate their approach. They claim existing methods do not scale well, or have other shortcomings compared to SIMPL. To robustly demonstrate that this model indeed scales better than most or all of these existing approaches while retaining accurate latent and neural tuning identification would require far more comparisons, discussion and evaluation. 

In particular, the authors model is motivated in a similar way to the GPLVM, which is discussed but not compared to. As the authors point out GP based models often do have issues with scalability. However,  there have been many steps toward improving the scalability of these models in recent years -- note that 1 and 2 uses inducing points, a well-known approach to improve scalability in GP  models. These models however have the same 'tuning curve' interpretations that the authors posit SIMPL has, and so it would be important to directly compare to them for both scalability and neural tuning identification. 

1 "Manifold GPLVMs for discovering non-Euclidean latent structure in neural data" 
2. Learning interpretable continuous-time models of latent stochastic dynamical systems

There is also no mention of switching linear dynamical systems models, which often scale better than GP based methods due admitting an EM based approach and utilizing forward-backward passing (see e.g. 3, 4, but there are many others). 

3 Bayesian learning and inference in recurrent switching linear dynamical systems.
4. A general recurrent state space framework for modeling neural dynamics during decision-making.

Still there are others in this space that scale well and admit flexible non-linear latent characterizations. See for example 4, 5. and of course LFADS, as the authors discuss. The authors also mention Pi-VAE (and there are of course now many other VAE based models used in neuroscience), but they don't provide a principled comparison or discussion of these approaches. 

4 Inferring Latent Dynamics Underlying Neural Population Activity via Neural Differential Equations
5  Collapsed amortized variational inference for switching nonlinear dynamical systems.

The choice of CEBRA as the only benchmark is not well motivated. CEBRA is a fundamentally different model it uses behavioral information for contrastive learning, and it was designed to visualize a behaviorally-informed latent space. In this sense it isn't an unsupervised model used purely on spiking data (like SIMPL as well as the ones cited above, among others) and it does not permit a tuning curve interpretation -- which the authors emphasize is an important component of SIMPL and existing in many of these other LVMs. Comparison to the GPLVM, as well as switching LDS, or other nonlinear dynamical models would be far more appropriate then CEBRA. 

In it's current state, it is unclear exactly how much better SIMPL scales than any of the existing approaches, and how it's latent identification or tuning identification may compare to these many state-of-the-art methods.

### Questions
None

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a new method they call SIMPL, an EM-style algorithm, aiming to recover low-dimensional, time-evolving latent variables from high-dimensional neural activity in large neural datasets. Their  approach fits tuning curves to observed behaviour and iteratively refines these through a two-step process (EM like). The originality of SIMPL lies in its novel combination of expectation-maximization (EM) techniques with behavior-driven initialization, providing a straightforward yet effective approach to refine neural representations. It’s main advantage is that it is a scalable and fast approach compared to existing popular methods like CEBRA.

### Strengths
SIMPL has only two hyperparameters. Minimal hyperparameters make it practical for experimentalists.

Rigorous comparison with existing methods (CEBRA).It outperforms CEBRA (whose latent embedding was noisier and had larger final error) and  is over 30 faster.

The capability to achieve results over 30 times faster than comparable methods while maintaining accuracy makes SIMPL particularly attractive for large-scale neuroscience research

Careful experiments on synthetic as well as real data.

### Weaknesses
Limited Evaluation on Non-Spatial Tasks: While the paper demonstrates impressive results in the domain of spatial navigation, it would benefit from evaluation on other tasks, such as the ones tested in the CEBRA paper so that the two methods can be directly compared in more diverse settings.

Scalability Concerns for High-Dimensional Latents: The authors mention that kernel density estimation (KDE) may not scale well to high-dimensional latent spaces. This potential limitation needs more elaboration. What alternatives might be suitable when dealing with truly high-dimensional data.

### Questions
How does the method perform with smaller neural populations? Can SIMPL perform well when there are limited neurons recorded. Some downsampling analysis on neural data can be helpful to check if the method is sensitive to neuron number.

How much data is required to optimize SIMPL? How does the mother deal with short recordings? For example, would a short half hour recording be sufficient to fit the model?

The author mentioned that they trained CEBRA on the synthetic grid cell data using out-of-the-box hyperparameters training for the default 10000 iterations. How much of the performance inferiority of CEBRA VS SIMPL comes from not carefully searching through the hyperparameter space? Is the comparison fair? Would increasing the iterations improve the performance of CEBRA?


The author mentioned that SIMPL finds a modified latent space with smaller, more numerous, and more
uniformly-sized place fields, suggesting the brain may encode space with greater resolution than previously thought. How could the author disentangle the possibility that this observation is a result of artifacts by their method versus reflecting true representation feature of the hippocampus?

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
3

### Summary
The paper introduces SIMPL, an EM-style algorithm for refining latent variables and tuning curves from spiking neural data. The key idea is using behavior as initialization and combining kernel density estimation with Kalman smoothing in an iterative optimization framework. The authors validate their approach on synthetic datasets and real hippocampal recordings, showing improvements in both computational efficiency and biological interpretability compared to CEBRA (a recent popular deep learning approach for learning latent embeddings from neural and behavioural data)

### Strengths
- The paper is a fresh departure from current latent variable models relying on deep neural networks, with a number of practical advantages (e.g., fast optimization, minimal hyperparameters (only 2), simple implementation based on standard components (Kernel Density Estimation and Kalman smoothing))
- The paper reveals scientific insights through the use of the proposed method (finer structure in place field representations, new interpretation of place field size distribution, relationship between behavioral uncertainty and neural encoding)
- The authors do a good job presenting the approach clearly and acknowledging the limitations of the proposed method.

### Weaknesses
The proposed approach relies on strong assumptions that seem to limit the application of the method for more complex cases beyond the datasets used here (e.g., settings where behavioral recordings are missing, higher dimensional latent states, latent spaces not dominated by behavior, non-smooth latent trajectories). It would be great to demonstrate the applicability of the approach on commonly used datasets in the current literature on LVMs in neuroscience (e.g., Neural Latents Benchmark).

### Questions
- The authors mention improved identifiability of the proposed approach. Is that solely because of the behavioral initialization?
- The authors say: "we see SIMPL as a specific instance of a broader class of latent optimization algorithms, ..." and then move on to the idea of replacing KDE with neural networks. Can the authors elaborate on that?
- Can the authors provide the intuition for choosing the kernel bandwidth in the KDE step? This seems critical for the method's performance.

### Soundness
3

### Presentation
3

### Contribution
3
