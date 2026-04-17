# ODEBrain: Continuous-Time EEG Graph for Modeling Dynamic Brain Networks

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Modeling neural population dynamics is crucial for foundational neuroscientific research and various clinical applications. Conventional latent variable methods typically model continuous brain dynamics through discretizing time with recurrent architecture, which necessarily results in compounded cumulative prediction errors and failure of capturing instantaneous, nonlinear characteristics of EEGs. We propose ODEBrain, a Neural ODE latent dynamic forecasting framework to overcome these challenges by integrating spatio-temporal-frequency features into spectral graph nodes, followed by a Neural ODE modeling the continuous latent dynamics. Our design ensures that the latent representations can capture stochastic variations of complex brain states at any given time point. Extensive experiments verify that ODEBrain can improve significantly over existing methods in forecasting EEG dynamics with enhanced robustness and generalization capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes ODEBRAIN, a neural ODE latent dynamic forecasting framework with integrating spatio-temporal features into spectral graph nodes. The continuous latent dynamics was modeled by a neural ODE. The model shows efficiency in forecasting EEG dynamics.

### Strengths
The paper extends graph-based EEG modeling into the continuous domain using Neural ODEs, with separating spatial and temporal encoders, the method combines structured connectivity with temporal uncertainty. The proposed method does show better performance than several baselines.

### Weaknesses
1. The objective function used to train the algorithm is not clearly defined.
2. The method should include comparisons with other continuous-time baselines, such as BrainODE, which is already cited in the related works. Such comparisons would help demonstrate the specific advantages of introducing the graph-based formulation.
3. While the proposed method introduces a graph structure, it might be the main distinction from existing ODE approaches, the paper does not clearly explain how and why incorporating the graph improves forecasting of temporal dynamics.
4. The data is processed by STFT, however, the details of STFT are not included in the paper, and how do different parameters (such as window, frequency bins, log scaling) influence the performance of the model?
5. The paper does not discuss limitations of the proposed method.

### Questions
1. In 4.1, how to decide top-tau?
2. Some typo: line 219, space before ‘Consequently’. Line 21 and line 23, ‘verifies’ to ‘verify’.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This well-written paper introduces ODEBRAIN, a continuous-time latent dynamics framework for multi-channel EEG that: (1) builds spectral EEG graphs (where nodes are channels; edges are top-k correlations in STFT space); (2) uses a dual encoder to initialize a Neural ODE (NODE) with a deterministic graph descriptor and a stochastic temporal descriptor; and (3) forecasts future graph node embeddings via a graph-prediction head with multi-step loss. The authors additionally visualize the learned vector field and argue it is clinically interpretable (e.g., “centers” during seizures). They claim the formulation is the first to explicitly cast EEG brain networks as a continuous-time dynamical system governed by a NODE and show gains on two real world data sets over discrete baselines.

### Strengths
The paper argues discretized, windowed EEG pipelines miss inherently continuous dynamics, motivating a NODE approach and posing concrete challenges (robust initialization; meaningful trajectory objectives).

The dual-encoder (graph plus stochastic temporal) initialization and a graph-forecasting head with multi-step loss are well aligned with continuous latent trajectory learning. The forward ODE is standard and the projection aims at future graph prediction, not just signal.

Visualizations of the learned vector field highlight identifiable structures during seizure vs. non-seizure (e.g., attractor-like “centers” only during seizures), which is a promising narrative for clinical insight.

On TUSZ/TUAB EEG data, ODEBRAIN outperforms CNN-LSTM, BIOT, EvolveGCN, DCRNN, and AMAG in AUROC/F1. Both single-step and multi-step settings are reported and ablations probe initialization choices, loss design, and horizon.

Data, preprocessing, solver tolerance, training hyperparameters, and an anonymous code link are provided to aid reproducibility.

### Weaknesses
Although the latent dynamics are continuous via NODE, inputs and supervision remain epoched STFT segments and edges are top-k correlations per epoch. The model forecasts at discrete horizons (1s/3s/11s), and training targets are per-epoch graphs. 

Sensitivity analyses related to top-\tau sparsity and normalized correlations are not reported.

The baselines considered are primarily discrete TGNs/transformers. The related-work cites graph ODEs, but the empirical table omits them.

NODE solvers can be costly/unstable. You specify RK45 and tolerances, but there’s no wall-clock / NFEs / memory vs. discrete baselines, nor sensitivity to tolerances/horizons

### Questions
Please clarify in what sense the approach exceeds a finely sampled discrete model, beyond the solver’s internal substeps; e.g., can ODEBRAIN answer arbitrary-time queries between epochs evaluated against held-out high-rate labels?

Edges use normalized correlation with top-\taup sparsification. Correlation graphs are sensitive to noise and volume conduction; what happens with different \tau, similarity metrics, or regularizers (e.g., shrinkage/graphical lasso/lagged connectivity)?

Since the proposed benefit is about continuous-time superiority, why aren’t Latent ODE-style baselines (e.g., NODE on per-channel embeddings without graph, and graph-ODE methods) and irregular-sampling setups considered?

### Soundness
3

### Presentation
4

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
This paper introduces ODEBRAIN, a framework for modeling dynamic brain networks from multi-channel EEG. Unlike existing work modeling EEG dynamics in discrete-time manner, the method aims to learn the continuous-time representation of brain state evolution by leveraging Neural Ordinary Differential Equation (NODE). To address the challenges of applying NODEs to EEG, the authors propose a dual-encoder architecture to obtain robust initialization for ODE solver, combining spectral graph features with stochastic temporal signals. Then a novel objective function is proposed to capture underlying EEG dynamics. Experiments on two seizure detection datasets demonstrate model effectiveness.

### Strengths
- The paper addresses an important problem that of modeling brain network in continuous time (with NODE).  
- The novel contributions mainly come from the proposed dual-encoder architecture and objective loss. 
- The learned dynamic field demonstrate potential for qualitative analyses of brain networks.
- The proposed method shows some improvement, especially on TUSZ.

### Weaknesses
- Several Technical details of the proposed method are not clearly stated / explained. How do the the temporal descriptor \Psi and the objective \Omega are defined? How the pooling of latent continuous trajectory z_t for downstream task is conducted?
- The paragraph “RQ3 concerns consistency in the graphs…” seems confusing. From Fig.4 I cannot interpret the similarity scores or similarity matrices from discrete or continuous predictor, and other discussion. It seems the figure misaligned with the content.
- Hyperparameter selection is not discussed. Why the paper uses top 3 correlation neighbors for each node when constructing the graph? How about the effectiveness of different GNN architectures?
- The complexity analysis for the model is missing. How is the training/inference time of ODEBRAIN compared to other baselines.
- What is the motivation for objective loss to predict graph structure rather than EEG signal forecasting? The proposed loss shows effectiveness for the prediction task, how it would perform on the typical EEG signal forecasting?
- The discussion of prior work on modeling EEG signals, capturing their nonlinearity for a variety of brain activity mining [1][3][4][8][14], modeling and control for analyzing EEG signals in the context of brain machine interfaces and epileptic seizures [5][6][12][13][15], modeling latent variables in EEG [7][11] and other brain activity data [9][10], as well as related papers using graph ODE for EEG data [1] are all missing. This needs improvement in both discussing existing methods and comparing against these methods.
[1] JPM. Pijn et al., "Nonlinear dynamics of epileptic seizures on basis of intracranial EEG recordings." Brain topography 9, no. 4 (1997): 249-270.
[2] Y. Chen et al., "EEG emotion recognition based on ordinary differential equation graph convolutional networks and dynamic time wrapping." Applied Soft Computing 152 (2024): 111181.
[3] Y. Xue et al., "Minimum number of sensors to ensure observability of physiological systems: A case study." In 2016 54th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pp. 1181-1188. IEEE, 2016.
[4] K. Lehnertz et al. "Seizure prediction by nonlinear EEG analysis." IEEE Engineering in Medicine and Biology Magazine 22, no. 1 (2003): 57-63.
[5] G. Gupta et al., "Re-thinking EEG-based non-invasive brain interfaces: Modeling and analysis." In 2018 ACM/IEEE 9th International Conference on Cyber-Physical Systems (ICCPS), pp. 275-286. IEEE, 2018.
[6] V. Tzoumas et al., "Selecting sensors in biological fractional-order systems." IEEE Transactions on Control of Network Systems 5, no. 2 (2018): 709-721.
[7] G. Gupta et al., "Learning latent fractional dynamics with unknown unknowns." In 2019 American Control Conference (ACC), pp. 217-222. IEEE, 2019.
[8] K. Lehnertz, "Epilepsy and nonlinear dynamics." Journal of biological physics 34, no. 3 (2008): 253-266.
[9] R. Yang et al., "Data-driven perception of neuron point process with unknown unknowns." In Proceedings of the 10th ACM/IEEE International Conference on Cyber-Physical Systems, pp. 259-269. 2019
[10] R. Yang et al., "Spiking dynamics of individual neurons reflect changes in the structure and function of neuronal networks." Nature Communications 16, no. 1 (2025): 6994
[11] G. Gupta et al., "Dealing with unknown unknowns: Identification and selection of minimal sensing for fractional dynamics with unknown inputs." In 2018 Annual American Control Conference (ACC), pp. 2814-2820. IEEE, 2018.
[12] X. Lu, "Detection and classification of epileptic EEG signals by the methods of nonlinear dynamics." Chaos, Solitons & Fractals 151 (2021): 111032.
[13] R. Martis et al. "Epileptic EEG classification using nonlinear parameters on different frequency bands." Journal of Mechanics in Medicine and Biology 15, no. 03 (2015): 1550040.
[14] M. Mercier et al. "The value of linear and non-linear quantitative EEG analysis in paediatric epilepsy surgery: a machine learning approach." Scientific reports 14, no. 1 (2024): 10887.
[15] G. Lepeu et al. "The critical dynamics of hippocampal seizures." Nature communications 15, no. 1 (2024): 6945.
Claiming that the novelty of this approach is that it captures the connectivity of spatio-temporal graphs to capture nonstationary changes is incorrect as can be seen from existing works that do just that.
- The paper states that the proposed approach “combines deterministic graph-based features with stochastic EEG representations to produce a robust initial state” but it is unclear how the stochastic EEG is accounted for.
- Can we always assume that the data supports that f_theta is continuous and differentiable function? Some nonlinearity of EEG may not support this assumption.
- The parameters N, d and T are barely mentioned in passing in section 4.2 but how they are selected in the case study and some guidelines to select them are missing.

### Questions
- Can we always assume that the data supports that f_theta is continuous and differentiable function? Some nonlinearity of EEG may not support this assumption.
- Several Technical details of the proposed method are not clearly stated / explained. How do the the temporal descriptor \Psi and the objective \Omega are defined? How the pooling of latent continuous trajectory z_t for downstream task is conducted?
- The paragraph “RQ3 concerns consistency in the graphs…” seems confusing. From Fig.4 I cannot interpret the similarity scores or similarity matrices from discrete or continuous predictor, and other discussion. It seems the figure misaligned with the content.
- Hyperparameter selection is not discussed. Why the paper uses top 3 correlation neighbors for each node when constructing the graph? How about the effectiveness of different GNN architectures?
- The complexity analysis for the model is missing. How is the training/inference time of ODEBRAIN compared to other baselines.
- What is the motivation for objective loss to predict graph structure rather than EEG signal forecasting? The proposed loss shows effectiveness for the prediction task, how it would perform on the typical EEG signal forecasting?
- The discussion of prior work on modeling EEG signals, capturing their nonlinearity for a variety of brain activity mining [1][3][4][8][14], modeling and control for analyzing EEG signals in the context of brain machine interfaces and epileptic seizures [5][6][12][13][15], modeling latent variables in EEG [7][11] and other brain activity data [9][10], as well as related papers using graph ODE for EEG data [1] are all missing. This needs improvement in both discussing existing methods and comparing against these methods.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents ODEBRAIN, a novel continuous-time EEG graph framework built upon Neural Ordinary Differential Equations (NODEs) to model dynamic brain networks. To tackle key challenges in learning temporal brain dynamics, the study introduces three main contributions.

First, a dual-encoder architecture is proposed to effectively initialize NODEs: one encoder extracts deterministic frequency-domain features to represent brain connectivity, while the other processes raw EEG signals to preserve stochastic variability. Their integration provides robust spatiotemporal representations for initializing the ODE solver.

Second, a trajectory forecasting decoder is designed to reconstruct graph structures from NODE latent trajectories. By incorporating a multi-step forecasting loss, the model explicitly predicts the evolution of brain networks over time, enabling accurate and continuous trajectory modeling.

Third, the paper introduces a novel gradient field–based metric derived from NODEs to quantify the dynamics of EEG brain networks. A case study on seizure data demonstrates the clinical interpretability and practical value of this approach.

### Strengths
- The paper writing is generally good although some parts can be further improved.
- The idea of using ODE solver in forecasting and predicting graph structure are interesting.
- The experimental results are good.

### Weaknesses
- Some parts are not clear.
- No discussion of the computational cost.
- No discussion of the architecture of $f_\theta$.

### Questions
- What is GRU (Gated Recurrent Unit)?
- In the formula of $z^g$ in line 260, is it should be $z^g_i$?
- Can you explain more $z^s$? Why is it called stochastic temporal embedding?

### Soundness
3

### Presentation
3

### Contribution
3
