# BayesENDS: Bayesian Electrophysiological Neural Dynamical Systems for Alzheimer’s Disease Diagnosis

- Avg Score: 3.20
- Decision: Reject
- Scores: 4, 2, 2, 6, 2

## Abstract
Alzheimer’s disease (AD) alters Electroencephalogram (EEG) through slowed oscillations and diminished neural drive, yet most AD-EEG pipelines are black-box classifiers, lacking a unifying mathematical account of how both neural activity and its interaction dynamics evolve over time. We introduce BayesENDS, a Bayesian electrophysiological neural dynamical system that explores the possibility of incorporating neuron spiking mechanisms into a Bayesian neural dynamical system. By introducing a differentiable leaky-integrate-and-fire (dLIF) prior, BayesENDS is capable of inferring population events and interaction dynamics directly from EEG—without spike or interaction annotations. The dLIF prior encodes membrane dynamics, rate/refractory constraints, and physiologically plausible frequency ranges, improving identifiability while yielding biologically plausible, subject-level biomarkers alongside AD predictions. Across synthetic event-sequence benchmarks and real AD EEG datasets, BayesENDS delivers superior performance to state-of-the-art baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Interesting idea (biophysically-informed latent events + graph).

The paper has some issues in evaluation design, missing experimental detail, and theoretical it’s not yet scientifically reliable.

### Strengths
Clear, modular presentation of components (EPDE, MELP, ERG).

### Weaknesses
1)The paper describes cohorts but does not specify subject-wise vs. epoch-wise splitting, preprocessing, artifact rejection, windowing, or cross-validation protocol—all crucial to avoid train/test leakage in EEG (e.g., multiple windows from the same subject).

2)Table 2 lists Accuracy/F1 only, yet the text claims gains in AUC from the priors (“boosts accuracy and AUC… raises F1/AUC”), which are not reported anywhere—a red flag.

3)Figure 1 shows EEGNet embeddings feeding BayesENDS, but it’s unspecified whether EEGNet is trained from scratch, frozen, or fine-tuned, and how its supervision interacts with the “unsupervised” event latents.

4)Only two AD datasets tested, with relatively small sample sizes (88 and 168 participants)

5)The paper claims the dLIF prior provides "biophysically plausible" dynamics, but this isn't  validated. The connection between scalp EEG and single-neuron LIF dynamics is tenuous at best, as EEG reflects aggregate activity from millions of neurons.

### Questions
If multiple windows were extracted from each subject, how did you ensure no data leakage between splits?

What was the window length and overlap used for segmenting continuous EEG recordings?

Table 2 reports only Accuracy/F1, but the text claims "boosts accuracy and AUC" and "raises F1/AUC." Where are the AUC values?

Can you provide complete performance metrics including sensitivity, specificity, and AUC with confidence intervals?

Were statistical significance tests performed between methods?

Is the EEGNet component pre-trained, trained from scratch, or fine-tuned jointly with BayesENDS?If pre-trained, on what dataset and task?

How does the supervised EEGNet training interact with the claimed "unsupervised" event discovery?

Are the EEGNet embeddings frozen during BayesENDS training or updated end-to-end?

How to validate issue 5 in the Weaknesses?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes BayesENDS, a Bayesian electrophysiology-inspired neural dynamical system that infers latent event timing per channel with a differentiable LIF prior, and (iii) learns a event-relational graph (ERG) based on inferred event timing. These are used as features for downstream Alzheimer prediction. This paper shows great performance in simulations studies and 2 Alzheimer datasets.

### Strengths
* Interpretable latents & graphs. The method is original and well-motivated by EEG biophysics. The dLIF prior and the ERG produce physiology-plausible latents and network patterns; the paper shows chord diagrams with plausible AD connectivity and diverse dLIF frequency distributions, supporting interpretability claims. 
* High novelty and clear theory hook. The KL to the dLIF event prior is handled via a tractable IVP-based bound evaluated during training, which is a neat theoretical contribution.

### Weaknesses
* Reproducibility gaps. Consider the complexity of the method, the paper does not provide enough detail for the paper to be reproduced. Including but not limited to optimization details, parameterization details, hyper-parameter tuning. All the details regarding evaluation is also omitted, including but not limited to splits for the real EEG experiments, 2 way or 3 way classification, how are metrics calculated.
* Subpar performance on public datasets. Based on some literature search on the datasets, several methods report substantially higher accuracies (often ≥ 90% [1] and up to the mid-90s) under various protocols, whereas BayesENDS reports 75.03% on Cohort A and 89.82% on Cohort B (Table 1). As the paper omits all the validation details, it is hard to evaluate the performance of the proposed method compared to these existing methods. Please perform a **thorough** literature review, compare and discuss these results.

[1] Zheng, Xiaowei, et al. "Diagnosis of Alzheimer’s disease via resting-state EEG: integration of spectrum, complexity, and synchronization signal features." Frontiers in aging neuroscience 15 (2023): 1288295.

### Questions
1. Is the plus sign for equation (3) a typo? Shouldn't the regularizes be minimized? 
2. Please clearly state 6.1 is with simulation study described in supplementary.

### Soundness
2

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
3

### Summary
The paper presents BayesENDS, a novel approach to classify Alzheimer’s disease (AD) from EEG data based on a novel Bayesian electrophysiological neural dynamical system. The problem is formulated as unsupervised latent-event and relation discovery in multi-channel EEG time series for labelled sequences (AD or no AD), in which BayesENDS learns channel-wise latent event dynamics p^((n)) and relational/graphical structure among channels G^((n)), which act as inputs to a downstream predictor for classification of AD versus no AD. The Bayesian neural dynamical system is trained end-to-end with a variational objective and consists of 1) an Event Posterior Differential Equation (EPDE) which yields next-event times; 2) a Mean-Evolving Lognormal Process (MELP) which samples inter-event intervals and the means of the log-normal mixture are parametrized by the outputs of the EPDE with reparametrized sampling; 3) a differentiable leaky-integrate-and-fire (dLIF) prior which provides biophysical rate and refractory constraints as well as plausible frequency ranges; and 4) a directed event-relational graph (ERG) prior which maps cross-channel event lags. The paper discusses the theorems of the proposed approach and empirical experiments. BayesENDS shows improved performance in comparison to baseline models on synthetic and real AD EEG datasets and provides biologically plausible biomarkers alongside AD predictions.

### Strengths
•	The paper addresses a relevant problem that is widely investigated in the field (identification of neurodegenerative diseases from non-invasive EEG data). 
•	The paper presents a novel approach by developing a Bayesian dynamical system to classify Alzheimer’s disease from EEG signals. 
•	The approach adds interpretable biomarkers to classification, providing clinicians with helpful additional information. 
•	The ablation studies aim to disentangle the contribution of the two priors of BayesENDS to the predictions.

### Weaknesses
•	In its present form, it is impossible to evaluate whether the paper presents meaningful results as too much information on the experimental design and results is omitted (see questions for an overview of missing information). 
•	The contextualization relative to prior work is very limited. 
•	The presentation of the results is chaotic and unclear. In particular, the results refer to information that is in the appendix (e.g. Figure 4). but without referring to the appendix. This gives the impression that referenced figures are non-existent and presented in an illogical order. 
•	The results of the ablation studies show a large variance (standard deviation) across runs. Although the paper does not define what a “run” is here, these results question the relative importance of each prior for the results. 
•	The biomarkers – differences in frequency distributions and EEG connectivity graphs – are only inspected visually. The claim that these are indeed interpretable biomarkers in EEG time series for AAD classification would be more convincing if the differences between groups (e.g. AD and no AD) in terms of these biomarkers are quantified. This is especially relevant because there appears to be considerable overlap between groups in e.g. frequency distribution and hence the “clear association” would be clearer if supported through quantification.

### Questions
The writing is clear, but the overall organization of the paper is lacking in terms of contextualization relative to prior work (very limited), details on experimental procedures (crucial information is omitted), and presentation of the results (chaotic, unclear). The manuscript includes abbreviations without introduction (for example, IVP and KL on p. 2, STRODE on p. 7). The results refer to information that is in the appendix but without referring to the appendix, giving the impression that referenced figures are non-existent and presented in an illogical order. 

As the paper addresses a relevant problem (Alzheimer’s disease classification from EEG signals) with an original approach that adds interpretable biomarkers to predictions, the paper could present a relevant contribution to the field. However, the paper omits too much information to enable evaluating the quality of the approach and the results (see below). 

•	Figure 1 shows that EEGnet is used to extract embeddings from the raw EEG signals, yet this is not described in the manuscript. What is the motivation for this approach? And what are the EEGnet specifications? Is this a pre-trained EEGnet or is EEGnet trained in the end-to-end pipeline?
•	What are the training parameters for BayesENDS for both datasets (i.e. data splits, cross-validation, training parameters)?
•	What are the training details for the benchmark models? And were these trained on raw EEG time-series or on EEGnet embeddings as well?
•	In Table 1, please provide a measure of variance by including the variance across “runs” (as shown in Table 2). Add a definition of “runs”?
•	How are the ablation experiments implemented? 
•	To what extent are the interpretable biomarkers (e.g. frequency distribution, connectivity) usable at the single-subject level to aid interpretation of a classification result? That is, what is the variability for these biomarkers?

### Soundness
2

### Presentation
1

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
The authors present a neural dynamical system to model EEG recordings. The model infers latent events using a physically plausible differentiable leaky-integrate-and-fire prior as well as a event relational graph prior. The model is trained in an unsupervised manner with a variational objective. The relational graph, latent priors and observed data are then passed to a prediction task between healthy controls and dementia afflicted recordings. The presented classifier outperforms other state of the art models (LSTM, transformers, convnets) on two datasets. Ablation studies show the spiking prior with a bigger lift than the graph prior, with both combined yielding the best results.
The authors then argue for the interpretability of the model showing (1) KDEs of frequency distributions of the spiking prior and (2) EEG connectivity graphs based on good old Pearson versus Bayesends. In the KDEs, the authors recover an expected result of higher theta/delta bands and lower alpha/beta in recordings from Alzheimer patients.

### Strengths
- clear performance increase against previously established baselines
- thorough theoretical description of the method
- Demonstration of the interpretability of the inferred priors showing their frequency distributions modulated per control and disease groups.
- the comparison of recovery of boundary times using STRODE versus BayesENDS on synthetic data is an asset, and shows due diligence toy dataset / synthetic proof of concept before scaling to real data

### Weaknesses
- The discussion is terse on interpretability. For example it is not clear what the graph on EEG connectivity brings for interpretability. See questions below for precise items.
- The recovery of boundary time cross plots is introduced abrubtly and the synthetic data protocol only mentioned in the appendix. This would probably need more context in the main text. Additionally the corresponding Figure 4 is buried in the appendix. It looks like the authors have moved this part between the main text and the appendix and left some information behind.

Minor comments:
- define IVP and STRODE
- Figure 4 is buried in the appendix in the review pdf

### Questions
- Connectivity figure: we can see a stronger similarity between frontotemporal dementia and Alzheimer in the BayesENDS connectivity versus Pearson but is it an expected result ?
-  Another example is with the frequency distributions. Would someone see the shift in central frequency by looking at the EEG trace itself ?  Is BayesENDS bringing better insights above and beyond such simple spectral analysis for already diagnosed patients ?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces BayesENDS for interpretable AD diagnosis from EEG. The model integrates electrophysiological principles into a probabilistic framework that infers latent neural events and a time-varying interaction graph directly from multichannel EEG data. Empirically, the model outperforms CNN, RNN, attention-based, and transformer-based EEG baselines across 2 public AD datasets, and has higher accuracy while producing interpretable physiological biomarkers.

### Strengths
1. This paper proposes a new Bayesian neural dynamical system for AD diagnosis using EEG data.
2. The method shows consistent gains over diverse baselines across multiple real-world EEG datasets.

### Weaknesses
1. Although the method tries to combine ephys-inspired mechanisms into a model for EEG data, the motivation is not very clear. Since EEG has poor temporal resolution, it may not be appropriate to make such a connection. I would appreciate some clarification of the motivation behind this.

2. The writing is not very clear, and the visualizations can be improved. Many paragraphs are not fully developed (e.g., the related work section).

3. Although the results are promising, the experiments are limited to EEG-based AD datasets. Testing on other neurodegenerative or cognitive tasks would demonstrate that the proposed method is robust.

### Questions
1. Given that the model involves multiple variational components, how computationally demanding is it? Could the authors provide a theoretical or empirical runtime comparison with baseline methods?

2. While the inferred latent frequencies and connectivity patterns appear biologically plausible, can the authors validate these findings against ground truth (for example, through simulation) to further support the interpretability claims?

### Soundness
2

### Presentation
2

### Contribution
1
