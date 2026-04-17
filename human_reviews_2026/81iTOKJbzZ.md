# Spatially and Temporally Guided Bayesian Optimization for Brain Effective Connectivity Learning from fMRI and EEG Data

- Decision: Reject
- Scores: 2, 4, 0, 2

## Abstract
Brain effective connectivity (EC) characterizes the causal and directional interactions among brain regions and plays a central role in understanding cognition and neurological disorders. Constructing EC networks from multimodal neuroimaging such as functional Magnetic Resonance Imaging (fMRI) and electroencephalography (EEG) is challenging, since most existing methods rely on feature concatenation or linear mapping, neglecting structural consistency and nonlinear cross-modal dynamics. In this work, we propose STBO-EC, a spatially and temporally guided framework for multimodal EC learning. First, we develop an anatomy-informed spatial alignment strategy that leverages known brain region coordinates to establish structurally consistent correspondences between EEG electrodes and fMRI regions. Second, we design a time-slice-based alignment and fusion mechanism to effectively bridge the temporal resolution gap between fast EEG activity and slow fMRI signals. Finally, to tackle the high dimensionality and nonlinear dependencies of fused multimodal data, we employ a low-rank parameterized Bayesian optimization method (DrBO), which enables efficient exploration of the exponential EC search space while providing uncertainty-aware inference. Experiments on two real EEG–fMRI datasets demonstrate that STBO-EC consistently outperforms state-of-the-art baselines across multiple evaluation metrics.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This study proposes a multimodal brain effective connectivity learning framework—**STBO-EC**, which aims to integrate EEG and fMRI signals to more accurately characterize the causal interactions between brain regions. The method achieves spatial and temporal alignment respectively and employs a Bayesian optimization approach to infer the causal connectivity network among brain regions.

### Strengths
A substantive assessment of the strengths of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage reviewers to be broad in their definitions of originality and significance. For example, originality may arise from a new definition or problem formulation, creative combinations of existing ideas, application to a new domain, or removing limitations from prior results.
The article has a well-organized structure and reads smoothly. The authors provide abundant visualization results and offer a fairly comprehensive discussion of the performance of the proposed method.

### Weaknesses
However, this paper presents a multi-stage training framework, which is more complex compared to end-to-end approaches. Moreover, each component tends to directly apply existing methods, resulting in limited algorithmic innovation. For more details on other issues, please refer to the *Question* section.

### Questions
1. In the spatial alignment task between EEG and fMRI, the authors apply an existing Gaussian kernel function to this alignment scenario, which lacks algorithmic innovation.
2. In the temporal alignment task, the authors design a novel loss function to fuse the two modalities. However, I question whether directly weighting the two modalities within the loss function is appropriate, since the physiological meanings and amplitude scales of the two data types may differ substantially.
3. In the final EC (effective connectivity) learning stage, the authors employ a Bayesian optimization-based structure learning method. Again, this represents an application-level innovation rather than a true algorithmic contribution.
4. Although the authors provide a relatively rich set of experimental results, I do not recommend merging the ablation and comparison experiments into a single table. In fact, it is unclear to me where the ablation experiments begin in Tables 1 and 2. The authors should clarify this labeling.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes STBO-EC, a three-module pipeline to infer effective connectivity (EC) from simultaneous EEG–fMRI:
 1. Neuroanatomy-guided spatial alignment projects EEG channels to fMRI ROI space with a Gaussian weighting based on electrode–ROI distances
 2. Block-wise temporal alignment & fusion bridges the sampling-rate gap by slicing the EEG to match fMRI frames.
 3. BO-based EC learning uses DrBO to optimize a decomposable score and return G ̂ = (V, E ̂, W ̂) 
Experiments are run on two real datasets: the visual categorization dataset and the XP2 dataset.

### Strengths
Key strengths include:
•	Well-structured multimodal fusion that enforces spatial correspondence and temporal synchronization.
•	Efficient causal search via DrBO
•	Clear algorithmic pseudocode and hyperparameter disclosure

### Weaknesses
The Paper has several weaknesses as listed below:
•	Novelty is mostly integrative. The core causal learner (DrBO) and the alignment pieces are known; the main contribution is the combination for EEG–fMRI EC.
•	Performance of the STBO-EC framework is modest on Visual Categorization.
•	Sensitivity to alignment choices (e.g., kernel width σ, fusion weight α=0.5) is not fully explored beyond general hyperparameter plots.

### Questions
1. Deeper sensitivity & ablations on the fusion design: vary σ and α, compare learned fusion vs. fixed α, and report impact on accuracy/AUC.
 2. Several ROI acronyms (e.g., FFA, PPA, SPL) are defined only in the appendix. For readability, please define each acronym at first mention in the main text (and figure captions).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
2

### Summary
The authors consider the problem of inferring brain effective connectivity using simultaneously recorded fMRI and EEG data.  The authors propose using a non-linear mapping to fuse EEG and fMRI data.  The authors then use a Bayesian optimization based directed acyclic graph (DAG) search method to identify effective connectivity.  The authors apply their method on two data sets with simultaneously recorded data and obtain encouraging results compared to baselines.

### Strengths
- The overall problem of working with multi-modal brain data, fMRI and EEG, is important and interesting
- The problem is also inherently challenging, as both types of data have significant signal processing challenges
- The authors’ experiments show encouraging results against a number of baselines

### Weaknesses
### Major
There does not appear to be technical novelty to the proposed approach to support the significance of the claimed contributions.  The modeling decisions appear to me to not be well justified in terms of the signal processing challenges of working with fMRI and EEG data.  There is no detailed discussion of relations to past works in the method.
- Section 3.1 (contribution 1) the authors state “The fundamental challenge in EEG-fMRI integration lies in bridging the spatial gap between scalp electrodes and brain regions” and they use a “biophysically-motivated projection that respects the underlying neuroanatomy.”  Yet the authors use a simple weighted averaging of the EEG signals based on Euclidean distances.  Without further justification it is not clear to me at all that this is meaningful.  There is a literature on EEG localization but in this paper there are *no discussions* in Section 3.1 how this is different from other attempts to infer localization of EEG recorded signals.
- Section 3.2  (contribution 2) 
    - for temporal alignment the authors simply replicate the fMRI value, then simply take an average of the EEG and fMRI values.   There is no discussion about whether this “fusion” accounts for the differences in measure units / signal scales between EEG and FMRI.  There is no discussion about preprocessing to normalize the data before combining. And how does the choice for \alpha affect the performance of the model? Further discussion and ablation study is important to understand the effect of these choices.
    - Confusingly, in (11) the authors propose training a non-linear mapping to minimize the MSE wrt the _known_ average of EEG and fMRI.   There is no discussion about what function class is used in this section.  This appears to be the main use of non-linearity which the authors criticize past works for not accounting for.
    - Lastly, in (14) a simple temporal smoothing is used.
- Section 3.3 (contribution 3)
    - The authors appear to apply a method DrBO to the fused signal.  
    - Duong et al is not cited in this section and there is no discussion if there are technical challenges addressed in applying it. (Section 2.3 does include a citation and background about the method, though also no discussion if the use of DrBO was straightforward or there were challenges overcome).   
    - the authors’ choice of acyclic graphs appears limiting, as feedback between brain regions cannot be captured.  There is no discussion of this
    - the likelihood $p(D| \theta, G)$ is denoted but never described – what distributions are you considering?  The choice of what family will likely play an important role in the modeling accuracy and computational complexity of the method, but there is no discussion.  


### Very Minor
- line 057 “did not achieve genuine integration” is too vague
- line 130 issue with Duong et al ref citation (no year displaying; should be parenthetical)

### Questions
(Several questions regarding technical novelty and method design are included in the Weaknesses section)

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The method feels like engineering stacking; current evidence does not substantiate the two core claims of practical value and generalizability.

Main Reasons
1) Insufficient evidence of “practical significance”

Only a proxy task. The authors acknowledge there is no EC ground truth in real data, so they evaluate solely by using the learned EC as features for a classifier (on both datasets). This does not show that the learned structure can guide any real decision or neuromodulation; it only shows that “the features help a particular classifier somewhat.” 

review

Small gains on a weak task. On the visual dataset the top accuracy is 0.42 vs. 0.39 for the runner-up; the paper itself concedes the overall accuracy is “not high.” Such increments are hard to translate into any meaningful application benefit. 

review

 

review

Experimental setup far from real use. One dataset has only 6 ROIs (very low spatial resolution); the other covers only the upper half of the brain (incomplete structural coverage). This can’t support claims about real closed-loop neurofeedback, clinical prediction, or large-scale cognitive studies. 

review

 

review

Physiology-aware signal handling is thin. Temporal “alignment” literally replicates each fMRI frame to match EEG blocks for fusion (rather than HRF deconvolution/lag modeling), undermining physiological interpretability and transferability. 

review

Bottom line: the paper shows no “real-world task” improvement (e.g., better neurofeedback outcomes, diagnostic/prognostic lift, or improved intervention planning). Thus the claimed “practicality” lacks a verifiable anchor.

2) Poor generalization (to datasets, subjects, tasks, and sites)

Very narrow data scope. Validation is limited to two public datasets; the visual set has only 93 multimodal samples. Sample size and task diversity are insufficient for broad generalization claims. 

review

 

review

No cross-dataset/site/task transfer. There is no OOD or cross-domain evaluation (across devices, centers, paradigms), and it is unclear whether strict subject-wise splits were used—only generic “5-fold RF / 10-fold KNN” is described. For multimodal neuroimaging, this is far from adequate. 

review

Authors themselves concede generalization/scale limits. The conclusion/limitations sections state that robustness and generalizability need strengthening, and that scalability to larger/higher-res data remains open. 

review

 

review

Heavy computation and complex hyperparameters. The method uses 100k candidates and 2000 iterations, with no clear scaling curves vs. node count/rank; this is unfriendly for moving to larger atlases or clinical-scale cohorts. 

review

Bottom line: current evidence only shows the method can slightly lift a classifier on two very similar setups; there is no quantification of robustness to cross-population/site/paradigm shifts—so “generalizability” is unsubstantiated.

Minor Issues (that further weaken the case)

Much of the neurophysiological interpretation is descriptive, lacking testable hypotheses and statistical designs. 

review

Most metrics are classification metrics; there are no graph-level EC quality measures (e.g., edge-PR, SHD/SID), making the claim “we learned better EC” difficult to support. 

review

Suggestions (for a major revision or a different venue)

Design real, actionable tasks: show operational gains in neurofeedback outcomes, clinical stratification/prognosis, or behavioral prediction—rather than only EC→classification proxy evaluation.

Systematic generalization testing: cross-dataset/site/task/device with strict subject-wise splits; report OOD/domain-shift performance and uncertainty.

Physiology-consistent temporal modeling: include HRF deconvolution/lag-aware baselines and ablations; avoid the “frame replication” misalignment. 



Scale & efficiency: provide wall-clock and memory curves vs. #ROIs/rank/sample size, and validate on higher-resolution atlases. 



Direct EC quality metrics: on (semi-)synthetic or interventional data, report graph-level measures and include misalignment/shuffling controls to prove the necessity of the alignment/fusion modules.

### Strengths
The method feels like engineering stacking; current evidence does not substantiate the two core claims of practical value and generalizability.

Main Reasons
1) Insufficient evidence of “practical significance”

Only a proxy task. The authors acknowledge there is no EC ground truth in real data, so they evaluate solely by using the learned EC as features for a classifier (on both datasets). This does not show that the learned structure can guide any real decision or neuromodulation; it only shows that “the features help a particular classifier somewhat.” 

review

Small gains on a weak task. On the visual dataset the top accuracy is 0.42 vs. 0.39 for the runner-up; the paper itself concedes the overall accuracy is “not high.” Such increments are hard to translate into any meaningful application benefit. 

review

 

review

Experimental setup far from real use. One dataset has only 6 ROIs (very low spatial resolution); the other covers only the upper half of the brain (incomplete structural coverage). This can’t support claims about real closed-loop neurofeedback, clinical prediction, or large-scale cognitive studies. 

review

 

review

Physiology-aware signal handling is thin. Temporal “alignment” literally replicates each fMRI frame to match EEG blocks for fusion (rather than HRF deconvolution/lag modeling), undermining physiological interpretability and transferability. 

review

Bottom line: the paper shows no “real-world task” improvement (e.g., better neurofeedback outcomes, diagnostic/prognostic lift, or improved intervention planning). Thus the claimed “practicality” lacks a verifiable anchor.

2) Poor generalization (to datasets, subjects, tasks, and sites)

Very narrow data scope. Validation is limited to two public datasets; the visual set has only 93 multimodal samples. Sample size and task diversity are insufficient for broad generalization claims. 

review

 

review

No cross-dataset/site/task transfer. There is no OOD or cross-domain evaluation (across devices, centers, paradigms), and it is unclear whether strict subject-wise splits were used—only generic “5-fold RF / 10-fold KNN” is described. For multimodal neuroimaging, this is far from adequate. 

review

Authors themselves concede generalization/scale limits. The conclusion/limitations sections state that robustness and generalizability need strengthening, and that scalability to larger/higher-res data remains open. 

review

 

review

Heavy computation and complex hyperparameters. The method uses 100k candidates and 2000 iterations, with no clear scaling curves vs. node count/rank; this is unfriendly for moving to larger atlases or clinical-scale cohorts. 

review

Bottom line: current evidence only shows the method can slightly lift a classifier on two very similar setups; there is no quantification of robustness to cross-population/site/paradigm shifts—so “generalizability” is unsubstantiated.

Minor Issues (that further weaken the case)

Much of the neurophysiological interpretation is descriptive, lacking testable hypotheses and statistical designs. 

review

Most metrics are classification metrics; there are no graph-level EC quality measures (e.g., edge-PR, SHD/SID), making the claim “we learned better EC” difficult to support. 

review

Suggestions (for a major revision or a different venue)

Design real, actionable tasks: show operational gains in neurofeedback outcomes, clinical stratification/prognosis, or behavioral prediction—rather than only EC→classification proxy evaluation.

Systematic generalization testing: cross-dataset/site/task/device with strict subject-wise splits; report OOD/domain-shift performance and uncertainty.

Physiology-consistent temporal modeling: include HRF deconvolution/lag-aware baselines and ablations; avoid the “frame replication” misalignment. 



Scale & efficiency: provide wall-clock and memory curves vs. #ROIs/rank/sample size, and validate on higher-resolution atlases. 



Direct EC quality metrics: on (semi-)synthetic or interventional data, report graph-level measures and include misalignment/shuffling controls to prove the necessity of the alignment/fusion modules.

### Weaknesses
The method feels like engineering stacking; current evidence does not substantiate the two core claims of practical value and generalizability.

Main Reasons
1) Insufficient evidence of “practical significance”

Only a proxy task. The authors acknowledge there is no EC ground truth in real data, so they evaluate solely by using the learned EC as features for a classifier (on both datasets). This does not show that the learned structure can guide any real decision or neuromodulation; it only shows that “the features help a particular classifier somewhat.” 

review

Small gains on a weak task. On the visual dataset the top accuracy is 0.42 vs. 0.39 for the runner-up; the paper itself concedes the overall accuracy is “not high.” Such increments are hard to translate into any meaningful application benefit. 

review

 

review

Experimental setup far from real use. One dataset has only 6 ROIs (very low spatial resolution); the other covers only the upper half of the brain (incomplete structural coverage). This can’t support claims about real closed-loop neurofeedback, clinical prediction, or large-scale cognitive studies. 

review

 

review

Physiology-aware signal handling is thin. Temporal “alignment” literally replicates each fMRI frame to match EEG blocks for fusion (rather than HRF deconvolution/lag modeling), undermining physiological interpretability and transferability. 

review

Bottom line: the paper shows no “real-world task” improvement (e.g., better neurofeedback outcomes, diagnostic/prognostic lift, or improved intervention planning). Thus the claimed “practicality” lacks a verifiable anchor.

2) Poor generalization (to datasets, subjects, tasks, and sites)

Very narrow data scope. Validation is limited to two public datasets; the visual set has only 93 multimodal samples. Sample size and task diversity are insufficient for broad generalization claims. 

review

 

review

No cross-dataset/site/task transfer. There is no OOD or cross-domain evaluation (across devices, centers, paradigms), and it is unclear whether strict subject-wise splits were used—only generic “5-fold RF / 10-fold KNN” is described. For multimodal neuroimaging, this is far from adequate. 

review

Authors themselves concede generalization/scale limits. The conclusion/limitations sections state that robustness and generalizability need strengthening, and that scalability to larger/higher-res data remains open. 

review

 

review

Heavy computation and complex hyperparameters. The method uses 100k candidates and 2000 iterations, with no clear scaling curves vs. node count/rank; this is unfriendly for moving to larger atlases or clinical-scale cohorts. 

review

Bottom line: current evidence only shows the method can slightly lift a classifier on two very similar setups; there is no quantification of robustness to cross-population/site/paradigm shifts—so “generalizability” is unsubstantiated.

Minor Issues (that further weaken the case)

Much of the neurophysiological interpretation is descriptive, lacking testable hypotheses and statistical designs. 

review

Most metrics are classification metrics; there are no graph-level EC quality measures (e.g., edge-PR, SHD/SID), making the claim “we learned better EC” difficult to support. 

review

Suggestions (for a major revision or a different venue)

Design real, actionable tasks: show operational gains in neurofeedback outcomes, clinical stratification/prognosis, or behavioral prediction—rather than only EC→classification proxy evaluation.

Systematic generalization testing: cross-dataset/site/task/device with strict subject-wise splits; report OOD/domain-shift performance and uncertainty.

Physiology-consistent temporal modeling: include HRF deconvolution/lag-aware baselines and ablations; avoid the “frame replication” misalignment. 



Scale & efficiency: provide wall-clock and memory curves vs. #ROIs/rank/sample size, and validate on higher-resolution atlases. 



Direct EC quality metrics: on (semi-)synthetic or interventional data, report graph-level measures and include misalignment/shuffling controls to prove the necessity of the alignment/fusion modules.

### Questions
The method feels like engineering stacking; current evidence does not substantiate the two core claims of practical value and generalizability.

Main Reasons
1) Insufficient evidence of “practical significance”

Only a proxy task. The authors acknowledge there is no EC ground truth in real data, so they evaluate solely by using the learned EC as features for a classifier (on both datasets). This does not show that the learned structure can guide any real decision or neuromodulation; it only shows that “the features help a particular classifier somewhat.” 

review

Small gains on a weak task. On the visual dataset the top accuracy is 0.42 vs. 0.39 for the runner-up; the paper itself concedes the overall accuracy is “not high.” Such increments are hard to translate into any meaningful application benefit. 

review

 

review

Experimental setup far from real use. One dataset has only 6 ROIs (very low spatial resolution); the other covers only the upper half of the brain (incomplete structural coverage). This can’t support claims about real closed-loop neurofeedback, clinical prediction, or large-scale cognitive studies. 

review

 

review

Physiology-aware signal handling is thin. Temporal “alignment” literally replicates each fMRI frame to match EEG blocks for fusion (rather than HRF deconvolution/lag modeling), undermining physiological interpretability and transferability. 

review

Bottom line: the paper shows no “real-world task” improvement (e.g., better neurofeedback outcomes, diagnostic/prognostic lift, or improved intervention planning). Thus the claimed “practicality” lacks a verifiable anchor.

2) Poor generalization (to datasets, subjects, tasks, and sites)

Very narrow data scope. Validation is limited to two public datasets; the visual set has only 93 multimodal samples. Sample size and task diversity are insufficient for broad generalization claims. 

review

 

review

No cross-dataset/site/task transfer. There is no OOD or cross-domain evaluation (across devices, centers, paradigms), and it is unclear whether strict subject-wise splits were used—only generic “5-fold RF / 10-fold KNN” is described. For multimodal neuroimaging, this is far from adequate. 

review

Authors themselves concede generalization/scale limits. The conclusion/limitations sections state that robustness and generalizability need strengthening, and that scalability to larger/higher-res data remains open. 

review

 

review

Heavy computation and complex hyperparameters. The method uses 100k candidates and 2000 iterations, with no clear scaling curves vs. node count/rank; this is unfriendly for moving to larger atlases or clinical-scale cohorts. 

review

Bottom line: current evidence only shows the method can slightly lift a classifier on two very similar setups; there is no quantification of robustness to cross-population/site/paradigm shifts—so “generalizability” is unsubstantiated.

Minor Issues (that further weaken the case)

Much of the neurophysiological interpretation is descriptive, lacking testable hypotheses and statistical designs. 

review

Most metrics are classification metrics; there are no graph-level EC quality measures (e.g., edge-PR, SHD/SID), making the claim “we learned better EC” difficult to support. 

review

Suggestions (for a major revision or a different venue)

Design real, actionable tasks: show operational gains in neurofeedback outcomes, clinical stratification/prognosis, or behavioral prediction—rather than only EC→classification proxy evaluation.

Systematic generalization testing: cross-dataset/site/task/device with strict subject-wise splits; report OOD/domain-shift performance and uncertainty.

Physiology-consistent temporal modeling: include HRF deconvolution/lag-aware baselines and ablations; avoid the “frame replication” misalignment. 



Scale & efficiency: provide wall-clock and memory curves vs. #ROIs/rank/sample size, and validate on higher-resolution atlases. 



Direct EC quality metrics: on (semi-)synthetic or interventional data, report graph-level measures and include misalignment/shuffling controls to prove the necessity of the alignment/fusion modules.

### Soundness
2

### Presentation
2

### Contribution
2
