# CREIMBO: Cross-Regional Ensemble Interactions in Multi-view Brain Observations

- Decision: Accept (Spotlight)
- Scores: 8, 6, 8

## Abstract
Modern recordings of neural activity provide diverse observations of neurons across brain areas, behavioral conditions, and subjects; presenting an exciting opportunity to reveal the fundamentals of brain-wide dynamics. Current analysis methods, however, often fail to fully harness the richness of such data, as they provide either uninterpretable representations (e.g., via deep networks) or oversimplify models (e.g., by assuming stationary dynamics or analyzing each session independently). Here, instead of regarding asynchronous neural recordings that lack alignment in neural identity or brain areas as a limitation, we leverage these diverse views into the brain to learn a unified model of neural dynamics. Specifically, we assume that brain activity is driven by multiple hidden global sub-circuits. These sub-circuits represent global basis interactions between neural ensembles—functional groups of neurons—such that the time-varying decomposition of these sub-circuits defines how the ensembles' interactions evolve over time non-stationarily and non-linearly.
We discover the neural ensembles underlying non-simultaneous observations, along with their non-stationary evolving interactions, with our new model, **CREIMBO** (**C**ross-**R**egional **E**nsemble **I**nteractions in **M**ulti-view **B**rain **O**bservations). CREIMBO identifies the hidden composition of per-session neural ensembles through novel graph-driven dictionary learning and models the ensemble dynamics on a low-dimensional manifold spanned by a sparse time-varying composition of the global sub-circuits. Thus, CREIMBO disentangles overlapping temporal neural processes while preserving interpretability due to the use of a shared underlying sub-circuit basis. Moreover, CREIMBO distinguishes session-specific computations from global (session-invariant) ones by identifying session covariates and variations in sub-circuit activations. We demonstrate CREIMBO's ability to recover true components in synthetic data, and uncover meaningful brain dynamics in human high-density electrode recordings, including cross-subject neural mechanisms as well as  inter- vs. intra-region dynamical motifs. Furthermore, using mouse whole-brain recordings, we show CREIMBO's ability to discover dynamical interactions that capture task and behavioral variables and meaningfully align with the biological importance of the brain areas they represent.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose CREIMBO that can extract interpretable neural subcircuits underlying multiregional neural signals by utilizing multi-session recordings that can have a variable number of recording units, trials, and durations. Through simulations, they show that their model successfully uncovers the ground truth dynamics when multi-session recordings are modeled together. In real data analysis, authors show that identified subcircuits can be sparse indicating specialized functionality, and reveal across-region interactions.

### Strengths
- Authors provide a new perspective over existing models of multi-regional neural models by allowing their model to leverage multi-session recordings that can help extract global and robust interregional interactions. While doing that, they keep the interpretability in-tact unlike deep-learning approaches. 
- Authors performed exhaustive simulation experiments to show the importance of their model formulation. 
- The paper is well-written and easy to follow. The proposed model architecture and training framework are intuitive and effective.

### Weaknesses
- Even though the paper is well written, the subfigures are very small and too crowded, which makes it hard to understand. Also, I think captions/labels for some appendix figures such as Fig. 11 should be improved. 
- It seems like the number of ensembles per region ($p_j$) is an important hyperparameter for CREIMBO. I wonder if the model would reveal some consistent subcircuits with small $p_j$, such that these subcircuits explain most of the variance and are consistent across sessions and subjects. Also, how would training times vary with large $p_j$ values such that $p \approx N$? Overall, I think the scalability of such a model is an important aspect since modern neural recordings can include hundreds of neurons from one region, in such case, max($p_j$) = 7, can limit the interpretability of the identified subcircuits. 
- In simulations (Fig. 2F), the authors show that the single-session model underperforms CREIMBO by a large performance gap. This can be caused by a small number of trials in each simulated session and short trials, but I could not find these details in the text (if it is in Fig. 11, I think it requires explanations in the caption). If this is indeed the case, I wonder how single-session models' performance would increase with longer sessions.  
- I think the biggest contribution of CREIMBO is its multisession modeling over other approaches like mDLAG. However, the modern recording session can have hundreds of trials of data that can be sufficient to train models to understand multiregional dynamics. Therefore, I think it would be nice to see how their model compares to mDLAG even if it operates on a single-session basis. Based on Fig. 22, for the real dataset considered in this study, using multiple sessions in modeling seems important, and a comparison to mDLAG would highlight the importance of multisession modeling. Even though mDLAG does not learn dynamic matrices for temporal evolution, its learned readout matrices and lag parameters would still indicate interregional interactions, and I wonder if interregional interactions learned by mDLAG would be as poor as 'Session # (SPARSE)' in Fig. 22. Also, did authors compare their model to SLDS variants in real data as done for simulations? Overall, I think this work would benefit from more baseline comparisons to existing approaches in real data.

### Questions
- In line with my previous comment in weaknesses item 2, the authors show subject 10 in Fig. 3 which has a small number of available neurons compared to some other subjects (for Screening task). Are the example subcircuits and latent states extracted for the Screening task? Do the identified subcircuits and A matrices look denser for a higher number of neurons?
- In the ablation study in Fig. 2 ‘All regions (sparse)’ A matrix, authors show that the CREIMBO cannot infer the true underlying connectivity matrix, which makes sense since the inductive bias on block-diagonal A matrix is removed. Do authors use the same A matrix initializations in this ablation study? If not, how would the results change if the same block-diagonal initialization is applied for ‘All regions (sparse)’ case? Can authors provide an intuition on why inferred latent factors deviate significantly from the true latents? Would the same hold in the K=1 case, in which, a similarity transform would exist between not block diagonal sparse, not block diagonal not sparse, and block-diagonal sparse A matrices? Also, are the authors showing trial and session averaged latent states in these figures? If so, how does single-session latents look like for CREIMBO?
- The block structure imposed on A matrix seems like no regions have shared latent states and interregional interactions are captured by temporal dynamics. Did the authors try having some latent factors shared across all regions? How would it change the performance?

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
4

### Summary
The recent developments in neural recording technologies allow recording from large populations of neurons from multiple brain regions. Latent space models are often used to analyze these datasets, but they are generally limited to the study of single or few populations of neurons recorded simultaneously. To overcome these limitations, the presented work introduces a new algorithm that can capture variability across recording sessions and across and within brain regions. The method assumes a shared latent representation across areas and structured priors given each session. The authors validated the method on simulated data and neural data, showing that the model can capture variability across and within brain regions.

### Strengths
The manuscript is clearly presented, referenced in the context of the relevant work and technically sound. The method was tested in simulations coming from the generative model, where the model was able to recover the parameter settings, and was able to better capture the variability compared to existing models. They also tested the model in neural recordings identifying functional connectivity between and within brain regions. Moreover, they tested the robustness to increasing levels of noise in the data.

### Weaknesses
Further work in needed to fully understand the impact of this work. The authors motivate their work from their ability to better understand behavioral tasks from asynchronous recordings of spiking neural activity. However, the authors only tested the model in a single dataset where they limit the analysis to functional connectivity. It would be relevant to assess behavioral variables, such as ability to decode the present image stimuli from the learnt representations. Since the emphasis of the model is brain functional connectivity, the method should be compared to alternative recording methods such as fMRI or the LFP present in the dataset. Moreover, the comparison to alternative models in the neural data is limited to one qualitative test. A comprehensive quantitative comparison, such as decoding or reconstruction performance, is needed to understand the capabilities of the proposed method. Along the same lines, it is not surprising that the model outperforms alternative methods when the simulated data is tailored to the given model. A more relevant comparison would be simulating neural data from a neural process with temporal, task and/or behavioral variability and fit the different models there. It would also be relevant to highlight the model strengths and weaknesses in simulated data. One of the motivations for this work is its application to spiking data, but the Gaussian assumption limits its application to this kind of observation model and must be handled in preprocessing. While the authors verbally list this and other limitations in the discussion, it would be informative to show the limitations and, more importantly, capabilities of the model as results.

### Questions
Can this model be applied to other organisms, like mice or rats? Extending the application to neuroscience animal models would greatly increase the impact of the presented work.
 
How difficult would it be to extend this model to use a Poisson observation model to better capture neural spiking activity?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Through the study, the authors have proposed a novel analytical approach (named as “CREIMBO”) for learning dynamics of latent representations from high-dimensional electrophysiology. The major advances of CREIMBO comes from that compression of high-dimension and extraction of dynamics were conducted simultaneously in CREIMBO, while keeping the interpretability. Through experiments with synthesized- and real data, the authors have properly demonstrated the validity of CREIMBO in the study. As CREIMBO contains conceptual novelty and its validated effectiveness, I believe this model can be one of useful candidate models for studying high-dimensional, partially overlapped, data, such as intracranial EEG or multi-array spike data. Thus, CREIMBO will be useful to neuroscientists.

### Strengths
The major strength of this study comes from its conceptual advances embedded in the proposed CRIEMBO. While there have been efforts to model the high-dimensional brain dynamics via deep learning, many of models failed to yield interpretable features, which is the most important aspect in the neuroscience field. Thus, the neuroscience field still tends to rely on relatively simpler yet interpretable models. In this regard, I believe CRIEMBO proposed in this study can be a good solution for this gap, clearly upholding the originality of this study. The authors thoroughly examined the validity of the proposed model using simulated- and experimental data, leading to the high quality of this work. Plus, the clarity of this paper is relatively high as the model was well described in the manuscript, whereas the reviewer believes there are some rooms requiring the attention of the authors. Altogether, the scientific significance of this paper is clear and can be interesting to the electrophysiology field.

### Weaknesses
While the overall strength of this study is obvious, there is a major weakness: Insufficient details in simulation study with synthetic data. It is unclear how the synthetic data was generated, e.g., what was the noise level, what kinds of parameters used for. Due to this uncertainty, it is nearly impossible to understand some of the intriguing findings in this study, especially with synthetic data. For example, there is very high inconsistency in performance across sessions in other methods, but not for CRIEMBO (Fig. 2M). While it can be interpreted as robustness of CRIEMBO, it is also possible the choice of benchmark models was not optimal for this type of synthetic data. Related to it, there was no comparison work with real data. Thus, the superiority of CRIEMBO needs further validation.

### Questions
1. Please provide details of how synthetic data was generated. 

2. In Fig. 2M and O, I observed the high inconsistency in reconstruction accuracy in other models, e.g.,  mp_rSLDS_Gauss (per tial). This is synthetic data and, I assume, each session contains similar level of noise. It is very unclear how other baseline models nearly completely fails but performed near perfectly in some sessions. My question is, does this inconsistency supposed to support the robustness of CRIEMBO?  
3. With real data, the authors demonstrated the validity of CRIEMBO. While I agree with the author’s claim, it does not necessarily lead to the strength of CRIEMBO. Are there any interesting differences or unique patterns that can be obtained from CRIEMBO vs. other baseline models? 
4. The authors checked the robustness of CRIEMBO over different noise levels, with real data. More common practice is evaluating the robustness of the models with synthetic data, one with grountruth. If there were any specific reasons why it was done with real data, please specify it.

### Soundness
4

### Presentation
3

### Contribution
3
