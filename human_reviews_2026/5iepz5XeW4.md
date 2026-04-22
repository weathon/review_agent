# BACE: Behavior-Adaptive Connectivity Estimation for Interpretable Graphs of Neural Dynamics

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Understanding how distributed brain regions coordinate to produce behavior requires
models that are both predictive and interpretable. We introduce Behavior-
Adaptive Connectivity Estimation (BACE), an end-to-end framework that learns
context-specific, directed inter-regional connectivity directly from multi-region intracranial
local field potentials (LFP). BACE aggregates many micro-contacts
within each anatomical region via per-region temporal encoders, applies a learnable
adjacency specific to each behavioral context, and is trained on a forecasting
objective. On synthetic multivariate time series with known graphs, BACE accurately
recovers ground-truth directed interactions while achieving forecasting
performance comparable to state-of-the-art baselines. Applied to human subcortical
LFP recorded simultaneously from eight regions during a cued reaching task,
BACE yields an explicit 8×8 connectivity matrix for each within-trial behavioral
context. The resulting behavioral context-specific graphs reveal behavior-aligned reconfiguration
of inter-regional influence and provide compact, interpretable adjacency
matrices for comparing network organization across behavioral contexts. By
linking predictive success to explicit connectivity estimates, BACE offers a practical
tool for generating data-driven hypotheses about the dynamic coordination
of subcortical regions during behavior.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BACE, a graph-learning framework that estimates directed, behaviour-specific brain connectivity from multi-region time series. BACE uses per-region GRU encoders and a learned adjacency matrix specific to each behavioural phase to forecast future neural signals while revealing interpretable regional interactions. BACE accurately recovers synthetic graphs, and real intracranial LFP recordings, with better performance when compared to a series of different baselines.

### Strengths
I don't have experience with the specific applied area, but assuming that the Introduction and Related Work sections are a good depiction of the field, this is indeed a very interesting work, and what a good domain-informed ML work should look like: there are domain-specific challenges and a new methodology is developed to tackle those challenges by creatively combining existing ideas. The experiments seem to show reasonable support that this method works, showing good performance across a synthetic and real-world dataset, across different baselines at a reduced computational cost. I should also comment that the Reproducibility analysis on section 4.3 is a really good one (ie, to check consistency of the model across different subsets of training data), as in my experience many deep learning models suffer from an annoying big lack of consistency when different seeds or training folds are used, and so I'd like to commend the authors for this specific experiment.

Despite these strengths, I have important concerns about the fairness of the comparisons used in this paper, and so for now I'm leaning towards rejection; hopefully this could change during the rebuttal period)

### Weaknesses
# Main Weakness
I have a main/key concern with this work, related to the fact that it seems that BACE's ability to learn "phase-specific" graphs depends on manually provided phase information. The model does not seem to infer the behaviour intervals; rather, it requires the trial data to be split by known behavioural phases (ie, phases seem to have been defined by marked events, I'm guessing done by doctors). This manual definition of contexts gives BACE additional information that baseline models did not have access to, raising fairness issues in the comparisons. If BACE knows the current phase (and uses a dedicated adjacency for that phase), it effectively has an "oracle" guiding it on when network changes can occur, whereas baselines had to model the entire sequence without such information. This could partly explain BACE’s superior performance. The paper should clarify whether phase indicators were indeed provided as inputs or used to select adjacency matrices during training. If so, the authors should justify whether the comparisons are fair, for instance could the baselines be extended with the same phase information? As it stands, it seems that none of the baseline models were given information about phase intervals or allowed to have different dynamics per phase. This means BACE had an advantage by effectively training a model with different parameters in each phase while others had to use one set of parameters across the whole trial.

To fairly assess the benefit of BACE, it would help to compare against at least one baseline that also leverages phase information. For instance, one could train separate baselines for each phase, or provide a one-hot phase indicator to a baseline as input. Without such comparisons, it’s unclear how much of BACE’s success comes from its novel architecture or just from privileged information. This issue needs clarification and possibly further experiments.


# Other Weaknesses

2. The paper makes a lot of bold claims in Introduction that I don't think are actually defended in the paper, and thus it would be good to get a clarification during the rebuttal period, specifically:
   1. proving/defending why contacts are high-dimensional and non-Euclidean (given we are talking I believe of only 8 such contacts, this claim seems difficult to defend?)
   2. Quantifying uncertainty
   3. Decoupling edge pattern from transmission strength
3. I think the paper can be improved with regards to clarity. 
   1. Firstly, I was only finally able to understand what these "behaviours" are in section 4.2. Up until then, the paper keeps mentioning behaviours (sometimes using other adjacent/related words like phase and context as if they were the same, adding to the confusion) as the key important part of this paper without any basic explanation to at least have an idea of what we are talking about.    
   2. Introduction mentions a "cued reaching task" which, given the broad computational background of ICLR, would require a brief explanation of what that is for better readability.
   3. Figure 1's caption mentions an adjacency $A_\phi$ that doesn't exist in the figure itself, while also saying "behavioral segment" in the caption while the figure instead uses the concept "Phase-specific".
   4. In the Methods section, regions are sometimes referred with letter "r" and other times with letter "i".
   5. What are the "contacts" mentioned on the the 3rd paragraph of Introduction which seem to be central to the motivation of this work?
   6. Figure 3, pane A, has some wiggly red lines under "ms"
4. I know this is mentioned in the Discussion section, but I have to highlight that the fact that only one participant was used means that we have no evidence that BACE could work for other individuals. Given this, one would expect more extensive comparisons to be more sure about the strengths of this work (as I'm detailing in this section)
5. A bit related to my main concern to this paper, I'm also concerned about how the baselines were trained. In the text it is said that models were optimised "under their standard hyperparameters", but that is not fair as it is well known how hyperparameters can be widely different for different datasets. Furthermore, given the strong claims on computational comparison, this is even more important; for example some of the models seem to have several hidden layers, and it is not clear whether a simpler structure would achieve similar results and thus making the claim of computational efficiency weaker.
6. A bit related to the previous point, it's unclear how much tuning of hyperparameters (e.g., GRU lengths, learning rate, etc.) was needed for this dataset. For example, the sliding windows and forecast steps were presumably chosen based on the task's timing, and the reliance on specific timings suggests the model might need careful setup for each new dataset. This isn't necessarily a flaw, but it means the method might require domain knowledge to deploy, and the paper doesn't discuss sensitivity to these choices, which in my opinion is an oversight given the reliance on a single dataset. Some justification of these design decisions and impact on interpretability would improve clarity and confidence that the approach can be adapted beyond the exact scenario presented.
7. No ablation study was done on the different components of the model to understand impact in performance. This ablation could be done in many different ways, but one that I think could be useful would be to use a single adjacency (ie, no phase differentiation), which would directly test how much benefit the behaviour conditioning provides in terms of forecasting accuracy.

### Questions
I believe I don't have any further questions beyond what I mention in the Weaknesses section. I'll reconsider only my score if the authors successfully tackle the main weakness I've identified, but a productive rebuttal would touch in all the points I've highlighted in Weaknesses except points (3) and (4).

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
4

### Summary
The authors propose a model that infers effective connectivity between multiple brain regions from local field potentials (LFP) by modeling the signal in each region as a combination of dynamics from other brain regions, and an internal regional generator. The authors describe their method from a graph neural network perspective, and use this framework to propose a model that achieves good reconstruction and simulation results.

### Strengths
The methods' results on the simulation set are encouraging and look quite good. Moreover, I believe the authors are working on an interesting problem, and that although this paper has a lot of potential, it needs more work before I can recommend it for publication. I commend the authors on their work, and encourage them to continue improving it.

### Weaknesses
Major weaknesses:
1) Baselines and missing related work. Given the interest in this field, I believe the authors are missing a large amount of relevant previous work and baselines. First, the authors do not compare any baselines on their simulation experiments, so it is unclear whether their proposed method uniquely can perform well on the simulation. Second, although the authors refer to their connectivity as 'effective', which I agree with, they do not compare to any commonly used effective connectivity measures such as Granger causality, dynamic causal modeling, transfer entropy, etc. Third, the authors do not compare or discuss any of the methods from a rich field of multi-region communication inference. Just to name a few papers that the authors should compare against or at the very least discuss are CURBD [1], DLAG [2], and MR-SDS [3], among others. All of these works are very closely related to the authors' work, but none are cited nor compared against. 
2) Predictive behavior can be due to the capacity of the single-region encoder/decoder. Given the use of recurrent neural networks and the specific formulation the authors' use, it is unclear to me how the authors ensure that most of the variance in a region is not just captured by a region's specific neural network. More specifically: the authors describe that their GRU creates embeddings for each region $\mathbf{H_r} \in \mathbb{R}^{T \times d}$, where T and d are the number of input temporal steps, and the number of latent dimensions for region r. This embedding is then multiplied by $W_{\text{self}}$ (which has no restrictions it seems as opposed to $A$) plus a region-interaction term involving A to obtain $\mathbf{Z_{i,t}}$. Subsequently, $\mathbf{Z}$ is used to forecast new time steps. How do the authors avoid that the GRU just learns to forecast the next timesteps without much influence from the interactions? If values in A are restricted to be between -1 and 1, then $W_{\text{self}}$ can just learn to have large values and become the dominant force in forecasting. Although the authors try to reduce this issue with what they call adjacency mixing, they use a (presumably) learnable factor $\alpha$, which could reasonably lead to a small impact of A on the forecasted time steps $y_{t-1}$. Moreover, in the LFP data the authors use, they do not use their model to generalize to unseen contexts/tasks, so the GRU can reasonably learn the temporal shape of the time series for each task. Moreso, the authors split based on each trial, so it is not unreasonable that the GRU can explicitly learn the temporal shape for each trial in the training set and that this shape is quite similar to the temporal shapes in the test set.
3) The method requires the model to know the context. This reduces the generalizability of the method.

Minor weaknesses:
- It is unclear to me what makes the method phase-specific. In fact the authors refer to the graph learner as phase-specific both in the text and Figure 1, but in Section 3 the paragraph heading is 'Context-specific graph learner'. I do not see how the method computes the phase of the signal (i.e. through a Fourier transform), or does phase here refer to context as in 'a different phase of experiment'? The authors should clear up their use of language, especially because there are many phase-based connectivity measures [4].
- Generally it is unclear to me exactly what the context-specific graph learner does or how it produces each A, are the A matrices just learnable parameters?
- The authors do not address the issue of scaling, specifically because they use a separate GRU for each region, the number of parameters would quickly grow with a larger number of brain areas under consideration.
- The authors provide limited neuroscientific grounding for their claim that the directed region-level graph should be different for different task contexts. The authors claim in the discussion that '... the model produced one connectivity matrix per behavioral segment.' but this is a design choice. There is some neuroscientific discussion in the sentences after this statement, but it lacks depth and a firm grounding of the results.
- The authors place a lot of focus on computation complexity, even though each of the methods they compare against is quite low in computational complexity, instead the authors should focus on better baselines and a broader discussion of the implications of their results.
- L116-117 The authors should introduce AMAG before using it in the text, a previous reference to AMAG on L59 is too far back for comfortable reading
- L308-309 The authors mention the Appendix, but not the exact section.

[1] Perich, M. G., Arlt, C., Soares, S., Young, M. E., Mosher, C. P., Minxha, J., ... & Rajan, K. (2020). Inferring brain-wide interactions using data-constrained recurrent neural network models. BioRxiv, 2020-12. \
[2] Gokcen, E., Jasper, A. I., Semedo, J. D., Zandvakili, A., Kohn, A., Machens, C. K., & Yu, B. M. (2022). Disentangling the flow of signals between populations of neurons. Nature Computational Science, 2(8), 512-525. \
[3] Karniol-Tambour, O., Zoltowski, D. M., Diamanti, E. M., Pinto, L., Brody, C. D., Tank, D. W., & Pillow, J. W. (2024). Modeling state-dependent communication between brain regions with switching nonlinear dynamical systems. In The Twelfth International Conference on Learning Representations. \
[4] https://direct.mit.edu/books/monograph/4013/chapter/167048/Phase-Based-Connectivity \

### Questions
-

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents BACE (Behavior-Adaptive Connectivity Estimation), a method that learns phase-specific directed graphs representing inter-regional neural connectivity from time-series data. The idea of behavior-adaptive, interpretable connectivity estimation is novel and well-motivated, and the paper is clearly written. However, the empirical validation is too limited: all real-data experiments use a single-subject, non-public intracranial LFP dataset, preventing reproducibility and generalization. The method also depends on manually defined behavioral phases and models effective rather than causal connectivity.

### Strengths
The paper introduces a novel and conceptually well-motivated framework, BACE, that jointly performs neural time-series forecasting and interpretable, behavior-adaptive connectivity estimation. Its main strength lies in combining predictive modeling with explicit, directed graph structures that reveal how neural interactions change across behavioral phases. The approach offers clear interpretability, allowing neuroscientific insights into dynamic brain networks, and demonstrates competitive forecasting accuracy compared to baselines such as DCRNN, GWNet, and AMAG.

### Weaknesses
The main weaknesses of the paper lie in its limited empirical validation and lack of reproducibility. All real-data experiments are conducted on a single-subject, private intracranial LFP dataset, which cannot be shared due to clinical restrictions, making it impossible for others to reproduce or verify the results. The method also depends on manually defined behavioral phase labels, restricting its applicability to structured experimental settings rather than continuous or naturalistic data. Moreover, the inferred graphs capture effective (predictive) rather than causal connectivity, so the biological interpretability of the results remains limited. Finally, the learned connectivity is static within each phase and has not been tested across multiple subjects, modalities, or tasks, which weakens the generalizability and robustness of the proposed framework.

### Questions
1. Limited and Non-Generalizable Experimental Validation

Although the authors release the code, the real dataset used in experiments is private and single-subject, consisting of intracranial LFP recordings from one clinical patient. This makes it impossible to reproduce the reported results or assess how well the model generalizes to other participants, tasks, or data modalities. The empirical evaluation is therefore too narrow to support strong claims about the model’s general effectiveness.

2. Dependence on Predefined Behavioral Phase Labels

The framework requires explicit behavioral phase annotations (“Wait”, “React”, “Reach”, “Return”) as inputs to learn phase-specific connectivity. This reliance on manually segmented data limits its applicability to real-world, continuous, or unlabeled neural recordings, where such phase labels are unavailable.

3. Limited Causal and Biological Interpretability

The model estimates effective (predictive) connectivity, not true causal interactions. The learned directed edges capture statistical dependencies that may not reflect actual neural influence. Without causal modeling or multimodal validation (e.g., cortical or anatomical data), the biological interpretability and neuroscientific conclusions remain tentative.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes behavior-adaptive connectivity estimation (BACE) to approximate the effective connectivity of brain signals recorded via electrodes using a neural network. On a synthetic dataset, the model successfully reconstructs matrices similar to the true effective connectivity. Application to real-world human data also suggests that the estimated effective connectivity captures clinically meaningful information.

### Strengths
- The method is validated on real-world human data, which is extremely difficult to acquire.
- The work addresses an important topic in the field of neuroimaging: the discovery of effective connectivity with interpretability.

### Weaknesses
- The choice of the modules within the architecture is rather simple. The paper would benefit from further explanations and comparative experiments on the architectural choice.
- More recent baseline methods can be included in the comparative experiments.
- The study remains proof-of-concept, with a single-subject data.

### Questions
### General
- Methodological rigor and contribution seem to be limited compared to the value of the real-world human data. The work may be of more interest to the readers from other conferences or journals.

### Major
- Please add explanations and/or experiments on current and other architectural choices.
- Please consider adding more recent baseline methods in the main comparative experiment.

### Soundness
2

### Presentation
2

### Contribution
1
