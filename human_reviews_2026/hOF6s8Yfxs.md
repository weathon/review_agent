# Hyperparameter search on the test set in the wild

- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Systems neuroscience has rapidly adopted machine-learning techniques, but has yet to develop a robust standardized methodology for assessing the performance of decoding models. Methodological issues can sometimes be subtle, arising as a consequence of experimental design. Here, in contrast, we investigate the consequences of post-hoc model selection: an issue which is neither subtle nor idiosyncratic. This occurs when a single test set is used to both select hyperparameters and evaluate performance, which favors models that overfit to ungeneralizable features of the test set. While the issues with this practice have been well documented within the ML literature, it has seen continued use in several domains, including systems neuroscience. To highlight this unfortunate practice, we performed a series of experiments using a selection of models from affected EEG decoding studies, finding that the overestimation of decoding accuracy in the affected studies was substantial: ranging from 0.74-1.24%. Moreover, we demonstrate that post-hoc model selection favors unstable model architectures, as the variability in their performance increases the likelihood that an instance of the model will coincidentally match the test set. Comparisons of model performance under post-hoc model selection may thus mislead researchers into developing increasingly complex and unstable models which fail to outperform simpler, more stable, ones.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackle the problem of post-hoc model selection in system neuroscience where a validation set is not available. It provides an analysis of the bias of such post-hoc selection through a number of experiments.

### Strengths
- The paper addresses an important difficulty regarding hyperparameter optimisation and model selection when no validation set is present.
- The experiments and hyperparameters used in the experiments are explained in details.
- The paper presents insights about the flows of the post-hoc models and recommend replacing it with more robust model-selection methods.

### Weaknesses
- The concrete contribution of the paper is missing. While it can be derived from reading the whole paper, it is essential to explicitly present the contributions (ideally in the introduction). This helps any future readers to understand the contributions.
- The position of the research relative to related work is not clear. There is no “related work” section in the paper, making the position of the paper in relation to the existing research and whether it addresses a significant research gap unclear.
- While the paper provides important analysis and conclusions about the flaws of current method, it does not propose nor present any concrete solutions: The authors wrote the following: we suggest that its use should be immediately discontinued in favor of more robust model-selection methods”, which is a broad statement and does not present a concrete solution.
- The code and data are not available for the review process. The authors mention in the paper that “All data and code used to generate these results will be made publicly available following acceptance”. However, it would have been better to include an anonymized link for the reviewers to assess the code.

### Questions
In the appendix of the paper, Table 4 presents the exact hyperparameters ranges used in the experiments. However, the rationale behind selecting these exact values is missing. Can you please elaborate more on the process and rationale of setting these exact values? Did you consider using automated methods (e.g., AutoML) or were the values set based on something else?

### Soundness
3

### Presentation
2

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
This paper raises the issue that post-hoc model selection can lead ML model performance to overfit test set features. Using an EEG dataset, the authors statistically demonstrate how models are affected by the problem(post-hoc model selection). They also point out that the problem can cause model overestimation, which in turn may lead to the development of unnecessarily complex and unstable model architectures.

### Strengths
- The paper highlights that post-hoc model selection can lead to the development of unstable model structures, which may negatively affect the design of suitable model architectures in the future.
- Through statistical tests, the authors identify biases caused by post-hoc model selection in deep learning models developed for EEG decoding tasks.

### Weaknesses
- The concept of post-hoc model selection is already a well-known issue. However, the related work section lacks a thorough review of prior studies that have addressed cherry-picking problems(post-hoc model selection).

- The paper identifies biases from post-hoc model selection using existing statistical tests, rather than developing a new methods. This work focuses more on demonstrating the severity of post-hoc model selection and suggesting the need for regulatory awareness. While the topic is critical, the paper may not fully align with the expectations of a main conference, which typically values methodological novelty. This work seems more suitable for a journal or review paper that focuses on ethical or methodological issues. If intended for a main conference, the paper would benefit from proposing a novel method to detect or mitigate post-hoc model selection effects, supported by mathematical justification and extensive experiments across multiple benchmarks.

- Although the paper title is “Hyperparameter Search on the Test Set in the Wild,” the experiments are conducted only on several models within an EEG decoding dataset. This issue is not unique to EEG decoding, yet the study does not examine diverse benchmarks or multiple model types. Therefore, the paper should either focus specifically on special domains like EEG decoding—where limited test sets make the problem particularly severe—and adjust the title accordingly, or broaden its scope to more general settings to strengthen its claims.

- The assumptions used in the appendix proofs do not appear sufficient to fully support the main argument. The results seem to rely on extremely small test sets and model performances that are already close to 1. In addition, a toy example demonstrating how these assumptions manifest in actual model outcomes would make the proof more convincing. At present, the assumptions seem simplistic, and if training does not converge properly, simply increasing the number of models may not guarantee convergence.

### Questions
- (W1) Are there no related works that have analyzed or addressed post-hoc model selection? Since cherry-picking is already a well-known issue, I assume that previous studies have examined this problem as well. In addition, this work appears somewhat similar to Kilgallen et al. (2025) [1], which introduced the repeated-stimulus confound. Could you elaborate on the differences and clarify the specific motivation behind their study?

- (W3) Why was only the EEG dataset used in this study? Was it chosen because such issues were observed in previous EEG decoding research? The problem of post-hoc model selection arises in performance evaluation across many ML applications, not just in EEG decoding. I would like clarification on whether the focus on EEG was due to practical limitations or deliberate scope.

- Was the t-test in Figure 3 performed on newly conducted post-hoc experiments, or based on existing model results? If the test set was used directly during experimentation, it would naturally lead to higher performance and thus significant results. What was the motivation behind conducting this analysis? Were there any specific observations or suspicions suggesting post-hoc model selection had occurred? In summary, it seems difficult to confirm post-hoc model selection solely from these results—what evidence led you to infer its presence?

- (W4) In the appendix, the 
$F(x) \in [0, 1]\ \forall x \in [0, 1]\ \text{ and }\ F(x) = 1 \Leftrightarrow x = 1.$ — is this an assumption, or a general property of the CDF? It does not seem to hold as a general property of cumulative distribution functions, so is it a condition required to satisfy Eq. (3)?

[1] Kilgallen, Jack A., Barak A. Pearlmutter, and Jeffrey Mark Siskind. "The Repeated-Stimulus Confound in Electroencephalography." arXiv preprint arXiv:2508.00531 (2025).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The current paper lays out the problem of post-hoc model selection in the context of EEG decoding, which is stated to be a well known problem in machine learning research but to be neglected in systems neuroscience. The authors propose three hypotheses for why this neglect occurs and attempt to test them using experiments with a subset of decoding models and tasks drawn from previous literature:
1. The magnitude of the bias caused by post-hoc model selection is perceived relatively modest in practice →  the paper shows that decoding accuracy substantially and consistently increases (i.e. positive selection bias) for all selected models and different decoding tasks for post-hoc compared to pre-hoc model selection, ranging from 0.74-1.24%. 
2. There is the assumption that despite overestimation of accuracy due to post-hoc model selection, relative improvements in performance are still meaningful across models or studies → the paper shows that relative differences in decoding accuracy between models are more unreliable for post-hoc model selection than pre-hoc model selection within the context of concept decoding. 
3. The belief that regardless of its technical incorrectness it is a relatively harmless practice in the broader context of the literature → using a direct manipulation of instability as implemented in a linear regression model, the paper shows that as instability increases, the selection bias increases, while true accuracy decreases. 

Based on this, the authors claim that the process of post-hoc model selection favours unstable models, which might steer the field into the wrong direction for further model development and yields problems with reproducibility. Thus, the authors argue for the discontuition of post-hoc model selection and for more robust model-selection methods within EEG decoding research.

### Strengths
- The current paper adds to the existing literature by explicitly targeting the effects of post-hoc model selection in the context of EEG decoding on model comparison (on top of the effects of repeated stimulus confound).  
- The paper systematically assesses the bias from post-hoc model selection for a diverse set of architectures and decoding tasks. 
- It illustrates the implications of post-hoc model selection on model comparison through the modelling of instability in additional experiments.

### Weaknesses
Main weakness:
- The novelty and contribution of the paper appear limited. The issue of hyperparameter tuning on the test set has been well established in the machine learning literature [1, 2], and the current work primarily highlights this known problem within a non–machine learning context. Furthermore, the paper does not provide many new insights or concrete methodological recommendations for mitigating this issue. Given this, the contribution remains largely conceptual and does in my opinion not warrant acceptance.

The following points are (comparatively) more minor and could be addressed through revision. They aim to improve the clarity, rigor, and presentation of the paper:
1. It would be helpful to clarify the scope of the paper more explicitly. As it stands, the study appears to focus on EEG decoding methods aimed at optimizing performance for applied contexts such as brain–computer interfaces. This focus differs from research that uses decoding accuracy as a measure of representational similarity or discriminative distance in more fundamental neuroscientific investigations (e.g. [3], which also assesses different decoding architectures, preprocessing steps and data partitioning schemes within this context). Clearly delineating this scope would help readers understand the intended application domain. This distinction is currently also not reflected in the paper’s title.
2. The manuscript would benefit from a clearer discussion of the limitations of the current studies and more concrete recommendations for future research. For instance, while the authors demonstrate how post-hoc model selection can introduce selection bias, it remains somewhat unclear what specific methodological approach they recommend to mitigate this issue. Based on the methods described, nested k fold cross-validation appears to be implied, though this approach is relatively well-established and may not constitute a novel contribution.
3. The paper would benefit from greater specificity and a clearer logical structure in its writing. This issue manifests in several ways:
    - Introduction: The introduction currently concludes with a paragraph summarizing previous neuroscientific work. It would be more effective to end with a concise statement of the research gap and the paper’s explicit contributions, which are currently missing.
    - Structure of sections: The manuscript does not clearly distinguish between sections. For instance, the discussion still contains elements of methods and results. I recommend creating a dedicated subsection on statistical analyses or hypothesis testing within the Methods section, and expanding the Results section (currently only a short paragraph) to include descriptions of figures and main findings. The Discussion should then focus solely on interpreting these results and outlining their broader implications.
    - Discussion organization: The Discussion is organized according to the proposed hypotheses explaining why post-hoc selection is often neglected within systems neuroscience as according to the author. However, the connection between these hypotheses, the corresponding experiments, and the subsection titles is not always clear. For example, the title of subsection 4.2 does not reflect the context of model comparison mentioned in the hypotheses or shown in Figure 3. In contrast, subsection 4.3 presents a clearer alignment between subtitle and experimental results, but its link to the overarching hypothesis remains weak (and the hypothesis itself may not be directly falsifiable).
       
    Overall, improving the structural coherence and explicitly aligning hypotheses, experiments, and interpretations would greatly strengthen the manuscript’s clarity and scientific narrative.
4. The overall tone of the paper can be read as quite dismissive towards previous neuroscientific work. For example figure titles like ‘the illusion of progress’ (Fig. 3) or ‘why bother with veracity when volatility is king?’ (Fig. 4), as well as expressions in the ethics statement referring to “faulty methods,” “false results,” and “time wasted,” could be interpreted as overly critical. The paper would benefit from adopting a more neutral and descriptive tone throughout, focusing on constructive critique rather than evaluative language.

[1] Cawley, G. C., & Talbot, N. L. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. The Journal of Machine Learning Research, 11, 2079-2107.

[2] Wainer, J., & Cawley, G. (2021). Nested cross-validation when selecting classifiers is overzealous for most practical applications. Expert Systems with Applications, 182, 115222.

[3] Guggenmos, M., Sterzer, P., & Cichy, R. M. (2018). Multivariate pattern analysis for MEG: A comparison of dissimilarity measures. Neuroimage, 173, 434-447.

### Questions
- The relationship between the repeated-stimulus confound and post-hoc selection bias remains unclear. How exactly is the repeated-stimulus confound defined in this context, and in what way does it interact with or contribute to post-hoc model selection bias?
- Could the authors provide more details about the dataset used, including the number of conditions, electrodes, and participants? Was this dataset selected because of its use in Kilgallen et al. (2025)? Furthermore, to what extent can the findings be generalized to other large-scale neural datasets, such as [4]?
- Could the authors clarify how EEG decoding was performed? Specifically, is the classifier trained separately for each time point, with performance averaged across time points and conditions, or is the neural data first averaged over time? Additionally, is a separate classifier trained for each participant, or is classification performed on data pooled across participants? The current methods section does not make these distinctions clear.
- Could the authors clarify how the hypothesis tests were conducted (in context of Table 3), given that the bias appears to be represented by a single value per model? A one-tailed t-test requires a distribution of values rather than a single estimate, so it is unclear what sample the test was performed on. Was the bias computed across cross-validation folds or participants? 
- Could the authors elaborate on the hypothesis presented in Section 4.3 regarding the continued use of post-hoc model selection? It is unclear how this hypothesis could be falsified, and consequently, the connection between the hypothesis and the results is unclear. 
- Could the authors clarify whether the range of overestimation in decoding accuracy (0.74–1.24%) mentioned in the abstract and Section 4.1 corresponds to the selection bias values reported in Table 3? In Table 3, the values appear to range from 0.14–1.44%; thus table and text seem to be inconsistent. 

[4] Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). A large and rich EEG dataset for modeling human visual object recognition. NeuroImage, 264, 119754.

### Soundness
1

### Presentation
2

### Contribution
1
