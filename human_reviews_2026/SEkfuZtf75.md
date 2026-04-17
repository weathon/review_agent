# EHR2Path: Scalable Modeling of Longitudinal Health Trajectories with LLMs

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Healthcare systems face significant challenges in managing and interpreting vast, heterogeneous patient data for personalized care. Existing approaches often focus on narrow use cases with a limited feature space, overlooking the complex, longitudinal interactions needed for a holistic understanding of patient health. In this work, we propose a novel approach to patient pathway modeling by transforming diverse electronic health record (EHR) data into a structured representation and designing a holistic pathway prediction model, EHR2Path, optimized to predict future health trajectories. Further, we introduce a novel summary mechanism that embeds long-term temporal context into topic-specific summary tokens, improving performance over text-only models, while being much more token-efficient. EHR2Path demonstrates strong performance in both next time-step prediction and longitudinal simulation, outperforming competitive baselines. It enables detailed simulations of patient trajectories, inherently targeting diverse evaluation tasks, such as forecasting vital signs, lab test results, or length-of-stay, opening a path towards predictive and personalized healthcare. We will release our code upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents EHR2Path, a transformer-based framework for modeling patient trajectories from EHR data using a structured text representation and a summarization mechanism that compresses long histories into compact tokens. The topic is timely and relevant, and the paper is well written and clear. However, several methodological and reporting limitations reduce the strength and interpretability of the results.

### Strengths
- Addresses an important and timely problem: scalable modeling of longitudinal EHR data using LLMs contributing to the growing shift from narrow outcome prediction toward holistic patient trajectory simulation.
- Conceptually interesting summarization mechanism that efficiently compresses long temporal contexts while maintaining continuity in modeling.
- Fairly novel unification of heterogeneous EHR modalities, structured, time series, and unstructured text into a textual representation, aligning with current trends in LLM-based clinical modeling.
- Generally well written and organized, with good motivation and consistent framing that make it accessible to both machine learning and clinical informatics audiences.

### Weaknesses
- Outputs only deterministic predictions without uncertainty estimates, limiting clinical usefulness. The absence of probabilities or confidence intervals prevents risk-aware decision support.
- Inference details are missing, decoding parameters such as temperature or top-k are never specified for either EHR2Path or ETHOS, hindering reproducibility and obscuring whether results depend on decoding strategy.
- Comparison with ETHOS is methodologically inconsistent: ETHOS uses stochastic Monte Carlo sampling, while EHR2Path is deterministic (?). Comparing them directly with deterministic metrics (F1, accuracy) is invalid without aligning inference methods.
- Inclusion of ETHOS results for both “restricted to data supported by ETHOS” and “full data” is contradictory, since ETHOS cannot process unstructured notes or ICU chart data. This raises concerns about what inputs were actually used.
- Test set details are incomplete: 5000 samples are mentioned, but class prevalences per task are omitted, making F1 and accuracy metrics difficult to interpret.
- Representation of diagnoses in the textual timelines is unclear, ICD categories are mentioned without specifying ICD-9 or ICD-10, undermining the claim of a unified representation. In addition, Appendix examples omit diagnosis tokens entirely, leaving this core modality undocumented.

### Questions
- What exact inference strategy was used, was temperature near zero or greedy decoding applied, or something else?
- How was ETHOS adapted, was Monte Carlo sampling not used or temperature modified, same question as above?
- For the full data ETHOS evaluation, what inputs were provided given unsupported modalities?
- What are the class prevalences for each downstream task?
- How are diagnosis categories represented textually, are ICD-9 codes converted to ICD-10 or standardized otherwise?
- Could you include an appendix example showing diagnosis representation in the timeline?
- Could you note in the appendix that the MIMIC-IV-ED extension was used and must be downloaded separately?
- Could you add a note in the Appendix that you used the MIMIC-IV-ED extension that has to be downloaded separately?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces EHR2Path, a novel model designed to address the challenge of modeling complex, longitudinal patient health trajectories using Large Language Models. The core problem it tackles is the failure of existing models to holistically interpret vast, heterogeneous Electronic Health Record data, which often comes from different clinical units like the Emergency Department (ED), hospital wards, and Intensive Care Unit. EHR2Path's solution involves transforming this diverse data—including structured fields, lab values, and unstructured text—into a unified, structured text representation. The key innovation is a novel "Masked Summarization Bottleneck," a mechanism that efficiently compresses long-term patient histories into topic-specific summary tokens, making the model highly token-efficient while capturing extended temporal context. The authors demonstrate that EHR2Path outperforms baselines on the MIMIC-IV dataset, showing strong performance in both next time-step prediction and the simulation of detailed future patient trajectories, such as forecasting vital signs and lab results.

### Strengths
A primary strength of EHR2Path is its comprehensive and scalable approach to data integration. Unlike previous models that often focus on a narrow set of features or specific outcomes , EHR2Path is designed to holistically process a patient's entire hospital stay, integrating all available structured and unstructured data from the ED, hospital, and ICU. The model's most significant technical strength is its novel Masked Summarization Bottleneck. This mechanism effectively solves the long-context problem in EHRs by compressing extensive patient histories into a compact set of summary tokens, allowing the model to be "much more token-efficient" than text-only approaches while leveraging long-term dependencies. This efficiency and holistic design translate to strong empirical performance, as the model not only outperforms trajectory simulation baselines but also proves to be a versatile foundation, outperforming specialized outcome prediction models when fine-tuned for those specific tasks

### Weaknesses
I didn't quite grasp the main contribution of this paper.

Predicting patient disease progression trajectories is an extensively studied problem. Researchers have commonly used models like RNNs, Transformers, and even Neural ODEs to forecast the temporal trajectories of high-dimensional data. To my understanding, these existing methods have already achieved a relatively high level of performance. Furthermore, techniques such as the Gumbel-Softmax trick already enable the joint prediction of continuous and categorical variables.

As I see it, the paper's motivation is not very clear. I don't fully understand the rationale for using an LLM for this task. It appears that the objectives of this paper can be achieved by existing methods, which can also be extended to handle unstructured text by incorporating embedding models. A particular drawback is the model's suboptimal performance; it does not seem to have been trained to a level sufficient for practical application.

### Questions
I noted in the Appendix that the training was performed based on a 0.5B quantized model. I am curious about the decision to perform SFT on a 4-bit model. Does applying SFT directly to a quantized version of such an extremely small model introduce any specific problems?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes EHR2Path, a model that represents electronic health record (EHR) data as structured text and predicts patient health trajectories over time. It integrates heterogeneous data sources such as vital signs, lab tests, and clinical notes within a unified language-based representation. A Masked Summarization Bottleneck compresses long-term patient history into compact embeddings, enabling efficient modeling of extended timelines. The model can simulate future states of patients and supports both trajectory forecasting and outcome prediction. EHR2Path is evaluated on the MIMIC-IV dataset, where it outperforms baseline methods like ETHOS in next-step prediction and long-term simulation, demonstrating improvements in forecasting tasks and generalization across different hospital settings.

### Strengths
Here are the strengths of the paper in my opinion:

1. The paper tackles an important problem: modeling longitudinal patient trajectories rather than isolated outcomes. This is very important step towards clinical AI as it resembles the standard approach clinicians take too.
2. The proposed Masked Summarization Bottleneck is a neat adaptation for handling long patient histories.
3. Evaluation is conducted on a publicly availble dataset (MIMIC-IV) with multiple baselines and clearly defined metrics.

### Weaknesses
While the work is technically sound and relevant, but the contributions and empirical analysis have limitations. The major weaknesses of the paper are as follows:

1. The novelty is incremental: the paper mainly combines existing concepts (EHR tokenization, transformer summarization, next-step prediction) without introducing a fundamentally new modeling idea.
2. The experiments are limited to one dataset, and there’s little analysis on generalization, interpretability, or clinical significance of the forecasts.
3. The discussion of results is brief; there is minimal exploration of failure cases, ablations, or design justification beyond empirical comparison.
4. Presentation is adequate but somewhat dense, and the framing could be clearer in how this work advances prior EHR modeling approaches.

### Questions
To me although the methodology looks nice and the paper is tackling the modeling from an important perspective, there are many key downstream experimental points that are missing to make the work strong: 

1. You state that the “Masked Summarization Bottleneck” compresses longitudinal context and forces outputs to attend only to summary tokens, with information-bottleneck framing. How does this differ in practice from known summarization or prefix-token schemes that restrict attention using masks or adapter tokens? Can you provide an ablation that replaces the custom mask with a simpler alternative?
2. The method appends m summary tokens per section and uses a mask to form a bottleneck. What is the performance vs. m trade-off and the compute cost vs. m trade-off? A plot of accuracy and wall-clock vs. m would help.
3. All experiments are on MIMIC-IV. A major weakness is that the authors do not provide stronger evidence for generalization. Can you add even a small held-out institution or a temporal split to demonstrate robustness? If that is not possible, can you simulate distribution shift in MIMIC-IV by patient subgroups or time periods and report stability?
4. Another suggestion is to use other publicly available EHR datasets to see how well the model generalizes there. To provide some examples, please check out NeurIPS D&B of the past 2 years. Specifically: https://som-shahlab.github.io/ehrshot-website/ can be a good candidate for the extra study.
5. Tables 1-3 present point estimates only. Please add confidence intervals or standard errors across seeds for next-step metrics and across simulation runs for longitudinal tasks.
6. How is LOS computed at training time without leaking future discharge timing into features that predict discharge?
7. How sensitive are simulation outcomes to errors compounding over long horizons? A breakdown of performance by simulation length would help.
8. You avoid heavy preprocessing, retain noise, and use natural language feature names and values. Could you show an explicit example that compares your textualization to code-based tokenization for the same patient slice, and then quantify the effect on accuracy and token counts?
9. Please include representative success and failure simulations for ICU vitals, inputs, and discharge timing, and discuss common error modes.
10. For diagnosis prediction tasks, show confusion matrices or top confusions for ICD categories.
11. Given that you are dealing with patient trajectories and you retain missing and incorrectly populated fields to embrace real-world noise, can you please quantify robustness: create stress tests that increase missingness or introduce synthetic noise and report degradation curves for key tasks.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces EHR2Path, a scalable model using LLMs to predict longitudinal patient health trajectories. The method transforms diverse and heterogeneous EHR data, such as vitals, labs, and notes, into a unified and structured text representation. To efficiently handle long-term patient histories, the authors propose the ``Masked Summarization Bottleneck'', which compresses extensive temporal data into compact summary tokens. Based on the evaluation over MIMIC-IV dataset, EHR2Path demonstrates good performance that outperforms baselines in both next time-step prediction and the simulation of full patient trajectories.

### Strengths
1. This paper studies the problem of trajectory prediction with LLMs, which is an interesting topic in healthcare.
2. The paper is generally written with good clarity, and thus it is easy to follow.
3. The experimental results demonstrate the effectiveness of proposed method compared to the baselines.

### Weaknesses
1. One of the major concerns is the limited technical contribution. The proposed Masked Summarization Bottleneck and Length-of-Stay Indicator appear to be relatively straightforward techniques. These components function more as engineering solutions rather than novel methodological contributions, which raises questions about the paper's technical novelty.

2. Another major concern is the insufficient model evaluation. Specifically, 

- a) The experimental comparisons for the Next-Timestep Prediction task are insufficient. Given that there are many existing models are capable of performing this task, a more comprehensive comparison is necessary. If the challenge lies in the mixed prediction types, the authors should consider conducting type-specific evaluations. For instance, isolating event prediction alone would be more feasible and would provide fine-grained insights into the model's performance across different prediction modalities.

- b) The evaluation methodology for trajectory prediction raises concerns as well. For time-sensitive outcomes such as mortality prediction, simple binary classification fails to capture prediction accuracy adequately, particularly since death occurs only once per patient, while the timing of predictions is critical in such scenarios but not considered. The experimental details in this section are unclear, and the evaluation approach requires substantial refinement.

3. One minor concern is the paper does not specify how many runs were conducted to obtain the reported experimental results. This is particularly critical for methods with LLM inference, as it is essential to assess the consistency or even hallucination concern of the model's predictions across multiple runs.

### Questions
1. Please refer to the above weakness for details.

### Soundness
3

### Presentation
3

### Contribution
2
