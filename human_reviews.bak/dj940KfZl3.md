# PIE: Simulating Disease Progression via Progressive Image Editing

- Decision: Reject
- Scores: 5, 5, 3, 5

## Abstract
Disease progression trajectories can greatly affect the quality and efficacy of clinical diagnosis, prognosis, and treatment. However, one major challenge is the lack of longitudinal medical imaging monitoring of individual patients over time. To address this issue, we propose Progressive Image Editing (PIE) method that enables controlled manipulation of disease-related image features, facilitating precise and realistic disease progression simulation in imaging space. Specifically, we leverage recent advancements in text-to-image generative models to simulate disease progression accurately and personalize it for each patient. We also theoretically analyze the iterative refining process in our framework as a gradient descent with an exponentially decayed learning rate. To validate our framework, we conduct experiments in three medical imaging domains. Our results demonstrate the superiority of PIE over existing methods such as Stable Diffusion Video and Style-Based Manifold Extrapolation based on CLIP score (Realism) and Disease Classification Confidence (Alignment). Our user study collected feedback from 35 veteran physicians to assess the generated progressions. Remarkably, 76.2% of the feedback agrees with the fidelity of the generated progressions. PIE can allow healthcare providers to model disease imaging trajectories over time, predict future treatment responses, fill in missing imaging data in clinical records, and improve medical education. Anonymous code for replicating our results can be found at https://anonymous.4open.science/r/PIE-3332.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a generative model based on DDIM called PIE (progressive image editing) to simulating disease progression using a text-conditioned stable diffusion model. The theoretical proofs show how the changes to the image being edited is bounded by a constant and converges. The approach is benchmarked on datasets involving lung x-rays, diabetic retinopathy, and skin lesions and shows promising performance against 2 other baselines, namely stable diffusion video and style based manifold extrapolation. The models are evaluated based on CLIP scores and classification confidence scores on the generated images. In addition to these experiments, a real-world edema progression and a user study is also shown to provide evidence that the disease progression makes sense.

### Strengths
1. The work is focussed on an important problem if simulating missing longitudinal data in medical imaging. Scarcity of such data is a genuine issue and to my knowledge, this work is among the few which attempts to do so without access to any temporal image data.
2. The proposed approach performs well on confidence score metrics and shows gradual improvements with number increasing of steps. The use of the real-world dataset which sequential images and the user study provide good evidence of the efficacy of the approach. Similarly, showing performance on 3 datasets from different problem types is also a big plus.
3. The availability of the code and the details in the supplementary are appreciated and a strong sign towards transparency and reproducibility. The experiments on ablations and sensitivity to hyperparamaters is also helpful for trying this approach and for future extensions of this work.

### Weaknesses
1. Even though the paper shows the editing process to be bounded and converging, I find it hard to understand why generating disease progression in images without any intermediate temporal information will lead to the correct intermediate pathologies in the image. Neither the text, nor the image have any information about what temporally intermediate stages of the disease can look like. Lacking this info, it's not clear how the progression is constrained to be realistic or biologically plausible. The real-world experiments on the edema dataset as well as the user study are most certainly helpful, but not completely convincing.
2. The paper proposes an interesting solution to a medical imaging problem, but is technically incremental in terms of the proposed method since it's a direct application of DDIM for conditional generation. 
3. The performance improvement with PIE is less significant on CLIP metrics. Additionally, all the similarity numbers on all datasets and baselines are usually high (>0.9 for a metric having a range of [-1,1]) which perhaps points to the fact that differences in this metric might not be hugely indicative of better fidelity, specially for medical images.

### Questions
1. Why are the confidence scores for other baselines so bad for the diabetic retinopathy dataset?
2. The recall, and in turn F1 scores for the simulated images is higher than the real ones in the case study. If this is indeed due to the simulated images accentuating the disease features, does that pose as a risk to this technique, specially in situations where it hallucinates or exaggerates pathologies?
3. Not a question, but the presence of RoI masks seems very important as without them, the model hallucinates significantly (as shown in the supplementary). It might be worth including this in the limitations sections or making this explicit for the readers if not already done so.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a method to generate realistic medical images corresponding to progression of diseases. The input is the image to be progressed and a text prompt describing the progression in the form of a clinical report. The method using Denoising Diffusion Implicit Models (DDIM) and text encoding using CLIP. It is evaluated on a dataset of chest X-rays (CheXpert), and skin cancer (ISIC 2018/HAM10000), and Diabetic Retinopathy. The approach is compared to Stable Diffusion Video and Style-Based Manifold Extrapolation. The results are evaluated qualitatively using visual examples and quantitatively by comparing CLIP embeddings of real and generated images and using the confidence score of a disease classifier. In addition 35 physicians and radiologists were surveyed using a questionaire on the realism of the generated images.

### Strengths
- The ability to simulate disease progression in medical images could have many relevant uses.

- Evaluated on a number of different medical imaging modalities.

- The results seem to be of good quality and the method novel.

- Trained model checkpoints will be made available on publication according to the supplement.

### Weaknesses
- A fundamental problem with the work is the focus and claims related to modelling of disease trajectories or progression. It is not entirely clear what the authors mean when they use these terms, and since this is a critical part of the work, this should really be defined.  Disease trajectory, I would understand to refer to the course of a disease over time. This could be in an individual or maybe as an average in a population. This would imply some predictive capability, and we are also told this in the abstract (see below). Yet there is as far as I can see no evidence that the proposed method can predict the future of individual patients or average patients. Instead it seems to me that what the approach is doing is instead to create images corresponding to different disease severities, which is certainly interesting, but a very different and generally easier problem. Loosely described, this could perhaps be called disease progression simulation, which is also a term used by the manuscript in places.

- "PIE can allow healthcare providers to model disease imaging trajectories over time, predict future treatment responses" - where is the evidence for this?

- "Specifically, we leverage recent advancements in text-to-image generative models to simulate disease progression accurately and personalize it for each patient." - how is it personalized?


- "The learning rate in this iterative process is decaying exponentially with each iteration forward, which means that the algorithm is effectively exploring the solution space while maintaining a balance between convergence speed and stability.", I don't think this is supported by evidence/references.

- "The physicians agree that simulated disease progressions generated by PIE closely matched physicians’ expectations 76.2% of the time, indicating high accuracy and quality." - is this a relevant measure to compare to? Are physicians able to predict actual progression?

- The question the physicians were asked appears to be "Does the below disease progression fit your expectation?" It is unclear if this is supposed to match a development in disease severity or what the specific development in this particular case would be expected to be.

- "However, all these methods have to use full sequential images and fail to address personalized healthcare in the imaging space. The lack of such time-series data, in reality, poses a significant challenge for disease progression simulation". I am uncertain about what is meant by "failing to address personalized healthcare in the imaging space". Could more precise wording be used? Also I feel like the authors are overly focused on the requirement of sequential data as a limitation. Longitudinal data exists for a reason and it may be much more difficult if not impossible to derive individualized progression models from cross-sectional data alone. I would suggest the authors think about the wording here and present it not as a limitation of previous methods but rather as a situation where the proposed approach could be used where previous models may not.

- Explain abbreviation DDIM

- "Due to the properties of DDIM, the step size would gradually decrease
with a constant factor.", what step size? No mention of step size before this point.

- Proposition 2 and 3, would benefit from some motivation, and explanation in text. There are variables and functions used without definition.

- "In addition, Proposition 2 and 3 show as n grows bigger, the changes between steps would grow smaller. Eventually, the difference between steps will get arbitrarily small. Hence, the convergence of P IE is guaranteed and modifications to any inputs are bounded by a constant." - I don't see how this follows. Could you help the reader a bit?

- What are the numbers presented in Table 1?

- "To further assess the quality of our generated images, we surveyed 35 physicians and radiologists with 14.4 years of experience on average to answer a questionnaire on chest X-rays." - why are the questions asked not

- "Furthermore, a user study conducted with veteran physicians confirms that the simulated disease progressions generated by PIE meet real-world standards.", what real world standards?

### Questions
- See the fundamental weakness mentioned in the above. Is it the authors intention to claim that the method can be used for prediction of future time points?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The manuscript presents a framework for progressively editing a medical image to simulate disease progression. The method is based on a diffusion denoising model that generates medical images based on text (medical report). The method is showcased in three medical applications to simulate enlarged disease sites and more server disease effects.

### Strengths
1. The task of editing medical images to inject or remove disease effects is of interest and is related to a long-standing problem of counter-factual generation.

2. The model generates visually authentic disease effects that are better than two comparison baselines.

### Weaknesses
1. Methodologically, the text condition seems to be a major part of the proposal. In fact, I believe it is the only mechanism that allows to model to "know" what is a "disease effect". However, it is discussed minimally in the method section, and is never discussed experimentally.

2. A core in these generative models in medical imaging is to show that the model does not hallucinate; the generated subject-specific disease should reflect realistic progression. The paper lacks quantitative evaluation on this aspect. The only experiment (Fig. 7) shows that the simulated disease effect deviates largely from the real case.

3. I'm having a hard time imagining what would be an ideal use scenario. The manuscript argues that the method can be used for "model disease imaging trajectories over time, predict future treatment responses, fill in missing imaging data in clinical records, and improve medical education". I'm not convinced it can do all of those things except for the last goal of "medical education", where the method can generate synthetic disease effects without showing an actual patient's data (see my questions below)

### Questions
1. It seems that the model cannot generate a deterministic progression trajectory as it mentions "We obtain at least 50 disease imaging trajectories for each patient". Why is this desired? How can such randomness contribute to "model disease imaging trajectories over time, predict future treatment responses, fill in missing imaging data in clinical records"?

2. I'm not sure why the model should generate "disease effects" from a healthy image (e.g. Fig. 5 3rd row). Isn't this contradictory to "predict future treatment responses" or "model disease imaging trajectories"? Healthy subjects should simply have healthy trajectories.

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In order to deal with the problem of the insufficient provision of necessary disease monitoring medical imagery and associated expert interpretation reports to assess the evolution of a patient disease, authors propose a method to derive disease evolution imagery based on available material from patients and evaluate its accuracy in predicting disease evolution by having generate devolution imagery assessed in comparison of expectations of medical experts.

### Strengths
Well written and tests the propose method/framework through various experimentations (3 different data sets/diseases).

### Weaknesses
The paper needs more clarifications regarding the experimental setting to support the drawn conclusions.

### Questions
- You state: “Moreover, disease progression exhibits significant variability and heterogeneity across patients and disease sub-types, rendering a uniform approach impracticable.”
Question 1: a- What was the number of available imagery per patient?
                 b - For various patients available materiel, was it of the same time frame (one time t, multiple over X months etc.)?  
                 c- Was the disease stage for available imagery uniform between patients?

- You state: " The study presented physicians with a set of simulated disease images and progressions, and then asked them to assess the accuracy and quality of each generated image and progression.”
Question 2: As opposed to presenting the generated evolution image (which might influence the expert judgement) or did you first ask for the expected evolution and then compare with generated result?

- You state: "which helps to establish a deeper understanding of the underlying mechanism”.
Question 3: Can clarify which explainability steps are specifically taken beyond confirmation of expected outcomes/progressions?

- You state: "“each (x, y) is from different individuals.”.
Question 4: a- Did you use only one (Image, text) pair per patient for for the 3 diseases/datasets?
                    b- Is this an experimentation choice to use worst case scenario (one imagery/interpretation text done by every patient) or are all your selected patient imagery diagnoses consisting of one single such imagery test?

Question 5: In Figure 2, It is not clear to us how the Denoising Diffusion Implicit Model is used to simulate the Cardiomegaly’s surface enlargement of the heart footprint in the X-ray. Can you clarify it further?"

- You state: "closely matched physicians’ expectations 76.2% of the time,”
Question 6: Is a global matching rate cross datasets/disease indicative of global performance?  

- You state: "For any given step n in PIE, we first utilize DDIM inversion to procure an inverted noise map. Subsequently, we denoise it using clinical reports imbued with progressive cardiomegaly information.”
Question 7: a- Is only one report used by patient or multiple?
                   b- If multiple, what is the report distribution among patient data used?
                   

- You state : "Raw text input could either be a real report or synthetic report, providing the potential hint of the patient’s disease progression”
Question 8: a- Do you mean expert/human generated for real and automatically/machine generated for synthetic repots?
                    b- Any detail by data set, of the proportions of  real/synthetic reports?
                    c- Any variability in the real reports vocabulary, abbreviations, styles?

- You state: ".. framework proposed to refine and enhance images”. 
Question 9: How do you define refinement of the images? Is it generating the predicted disease progression images?

- You state: ".. use of additional prompts for small and precise adjustments to simulate semantic modification” & “control over specific semantic features of the image”. 
Question 10: As this is first introduction of semantic features in this work, can you indicate which image semantic features you are targeting (presumably by disease)? 
 
- You state: “the disease-changing trajectory that is influenced by different medical conditions.” 
Question 11: Care to clarify. Which ones?

- You state: “PIE also preserves unrelated visual features from the original medical imaging report"
Question 12: a- Care to clarify "unrelated visual features"
                     b- “unrelated” to disease features? 
                     c- What about modifications to non-disease areas (unwanted behavior akin to false positive disease feature)?

- You state: “Each of these datasets presents unique challenges and differ in scale”
Question 13: By “Scale”, do you mean size of the data sets?"

- You state: ““represent whether the simulation results are aligned to target disease”
Question 14: Do you mean “expected disease progression"?


General remarks:
- Please always provide meaning of acronyms in-extenso when first used (HMM, DDIM, ROI, SD Video).
- Figure 1:  You might just in one sentence introduce the reader to what “Cardiomegaly” is supposed to manifest as in the X-ray.
- Figure 2: Barely readable. Explaining the concentric circle representation might help.
- Figure 3: Illustrations are barely readable. “Red” portions are hard to assess. May be differential images (disease progression from previous stage) might be more readable.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
