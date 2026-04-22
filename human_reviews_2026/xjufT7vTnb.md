# TIDAL: A Temporal Causal Diffusion Framework for Visualizing Knee Osteoarthritis Treatment Outcomes

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Generating realistic patient-specific counterfactual images of treatment outcomes from longitudinal medical imaging is a challenging task, complicated by confounding and selection bias in observational datasets. To address this challenge, we propose TIDAL (Temporal IPW Diffusion Adversarial Learning), a novel longitudinal causal diffusion framework that integrates causal inference techniques directly into diffusion model training. TIDAL utilizes a Stable Diffusion backbone conditioned on patient history and incorporates two key causal adaptations: (1) Temporal Inverse Propensity Weighting (IPW) that reweights the diffusion loss based on treatment propensity scores; and (2) Domain Adversarial Training that encourages treatment-invariant representations. We demonstrate TIDAL's effectiveness by simulating knee osteoarthritis (OA) progression with longitudinal X-rays from the Osteoarthritis Initiative (OAI). Performance is assessed using image fidelity metrics and observed treatment effects for OA features like Kellgren-Lawrence grade. Our experiments show that TIDAL significantly outperforms baseline approaches, achieving 21.52\% reduction in image generation error and 18.43\% improvement in observed treatment effects, demonstrating significant improvements for longitudinal medical counterfactual generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a combination of diffusion, causal adjustment as well as adversarial learning to predict treatment results (X-ray). The results show some significant improvement over baselines as well as ablated components in both automated metrics and medical metrics.

### Strengths
1. The problem of the paper, predicting treatment results, is very important and interesting.
2. The paper provides some level of ablations, providing some interesting insights.

### Weaknesses
Disclaimer: I am not an expert in medical imaging, so I am purely commenting from the machine learning perspective.
1. The novelty of the method is limited and I don't see there is much insight offered on why the method is better beyond empirical evidence.
2. The generalization of the method is questionable. More specifically, it doesn't show that it works on other diseases/treatments which can be captured by X-Ray.
3. The evaluation metric is not really convincing (see the question section for elaboration)

### Questions
Since the (KL) Grade and JSN Medial Grade are all graded by models trained from the same limited data distribution instead of human radiologists, wouldn't adversarial training in some sense be optimizing (hacking) against the scorer? Table 1 and table 2 both show that adversarial training is bringing the majority of improvement.

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
3

### Summary
This paper presents TIDAL (Temporal IPW Diffusion Adversarial Learning), a new longitudinal causal diffusion framework for generating patient-specific counterfactual medical images. The model visualize counterfactual medical image, such as knee X-ray images generation under different treatment scenarios. The key chellenge resolved from prior works are confounding and selection bias in observational datasets.

The author proposed causal inference into diffusion models via to contributions:
- A Temporal Inverse Propensity Weighting(IPW) : Using a RNN based model that re-weight the diffusion loss for estimated treatment propensity scores to balance covariate distributions over time.
- Domain Adversarial Training : using a disccriminator that encourages treatment-invariant representations to reduce confounding.

The model is evaluated on a public dataset, Osteoarthritis Initiative, generating longtitudinal knee X-rays. The author mentioned the result shows that TIDAL achieves 21.5% reductoin in image generation error and 18.4% improvement in treatment.

Overall, the paper has good soundness and fair novelty, and the context on counterfactual image generation and prediction is a frontier. The writing is clear and good to follow.

### Strengths
1. The author proposed a novel architecture to integrate causal inference with diffusion modeling. It combines the Temporal IPW and adversarial training within stable diffusion v1.5, which novel in longitudinal, counterfactual medical image generation, for example Kee Osteoarthritis treatment. The author also claimed their work is the first in the area.

2. The paper provides solid theoretical justification through a risk decomposition theorem.

3. Good experiment performance compared to baselines with both quantitative results and qualitative analysis.

### Weaknesses
1. The paper proposed to use alternate adversarial training, which may cause training instability and difficulties in hyperparameter tuning. It could be better to discuss more on the training convergence and robustness.

2. The paper would benefit from a more detailed validation of the IPW model. Such as its accuracy and the resulting covariate balance after reweighting.

### Questions
The ablation mostly focuses on the adversarial loss weight lambda. Would it be possible to include ablations removing either the IPW or the adversarial component individually, to show their distinct contributions to performance? This would clarify which module provides the main causal benefit.

Do authors expect the framework to generalize to other modalities (e.g., MRI, CT) or other longitudinal conditions?

### Soundness
4

### Presentation
3

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
This paper presents **TIDAL (Temporal IPW Diffusion Adversarial Learning)**, a framework for generating counterfactual medical images that simulate treatment outcomes over time. The method extends causal diffusion approaches (e.g., DiffPO) by integrating **temporal inverse propensity weighting (IPW)** and **domain adversarial training** into the diffusion process, aiming to address confounding in longitudinal datasets. Experiments on the Osteoarthritis Initiative (OAI) dataset are reported, where TIDAL is claimed to generate more realistic and treatment-consistent knee X-rays than baseline diffusion and ablated variants.

### Strengths
- The topic is timely and relevant — causal generative models for longitudinal and medical imaging are of clear scientific interest.  
- The proposed combination of **temporal IPW with diffusion modeling** is a reasonable extension of prior work and could, in principle, improve fairness and causal faithfulness in generative models.  
- The empirical setup demonstrates some thoughtfulness, evaluating both reconstruction (MSE, SSIM) and treatment consistency.  
- The paper shows awareness of recent work in causal generative modeling (e.g., DiffPO) and attempts to adapt it to a medical longitudinal context.

### Weaknesses
- **Writing and structure:** The paper is **poorly written and difficult to follow**. The introduction lacks a clear motivation or narrative. Technical details (e.g., temporal IPW) are given early without context or intuition. The exposition frequently focuses on mathematical detail rather than providing a conceptual story.
  - Theorem 1 and the empirical risk bound feel unnecessary and could be placed in the appendix.  
  - Line 370 is confusing: does “alternated between” mean that the IPW and diffusion models are trained jointly or in alternating steps?  
  - It is unclear whether multiple treatments are modeled or just one.
  - A background section is probably needed, as it currently feels really entangled with methodology, confusing readers on novelty too.
- **Incremental contribution:** The approach appears to be an **adaptation of DiffPO** for the longitudinal setting.
- **Lack of clarity in methodology:** The motivation for using domain adversarial training instead of just a simple stabilized IPW is not justified.  
- **Weak evaluation:**  
  - Only **ablations** are reported; there are **no comparisons with other published baselines** (e.g., DiffPO, Diff-CF).  
  - No information about **seeds, statistical significance**, or robustness.  
  - Evaluation uses only **MSE and SSIM**, which measure reconstruction quality but not causal validity or perceptual realism. At least one realism metric such as **FID/sFID** should have been included.  
  - No human, expert, or clinical evaluation is performed.

### Questions
1. Could you better explain *why* IPW is beneficial for this task, beyond citing standard causal adjustment? What confounding pattern is it correcting in your dataset?  
2. Why did you choose domain adversarial training instead of stabilized IPW?  
3. What exactly does “TIDAL alternated between” mean — are the IPW and diffusion modules optimized alternately in a loop?  
4. Are multiple treatment types modeled? If so, how are they handled in training and sampling?  
5. Why not include existing baselines or realism metrics (FID/sFID)? Without them, how can we assess if TIDAL provides genuine gains?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors generate counterfactual X-ray images to visualize the patient knee after osteoarthritis treatment effects. X-ray image pairs recorded at different time points, the time difference, patient history, the treatments, etc. are utilized to fine-tune a stable diffusion model and train an IPW RNN, a discriminator, a generator, etc. Finally, the authors show their algorithm performance on the osteoarthritis dataset.

### Strengths
It is an easy-to-read paper as written in an intuitive approach.  Although it was clear from the text how the image pairs were created, Figure 2 helps to understand it better. For evaluating the model performance, the authors used a pre-trained model to predict the clinical variables for the generated images. I appreciate that the authors did not keep their evaluation limited to only image quality.

### Weaknesses
Here are my comments:

Major:
1. How is multiplying the propensity score with diffusion loss equivalent to reweighting each sample with the propensity score in standard IPW? What is the proof?
2. My understanding is that the goal of the adversarial training is to control the confounding bias. However, the authors mentioned at the beginning of Section 3.4 that IPW training is unstable due to some treatment probabilities being close to zero. The connection between these two concepts is not explicit in the paper.
3. The experiment results are limited. The authors should show experiment results on synthetic datasets to illustrate that the proposed algorithm can generate correct counterfactual images at different time points for the same initial images.
4. The authors cited multiple papers that perform counterfactual diffusion. However, they did not compare their performance with any such baselines. To my understanding, the results presented in Tables 1 and 2 are more like ablation studies where the baselines refer to different variants of the proposed algorithm, i.e., progressive inclusion of model training in the algorithm. Since the authors have image pairs, the time difference, and patient history, they should be able to use this dataset to evaluate existing algorithms and compare their performance against existing works.
5. The novelty of the proposed algorithm is not obvious. It appears like multiple models with known capabilities are connected with each other. I would request the authors to make it clear.

Minor:
1. In Equation 2, it is not mentioned what $k$ refers to. Probably the index for the treatments. If so, does that mean the authors train $K$ LSTM models? Why can’t they train one model and output a vector of treatment probabilities?
2. “Assuming conditional independence of treatment assignments within the interval given the conditioning variables”  this probably refers to the assumption that treatments don’t interact with each other. This is an important assumption and should be explained explicitly.
3. Figure 1 does not represent the execution steps properly. Also, where is the generator $G$ in Figure 1?
4. It is not mentioned how the hyperparameter $\lambda_{adv}$ is chosen.
5. What did the authors mean by “a target (intervention) policy” in lines 315–316? What is the definition of policy in this context? How is the policy different from $P(A \mid H)$? Where do the authors get the estimation of this policy to use it in Equation 10?
6. Difference between the LSTM and the treatment generator is not clear.
7. The authors mentioned that they use the stable diffusion model. However, they did not provide enough details on how that helps their training.

### Questions
Below I share my questions:

Questions:
1. Line 312: Should it be $(P(A \mid H))$?
2. How many X-ray images did each patient have in the considered dataset?
3. In Figure 1, the target image is different from the image shown in historical patient covariates. Why are these different? At what time points were these images collected?
4. How significant is the performance in Table 1 when the predicted noise MSE changes from 0.1361 to 12.94 and MSE changes from 0.0079 to 0.0062? Although these metrics are dataset-dependent, the change appears very small. How do the authors justify this?

### Soundness
3

### Presentation
3

### Contribution
2
