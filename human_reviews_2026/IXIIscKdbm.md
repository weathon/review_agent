# A Multimodal Label Forecasting Method for Aperiodic Visuo-Motor Time Series

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Deep learning models have been increasingly applied to Time Series Forecasting (TSF) in recent years. Transformer-based and MLP-based models have both been used effectively on many real-world TSF regression benchmarks, and there is ongoing debate as to which family of methods is best.  While these benchmarks have drawn much attention, it is also worth noting that many current datasets and methods assume approximate periodicity in the time series.  In this work, we focus on a new TSF task without periodicity: anticipating falls during humanoid locomotion, on the basis of egocentric vision and proprioception.  When the locomotion trajectories are sufficiently diverse, periodicity is violated.  We contribute two new benchmark datasets (one from simulation, one from real hardware), showing that periodicity is violated and recent deep TSF methods struggle on these benchmarks.  We also propose a novel deep learning architecture that exploits both endogenous and exogenous variables and a training process that rigorously enforces i.i.d sampling of training examples. Our results show statistically significant improvement over prior art in multiple experimental conditions, by 12.73\% or more on the real data and 10.40\% or more on the simulation data. Code and datasets will be available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This submission presents a new benchmark and method for Time Series Forecasting (TSF). It argues that the community is overly focused on periodic regression tasks and targets a new fine-grained classification challenge: predicting per-time-step fall labels for a humanoid robot using aperiodic, multimodal sensory input.

To this end, it presents two valuable new datasets (RP: real-world, SimP: simulated) containing egocentric video, proprioceptive data (actual joint angles), and planned motion trajectories. In addition, a EMP model is proposed which uses three parallel encoders to process this data. A key feature of this model is its use of the planned trajectory as an exogenous variable, including the portion of the plan that extends into the future ($P$ steps).

Experiments show that EMP (even the ablated version MP) outperforms several TSF models like TimesNet and FlowFormer and the domain-specific EgoFalls on this new benchmark. The authors conclude that EMP is a superior baseline for this new problem, and they attributes EMP’s success to its multimodal architecture and designs for handling aperiodic data.

### Strengths
**(S1)** Clear problem formulation and new benchmarks. This work provides two new datasets (RP and SimP) for TSF classification under aperiodic, real-world robotic conditions. This fills a substantial gap in existing TSF benchmarks that focus on periodic or stationary signals. The new task of per-time-step fall prediction is of clear practical and research importance.

**(S2)** The use of exogenous data. To my knowledge, the key contribution of this work is its use of the planned motion trajectory as an exogenous variable. Leveraging the known future $N+P$ steps of this plan is a fresh and powerful advance. This is a far more meaningful use of exogenous data than the field's standard (and rather weak) reliance on simple time-stamp encodings.

**(S3)** Thorough experiments. The empirical work is robust. It first shows that SOTA regression models fail on this data in Tab. 1, justifying their new method. The comparison with relevant classification and domain-specific baselines shows big improvements with significant p-values in both simulated and real robot scenarios. What I appreciate the most are the multiple repetitions, standard deviations, and statistical tests (Walsh's t-tests and p-values), which are applied to support claims of significance, rather than just cherry-picked runs.

**(S4)** Ablation studies and analysis. Tab. 3 and Tab. 4 provide ablation results to break down the contributions of each encoder (vision, proprioception, and trajectory) and the significance of future planned trajectories especially in simulation. Particularly, MP ablation (removing vision) reveals that the model's strength derives mainly from the proprioceptive and trajectory inputs, not the vision stream. Training overhead for added modalities is also reported for efficiency evaluation.

### Weaknesses
**(W1)** This work’s core premise seems to be directly contradicted by its method. It is built on the premise of tackling aperiodic data. However, in Line 272, the Trajectory Encoder is designed specifically to exploit the "more reliable periodicity... in a repetitive gait" of the planned trajectories, and uses an FFT-based filter to do so. This is a direct contradiction. I recommend the authors add clarifications in the revised manuscript to address this point.

**(W2)** IMHO, the comparison seems a bit unfair. The baselines were almost certainly not given access to the same powerful exogenous data (the future plan) that the EMP model uses. In other words, the MP ablation (without vision) achieves 90.40% on SimP, while the best baseline (FlowFormer) scores only 80.00%. This may suggest the 10.4% gap is not due to EMP architecture, but rather that MP has access to $N+P$ future plan and baselines do not. I look forward to the author's response on this and hope they can incorporate insightful comments into the revision.

**(W3)** An unexplained "sim-to-real" gap exists. Key components (like the future plan and the vision encoder) are critical in simulation but are marginal or even detrimental on the real-world data. Specifically, the future plan is essential on SimP (with an 11.7% drop when removed) but irrelevant on RP (only a 0.01% drop). The explanation now (“limited diversity”) is a bit insufficient. 

**(W4)** The task definition is a bit ambiguous. It is unclear whether the model is truly forecasting the onset of a fall or simply learning a trivial persistence model (i.e., "if I am already falling, I will still be falling in $P$ steps"). In this case, it is not a forecasting task. I believe this needs further clarification.

**(W5)** Several limitations in RP dataset. The RP dataset is limited in size (110 episodes) and diversity of environments and trajectories. Also, there are not enough scenarios for each environment (only three locations), which could affect the validation.

**(W6)** The title of Sec. 2.3 is too general. Since it is specifically about multimodal time series forecasting, revising the title to “Multimodal TSF” would better reflect its scope and avoid confusion with general multimodal representation learning.

**(W7)** Some notation issues reduce readability. (i) The symbol **M** (in Line 156, Sec. 3.2.1) and subsequent uses are overly bolded, which is a bit wired in the entire manuscript. (ii) Similarly, **V**, **K**, and **T** in Eqs. 8–10 suffer from the same issue. (iii) Some operators and function names are typeset in italics, which are commonly noted as upright (\text{} or \operatorname{}) to distinguish them from variables. Specifically, (e.g., “Spa()” and “TepMLP()” in Eq. 6, “Conv3D” in Eq. 9, “ConvBlk” in Eq. 10, “Perturbation” and “factor” in Eqs. 11–12).

### Questions
Most of my concerns and related recommendations have been stated in the Weaknesses section. I encourage the authors to focus their efforts on addressing those points, as they are critical for strengthening the manuscript in the rebuttal stage.

The following are more specific, minor questions to help the authors think more deeply about certain design choices and experiment setups, which I hope might be helpful for this and future work:

**(Q1)** Given that ablation shows unimodal MP often matches or outperforms the multimodal EMP (especially for certain $P$ values on RP), what regularization strategies were attempted, if any, to stabilize multimodal learning? What insights can the authors offer about the situations/environments where fusion of vision actually delivers benefit?

**(Q2)** Would EMP's architecture (vision + proprioception + exogenous planning) port well to different domains, or is the value mainly restricted to robotic forecast settings where planned forward trajectories are known and the main source of exogenous data?

**(Q3)** Have variants where DCT/iDCT is replaced with learned temporal filters like Conv kernels been tried? How significant is the DCT feature transformation compared to other modeling components?

---

## Justifications:

The main idea of this paper is fresh and strong. The new datasets and the focus on exogenous future plans are great advances, and the authors have clearly identified a valuable problem. More importantly, I can catch the soundness of the experiments and analysis.

However, several concerns exist as listed in Weaknesses. The model uses future trajectory plans as input, while baselines appear not to. The difference in performance is probably due to unequal information, not better design. Additional concerns include the contradiction between the aperiodic framing and the method’s reliance on periodic gait, the sim-to-real gap, and whether the task truly involves forecasting or label persistence.

Therefore, I cannot recommend acceptance at this stage and give a rating of 4. I would be glad to raise my rating if thoughtful responses and improvements are provided in the rebuttal stage. I am also open to follow-up discussions with the authors to help further strengthen this work.

I hope these comments help my fellow reviewers and ACs understand the basis of my recommendation.

### Soundness
2

### Presentation
3

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
This paper focuses on the task of predicting time series that lack periodicity. The specific contribution lies in providing both a real and a simulated dataset to determine whether the robot has fallen. Each frame in the dataset has its own label for each time step. Additionally, a planned joint trajectory is used as the exogenous time series to assist in label prediction.

### Strengths
- The paper is clearly presented

### Weaknesses
The issue of non-periodic label prediction at each time step emphasized in this work is essentially a time series classification or anomaly detection problem. This is common in general time series classification or anomaly detection tasks, as seen in the widely used UCR database, which contains many data samples from the robotics application field. Therefore, from this perspective, the novelty of this work does not seem significant to me. Specifically, the label forecasting mentioned in the title can more accurately be described as classification.

Additional suggestions:

- The comparative methods could be thoroughly compared with multimodal methods related to time-vlm.
- This work also seems relevant to multimodal action recognition or motion prediction, and there should be many related studies in the field of computer vision.
- The term $𝐿_{t+P}$  in line 134 is ambiguous. It is unclear whether it refers to a single value or P values, since you mentioned "occurring P time steps in the future".
- There is too little information regarding vision sequences in the text.
- More visual analyses should be included, discussing the challenges of the classification tasks studied (beyond the so-called non-periodicity).

### Questions
- What are the vision sequence inputs you meant in the model figure?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a new robotics benchmark for aperiodic time series forecasting (TSF): predicting falls in humanoid locomotion using egocentric vision and proprioception. The authors contribute two datasets (real hardware “RP” and simulated “SimP”)  from a humanoid robot. They also propose a multimodal method that leverages observed states, planned trajectories (exogenous inputs), and vision for per-timestep classification, showing improvements over existing TSF baselines.

### Strengths
Overall, the paper is well-written and easy to follow. Given the lack of aperiodic benchmarks, the two collected datasets are very valuable. The motivation for aperiodic TSF and for per-timestep label forecasting is clear. The proposed multimodal approach is principled, and the consistent accuracy gains over baselines support its effectiveness.

### Weaknesses
Some concerns:

1. The authors mentioned the Exchange Rate dataset as an exception to periodicity among standard benchmarks, but do not report any experiments on it. All reported experiments are on their two datasets (RP and SimP) for the main per-timestep classification task.

2. To make the proposed EMP method more reliable, the authors can consider evaluating EMP/MP on at least one public aperiodic dataset (e.g., Exchange Rate) to complement the new benchmarks.

3. In the RP/SimP comparisons of EMP and other baselines, please clarify whether any baselines are multimodal. If not, can the authors add a simple multimodal baseline for fair comparison?

4. The authors mentioned that MP outperformed EMP on the RP dataset when P = 18 and attributed this to the possible vision overfitting. Can you substantiate this with any targeted checks?

### Questions
See weaknesses.

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
4

### Summary
The paper introduces a multimodal architecture for per time‑step label forecasting in aperiodic visuo‑motor streams. The task is to predict whether a humanoid robot will have fallen P steps in the future given a short history of egocentric video and proprioception, augmented with known future exogenous inputs (planned joint trajectories). The authors contribute: (i) a fine‑grained TSF classification task distinct from standard regression or whole‑sequence classification; (ii) two benchmark datasets, RP (real Poppy humanoid, 110 episodes at ~16 fps across three indoor locations) and SimP (PyBullet simulation, 2k episodes at 30 fps) with analyses arguing limited periodicity; and (iii) the EMP baseline that merges a DCT‑based motion encoder, a low‑frequency 3D‑Conv vision encoder, and a trajectory encoder that filters top‑k rFFT components.

### Strengths
- The problem statement is clearly written and easy to follow, and the overall experimental setup is transparent at a high level

### Weaknesses
- The problem being solved is not quite complicated, forecasting whether a robot is going to slip/fall or not N steps into future is a simple classification problem and has been tackled even simply with proprioception (e.g., paper on Predictive Proprioception, IROS 2022).

- It is not understood why a complicated architecture is required for solving this problem. 

- Baseline coverage is too narrow; simple methods are missing.The comparison set is restricted to recent deep TSF models (TimesNet, FlowFormer) and an egovision fall detector (EgoFalls). There are no non‑deep baselines (e.g., logistic regression, linear/regularized classifiers, gradient‑boosted trees, or simple temporal CNN/LSTM baselines) trained on engineered features from joint states and known‑future trajectories, exactly the signals that appear most predictive here. Given that MP is already strong, classical models with derivatives of joint angles, short‑window statistics, and trajectory deltas could be competitive and would directly test the claim that deep multimodal fusion is necessary.

- Proprioception‑only literature is under‑engaged. The related‑work section does not engage with prior work that uses proprioception alone for near‑term failure or slip prediction in locomotion (as mentioned above). The manuscript should position its contribution relative to proprioception‑based forecasting and control papers that demonstrate strong predictive signals from joint histories alone, and then justify when and why egovision materially helps this particular forecast.

### Questions
See weakness section

### Soundness
2

### Presentation
3

### Contribution
1
