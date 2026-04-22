# Great GATsBi: Hybrid, Multimodal, Trajectory Forecasting for Bicycles using Anticipation Mechanism

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 4, 6

## Abstract
Accurate prediction of road user movement is increasingly required by many applications ranging from advanced driver assistance systems to autonomous driving, and especially crucial for road safety. Even though most traffic accident facilities account to bicycles, they have received little attention, as previous work focused mainly on pedestrians and motorized vehicles. In this work, we present the Great GATsBi, a domain-knowledge-based, hybrid, multimodal trajectory prediction framework for bicycles. The model incorporates both physics-based modeling (inspired by motorized vehicles) and social-based modeling (inspired by pedestrian movements) to explicitly account for the dual nature of bicycle movement. The social interactions are modeled with a graph attention network, and include decayed historical, but also anticipated, future trajectory data of a bicycles neighborhood, following recent insights from psychological and social studies. The results indicate that the proposed ensemble of physics models - performing well in the short-term predictions - and social models - performing well in the long-term predictions - exceeds state-of-the-art performance. We also conducted a controlled mass-cycling experiment to demonstrate the framework's performance when forecasting bicycle trajectories and modeling social interactions with road users.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces Great GATsBi, a hybrid multimodal framework for bicycle trajectory prediction that combines physical dynamics and social interaction modeling. The physics module includes constant velocity, constant acceleration, kinematic, and Kalman filter models, encoded through LSTMs. The social module integrates neighbor trajectories through GATs to model inter-agent interactions. Evaluated on a multi-cyclist dataset collected from the authors' controlled real-world experiment, Great GATsBi outperforms Social-LSTM and Social-BiGAT in ADE/FDE over 1s~4s horizons and demonstrates modest generalization on pedestrian datasets ETH and HOTEL.

### Strengths
The paper introduces anticipation and perception decay mechanisms in social interaction modeling, drawing on psychological principles of human decision-making to explain cyclists’ characteristics of “limited memory and foresight” when predicting neighbors’ behaviors.

The authors collected a controlled real-world cycling dataset containing multi-cyclist interaction behaviors, which effectively validates the model’s capability to represent such scenarios.

### Weaknesses
1. The paper oversimplifies the problem by ignoring road and environmental context, which limits its real-world applicability.
2. The model architecture is simple and largely follows existing pedestrian trajectory prediction frameworks; it remains within the standard GAT + LSTM paradigm with little technical novelty.
3. The chosen physical dynamics models are generic for all vehicles and cannot adequately capture the unique dynamics of bicycles.
4. The social interaction modeling remains oversimplified, especially under dense or complex traffic conditions. Although the authors emphasize that the main challenges of pedestrian and cyclist trajectory prediction stems from sudden trajectory changes, the current model still relies solely on historical trajectories, neighbor anticipation, and perception decay, while neglecting richer and more dynamic social behaviors [1]. I am not sure whether only these two mechanisms are sufficient to address above key challenge in bicycle trajectory prediction.

5. (1) The experimental setup is closed and limited. It lacks mixed-traffic scenarios involving vehicles, whose psychological and physical influence on cyclists could be crucial but is ignored. (2) The uncertainty evaluation only shows a single trajectory and does not convincingly demonstrate the model’s generative advantages. (3)The authors claim strong long-term prediction performance, but the actual prediction horizons (1–4 s) is too short to support this claim.

[1] Martín-López IM, García-Taibo O, Aguiló A, Borràs PA. Understanding Factors Influencing Cycling Behaviour Among University Students and Staff: A Cross-Sectional Study. Social Sciences. 2025; 14(5):261. https://doi.org/10.3390/socsci14050261

### Questions
1. What challenges do the authors identify for bicycle trajectory prediction compared to pedestrian trajectory prediction?

2. What technical innovations does the GATsBi introduce compared to existing pedestrian trajectory prediction research?

3. How are the hyperparameters $\lambda_h$ and $\lambda_p$ set in the perception decay module?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a hybrid multimodal trajectory prediction framework, called Great GATsBi, designed specifically for cyclists, who represent a vulnerable road user group. The framework addresses the “dual nature” of cyclist behavior, which combines characteristics of both motor vehicles and pedestrians. It consists of a physical module that captures vehicle dynamics and a social module that models social interactions using a graph attention network. Its core innovation includes a psychology-inspired anticipation mechanism and a perceptual-decay concept, along with the introduction of a large-scale cycling-experiment dataset collected in a controlled environment to study social dynamics. Experimental results show that the model outperforms multiple baseline methods on the proposed dataset.

### Strengths
1.	This paper clearly identifies cyclist trajectory prediction as an understudied problem and offers strong evidence for the mixed behavioral patterns observed in cyclists. Inspired by psychology, the paper introduces the concept of “anticipation” to model interactions between agents, providing a novel and meaningful research direction for the field.
2.	The authors have developed and plan to release a new large-scale cycling dataset. By collecting data on a controlled circular track, the dataset isolates social dynamics from confounding factors such as road topology, thereby filling a gap in existing datasets and providing a valuable resource for the research community.
3.	Using their self-constructed dataset, the authors systematically demonstrated the superiority of their hybrid method and the contribution of each component through extensive benchmark comparisons and ablation studies. In addition, qualitative analysis further illustrated the model’s effectiveness in an intuitive manner.

### Weaknesses
Weaknesses
1.	The paper’s core concept of an “anticipation mechanism” is ultimately implemented using a simple constant-velocity model. This implementation falls significantly short of the complex cognitive processes underlying human anticipatory behavior, which involve intention and planning. As a result, the depth and persuasiveness of the technical contributions are substantially weakened.
2.	The model performed poorly on the standard ETH/HOTEL pedestrian dataset, achieving results significantly worse than even simple baseline methods. This suggests that the model may have overfitted the simple and homogeneous dynamics of its self-built dataset, resulting in limited applicability to complex real-world scenarios and weak generalization ability.
3.	The baseline models used for comparison in this paper (SocialLSTM and Social-BiGAT) are relatively outdated. The paper does not include comparisons with more recent and advanced architectures that have become SOTA in trajectory prediction.
4.	The ethics statement asserts that “no human subjects are involved,” yet the appendix reports obtaining written consent from “volunteering participants.” This inconsistency should be clarified.

### Questions
1. Why did you choose a simple constant-velocity model to implement the complex “anticipation” function? Have you explored using more sophisticated models to generate predictions of neighboring agents’ trajectories?
2. Aside from dataset scale, what do you consider to be the fundamental causes of the model’s poor performance on the ETH/HOTEL dataset? How could the model be improved to better adapt to more complex real-world environments?
3. How does the expected performance of your model compare with SOTA methods based on Transformers or diffusion models?
4. Please clarify the inconsistency between the ethics statement and the description of human participants in the appendix.

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
This paper presents yet another LSTM model for cyclist motion prediction. Unlike previous methods, this paper innovates a fusion mechanism that makes predictions using (1) purely physics and (2) social interactions. These two predictions are then combined. The social interaction part additionally uses a constant velocity model to interpolate future states and encode them, mimicking a human’s anticipation of the other cyclists. The output is a Gaussian Mixture Model that captures the multi-modal nature of the predictions. The authors collected their own dedicated cyclist data with a length of 270 minutes and compared it against SocialLSTM Social-BiGAT, and observed some improvement in terms of ADE and FDE.

### Strengths
-	Separately predicting with bicycle dynamics and social interaction, then fusing them for motion prediction, is novel.
-	The authors organized some cyclists and collected 270 minutes of cycling data with a drone, which was then annotated with YOLO and conventional vision algorithms.
-	The experiments show overall improved results compared to SocialLSTM.

### Weaknesses
-	The newly collected cycling data are simply just riding within a roundabout, with no enforcement on the interaction between cyclists. The readers can imagine that, if everyone just rode in an orbit, no interaction could happen at all, thus raising doubt on the necessity to model interactions. Furthermore, the lack of diverse scenarios beyond roundabouts makes readers concerned about the model’s generalizability.
-	The fusion is essentially a mixture-of-experts where the experts are a combination of a rule-based predictor and a learned predictor. The contribution is thus incremental.
-	The authors designed a parallel, independent prediction mechanism in the early stage of the model. There are many alternate designs. For example, since the const_v, const_acceleration, and kinematics can be done with simple differentiable linear algebra, how about making these as layers **after** the social module? In this case, the dynamics can be enforced explicitly on top of a learned prediction, similar to [1]. 
-	In the experiments, we see the improvements are fairly small, i.e., often ~0.01m. Intuitively, this value is smaller than the labeling noise when drawing bound boxes on drone images. It is recommended to perform multiple rounds of experiments on the same configuration and see if the improvement is consistent.
-	There is no ablation study on ablating each combination of the physics-based predictor. Some of them could be redundant with each other.
-	Line 068: a reference is compiled as a question mark.
-	Line 257 typo: Gauss -> Gaussian
-	Line 392: SocialLSTM seems to have a better result at 4s FDE, but is not bolded. 

References
[1] Jiao, Ruochen, et al. "Kinematics-aware trajectory generation and prediction with latent stochastic differential modeling." 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024.

### Questions
-	Section 3.3: Are both modules predicting the same horizon?
-	Eq.4: Since each trajectory is discrete, should that be a sum rather than an integral?

### Soundness
2

### Presentation
2

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
This paper proposes Great GATsBi, a hybrid multimodal framework for bicycle trajectory forecasting. It integrates physics-based modeling (capturing vehicle-like dynamics via constant velocity, constant acceleration, kinematics, and extended Kalman filtering) and social-based modeling (capturing pedestrian-like social interactions via Graph Attention Networks, GATs). The framework also incorporates an anticipation mechanism and perception decay inspired by psychological and social science insights. It is evaluated on a self-collected controlled mass cycling dataset (circular track, varying traffic density) and generalized to pedestrian datasets (ETH, HOTEL). The paper claims Great GATsBi outperforms physics-based and social-based baselines, addressing the gap of neglecting bicycles’ dual behavioral nature in prior trajectory forecasting work.

### Strengths
1. The model develops a hybrid framework that combines physics-based and social-based modeling to effectively match bicycles’ dual behavioral traits of vehicle-like dynamics and pedestrian-like flexibility.
2. The social module innovatively integrates psychological and social science insights (neighbor trajectory anticipation and perception decay) to make social interaction modeling more realistic.
3. A high-quality controlled mass cycling dataset is built to avoid external interferences, providing reliable support for verifying the model’s performance in bicycle dynamics and social interactions.

### Weaknesses
1. Risk of Circular Logic in the Core Innovation: The "anticipation mechanism" in the social module requires predicting the future trajectories of neighboring agents (using a simple const.v model, mentioned in ilne 239) to serve as input for forecasting the ego agent's future. This creates a potential circular argument: predicting agent A's future relies on first predicting agent B's future, which is itself a challenging prediction problem. The model sidesteps this fundamental issue rather than solving it. If the const.v predictions for neighbors are unreliable, this "anticipated" input may introduce noise rather than beneficial information.
2. Unclear Motivation for Physics Models: The physics module ensembles four models, but their fusion is performed opaquely through LSTM encoding and concatenation. The paper fails to justify why this specific combination of models is necessary and sufficient, nor does it provide significance analysis to demonstrate each model's unique contribution. Notably, since the simplest const.v model performs best among the individual baselines (Table 1), the motivation for including the more complex and poorer-performing kinematic and xkalman models is questionable. This appears more like model stacking than a deliberate design. An ablation study comparing different subsets (e.g., 2 or 3 models) is needed to substantiate that using all four is optimal.
3. Writing: Line 68 appears to have a missing citation.
4. Figures: Figure 1 is blurring, and the overlaid trajectories are difficult to discern.
5. Social Graph Construction: Line 284 states "at most five neighbors at a distance below 20m are considered," but the specific selection strategy (e.g., the five closest? random selection?) is not specified. This strategy can significantly impact the results and should be discussed or ablated.
6. Lack of Novelty: The core methodology primarily combines existing techniques: GATs for social modeling (from Social-BiGAT), physics model ensembling, and multimodal output (GMM).

### Questions
Please refer to the weaknesses for my main concerns.
In addition, I would like the authors to clarify the following:
1. Could the authors further clarify how the proposed anticipation mechanism avoids the potential circular reasoning problem identified in the weaknesses?
2. Could the authors provide an ablation study on the physics model ensemble (e.g., using different combinations of the four models) to conclusively demonstrate the necessity of including all of them?
3. Could the missing citation on line 68 be added, and could the resolution of Figure 1 be improved?
4. Could the authors specify the neighbor selection strategy for the social graph and potentially include an ablation study on its impact?
5. Why were bicycle-specific baselines excluded from comparisons? If pedestrian baselines (SocialLSTM/Social-BiGAT) were adapted to bicycle dynamics (e.g., adjusting for speed), would their performance narrow the gap with Great GATsBi?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Great GATsBi, a hybrid multimodal framework for bicycle trajectory forecasting. It combines physics-based modeling of motion dynamics with social interaction modeling using a graph attention network. Inspired by psychology, the model includes anticipation and perception decay mechanisms to better capture how cyclists respond to their surroundings. Experiments on a real-world cycling dataset show clear improvements over physics-only and social-only baselines, and the method generalizes reasonably well to pedestrian prediction.

### Strengths
Combines physical and social modeling in a clear and interpretable way.

Provides extensive experiments and a new real-world dataset.

### Weaknesses
Technically, the framework still relies on conventional components (LSTM + GAT + GMM). Is there any plan to explore whether transformer-based or VLMs could further improve representation and generalization?

The model’s transferability to complex urban or mixed-traffic environments has not been validated, as experiments are limited to a controlled cycling scenario.

### Questions
pls refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
