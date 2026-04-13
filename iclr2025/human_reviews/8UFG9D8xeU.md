## Human Reviewer 1

### Summary
The paper presents a novel approach inspired by inverse reinforcement learning, proposing **Direct Preference Alignment from Occupancy Measure Matching Feedback**. This method aims to align generated behaviors with expert demonstrations by matching occupancy measures in a semantically meaningful feature space. This method does not rely on additional human annotations or complex reinforcement learning but instead leverages the implicit preferences encoded in expert demonstrations. DPA-OMF ranks model-generated samples based on their alignment with expert behaviors using occupancy measure matching in a semantically meaningful feature space. The model is capable of handling up to **128 agents** using a **1M token-prediction model**.

#### Strengths:
1. **Scaling Experiments:** The paper includes comprehensive ablation studies, highlighting the model’s performance when scaling up the number of agents.
2. **Detailed Experimental Setup:** The authors provide thorough descriptions of the experimental setups, including parameters and conditions, contributing to the reproducibility and clarity of their results.

#### Weaknesses:
1. **Some miss proofs in the paper:** "These algorithms collect preference rankings from humans over model generations and directly update the model to maximize the likelihood of preferred behaviors over unpreferred ones." and " Human annotators must analyze intricate and nuanced motions, which is a time-consuming process, making the scalability of direct alignment methods difficult in these scenarios." need more proofs. I think there should be some citations or experiments to show.
2. **Overuse of Colors and Fonts:** The excessive use of different colors and fonts in the main text affects the readability and cohesiveness of the presentation. A more consistent design would improve the clarity of the paper.
3. **Visual Clarity of Images:** Some images in the paper are difficult to interpret due to potential resolution, contrast, or layout issues, which could hinder the reader’s ability to understand the visual data being presented.


I will refine this review based on the author's rebuttal and feedback from other reviewers. As the discussion progresses, further improvements or adjustments to the evaluation will be considered.

### Strengths
1. **Scaling Experiments:** The paper includes comprehensive ablation studies, which highlight the model’s performance when scaling up the number of agents.
2. **Detailed Experimental Setup:** The authors provide thorough descriptions of the experimental setups, including parameters and conditions, contributing to the reproducibility and clarity of their results.

### Weaknesses
1. **Some miss proofs in the paper:** "These algorithms collect preference rankings from humans over model generations and directly update the model to maximize the likelihood of preferred behaviors over unpreferred ones." and " Human annotators must analyze intricate and nuanced motions, which is a time-consuming process, making the scalability of direct alignment methods difficult in these scenarios." need more proofs. I think there should be some citations or experiments to show.
2. **Overuse of Colors and Fonts:** The excessive use of different colors and fonts in the main text affects the readability and cohesiveness of the presentation. A more consistent design would improve the clarity of the paper.
3. **Visual Clarity of Images:** Some images in the paper are difficult to interpret due to potential issues with resolution, contrast, or layout, which could hinder the reader’s ability to understand the visual data being presented.

### Questions
Current experiments have demonstrated the feasibility of the approach on a 1M-scale model. Will it still be effective on larger-scale models?

### Soundness
3

### Presentation
1

### Contribution
2

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
This paper introduces a novel alignment from demonstration (AFD) strategy for multi-agent motion generation in the autonomous driving setting. Compared to direct annotation of preferences by humans, AFD scales better for the multi-agent setting. However prior AFD methods assume all base model's (or fine-tuned version of it) motion samples to be non-optimal while all demonstrations to be optimal. These alignment strategies are inefficient compared to the proposed method, which also compares the relative quality of generated samples among themselves. The paper shows improved alignment after their proposed AFD measured in terms of collision / progress / comfort features in the autonomous driving motion prediction tasks.

### Strengths
The paper is very well motivated and presented. The idea is novel and results are well explained by their visualizations. The proposed optimal transport based distance metric is compared to L2-distance baseline. Insights such as why their method works better than supervised fine-tuning as training continues, preference scaling, preference vs exploitations are also investigated.

### Weaknesses
1. There is a lack of discussion of the assumptions or limitations of the proposed method. For example, one assumption is that the OT-based distance between demonstration and generated samples captures the preferences in a monotonic fashion. Is this always the case in self-driving setting? Another assumption is that asking humans to provide dense trajectory demonstration for multiagent interactions is easier to just rank them (even though there are might be many more pair-wise rankings). Do you have any statistics or references that show the prior scale better than the latter annotation scheme? 

2. The biggest concern is that there is limited if not no comparison with methods this paper is set out to improve: prior AFD methods that assume all base model's (or fine-tuned version of it) motion samples to be non-optimal. While the paper does compare their OT-distance metric works better than L2 distance as well as AFD works better than SFT, their main motivation of improving prior AFD methods is not validated.

3. While they show OT distance works better than L2 distance and AFD better than SFT, the improvement in table 1 results is quite incremental, limiting the contribution of the work.

### Questions
1. Can you compare your method to prior AFD methods and show both quantitatively and qualitatively (in figures) why comparing model generations among themselves help? Can you show some examples of the bias introduced by the heterogeneity of the preference data?
2. What are limitations of AFD? How many human direct annotations do you need vs how many demonstrations of how many cars do you need? How does alignment improves in terms of the labels provided in both cases? (scaling concern in multiagent setting) 
3. If demonstrations are multi-modal, will your method of comparing sampled based of OT-distance metric introduce conflicting gradients and leading to mode collapse?
4. How does your OT-distance metric factor in collision / progress / comfort features?
5. Do you have qualitative figures that help readers understand why L2 distance drop at the end in Fig 3? Why does L2 distance metric will lead to missed turn in Fig5?

### Soundness
2

### Presentation
4

### Contribution
2

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper introduces a method for aligning a token-based motion forecasting model better with demonstrations. The method is based on fine-tuning a pretrained model using a contrastive approach.

### Strengths
- **S.1:** The method shows great results on bringing a 1M parameter model up to the performance of larger models.
- **S.2:** The writing is mostly clear.
- **S.3:** The SFT comparison and Fig.4 are interesting.

### Weaknesses
- **W.1:** I don't fully understand from the paper how the embedding works, the agent feature encoder. Could you please either give me some implementation details or some better high-level overview?
- **W.2:** Some figures are confusing. Fig.1: What are the orange lines on the left above the motion token pred. model? A-hat is not explained. Fig.3: I don't know how to read this diagram. What's the takeaway? Fig.6: I'm completely lost as to what I'm supposed to do with these.

### Questions
- **Q.1:** The writing could use a proofreading pass. There are some minor spelling issues throughout.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3