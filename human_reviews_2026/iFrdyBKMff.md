# PreferThinker: Reasoning-based Personalized Image Preference Assessment

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Personalized image preference assessment aims to evaluate an individual user's image preferences  by relying only on a small set of reference images as prior information. Existing methods mainly focus on general preference assessment, training models with large-scale data to tackle well-defined tasks such as text-image alignment. However, these approaches struggle to handle personalized preference because user-specific data are  scarce and not easily scalable, and individual tastes are often diverse and complex. To overcome these challenges, we introduce a common preference profile that serves as a bridge across  users, allowing large-scale user data to be leveraged for training profile prediction and  capturing complex personalized preferences.  Building on this idea, we propose a reasoning-based personalized image preference assessment framework that follows a \textit{predict-then-assess} paradigm:  it first predicts a user's preference profile from reference images, and then provides interpretable, multi-dimensional scores and assessments of candidate images based on the predicted profile. To support this, we first construct a large-scale Chain-of-Thought (CoT)-style personalized assessment dataset annotated with diverse user preference profiles and high-quality CoT-style reasoning, enabling explicit supervision of structured reasoning. Next, we adopt a two-stage training strategy: a cold-start supervised fine-tuning  phase to empower the model with  structured reasoning capabilities, followed by reinforcement learning to incentivize the model to explore more reasonable assessment paths and enhance generalization. Furthermore, we propose a similarity-aware prediction reward to encourage better prediction of the user's preference profile, which facilitates more reasonable assessments exploration. Extensive experiments demonstrate the superiority of the proposed method. Our code and dataset will be  publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new CoT-based methodology for personalized image preference assessment and introduces a large-scale CoT-style dataset. Most existing work captures general human preferences and cannot handle personalized ones. The paper introduces the concept of "common preference profile" and proposes a two-stage "predict-then-assess" method. The interpretable preference profile and reasoning steps help MLLMs to achieve better alignment for each user's personalized image synthesis.

### Strengths
1. The innovative “Common Preference Profile” efficiently bridges the gap between large-scale general preference alignment and personalized preference modeling.
---
2. The proposed "predict-then-assess" paradigm decomposes the task into *profile prediction* (summarizing human preference across several pre-defined dimensions) and *assessment reasoning* (interpretation and scoring). This decomposition provides **better modularity for image generation** and interpretability.
---
3. The paper empirically verifies that carefully designed reasoning structures—incorporating CoT reasoning to (1) predict each user's preference profile and (2) assess candidate images with the uncovered preference profile—achieve state-of-the-art performance on human preference prediction.

### Weaknesses
The discrete elementary preference profile may oversimplify the real-world human's nuanced preference, as it reduces complex, context-dependent aesthetic judgments into a fixed set of categorical dimensions that may not capture the subtle variations and contradictions inherent in individual taste.

### Questions
Humans have inconsistent preferences across different topics and scenarios. How does the proposed elementary preference profile approach address this complexity? Can you clarify the motivation behind using independent element-wise preference profiles rather than capturing the continuous latent representation for users' preferences?

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
This paper presents PreferThinker, a reasoning-based personalized image preference assessment system. The key idea is to predict a preference profile to bridge various uses, allowing large-scale user data to be leveraged for training profile prediction and capturing complex
personalized preferences. To this end, a CoT-style dataset annotated with preference profiles and high-quality reasoning for interpretability supervision is constructed. A two-stage a two-stage training strategy comparing Cold-start SFT and reinforcement learning is utilized to enpower the model with reasoning capabilities. Experiments on the constructed dataset shows that the proposed method outperforms existing approaches.

### Strengths
This paper addresses personalized image preference assessment from a novel visual preference profile based interpretable perspective. 

A large-scale Chain-of-Thought (CoT)-style personalized assessment dataset annotated with diverse user preference profiles and high-quality CoT-style reasoning is contructed, enabling explicit supervision of structured reasoning. 

Experiments demonstrate the superiority of the proposed method.

### Weaknesses
My main concerns is how the proposed method generalizes to real-world images. To costruct a large-scale CoT-style dataset that provides high-quality reasoning supervision, the authors propose to combine several random profiles with initial prompts and feed into a text-to-image model to generate each user’s reference images (preferred and non-preferred) and two candidate images. However, the generated images ,as shown in paper and supplementary, lack photorealism, and would also cover a very limtied range of categories. Though the experiments demonstrate the superiority of the proposed method, the main experiments are conducted on the collected dataset, failing to validate the generalization to real-world images. 

On the other hand, it is still unclear to me whether the compared methods in paper are trained or fine-tuned on the collected dataset for fair comparison of assessment accuracy. Moreover, the evaluation on the PickaPic dataset is somehow confusion to me. As described in paper, the experiment on PickaPic reflects general preferences, rather than personalized preference. However, the results in Table 1 show that the proposed method ranks second on unseen PickaPic data for general preference assessment, this makes me doubt whether the proposed method can indeed extract and understand personalized visual preference profile. 

Currently, the employed visual preference profile consists of five visual elements while the user study is conducted with 15 visual elements. Therefore, I would like to see experimental analysis on how the number of visual elements in a visual preference profile affects the final results. 

In fact, it is difficult for a person to rate his/her visual preference to an image using an absolute score, but it is easy for a person to perform relative preference comparison between two images. Hence, I suggest the authors to evaluate the proposed method on such preference ranking data.

### Questions
What is the limitation of the proposed method?

How does the effectiveness of the proposed method on personalized image generation?

In my opinion, the current 5-element visual preference profile mainly summarizes the visual tone of an image, I wonder whether it is possible to make the model learn to find the visual elements that determines his/her visual preference?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces PreferThinker, a new framework for personalized image preference assessment.
The authors pointed out that currently there is a lack of user-specific data.
To overcome this issue, the paper proposes a common preference profile for bridging various users.
This profile enables the model to leverage large-scale data to learn user preferences.
The PreferThinker framework operates on a "predict-then-assess" paradigm: It first predicts a user's visual preference and non-preference profiles based on a small set of reference images. Then, it uses this predicted profile as a criterion to generate interpretable, multi-dimensional scores and a Chain-of-Thought (CoT) assessment for candidate images.

### Strengths
This paper tries to tackle the problem of personalized preference assessment by using the idea of common preference profile as a bridge between users. This idea is novel. Beyond this, the paper also introduces a new, large scale dataset for personalized assessment. The experiments are robust, covering seen vs. unseen profiles , single vs. multi-preference users , and robustness to the number of reference images.

### Weaknesses
The primary weakness is that the main dataset, PreferImg-CoT, is built on simulated user preferences. While the simulation pipeline is well-designed (based on a user study to find 5 key elements ), simulated profiles may not capture the full, complex, and sometimes contradictory or hard-to-articulate nature of real human preferences.

### Questions
Given the primary limitation is the simulated dataset, could you discuss the feasibility of collecting a (perhaps smaller) "gold standard" test set with real personalized data?

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
This work proposes PreferThinker, a system for personalized image reward scoring with chain of thought reasoning to predict a user's preference profile. To train this system, the authors first construct PreferImg-CoT, using Claude 3.7 to generate a reasoning trace of each individual user's score given reference images. The authors use this dataset to train Qwen-2.5-VL-7B with SFT and GRPO, and evaluate on their PreferImg dataset and Pick-a-Pic v1.

### Strengths
* This work proposes an interesting system to predict a user's preference profile for text-to-image generation, and then score generated images for that individual
* This work contributes a new large scale synthetic preference dataset PreferImg created from 80K synthetic user preference profiles with attributes that the authors choose after a real-world user study. This dataset can be a valuable resource for the text-to-image reward modeling community
* The authors demonstrate that their dataset can learn rewards that generalize well when new users have similar preference profiles to users seen during training (Tab 1) via cold start SFT and RL training of Qwen 2.5 VL

### Weaknesses
* The authors argue that *"although each user’s personalized preferences are unique, the key visual elements that shape these preferences are shared"* (L197-198), and they mention discrete attributes that users rank highly as important to them (art style, color, detail, art medium and saturation). I feel that this a strong assumption to make - what about individual preference differences that are more semantic in nature for a given prompt? Is it possible to discretize real-world user preferences of generated images in this manner?
* **My primary concern**: PreferThinker appears robust to unseen users when the distribution of their preference profiles is shared with seen users (I.e. they are both sampled from PreferImg). How robust is PreferThinker to preferences outside this distribution? Since PreferImg is constructed with 80K synthetic preference profiles, it is unclear to me how robust these preference profiles will be to real-world datasets. While the authors do evaluate on Pick-a-Pic, PreferThinker's accuracy (67%) does not seem compelling when compared to extremely lightweight pluralistic reward modeling baseline [1], which gets 71% accuracy on V1 and 70.5% on V2 (no-leakage) with just 6M trainable parameters 
* The presentation has room for improvement, especially the figures, which are quite small, dense, and difficult to read (especially Figure 8). Figure 1 and 2 can be made bigger and Fig 3 moved to the Appendix. It is very hard to parse Table 2 and 3 as their captions are nearly contiguous.

[1] Chen et al., "PAL: Sample-Efficient Personalized Reward Modeling for Pluralistic Alignment", ICLR 2025.

### Questions
* Which version of Pick-a-pic is used in your experiments, v1 or v2?

### Soundness
3

### Presentation
2

### Contribution
2
