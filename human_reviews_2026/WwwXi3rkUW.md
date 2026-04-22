# Co-Evolving Latent Action World Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Adapting pre-trained video generation models into controllable world models via *latent actions* is a promising step towards creating generalist world models. The dominant paradigm adopts a two-stage approach that trains the latent action model (LAM) and the world model separately, resulting in redundant training and limiting their potential for co-adaptation. A conceptually simple and appealing idea is to directly replace the forward dynamic model in LAM with a powerful world model and train them jointly, but this is non-trivial and prone to representational collapse. In this work, we propose **CoLA-World**, which for the first time successfully realizes this synergistic paradigm, resolving the core challenge in joint learning through a critical warm-up phase that effectively aligns the representations of the from-scratch LAM with the pre-trained world model. This unlocks a co-evolution cycle: the world model acts as a knowledgeable tutor, providing gradients to shape a high-quality LAM, while the LAM offers a more precise and adaptable control interface to the world model. Empirically, CoLA-World matches or outperforms prior two-stage methods in both video simulation quality and downstream visual planning, establishing a robust and efficient new paradigm for the field.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a joint training framework to learn a world model based on latent actions, including a warm-up phase where only the latent action model is optimized, and a joint-training phase. Experiments are conducted on several benchmarks to demonstrate the efficacy of the proposed framework.

### Strengths
- The paper is well-written. The proposed method is simple to follow, and I think it is easy to reproduce.
- The proposed method is sound. Given a well-pretrained world model, it is promising to learn both the latent action model and the world model jointly.
- The idea of adapting pre-trained video generative models to world models is promising and is an interesting topic for the community.
- Experiments include both seen and unseen datasets, which demonstrate the efficacy of the proposed method.

### Weaknesses
- The advantages of the proposed method are not obvious: **1) efficiency**: Can the proposed joint-training method save computation or accelerate training speed compared with the baseline? As the paper finetunes a large-scale pre-trained world model, the comparison of the required computation budget should be demonstrated clearly. **2) Applicability**: The proposed method requires a well-initialized, pre-trained world model for stable training. While the paper only studies Open-Sora, it is natural to ask, can this method work well for other models? If the pre-trained model is specifically worse than Open-Sora, it is challenging for the proposed method to perform well. **3) Scalability**: The baseline is scalable, which has been proven by Genie. Is the proposed method scalable with increased dataset size or model parameters?
- It is better to include qualitative results like video demos to show the performance improvements of the method. Moreover, I suggest including the video prediction performance at the warm-up stage and later joint-training stage for presenting the learning dynamics.
- Do the authors have any ideas for the data size used in the warm-up phase? Are there any monitoring metrics for better designing the warm-up phase to generalize to different training tasks and pre-trained models?

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a joint learning approach for the latent action model and latent action world model. It leverages a critical warm-up phase to avoid collapse. Such joint learning makes the world model stronger controllable and sample efficient.

### Strengths
* The problem that this paper would like to address is clear. Separately learning the latent action model and world model will hinder their in-depth mutual improvement.
* This paper conducts in-depth analysis of why direct joint learning is relatively difficult.
* The presentation is good for easy understanding.

### Weaknesses
* Lack of visualization comparison between the 2-stage and joint. The visualization results can better help to understand the benefits of joint learning.
* There is no significant improvement compared to the 2-stage for the performance of latent action or world models. It's very hard to see the benefit of joint learning
* Training budget is not the perfect metric to evaluate efficiency. The FDM model is much smaller than the world models.

### Questions
* What about the performance of LAM30K+WM52K?

### Soundness
1

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
4

### Summary
This paper proposes a latent action world model learning approach that jointly updates the latent action model (LAM) and the world model. The approach includes a warm-up phase designed to align the representations of the LAM and a pre-trained world model, after which both models are updated synergistically. This framework mitigates the representational collapse problem during joint learning of the LAM and world model, making it effective. Experimental results demonstrate that this joint learning approach produces a better latent action space, superior video simulation quality, and improved visual planning performance compared to a two-stage approach.

### Strengths
1.	The representational collapse problem addressed in this paper is well-motivated and thoroughly analyzed. The use of metrics such as utilization, maximum usage, and entropy to assess the quality of the latent space is intuitive and effective.
2.	The method proposed to address the representational collapse problem is simple, intuitive, and effective.

### Weaknesses
1.	The improvement in metrics such as linear probing loss and video prediction performance is marginal. It is difficult to accept that joint learning is definitively better.
2.	As shown in Fig. 2, the joint learning of a randomly initialized inverse dynamics model (IDM) and world model (WM) ultimately approaches the performance of a pre-trained IDM. This suggests that joint training from scratch may be a viable approach if a pre-trained video prediction model is unavailable, which contradicts the authors' claim of being the first successful joint training framework.
3.	More comparisons of video simulation quality and visual planning performance with other paradigms, such as joint training from scratch (PreLAR) or iterative learning (AD3), should be included and discussed.

### Questions
1.	In Fig. 2 and Fig. 3, the best metrics (such as utilization, maximum usage, and entropy) of the latent action space may be influenced by the distribution of actions in the training data. It would be beneficial to use normalized metrics or provide statistical results to demonstrate that the data is balanced.
2.	The authors use OpenSora as a pre-trained world model. Would using other types or architectures of pre-trained world models yield different results?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper aims to improve the pretraining of the generalist action-conditioned world models. To achieve that, it proposes to replace the forward dynamics model in previous literature with the world model, and jointly pretrain the latent action model and the world model. To avoid representation collapse, the paper further designs a warm-up stage. The paper conducts downstream applications in multiple decision making scenarios in terms of video prediction quality and visual planning performance.

### Strengths
S1) The paper presents a very detailed and inspiring analysis of different integration methods of latent action model, which I believe will be highly instructive to the readers.

S2) The proposed method is well-motivated, simple, and effective.

S3) The proposed method is validated on multiple important settings, demonstrating its superiority.

### Weaknesses
W1) The paper is not the first practice to replace FDM in the latent action model with a generative world model. UniSkill [1] has already explored this setting, which is missing in the discussion of the paper.

W2) My major concern is the training efficiency of the proposed method. Previous two-stage methods can enjoy a high training throughput by using a light-weighted FDM. However, this paper uses OpenSora, which is more computational heavy. It is important to consider the training efficiency in comparison.

W3) Another concern is about the expressiveness of the jointly learned latent actions. As shown in Figure 2, "IDM (rand) + WM (rand)" has better codebook usage compared with "IDM (pre) + WM (pre)". In my opinion, "IDM (rand) + WM (rand)" is similar with the case of the two-stage method, which trains IDM and FDM from scratch, while "IDM (pre) + WM (pre)" is the proposed method. Hence, does that indicate that prior method learns a latent action codebook with larger capcacity? As the pretrained WM exists some prior knowledge, it may undermine the learning of an expressive codebook. On the contrary, the two-stage method could motivate the learning of codebook to represent more action information since the FDM has limited prediction capability.

---

[1] UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations

### Questions
Q1) In AdaWorld, it is demonstrated that a continuous latent action space can perform better than discrete latent actions in application. Why this paper doesn't consider to use continuous latent actions?

### Soundness
3

### Presentation
4

### Contribution
3
