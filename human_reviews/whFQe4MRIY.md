# MI-NeRF: Learning a Single Face NeRF from Multiple Identities

- Decision: Reject
- Scores: 8, 6, 6, 6

## Abstract
In this work, we introduce a method that learns a single dynamic neural radiance field (NeRF) from monocular talking face videos of multiple identities. NeRFs have shown remarkable results in modeling the 4D dynamics and appearance of human faces. However, they require expensive per-identity optimization. To address this challenge, we introduce MI-NeRF (multi-identity NeRF), a single unified network that models complex non-rigid facial motion for multiple identities, using only monocular videos of arbitrary length. The core premise in our method is to learn the non-linear interactions between identity and non-identity specific information with a multiplicative module. By training MI-NeRF on multiple videos simultaneously, we significantly reduce the total training time, compared to standard single-identity NeRFs. Our model can be further personalized for a target identity. We demonstrate results for both facial expression transfer and talking face video synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces MI-NeRF, a novel approach to learn a single dynamic neural radiance field (NeRF) from monocular videos of talking faces across multiple identities. The core innovation is the incorporation of a multiplicative module that distinguishes between identity and expression information, inspired by TensorFaces. By simultaneously training on multiple videos, MI-NeRF significantly cuts training time while achieving state-of-the-art performance in tasks like facial expression transfer and talking face video synthesis.

### Strengths
1. The proposed MI-NeRF robustly adapts to unseen expressions and identities, requiring minimal retraining, thereby saving computational time and effort.
2. The Multiplicative module in MI-NeRF effectively separates identity from expression, ensuring consistent identity portrayal even with dynamic expressions, a challenge for some other models.
3. MI-NeRF achieves superior performance in visual quality and lip synchronization, making it a leading solution in realistic talking face synthesis.

### Weaknesses
Here are some concerns:

1. The paper utilizes learnable latent codes to capture time-varying information. However, there's a potential ambiguity regarding how the model ensures these codes don't inadvertently encode identity or expression information. A more rigorous analysis or mechanism would have been beneficial to validate that these latent codes truly only represent unique, time-varying elements without overlapping with identity and expression descriptors.

2. Maintaining consistency in novel views is inherently tied to the accurate modeling of 3D information. When MI-NeRF constructs facial identities and expressions over time, it is imperative for it to ensure that these reconstructions remain consistent and accurate, even when viewed from unseen angles. This is not merely a question of aesthetics, but of the system's fidelity to real-world dynamics. Without evaluation or visualization of its ability to maintain this consistency, one could question its versatility and applicability in varied real-world settings.

3. Some concerns regarding the dataset:
i) The collected videos lead to concerns about data privacy and the feasibility of public release. Ethical considerations surrounding personal data must be paramount. And will the authors be able or allowed to release the data?
ii) Within 140 videos, It's unclear how many unique individuals are represented and whether multiple videos pertain to the same person. Additionally, questions arise regarding the consistency of identity codes across different videos during training.
iii) The steps taken to prepare the data before its integration into MI-NeRF are essential. Without clear details about preprocessing, it's challenging to replicate results or trust the dataset's integrity.

4. While the research highlights the limitations of the GAN-based method, Wav2Lip, it doesn't delve deeply into how MI-NeRF stands compared to other recent state-of-the-art GAN approaches. 

5. MI-NeRF's uniqueness largely stems from its multiplicative module. However, this also means the model's overall success heavily depends on this single component. If the module faces challenges in complex real-world scenarios, the entire model's performance could be compromised. A deeper discussion about the potential limitations of this module would be beneficial for a comprehensive understanding of its robustness.

### Questions
1. Could the authors shed light on the exact nature of the per-frame latent codes and how they ensure these codes are devoid of identity or expression specifics? 

2. Could there be a more detailed comparison between MI-NeRF and advanced GAN-based methods, particularly in the context of face video synthesis?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, authors propose a NeRF based method for dynamic face synthesis from monocular talking face videos. Unlike single-identity models, the proposed method uses a single network to train on videos of multiple identities and do a fast fine tuning when applying on a new identity. This single network is trained by imposing a multiplicative structure between identity embeddings and expression embeddings.  Quantitative and qualitative experiments show that the proposed multiplicative structure helps disentangling the identity and expression. The training time is significantly reduced compared to single-identity models.

### Strengths
The proposed method is simple and effective. 

Built on top of earlier multi-identity NeRF methods, this work basically imposes a multiplicative structure between identity embeddings and expression embeddings. The structure is in the form of an element-wise product between two embeddings and can be extended with high-degree interactions. The method shouldn't be hard for any readers to re-implement.  

Ablation studies show the multiplicative structure is well designed, and each piece of it can help improve the factor disentanglement and visual quality. When compared with existing works, the proposed method shows superior performance for expression transfer and lip synced video synthesis tasks.

### Weaknesses
The comparison with other multi-identity NeRF-based methods can be improved. 

Fast training time is a known benefit for multi-identity NeRF-based methods. For the training time evaluation, it misses the comparison with other multi-identity methods, like HeadNeRF.

More multi-identity NeRF-based methods might be included in the quantitative comparison, especially in the expression transfer experiments.

### Questions
Is there a training time benchmarking (Figure 3) for identities methods, like HeadNeRF?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a method to learn a single dynamic NeRF for talking face videos of multiple identities. Expression, identity, and time-varying parts are separately modeled and interact with each other to predict the color and density of NeRF. Therefore, this method can use only a single NeRF to model the common geometry for diverse faces.

### Strengths
1. The whole writing is well organized and the proposed method is clearly described;
2. The method for disentangling the identity and expression sounds reasonable, and the experiments show good disentanglement;
3. Shown results outperform the competitors and the training time is significantly reduced.

### Weaknesses
The paper further proposes a high-degree interaction module; however, I don't see any usage of this module in the whole method, as well as any experiment analysis about this module.

### Questions
Please refer to the weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work is about developing a single NeRF for multiple face identities. It uses the 3D morphable face model to estimte the 3D first, and then applys to NeRF for building the model. Experiments are shown both quantitatively and quanlitatively.

### Strengths
It argues that a single face NeRF model can be developed for multiple identities.

Some interesting results are obtained.

### Weaknesses
It is unclear how the face expressions are aligned or handled. Even different identities can be handled together, how to deal with the different expressions for different identities? Do you use a cononical face model to separate the expressions? 

It is unclear how to handle diferent lighting conditions. 

There is no quantitative comparisons between the proposed method and the state of the art. It is unclear if the proposed method can outperform the existing works.

### Questions
As listed in the weakness part. 

Further, the authors should give clearer statements on how to deal with different identities for the NeRF model learning. Although some equations are given, it is still difficult how to do this. It needs more detailed descriptions on this point.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
