# SMAFace: Sample Mining Guided Adaptive Loss for Face Recognition

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
Traditional face recognition (FR) algorithms often rely merely on margin-based softmax loss functions. However, due to noisy training data and varied image quality in datasets, these models often falter when dealing with low-quality images. To address this issue, we introduce SMAFace, an innovative FR algorithm that enhances performance by incorporating sample mining into conventional margin-based methods. At its core, SMAFace focuses on prioritizing information-dense samples, namely hard samples or easy samples, which present more distinctive features. In this study, we employ a probability-driven mining strategy, enabling the model to adeptly navigate hard samples, thereby bolstering its robustness and adaptability. The mathematical evaluation and empirical tests of SMAFace indicate its effectiveness. Moreover, experimental results reveal that our approach surpasses the state-of-the-art (SoTA) on four renowned datasets (CPLFW, VGG2-FP, IJB-B and TinyFace), highlighting its potential and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, the authors propose their method, namely SMAFace, to leverage the technique of sample mining and margin-softmax for learning face recognition network. In the experiments conducted by the authors, the proposed method shows certain advantage in the results.

### Strengths
The paper is straightforward and easy to follow.

### Weaknesses
-	The motivation of this work is ambiguous. In Abstract, the authors introduce the challenge of noisy training data, but the proposed method is developed for improving model training on hard samples, rather than noisy sample learning.
-	The definition of ``information-dense’’ sample is confusing. 
-	Why easy sample belongs to so-called ``information-dense’’ along with hard sample? Why easy sample present distinct feature as hard sample does?
-	The definition of $\gamma$ is not given. Also does the ``GST’’.
-	The index of the equation between Eqn3 and Eqn4 is not given. (Let us note it as Eqn3.5) Eqn3.5 is also confusing, including two similar sub equations. The condition is not given.
-	The color notion (red and blue) in Tab2 is not defined.

### Questions
Please refer to the weakness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper incorporates sample mining into margin-based methods to improve performance in dealing with low-quality images. It prioritizes information-dense samples and employs a probability-driven mining strategy to enhance robustness and adaptability. Experimental results show that SMAFace outperforms state-of-the-art methods on some datasets.

### Strengths
SMAFace introduces an adaptive training method for hard-negative mining. Compared to the counterpart without this module, it improves the accuracy of face recognition.

### Weaknesses
1.The quality of writing and layout of the paper need improvement.
2.By integrating marginal softmax, there are too many hyperparameters. It seems more like a parameter-tuning technique, lacking novelty.
3.The results show only a slight improvement, without demonstrating significant effects.

### Questions
None

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed a new SMAFace face recognition algorithm that adaptively integrates sample mining into the margin-based loss function, which is claimed to handle low-quality face images well. The paper also proposed a so called scaling term to analyze face recognition method. The proposed method was evaluated 4 public datasets and compared favorably with baseline methods.

### Strengths
The paper presented an in-depth discussion about adaptive loss function for low-quality face images. Then the paper proposed to integrate hard sample mining to dynamically adjust weight coefficient based on the probability of the correct class, which appears an effective way to further improve face recognition performance for low-quality face images.

### Weaknesses
First of all, the paper shall be self-contained within 9 pages with/without the Appendix. However, this paper appears to take the Appendixes as indispensable components, with many references to the Appendixes in the main text.  If the main text is the just the outline and proof and detailed explanation are in the Appendixes, this may violate the 9-page limit in some sense. 

For example, if the scaling term is the 2nd contribution of the paper, why only explains its details in Appendix?

“Regarding the adaptive margin function, we define it as” … “For a detailed explanation, please refer to Appendix A”, why not explain the key contribution of adaptive margin function in the main text?

The paper is not well-written and hard to follow. To list a few examples: “which as been presented in the Appendix” in the 4th paragraph in the introduction; “when a sample is more hard”; “Combining it with fields beyond FR promises encouraging results”.

Important reference missing:
DeepFace: closing the gap to human-level performance in face verification, CVPR 2014. 
Deep learning face representation by joint identification-verification, NIPS 2014

### Questions
Other than the loss function, SMAFace pretty much followed Deng et al (2019a), e.g., using the backbone ResNet, what the performance would be if using the new ViT network?

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair
