# Detecting Deepfakes Without Seeing Any

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 5, 3

## Abstract
Deepfake attacks, malicious manipulation of media containing people, are a serious concern for society. Conventional deepfake detection methods train supervised classifiers to distinguish real media from previously encountered deepfakes. Such techniques can only detect deepfakes similar to those previously seen, but not zero-day (previously unseen) attack types. As current deepfake generation techniques are changing at a breathtaking pace, new attack types are proposed frequently, making this a major issue. Our main observations are that: i) in many effective deepfake attacks, the fake media must be accompanied by false facts i.e. claims about the identity, speech, motion, or appearance of the person. For instance, when impersonating Obama, the attacker explicitly or implicitly claims that the fake media show Obama; ii) current generative techniques cannot perfectly synthesize the false facts claimed by the attacker. We therefore introduce the concept of “fact checking”, adapted from fake news detection, for detecting zero-day deepfake attacks. Fact checking verifies that the claimed facts (e.g. identity is Obama), agree with the observed media (e.g. is the face really Obama’s?), and thus can differentiate between real and fake media. Consequently, we introduce FACTOR, a practical recipe for deepfake fact checking and demonstrate its power in critical attack settings: face swapping and audio-visual synthesis. Although it is training-free, relies exclusively on off-the-shelf features, is very easy to implement, and does not see any deepfakes, it achieves better than state-of-the-art accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents a novel approach to detecting deepfakes that does not rely on direct visual inspection of the manipulated media. Instead, the authors employ fact-checking techniques to discern between authentic and manipulated content. They introduce a novel metric termed the "truth score," quantifying the probability that a provided caption accurately describes the corresponding image. Subsequently, this metric is utilized to compare truth scores between genuine and manipulated images, demonstrating that the latter tend to exhibit lower truth scores. Furthermore, the authors introduce a new methodology for deepfake verification, denoted as FACTOR, which integrates the truth score with additional attributes such as image quality and metadata, thereby enhancing detection precision. In summary, this paper's contributions encompass a robust approach to deepfake detection, resilient against zero-day attacks, the introduction of a metric for assessing caption veracity, and a comprehensive methodology for deepfake verification, amalgamating multiple features to enhance detection accuracy.

### Strengths
The methodology presented in this paper is ” simple but effective”. The authors approach the problem from the perspective of fact-checking, offering a training-free method, which is particularly intriguing.

The paper is well-structured, providing clear explanations of the methods employed and the corresponding results. The authors offer detailed accounts of their experiments, substantiating the effectiveness of their approach with various types of data.

The introduction of the novel deepfake detection method, FACTOR, showcases its heightened robustness against zero-day attacks compared to existing methodologies.

Overall, this paper's contributions have the potential to elevate the technological prowess of deepfake detection, holding broader implications for media forensics and fact verification.

### Weaknesses
While the authors have conducted a substantial number of experiments, there are instances where certain experiments appear to oversimplify the problem. For instance, in the deepfake detection from text-to-image, the use of diffusion models to generate data may not fully capture the complexities present in real-world scenarios of deep deception.

The authors have separately validated the deep deception in three distinct aspects. However, there are occasions where these aspects may not be entirely independent. In such cases, it raises questions about whether FACTOR can still exert a positive influence.

It is hoped that the authors will consider incorporating the ideas presented in this paper into supervised methods in future work. This would be a particularly intriguing avenue to explore.

### Questions
Addressing zero-day attacks with supervised methods is currently considered challenging. How does the proposed approach in this paper contribute to changing this situation? Can supervised methods leverage the concept of FACTOR in any way to mitigate the challenges posed by zero-day attacks?

The currently validated data appears to consist of features that are relatively prominent. When faced with a large-scale dataset of natural content, quantifying the true impact of FACTOR becomes considerably challenging. Although this may present a difficulty for the authors, if supplemental experiments were conducted in this area, it would greatly enhance the comprehensiveness and excellence of this paper.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a new type of DeepFake detection task. Instead of classifying a piece of data (e.g., a fake image, fake video) as fake, the new goal is to make the real/fake prediction based on how well that piece of data aligns with the additional fact claimed by the attacker. So, a deepfake Obama face is fake if the contents of the image do not match the identity being claimed (Obama). The authors present three variants of this task - (i) Face swapping detection, where the additional fact is the identity being claimed, (ii) Audio-visual deepfake detection; additional fact = audio/video are matching, (iii) detecting fake images from text-to-image models; additional fact = the prompt is corresponding to the image. For these tasks, the authors present results on the relevant datasets, comparing to baselines (which are solving a different task), and report improvements over them using appropriate metrics.

### Strengths
1. The authors have tackled a new form of deepfake detection. While I do have some concerns about the practical nature of that setting (see weaknesses), it might still be worthwhile to think about some different ways of thinking about - what makes a data point fake.

2. The authors have made an effective use of some of the large pretrained models to solve the detection task. Their main point about potential usage of these feature spaces without needing to finetune them on any supervised dataset is practically helpful.

3. The above point will be especially useful when thinking about general purpose fake image/video detectors, where we do not want to keep on training new models for every manipulation method.

4. Their proposed method is simple, and seems effective enough against other baselines (although they are solving a slightly different task; see weaknesses/questions).

### Weaknesses
1. My major concern is with the new form of problem statement for detecting deepfakes for two out of the three settings presented in the paper: (i) detecting face-swapped images (Section 5), and (ii) detecting images generated by text-to-image diffusion models. For (i), when we see DeepFakes in the wild, the attacker does not explicitly give us a well structured label that we can use as identity. Even if we somehow do, it is not clear how we will use the label to collect the reference set. My point is not that it cannot be done (the authors have described a hypothetical scenario in which it can be done), but is rather that it does not seem that practical. Similarly for the (ii) case; in the wild (e.g., internet), we will simply encounter an image and likely not encounter any text associated with it. What the authors are trying to study - how good image-text alignment is - is a different, and I believe, a separate problem. 

2. In Section 7, the final score being the difference of two “truth-scores” (BLIP - CLIP) does not seem principled. It is not clear why someone would arrive at this particular way of computing the scores instead of, for example, the average of the two scores. Right now, it simply seems that one way (difference of scores) can distinguish the real image-text pairs from fake ones and hence the authors have used them. However, that does not tell us how generalizable the metric is. For example, will the same metric work in detecting fake images from other open-source text-to-image models? 

3. In sections 5, 6, it seems that the feature extractor that one uses might influence how well the method works. Currently, there is no discussion on the effect of different feature extractors. For example, what would happed if you used a different pre-trained network for phi_id (Section 5). What properties of those pretrained networks are important; is the size of pretraining dataset important? Does the pretraining dataset have to be of very similar domain of images? etc. In summary, the feature space used should be studied more thoroughly.

4. Overall, I believe that the task that the authors are trying to solve is not the same task that the baselines are trying to solve. The baselines are trying to solve, what I would call, instance-level deepfake detection; where you have to make a prediction whether that instance (e.g., a deepfake face) is real or not "regardless of the associated fact". The problem being tackled by the authors, as mentioned before, is different. So, while I am not saying that authors' method has any inherent advantage. But it does not seem to be an apples to apples comparison.

### Questions
Questions:

1. Figure 1; is that the best that current generative models can do? While I cannot pinpoint the exact results, given what progress has taken place in deepfakes, I would have expected a much better Obama face wearing Trump's suit.

2. For the Audio-Visual DeepFake detection, what is the train/test split? 


Comments:

1. For all the tables, the authors should report the mean performance across different datasets; for all the methods (baselines + their own method).

2. In the discussion of Table 3, the authors say that their method sometimes outperforms the supervised baselines even though they do not require labeled data is a bit misleading. Even though they do not need supervised data to train their model, they do need labeled data in the form of a reference set to do test time inference. 

3. I believe the authors should change their title to account for the modified nature of deepfake detection.

### Soundness
2 fair

### Presentation
3 good

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
This work proposes to address deepfake detections by focusing on checking the claimed facts (e.g. identify, motion).

For a potentially faked sample with a claimed fact, they use a pre-trained feature extractor to embed it and compare the embedding to the embedding of a real sample with the same claimed fact. If the cosine similarity between them are high, they predict the sample real; Otherwise they predict the sample as deepfake.

They evaluate this for the cases of detecting face wwapping, Audio-Visual Deepfake and deepfake from text-to-image models.

### Strengths
1. Detecting Deepfakes is a hot topic.
2. The proposed method has a very simple formulation.

### Weaknesses
**1. Falsely claimed robustness against zero-day attacks.** 
The arguments 'the proposed framework is robust against zero-day attacks because it uses pre-trained feature extractor without tuning on deepfake samples' is not justified. It is essentially claiming the following impossibility for generative models: Given a feature extractor, one cannot train a generative model to make the embedding of the generated samples close to a target (real) sample. This is a very strong and likely wrong claim as least for existing approaches.

**2. The limitation/effect of the reference sample (i.e. the real sample with the claimed fact) is not properly investigated and discussed.**
How can one retrieve such a reference sample in practice? How will distribution shifts (e.g. different angles, different cameras, different places) affect the effectiveness of the propose detection method? In most if not all existing experiments, the reference samples come from the same dataset as the real samples, which renders the evaluation less practical. One way to have some preliminary results on these is to apply some augmentations (e.g. rotations, blurring, noises, cropping) to only the reference samples, and see if the performance remains.

**3. The requirement to the feature extractor is over-simplified.** The assumption is basically requiring the pre-trained feature extractor to naturally distinguish real samples from fake ones. This is also a rather strong assumption (it is in fact related to Weakness 1). Please provide some justifications. It is worth noting that at least for now, fooling a pre-trained feature extractor should be considered relatively easier than 'encode the false fact into fake media with sufficient accuracy' given that 1) adversarial robustness remains an open challenge and 2) the pre-trained feature extractor has never seen deepfake samples according to your assumption, which means it may not need to detect 'tiny flaws' when it is trained.

**4. Missing important details in some experiments.** For example: What are method A and method B in section 5? How do you decide the threshold for predicting real/fake? What are the thresholds?

### Questions
Please refer to Weakness section above.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair
