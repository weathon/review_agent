# Synthesizing Physical Backdoor Datasets: An Automated Framework Leveraging Deep Generative Models

- Decision: Reject
- Scores: 3, 5, 5, 5

## Abstract
Backdoor attacks, representing an emerging threat to the integrity of deep neural networks, have garnered significant attention due to their ability to compromise deep learning systems clandestinely. 
While numerous backdoor attacks occur within the digital realm, their practical implementation in real-world prediction systems remains limited and vulnerable to disturbances in the physical world. 
Consequently, this limitation has given rise to the development of physical backdoor attacks, where trigger objects manifest as physical entities within the real world. 
However, creating the requisite dataset to train or evaluate a physical backdoor model is a daunting task, limiting the backdoor researchers and practitioners from studying such physical attack scenarios. This paper unleashes a framework that empowers backdoor researchers to effortlessly create a malicious, physical backdoor dataset based on advances in generative modeling. Particularly, this framework involves 3 automatic modules: suggesting the suitable physical triggers, generating the poisoned candidate samples (either by synthesizing new samples or editing existing clean samples), and finally refining for the most plausible ones. As such, it effectively mitigates the perceived complexity associated with creating a physical backdoor dataset, transforming it from a daunting task into an attainable objective. Extensive experiment results show that datasets created by our framework enable researchers to achieve an impressive attack success rate on real physical world data and exhibit similar properties compared to previous physical backdoor attack studies. This paper offers researchers a valuable toolkit for studies of physical backdoors, all within the confines of their laboratories.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper presents a framework for generating physical backdoor datasets using advances in generative modeling. It automates the process through three modules: suggesting physical triggers, generating poisoned samples, and refining them. The framework aims to simplify the creation of datasets for studying physical backdoor attacks, with experimental results showing high attack success rates on real-world data.

### Strengths
1. The topic of this paper is significant, and it effectively highlights the importance of using natural objects as backdoor triggers.
2. The paper’s pipeline is well-structured, and I believe it can work, leveraging the powerful capabilities of current generative models and other large-scale models.

### Weaknesses
1.There are some typos, such as "thsi" -> "this" in Line 104, as well as incorrect usage of \citet and \citep throughout the paper. Additionally, I couldn't quickly grasp the intended meaning of Fig. 2, as the explanation is unclear and lacks a detailed diagram.


2. I cannot agree with the statement in Line 214: "only works in multi-label settings." In fact, the key ideas from Wenger et al. (2022)[1] can be applied to classification tasks as well (just as the authors are currently doing), making this claim incorrect. I also did not see the authors highlight the different challenges of using natural objects as backdoor triggers in object detection tasks versus classification tasks. Furthermore, Zhang et al. (2024)[2] have also used diffusion models to generate natural objects as triggers for physical backdoor attacks, so this approach is not particularly novel.

3. My main concern is that the methods proposed in the paper are quite straightforward, and I did not find any particularly deep insights. Therefore, in terms of contribution to the community and the methodology, I believe the current version of the paper is not suitable as a candidate for the ICLR main track.


[1] Wenger, Emily, et al. "Finding naturally occurring physical backdoors in image datasets." NeurIPS 2022.

[2] Zhang, Hangtao, et al. "Detector collapse: Backdooring object detection to catastrophic overload or blindness." IJCAI 2024.

### Questions
The major challenge of using benign features as triggers is that clean training datasets might already contain these features (such as books), leading to conflicts between the trigger pattern and benign features, potentially hindering backdoor learning. I am very interested to know how the authors have approached and tried to resolve this issue.

Code implementation: I would like to see the code made open source

In conclusion, I think using natural objects as triggers is a good idea, but the challenges lie in addressing the potential conflict between benign features and the trigger, which could cause the backdoor training to fail (e.g., a significant drop in clean ACC).

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a framework to generate poisoned backdoor datasets, which consists of three components: 1) trigger suggestion, 2) trigger generation, and 3) poison selection. The motivation is to provide a more practical, generalized, and automated framework. The advantage is that this paper takes into account physical images captured from various devices.

### Strengths
- The built dataset contains physical data captured by various devices.
- The framework automatically chooses the most suitable trigger, which is usually not considered by previous works.
- It is easy to follow the presentation of the paper.

### Weaknesses
- Lower ASR even compared to clean label attacks, such as LC [1] and Narcissus [2]. The authors need to explain if there are any challenges behind the lower ASR.
- Evaluated by very old defenses. The attack in this paper should also consider recent defenses, such as BTI-DBF [3] and IBD-PSC [4].
- Only one dataset (5 classes is very small) and one small architecture. Considering larger datasets, such as a subset of ImageNet with 100 classes but fewer samples in each class. 
- According to Figure 1, the VQA model is a part of the "trigger suggestion" component, so it is not an individual contribution.
- The motivation is not clear to me. For example, in section 3, the authors mention the previous method only works in multi-label settings, but the experiments in this paper are also conducted on a multi-label (5-class) dataset. It looks like this paper does not solve the problem raised. It would be better if the authors could clarify how their framework addresses the limitations of previous methods.

[1] Label-Consistent Backdoor Attacks

[2] Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information

[3] Towards Reliable and Efficient Backdoor Trigger Inversion via Decoupling Benign Features.

[4] IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency

### Questions
- Are there any experiments conducted on physical devices? As far as I can see, the authors use the devices in Appendix B to only build the dataset. Is it possible to also apply the trained model (by the poisoned dataset) to a physical device for the classification task?

- As the paper offers a toolkit for backdoor studies, do the authors consider open-source their code?

- The authors aim to provide an "effortless" framework. Are there any results about time consumption?



typo: "thsi" (line 104)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes an automated framework for generating physical backdoor datasets using generative models to make physical backdoor attack more accesible. The framework has modules: 

Trigger Suggestion - Uses VQA models to suggest suitable physical triggers
Trigger Generation -  Create poisoned samples by generating or editing
Poison Selection - Use ImageReward to select the best poisoned images

### Strengths
Overall, this is good work that makes physical backdoor attack research more accessible by providing a framework to generate datasets, which is usually the most tedious part.

The three-phase design makes good sense, and the results are considered comprehensive, as many aspects have been discussed, such as the common accuracy on clean inputs, attack success rate, as well as resilience, saliency heatmap, and dataset entropy.

### Weaknesses
My major concern is that the design of the framework appears to be a straightforward combination of a few existing solutions, i.e., a pretrained VQA model for trigger suggestion, stable diffusion or instruct diffusion for trigger generation/editing, and ImageReward for final poisoned data selection. Additionally, there is limited to no further customization or modification of these existing works to make them more integrated or collaborative. Therefore, the innovative contribution of this paper is very limited.

It also seems that we have limited control over the framework in terms of poisoned data generation. For some complicated datasets, it may be difficult to precisely control the size, type, or position of the trigger. This functionality is critical for certain tasks, given the diverse settings in the physical world. It would be beneficial to discuss this aspect in the paper and potentially incorporate it into the framework design.

As a physical backdoor dataset generator, it is more important to compare the quality of the generated images to real images, poisoned images from other related works (semantic trigger backdoor attacks have been around for quite some time), and edited images using traditional methods such as Photoshop. MORE evidence and examples need to be provided, especially in the appendix.

### Questions
Could you carefully justify the technical contribution of this framework?
Please discuss the customizability of the framework as pointed out in the weaknesses of the paper.
Please add more examples and comparisons of the generated poisoned images.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper presents a framework that can synthesize physical backdoor datasets. The framework consists of three modules: a trigger suggestion module that recommends suitable physical objects as triggers, a trigger generation module that creates or edits images to contain these triggers using advanced generative models, and a poison selection module that filters for the most natural-looking results. The paper demonstrates that the framework can produce datasets that achieve high attack success rates in real-world scenarios while maintaining similar properties to manually collected physical backdoor attack datasets.

### Strengths
- Creating physical backdoor datasets is an interesting topic and can make contributions to related work.
- Detailed discussion on improvements over existing techniques.

### Weaknesses
- The paper does not explain how the models in each module are trained. Also, the inputs and outputs of each step are not clear. As I understand it, in step 1 (trigger selection), the final output is a trigger. The model in step 2 then tries to attach the trigger to an image to generate the Trojan dataset. However, it seems that triggers need to be specified when training the model. So, how can this model be generalized to different triggers? Also, it is not clear what the training data is.

- From the experimental results so far, it is difficult to evaluate the realism of the generated dataset. It might be helpful to provide some generated examples.

- When specifying triggers, let's say “a car”. trigger generation may generate different cars based on its own understanding. Discussing how to ensure consistency of triggers and minimize the impact on relevant benign samples could be helpful.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
