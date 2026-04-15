# Warfare: Breaking the Watermark Protection of AI-Generated Content

- Decision: Reject
- Scores: 6, 3, 6, 6

## Abstract
AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content).  
A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to the inference process of existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces WARFARE, a strategy to remove or forge invisible image watermarks, by introducing a generative adversarial network based on a user-specific watermarked dataset and its regeneration-attacked counterpart. The method has demonstrated its capability to successfully remove and forge "StegaStamp" watermarks, utilizing the pre-trained diffusion model "DiffPure" without any need for additional fine-tuning. Moreover, the paper highlights that WARFARE performs well even with limited fine-tuning data which indicates strong few-shot generalization ability, achieving significant time efficiency—boasting a speedup of 5,050 to 11,000 times compared to conventional regeneration attacks.

### Strengths
This paper is well-written and easy to read. The forging attack threat model is interesting and the solution is novel. The experiment result is clear.

### Weaknesses
While I appreciate the novelty in the proposed forging component of this paper, I find the overall contribution to be somewhat incremental. My main concern centers around the assumption regarding the mediator dataset. Specifically, the authors assume that "the mediator dataset can be considered as being drawn from the same 'non-watermarked' distribution, which is distinct from the 'watermarked' distribution." This is a critical assumption underlying why the proposed method is effective. However, whether this assumption holds entirely depends on the interaction between the watermarking technique and the methodology employed to generate the 'non-watermarked' distribution (i.e., the regeneration attack). In my view, this approach inherently relies on the success of a preceding watermark removal attack. The author could benefit from discussing the potential limitations or edge cases where this assumption may not hold.

My reasons for questioning the contribution of this paper are twofold:

Technical Dependency: The success of the proposed attack appears to hinge on the success of a prior watermark removal attack. This reliance raises concerns about the robustness and independence of the proposed methodology. 

Impact on the Community: As a peer in the watermarking community, I find it difficult to see the positive social impact of this work. Previous research on regeneration attacks contributed by highlighting ways to remove watermarks, thereby motivating the community to enhance watermarking algorithms to withstand such attacks. In contrast, this paper builds directly upon existing attack methodologies, and it is unclear how this approach benefits the community in terms of advancing robustness or offering constructive insights for defense mechanisms. Could you give us some examples of how this work might indirectly contribute to improving watermarking techniques or to address potential misuse concerns?

### Questions
In section 6, you chose to use two-bit strings for one user, and the resultant bit error rate differs by around 7%. And from your Table 8, it seems that you use unique and fixed bit strings for all experiments. I wonder, for your results in table 1-6, if you test on different bit strings, what would the standard deviation look like? 

Some minor questions:
1. In section 3.1, you mentioned 'We only consider the steganography approach, as it is much more robust and harder to attack than the signal transformation approach.' Can you elaborate on what you mean by the steganographic approach, and what is the difference between the signal transformation approach?
2. On the Table 3 caption it mentioned that the second best results are with underline, however for table 4, the second best result for bit ACC in the "watermark remove" section seems to be DM_l to me.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a method to break watermark-based detection of AI-generated content, in particular AI-generated images. The idea is to collect some watermarked images, denoising them as mediators, and then train a GAN to map images to remove and forge watermarks. Evaluation is performed on two simple datasets CIFAR10 and CelebA.

### Strengths
AI-generated content detection is a timely topic. 

Watermark for AI-generated content is an important topic.

### Weaknesses
The proposed method is not intuitively sound. The method uses watermarked images and mediators (denoised versions of watermarked images) to train GAN. This does not sound a correct approach. The mediators are different from clean images, even the distributions are likely different. 

Evaluation is performed on CIFAR10 and CelebA datasets. These are toy datasets, and shouldn't be used any more in the era of generative AI. Or we cannot draw any conclusion based on such toy datasets. 

The proposed method relies on training GAN, which is challenging for large datasets. 

Evaluation only considers simple baselines. There are some recent works on breaking watermarks in the context of AI-generated content detection, especially image watermarks. The paper needs to perform a more comprehensive literature review and compare with multiple recent works. 

DiffPure si a weird method to remove adversarial perturbations. Given an image, if you add enough noise and then denoise it, the denoised image may be completely different from the original image, and of course it does not have adversarial perturbation any more, but the image has changed.

### Questions
Please see the above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper discusses the vulnerabilities of watermarking techniques used to protect AI-generated content (AIGC). It introduces "Warfare," a methodology that can effectively remove or forge watermarks in AIGC using a pre-trained diffusion model and a generative adversarial network (GAN). The study highlights that existing watermarking schemes are fragile and can be easily bypassed, posing significant risks to intellectual property and content regulation. The authors emphasize the need for more robust watermarking methods to ensure the security and integrity of AI-generated content.

### Strengths
1. **Innovative Approach**: The paper introduces **Warfare**, a novel methodology that effectively combines a pre-trained diffusion model and a generative adversarial network (GAN) to remove or forge watermarks in AI-generated content. This unified approach is both innovative and practical.
2. **Comprehensive Evaluation**: The authors conduct **extensive experiments** on various datasets (e.g., CIFAR-10, CelebA) and settings (e.g., different watermark lengths, few-shot learning). The results demonstrate the high success rates and efficiency of Warfare, providing strong evidence of its effectiveness.
3. **Practical Relevance**: The study addresses a **real-world threat** by showing how existing watermarking schemes can be easily compromised. This highlights the need for more robust watermarking methods and has significant implications for the protection and regulation of AI-generated content.

### Weaknesses
1. **No Connection to AIGC**: Although the paper emphasizes AIGC (AI-Generated Content) in the title and introduction, the actual work appears unrelated to AIGC. Instead, it focuses solely on watermarking images and proposes two specific attacks in this context. While the authors mention SD1.5, this does not establish a meaningful link to AIGC. I recommend that the authors revise their claims about AIGC to reflect this focus.

2. **Training Success Risks**: The proposed method relies on a GAN model, yet training GANs successfully is known to be challenging. This difficulty poses a risk of failure in achieving the desired outcomes. It would be beneficial for the authors to delve deeper into addressing these potential risks.

3. **Efficiency Concerns**: While the authors assert that the proposed Warfare method is 5,050 to 11,000 times faster than the inference process of current diffusion model-based attacks, training a GAN is non-trivial and can be very time-consuming. For a comprehensive efficiency comparison, it would be valuable to account for the time required to train the GAN model.

### Questions
1. Given the challenges in training GAN models, what specific measures or adjustments do the authors recommend to mitigate the risks of unsuccessful training?
2. How does the training time for the GAN model impact the overall efficiency of the proposed method, especially when compared to diffusion model-based attacks?
3. Have the authors considered including the GAN training time in their efficiency comparison? If not, what would be the estimated impact of this addition on the reported performance gains?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Watermarking has recently been seen as a promising solution for protecting AIGC content. However, this paper challenges that perspective by introducing a unified approach for watermark removal or forgery. The proposed method, *Warfare*, leverages a pretrained diffusion model to first recover an unwatermarked version of a watermarked input. Then, it trains a GAN using 1) pairs of watermarked and unwatermarked inputs for watermark removal, or 2) pairs of unwatermarked inputs and inputs with forged watermarks for watermark forgery. Empirical results demonstrate that *Warfare* can effectively remove or forge watermarks.

### Strengths
- The proposed method is simple yet effective, demonstrating strong few-shot generalization capabilities.

- The research topic is timely, as AIGC has raised widespread societal concerns.

- The writing is clear and easy to follow.

### Weaknesses
# Weaknesses

### 1.  Overclaimed Speedup and Hidden Training Costs:
While the paper claims significant speedup in watermark removal at inference time, this improvement may be overstated. Although *Warfare* is indeed faster than purely diffusion-based methods during inference, it still relies on a pretrained diffusion model to generate the initial training data, a step that is computationally intensive and time-consuming. This dependency introduces a hidden time cost that is overlooked in the paper’s analysis. The authors could strengthen their claims by providing a more granular breakdown of time costs, including data preparation and model setup, rather than focusing solely on inference speed. Such clarification is crucial to accurately represent the efficiency gains, especially for potential real-world applications where both training and inference times play a role in assessing practicality.

### 2.  Unclear Justification for Integrating a GAN for Watermark Removal:
The strategy to leverage a GAN for watermark removal in *Warfare* is not well-justified within the paper. A pretrained diffusion model alone might be capable of removing watermarks by denoising the inputs, making the addition of a GAN seem redundant. If the GAN is intended to improve processing speed, the authors should factor in the time required for training this additional model, including the data acquisition for GAN training, which may offset any intended acceleration. A direct comparison between results using only the diffusion model and those using both the diffusion model and GAN would clarify the unique benefits, if any, that the GAN brings to the process. This clarification would help readers understand whether the GAN truly adds value or introduces unnecessary complexity.

### 3.  Questionable Black-Box Claim:
The paper claims that *Warfare* operates in a black-box setting; however, this claim could be somewhat misleading, as the evaluation primarily focuses on a limited set of watermarking schemes, specifically *StegaStamp*. This challenges the notion of a fully black-box approach, as the hidden information is limited to the embedded watermark and the hyperparameter *bit string length*, while the watermarking algorithm (*StegaStamp*) is determined.

### Questions
- Is using only the pretrained diffusion model sufficient for effective watermark removal?

- What is the time cost associated with data preprocessing?

### Soundness
3

### Presentation
3

### Contribution
3
