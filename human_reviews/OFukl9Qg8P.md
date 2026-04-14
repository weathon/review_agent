# Resolution Attack: Exploiting Image Compression to Deceive Deep Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Model robustness is essential for ensuring the stability and reliability of machine learning systems. Despite extensive research on various aspects of model robustness, such as adversarial robustness and label noise robustness, the exploration of robustness towards different resolutions, remains less explored. To address this gap, we introduce a novel form of attack: the resolution attack. This attack aims to deceive both classifiers and human observers by generating images that exhibit different semantics across different resolutions. To implement the resolution attack, we propose an automated framework capable of generating dual-semantic images in a zero-shot manner. Specifically, we leverage large-scale diffusion models for their comprehensive ability to construct images and propose a staged denoising strategy to achieve a smoother transition across resolutions. Through the proposed framework, we conduct resolution attacks against various off-the-shelf classifiers. The experimental results exhibit high attack success rate, which not only validates the effectiveness of our proposed framework but also reveals the vulnerability of current classifiers towards different resolutions. Additionally, our framework, which incorporates features from two distinct objects, serves as a competitive tool for applications such as face swapping and facial camouflage. The code is available at https://github.com/ywj1/resolution-attack.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces the “Resolution Attack”, a novel form of adversarial attack. Specifically, it generates images with semantic ambiguity across different resolutions to achieve the attack. To achieve this, the paper leverages diffusion models and a proposed staged denoising strategy to introduce two types of resolution attacks: RA (adversarial samples generated conditioned on dual text) and RAS (adversarial samples generated conditioned on images). Experimental results demonstrate a high attack success rate and show further potential in applications such as facial forgery.

### Strengths
- Well, I feel this work interesting as it presents an attack manner that is fundamentally different from previous adversarial attack approaches, that, leveraging the dual semantics introduced by resolution changes to confuse classifiers, which is quite refreshing. 
- The authors' proposed optimization framework incorporates a Dual-Stream Denoising (DS Module) that locates perturbation regions in high- and low-frequency areas to achieve the attack, providing a good baseline for this attack manner. 
- The experiments and ablation studies presented are comprehensive and solid.

### Weaknesses
Some concerns: 
- (minor, discussion) Does the author thoroughly investigate the problem being studied? Specifically, are there existing resolution-based attack methods, and if so, should they be considered baselines? I suggest the author provide a more in-depth discussion of related work. 
- (major, experimental) Although the effectiveness of the attack method on deep classifiers has been well demonstrated, all the models used are CNN-based classifiers with relatively low input resolution (224×224). In contrast, I am more interested in whether these adversarial samples can disrupt existing vision-language models (VLMs, e.g., CLIP, BLIP-2, LLAVA, etc.), which possess more excellent generalization capabilities than task-specific models. I think adding some evaluation results on VLMs, such as zero-shot classification or even image captioning and VQA, would significantly enhance the assessment of this method's value. 
- (minor, experimental) I suggest including experiments under defensive settings to evaluate whether these adversarial samples can still successfully confuse classifiers when facing adversarial preprocessing (e.g., different resolution compression methods or adversarial purification). 
- (minor discussion) Given the potential risks associated with this work, I also recommend discussing the societal impacts and potential defence strategies at the end of the paper.

### Questions
This paper is sufficient excellent, but I hope the authors can address the above concerns.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, the authors propose a new attack method called as resolution attack, i.e., an automated low-resolution image generative framework capable of generating dual-semantic images in a zero-shot manner. Experiments are conducted to verify the effectiveness of resolution attack.

### Strengths
1. A new method called as resolution attack is proposed, which is interesting.  
2.  A reasonable technique is proposed to implement the proposed idea
3. The paper is easily understood.

### Weaknesses
1. It seems the test images are important for the success of resolution attack, because when an image is compressed into a low-resolution image, the semantic content is easily changed. In the paper, the authors don’t give the test dataset, so I don’t know whether the proposed method  is still effective for all images. 
2. The method should be compared with the compression method that randomly reduce the resolution of the test image. Compared with this baseline, the readers can better see the effectiveness of this attack. 
3. Could the proposed method is till effective for vision transformer? The authors should give the discussion.

### Questions
see the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel "resolution attack" that deceives both classifiers and human observers by creating images with distinct semantics at different resolutions. The proposed automated framework generates dual-representation images in a zero-shot manner, using generative priors from large-scale diffusion models and a staged approach to achieve smooth transitions across resolutions.

### Strengths
1. The designed "resolution attack" is very innovative and presents interesting effects.
2. It reveals the vulnerability of classifiers under the "resolution attack."
3. The writing is clear, and the ideas are easy to follow.

### Weaknesses
1. The classifiers tested in the article are primarily based on CNN methods and do not include evaluations of the latest classifiers. I am curious about how transformer architecture models perform under the "resolution attack," and whether methods using feature pyramids could better overcome the "resolution attack." The authors could provide more analysis in this area.
2. Although the authors validate the effectiveness of their method on existing classifiers, they do not provide a detailed comparison with other attacks (such as adversarial attacks). The authors might consider analyzing how the proposed "resolution attack" compares to other generative attacks that also utilize the diffusion model.
3. The paper touches on applications like face swapping and camouflage without addressing the potential ethical implications.

### Questions
I am curious about the performance of transformer architecture classifiers in comparison to CNN architecture classifiers under the "resolution attack," and whether methods using feature pyramids can better overcome the "resolution attack."

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Ensuring model robustness is critical for the stability and reliability of machine learning systems. Although substantial research has addressed areas such as adversarial and label noise robustness, robustness against variations in image resolution—especially high-resolution images—remains underexplored. To bridge this gap, this work introduces a novel form of attack: the resolution attack. This attack generates images that display different semantics at varying resolutions, deceiving both classifiers and human observers. The resolution attack is implemented through an automated framework designed to create dual-semantic images in a zero-shot manner. Specifically, it leverages large-scale diffusion models for their comprehensive image construction capabilities and employs a staged denoising strategy to achieve smooth transitions across resolutions. Using this framework, resolution attacks were conducted on several pre-trained classifiers, with experimental results demonstrating a high success rate. These findings validate the effectiveness of the proposed framework and highlight the vulnerability of current classifiers to resolution variations. Additionally, this framework, integrating features from two distinct objects, presents a powerful tool for applications like face swapping and facial camouflage.

### Strengths
1. This paper introduces a new test-time attack, the resolution attack, which demonstrates effective performance across several classifiers.

### Weaknesses
1. The attack scenario presented is quite limited. The paper uses a high resolution of $512 \times 512$, but the size of low-resolution images is unclear, though figures suggest it might be $32 \times 32$. In practice, this resolution is only common in classifiers trained on small-scale datasets like MNIST or CIFAR-10/100, whereas real-world classifiers, such as those trained on ImageNet, typically use a $224 \times 224$ input resolution, making this type of attack less applicable.

2. It is expected that this attack could succeed, as adversarial images here are generated from scratch. However, in practical scenarios, adversarial images are typically crafted by modifying an uploaded clean image to influence predictions while maintaining visual similarity to the original. Generating adversarial images from scratch, therefore, has limited practical application.

3. The module proposed lacks novelty, as the separation of control between modules for detail and shape generation is quite standard.

4. Key implementation details are missing, such as the resolutions at which images successfully achieve adversarial effects.

### Questions
See weakness above.

### Soundness
2

### Presentation
2

### Contribution
2
