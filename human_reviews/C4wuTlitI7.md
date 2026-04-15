# Understanding Masked Autoencoders From a Local Contrastive Perspective

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 3

## Abstract
Masked AutoEncoder (MAE) has revolutionized the field of self-supervised learning with its simple yet effective masking and reconstruction strategies. However, despite achieving state-of-the-art performance across various downstream vision tasks, the underlying mechanisms that drive MAE’s efficacy are less well-explored compared to the canonical contrastive learning paradigm. In this paper, we explore a new perspective to explain what truly contributes to the “rich hidden representations inside the MAE”. Firstly, concerning MAE’s generative pretraining pathway, with a unique encoder-decoder architecture to reconstruct images from aggressive masking, we conduct an in-depth analysis of the decoder’s behaviors. We empirically find that MAE’s decoder mainly learns local features with a limited receptive field, adhering to the well-known Locality Principle. Building upon this locality assumption, we propose a theoretical framework that reformulates the reconstruction-based MAE into a local region-level contrastive learning form for improved understanding. Furthermore, to substantiate the local contrastive nature of MAE, we introduce a Siamese architecture that combines the essence of MAE and contrastive learning without masking and explicit decoder, which sheds light on a unified and more flexible self-supervised learning framework.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
CL and MIM are the two important self-supervised pre-training methods for computer vision. This paper discusses the connection between them and proposes a new method, which may benefit the community. This paper is well-organized and well-written.

### Strengths
The motivation is interesting and the writing is good. The authors provide an interesting finding that MIM and CL methods have a close relationship. It motivates the authors to propose a new method by combining CL and MIM. Moreover, The experiments show the fair performance of the proposed method.

### Weaknesses
1. There are currently too few experiments to demonstrate the effectiveness of the approach. At present, the lack of experiments includes longer-epoch training, benchmark object detection, benchmark semantic segmentation, and other experiments.
2. The paper should outperform other contrastive + masked methods, e.g., MST[1], iBoT[2] which were proposed two years ago.

[1] Mst: Masked self-supervised transformer for visual representation. NeurIPS2021.
[2] iBOT: Image BERT Pre-Training with Online Tokenizer. ArXiv2021.

### Questions
Please refer to Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper empirically analyzes the behaviors of the decoders in MAE. The author finds that 1) the first layer of the decoder primarily relies on the positional information while the subsequent layers obtain higher-level semantic information 
2) the receptive field of the decoder is limited. Based on that, this paper reformulates the objective of MAE and proposes a new architecture that combines the spirits of MAE and contrastive learning.

### Strengths
1. The writing is well and easy to follow. And the main messages based on the analysis look solid and insightful.
2. The representations learned by Uni-SSL (without masking techniques) are more similar to the mask models instead of contrastive models, which is interesting and verifies the analysis proposed in this paper.

### Weaknesses
1. As shown in Figure 3, the attention distance increases with the larger mask ratio. However, both the fine-tuning and linear accuracy are not monotonous in MAE. Is it possible to discuss the trade-off in the choice mask ratio based on the analysis in this paper?
2. In Figure 5(b), the fine-tuning accuracy of DINO is higher than MAE and Uni-SSL while in Table 2 it is the opposite. What differences have I missed?
3. The paper focuses on analyzing the behavior of the decoders in MAE. Is it possible to provide some insights about how to design a better decoder?
4. The connections between the analysis of the decoder, the analysis of the training objective, and the design of the new framework are a little confusing. It would be better to provide a more detailed discussion about that.

### Questions
see my comments above.

### Soundness
3 good

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work has two main contributions: 1)  a comprehensive understanding of the role of the decoder part in MAE, uncovering the fact that the reconstructive decoder part learns local features within a limited receptive field. This work statistically analyzes the similarity among the learned attention maps for all mask tokens on the ImageNet validation set. These findings elucidate that the initial decoder layer predominantly depends on token positional data, whereas in the subsequent layers, the decoder progressively combines more advanced semantic information while maintaining positional guidance. 2) proposing a Siamese architecture combining MIM and contrastive learning in a unified manner.

### Strengths
This work has explored the decoder’s role of MAE in helping the encoder learn “rich hidden representations” in a generative manner, uncovering the fact that the decoder part enables to learn local features.

### Weaknesses
### Comparison to prior works

This work proposed a combination of Masked Image Modeling (MIM) and contrastive learning (CL) using Siamese architecture, however, this strategy has already been explored in several methods (iBOT [1], CAE [2], and CMAE [3]).  Moreover, this work just commented on these works, not comparing the proposed methods with these prior works. It seems that outstanding points of this work do not exist compared to the prior works.

### Weak experiments

1) very short training epoch: This work was trained for only 100 epochs, which is very short to compare with state-of-the-art methods

2) lacks comparison with prior works (iBOT [1], CAE [2], and CMAE [3]). It would be better to compare with the [MIM+CL] combination methods.

[1] Zhou et al., iBOT: Image BERT Pre-Training with Online Tokenizer, ICLR 2022.  
[2] Chen, et al., Context Autoencoder for Self-Supervised Representation Learning, Arxiv, 2023.  
[3] Huang et al., Contrastive Masked Autoencoders are Stronger Vision Learners, Arxiv, 2023.

### Questions
No questions.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper discusses MAE is a local-level contrastive learning, and propose a approach to self-supervised learning. This paper is well-written.

### Strengths
The authors provide an finding that MAE is local contrastive learning, though some previous paper express the same opinion. Moreover, the paper propose a new method by combining contrastive learning and MAE. Finally, the writing is good.

### Weaknesses
1. The paper should cite and compare with related work, like MST, iBoT, CMAE, and so on. They are contrastive + masked methods.


2. The main opinions of this paper is proposed by previous work. Hence, the authors should not ignore them.

3. Current experimental results cannot demonstrate the method is effective. The authors should show detection and segmentation experiments in MAE to fairly compare with MAE and other related work.

### Questions
Please refer to Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
