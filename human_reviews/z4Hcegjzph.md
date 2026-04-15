# Pre-training with Random Orthogonal Projection Image Modeling

- Decision: Accept (spotlight)
- Scores: 6, 6, 8

## Abstract
Masked Image Modeling (MIM) is a powerful self-supervised strategy for visual pre-training without the use of labels. MIM applies random crops to input images, processes them with an encoder, and then recovers the masked inputs with a decoder, which encourages the network to capture and learn structural information about objects and scenes. The intermediate feature representations obtained from MIM are suitable for fine-tuning on downstream tasks. In this paper, we propose an Image Modeling framework based on random orthogonal projection instead of binary masking as in MIM. Our proposed Random Orthogonal Projection Image Modeling (ROPIM) reduces spatially-wise token information under guaranteed bound on the noise variance and can be considered as masking entire spatial image area under locally varying masking degrees. Since ROPIM uses a random subspace for the projection that realizes the masking step, the readily available complement of the subspace can be used during unmasking to promote recovery of removed information. In this paper, we show that using random orthogonal projection leads to superior performance compared to crop-based masking. We demonstrate state-of-the-art results on several popular benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on the task of MIM, by designing a new corruption process via the random orthogonal projection. Such a ROP strategy results in a more efficient and effective pre-training method. Another advantage of ROP is the guaranteed bound on the noise variance during the corruption process.

### Strengths
1. This work is based on the sound theory of random orthogonal projection.
2. ROPIM is able to achieve more superior performance in a shorter pre-training time.
3. The decoder only contains one linear layer, being more slight comparing with MIM.
4. The experiments verify its effectiveness on several downstream tasks, including classification and segmentation.

### Weaknesses
1. It is a little difficult to understand why does the proposed method is more superior than MIM. According to my understanding, the ROP strategy randomly discards some local (not global) patterns during corruption as shown in Fig. 9. This is very similar to MIM. So, I can't intuitively catch what results in the superiority of ROP in the field of MIM. It would be better to have a discussion in the paper.

### Questions
1. Proposition 1 gives a complicated way to generate a projection matrix $P$ that contains only three elements, namely {-1, 0, 1}. This procedure relies on two auxiliary variables of $h$ and $s$. Is it possible to directly sample from {-1, 0, 1} for each entry of $P$?

2. Following question 1, the projection matrix $P$ is composed of {-1, 0, 1}. Generally speaking, "0" denotes discarding some embed patch in projection. It is hard to understand the function of "-1".

3. In proposition 1, I guess: $h \in I^d_{K^{'}}$ should be $h \in I^K_{K^{'}}$?

4. In proposition 2, the notation for $\phi^{'}$ is missing.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper considers the self-supervised learning problem, which has attracted much attention recently. While the masked image modeling, e.g. masked autoencoder, has recently shown very promising performance in self-supervised learning for visual pre-training, the authors aim to improve it by proposing a new image modeling. Specifically, unlike the masked image modeling applying random crops to the input and learning to recover the masked inputs with an encoder-decoder network, this paper considers a random orthogonal projection modeling which uses a random subspace for the projection and then learns to recover the complement of that subspace. Provided experiments show that the proposed random orthogonal projection can yield better performance than the crop-based masking.

### Strengths
Though simple and straightforward, to my knowledge the proposed random orthogonal projection modeling for self-supervised learning is novel. Provided experimental results have demonstrated better performance of the proposed random orthogonal projection method in comparison with the masked image modelling using crop-based masking.

### Weaknesses
The proposed method is somewhat heuristic.

### Questions
Some of the Propositions in Section 3.1 are rather straightforward and would be better not be expressed as Proposition.

$\ell_1$ loss is used in the reconstruction loss, would the MSE loss yield worse performance?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The work proposes to use random projections in the place of masked images for pre-training the ViTs. The work is shown to lead to better performance for classification tasks compared to the MIM methods. The work seems novel enough where they replace binary mask with floating point mask.

### Strengths
+ The use of linear algebraic projection technique in both the method and the loss function for reconstruction.

+ The results are achieved better with a considerably smaller number of epochs.

+ The work is well-motivated from the basics and seems reproducible. 

+ The transfer learning results are added value to the work as such. 

+ The work should be useful as a pre-trainer for several ViT based applications.

### Weaknesses
- I'm not sure if all the recent works on MIM have been compared with. Authors are requested to comment on this.

### Questions
Are there any recent works which have been exempted from comparison?

Apart from classification and semantic segmentation, do the authors have results on any other applications?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
