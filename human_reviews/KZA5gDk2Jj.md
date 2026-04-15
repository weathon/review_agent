# Recovery of Training Data from Overparameterized Autoencoders: An Inverse Problem Perspective

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 5, 3

## Abstract
We study the recovery of training data from overparameterized autoencoder models. 
Given a degraded training sample, we define the recovery of the original sample as an inverse problem and formulate it as an optimization task. In our inverse problem, we use the trained autoencoder to implicitly define a regularizer for the particular training dataset that we aim to retrieve from. We develop the intricate optimization task into a practical method that iteratively applies the trained autoencoder and relatively simple computations that estimate and address the unknown degradation operator. We evaluate our method for blind inpainting where the goal is to recover training images from degradation of many missing pixels in an unknown pattern. We examine various deep autoencoder architectures, such as fully connected and U-Net (with various nonlinearities and at diverse train loss values), and show that our method significantly outperforms previous methods for training data recovery from autoencoders. Importantly, our method greatly improves the recovery performance also in settings that were previously considered highly challenging, and even impractical, for such retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a novel approach for recovering images on which an auto-encoder was trained. The method assumes access to a set of degraded training images. Specifically, the degrading is done via a noisy *linear* operator. The authors study a particular variation of such an operator which erases the image pixels (i.e., diagonal matrix with $\\{0,1\\}$ entries). On a high-level, the method consists iof alternating steps of estimating the image $\hat{\mathrm{x}}$ and degrading operator $\hat{\mathrm{\boldsymbol{H}}}$ via an ADMM-like algorithm. The authors demonstrate the superiority of  their approach by comparing with DDNM method and iterative application of trained autoencoder (Radhakrishnan et al.), while also validating a variation of their method which has an access to the true pixel mask $\hat{\mathrm{\boldsymbol{H}}}$.

### Strengths
- a novel method for image recovery, that seems to be empirically superior to the existing alternatives
- the methodology is based on a well-established ADMM method

### Weaknesses
- the type of degrading of the images is limited to noisy linear ones
- the method still seem to assume a particular structure of $\hat{\mathrm{\boldsymbol{H}}}$ (i.e., diagonal for pixel erasure), e.g., the choice of the regularizer $\phi$ and etc.
- having access to a degraded training samples for recovery is a bit less practical than one can imagine
- the comparison and overall experimental evaluation seems a bit lacking

### Questions
- do the authors think that their method can potentially treat a general form of unknown $\hat{\mathrm{\boldsymbol{H}}}$? If so, I would be delighted to see some numerical evidence for that, even less rigorous would do

- it would be interesting to see, if the method still performs well under a mismatched scenario, i.e., the degrading process itself is not linear, but one can assume a certain form of $\hat{\mathrm{\boldsymbol{H}}}$ that replicates it close enough

- It would be interesting to see whether the method is able to perform well on an image which is not used during the training but close enough: pick some simple dataset and subsample images of a certain class and than look at the performance of the unused remaining ones

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposed an ADMM algorithm to recover training data from degraded observations. In detail, this paper looks at masking-based linear degradation functions similar to those in linear inverse problems, for AEs that can almost perfectly fit the training data. The proposed method outperforms previous techniques and improves recovery performance. The experiments also show a strong correlation between overfitting and recovery performance of the proposed method.

### Strengths
- The paper is clearly written and easy to follow. The formulation follows naturally from the reconstruction task, and all algorithm design follows naturally from the training objective. The decomposition of all parts of the algorithms is also straightforward. 
- It is an interesting application of ADMM to data reconstruction tasks and can be potentially used for the general neural network architectures with the plug-and-play method. There is theory provided to support this change.
- The experimental results show improvement over a number of baselines, and even under the noisy degradation case.

### Weaknesses
- The motivation of this paper is unclear. The paper mentions privacy challenges in modern ML at the beginning of the paper, but there lacks connection between the proposed data reconstruction technique and realistic privacy challenges. There should be more discussion (and examples) on how this formulation can represent realistic privacy concerns. For instance, one speculative example can be generating sensitive information with appropriate prompts in an LLM, which seems to be related to masking degradation. Another example can be image copyright issues, which further establishes a connection to membership inference. I hope the authors can draw a connection between the proposed formulation and these practical challenges, and make a clear statement on the attack taxonamy (e.g. black- or white-box, number of queries, etc.). 
- The word "overparameterized" is not appropriate in my thoughts. It usually means the middle hidden layer is much wider than input/output dimensions, which is in contrast to AEs where the latent dimension is usually smaller. Overparameterized networks may not always interpolate training data; they usually do under sgd or gd, but not under some other optimizers; and extrapolation may happen together with interpolation, which indicates non-overfitting. Based on your assumption, it is more accurate to use words like interpolating or overfitted AEs. 
- While there is theoretical justification for using plug-and-play to avoid the explicit definition of $s_f$, there isn't convergence analysis on the proposed algorithm. The initialization selection also seems to be heuristic. It is therefore reasonable to doubt how robust the proposed algorithm is, as well as its potential to generalize to a wider range of problems. 
- There isn't a proposed algorithm for noisy degradation that leverages $\sigma_{\epsilon}$ assuming it's known, which limits the scope the proposed task. While the experiments for small $\sigma_{\epsilon}$ shows the scalability of the proposed method, it is possible to fail for larger $\sigma_{\epsilon}$, which is not discussed in the paper. 
- The experiments are only conducted on very small training sets. From my understanding this is to ensure that the model overfits. However, it is very far away from practical scenarios, where most ML models under trustworthy concerns are giant and trained on massive data. It would be more interesting to look at (pretrained) models trained on the full datasets. These models can overfit on parts of the training set and not overfit on others; this can help us better understand the effect of overfitting to reconstruction. 
- There is no discussion on what will happen if you input a degraded sample $\notin$ training set but close to some $x_i$.

### Questions
The questions correspond to the weakness mentioned above.
- Motivation and connection to realistic privacy concerns?
- Any convergence and initialization analysis? 
- Is there an algorithm for noisy degradation? 
- What is the maximum noise for the proposed method to perform well? 
- Any experiments for standard full training sets? 
- What does the proposed method output for non-training samples?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on developing methods that can recover degraded training data samples from overparameterized autoencoders (AE).

They formulate the task as an inverse problem and proposes an iterative optimization method to solve the optimization.

The proposed method significantly outperformed the baseline method on both FC AE & UNet AE and CIFAR-10 images.

### Strengths
The formulation of the recovery problem is interesting and makes sense to me.

### Weaknesses
My main concern is on the assumption and experimental setup.

This work assumes that the autoencoder can overfit to the real data. I'm unsure whether this could happen in practice if we train AE on a real-world dataset. The largest data the AE is trained on is 25,000 images. Wondering that will happen if we apply the proposed method on a AE trained on a  real-world data, e.g., the AE used in Stable DIffusion that has been widely used by tons of work on image generation.

The current experimental setup is limited to small-scale. Would be great to see results on large-scale setup.

### Questions
See Weakness

### Soundness
2 fair

### Presentation
3 good

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
The consider the linear inverse problem of blind inpainting for a particular set of images which are used priory to train an overparameterized autoencoder. The use ADMM and the trained AE in a plug-and-play prior scheme to estimate the degradation operator on the training image, and fully recover the training image.

### Strengths
This problem is of particular interest to both the inverse problem community, and those interested in privacy issues concerning revealing the training data of a trained network. The paper is written clear and was easy to follow.

### Weaknesses
I find some motivations of the framework not well-suited for wider applications. They consider recovery of training images from degraded measurement of the image. How is this method applicable to inverse problems on data not appeared in the training data? How is the recovery when there is no measurement of the training image? The authors indeed shows (ans to my first question) that their approach cannot be used to solve general inverse problems on a test set. How applicable this method is on inverse problems with other measurement operators (e.g., additive Gaussian noise, Gaussian blurring, random inpainting, motion blur)? The experiments are not comprehensive.


The paper applies plug-and-play approach which is used is solving a general inverse problem using a denoiser on recovery of training images; they replace the denoiser with an autoencoder (the concept is exactly the same). The contribution of the method for inverse problem is limited.

Prior works on AE have shown that AE can recover training images. However, the reported numbers for the baseline is super low.

See my questions.

Minor comments

- I do not find the usefulness of the Theorem 1 for the general overparameterized, untied autoencoder. In general, (16) has been used and well-motivated by the plug-and-prior literature, so not clear why the authors on discussing usage of a network as a proximal operator for implicit regularization.

-  Please provide appropriate citations for alternating-minimization method (6), (7). This is a well-known procedure in dictionary learning. One example is [1].

- The last two paragraphs in Section 1 statements are very vague (has no citation) and not clear what the exact comparison is. See my question in Q section.


[1] Chatterji, N. S., & Bartlett, P. L. (2017). Alternating minimization for dictionary learning: Local convergence guarantees. arXiv preprint arXiv:1711.03634.

### Questions
1. The paper consider linear inverse problem of blind inpainting. I wonder how the performance of the framework is on other linear inverse problem (additive Gaussian noise, Gaussian blurring, random inpainting) or non-linear inverse problems (motion blur)?

2. Can the authors elaborate how much was the "small training dataset size" that was used in prior works?  The authors argued that their method can be applied on large training set images unlike prior work. The main experimental results include 600 images, and 50 images. Could the author elaborate on large training set?

3. Could the author elaborate which prior methods the outperform by citing (above section 2)? Does prior works try to recover training images given some degraded measurement or without having any measurement from it? Please elaborate, as this is crucial for fair comparison between the methods.

4. "our results also demonstrate the reduction in the recovery ability as the autoencoder is trained to a
higher train loss and less overfits its dataset. This, as well as our other results, are useful to understand the privacy risk of training data recovery in autoencoders." Is this a new finding? or I find this trivial.

5. The definitions of image regularizer and H regularizer in (5) are not rigorous and is not defined. Please elaborate on the wording "probable", and how the regularizers are implemented. Are they smooth? differentiable? ...

6. What is the motivation toward using ADMM as opposed to vanilla gradient on regularized objective? Why the splitting provides benefit in this case?

7. For (18), why not defining H only as a diagonal matrix in the first place? Then (18) is not needed.

8. Can authors elaborate on how they define "recovery"?

9. Can the author explain why the "AE iteration only does not work"? Providing a visualization on the iterations on AE to find its fixed point can be helpful.

10. Possible to visualize some failed examples?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
