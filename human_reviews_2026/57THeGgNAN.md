# Generalization of Diffusion Models Arises with a Balanced Representation Space

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 2, 8, 6, 6

## Abstract
Diffusion models excel at generating high-quality, diverse samples, yet they risk memorizing training data when overfit to the training objective. We analyze the distinctions between memorization and generalization in diffusion models through the lens of representation learning. By investigating a two-layer ReLU denoising autoencoder (DAE), we prove that: *(i)* memorization corresponds to the model storing raw training dataset in the learned weights for encoding and decoding, yielding localized, spiky representations; whereas *(ii)* generalization arises when the model captures local data statistics, producing balanced representations. Furthermore, we validate our theoretical findings on real-world unconditional and text-to-image diffusion models, demonstrating that the same representation structures emerge in deep generative models with significant practical implications. Building on these insights, we propose a representation-based method for detecting memorization and a training-free editing technique that allows precise control via representation steering. Together, our results highlight that *learning good representations is central to novel and meaningful generative modelling*. Code is available at https://github.com/la0ka1/diffusion-gen-from-rep.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this work, authors study the representation space of diffusion models, showing especially for simplified scenario with 2-layered DAE that memorization can be attributed to the spikiness of activations. In particular, authors show that for overparameterized models examples are encoded in single neurons, while this is not the case for generalized underparametrized scenario.

### Strengths
- Insightful formulation with ReLU DAE which clearly introduces the main claims of the paper
- The theoretical contribution of the paper is clear and sound

### Weaknesses
- The paper's central observation—that memorization is linked to "spiky" neural activations—appears to be a rediscovery of the main finding from [1]. The authors propose using "the standard deviation of intermediate features" as a proxy for this spikiness. This metric is functionally identical to the z-score introduced in [1], which also identifies memorized examples by measuring the number of standard deviations an individual neuron's activation is from the mean.
- The analysis is restricted to a comparison between two well-understood extremes: an overparameterized model that memorizes and an underparameterized model that generalizes. This setup does not address the more complex and realistic scenario where a single, large-scale model simultaneously memorizes some training examples while generalizing others.
- In Section 3.1 there is an interesting example of memorization in overparametrized setup, where authors show that small DAE with large enough latent space (bigger than the dataset) can be optimized by memorizing individual samples (following block-wise structure). While this is a viable example, I am not sure that the conclusions are rigorous enough. I agree with comments presented in the first dot, but given the fact that the analysed solution is only the one of many local minimas, we cannot be sure that other minimas do not promote some generalization via neurons entanglement. 
- The paper makes the strong claim that “our findings show that the representation space is not a byproduct but a determining factor for generation.” This implies a causal relationship that the experiments do not support.
- The conclusions on steering and editing representations are drawn from an insufficient sample size of just 8 qualitative examples. There is no quantitative evaluation to validate these claims.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present a study on diffusion model generalization. They use a two-layer Relu network in a denoising autoencoder framework. In this framework, and assuming a type of cluster-separability, the authors present a theorem relates the optimal weights to the data clusters. They then proceed to derive properties for the low and high data limits, demonstrating they relate to memoization and generalization, and explain how these two domains have different weight properties. The authors then demonstrate these properties exist in pretrained diffusion models and how they can explain limitations of "steering" where the data is sparse.

### Strengths
Overall I very much enjoyed reading this paper, here are some specific points of strength:
- The paper is clearly organized, and concepts are well explained and presented.
- I very much like the paper is centered around rigorous theory, but makes it approachable in presentation, and demonstrates the strengths and downstream applications their theory through practical experiments with real datasets and models.
- This is a paper that makes a strong contribution toward understanding diffusion model generalization fundamentals, which is still quite poorly understood. The authors not only deliver great insight for people working on understanding this generaliation, but also make it approachable for practitioners working on downstream tasks

### Weaknesses
- There are a few things about definition 3.1 that seem odd to me. What 3.1 says is that the cluster means must be separated by at least some angle related to beta. This seems like a very specific type of clustering. It excludes for example clustering in the same direction (imagine a multi-modal gaussian with a sequence of modes in one dimension). Perhaps this is a common assumption I am not aware of. Could the authors defend this choice? Is it necessary for the theorems?
- It seems to me that depending on beta, the assumption the authors use sets some limitations on the ratio between the number of clusters and the data dimension. When I think about such clusters and data dimension in the image space, I would argue that the number of clusters is much larger than the data dimension. This would limit beta for such datasets, and because of that make theorem 3.1 less applicable since it assumes beta < 1. Can the authors comment?
- A small comment here is that the "margin" gamma (line 184) does not seem to be defined in the text.
- It seems surprising to me that the "spikiness" observation from Cor. 3.2 holds in more complex networks than two-layer Relu networks. While it seems like a perfectly reasonalbe intermediate state for the two-layer net, for more complex networks it seems like that would be much more complex. Beyond empirical results, can the authors give some intuition for why this would hold?

### Questions
- (see also weaknesses)
- Do the authors use sigma in the denoiser network? It is common practice for sigma to be part of the denoising representation. Would adding versus omitting it change any results?
- Can the authors confirm that in the low data case, the network's output matches the optimal empirical denoiser? It seems like this would be the case, but I did not find that back in the paper. It would be useful to the reader to make a statement about that. 
- Related to the last question: is the reason that the under-parametrized networks do not output the optimal empirical denoiser that the network does not have enough capacity to learn it?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to understand memorisation and generalisation of diffusion models from the representation perspective. Specifically, the authors built a theoretical framework in a two-layer ReLU diffusion model, which is more manageable in the theoretical sense. Firstly, the authors analysed the optimal solution of the above simplified diffusion model, which is over-parameterized. Through the analysis, they found that spiky representations happened, which are a a signal of memorisation. Secondly, the authors analysed the scenario of under-parameterisation, and showed that balanced representation as a signature of generalisation. Finally, the authors demonstrate the theoretical analysis can result in real-world impact, such as better memorisation detection metric and a steering approach for image editing.

### Strengths
This paper benefits from both theoretical analysis and further empirical impact. Especially, through the theoretical analysis, the authors showed that data representations (spikiness or balance) could be signals for memorisation/generalisation. Such signals could be used for memorisation detection, and have high accuracy than previous metrics and meanwhile is prompt-free.

### Weaknesses
I have the following concerns or questions, which may need authors' clarifications. Also please correct me if  I am wrong.

1. In case 1 of over-parameterisation, the authors found that with the optimal solution, the representations of a **single training sample** exhibits spikiness (in line 269). I am wondering if we input a different sample (not in training data) to such a neural network, whether the learned representations become a zero vector. 

2. In case 2 of under-parameterisation, the authors found that with the optimal solutions, the representations of a **single training sample** exhibits balance (in line 369), hence smaller std. I am also wondering what would happen if we input a different sample (not in training data, or a generalised sample). 

3. It seems that the authors consider two neural networks, one could memorise all samples, one could generalise new samples. However, for a trained diffusion model, it could both memorise and generalise. Why is the spikiness in representations an indicator for memorisation? 

4. The authors mainly discuss about optimal solutions for diffusion model. However, [1,2,3] show that as long as the number of training samples is finite, whether diffusion model is over-parameterised or not, there exists a theoretical optimum which could always generate memorised training data. Can you clarify the connection between such a theoretical optimum and the optimal solution shown in this paper? Is this because the model family of two-layer ReLU network cannot represent the theoretical optimum?

References:\
[1] Yi et al. On the generalization of diffusion model. 2023.\
[2] Gu et al. On memorization in diffusion models. 2023.\
[3] Kamb et al. An analytic theory of creativity in convolutional diffusion models. ICML 2025.

### Questions
See the weakness.

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
3

### Summary
This paper investigates the mechanisms of memorization and generalization in diffusion models, starting with a theoretical analysis based on a two-layer ReLU denoising autoencoder.

The claim is that mem. or gen. is determined by the learned representations: (1) memorization corresponds to learning spiky, sample-specific representations and weights that store the training data sparsely, when model parameters $p$ are larger than data size $n$ ($p\geq n$), and (2) generalization arises when the model learns balanced representations that capture local data statistics, when it is under-parameterized ($p \ll n$).

Based on the analysis, the authors introduce two practical applications: a highly efficient and effective method for detecting memorized content by measuring the spikiness of representations, and a training-free image editing technique based on representation steering, which demonstrates that generalized samples are more editable than memorized ones. The paper also validates these methods on some real-world models, including EDM, DiT and SD1.4.

### Strengths
- The paper presents a valuable contribution by proposing a more fundamental, representation-centric framework to explain the behaviors of memorization and generalization in diffusion models.
- The quality of the theoretical analysis is high, and the findings are shown to be consistent with observations in real-world models, as validated through two practical applications.
- The memorization detection method demonstrates high performance and broad applicability across different models and datasets.
- The observation that generalized samples are more steerable, while memorized samples exhibit brittle editing behavior, provides a novel and interesting insight.

### Weaknesses
- The theoretical framework is built upon a two-layer DAE. While empirical results suggest the conclusions hold more broadly, I do not think it is very clear why the findings based on the linear projection dimension $p$ vs. data size $n$ ($p\geq n$ or $p \ll n$) should transfer to deep, multi-layered UNet / DiT.
- The image editing experiments are limited in scope and somewhat unclear in their details.
  - The representation steering method is only demonstrated on Stable Diffusion 1.4, which is somewhat old in this area. Its effectiveness and the observed behavior have not been tested on more recent models like DiT / MM-DiT based text-to-image models.
  - The paper lacks a justification for how the "encoder" $g_\theta$ and "decoder" $h_\theta$ are determined (Line 461, Lines 990-992). The appendix specifies that features are extracted from 6 distinct layers in up_blocks.0 and up_blocks.1, but how these specific layers are selected is not provided. Additionally, I assume they form a collection of features with the size $100\times C\times H\times W \times 6$. Are these features averaged, or is the steering performed on 6 layers in parallel?
- The image editing approach requires generating 100 reference images to compute a mean representation for the target concept, which makes it inapplicable to image-guided editing scenarios using a few (<10) provided reference images. Furthermore, will the reference image generation procedure affect the editing quality? For example, how the editing process would be affected if the generated reference samples were themselves memorized or of low quality.

### Questions
Please refer to the weaknesses regarding the image editing approach and experiments.

### Soundness
3

### Presentation
3

### Contribution
2
