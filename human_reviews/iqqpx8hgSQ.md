# RAQ-VAE: Rate-Adaptive Vector-Quantized Variational Autoencoder

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 5, 5

## Abstract
Vector Quantized Variational AutoEncoder (VQ-VAE) is an established technique in machine learning for learning discrete representations across various modalities. However, its scalability and applicability are limited by the need to retrain the model to adjust the codebook for different rate requirements or encoding efficiency. We introduce the Rate-Adaptive VQ-VAE ($\textbf{RAQ-VAE}$) framework, which addresses this challenge with two novel discrete (codebook) representation methods: a model-based approach using a clustering technique for existing pre-trained VQ-VAE models, and a data-driven approach utilizing a sequence-to-sequence (Seq2Seq) model for variable-rate codebook generation. Our experiments demonstrate that RAQ-VAE achieves effective reconstruction performance across multiple rates, often outperforming conventional fixed-rate VQ-VAE models. This work enhances the adaptability and performance of VQ-VAEs, with broad applications in data reconstruction, generation, and computer vision tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on changing the codebook of a VQ-VAE to have more (or fewer) codewords. It proposes two different approaches:

1. model-based: Train a VQ-VAE and then use differentiable k-means clustering to split (or consolidate) codewords and fine-tune the pre-trained VQ-VAE architecture. 

2. data-based: Use a LSTM during VQ-VAE training to adapt the codebook to different sizes (i.e. train a VQ-VAE with 1024 codes, and use the LSTM during training to learn how to adapt the existing codewords to other sizes like 512, 8192, etc.).

The paper compares these two methods with training a VQ-VAE from scratch using other codebook sizes as well as randomly selecting codes from a pre-trained VQ-VAE with a large number of codewords.

### Strengths
**Problem:** This paper tackles an interesting problem: how to adapt the codebook of a pre-trained VQ-VAE to other applications where a different information rate is preferable.  

**Writing:** The paper is well-written and easy to follow.

**Uniqueness / Novelty:** It is my view this application (or problem setting) is quite novel; I have not seen other work in the VQ-VAE literature addressing this question. 

**Method:** Two different methods are proposed and explored (data-based and model-based). While both methods use existing machine learning components (i.e. this paper is not proposing a new building block or algorithm), they're used in a new setting that seems very fitting. A caveat to the last sentence is that the IKM algorithm to increase the codebook size seems to be a new contribution.

**Evaluation:** The empirical evaluation is comprehensive and sufficient to show both methods work (to varying degrees).

### Weaknesses
**Applicability:** It is unclear to me when the approaches presented in this work would be used. As likelihood-based or score-based generative models are often quite a bit more complex (and expensive) to train than VQ-VAEs, simply training a new VQ-VAE with the desired number of codebook vectors seems like it would not be an issue. Perhaps the authors can augment the introduction with specific examples where their approach would be desirable?

**Weak Baselines:** This work primarily compares with randomly selecting codes from a VQ-VAE to form the adapted codebook size. Unfortunately, many VQ-VAEs have low codebook utilization, so randomly selecting codes would likely incorporate codes that are never actually used by the model. A stronger (and very simple) baseline would be to select the $k$-most-used codes as measured on the validation (or training) set.

### Questions
See weaknesses. Additionally:

1. **Codebook utilization:** It is unclear how codebook utilization (or rate) changes with the adapted codebook sizes. Could this value be included for the experiments in Table 1 for the reconstructions? Specifically, for each batch of examples, count the number of unique codewords that are used and divide by the size of the codebook.

2. In Figure 2, it appears that the VQ-VAE trained with a codebook of size 1024 has the same (or lower) perplexity as both RAQ-VAE variants with K = 128 in both the top and bottom rows. How is the is the case when the VQ-VAE has such a larger vocab size? Is the codebook utilization for both of these models really around 10%?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces two codebook representation methods for VQ-VAEs: model-based and data-driven Rate-Adaptive VQ-VAEs (RAQ-VAEs), which allow for changes in compression rates without the need to retrain the models. The model-based RAQ-VAE employs differentiable k-means clustering (DKM) and a newly proposed inverse functional DKM. The data-driven RAQ-VAE utilizes a Seq2Seq model to generate rate-adaptive codebooks. The two proposed methods provide trade-offs between performance and cost. Extensive experiments demonstrate the effectiveness of the proposed methods in achieving rate adaptation without retraining.

### Strengths
* The problem this paper addresses is clearly articulated, and the motivation for this work is clear.
* This paper proposes two complementary and intriguing algorithms that appropriately address rate adaptation without the need for retraining.
* Extensive experiments have been conducted to verify the capabilities of the proposed methods.
* Discussion about performance limitation is provided.

### Weaknesses
* Although some discussion regarding performance limitations is provided, the degradation of reconstruction quality is not negligible in certain cases. In particular, as shown in Figure 2, the model-based RAQ-VAE fails to improve reconstruction performance even when the codebook size is increased exponentially.

* There are several parts that could benefit from clearer explanations:
  - A proof of the inequality, \$ \mathcal{L}_\mathrm{VQ}(\phi, \theta, \mathbf{e}; \mathbf{x}) \geq \mathcal{L} _\mathrm{RAQ}(\phi, \theta, \psi, \mathbf{e}; \mathbf{x}) \$, in the Training Procedure on page 6 is not provided.
  - It appears that a sampling scheme for \$ \tilde{K} \$ is needed during the training of data-driven RAQ-VAEs, but I could not find an explanation for this.
  - The paper mentions that while image quality declines slightly at the lowest bits per pixel, future work combining RAQ-VAE with advanced priors could further enhance the fidelity of generated images, as stated in Section 5.1. However, the reconstruction performance of the autoencoder may serve as an upper bound for the performance of generative models based on that autoencoder. It seems that efforts to improve advanced priors are less likely to enhance the fidelity of generated images when starting from a poor autoencoder. A further discussion on how to overcome this bottleneck would strengthen the paper's arguments.

### Questions
* Could you please demonstrate a proof of the inequality \$ \mathcal{L} _\mathrm{VQ} \geq \mathcal{L} _\mathrm{RAQ} \$?
* Could you clarify how to sample \$ \tilde{K} \$ during the training of data-driven RAQ-VAEs?
* Could you provide additional insights on how to address the reconstruction performance bottleneck to enhance the performance of generative models?
* Minor comment: there are unpaired parentheses in Algorithm 2, e.g., $Q(f_\phi(x)|e)) $

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper extends the VQ-VAE to the rate-adaptive VQ-VAE (RVQ-VAE) to enable the codebook adjusting the different data or the model scales. Experiments on reconstruction shows the RVQ-VAE outperforms the VQ-VAE.

### Strengths
1-A new idea is proposed to extend the VQ-VAE to adjust the he different data or the model scales.

2-The experiments on the reconstruction shows its superority.

3-The paper is easy to follow.

### Weaknesses
1-Experiments on the reconstruction is not enough. The authors are suggested to included the experiemnts on 1-codebook usage, 2-image generation etc, to show the advantage of the method. Also the update strategy ( exponential moving average) of RVQ-VAE is also encouraged. I think this work is not thorough studied yet.

### Questions
1-I want to see thorough experiments of the proposed RVQ-VAE.

2-I am also not quiet sure the motivation of this work.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces two codebook representation methods, model-based and data-driven approaches, which allow for varying the quantization rate within a single VQ-VAE model.

### Strengths
- RAQ-VAE allows a single model to adapt its codebook size without retraining to achieve variable rates for representations.
- RAQ-VAE has practical implications, particularly in applications that need adjustable compression rates such as real-time communications.

### Weaknesses
- The idea of rate adaptability is not new in the domain of neural compression, and the paper does not fully establish how RAQ-VAE advances beyond these existing frameworks. For instance, variable-rate neural image compression methods based on autoencoders and VAEs are prevalent, which is also mentioned in the related works.
- The experimental validation compares RAQ-VAE primarily to conventional VQ-VAEs. It is desirable to see if RAQ-VAE works with other methods, like more recent SQ-VAE and FSQ.
- The model-based approach is very weak.

### Questions
Why do we need the model-based approach?

### Soundness
3

### Presentation
3

### Contribution
2
