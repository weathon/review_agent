# Glauber Generative Model: Discrete Diffusion Models via Binary Classification

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 6

## Abstract
We introduce the Glauber Generative Model (GGM), a new class of discrete diffusion models, to obtain new samples from a distribution given samples from a discrete space. GGM deploys a discrete Markov chain called the heat bath dynamics (or the Glauber dynamics) to denoise a sequence of noisy tokens to a sample from a joint distribution of discrete tokens. Our novel conceptual framework provides an exact reduction of the task of learning the denoising Markov chain to solving a class of binary classification tasks. More specifically, the model learns to classify a given token in a noisy sequence as signal or noise. In contrast, prior works on discrete diffusion models either solve regression problems to learn importance ratios, or minimize loss functions given by variational approximations.  We apply GGM to language modeling and image generation, where images are discretized using image tokenizers like VQGANs. We show that it outperforms existing discrete diffusion models in language generation, and demonstrates strong performance for image generation without using dataset-specific image tokenizers. We also show that our model is capable of performing well in zero-shot control settings like text and image infilling.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes the Glauber Generative Model (GGM), a new discrete diffusion model framework. Unlike existing methods that often rely on regression or variational approximations, GGM frames the core denoising step as a binary classification problem: for each token in a noisy sequence, the model classifies it as either "signal" (belonging to the original data) or "noise" (introduced during the diffusion process). This simplification allows for efficient training using standard binary classification losses and avoids the complexities of estimating transition probabilities in high-dimensional discrete spaces. GGM utilizes the Glauber dynamics, a Markov chain that flips one token at a time based on conditional probabilities, to iteratively refine noisy input towards a clean sample. The authors apply their model to both text and image generation, demonstrating competitive performance on language modeling benchmarks and promising results on image synthesis using a generic VQGAN tokenizer. Notably, GGM outperforms other discrete diffusion models on language modeling while using fewer parameters. In image generation, it achieves decent FID scores compared to GANs and continuous diffusion models, though a gap to state-of-the-art remains. The authors also showcase the model's ability to perform zero-shot text and image infilling, highlighting the framework's potential for controlled generation.

### Strengths
- The reframing of discrete denoising as a binary classification task is a novel contribution. This  simplifies the learning problem compared to prior discrete diffusion models. The use of Glauber dynamics, while not new in itself, is creatively applied here within a diffusion framework and offers a theoretically grounded approach to discrete data generation.
- The paper demonstrates strong empirical results on language modeling, exceeding the performance of other discrete diffusion models and approaching that of much larger continuous models. The image generation results, while not state-of-the-art, are compelling given the use of an off-the-shelf VQGAN tokenizer and suggest further potential with dataset-specific tokenizers.

### Weaknesses
- The model training only differs from absorbing discrete diffusion(MD4, MDLM) by using NCE-style loss vs cross-entropy based loss, and since the regular absorbing diffusion models assume factorized denoising model distribution, this seems to me that the training objective of proposed model do not differ from previous works by much. So can you use a regular absorbing diffusion model for the GGM style sampling?
- Similarly, the defined forward process can also be seen as a special case of D3PM, and it might have simpler/different forms if considering the continuous time limit, which is absent in current work.
- The use of Glauber dynamics, which updates one token at a time, necessitates long sampling sequences (T=4096). Exploring alternative sampling strategies or modifications to the dynamics that enable faster convergence could significantly improve efficiency. Analyzing the impact of varying sequence lengths *T* on both sample quality and computational cost would be valuable.
- Although GGM achieves respectable FID scores, a significant gap remains compared to state-of-the-art GANs and continuous diffusion models on image generation. The authors acknowledge the limitations of the generic VQGAN tokenizer and suggest exploring dataset-specific tokenizers. However, this reliance on VQGANs, even with improvements, might inherently restrict the model's ability to capture fine-grained details and diversity compared to methods operating directly in pixel space.

### Questions
Will the authors open-source the code for training and evaluating the proposed models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the Glauber Generative Model (GGM), a new approach to generating discrete data like text and images using a process inspired by physics called Glauber dynamics. GGM works by changing one element at a time and simplifies the learning process to basic yes/no decisions about whether each element is meaningful or random noise. When tested on language generation, GGM performs better than similar existing models while using fewer computational resources, though it does not yet match the quality of top language models. For image generation, GGM produces good results without needing specialized preprocessing. While there is still room for improvement, the authors argue that GGM's straightforward approach makes it a promising direction for future research in generation systems.

### Strengths
This work introduces a theoretically elegant solution by reducing a complex generative task to simple binary classification problems. This makes the approach more straightforward to implement and understand compared to previous methods.

The model achieves strong empirical results, outperforming existing discrete diffusion models in language generation while using fewer parameters. It also shows competitive performance in image generation without requiring specialized tokenizers.

The method is versatile, working effectively across different domains (text and images).

The approach is computationally efficient, with lower complexity than previous methods, making it more practical to implement and scale.

### Weaknesses
Despite its improvements over other discrete diffusion models, GGM still performs worse than state-of-the-art methods like GPT models for text generation and continuous diffusion models for image generation.

The experimentation focuses mainly on comparing with other discrete diffusion models rather than providing comprehensive comparisons against the broader range of current generation methods. 

The paper does not provide detailed analysis of computational costs and training times compared to other methods.

### Questions
How would the model perform if given more data and computing power? 

How does the model perform on NLP tasks? 

How stable is the training process? 

How well does the model handle long-range dependencies in text? 

Why don't you use the more common name "Gibbs sampler" or "time inhomogeneous Gibbs sampler"?

Is the classification-based learning method related to noise contrastive estimation?

### Soundness
3

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
This paper introduces a new approach to discrete diffusion models called the Glauber Generative Model. The key innovation is using the Glauber dynamics from statistical physics as a discrete Markov chain to denoise sequences of tokens. The authors reduce the denoising process to a series of binary classification tasks, making it simpler than previous approaches. They evaluate GGM on language and image generation tasks, showing it outperforms existing discrete diffusion models for language generation. It also demonstrates competitive performance on image generation without using dataset-specific tokenizers.

### Strengths
1. Good Physical Intuition: The use of Glauber dynamics in the paper is well-motivated and mathematically principled. Glauber dynamics serves as a discrete analog to Langevin dynamics, which underlies many continuous diffusion models, making it a natural and theoretically elegant choice for discrete diffusion.
 
2. Theoretical Innovation: The paper presents a novel theoretical framework that elegantly reduces the complex problem of discrete diffusion to simple binary classification tasks. This provides a more efficient approach compared to previous methods that required learning full transition matrices.

3. Plenty of Empirical Results: GGM outperforms existing discrete diffusion models on language generation tasks while using fewer parameters. For example, it achieves better generative perplexity than previous methods with fewer parameters and comparable performance. Besides, it also achieves strong performance for image generation without using dataset-specific image tokenizers.

### Weaknesses
1. Performance Gap: Despite improvements over other discrete diffusion models, the performance of GGM is still worse than state-of-the-art autoregressive models for language generation and continuous diffusion approaches.

2. Computational Requirements: The model requires a relatively large number of denoising steps compared to other approaches. While each step may be simpler, the total computational cost during generation could be large.

3. Limited Ablation Studies: The paper doesn't explore how different architectural choices and hyperparameters affect performance. For example, there's limited analysis of how the number of denoising steps or the choice of noise distribution impacts results.

### Questions
1. Could the author provide theoretical analysis or bounds on the convergence rate of GGM compared to standard Glauber dynamics?
2. Could the author give ablation studies about the number of denoising steps, different round-robin scheduling strategies for token positions and vocabulary size?
3. How do the dataset-specific tokenizers and general-purpose tokenizers affect the result of image generation?

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
The paper proposes a new type of discrete diffusion model. The denoising process is defined as flip one token at a time, and the model predicts whether the token at the given location is noise. The proposed GGM is tested on both text and image, and shows strong generative perplexities compared with discrete diffusion model baselines.

### Strengths
- The method is innovative and yields promising results.
- The paper is clearly written, with a thorough discussion of related work.
- The experiments are comprehensive and well-documented.

### Weaknesses
Clarification questions:
- In Algorithm 2, line 5, the step is executed "for all a." Does this require $|\mathcal{X}|$ forward passes per model? If so, could this impact efficiency? Is there a way to parallelize this?
- Why can the model take a mask token as input? According to Eq. (1), $X_{t+1}$ inherits $X_t$ when $Z_t = \phi$. When exactly does $Z_t$ take the value $\phi$?

### Questions
Please see Weakness clarification questions.

### Soundness
3

### Presentation
3

### Contribution
3
