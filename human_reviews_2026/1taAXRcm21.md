# A Unification of Discrete, Gaussian, and Simplicial Diffusion

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
To model discrete sequences such as DNA, proteins, and language using diffusion, practitioners must choose between three major methods: diffusion in discrete space, Gaussian diffusion in Euclidean space, or diffusion on the simplex. Despite their shared goal, these models have disparate algorithms, theoretical structures, and tradeoffs: discrete diffusion has the most natural domain, Gaussian diffusion has more mature algorithms, and diffusion on the simplex in principle combines the strengths of the other two but in practice suffers from a numerically unstable stochastic processes. Ideally we could see each of these models as instances of the same underlying framework, and enable practitioners to switch between models for downstream applications.  However previous theories have only considered connections in special cases. Here we build a theory unifying all three methods of discrete diffusion as different parameterizations of the same underlying process: the Wright-Fisher population genetics model. In particular, we find simplicial and Gaussian diffusion as two large-population limits. Our theory formally connects the likelihoods and hyperparameters of these models and leverages decades of mathematical genetics literature to unlock stable simplicial diffusion. Finally, we relieve the practitioner of balancing model trade-offs by demonstrating it is possible to train a single model that can perform diffusion in any of these three domains at test time. Our experiments show that Wright-Fisher simplicial diffusion is more stable and outperforms previous simplicial diffusion models on conditional DNA generation. We also show that we can train models on multiple domains at once that are competitive with models trained on any individual domain.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses discrete data generation using diffusion models and proposes a unified framework that integrates direct discrete diffusion, simplex-based diffusion, and embedding-based Gaussian diffusion. Inspired by genetics, this approach enhances performance.

### Strengths
The paper reveals an intriguing connection between neutral evolution in genetics and the diffusion process in diffusion models. This bridge between the two fields allows for mutual conceptual enrichment and represents a high level of innovation.

The proposed SSP method is very interesting and inspiring, as it provides an intermediate layer that unifies all perspectives.

According to the experiments presented, the proposed method achieves significantly better performance than traditional approaches.

### Weaknesses
- The relationship between genetic drift and the n-simplex method is not clearly explained, and the paper would benefit from providing a more intuitive illustration of their connection.

### Questions
- There is a previous work of "Diffusion Evolution" that connects backward diffusion and forward evolution process. The authors proposed the opposite picture: the diffusion process corresponds to reverse evolution, while the denoising process represents forward evolution, naturally incorporating selection and reproductive isolation. How do the authors of the present paper view this distinction? Is there any conceptual connection between the two frameworks?
- Although this paper mainly focuses on discrete diffusion, it also builds a bridge between discrete and continuous spaces. So, I’m wondering if the framework presented in the paper can be applied to continuous diffusion?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a one-shoe-fit-all framework that perform diffusion on discrete, gaussian, and simplical data based on Wright-Fisher model. The paper shows a maththematical connection between them, allowing to training multiple types of data at once and yield competitive performance on DNA geneartion.

### Strengths
1. Establishing a connection among discrete, Gaussian, and simplex which has a root from Wright-Fisher -- a human population genetics model. This insight is crucial for building next AI model where a model is not limited to any particular type of data.

### Weaknesses
1. Limited practical applicability of multi-domain inference: While Figure 7 demonstrates competitive performance when the model handles different data domains at test time, this scenario has limited real-world relevance. In practice, each modality has distinct statistical properties—images typically follow continuous Gaussian distributions, while text is inherently discrete. Using a discrete or simplex representation for images, or Gaussian diffusion for language, is suboptimal for each respective modality. This raises questions about whether a unified framework provides practical advantages over domain-specific approaches.
2. The paper lacks experiments on combined domains like image and text. Since the primary contribution is demonstrating the advantages of a unified framework, the authors should have a baseline to train and evaluate a model on both language and vision domains jointly. Without such experiments, the claimed benefits of the unification approach remain unsubstantiated.
3. The paper lacks vision experiments, despite images being continuous data that naturally align with Gaussian distributions.
4. The paper does not provide a suitable choice of parameterization for a specific data domain. 


Minor concenrs:
1. Paper presentation is quite dense. Theorem 4.1 and 5.1 requires multiple readings to understand.

### Questions
Q1. The authors claimed the proposed method enable the flexibility of performing diffusion at all three domains at test time. I wonder if this still holds for large model (e.g. Stable Diffusion, Large Diffusion Language Model) where the modality-specific model is guaranteed with the scaling laws. 

Q2. The experiment setting is limitted to a single modality, no experiment demonstrates the practical value of the unified framework across genuinely different data modalities. For example: Can a model trained on discrete text data leverage the continuous parameterization for improved performance? Can the same model joinly learn from discrete language tokens and continuous image embeddings?

Q3. What's the effect of the ζ parameter in practical settings? Ablation is needed.

### Soundness
3

### Presentation
3

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
This paper unifies various approaches to diffusion under the view of population genetics (Wright-Fisher). They then use this theoretical result to draw from the literature of Wright-Fisher diffusion, practically improving existing simplicial approaches for DNA generation. They finally demonstrate that diffusion can be represented by a sufficient statistic parameterisation that is independent of the timestep and diffusion paradigm.

### Strengths
- Paper is well presented with a good mixture of visualisations for better understanding. 
- Good theoretical contribution, being able to tie together disparate concepts into a unified framework will be helpful for future researchers aiming to navigate the field. Note that I am a little unconfident here with regards to the specifics of the theory, as I did not check it carefully, and it is out of the domain of my knowledge. 
- Demonstrated practical impact from the Wright-Fisher perspective, improving performance on an applied task.
- The implications of SSP are somewhat interesting

### Weaknesses
The paper load for ICLR this year has been large, and so I have not been able to spend as much time as I would like on reviewing. I encourage the authors to correct any errors/misunderstandings I may have with regards to the paper.


1. **Experimental weaknesses**
    1. The DNA experiment is quite limited in scale (it is also not clear why the stabilised simplicial diffusion surpasses dirichlet FM).
1. **SSP motivation unclear**
    1. Although the idea of learning multiple diffusion paradigms in a single model is interesting, I think more clarity on the practical utility of such an approach might be of benefit. For example, I don't see a practical benefit in training an image generation diffusion model with an additional discrete diffusion objective on the 256 intensity levels. That is to say, I don't see that much benefit in SSP over comparing different diffusion approaches and picking the individual best.
1. **Better intuition**
    1. For the general reader interested in diffusion, the theoretical results of the paper are a little hard to intuit. However, I also appreciate that the authors have already included a number of helpful visualisations and I found the language and notation to be clear.
1. **Missing relevant references**
    1. [1] is a recent large scale DNA generation model that achieves SoTA by combining autoregressive and discrete diffusion/flow models.
    1. [2] propose to use gaussian diffusion for categorical data, but constrain the *clean data* distribution to the simplex hyperplane and train using cross entropy. 


[1] Li et al, Absorb & Escape: Overcoming Single Model Limitations in Generating Genomic Sequences, NeurIPS 2024

[2] Eijkelboom et al, Variational Flow Matching for Graph Generation, NeurIPS 2024

### Questions
See above

My weaknesses are all fairly minor and I believe the main contribution (theoretical) is good.
However, this is a little unconfident as I am not knowledgeable about the theoretical domain explored in the paper. As such, I am unlikely to increase my score as I cannot confidently verify the main contribution.

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
This paper proposes a unified formulation of discrete, simplicial, and Gaussian diffusion that, in different limiting cases, converges to each of these three diffusion formulations. The method is based on introducing repeats of tokens in a discrete sequence and diffusing each token independently with a Markov transition kernel. The theoretical results show that, in the limit of an infinite number of repeats, the proposed scheme converges to Gaussian diffusion. The proposed approach is evaluated on the protein modeling task.

### Strengths
The paper is well written, and the motivation and importance of the work are clearly explained.

The idea of introducing repeats is interesting and novel, as are the theoretical convergence results.

The source code is provided and well-structured.

### Weaknesses
Training a discrete diffusion model for an extended sequence is computationally expensive, so it is unclear why a practitioner would use the proposed approach. This is especially true for Gaussian cases, where working with large sequence lengths becomes infeasible. For example, if one uses a transformer with the number of repeats equal to 10, the full-attention complexity becomes 100 times more expensive. The computational efficiency is not discussed in the main part of the paper. Sequence length used in the evaluation experiments is 200 which is too small for a practical image or language modeling tasks.

Experimental verification is limited to the protein modeling task. There are no evaluations in visual or language domains, for example, on MNIST, where one could generate samples with both Gaussian and discrete diffusion and visually assess sample quality. This raises questions about the generalizability of the proposed approach.

### Questions
Line 101: If we consider softmax instead of argmax, the connection between gaussian and discrete diffusion in continuous time is achieved via Ito’s lemma. What will fail if we consider zero temperature in softmax? 

Line 111: Could authors explain why simplex diffusion sacrifices the ability to calculate a likelihood? If we have an ODE as in [1] we could compute likelihoods via integration of Jacobians of the vector field.

Line 182: Why is the closeness of ELBOs described as paradoxical? The ELBO is continuous with respect to weak convergence of probability measures. If we take a distribution that is a finite combination of delta measures and convolve it with a Gaussian kernel of small variance, then for sufficiently small $\sigma$ the two distributions are close in Wasserstein distance (and hence in the weak topology). Therefore, if a probabilistic quantity is continuous in the weak topology, we should indeed expect these two distributions to yield close values of that quantity by definition of continuity.

What is the computational overhead of the proposed approach? 

The reviewer is willing to raise the score if the weaknesses regarding computational complexity and generalizability beyond the protein modeling task are addressed.

[1] Stark et al. Dirichlet Flow Matching with Applications to DNA Sequence Design

### Soundness
3

### Presentation
4

### Contribution
4
