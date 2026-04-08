## Human Reviewer 1

### Summary
The authors propose a latent stochastic interpolant method, where they aim to learn a generative model with a low-dimensional feature. To ensure the condition of the stochastic interpolant, they come up with a novel parameterization of the stochastic interpolant, conditional on the sample $z_0$ and the encoded output $z_1$. They further optimize the training objectives for reducing the variance in the training and improving the performance. The experiments were conducted on ImageNet and show noticeable improvements in the computational cost.

### Strengths
The goal of jointly optimizing the encoder and the generative model is challenging. And the authors' proposal seems to help address the problem.

The experiments on varying initial distributions show that the method might help to be used in the case where the Gaussian might not be ideal.

The overall presentation is clear.

### Weaknesses
The authors should consider giving a more detailed benchmark, including more models with a pre-trained encoder, and discuss the advantages over the comparison methods.

It would be good if the encoder is evaluated for its linear probing accuracy, since it's very useful to see if the encoder is meaningful or not.

### Questions
Could you show more results regarding the encoder's performance? How does it perform when you train the interpolant model using varying initial conditions and training objectives?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes Latent Stochastic Interpolants (LSI), which extend stochastic interpolants into a jointly trained latent-variable model by learning the encoder, decoder, and latent dynamics together through a continuous-time ELBO. The method gives a nice unifying view of SI inside latent models, keeps the flexibility of arbitrary priors, and preserves simulation-free training, while avoiding heavy computation in pixel space. I haven’t checked every technical detail, but the approach feels clean, scalable, and grounded in solid continuous-time modeling ideas. It seems to address a hard and meaningful problem in a practical way, and the flexibility to apply this direction beyond ImageNet is the kind of capability I’d like to see more of at ICLR.

### Strengths
* This paper tackles a non-trivial and meaningful problem in generative modeling.

* The approach of using joint training of encoder + latent dynamics + decoder feels principled and elegant.

* Flexible priors and continuous-time formulation are nice advantages.

* Experiments on ImageNet show competitive performance, and this general framework could extend to many domains.

* The paper is generally well written and easy to follow.

### Weaknesses
* Evaluation is mostly on ImageNet, so the broader impact still needs to be validated.

* It’s not fully clear how robust the training is across architectures and hyperparameters.

* The paper is heavily reliant on the supplementary materials and mathematical details.
This makes the paper less accessible to the readers who are not interested in going through the theoretical details.
The authors could provide the key insights in a more intuitive way, possibly using graphical illustrations.

### Questions
* How sensitive is performance to design choices in the latent SDE or encoder noise scale?

* Any results (even preliminary) on other modalities or conditional settings?

* Please provide some failure examples, and discuss the limitations of the proposed method.

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
10

### Confidence
4

---

## Human Reviewer 3

### Summary
The key idea of this paper is to jointly train an encoder, decoder, and latent stochastic interpolant model using a continuous-time ELBO. Unlike traditional SI, which requires direct access to samples from both prior and target distributions, LSI constructs interpolants in the latent space, allowing end-to-end optimization.

### Strengths
1. The paper provides generalization of stochastic interpolants into latent spaces, enabling joint encoder–decoder–generator learning.

2. The derivation of a continuous-time ELBO in latent space seems theoretically grounded.

3. Demonstrates comparable or better FIDs across multiple resolutions on ImageNet

### Weaknesses
The assumptions of linear SDEs and Gaussian posteriors may limit its expressivity. It’s unclear how much these approximations affect performance or generalization.

### Questions
This paper is well motivated and well presented. I do not have many questions, but I am curious about whether the method could be extended to a learnable prior, also formulated within the joint learning scheme? The proposed method mentioned "arbitrary prior", which, however, are mainly simple, known distributions (e.g., Gaussian, Laplace). Would you also consider a learnable prior (e.g., an EBM prior or another more sophisticated prior)?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2