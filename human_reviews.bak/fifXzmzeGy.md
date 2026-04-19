# Scalable Bayesian Learning with posteriors

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Although theoretically compelling, Bayesian learning with modern machine learning models is computationally challenging since it requires approximating a high dimensional posterior distribution. In this work, we (i) introduce **_posteriors_**, an easily extensible PyTorch library hosting general-purpose implementations making Bayesian learning accessible and scalable to large data and parameter regimes; (ii) present a tempered framing of stochastic gradient Markov chain Monte Carlo, as implemented in posteriors, that transitions seamlessly into optimization and unveils a minor modification to deep ensembles to ensure they are asymptotically unbiased for the Bayesian posterior, and (iii) demonstrate and compare the utility of Bayesian approximations through experiments including an investigation into the cold posterior effect and applications with large language models.

_**posteriors**_ repository: https://github.com/normal-computing/posteriors

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes `posteriors`, a new PyTorch library for scalable Bayesian learning. Unlike previous libraries, `posteriors` is functional: The user can freely code their own minibatch-wise loss function and `posteriors` will use it to compute the posterior. Various Bayesian deep learning methods are available, such as the (linearized) Laplace approximation, variational inference, and stochastic-gradient MCMC.

The key selling points of `posteriors` is its composability, extensibility, swappablilty, and its functional nature. For instance, performing Bayesian inference on parameter-efficient fine-tuning in LLMs is a breeze since `posteriors` does not need to know the underlying model.

Moreover, the authors discussed a parallel deep ensemble technique and applied it to the Llama-3-8b model, effectively enabling this latest LLM model to detect out-of-distribution data. Further experiment results on inference techniques available in `posteriors`, such as the linearized Laplace for continual learning, were also presented in this paper.

### Strengths
I like the approach of `posteriors` in making Bayesian inference easier to perform, even in highly sophisticated models like LLMs. Indeed, by focusing in composability and by being functional-first means that `posteriors` is not restricted to a small class of models.

Moreover, I'm impressed by how clean and general the code of `posteriors` from the users' point-of-view (Fig. 3). I believe that `posteriors` is a useful addition to the community, enabling easy-to-do Bayesian inference on various models.

### Weaknesses
While the main contribution, i.e. the `posteriors` package, is great, the main weakness of this work is the presentation in the paper itself.

The story-telling of the paper is rather haphazard, badly motivated, and rather disjoint. The main purpose of this paper is to describe `posteriors`, to report the details of the package, and to show that it is useful. However, right from the introduction, the authors fail to motivate why do the community needs `posteriors` while other libraries such as `laplace-torch` exists, which also support sophisticated models like [LLMs](https://aleximmer.github.io/Laplace/huggingface_example/) and [reward models](https://aleximmer.github.io/Laplace/reward_modeling_example/). Indeed, the authors only motivate and compare `posteriors` with packages for traditional Bayesian inference (Line 86) and other functional libraries like Pyro in the Related Work section. Please note that I'm not saying that `posteriors` should not exist, but the motivation and comparison should be made stronger to make the paper more appealing to the readers, who are, ultimately, potential users.

In Section 3, I am underwhelmed by the fact that the authors did not describe their library in detail. As a library paper, I would expect the paper to be structured as in e.g. [PyTorch's paper](https://arxiv.org/pdf/1912.01703). I suggest the authors expand Section 3 more and reduce unnecessary discussion like Sec. 4, which ultimately does not contribute to the description of the library itself.

Indeed, Section 4 is rather disjoint from the rest of the paper. It is much more suitable for a method paper. If the main purpose, as the authors intended, is to showcase `posteriors`' extendability, then this section will be much better used to show _how_ to implement the method with `posteriors`, not to propose the method itself. 

I am also confused about the selection of the experiments, especially Sec. 5.1. There, the authors only studied the cold posterior effects, which reads more like a validation of the parallel MCMC method proposed in Sec. 4. Again, it reads like a method paper, and thus disjointed from the rest of the paper.

Secs. 5.2 and 5.3 are fine in terms of the problems. But they would be much stronger if the authors showcased _how_ `posteriors` play a part in making this possible. For example, the code needed to construct such a Bayesian LLM is very short with `posteriors`. The failure to do so made the paper less coherent, and ultimately, it is hard for a reader to fully understand the benefits of `posteriors`.

Lastly, I noticed quite a bit of typo/misformatting in the paper, e.g. incorrect use of `\citep` and `\citet` (Line 157 etc), incorrect reference formatting like "A.1" instead of "Appendix A.1" in line 124.

In sum, I believe the paper needs several more writing iterations to make the story coherent, thus fully showcasing the potential of `posteriors`, which I do really like.

### Questions
The authors mentioned that `posteriors` is unable to provide model-aware techniques such as KFAC-Laplace. I believe that the future of Bayesian foundation models requires model-aware approaches: E.g. the current Laplace-LoRA with KFAC covariance (Yang et al., ICLR 2023) is insufficient since KFAC combined with LoRA has the same costs (at least memory-wise) as KFAC on the original weight matrices, thus negating the benefits of PEFT. (To see this, notice that KFAC on LoRA implies Kronecker factors of $n \times n$, $k \times k$, $k \times k$, and $m \times m$ for $n \times m$ original weights.) How does `posteriors` handles these limitations?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces "posteriors", a PyTorch-based library designed to enhance scalability and usability of Bayesian inference for deep learning. The focus is on making Bayesian learning feasible for large datasets and models, often challenging due to the computational demands of approximating posterior distributions. Authors proposes a novel approach by reframing stochastic gradient MCMC as a form of gradient descent, creating a unified pathway between them. Experimental results highlight improvements in generalization, continual learning, and out-of-distribution detection, facilitated by scalable, minibatch-focused Bayesian techniques.

### Strengths
- the paper is mostly clear and concise, presents a straightforward approach, keeping primary discussions in the main sections and moving technical details to the appendix.
- the paper provides extensive experimentation, including large datasets and Bayesian inference with large-scale models, showing the library's potential across applications
- the library is composable, extendable, and scalable; and promotes compatibility with pytorch and other popular libraries, and could therefore have a relevant impact in the community
- the paper unifies stochastic gradient MCMC with traditional gradient descent, enabling efficient transitions between sampling and optimization
- the library is open-source, and provides a functional API that aligns with pytorc , and so facilitates user contributions

### Weaknesses
- the clarity of writing in some technical sections could be improved. Certain descriptions of key concepts, particularly the tempered SGMCMC framework (section 4), would benefit from a more step-by-step breakdown. I felt it was a bit too rushed.
- the main text could further emphasize how "posteriors" differentiates itself technically, showing specific implementations or optimizations unique to this library (in a similar spirit to Figure 3). The current presentation could be misinterpreted as implementing previously established methods rather than providing a novel platform for Bayesian inference
- although the experiments are robust, comparisons to other popular Bayesian libraries are quite limited. Explicitly comparing computational efficiency or accuracy with other Bayesian packages would better position "posteriors" in the current landscape.

### Questions
- could the authors clarify how "posteriors" manages scalability in large parameter regimes, particularly regarding memory management with large covariance matrices?
- how does "posteriors" compare in ease of use and computational efficiency with similar Bayesian packages like Pyro or BlackJAX?
- Given that parallel SGMCMC shows promising results in certain tasks, do the authors anticipate specific scenarios or types of models where it may not perform well, and if so, why?

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
4

### Summary
I have increased my score from 5 to 6. 
----- 
This paper presents a pytorch library called `posteriors` for general-purpose Bayesian learning tasks, which is featured by (1) composability, (2) extensibility, (3) functionality, (4) scalability, and (5) swappability. The implementation includes a tempered version of Parallel SGMCMC, which is shown to combine the favorable properties of deep ensembles and serial SGMCMC. Empirically, the library has been tested for cold posteriors, continual learning with LORA, and finetuning of Llamma3 with Bayesian learning, demonstrating the benefits of Bayesian paradigm for machine learning.

### Strengths
The paper is in general well-written and the empirical results are presented clearly. I like the key features of the proposed library which are lacking in existing libraries. Below are the key strengths. 

- The key features of the proposed library as sketched in section 3 are important for modern Bayesian deep learning. 

- The experiments cover various scenarios in current deep learning landscape. 

- the library is open-sourced and supports customization.

### Weaknesses
I think the paper can be improved by better scoping the problems that the library aims to solve, the inference approaches provided by the library, and the practical tradeoffs between the proposed Bayesian approach versus the non-Bayesian approaches (e.g. SGD, LoRA and Deep Ensemles and their various combinations). 

In particular, the following questions should be addressed:


- What problems can the proposed library solve? Examples include disentangling various sources of uncertainties, continualing learning, finetuning, standard classification / regression,  classical Bayesian inference problems e.g. time series, hierarchical modeling? Detailing these problems will help readers understand what set this approach different from the existing ones such as Stan and BlackJAX. And perhaps a table that outlines the differences among these approaches can be helpful. 

- What are the key inference algorithms currently supported in the library?

- What is the practical time and memory complexity of the current implementations when compared against the existing non-Bayesian approaches? 

-- What are the limitations / scenarios that the current library does not support?

Overall I think the proposed library is very promising. And the throughout the paper some of the above points have been touched upon briefly. However, a comprehensive discussion is lacking, which is critical to the potential target users of the library. I will consider raising my scores if the above points are addressed during rebuttal.

### Questions
- figure 4: only 4 lines are plotted but 5 lines are included in the legend box?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces the `posteriors` pytorch library for programmatic approximate Bayesian inference with a focus on deep neural networks as models, which further contains some advances, like a variant of stochastic gradient Markov chain Monte Carlo. The paper assesses the implemented methods on a variety of problems.

### Strengths
- Software release that allows other researchers and practitioners to build on the proposed and baseline methods. Overall appears to contain MCMC-based methods, variational inference, and Laplace approximations, all of which are relevant and practical methods.
- Clearly written motivation for Bayesian inference in general
- Interface seems very clear and flexible due to functional paradigm
- Additional contribution in form of a new SGMCMC algorithm?
- Convincing experiments on modern architectures like lora and llama.

### Weaknesses
- Only diagonal variational and Laplace approximations, which are known to have certain issues and are generally outperformed by more structured posterior approximations. For example, Laplace approximations perform better across the board when using layer-wise Kronecker-factored structure (Ritter et al, ICLR 2018, https://discovery.ucl.ac.uk/id/eprint/10080902/1/kflaplace.pdf). Apparently, extending the library to such approximations seems out of scope? (lines 209-210)
- I find the interweaved SGMCMC contribution hard to understand and it appears to break the story a bit. Maybe it misses parts of the appendix to be understandable by itself.

### Questions
- Would it be possible to extend the library to non-diagonal parametric posterior approximations in the future?
- Could the authors clarify their SGMCMC contribution in relation to recent prior work on such algorithms?
- How were priors chosen in the experiments on the cold posterior effect? For VI and Laplace, these can have quite a strong effect and I couldn't find any information on that.
- How would Figure 7 look like for the MAP as a comparison? I have seen claims that large models are already fairly well calibrated when looking at confidence itself.

### Soundness
3

### Presentation
3

### Contribution
3
