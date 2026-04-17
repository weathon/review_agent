# Sensitivity Analysis for Diffusion Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Training a diffusion model approximates a map from a data distribution $\rho$ to the optimal score function $s_t$ for that distribution. Can we differentiate this map? If we could, then we could predict how the score, and ultimately the model's samples, would change under small perturbations to the training set before committing to costly retraining. We give a closed-form procedure for computing this map's directional derivatives, relying only on black-box access to a pre-trained score model and its derivatives with respect to its inputs. We extend this result to estimate the sensitivity of a diffusion model's samples to additive perturbations of its target measure, with runtime comparable to sampling from a diffusion model and computing log-likelihoods along the sample path. Our method is robust to numerical and approximation error, and the resulting sensitivities correlate with changes in an image diffusion model’s samples after retraining and fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This article introduces a method for computing the sensitivity of diffusion models, modeled as the directional derivative of both the learned scores and the generated samples. The computation requires only the pre-trained score estimator and its spatial derivatives. Experiments on synthetic and real-world datasets validate the method's effectiveness.

### Strengths
- **Originality**: The core idea of quantifying the influence of training distribution perturbations is novel.

- **Quality**: The motivation for the sensitivity calculation is well-justified, and the mathematical deductions are both clear and sound.

- **Clarity**: The paper is well-structured; the theory is presented logically and is easy to follow.

- **Significance**: Sensitivity analysis is highly significant for ensuring the safety and reliability of diffusion models in real-world applications.

### Weaknesses
- The practical use of this sensitivity study is not well presented. 
- The numerical comparison is limited. In particular, the baseline method used for the real-world data experiments (from 2013) is outdated and does not convincingly demonstrate superiority over modern techniques.
- Some descriptions are not necessary. For example:
  - Equation 1, which formulates the optimal score, is not strongly correlated with the core content and could be removed to improve focus.
  - In Sec. 3.3, there is no need to illustrate the reason for choosing $\tilde{t}_{1} < t_{1}$ since it is not strongly correlated to the content. 
  - For explaining how to compute the perturbation in score and samples, an algorithm could be more illustrative than just sentences in Sec. 3.3 and Sec. 4.1.

### Questions
- What is the application scenario of this study? How does it apply to the safety of deploying diffusion models?
- In Figure 4, why does the first-order approximation seem to fail for $\bar{\eta} = 10^{-6}$?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work derives closed-form directional derivatives of score functions w.r.t. data distributions given another external distribution, which characterizes certain input sensitivity of diffusion models. The results are also extended to the solution path case (sensitivity ODEs) when sampling (with probability flow ODEs or certain SDEs). This helps to predict changes in model samples after retraining and fine-tuning.

### Strengths
1. The paper is well-written and clearly organized, which is fluent to read. 
2. The sensitivity problem of diffusion models is novel, and quite different from that of general neural networks due to the dynamic formulation of diffusion models. This work takes an initial step towards this direction. 
3. The derived theoretical result is clean and insightful. 
4. Numerical verifications are consistent, with meticulous robustness studies covering main components of the proposed calculation method.

### Weaknesses
1. It would be clearer to add a dedicated algorithm of the proposed calculation method (e.g. Thm. 3.1). 
2. Following 1, how can we compute score functions *accurately* in Eq. (2) in detail? How can we compute probability densities in Eq. (2) *efficiently* beyond neural ODEs? It would be better to include self-contained algorithms regarding them. 
3. Following 1, I also suggest to formulate in-paragraph discussions in Sec. 3.3 as separate algorithms. 
4. For Eq. (1): It is just the score function of $Z_t$, right? What is the meaning of "optimal solution to this problem (score-matching)" (Line 107)?
5. What are the computation and memory complexity of Eq. (2) and Eq. (3) w.r.t. time steps & data dimensions? 
6. It would be more readable to provide self-contained supports in former references for key quantities (at least in appendices), e.g. CCoV, sensitivity equations, and entropic optimal transport (OT) coupling. 
7. Experiments are mainly conducted on MNIST and CelebA datasets. How about the performance on standard Cifar (more diversity) and the efficiency on large-scale ImageNet datasets? 
8. Can authors provide more detailed instructions on how the correlation is calculated in e.g. Fig. 6? 
- Although the proposed calculation method outperforms OT (baseline), it is still far away from 1. What are potential sources of this gap? 
- It is not clear why the correlation is better for larger datasets like CelebA instead of smaller MNIST?

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the differentiablity of the diffusion model training map, that is the map that assigns a score function to a dataset.
An algorithm is proposed, which allow studying the sensitivity of the diffusion model to addition or removal of dataset points (images).
This is then exploited to define sample sensitivity.

### Strengths
* The main strength of the paper is to establish Theorem 3.1.
* Original problem

### Weaknesses
I don't see where the term "sample sensitivity" is clearly defined? I understand that this is the derivative involved in Eq (4) as confirmed by Fig. 10 caption, but this is not clear from the text.

Theorem 3.1 suppose bounding support but all examples use mixture of Gaussians. A comment on that point seems necessary.

**Experiments:**
* Section 4.1: Why use $d=100$ with mixture of isotropic Gaussians. This looks like to poor a model.
* Section 4.2: As explained in Section C.1.2, here the work is done in $d=10$ with a bimodal Gaussian mixture. This is a very simple model to learn with a network. 
* Sections 4.2 and 4.3: Why is it enough to measure the experiment by only requiring better correlations? How is this computation stable since for most images/points the output should be very close to zero?
* Figure 6: The OT baseline is based on empirical OT between samples. This could be replaced by a parameterized transport map trained on the whole dataset (see eg Korotin et al ICLR 2023 and references therein).
* Figure 8: What is displayed in figure 8, 

Minor remarks:
* No hyperlink on cross-references

### Questions
See questions regarding experiments.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Diffusion models and the mathematical objects that define them (the score function and the learned distribution) are functions of training data. This means that, if the model were trained on additional data, or some training example were removed from the training set, these mathematical objects would be different, and hence generated samples would be different. How different would they be? How sensitively does a given diffusion model depend on one of its training examples?

This is the question the authors seek to address in their paper. They formalize the question mathematically in terms of the Frechet derivative of a perturbation map, use this formalization to motivate a specific numerical approach to measuring sensitivity to training data, and then show that their method works in a number of experiments. They also place their work in the context of related machine learning work, e.g., related to influence functions.

### Strengths
Overall, I really like this paper. There is a clear goal---How do we formalize and measure the extent to which a diffusion model depends on small perturbations to its training set?---along with math, an algorithmic approach, and experiments to back it up. The paper is also reasonably well-written and clear throughout. 

Lots of details seem to have been carefully attended to. I also like how the sequence of figures shown slowly build up the ideas of the paper, and gradually show that the method works.

### Weaknesses
I think my overall complaints are minor. One of them is that a lot of space is dedicated to showing that the approach yields sensible results (e.g., Figs. 4-7), but not much space is dedicated to showing what the sensitivity results actually are and what we learn from them. Some SI figures show a bit of this (Figs. 9 and 10); I think the paper would be better if some of these were moved to the main text.

Relatedly, some material currently in the main text could probably be moved to SI. For example, Eq. 1 doesn't seem to be used anywhere, and doesn't say anything interesting. The proof that the method is reasonable is belabored a bit, and maybe some of the details of Sec. 4 could be moved to SI. 

The figures could be improved slightly. Many of the figures have small text or labels, and would be improved by making those things bigger (Figs. 4-7 especially). In Figure 8, it would be helpful to also include (i) the perturbation, and (ii) the images pre-perturbation, as opposed to just the sensitivities and post-perturbation images. It's hard to parse the figure without this extra info.

### Questions
1. Can the authors say more about the 'optimal transport baseline' in the main text? I didn't understand that part.

2. Are there interesting things worth sharing about what we learn from measuring a bunch of sensitivities? Are certain kinds of data points (e.g., outliers) more influential than others (e.g., non-outliers)?

### Soundness
4

### Presentation
3

### Contribution
3
