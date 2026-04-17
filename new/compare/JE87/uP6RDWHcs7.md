# Review

## Summary
This paper introduces Marginal Flow, a density estimation framework that addresses common limitations in existing methods, such as expensive training, slow inference, and architectural constraints. Marginal Flow defines a parametric distribution $q(x|w)$ with latent parameters $w$, which are sampled from a learnable distribution $q_{\theta}(w)$. This approach allows for efficient density evaluation and sampling without directly optimizing the latent variables, making it faster than competing models.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
The paper is well-written, with a clear and logical structure that makes it easy to follow. The experiments are comprehensive, demonstrating the model's performance across various tasks, including synthetic datasets, simulation-based inference, distributions on positive definite matrices, and manifold learning in latent spaces of images.

## Weaknesses
1. The paper lacks theoretical analysis of the marginal distribution $q(x)$, which is crucial for understanding the model's performance. Without this analysis, it's difficult to assess the model's ability to accurately capture the target density or to provide meaningful guarantees about its behavior.

2. The paper does not provide a detailed analysis of the sampling process, particularly regarding the sampling of the latent parameters $w$. A deeper exploration of the sampling algorithm, including its convergence properties and potential pitfalls, would strengthen the paper's contributions and provide a more comprehensive understanding of the model's strengths and limitations.

3. The paper does not discuss the impact of the number of mixtures $N_c$ on the model's performance. Understanding how the choice of $N_c$ affects the model's ability to capture complex distributions is critical for practical applications.

## Questions
1. How is the number of mixtures $N_c$ chosen, and how does it affect the model's performance? Is there an upper limit to $N_c$ before the model's performance degrades?

2. How does the choice of the parametric distribution $q(x|w)$ affect the model's performance? Have other distributions been considered, and how do they compare to the chosen distribution?

3. Could the authors provide a theoretical analysis of the marginal distribution $q(x)$? This would help to better understand the model's ability to capture the target density.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4