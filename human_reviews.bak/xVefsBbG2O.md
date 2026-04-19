# Diffusion Models are Evolutionary Algorithms

- Decision: Accept (Poster)
- Scores: 8, 6, 5, 8

## Abstract
In a convergence of machine learning and biology, we reveal that diffusion models are evolutionary algorithms. By considering evolution as a denoising process and reversed evolution as diffusion, we mathematically demonstrate that diffusion models inherently perform evolutionary algorithms, naturally encompassing selection, mutation, and reproductive isolation. Building on this equivalence, we propose the Diffusion Evolution method: an evolutionary algorithm utilizing iterative denoising -- as originally introduced in the context of diffusion models -- to heuristically refine solutions in parameter spaces. Unlike traditional approaches, Diffusion Evolution efficiently identifies multiple optimal solutions and outperforms prominent mainstream evolutionary algorithms. Furthermore, leveraging advanced concepts from diffusion models, namely latent space diffusion and accelerated sampling, we introduce Latent Space Diffusion Evolution, which finds solutions for evolutionary tasks in high-dimensional complex parameter space while significantly reducing computational steps. This parallel between diffusion and evolution not only bridges two different fields but also opens new avenues for mutual enhancement, raising questions about open-ended evolution and potentially utilizing non-Gaussian or discrete diffusion models in the context of Diffusion Evolution.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper explores the theoretical and practical parallels between diffusion models and evolutionary algorithms, proposing the "Diffusion Evolution" approach, which adapts diffusion models for evolutionary tasks. This new method, Latent Space Diffusion Evolution, enhances the ability to find diverse and optimal solutions in high-dimensional parameter spaces, particularly within reinforcement learning.

### Strengths
he core idea of treating diffusion models as evolutionary algorithms is innovative. It extends diffusion models to broader applications by framing them as tools for evolutionary tasks, potentially contributing new knowledge to both evolutionary biology and AI communities.

Technically: The paper provides thorough mathematical grounding for the equivalence between diffusion and evolution, specifically explaining diffusion as a probabilistic denoising process analogous to evolutionary mechanisms like mutation and selection.

Experimentals: Comprehensive experiments benchmark Diffusion Evolution against traditional algorithms (e.g., CMA-ES, PEPG) ascross various fitness landscapes, demonstrating its strength in maintaining diversity and achieving multiple optima.

Applications in High-Dimensional Problems: The adaptation of Diffusion Evolution to high-dimensional tasks via latent space diffusion showcases its practical viability in reinforcement learning contexts, with empirical results suggesting its potential in complex environments.

### Weaknesses
Complexity of Explanation: The theoretical connections between diffusion and evolution, while compelling, are highly complex, and the paper occasionally sacrifices clarity for depth. This might limit accessibility to a broader audience.

Although promising, the methodology may have limitations in scenarios requiring open-ended evolution, a challenge acknowledged briefly. More thorough discussion could help set realistic expectations for potential users.

While the Diffusion Evolution algorithm demonstrates superiority in finding diverse solutions, certain comparisons (e.g., high-fitness solutions) against traditional evolutionary algorithms lack statistical significance or detailed analysis of computational cost versus benefit, particularly in high-dimensional settings.

### Questions
The work potentially contributes new insights and methods for both diffusion model and evolutionary algorithm communities, even if the practical impact in specific real-world applications remains to be validated. Can the authors address this?

Can statistical tests be added?

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
4

### Summary
This manuscript introduces an innovative perspective by interpreting the diffusion model as an evolutionary algorithm, highlighting the mathematical similarities between the diffusion and evolutionary processes. It proposes a Diffusion Evolution Method that employs iterative denoising to heuristically optimize solutions within the parameter space, drawing an analogy between the generative process of the diffusion model and the selection and mutation mechanisms in biological evolution.

### Strengths
The idea of regrading the diffusion model as an evolutionary algorithm is interesting.

### Weaknesses
1. The mathematical demonstration process in this manuscript is insufficient. The authors conceptualize the evolution process as a transformation of the probability density function, deriving the diffusion evolution algorithm from the Bayesian formula. However, this derivation overlooks key complexities inherent in evolutionary dynamics, such as genetic drift and gene recombination. Additionally, the application of the Bayesian formula assumes conditional independence, which may not be held in the context of evolutionary processes due to the interactions and competition among individuals. Furthermore, the manuscript does not provide a clear explanation of how to effectively map a complex fitness function into a probability density function, which is essential for the practical implementation of the proposed model.
2. Since the author claims that the diffusion model functions as an evolutionary algorithm, the experimental section must reflect this comparison appropriately. Specifically, in scenarios involving multiple tasks and comparisons, aspects such as speed, performance metrics (including best, worst, median, and average results), and the stability of performance should be examined. Additionally, the model's excellence should be illustrated through the mean variance of the results. Currently, the scope of experiments conducted is insufficient and does not adequately support the claims made in the manuscript.
3. There are several grammatical errors and unclear expressions throughout the article. For example, the definitions of certain terms lack clarity, the derivation process for the formulas is inadequately explained, and the labeling of the charts is inaccurate.
4. The authors categorize diffusion models as evolutionary algorithms on the basis that both methods perform distribution transformation. However, this classification may be overly broad. For instance, semantic segmentation models also involve transforming distributions—from real images to pixel-level segmentation results. Should we, therefore, classify these segmentation models as evolutionary algorithms as well? This comparison seems inaccurate. A more robust argument would establish that the Markov process within a non-equilibrium thermodynamics framework (diffusion) function as an unconstrained parameter optimization technique (evolution). If this argument cannot be substantiated, I recommend revising the title to better reflect the content.
5. The manuscript presents two primary sets of experiments: multi-objective evolution and latent space diffusion evolution. However, the multi-objective evolution experiments are limited to a two-dimensional parameter space and a simplistic fitness function, which fails to demonstrate the proposed approach's efficacy in high-dimensional parameter spaces or with more complex fitness functions. Similarly, the latent space diffusion evolution experiments focus solely on the CartPole task, lacking validation across a broader range of reinforcement learning tasks. Furthermore, the experimental results do not include any statistical significance tests, making it challenging to determine whether the observed improvements in algorithm performance are statistically significant. This omission significantly undermines the credibility of the findings presented.

### Questions
In the experiments section, the authors select three different evolutionary strategies for performance comparison. However, the rationale for choosing these specific methods is not clearly articulated. I would appreciate more insight into this selection process, particularly since each of these methods was introduced over five years ago.

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
This paper proposes a new evolutionary algorithm inspired by diffusion models. It views the evolution processes as the denoising process of diffusion models, and designs an evolutionary algorithm named Diffusion Evolution, which is based on the denoising framework of the famous diffusion model DDIM. During evolution, it takes each individual in the population as a solution in the denoising process, and updates it with similar updating rules in DDIM. To facilitate better performance, it further follows previous works to optimize in the latent space. Experiments on several benchmark functions and a simple cart-pole controlling problem show that compared with classic baseline methods include CMA-ES, OpenES and PEPG, the proposed Diffusion Evolution can obtain more diverse solutions with good fitness.

### Strengths
The paper is well-written and easy to follow. The proposed method Diffusion Evolution is well-motivated, with clear explanation and illustration.

### Weaknesses
As there has been a variety of evolutionary algorithms with various inspirations, the most important issue when proposing a new method should be clarifying its strength compared with previous methods, from the perspectives of theoretical analysis and extensive experiments. However, the proposed Diffusion Evolution in this paper is lack of theoretical analysis. Meanwhile, the experiments are much too simple. For example, only five synthetic functions and a simple cart-pole controlling problem are included. As the diversity of solutions seems to be the strength of the proposed Diffusion Evolution, except for methods like CMA-ES, which are not specified for diversity, comparison with other kinds of evolutionary algorithms like Quality-Diversity (QD) are necessary.

### Questions
As mentioned in the appendix, different Alpha and noise schedule settings are tested, what about the detailed experimental results? Is there any analysis about the results of different settings, which could be useful for the users?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors try to connect Diffusion Models and Evolutionary computation by arguing that both processes do iterative refinements through an update rule plus some perturbation. In evolutionary algorithms the update comes in the form of natural selection, in diffusion is the denoising phase. The perturbation corresponds to mutation for evolution, and the diffusion phase for diffusion models. 

From the above connection an evolutionary algorithm that performs in its iteration a diffusion process is proposed.  Instead of aiming to recover some distribution of the data, the goal is to turn the random initial population points towards a distribution centered around the optimized function optimum.

### Strengths
By connecting methods from different fields the work proposed an interesting new evolutionary algorithm that can be further improved using variations and techniques found in diffusion models.

The text is very clear and the math derivation from diffusion to an EA algorithm is easy to follow. Figure 1 really helps to visualize how the population of the diffusion EA spreads to the higher fitness regions.

### Weaknesses
Maybe I missed it but there is no discussion on how to choose the mapping to a density function g() or which g() was used for the experiments. It seems is important for the search to work properly to have a mapping that makes it clear which points should be paid more attention by assigning to them bigger weights. Maybe there is some connection to common selection strategies in evolutionary algorithms that could be discussed.

### Questions
In the text is not mention what g could be or some options. It could be like the utility function used in Natural Evolutionary Strategies? because it seems to assign more weight/importance to high ranking solutions.

I think it would be very interesting in a  future work to explore more the latent diffusion version of the algorithm, and techniques to understand which parameters could be ignored or which parameters should be perturbed together.

### Soundness
3

### Presentation
3

### Contribution
3
