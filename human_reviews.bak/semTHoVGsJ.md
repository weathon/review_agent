# Density estimation with LLMs: a geometric investigation of in-context learning trajectories

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Large language models (LLMs) demonstrate remarkable emergent abilities to perform in-context learning across various tasks, including time series forecasting. 
This work investigates LLMs' ability to estimate probability density functions (PDFs) from data observed in-context; 
such density estimation (DE) is a fundamental task underlying many probabilistic modeling problems. 
We leverage the Intensive Principal Component Analysis (InPCA) to visualize and analyze the in-context learning dynamics of LLaMA-2 models. 
Our main finding is that these LLMs all follow similar learning trajectories in a low-dimensional InPCA space, which are distinct from those of traditional density estimation methods like histograms and Gaussian kernel density estimation (KDE). 
We interpret the LLaMA in-context DE process as a KDE with an adaptive kernel width and shape. 
This custom kernel model captures a significant portion of LLaMA's behavior despite having only two parameters. 
We further speculate on why LLaMA's kernel width and shape differs from classical algorithms, providing insights into the mechanism of in-context probabilistic reasoning in LLMs.
Our codebase, along with a 3D visualization of an LLM's in-context learning trajectory, is publicly available at https://github.com/AntonioLiu97/LLMICL_inPCA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper titled “Density Estimation with LLMs: A Geometric Investigation of In-Context Learning Trajectories” investigates the ability of large language models (LLMs), specifically LLaMA-2 models, to perform density estimation (DE) from in-context data. Density estimation is crucial for probabilistic modeling, where the goal is to estimate a probability density function (PDF) based on observed data. The authors leverage Intensive Principal Component Analysis (InPCA) to visualize and analyze the in-context learning dynamics of these models, showing that LLaMA-2 follows distinct low-dimensional trajectories in probability space when compared to traditional methods such as histograms and Gaussian kernel density estimation (KDE). A key insight from the paper is that LLaMA-2’s in-context DE process can be interpreted as an adaptive KDE with a dynamic kernel shape and width, which captures the model’s behavior with high precision. This adaptive KDE model, with only two parameters, successfully explains the geometric features of the model’s DE trajectories, offering a deeper understanding of how LLMs perform probabilistic reasoning in-context. The paper also speculates that LLaMA-2 may use a mechanism similar to the dispersive induction head to adjust predictions based on observed data.

### Strengths
Originality: The paper contributes a novel approach to understanding the in-context learning capabilities of large language models (LLMs) by framing their density estimation (DE) processes as an adaptive form of kernel density estimation (KDE). This is an innovative application of KDE in a context that diverges from traditional statistical methods. The use of Intensive Principal Component Analysis (InPCA) to visualize and interpret learning trajectories in probability space adds further originality, combining ideas from both geometric analysis and machine learning in a fresh way. The concept of interpreting LLaMA’s learning trajectory as a kernel-like algorithm, with an adaptive kernel, is particularly creative and represents an advancement in understanding how LLMs process probabilistic information.

Quality: The research is rigorous and well-supported by both theoretical and empirical analyses. The application of InPCA, along with detailed comparisons between LLaMA’s trajectories and traditional DE methods like histograms and Gaussian KDE, demonstrates the authors’ thoroughness in exploring the model’s behavior. The paper is grounded in solid mathematical foundations, using well-established statistical measures such as the Hellinger distance to quantify and compare learning trajectories. The methodology is well-executed, and the visualizations (e.g., 2D embeddings of DE trajectories) are informative and enhance the quality of the research.

Clarity: The paper is generally clear in its explanations of complex concepts, especially given the technical nature of the topic. The authors effectively explain the connection between KDE and LLMs’ in-context learning, providing sufficient background for the reader to follow the argument. The figures and visualizations, such as those illustrating LLaMA’s DE process and its comparison with classical methods, are well-designed and help clarify the main findings. However, some sections, particularly around the technical details of InPCA and bespoke KDE modeling, may require more careful reading by less specialized audiences.

Significance: The significance of the work lies in its contribution to the growing field of in-context learning for LLMs. By providing a new geometric and probabilistic interpretation of LLaMA’s behavior, the paper has the potential to impact future research on both LLMs and kernel-based methods. The idea of modeling LLMs’ probabilistic reasoning with an adaptive kernel framework offers a new lens through which to study these models, which could inform the development of more efficient or interpretable LLMs in various domains. Moreover, the paper opens avenues for future exploration, such as investigating the role of dispersive induction heads in continuous probabilistic learning, making it a significant contribution to both theoretical and practical applications in AI.

### Weaknesses
While the paper “Density Estimation with LLMs: A Geometric Investigation of In-Context Learning Trajectories” offers a strong contribution, there are several areas where it could be improved. 

1. Insufficient exploration of broader datasets: The paper focuses primarily on Gaussian and uniform distributions for testing LLaMA’s in-context density estimation (DE) capabilities. While these distributions are useful for illustrating the model’s ability to perform DE, they are relatively simple and may not fully capture the complexity of real-world probabilistic learning tasks. Incorporating more complex and diverse target distributions, such as multimodal or skewed distributions, would significantly strengthen the generalizability of the findings. By broadening the scope of the experiments, the authors could better demonstrate the robustness of LLaMA’s DE across a wider variety of data structures.

2. Incomplete explanation of adaptive kernel choices: The core contribution of the paper is the interpretation of LLaMA’s in-context DE process as an adaptive KDE, but the reasoning behind the specific choices of kernel shapes and widths, and their relationship to LLaMA’s behavior, could be made clearer. For instance, while the authors introduce a flexible kernel with a shape parameter, the empirical results seem to focus primarily on the Gaussian kernel and only touch briefly on other possibilities. Expanding this section to include more detailed justifications for the selection of kernel parameters and providing more comparative results with different kernel shapes would offer deeper insights into why LLaMA behaves as it does.

3. Theoretical depth versus empirical validation: The paper proposes several intriguing theoretical concepts, such as the idea of dispersive induction heads, but these remain speculative without much empirical validation. The authors propose that LLaMA employs a dispersive induction mechanism to handle in-context density estimation, yet this idea is not thoroughly tested or backed up with concrete evidence. To improve the work, the authors could design specific experiments or ablation studies to directly probe the existence of this mechanism, or alternatively, provide more substantial theoretical evidence. This would move the idea from speculation to a more solid theoretical contribution.

4. Comparative analysis with more models and baselines: The paper primarily compares LLaMA’s performance against classical KDE and Bayesian histograms. However, it would benefit from including a broader set of baseline models, including other transformer-based LLMs or models specifically designed for probabilistic reasoning. For instance, comparing LLaMA’s performance against other architectures known for in-context learning abilities could provide more insight into how LLaMA’s density estimation compares to other state-of-the-art models. This would make the experimental results more compelling and allow for a better understanding of LLaMA’s unique contributions.

### Questions
1. Clarification on Kernel Selection and Adaptation: How were the kernel shapes and bandwidth schedules chosen, and why were they deemed the most appropriate for LLaMA’s in-context learning trajectories? Did you experiment with non-Gaussian kernel shapes beyond those briefly mentioned?

2. Scope of Data Distributions: Why did you primarily focus on Gaussian and uniform target distributions? How might LLaMA’s in-context DE capabilities generalize to more complex, real-world distributions, such as multimodal or skewed distributions?

3. Empirical Validation of Theoretical Concepts:How can you empirically validate the proposed concept of “dispersive induction heads”? What experiments would directly test this idea beyond the adaptive KDE model?

4. Comparison with Other Transformer Models: How does LLaMA-2’s performance in in-context DE compare to other large language models  or other architectures known for probabilistic reasoning?

5. Generalization Beyond DE Tasks: How do you see the techniques and findings in this paper extending to tasks beyond density estimation, such as other forms of probabilistic reasoning or inference? Could these methods be generalized to conditional density estimation or more complex probabilistic models?

6. Impact of Tokenization on DE Process: How does the tokenization process impact the DE process in LLaMA? Could different tokenization schemes affect the observed DE trajectories or results?

7. Role of Model Size: How does the model size (7B, 13B, 70B) impact the in-context density estimation process? Are there any trends or differences in the DE trajectories of these models, and do larger models perform better in this context?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work explores how LLMs infer probability density functions, derived from in-context observations. It uses a variant of Principal Component Analysis to help visualise the learning dynamics. Llama-2 models are chosen for the experiments, and these are fed data from target distributions which are either uniform or Gaussian. The results are found to be comparable to certain forms of kernel density estimation.

### Strengths
The paper is well written and clearly presented.

The approach is novel, and the topic is of high interest.

There is a large number of supportive experiments and these are presented with well chosen visualisations.

### Weaknesses
I have a few concerns regarding the scope of the experiments:

- Models
The empirical results would be significantly more compelling if they weren't all using the Llama-2 family of models, as then the results could potentially be claimed to apply more generally to LLMs. As it stands, the title of the paper may be considered slightly misleading since conclusions can only be drawn on this specific family of models, and not LLMs in general.

- Distances
When visualising the results, it would be beneficial not to solely rely upon the Hellinger distance but to also show results derived from other distances such as the KL divergence or Wasserstein distance. Even if this does not lead to such compelling visualisations, this would provide a much more complete picture of how well the predicted distributions align with the targets.

- Distributions
The chosen target probability distributions are very simple (Gaussian & uniform) so this also limits the conclusions the reader can draw. How would the LLM respond to data from a bimodal distribution, for example?

- Uncertainties
In order to establish a robust, reproducible set of empirical results, it would be strongly advisable to assign uncertainty estimates to the key quantities involved, such as the bespoke KDE parameters.

### Questions
While my main concerns relate to the empirical results, as outlined in the 'weaknesses' section, I have some smaller questions below:

> a uniform distribution over the domain (0, 100), showing a state of neutral ignorance, which is a reasonable Bayesian prior
I'm not so sure that is a reasonable Bayesian prior. To quote David Mackay's book, "ignorance does not correspond to a uniform probability distribution". I would be surprised if Llama has a priori a uniform prior over the digits, since 1 is much more often the leading digit than others. 
As David MacKay argues in his book on Information Theory, "ignorance does not correspond to a uniform probability distribution". Given that lower digits appear more frequently as leading digits (in line with Benford's Law), it would be surprising if LLMs such as LLaMA  adopted a uniform prior over digits without explicit conditioning. Could this be the reason we can see in Figures 2 and 3 that the Llama trajectory starts some distance from the grey point marked as 'ignorance'?

> While LLaMA-2 has a context window of 4096 tokens (equivalent to ∼1365 comma-delimited, 2-digit data points), we limit our
analysis to a context length of n = 200.
Could you clarify here whether n=200 relates to max number of tokens or data points? (I would interpret this sentence to imply n = 200 numbers, except I notice Figure 1 seems to illustrate 200 datapoints which would correspond to many more than 200 tokens)

The paper frequently refers to the target function as a probability density function (PDF). However, since the outputs are discrete values, perhaps it would be more accurate to describe this function as a probability mass function (PMF)?

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
This paper investigates the behavioral characteristics of LLMs (more specifically, LLaMA-2) in performing probability density estimation from in-context by using InPCA. It finds that compared to traditional KDE and Bayesian histogram estimation, the proposed KDE with adaptive kernel width and shape resembles their behavior more closely.

### Strengths
1.	Probability density estimation is a classic and important problem in statistics and this paper explores the underlying mechanism of utilizing LLMs' in-context capabilities for probability density estimation, which is novel to me.

2.	The authors designed various experiments that are easy to understand, providing support for their viewpoints and interesting findings regarding the potential patterns of density estimation in LLMs.

3.	The paper has a clear structure and is easy to follow.

### Weaknesses
1.	Although using LLMs for probability density estimation is relatively novel compared to traditional methods, the scope of the discussion in the paper is somewhat limited (for example, it focuses on the one-dimensional case where continuous probability densities are discretized into 2-digit numbers). It would be interesting to explore whether LLMs can still make effective predictions and the analysis results remain consistent in more complex scenarios.

2.	When estimating more complex distributions, such as those with higher curvature (l = 0.02) as shown in Figures 32 - 34, the context length may limit the model's estimation performance. It would be worth investigating whether increasing the context length would lead to better estimation results.

### Questions
1.	It is observed in part b of Figure 2 that, for both n = 10 and n = 50, the estimated density functions appear to have an upward bias at the edges (when x is close to 0 and 10). Could the authors provide some explanation for this?

By the way, it can be seen that when the context length is limited in Figure 2 b, the LLMs' estimation of the Gaussian distribution seems to resemble the scenario where the proposed bespoke KDE method has a smaller s (depicted in Figure 5, s = 1 for example). Therefore, as seen in Figure 13, the estimation using the exponential KDE appears to be closer to LLaMA's prediction pattern than that using the Gaussian KDE. This might also provide us with some insights.

2.	When using Bayesian histogram to estimate a Gaussian distribution, the authors propose that its slow convergence is 'likely due to the strong influence of its uniform prior'. However, when estimating a uniform distribution, it does not seem to converge faster either. Therefore, this explanation is somewhat contradictory. Furthermore, what would happen if \alpha is not set to 1 (for example, if \alpha = 0)?

Typos:
It seems like there is a missing citation (squared exponential kernel (?)) in line 1253.

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
3

### Summary
This paper is looking at the classical problem of density estimation with the use of large language models. The question of interest here is how the large language model estimates a density by relying on the data accessible to it "in context". A simple experiment is used to validate the proposal, namely, checking whether presenting an LLM (in this case, the LLaMA-2) with a sequence of points sampled iid from an underlying distribution results in the *next* sampled point eventually converge to come from the true distribution (with the increase of the number of observed data points).

This, of course, brings the question of how exactly does convergence of this process occur. The authors propose looking at low dimensional characteristics of the density estimate trajectories. Heuristically, at least, it appears that the LLM performs a kind of adaptive Gaussian estimation.

### Strengths
I think the question is a nice one and this is certainly (at least to me) an interesting paper on density estimation. It is not obvious a priori that the LLM will be able to do density estimation as it does. The trajectory analysis is certainly interesting. I would think this paper can allow us to do density estimation in automated ways. The experiments are well-designed for the most part.

### Weaknesses
I wonder what properties of the density are important here. I don't believe checking on a Uniform is sufficient - uniform densities are easy to estimate with a flexible kernel and indeed that is what the experiments show, this is not surprising. What is more of interest is to try a heavy-tailed example like a Cauchy or a t with a small number of degrees of freedom.

### Questions
What about even the bivariate normal case? The biggest issue with kernel density estimation methods is scaling with dimensionality. The refinement scheme seems sensible and easily adapted to the bivariate case, so would it be possible to run this scheme on bivariate normal data to see how well learning happens there. How are the correlations learned, for example?

### Soundness
3

### Presentation
4

### Contribution
3
