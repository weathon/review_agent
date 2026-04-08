## Human Reviewer 1

### Summary
This paper proposes a new method to sample from tempered distribution of pretrained diffusion models.  The key method relies on re-scaling the score function with a factor depending on the variance of the data distribution and the temperature factor. This scaling is motivated from the mixture-of-gaussian data assumptions, and are empirically validated on real-world data. The method is benchmarked against a variety of tasks including synthetic 2D examples, text-to-image generation, protein generation, depth estimation, pose prediction. and robotic manipulation.

### Strengths
- The proposed method provides a training-free, plug-and-play inference-time method for tempered sampling in diffusion models (although in somewhat heuristic way). It is simple to implement, as it only involves rescaling the score function in certain ways.

- In text-to-image generation tasks, the performance is improved with k slightly smaller than 1, and in robotic manipulation task, the performance is improved with k larger than 1 for most problems, when compared to the base sampling method.

### Weaknesses
- While the proposed method is motivated by closed-form derivations under Gaussian assumptions, it remains a heuristic approach in the general case.

- The juxtaposition with the Mixture of Gaussian case in Section 3.2 should be clarified in the main text. This case represents a more realistic approximation than the single Gaussian scenario. However, the current argument in the main text feels somewhat hand-wavy. In particular, it is unclear how the proposed reasoning applies when t lies in the mid-range of [0,1], since the arguments at the extreme points do not extend naturally to this region. I also found the structure in the relevant appendix section difficult to follow.

- The presentation in Section 5.3 on depth estimation could be improved for clarity. Please refer to my detailed comments in the “Questions” section below.

- The empirical results appear mixed. For instance, the proposed method consistently underperforms CNS in the pose prediction task. In Figure 6, the conclusions are also not straightforward: the best FID is achieved by CNS under certain settings, while the best designability is obtained by the proposed method under others. I also find the linear trend fits for CNS and the proposed method confusing; it is unclear how fitting a single linear line across results from different hyperparameter settings is justified.

- It is difficult to assess the statistical significance of the results, as standard errors are not reported in the tables and error bars are missing from the plots.

### Questions
- Can you explain more on the interpretation of $\sigma$ in the first paragraph of Section 4.2? In particular, as $\sigma$ corresponds to the variance of the data distribution, why is it that it "indicates how early we want to steer the sampling process"? In the same paragraph, I also found the term "initial Gaussian distribution" confusing. While from the previous context I understand that this refers to the data distribution under gaussian assumption, it can be overloaded with the initial gaussian of the reverse diffusion process.

- Can you explain the task in Sec 5.3 depth estimation, how diffusion model is used and why a tempered sampling approach would be favorable? Also can you provide more context and interpretation of Figure 7?

- Why is it that CNS is only benchmarked against in some of the tasks, and not others. For example, results for CNS are not included in Figure 4,5,7 and Table 3. 

- The following paper may be a good reference for tempered sampling and baseline to compare to. 

Skreta, Marta, et al. "Feynman-kac correctors in diffusion: Annealing, guidance, and product of experts." arXiv preprint arXiv:2503.02819 (2025).

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper proposes Temporal Score Rescaling (TSR), a method designed to steer the distribution of diffusion and flow models during inference.
Compared to Constant Noise Scaling (CNS), TSR is less prone to mode collapse.
Under relatively strong assumptions, TSR provides bounded error guarantees with respect to the ideal Gaussian mixture distribution, whereas CNS fails to achieve true temperature scaling.
The authors validate the superiority of TSR over CNS through experiments on synthetic data and conduct extensive evaluations on real-world datasets.
Experimental results show that across various quality–diversity trade-off scenarios, TSR consistently achieves a better Pareto frontier than CNS.

### Strengths
- Bounded error: Under independence and well-separatedness assumptions, the temperature sampling estimated by TSR has exponential and polynomial error bounds, which vanish with the score estimation error.

- Extensive experiments: TSR’s effectiveness is clearly demonstrated on synthetic data, and its superiority in achieving a better Pareto frontier is demonstrated on real-world datasets. The experiments span multiple domains and data modalities, confirming the general applicability of the method.

- Clear presentation.

- The idea is simple yet effective, with strong potential for broad applications.

### Weaknesses
- Mode collapse: Although TSR claims to avoid mode collapse under strong assumptions (e.g., Gaussian mixture distributions with well-separated components), this claim lacks solid empirical support on high-dimensional real-world data. The paper provides no direct quantitative verification of mode preservation.

- Sigma: In the theoretical derivation, σ originates from the variance of the underlying data distribution; however, in real-world experiments, it becomes a tunable hyperparameter. This deviation may weaken the theoretical interpretability of TSR and increases the tuning overhead across different tasks.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper study a test-time sampling approach for diffusion models, creating a sharper or broder distribution than the one at training to sample data from. Particularly, they propose to apply temperature sampling technique to diffusion model to control the diversity of the sampled output by rescaling the learned score functions. The method demonstrates the applications to both stochastic and deterministic samplers and validates the performances on 5 different tasks, such as image generation, pose estimation, depth prediction, robot manipulation, and protein design.

### Strengths
1. Though the idea of applying temperature scaling is not new, the authors propose an elegant way to effectively alter the learned score function without stripping away the modes and structure of data. The main idea is to rescale the variance of intermediate variables of diffusion model while preserving the mean (modes).
2. The exposition is easy to follow.

### Weaknesses
1. Since this method introduces additional two parameters $(k, \sigma)$ for tuning, these are varied between tasks. It is better to have a conclusion on the range of them per task. 
2. There is no mention of a systematic way for choosing an optimal parameter set. This is extremely important for test-time scaling techniques these days.



- L056: "For example, ... we want the more likely estimate(s) as output". "more likely estimates(s)" is not very clear, it is different than what model have been trained on?

### Questions
Q1. Look at Figure 10, the method seems not bring any performance gain (even worse), expecially for small inference steps (< 100). Any rationale behind this? 

Q2. Will the findings of optimal $(k, \sigma)$ be hold for different model version like StableDiffusion vs Flux? 

Q3. I wonder if the finding is applicable to video domains since video is more dynamic and complex. This could be an useful test.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 4

### Summary
The authors consider the problem of temperature rescaling. That is, given a diffusion model (or equivalently a flow matching model) sampling the density $p$ they aim to sample from $p_k(x)\propto \exp(1/k \log p(x))$. To this end, they rescale the score functions by a time-dependent factor, where the time-dependence is derived from the isotropic Gaussian case. The method is tested on a wide range of applications.

### Strengths
The idea is very simple, clearly presented and the paper includes experiments for plenty of applications.

### Weaknesses
There is no justification why the proposed method should make sense despite the trivial case of an isotropic Gaussian (or seperated Gaussian mixtures, which are quite the same). More generally, I am quite sure that the generated distribution with TSR does not coincide with the target distribution $p_k$ whenever $p$ is not an isotropic Gaussian. The paper does not include any analysis or bound of this error. The claim that the method has the intended effects is purely empirical.

The rest of the paper is a tour through different applications, each of them using a well-tuned diffusion model taken from the literature. While slight adaptions can lead to minor improvements, it reduces the image quality drastically for larger adaptions. Most likely, the main reason is that changing the temperature on image datasets is not really a sensible task.

In summary, the paper consists out of a small implementation trick and I wouldn't consider the methological improvement over prior work as high enough.

### Questions
I have a couple of questions regarding the experiments:

- In the toy experiments, where the authors compare CNS with TSR using the same scaling parameter $k$. Is there any reason to believe that the scaling parameter $k$ in CNS corresponds to the $k$ in TSR (which corresponds to the distribution $\propto \exp(1/k \log p(x))$)? 

- The example from Fig. 2 is missing descriptions. How exactly is TRS used for conditional sampling? Or is it taking already the conditional model and adjusting the conditional scores? In this case: Why is it relevant that it has been a conditional distribution in the first place?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
4