# DAK-UCB: Diversity-Aware Prompt Routing for LLMs and Generative Models

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
The expansion of generative AI and LLM services underscores the growing need for adaptive mechanisms to select an appropriate available model to respond to a user's prompts. Recent works have proposed offline and online learning formulations to identify the optimal generative AI model for an input prompt, based solely on maximizing prompt-based fidelity evaluation scores, e.g., CLIP-Score in text-to-image generation. However, such fidelity-based selection methods overlook the diversity of generated outputs, and hence, they can fail to address potential diversity shortcomings in the generated responses. In this paper, we introduce the *Diversity-Aware Kernelized Upper Confidence Bound (DAK-UCB)* method as a contextual bandit algorithm for the online selection of generative models with diversity considerations. The proposed DAK-UCB method incorporates both fidelity and diversity-related metrics into the selection process. We design this framework based on prompt-aware diversity score functions that decompose to a two-sample-based expectation over prompt-output pairs in the previous generation rounds. Specifically, we illustrate the application of our framework using joint kernel distance and kernel entropy measures. Our experimental results demonstrate the effectiveness of DAK-UCB in promoting diversity-aware model selection while maintaining fidelity in the generations for a sequence of prompts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose DAK-UCB a diversity-aware bandit algorithm for selecting the best generative model for a sequence of prompts. The main contribution lies in adding diversity scores to the bandit as well as allowing for a mixture of generative models, instead of selecting a single model per prompt. The experiments show that, compared to simple baselines, the proposed approach 1) achieves better diversity-aware scores, 2) is able to discover the most diverse generative model and 3) does not fall for the trap of selecting generative models that increase diversity in images but based on irrelevant prompts.

### Strengths
- Using UCB or a mixture UCB for the selection of a generative model is not new. However, the authors add diversity scores as an additional criterion, which is an original idea. Going beyond the images themselves and incorporating further available information is important in making a more informed choice between generative models.
- The work is grounded in a thorough mathematical argument that is worth sharing with the community and may prompt further research.
- The authors conduct many experiments trying to showcase the strengths of their method in various different contexts.
- The experimental results look promising: Making UCB diversity-aware appears to work as expected, both on (conditional) image generation and text / caption generation tasks.
- The appendix includes ablation studies on the sensitivity to hyperparameter choices, such as the Kernel function.

### Weaknesses
- The motivation is weak. For example, to get diverse data with respect to gender, I can just define an appropriate prior distribution of prompts. Hence, the practical benefit over existing models is not always clear.
- $D_g$ is defined twice (in Equation 8 and in line 312 with different notation). $s_g$ is defined twice as well (in Equation 8 and in line 310 with different notation). In general this part is rather inconsistent in notation. Similarly, first $t$ is defined as a text prompt (line 186) and later as a round (line 309).
- I would have expected a comparison to the already existing, and cited, mixture UCB to show the value of adding diversity scores.
- The presented metrics in Figure 2 are the same ones used for the optimization. Therefore, it is no surprise that the proposed models perform best here. Possibly using additional metrics that are not directly part of the algorithm could strengthen the argument.
- Some implementation details remain unclear. For example, after equation 6 and 7 the prompt $t'$ is suddenly no longer considered.
- The method balances fidelity and diversity but the costs in terms of fidelity have not been directly addressed, e.g., there is no CLIP Score in Figure 2.
- The actual costs associated with using this approach for increasing diversity remain unclear. No running times for the experiments are mentioned.  I expect the exploration phase to incur monetary costs that may not be trivial, depending the number of generative models, the complexity of the prompt, etc. While regret bounds are provided, the actual incurred costs for the experiments would be interesting.

### Questions
- I am not convinced by the practical use case of the method. If we are interested in diversity along a certain dimension, could we not specify an appropriate prior distribution of prompts, e.g., a 50/50 gender distribution, use existing algorithms to select the most appropriate generative model, and then conditionally sample 50% male and 50% female images? What would be the limits of this approach?
- When do I need to rerun the algorithm? For instance, is a re-training required whenever the user prompt or prompt distribution changes? Could any information from the previous model be salvaged for a warm start?
- In Equation 6 and 7, $t'$ is just another prompt drawn from $P_t$. Should this step not appear in the algorithm?
- In Figure 3, it seems unnatural to have models conditioned on cat or dog classes but keep the prompt about a general "animal". This seems rather detached from how the conditioning is applied in practice. Would you expect different results when being more specific in the prompt?
- In Figure 4, why does the mixture DAK-UCB lead to a worse CLIP score than DAK-UCB, should the former not be nested in the latter?
- How to interpret the expert selection ratio in Figure 4? Should we not expect to select each cluster 25% of the time (assuming the cat, dog, car and cake clusters have equal size)?
- Can the model be used in other domains such as protein generation, where diversity has a clear benefit?

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
2

### Summary
The paper proposes DAK-UCB, a contextual-bandit algorithm for the online selection of generative models with diversity considerations. It incorporates fidelity and diversity-related metrics into the selection process. The experimental results demonstrate considerable improvements.

### Strengths
1) The topic of this paper is meaningful. The new method of delicate online model selection is valuable for the community.
2) The experimental results on benchmarks demonstrate considerable improvements.

### Weaknesses
1) I do wander the cost of Mixture-DAK-UCB: it solves quadratic program each round and stores matrices, will this bring scalability concerns? 
2) More details and analysis of the bias evaluation beyond gender is necessary.

### Questions
See the weakness.

### Soundness
3

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
This paper recasts prompt‑conditioned model routing as a contextual bandit that balances generation quality and diversity when choosing among generative models. Building on Upper Confidence Bound, which optimizes only fidelity, the proposed DAK‑UCB jointly estimates prompt fidelity and diversity using separate regressors and an optimism bonus. The method also extends to prompt‑conditioned mixture selection that learns routing probabilities. The authors evaluate DAK‑UCB on the MS‑COCO validation set and on a synthetic dataset created with GPT‑4o to assess its effectiveness.

### Strengths
1. The paper presents a complete and clearly derived theory. It details how KD and RKE are extended to metrics suitable for joint variables and provides the theoretical basis for a diversity‑aware extension of UCB.
2. DAK‑UCB is applied to image generation, text generation, and image captioning, and the experiments validate the approach across these tasks.

### Weaknesses
1. The only major difference from PAK‑UCB is the shift from prompt‑aware selection alone to a joint selection based on both prompt fidelity and diversity. Many other aspects, including the theoretical tools and the choice of backbone models in experiments, overlap substantially, which reduces the perceived novelty.
2. The DAK‑UCB score relies on CLIP text embeddings and DINOv2 image embeddings as the algorithm’s inputs, and the evaluation uses the same embeddings. Prior work indicates that CLIP and DINOv2 can encode biases stemming from imbalances in their training data across geography and race. Using biased embeddings may affect distances between text‑image vector pairs and thereby influence routing, yet the paper does not analyze whether this impacts DAK‑UCB’s outcomes.
3. In this work, fidelity refers only to alignment between an image and its prompt, not to the realism of the image itself. It therefore remains unclear whether DAK‑UCB can select diffusion models that produce more realistic images, for example better FID.

### Questions
See “Weakness”

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the online selection of generative models for given prompts, with explicit consideration of both fidelity and diversity in the generated outputs. 
The existing prompt-aware selection methods only focus on fidelity (e.g., CLIP-Score), often overlooking diversity, which can lead to repetitive or biased outputs. To overcome this, the authors propose a method, Diversity-Aware Kernelized Upper Confidence Bound (DAK-UCB), that extends the kernelized UCB framework by incorporating diversity metrics 
(joint kernel distance (JKD) and joint Rényi kernel entropy (JRKE)) into the selection process to balance fidelity and diversity. The authors also propose a mixture variant (Mixture-DAK-UCB) that allows for prompt-conditioned mixtures of models. The approach is validated on text-to-image and language model generation tasks, showing improvements in both diversity and fidelity metrics compared to baselines.

### Strengths
**The following are the strengths of the paper:**
1. This paper considers the online selection of generative models for given prompts that explicitly consider both fidelity and diversity in the generated outputs, overcoming repetitive or biased outputs.

2. The authors propose a new algorithm, Diversity-Aware Kernelized Upper Confidence Bound (DAK-UCB), that incorporates diversity metrics into the selection process to balance fidelity and diversity, while having theoretical guarantees on the regret upper bound (better bound for Sup-DAK-UCB).

3. The authors empirically validated the proposed algorithm on text-to-image and language model generation tasks, showing improvements in both diversity and fidelity metrics compared to baselines.

### Weaknesses
**The following are the weaknesses of the paper:**
1. The kernel methods and the estimators may lead to high computational overhead (due to kernel matrices), especially when the number of prompts (or models) increases. There should be more details on how the computational cost varies with the increase in samples (number of iterations)

2. It is unclear why the paper only focuses on JKD and JRKE, as there are other diversity metrics.

3. There is not much novelty in the regret bounds, as the derivation depends on existing results, which make many assumptions (normalized kernels, sub-Gaussian noise, RKHS boundedness), which may not hold in all practical settings, especially for LLMs.

4. The diversity-aware selection may degrade fidelity, and there needs to be a discussion about this trade-off. Furthermore, there should be a discussion about the choice of kernel functions, as it may depend on the prompts and models.

### Questions
Please address the weaknesses of the paper.

### Soundness
2

### Presentation
3

### Contribution
2
