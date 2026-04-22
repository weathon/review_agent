# Identifiable Interpretation in Generative Models via Causal Minimality

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 2, 6

## Abstract
Deep generative models, while revolutionizing fields like image and text generation, largely operate as opaque ``black boxes'', hindering human understanding, control, and alignment. While methods like sparse autoencoders (SAEs) show remarkable empirical success, they often lack theoretical guarantees, risking subjective insights. Our primary objective is to establish a principled foundation for interpretable generative models. We demonstrate that the principle of causal minimality -- favoring the simplest causal explanation -- can endow the latent representations of diffusion vision and autoregressive language models with clear causal interpretation and robust, component-wise identifiable control. We introduce a novel theoretical framework for hierarchical selection models, where higher-level concepts emerge from the constrained composition of lower-level variables, better capturing the complex dependencies in data generation. Under theoretically derived minimality conditions (manifesting as sparsity or compression constraints), we show that learned representations can be equivalent to the true latent variables of the data-generating process. Empirically, applying these constraints to leading generative models allows us to extract their innate hierarchical concept graphs, offering fresh insights into their internal knowledge organization. Furthermore, these causally grounded concepts serve as levers for fine-grained model steering, paving the way for transparent, reliable systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the authors introduce a theoretical framework for hierarchical selection models, where higher-level concepts emerge from structured combinations of lower-level variables to capture complex dependencies in data generation. The authors derive minimality conditions—expressed as sparsity or compression constraints, under which the learned representations align with the true latents. Empirically, the authors demonstrate that applying these constraints to Stable Diffusion v1.4 using SAEs uncovers their inherent hierarchical concept graphs, providing insights into their internal organization and enabling causally grounded, fine-grained model control. Authors demonstrate the effectiveness of their method on unlearning, fine-grained image editing, and controllable image generation tasks.

### Strengths
* To my knowledge, the paper introduces a novel and interesting way of obtaining hierarchical concept graphs using SAEs. Such hierarchical graphs capture the progression of low -> high level concepts. The connection of this modeling and how progressively higher-level concepts emerge during sampling process of diffusion models is complementary to our current understanding of diffusion models [1,2,3].
* The manuscript is well written and contextualized well for the most part.
* The authors provide a theoretical justification on why hierarchical SAEs might be able to capture the true latent variables of the data-generation process, which makes the results more grounded. I would like to note that I am not familiar with hierarchical models. Therefore, I could not verify the proofs of theoretical claims (hence the lower confidence).

### Weaknesses
* Some empirical implementation details are missing. These include annotating discovered concepts by SAE, implementation of SAE w/o hierarchy, etc. (see questions section).  
* Discussion about several related works are missing. In particular, the evolution of high- and low-level semantics through different stages of the diffusion process has been observed and discussed in [1,2,3]. 
* Empirical evidences are based on a single model (SDv1.4). There should at least be a discussion on why authors expect their method to work for other models and possible architectures (such as DiT rather than U-Net).
* Please see the questions below as well.

### Questions
***Questions and Suggestions:***
* I would recommend adding a section (possibly in the Appendix) on how the causal discovery algorithm works in the context of SAEs. 
* How does the SAE baseline (no hierarchy) work/implemented?
* Related to this: how do you annotate the concepts discovered by SAEs?
* How do you combine features from different transformer blocks? Do you train a single SAE or separate SAEs but the discovered concepts are matched across layers with the “Causal Discovery Process”?
* Is 10k prompts sufficient for training SAEs? Several related work [2,4] use significantly higher number of prompts.
* How were the distinct timesteps 899, 500, 100 selected?
* What are the sample sizes for the numerical results provided in the manuscript (Tables 1,2,3)?

***
***References:***

[1] Patashnik, Or, et al. "Localizing object-level shape variations with text-to-image diffusion models." Proceedings of the IEEE/CVF international conference on computer vision. 2023.

[2] Tinaz, Berk, Zalan Fabian, and Mahdi Soltanolkotabi. "Emergence and Evolution of Interpretable Concepts in Diffusion Models." arXiv preprint arXiv:2504.15473 (2025).

[3] Mahajan, Shweta, et al. "Prompting hard or hardly prompting: Prompt inversion for text-to-image diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[4] Surkov, Viacheslav, et al. "One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models." arXiv preprint arXiv:2410.22366 (2024).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the interpretability of generative models by identifying and steering latent features that lead to specific prediction outcomes. The intention is to differentiate from existing methods, such as sparse autoencoder based approaches, by adopting a theoretical framework. It mainly leverages concept decomposition/aggregation to achieve this goal.

### Strengths
The paper applies causal minimality principles to the interpretability of complex generative models.

### Weaknesses
1. The hierarchical model, $P(V_{l+1} | V_l)$ defined around line 179 is oversimplified as a higher level concept is derived not only from the lower level features, but also from similar features in other datasets, e.g., through data labelling. Without considering that concepts are derived from multiple objects and are often context sensitive, reducing generative models to simple hierarchical structures does provide useful interpretability. 

2. Some key terms, causal minimality, concept sparsity, parsimony, and identifiability, are used without clear definitions or an explanation of how they relate to one another.

3. The experiment setting and evaluation objectives are not clearly described. It is unclear why unlearning baselines are chosen for comparison.

4. Decomposition based interpretability has a history. The authors may want to dig out early work such as [1] to understand its limitations in complex tasks.

References:

[1] Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C. and Su, J.K., 2019. This looks like that: deep learning for interpretable image recognition. Advances in neural information processing systems, 32.

### Questions
see weaknesses

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a theoretical hierarchical selection model for generative models where higher level concepts would emerge from constrained combinations of lower level concepts. Based on their theory, they use sparse autoencoders trained at different timesteps in the diffusion process and a causal discovery algorithm to construct a hierarchical concept graph. With this, they perform concept steering to enable downstream usecases like compositional editing and model unlearning.

### Strengths
* The paper focuses on finding hierarchical concepts in generative models, which is an interesting topic and seems relatively under-explored.

### Weaknesses
* Fig. 2: It is unclear to me how these hierarchical concept graphs are created. It seems that more noisy timestep SAEs capture high-level concepts and less noisy timestep SAEs capture low-level concepts. But this cannot be concluded from visualizing 2-3 neurons from an SAE (which has 5120 neurons based on the appendix). There needs to be more rigorous quantitative evaluation (or more extensive qualitative evaluation) to show this. For example, for each timestep, we could try to quantify the spread of the activations across spatial dimensions. And with this, if we find that more noisy timesteps have less concentrated activations than less noisy timesteps on average across all neurons, then we could possibly make the conclusion that the paper makes.
    * This is a major problem because we already know from other papers like Revelio or Saeuron that SAEs can be used to control image generative models (and multiple SAEs would enable finer-grained control). But the existence of a hierarchical concept graph in these models is not validated by being able to steer the generation.

* The experiments in the paper are focused only on Stable Diffusion 1.4 which is an old model in 2025. I suggest adding results with newer models with better generation capabilities like Flux or Stable Diffusion 3.5. In the current draft, it is unclear if the method and analysis apply only to SD 1.4 or can work more generally across generative model types.

* The paper is quite difficult to follow and I find it hard to connect the empirical ideas to the theoretical ones. It would be great if some of the theoretical ideas could be simplified or connected better with the empirical ideas (within Sec. 3 and 4 itself instead of introducing it later in Sec. 5).

* Some related work on interpretability and steering in generative models via concept bottlenecks [W1, W2] is missing.

### References

[W1] Ismail et al., "Concept Bottleneck Generative Models", ICLR 2024

[W2] Kulkarni et al., "Interpretable Generative Models through Post-hoc Concept Bottlenecks", CVPR 2025

### Questions
Please address my questions/comments in the weaknesses section

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Authors develop theoretical framework for causal identifiability of hierarchical models, motiveted by text to image deep generative models. Key idea is to realize that higher level concepts emerge from the composition of lower-level concepts, not as  in classic causal hierarchies that higher level concepts are causes of lower level concepts. Authors also perform experiments.

### Strengths
- Innovative and important result for modern deep generative models. It is necessary to open up the black box of modern LLMs and diffusion models.  
- Experimental results are also convincing.

### Weaknesses
- From the definition of selection mechanism, I see that higher level concepts need to appear in the lower level concepts. Image concepts are defined as being from R^dim2 and text concepts from N^dim1. If as in Fig 2 text is the highest level concept, then by the selection defintion those interger numbers need to appear in the lowest level? So, does it mean then that only useful concepts, finally in terms of text to image generation are integers appearing in the natural images used to train the model? Or is it, as I guess, that Fig2 is just conceptual and does not reflect the reality? 
- In the VAE  identifiability lit, such as (Simon Buchholz, Bernhard Schölkopf Proceedings of the 41st International Conference on Machine Learning, PMLR 235:4785-4821, 2024) talk about approximate identifiability or weak and strong identifiability (https://arxiv.org/abs/2206.00801). It would be useful to contrast your work with theirs.

### Questions
-

### Soundness
4

### Presentation
3

### Contribution
3
