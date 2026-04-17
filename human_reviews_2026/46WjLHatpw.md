# Stochastic activations

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
We introduce stochastic activations. This novel strategy randomly selects between several non-linear functions in the feed-forward layer of a large language model. In particular, we choose between SILU or RELU depending on a Bernoulli draw. Our strategy circumvents the optimization problem associated with RELU, namely, the constant shape for negative inputs that prevents the gradient flow. We leverage this strategy in two ways:

(1) We use stochastic activations during pre-training and fine-tune the model with RELU, which is used at inference time to provide sparse latent vectors. This reduces the inference FLOPs and translates into a significant speedup in the CPU. Interestingly, this leads to much better results than training from scratch with the RELU activation function. 

(2) We evaluate stochastic activations for generation. This strategy performs reasonably well: it is only slightly inferior to the best deterministic non-linearity, namely, SILU combined with temperature sampling. This offers an alternative to existing strategies by providing a controlled way to increase the diversity of the generated text.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes training LLMs with stochastic activations to strike a balance between performance sparsity/efficiency. The performance seems promising but it is unclear how much practical benefit this can have on GPUs.

### Strengths
1. Ideas and experiments are clear and easy to understand.
2. Training with stochastic activations is shown to have performance on par with SiLU activations, resolving a major weakness of ReLUs.
3. Experiments are thorough.

### Weaknesses
1. Practical use case for this method is unclear since GPU acceleration is not demonstrated.
2. The generalizability is unclear. Will this setup work with pairs of activations that are less similar (e.g., ReLU + tanh)? What about non-ReLU sparse variants (e.g., soft/hard thresholding)?
3. Lack of structured sparsity baselines.

### Questions
1. Line 143: Unremoved comment
2. Hard to see what is going on at the tail in figure 4. Could you shrink the y-axis range?
3. Can you comment on when/how your method is preferred over post-training sparsity (e.g., [1]) and inference-time sparsity algorithms (e.g. [2,3])?

[1] Zhang, et al., MoEfication: Transformer Feed-forward Layers are Mixtures of Experts, 2021.
[2] Dong, et al., Prompt-prompted Adaptive Structured Pruning for Efficient LLM Generation, 2024.
[3] Lee, et al., Cats: Contextually-aware thresholding for sparsity in large language models, 2024.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces two distinct methods for training with multiple activation functions. The first method, Swi+FT, is a deterministic two-stage process: the model is first trained with SiLU, then the final portion of training is completed with ReLU. The goal is to achieve the high quality of a SiLU-trained model while retaining the sparsity and inference-speed benefits of ReLU, which the authors demonstrate on CPUs. The second method, StochA, trains using a stochastic activation that randomly interpolates between ReLU and SiLU (controlled by a probability p). This is proposed as a novel, inference-time tool to control output diversity, similar to temperature sampling. Both methods achieve performance similar to SiLU, with Swi+FT being 1.65x faster on CPUs and StochA providing activation swapability as a tool for increasing generation diversity.

### Strengths
- The paper successfully introduces and validates two distinct, novel methods for training with multiple activation functions. 
- The primary method, Swi+FT, is well-motivated and its "best of both worlds" claim is supported by the data. The results clearly show it produces a final ReLU model with the high test-set quality of a SiLU-trained model, while retaining ReLU's sparsity for a 1.65x speedup on CPUs.
- The experiments are robust, validating these findings across two different model scales (1.5B and 3B) on various downstream tasks.
- The results show a positive scaling trend: the quality benefit of Swi+FT over the ReLU baseline increases with the larger model, suggesting its value may grow with model size.
- The second method, StochA, is also shown to be effective, achieving SiLU-level performance while providing a new, functional lever for controlling output diversity at inference time.

### Weaknesses
- W1. The 1.65x speedup from the Swi+FT method is demonstrated exclusively on CPUs. This limits the work's impact, as large-scale models are predominantly deployed on GPUs/TPUs where the benefits of activation sparsity are often different or less pronounced. For true "edge device" applications, this "decent" speedup is likely insufficient, as activation sparsity is only one part of a much larger optimization problem.
- W2. The diversity claim for Stocha is not convincingly supported or motivated. The diversity analysis fails to compare StochA against the most common and simple tools we already have, namely temperature and nucleus (top-p) sampling. It is unclear if StochA offers any real benefit over these established methods. 
    - The evaluation also lacks qualitative examples or application to a task where controlled diversity is critical (e.g., RL exploration), making it hard to judge the usefulness of this new diversity lever.
     - Finally, the paper is missing comparisons to highly relevant prior work. For the StochA method, there is no comparison to other stochastic activation functions (like ASH).
- W3. More broadly, the paper doesn't make a strong case for its novelty in a somewhat saturated field, and it's unclear if this complex approach is truly superior to simpler, established adaptive activation functions.

### Questions
- The paper's motivation is weakened by the lack of clear justification for the complexity of its novel activation function compared to simpler stochastic regularization techniques. The authors do not discuss why this approach is necessary or superior to existing stochastic dropout methods (like Layer Dropout) which offer clearer, multi-faceted benefits in practice (e.g., dynamic model sizing or uncertainty estimation).
- To validate the quality claims of the $\text{Swi+FT}$ method, could the authors confirm that the total training compute (total number of iterations) was held constant or until convergence across the $\text{SiLU}$, $\text{ReLU}$, and $\text{Swi+FT}$ baselines?
- What are the benefits of $\text{StochA}$ compared to the most widely used diversity levers (e.g., temperature sampling and top-p sampling? Evaluating on a use case where controlled diversity is critical as in RL for LLMs would be helpful. 
-  The practical benefits of the StochA method need stronger justification. The authors should explicitly discuss and quantify how the benefits of StochA (controlled diversity) compare with existing diversity methods such as temperature and top-p sampling.
- The Related Work section contains an editorial comment that should be removed for the final version.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces two new methods to leverage the expressivity of SiLU while, at the same time, the sparsity and computational efficiency of ReLU at inference time. The first method Swi+FT trains on SiLU, then adapts the network to ReLU for the last 5-10% of training. The second method StochA samples the activation as Bernoulli(p) during training or test time. The product is a ReLU-based model at inference time that outperforms ReLU-based training.

### Strengths
I enjoyed this paper for several reasons. The approach seems **novel** (I may be wrong, I'm not an expert in this field), yet **simple**. The paper is a largely empirical one: the authors did **extensive experiments** which validate their main thesis that SiLU's expressivity can be exploited during training to make better models; then ReLU's sparsity can be exploited at inference time to make them fast.

### Weaknesses
The reason my score is not higher is for the following reasons. In its current state, the paper is quite difficult to read and some evaluation points not justified. If both points are resolved (should be easy to fix), I will increase my score to "accept". 

1. The main area where the paper can be improved is the presentation. Overall, there are many experimental results, which makes the takeaways get lost. For instance, Table 2 is extremely difficult to parse. Please distill Table 2 to the key results most salient for the paper, add **bolding** for the best performances, and move the rest to the appendix. In general, visually highlighting the takeaways for each paragraph of subsection is necessary given the density of the results.

2. Diversity of generations ablations. Why does the F1 score signal higher diversity? I would have expected to see an actual text diversity metric, e.g., type-token ratio, entropy, etc.

### Questions
l271 Typo: The following only applies to the StochA strategy: we evaluate the performance when if leverage the randomness at test time

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To balance the sparsity and efficiency of activation functions during LLM training, this paper introduces stochastic activations that randomly select ReLU and SILU. Specifically, in experiments, the first method, Swi+FT, use SILU at 10% of the training steps and ReLU for the rest steps; the second method, STOCHA, randomly selects ReLU and SILU during training and uses ReLU for inference. Experimental results show the comparable performance and reduced inference times of the proposed stochastic activations.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed stochastic activation is simple while addressing the sparsity and computational efficiency of ReLU.
3. The experimental results show comparable performance of stochastic activations and reduced inference time.

### Weaknesses
1. This paper only proposes to switch activation functions between ReLU and SILU for LLM pretraining. An in-depth analysis should be conducted: why switching the activation function during pretraining and randomly selct activation functions both retain comparable performance? How models behave differently using a single or combination of these activation functions? Can we have some theoretical explanation of stochastic activations?
2. The experiments are conducted on pretraining small-scale transformers (1.5B and 3B), which may not generalize to large-scale models. Why not apply this method to post-training?
3. As shown in Figure 5, the performance of stochastic activations only outperform SILU on specific range of $\alpha$ in 1.5B model and always underperforms SILU in 3B model. It seems the performance is fragile because it depends on dedicated selection of hyperparameter and behaves differently on different models.

### Questions
See Questions mentioned in the point 1 and 2 in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
