# On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
In this work, we present a simple yet theoretically motivated improvement to Supervised Fine-Tuning (SFT) for the Large Language Model (LLM), addressing its limited generalization compared to reinforcement learning (RL). Through mathematical analysis, we reveal that standard SFT gradients implicitly encode a problematic reward structure that may severely restrict the generalization capabilities of model compared to RL. To rectify this, we propose Dynamic Fine-Tuning (DFT), stabilizing gradient updates for each token by dynamically rescaling the objective function with the probability of this token. With just a single-line change, the method outperforms standard SFT on multiple difficult benchmarks and base models, from math reasoning to code generation and multi-modal tasks, demonstrating improved generalization. Additionally, \model~achieves competitive results in offline RL settings, providing an effective yet streamlined alternative. By bridging theoretical insights with practical solutions, this work advances the state of SFT. The source code will be available at https://github.com/yongliang-wu/DFT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Dynamic Fine-Tuning (DFT), a method that improves the generalization of Supervised Fine-Tuning (SFT). The authors first provide a analysis by framing SFT from a reinforcement learning perspective, identifying that its implicit reward is inversely proportional to the model's confidence, which leads to unstable training. DFT rectifies this by reweighting the SFT loss with the token probability, a simple modification that is shown through extensive experiments to deliver performance gains across diverse models and tasks.

### Strengths
- The theoretical motivation is clear, connecting the SFT gradient to policy gradients and pinpointing the problematic inverse-probability weighting as the root cause of poor generalization.
- The proposed DFT is simple to implement, requiring only a minor modification to the standard SFT loss function, yet it yields substantial empirical improvements.
- The evaluation is comprehensive, spanning multiple reasoning domains including mathematics, code generation, and multi-modal tasks.

### Weaknesses
- My main concerns lie in the potential conceptual limitations and unintended consequences of the core reweighting mechanism. The strategy of down-weighting low-probability tokens encourages the model to ignore what it finds hard. While this appears to prevent overfitting on noisy or rare patterns in the presented experiments, the paper does not adequately explore the boundary conditions under which this approach might fail. A more thorough discussion or ablation study on when this *hard example ignorance* becomes detrimental would strengthen the paper's claims.
- The authors mention avoiding numerical instability as a reason for token-level weighting but lacks detail on how it handles token probabilities that are extremely close to zero, which could cause vanishing gradients for those specific tokens.
- The implicit assumption that a uniform reward of 1 is optimal for all expert demonstrations might be an oversimplification, as it treats demonstrations of varying quality, complexity, or importance equally.

### Questions
- How does the DFT method influence the calibration of the model's output probabilities? 
- The decision to apply weighting at the token-level was justified by stability. Did the authors investigate applying a sequence-level probability as the weight, perhaps with safeguards like clipping, and how did its performance compare?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Proposes a modification to supervised finetuning (SFT) for language models, called Dynamic Fine-Tuning (DFT). Instead of minimizing the standard cross-entropy loss, DFT weights each gradient of an output by the probability of that output. This change is motivated by framing SFT as a reinforcement learning method. Empirical evaluations demonstrate that DFT outperforms SFT across several benchmarks and base models.

### Strengths
1. Relatively well-written and easy to follow.

2. The idea behind DFT is intuitive and conceptually simple.

3. Experiments demonstrate that DFT leads to substantial performance gains over SFT across several models and settings, covering mathematical reasoning, code generation, and multi-modal tasks.

### Weaknesses
1. The theoretical motivation behind DFT is imprecise and includes several unsubstantiated claims.

    - The dependence of SFT on $1 / \pi_\theta (y | x)$ is fake, in the sense that it is obtained by multiplying and dividing by $\pi_\theta (y | x)$. In particular, it is not true that the gradient of SFT grows unboundedly due to the term $1 / \pi_\theta (y | x)$ in Equation (6), as this term cancels out with the expectation.

    - The claim of SFT having sparse rewards does not make much sense to me. In SFT, you are guaranteed access to the output $y*$ with the optimal reward of 1 (according to the formulation in the paper). Such a claim about sparseness would be true if one would actually sample outputs from $\pi_\theta$ until landing on $y^*$, but this is not what SFT does.

    - In several places (e.g., line 51 and lines 167 to 171), a connection is being made between large gradient norm, unstable training, and overfitting. Regardless of the fact that, as mentioned above, the gradient norm of SFT does not grow unboundedly when $\pi_{\theta} (y^{*}|x)$ is small, why would large gradient norms lead to overfitting? When making such claims you need to either ground them in the literature or theoretical evidence. Otherwise, it is best to refrain from making them or clearly hedge them as intuition to not mislead readers.

    To be clear, if a method works well in practice, I do not believe that theoretical motivation is strictly necessary. There are plenty of heuristics that are useful in practice, yet lack theoretical backing. However, if one does choose to motivate their method theoretically, that motivation should be sound and clearly separate intuition from formal arguments, which can be backed by results in the literature or new results. Another downside of imprecise theoretical motivation is that it may hide the true reason for why DFT improves upon SFT.


2. The paper lacks a discussion of potential limitations of DFT. For example, weighting outputs by their current probability should induce a tendency to prefer high probability outputs, thereby sharpening the model’s distribution. This suggests that the performance of DFT may highly depend on the initial model, to a greater extent than SFT. Does DFT indeed improve over SFT across the board, or when finetuning on domains further away than the model's current capabilities SFT works better? Are there any other potential limitations of DFT or do the authors always recommend using DFT over SFT?



Review Summary and Recommendation
---

Overall, DFT seems to bring substantial benefits compared to SFT across several practically relevant settings. The fact that such an improvement can be achieved by a simple (in a good way) tweak to SFT is quite remarkable. I therefore tend toward recommending acceptance. However, I do strongly suggest either toning down some of the imprecise and unsubstantiated claims made regarding SFT (see weaknesses portion of the review) or grounding them in a formal analysis or empirical evaluation.


Additional (More Minor) Comments
---

1. In Equation (7) there is a minor issue with the definition of DFT: $L_{SFT}$ was defined as an expectation over the training data, while here it is treated as if it is the loss over a single example $(x, y^*)$.

2. Typo in Equation (8): I believe there should be a minus sign there (i.e., the objective is to minimize the negative log probability and not the log probability).

### Questions
--

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper observes that supervised fine-tuning (SFT) encodes a sparse reward that limits its generalization ability when compared to reinforcement learning (RL). Due to the sparse nature of the reward, using SFT may lead to optimization difficulty and/or poor generalization. The paper introduces a method called dynamic fine-tuning (DFT) that reweighs the SFT gradient to avoid the problem observed with SFT. The effectiveness of DFT over SFT is analyzed via experiments on several language models. Additional results are provided with other setups including coding and multimodal models.

### Strengths
- The analytical arguments are clear and presented in an easy-to-follow manner. The proposed fix via weighing the gradient follows naturally from the analytical argument.

- Experimental results that show improvement over SFT models are fairly extensive. These experiments cover a variety of language model families (Llama, Qwen, DeepSeek). 

- The impelemtnation included in the appendix is easy to understand and use by a general ML practitioner.

### Weaknesses
-While the analysis is good, there may be other papers that have made observations about sparse rewards and done the derivation perhaps under a different guise. I appreciate the authors referring to GOLD (Pang & He, 2021) in the paper as I learned about this method as well. A follow up ICLR Blog Post does a derivation that looks similar to what is included in the paper (https://iclr-blog-track.github.io/2022/03/25/text-gen-via-lfd). So the derivation, while insightful, is perhaps not as valuable as the empirical results. 

- Table 1 reports SFT vs DFT results. I wonder why there are no results available with RL (SFT + RL, DFT + RL perhaps?). I am curious to understand whether DFT-based models can be effective when used to further train with RL. So the claim on DFT surpassing RL may need to be qualified as I do not see any RL methods in Table 1

- Similar comment can be made for other setups like coding and multimodal models considered in the paper

### Questions
- Please check the ICLR Blog Post I shared above and see if that merits being cited in the paper. While the analysis is still useful my suggestion is to focus on empirical results as the main contribution in the paper

- I would like to understand why there are no RL results in Table 1 (and others).

- Are DFT-based models RL-able? 

Overall, this is a well written and simple to follow paper. I look forward to interacting with the authors, other reviewers and AC to get a deeper understanding of this work and make a decision.

The following paper may be of interest to the authors: 
"Simplify RLHF as Reward-Weighted SFT: A Variational Method" https://arxiv.org/abs/2502.11026

(Note: I am sharing the above in the spirit of helping out a colleague in the research community. Citing the above is not necessary for me to recommend an acceptance.)

### Soundness
2

### Presentation
3

### Contribution
2
