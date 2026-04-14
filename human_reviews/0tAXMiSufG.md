# BOND: Aligning LLMs with Best-of-N Distillation

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6, 6

## Abstract
Reinforcement learning from human feedback (RLHF) is a key driver of quality and safety in state-of-the-art large language models.
Yet, a surprisingly simple and strong inference-time strategy is Best-of-N sampling that selects the best generation among N candidates.
In this paper, we propose Best-of-N Distillation (BOND), a novel RLHF algorithm that seeks to emulate Best-of-N but without its significant computational overhead at inference time. Specifically, BOND is a distribution matching algorithm that forces the distribution of generations from the policy to get closer to the Best-of-N distribution. We use the Jeffreys divergence (a linear combination of forward and backward KL) to balance between mode-covering and mode-seeking behavior, and derive an iterative formulation that utilizes a moving anchor for efficiency. We demonstrate the effectiveness of our approach and several design choices through experiments on abstractive summarization and Gemma models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This article models the alignment of LLMs as a distribution-matching problem. Specifically, it considers the distribution induced by Best-of-N sampling as the target distribution and optimizes the Jeffreys divergence with respect to it for balancing the characteristics of forward & backward KL-based optimization. Additionally, this work derives an iterative form, updating the reference model during training.

### Strengths
1. This work implements the LLM alignment problem through Best-of-N distillation, which can be a sound direction for the development of algorithms in this field.
2. This work formulates and discusses the BoN distribution and its relationship with the general RLHF target. 
3. This work proposes to utilize Jeffreys divergence to balance the mode-covering and mode-seeking behavior introduced by forward- and backward-KL- KL optimization.
4. This work further integrates their method with EMA techniques and proposes an iterative version.

### Weaknesses
1. My biggest concern is the absence of direct comparison with close-related works http://arxiv.org/abs/2407.06057, http://arxiv.org/abs/2406.00832. It would be much more convincing if there is a clear comparison like Figure 1 in http://arxiv.org/abs/2407.06057, rather than a simple baseline REINFORCE presented in Figure 7 in this work.
2. More discussion about why BoN distribution as the target can be included.
3. Other possible additional analyses are discussed in the Questions.

### Questions
#### Question 1:
Line 298, why is the proposed approach "equivalently to distilling a Best-of-$N^M$ of the initial distribution $\pi_{ref}$"? Is this a qualitative or quantitative statement?
#### Question 2:
Figure 4, seems like a linear relationship between # steps and r(y) for iterative BOND.
For Figure 5, it seems like another kind of trending between $r(y)$ and # steps or $KL(\pi||\pi_{ref})$.
Figures 6 and 7, present a log-like trend between the KL and reward similar to the REINFORCE algorithm.
A similar trend is also observed in http://arxiv.org/abs/2406.00832 and http://arxiv.org/abs/2407.06057.
As discussed in http://arxiv.org/abs/2204.05862 and my experience, there can be an approximately linear relationship between $\sqrt{D_{KL}}$  and $r$ for BoN sampling. 
It would be interesting if the author could provide some empirical or theoretical intuition about such a relationship.
Does this indicate that even though performing a BoN distribution match, it is still more similar to the general policy-gradient RL algorithm (which may try to match another distribution)?
#### Question 3:
Why is there an approximately linear relationship between #steps and KL in, as presented in Figures 5 and 6 for the BOND algorithm?
For the REINFORCE algorithm, it seems a quite different trend.
#### Question 4:
Figure 4 presents consistent trending between KL and $\log_{p\le(y)}$ across varying $N$ and algorithm, which is quite interesting.
Is there any explanation for this phenomenon?
#### Question 5:
Is there any analysis or comparison of the reward overoptimization of this algorithm?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper is essentially distilling inference-time best-of-N sampling into training time. Specifically, the authors propose to train the policy to match the best-of-N distribution (which is an analytical form derived by the authors). The distribution matching is done through minimizing a combination of forward and backward KL. The behavior of J-BOND appears better for reward vs. num steps as well as KL(policy || ref policy) vs. num steps, compared to REINFORCE with various beta values.

### Strengths
The motivation of distillation is great and this direction should be deeply explored. The paper is written carefully. The algorithm is clearly written. I checked the derivations and they seem correct so far.

REINFORCE is a natural baseline and the authors have attempted multiple beta values for the baseline.

### Weaknesses
Have the authors tried reward shaping techniques for RLHF baseline, e.g., making the reward values more peaky – either very large or very small? 

I’d appreciate a more comprehensive discussion on how much the authors expect this technique to benefit downstream tasks.

It’ll be great if the authors can include more discussion on whether the baseline B is important or not in the algorithm.

What ablations on beta and gamma in Algorithms 2 (balancing among forward KL, backward KL, additional regularization) would likely benefit downstream tasks more? It's still unclear to me why we want to put an equal/large weight on backward KL. More motivation would be nice.

### Questions
Would the curves in Figure 4(a) converge to similar spots if you run the algorithms long enough?

Do the authors expect that the conclusions would be similar for much larger models?

Writing: 
- pi(x,y) makes it look like a joint distribution; do the authors mean pi(y|x)?
- 3e-6 means 0.000003; 3e^(-6) means 0.007. Which one are you referring to?

### Soundness
3

### Presentation
2

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
This paper introduces a distribution matching-based Best-of-N distillation method that simulates the Best-of-N distribution space, while reducing the time overhead of N inferences to just one. Starting from the theoretical distribution of BoN, the authors construct the Iterative BOND algorithm based on Quantile estimation and the choice of Jeffreys Divergence, and further propose the more practically meaningful J-BOND algorithm.

### Strengths
1. Rigorous Theoretical Analysis: This work rigorously analyzes the distribution characteristics under Best-of-N sampling and establishes its connection with standard RLHF, as well as the specific reward value $r_{BOND}$ under this correlation. This provides a reliable theoretical foundation for the work, rather than being based on naive assumptions.

2. Some Degree of Novelty: Although there is some concurrent work, the idea of distilling distributions from Best-of-N is fairly novel and important.

3. Consideration of Practical Efficiency: I appreciate the authors' consideration of the practical efficiency of the algorithm. The proposed J-BOND algorithm theoretically has lower sampling complexity, which should increase the efficiency of RLHF.

### Weaknesses
1. Lack of Important Baselines: Given that the main purpose of the paper is to distill Best-of-N sampling, BoN performance should straightforwardly serve as an important baseline to analyze pros and cons in terms of performance and efficiency. Moreover, other concurrent BoN distillation algorithms [1] should also be considered.

2. Lack of Downstream Validation: The main metrics in the paper, such as reward value and KL divergence, cannot be directly equated to the model's performance on downstream tasks. For an RLHF method, it is necessary to conduct experiments on downstream tasks and present more intuitive metrics to demonstrate the model's alignment performance.

3. Insufficient Experimental Setup: The paper lacks exploration of several issues. For instance, BoN sampling heavily depends on the Reward Model, and the influence of different RMs on the BOND algorithm is not investigated. Additionally, a more nuanced exploration of Jeffreys Divergence with smoother β variations could be included; and the comparison between J-BOND and standard Iterative BOND lacks investigation.

[1] Variational Best-of-N Alignment, https://arxiv.org/pdf/2407.06057

### Questions
1. Can the authors compare with more fundamental baseline methods, such as BoN or other BoN distillation algorithms?

2. Can the authors supplement additional experiments, including downstream validation and more ablation studies as discussed in Weakness 3?

3. Can the authors prove more clearly the advantages of the BOND algorithm over other Alignment algorithms, in terms of both performance and efficiency, to make the argument more convincing?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Best-of-N Distillation (BOND), a novel alignment tuning algorithm designed to emulate the Best-of-N sampling method in a more computationally efficient manner. BOND aims to achieve the same high-quality output as Best-of-N sampling without the inference-time computational overhead, by aligning model outputs to match the distribution of the Best-of-N candidates. 
In addition, to ensure stability and scalability, the authors introduce an iterative approximation strategy that operates effectively even with a minimal sample size (e.g., 2 or 4). 
Further, based on the two types of loss function derivation aiming forward and reverse KL respectively, the author leverages Jeffreys divergence proposing J-BOND, an enhanced algorithm incorporating iterative distribution matching with an exponential moving average (EMA) anchor. J-BOND demonstrates effectiveness in maintaining a stable training and superior KL-reward trade-off through experiments.

### Strengths
The paper is well-structured and clearly presents its methodology, with detailed explanations and algorithms that allow readers to follow the progression. From iterative BOND to the addition of KL regularization in Sections 4 and 5, the additional experimental results effectively support these methodological advancements. 
BOND is notable for its originality, offering a practical and computationally efficient alternative to traditional RLHF that achieves a superior KL-reward balance without requiring commitment to a specific regularization level. The work is significant in its potential impact on RLHF practices, as it provides a scalable solution for optimizing performance and efficiency while minimizing trade-offs between KL divergence from the reference distribution and reward maximization.

### Weaknesses
The paper relies heavily on the Jeffreys divergence without sufficient comparative analysis against alternative divergence metrics. The mode-covering and mode-seeking behavior property paper mentioned about are only observed in lower dimension such as multimodal distribution in 1-dimension. An inclusion of other divergence types, especially in the iterative stages, could offer clearer insights into the unique advantages of Jeffreys divergence. Further, relevant literature on divergence measures in alignment tuning, should also be cited to contextualize this choice.
- Go, Dongyoung, et al. "Aligning language models with preferences through f-divergence minimization." Proceedings of the 40th International Conference on Machine Learning. 2023.

As the paper discusses the method’s efficiency, the paper would benefit from explicit comparison of the computational cost saved by BOND relative to traditional Best-of-N sampling, or comparisons with sampling approaches used in RLHF. This would clarify BOND’s potential advantages in real-world applications.

Additionally, while the paper addresses the challenge of sampling size N through iterative approximation, showing practical advantages like non-saturation compared to non-iterative BOND, this helpful randomness raised in iterative BOND is solely introduced by approximation randomness, which lacks controls or specific directions. This calls into question whether the proposed algorithm genuinely achieves a distilled $\text{Bo}N^M$ distribution. 
The substantial difference in $r(y)$ between iterative and non-iterative BOND  in Figure3 suggests a potential vulnerability to reward hacking, as discussed in Gao et al.
- Gao, Leo, John Schulman, and Jacob Hilton. "Scaling laws for reward model overoptimization." International Conference on Machine Learning. PMLR, 2023.


The introduction combines related works and problem setup, which could be structured more effectively. Detailed discussions on RLHF and Best-of-N would be more suitable in a separate related works section or could be incorporated into the problem setup. In the introduction, it would be clearer to emphasize the limitations of existing methods and highlight the advantages of the proposed approach over current methods.

### Questions
In Section 4.2, the authors utilize 32 Monte Carlo samples to estimate the backward and forward KL divergences between the training policy and the reference distribution. Given the high dimensionality of these distributions, this sample size seems insufficient to reliably capture the divergence and may introduce substantial estimation variance. A sensitivity analysis showing how the estimator's variance changes with an increasing number of Monte Carlo samples would strengthen the results. Alternatively, using a larger sample size for these estimates could enhance the reliability of the reported divergences.

While BOND’s benefits, such as improved KL/reward trade-offs and dynamic regularization, are discussed in the body of the paper, they are not clearly summarized in the introduction or abstract. A brief overview in these sections would effectively communicate BOND’s main advantages over traditional RLHF approaches, aiding readers in understanding its unique contributions and practical value.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper is focusing on the RLHF alignment problem, in particular on emulating the Best-of-N distribution which is known to perform very well, but is very costly at inference time (for each prompt it requires drawing N candidate generations from the reference model and selecting the one with highest reward according to a reward model). The authors propose the BOND (Best-of-N Distillation) algorithm designed to force the distribution of generations from the finetuned policy to be close to the Best-of-N distribution, requiring the generation of just a single sample (instead of N). To this end, BOND regards the alignment problem as a distribution matching problem and distills the Best-of-N distribution by finetuning the reference policy to imitate the Best-of-N distribution. To stay close to the original reference model, the authors incorporate a KL regularization term that considers both the forward and backward divergence (Jeffrey divergence). In addition, they incorporate Monte-Carlo quantile estimation, and exponential moving anchor, resulting in the J-BOND algorithm. The authors conduct experiments on the abstractive summarization task (XSum dataset) and aligning GEMMA using J-BOND.

### Strengths
The paper makes multiple contributions, namely theoretical derivation for the Best-of-N distribution and a practical RLHF finetuning algorithm that distills the Best-of-N distribution into a policy which is sample efficient and requires just one single sample at inference time

The authors are making a lot of engineering design choices in their proposed model, and carefully analyze the role of each component in the performance of the proposed algorithm

To regularize the model and ensure it is not steering too far from the reference model (the supervised finetuned policy), the authors use a combination of both forward and reverse KL, namely Jeffrey divergence. While the forward KL ensures mode covering behavior, the reverse KL is used for mode seeking behavior; their combination results in better aligned policies that combine the advantages of both divergences

Applying BOND recursively (Iterative BOND) improves the sample efficiency of the BOND algorithm and works for very small values of n (2, 4); its reward/KL tradeoff is comparable to the non-iterative BOND while being more sample efficient

The J-BOND algorithm presents better reward/KL trade-off compared to the REINFORCE algorithm with different values of \beta and does not require using a specific regularization strength

The paper is well written, well-motivated, presents theoretically and experimentally sound insights that would benefit the research community

### Weaknesses
The paper combines a lot of distinct ideas already proposed in previous works - it would be good to actually clearly articulate what the novel contribution is. Besides, the comparison with concurrent works is not very clear, in particular the difference with (Amini et al, 2024), WARM, WARP (Rame et al, 2024). 

Figure 4 - It would be interesting to see how the performance of Best-of-N compares to the proposed algorithm J-BOND and REINFORCE

Algorithm 1, line 330 - \pi_t is not defined

Line 329 - \pi \in Capital \pi -Capital \pi  in Algorithm 1 is not defined

Line 456 - “a large previously trained reward model r(.)” - please provide details

Lines 481-482 - there are not many details about the hyperparameters of the REINFORCE algorithm 

The authors are conducting experiments on Gemma 2B and 7B models, while results are convincing it would be good to see if they hold with other models and other tasks than summarization

### Questions
How does J-BOND performance compare to (Amini et al, 2024) and other concurrent works?

There are three algorithms discussed in the paper, namely BOND, Iterative-BOND and J-BOND. Is it always preferable to use JBOND or do you recommend using each algorithm in particular situations?

Will the code be made publicly available to serve the research community?

### Soundness
3

### Presentation
3

### Contribution
3
