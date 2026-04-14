# Fine-Tuning of Continuous-Time Diffusion Models as Entropy-Regularized Control

- Decision: Reject
- Scores: 5, 6, 6, 6

## Abstract
Diffusion models excel at capturing complex data distributions, such as those of natural images and proteins. While diffusion models are trained to represent the distribution in the training dataset, we often are more concerned with other properties, such as the aesthetic quality of the generated images or the functional properties of generated proteins. Diffusion models can be finetuned in a goal-directed way by maximizing the value of some reward function (e.g., the aesthetic quality of an image). However, these approaches may lead to reduced sample diversity, significant deviations from the training data distribution, and even poor sample quality due to the exploitation of an imperfect reward function. The last issue often occurs when the reward function is a learned model meant to approximate a ground-truth "genuine" reward, as is the case in many practical applications. These challenges, collectively termed "reward collapse," pose a substantial obstacle. To address this reward collapse, we frame the finetuning problem as entropy-regularized control against the pretrained diffusion model, i.e., directly optimizing entropy-enhanced rewards with neural SDEs. We present theoretical and empirical evidence that demonstrates our framework is capable of efficiently generating diverse samples with high genuine rewards, mitigating the overoptimization of imperfect reward models.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper provides a theoretical framework that frames diffusion model fine-tuning as entropy-regularized stochastic optimal control. The ELEGANT method combines terminal reward optimization, entropy regularization against the pre-trained model, and simultaneous learning of both drift terms and initial distributions. This approach maintains the bridges (posterior distributions) of pre-trained models, theoretically proves to stay within the support of the training distribution, and shows connection to classifier guidance. Compared to PPO-based methods, it also demonstrates advantages in computational efficiency.

The researchers evaluated the method on three tasks: GFP (protein) sequence generation, TFBind (DNA) sequence generation, and image generation with aesthetic quality optimization. The experimental results show that ELEGANT outperforms other baseline methods (PPO+KL, Guidance, NO KL) in terms of achieving higher genuine rewards, maintaining lower KL divergence from the pre-trained distribution, and training speed and efficiency.

### Strengths
The originality of this work is out of question. This paper address the diffusion alignment problem via the continuous-time stochastic control perspective. The quality and the clarity of this paper is reasonable. The authors have conducted a series of numerical experiments to demonstrate its effectiveness. 

Advantages:
- Maintains bridges (posterior distributions) of pre-trained models
- Theoretically proven to stay within support of training distribution
- Shows connection to classifier guidance
- More computationally efficient than PPO-based methods
- Better at mitigating overoptimization

### Weaknesses
Algorithm 1 presents a method with three stages: value function learning and solving two neural SDEs. This seems computational heavy, especially the Neural SDE solving should be hard due to the storation of computation graph when the SDE is parameterized by huge neural networks. It seems impossible for stable diffusion as usually it is only possible to forward pass with batch size = 4 on 80G GPU, let alone the computation graph along a trajectory. I wonder if more clarification can be made about the computation implementation and speed. 

There are several other works that should be compared or mentioned: 
- Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control
- Improving GFlowNets for Text-to-Image Diffusion Alignment, 2024
- Implicit diffusion: Efficient optimization through stochastic sampling, 2024.

The pretrained diffusion model is not continuous time. How to adapt it to the framework proposed in this paper?

There is severe reward overoptimization issue for aesthetic score. Would there be a trade-off when we tune the value of alpha?

The writing is not clear in some places. For example, is f, u, q parameterized neural networks, or abstract notation? Does v mean value function or initial distribution? I can roughly guess their meaning, but this may not be easy for people who are not familiar with SDE.

### Questions
N/A

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
3

### Summary
The paper presents a novel approach for fine-tuning diffusion models. The proposed method, ELEGANT, aims to address the challenge of overoptimization when fine-tuning diffusion models based on a learned reward function. Overoptimization occurs when the model produces samples with high nominal rewards that are outside the support of the data distribution on which the diffusion model was trained. To prevent overoptimization, the authors propose an entropy-regularized objective that is optimized over trajectories, i.e., path measures, formulated as a stochastic optimal control problem.  The authors introduce another learnable SDE to obtain approximate samples from the intractable initial distribution. The numerical evaluation shows that the proposed approach is on par or better compared to the considered baselines in terms of reward while achieving a lower KL divergence.

### Strengths
- The problem of overoptimization is well-motivated and clearly explained.
- Using entropy-regularization is a sensible and approach to counteract overoptimization.
- Interesting connections and explanations, such as Feynman-Kac, Bridge Preservance, and Classifier Guidance.
- Numerical results show that entropy regularization helps to stay closer to the support of the original diffusion model.

### Weaknesses
Things that might improve the clarity and readability of the paper:

- Readers unfamiliar with KL divergences on path measures might wonder where the objective function (5) is coming from. I think it would be great to include a derivation (which is already part of the proof of Theorem 1 but it would be good to make it more explicit).
- The same holds for the optimal entropy-regularized value function (Equation after Eq. (5)).
- The authors make it sound like the Feynmac-Kac (FK) formulation is introduced to provide intuition. However, to the best of my knowledge, it is used to learn the approximate optimal initial value function $\hat a$. It would be helpful to make this connection more explicit in the main part of the paper.
- Moreover, the optimal initial value function $v_0^*$ is used in objective Eq. (8). Only later in the paper is explained how it is obtained. The authors should at least include a sentence that an explanation for how to obtain $v_0^*$ is deferred to later in the document.
- The text between Lemma 1 and 2 uses an expression for the optimal drift that, to the best of my knowledge, stems from the HJB equation. This is never mentioned in the main text and potentially could cause confusion. It would be good to explain where the formula for the optimal drift is coming from
- From my perspective, the main body of the paper contains content that is not integral to understanding the methodology, such as the section on Diffusion Bridges or Algorithm 2 & 3 and instead defers parts that are essential for understanding the paper to the appendix, in particular, the approximation of optimal initial value function.
- The term entropy regularization actually refers to a KL regularization which could potentially cause confusion.
- Using a capital letter for describing the value function may help to distinguish it from the initial distribution $\nu$ due to their similarity.

Weaknesses:

- While inference time is already a bottleneck for diffusion models, the proposed approach requires simulating two SDEs.
- The approach requires solving two stochastic optimal control problems and an additional regression problem for obtaining the initial value function.
- The method requires a) gradient checkpointing and b) low-rank modules to train, making the approach inconvenient to use.
- My main concern with the paper is the following: The terminal reward $v_0^*$ for learning the drift $q$ is approximated with a neural network using a regression objective. Hence, the learned model could be arbitrarily bad in regions with no data which is also the cause for the overoptimization problem. It thus seems like the problem has just shifted to the initial value function estimation and thus not properly addressed.

### Questions
See Weaknesses.

### Soundness
4

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
The authors propose a new method for fine-tuning diffusion models by maximizing the value of the reward functioning in a goal directed manner. They propose an objective that maximizes a reward function towards the desired goal, jointly with the negative KLD to the data distribution (to stay close to the pre-trained diffusion model). The closed-form expression of the optimal solution is an energy based model (untractable). The authors show that they can augment the pre-trained diffusion model to sample from this EBM. This is achieved by adding a drift term and learning the initial distribution.

### Strengths
* The problem is interesting. The approach of augmenting a diffusion model with a term to approximate a different distribution is elegant.

* Approximating EBMS is a hard problem. This seems to be another nice solution in another useful special setup.

### Weaknesses
* In the write-up, the authors state that the methods is an entropy-regularized control against the pretrained diffusion model but in the equations, the data distribution is used instead of the pre-trained distribution. Could you clarify?

* Approximation errors when solving of E9 and E10. Can [1] be used instead? Similarly to the proposed approach, the entropy of the EBM policy can be approximated in closed form. The parameters of the samplers (also initial distribution) can be simply learnt via optimizing the KLD. [1] seems to be more parameter and computationally efficient. 

[1] Messaoud, Safa, et al. "S $^ 2$ AC: Energy-Based Reinforcement Learning with Stein Soft Actor Critic." arXiv preprint arXiv:2405.00987 (2024).

* Missing references to proofs in appendix (eq5, theorem 1, corolalaries 1 and 2)

### Questions
(1) Can p_{data} be replace by the distribution of the pre-trained model? What are the pros and cons of using each of these distributions.

(2) Why is your approach of modifying the diffusion process better than using a parametrized SVGD as in [1]? 

[1] Messaoud, Safa, et al. "S $^ 2$ AC: Energy-Based Reinforcement Learning with Stein Soft Actor Critic." arXiv preprint arXiv:2405.00987 (2024).

### Soundness
3

### Presentation
3

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
This study focuses on the overoptimization issue in the finetuning of diffusion models. The authors point out that recent methods of finetuning diffusion models usually suffer from the overoptimization problem, which shifts the generation distribution towards a high nominal reward value but away from the natural distribution. To this end, the authors propose a new finetuning method to adjust the generation distribution. They reformulate SDE with an additional drift term and a different initial distribution. Then, they provide the optimal solution to the drift term and initial distribution and develop an optimization algorithm to solve it. Experimental results on biological sequence tasks and image generation demonstrate that the proposed method achieves high reward values while keeping a small value of KL divergence from the original distribution.

### Strengths
+ The overoptimization problem is important in finetuning diffusion models, and this study provides a sound way to address it. The authors formally formulate the overoptimization problem and the desired properties in Section 4.
+ This study involves both solid theoretical analysis and interpretations that are easily understood. 
+ The evaluation based on the comparison between nominal reward and genuine reward is interesting and persuasive.

### Weaknesses
- The motivation for learning a different initial distribution is unclear. Why does the initial distribution affect the finetuning of diffusion models, and why should we finetune the model on a different initial distribution? On the other hand, does it affect the model performance on general tasks? In other words, given the model finetuned with the learned initial distribution, if users sample the generation from the original Gaussian distribution in the sampling phase, how about the quality of generations?
- Equations in Lines 372-377 are slightly confusing. What do $y_0$, $y_t$, and $y_T$ mean and how to obtain them in implementation?
- The connection between the added drift term and the classifier guidance is interesting. Given that they have a similar mathematical formulation, I expect a theoretical analysis of their difference in performance.
- How was $\alpha$ in PPO+KL set in experiments of Figure 3? Have you tried different settings of $\alpha$ for PPO+KL and ELEGANT for comparison? I notice that in Figure 5 and Figure 6, $\alpha$ was set to 0.001 for PPO+KL, which was very small, while ELEGANT used $\alpha=\{5,10\}$. I am not sure whether such a comparison is fair.
- Generations of ELEGANT still exhibit the overoptimization phenomenon. For example, the first line of Figure 3(c), especially the image of ``panda,’’ looks unnatural. Images in Figure 5 also have unnatural backgrounds and colors.
- What is the additional computation cost and memory cost of learning $q$ and $u$?

### Questions
Please refer to the above weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
