# Inference-Time Diffusion Model Alignment via Random Ordinary Equations

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Aligning diffusion models (DM) with human preferences is a challenging yet practical task. Recent efforts focus on training-free methods, but usually adopt high-dimensional action spaces or require differentiable rewards. To address these issues, we propose a novel inference-time alignment framework based on random ordinary differential equation sampling. Specifically, we first formulate DM alignment as a max-encountered-reward optimal control problem. Then, by fixing the process noise and optimizing the perturbation strength, we obtain a 1-D action space, which integrates naturally with Monte Carlo tree search. We can thus perform trajectory search to derive the optimal control in a gradient-free manner, therefore supporting non-differentiable rewards. We also provide theoretical guarantees and empirical evidence to support and validate our method. Experiments show that our method demonstrates sufficient sample diversity and successfully aligns pre-trained DMs with reward functions defined on clean image domains.Our method outperforms traditional inference-step scaling, achieving higher best rewards. Meanwhile, it has significantly higher parameter efficiency than existing approaches adopting high-dimensional action spaces. Our approach can be plug-and-play integrated into any multi-step inference DMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
- This paper proposes a novel diffusion alignment method that operates efficiently at inference time.  
- The approach is based on Monte Carlo Tree Search (MCTS), with the key originality lying in the combination of RODE sampling, a one-dimensional action space, and a max-reward control strategy.  
- The proposed method is validated through extensive experiments, demonstrating that  
  (a) it efficiently and actively searches for high-reward trajectories, and  
  (b) its sampling process is both stable and diversity-controllable owing to the RODE formulation.  
- The method is applicable to non-differentiable rewards, since it operates in a gradient-free manner.  
- The authors also provide theoretical guarantees regarding sampling stability and score estimation error bounds.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Strengths
- This paper presents a novel diffusion-alignment method that operates efficiently at inference time.  
- The proposed method is supported by extensive experiments, evaluated in terms of "Best Reward" and "Parameter Efficiency".  
- The experiments also provide useful insights into the behavior of the proposed alignment method:  
  (a) Proposition 1 and the experimental result on sample diversity (measured by the mean pairwise distance) are well connected, as discussed around line 405.  
  (b) The visualization of the denoising path on the data manifold (Figure 4) effectively illustrates the exploration capability of the proposed approach.  
- The authors also provide theoretical guarantees regarding the stability of the method and the bound on score-estimation error.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Weaknesses
- The theoretical argument, especially Proposition 2, needs to be clarified.  
- It is not entirely clear whether the authors can rigorously justify the following conclusions based on their propositions:  
  (1) that $p^{(R)}$ approximates the true data distribution as well as $p^{(S)}$; and  
  (2) that using an SDE-trained score network within RODE sampling does not significantly increase the score-estimation error.  
- See Question (2) below for more detailed comments.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Questions
1. **On the max-reward control strategy**  
   Can the authors provide a more detailed explanation of why the proposed max-reward control strategy performs better compared to FK steering [1] (which is also cited in the paper)?  
   This kind of approach seems natural since the intermediate reward is explicitly defined (see the theoretical discussions in, e.g., [2]).

2. **On Proposition 2 and its interpretation**  
   (a) Was $M_{t_{k+1}}$ introduced before Proposition 2?  
   It might be helpful if the authors explicitly describe all quantities appearing on the right-hand side.  
   Also, is $M_{t_{k+1}}$ exponential in the number of steps?  
   (b) It seems that the right-hand side does not vanish even if $\Delta t \to 0 $.  
   Can the authors still claim that $p^{(S)}$ and $p^{(R)}$ are sufficiently close under some limiting regime?

3. **On Assumption 8**  
   Is Assumption 8—stating that the mapping $\Psi_t$ is twice continuously differentiable—reasonable or realistic in practice?  
   One of the main strengths of your work is that the proposed MCTS framework can handle **non-differentiable rewards**.  
   Therefore, it would be desirable if the underlying diffusion dynamics (and the mapping $\Psi_t$ ) could also relax the differentiability requirement, so that both the diffusion model and the reward function may be non-differentiable.


---

**References**

[1] Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, and Rajesh Ranganath. *A general framework for inference-time scaling and steering of diffusion models.* arXiv preprint arXiv:2501.06848, 2025.  

[2] Uehara, M., Zhao, Y., Black, K., Hajiramezanali, E., Scalia, G., Diamant, N. L., ... & Levine, S. (2024). *Fine-tuning of continuous-time diffusion models as entropy-regularized control.* arXiv preprint arXiv:2402.15194.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies inference-time alignment of diffusion models with respect to general reward functions. The key idea is to fix the process noise and treat the interpolation coefficients between ODE and SDE updates as optimization variables, a scheme we term RODE sampling. These coefficients form a low-dimensional vector that we optimize using Monte Carlo tree search. Experiments first confirm that RODE sampling provides sufficient sample diversity for effective exploration, and then demonstrate strong performance on several text-to-image diffusion alignment tasks.

### Strengths
1. Framing the ODE–SDE interpolation factors as optimization variables is an elegant idea. This parameterization is inherently low-dimensional and parameter-efficient, making it a natural fit for bandit-style algorithms such as MCTS.
2. The authors validate the method’s effectiveness across diffusion models with diverse architectures.

### Weaknesses
1. Baseline comparisons are limited. It’s unclear how RODE fares against simple baselines, especially best-of-n sampling under pure ODE or pure SDE trajectories. Since the process noise is fixed, the method largely behaves like a local search (though augmented RODE partially mitigates this), raising doubts about its advantage over global search strategies such as best-of-N.
2. The paper lacks an analysis of optimization time, which is critical for assessing practicality. Reporting wall-clock runtime, the number of reward evaluations, MCTS budgets, and how costs scale with image resolution and model size would greatly clarify the method’s efficiency.

### Questions
Some works like DNO explore gradient-estimation optimization approach for non-differentaible reward. How do you compare the bandit-type algorithm like MTCS to those gradient-estimation based optimization approach?

DNO: Tang, Zhiwei, et al. "Inference-Time Alignment of Diffusion Models with Direct Noise Optimization." Forty-second International Conference on Machine Learning.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper developed a randomised ordinary differential equation (RODE) serving as value estimation for the MCTS algorithm. The theoretical guarantee ensures that the Lipschitz constant bounds the mean of the distribution between SDE and RODE.

### Strengths
## Presentation: ~45th percentile

The mathematical writing and explanation are intelligible, but they have some issues (see below).

## Soundness: 30th-70th percentile

This paper provides a theoretical guarantee for RODE to estimate the MCTS value, offering a transparent analysis with mathematical arguments.

## Contribution: 40th~75th percentile

The formulation of RODE is new to me and may be useful for further work other than alignment tasks. 

## Note:

I hope the AC is aware that the rating is calibrated using estimation of percentiles to reduce evaluation noise effectively.
The rating is simply the mean of the three aspects.

### Weaknesses
## Presentation

### Writing

My reading flow was sometimes interrupted due to the lack of connection between sentences/paragraphs. For example, your abstract reads like a list of bullet points without a strong hierarchical presentation; the structure of the first introduction paragraph is

```
for work from a list of literature:

  Introduce literature

  Explain the weakness of the literature, compared to your method.

for work from a list of literature:

  Suggest your solution to address the weakness you mentioned earlier.
```

You can save your readers' effort by putting the weakness and the solution together so that they don’t need to frequently move their pointer while consuming a paragraph of 20 lines.

Such writing is pervasive throughout the manuscript, but I only list the abstract and introduction as they are the most important.

### Literature survey
**Citing Wikipedia** is not considered standard academic practice (and you did it five times). As an encyclopedia, it is a tertiary source that summarises information and is the last type of citation source. You must conduct a thorough literature review and replace this citation with the original peer-reviewed sources that contain the foundational knowledge. I have to admit that these citations leave a terrible impression when reviewing this paper, although I have to make it independent of judgement of other aspects.

### Minors
1. I suggest using Big-O/littel-O notation in equation 10.
2. In Equation 8, it would be better if you juxtapose the original SDE induced by DDIM and the RODE, so your reader can compare them.
3. If your work does not involve a reward encountered during the process (despite the reward at t=0), it is a bad idea to introduce them in Section 3.1 and throughout your entire paper, as I feel confused and effortful when reading the paragraphs from lines 160-179.


## Soundness
**ODE-based baseline:** It is known that DDPM, DDIM, and any score-based SDE can be reformulated as Karra’s SDE [1] bidirectionally. Therefore, these formulations are theoretically equivalent, although the conversion in practice is not very trivial. Assuming the equivalence, you state

> We are the first to extract an RODE sampling from the DDIMs

> To the best of our knowledge, we are the first to model DM alignment as a max-reward control/RL problem.

I expect you to clarify the difference between your and ODE-based value estimation baselines [2, 3] in your paper, and the experimental comparison of them is also missing. I raise a doubt about your performance because the figures in [3] are significantly larger than yours.

[1] Karras, Tero, et al. "Elucidating the design space of diffusion-based generative models"

[2] Ma, Nanye, et al. "Inference-time scaling for diffusion models beyond scaling denoising steps"

[3] PH, Yeh et al. “Training-free Diffusion Model Alignment with Sampling Demons”

### Questions
See weaknesses

### Soundness
2

### Presentation
2

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
This paper proposes a novel inference-time scaling method for diffusion model alignment via modifying the denoising trajectory with the strength of the added noise as the exploration action (while the added noise is fixed). Compared to the commonly used noise/eps as action (which is high-dimensional), the proposed method adopts such a low-dimensional action that can be easily integrated into UCB-based Monte Carlo Tree Search (MCTS) for efficient online exploration. The authors show (both theoretically and empirically) that this new action space admits less variance than the noise/eps as actions paradigm. This method is plug-and-play for various reward signals (including non-differentiable ones).

### Strengths
+ The formulation of using the perturbation strength as an action while fixing the added noise is novel and intriguing. I like the explanation and the visualization of the reduced variance. 
+ I am a bit surprised the method, while simple conceptually, can do inference-time scaling pretty well. A major challenge in visual diffusion model fine-tuning is the high-dimensional action space. Since the action space is low-dimensional, it might open up a door for other online tuning methods, such as policy optimization (also optimizing the diffusion model itself).

### Weaknesses
+ I am not particularly impressed by the claimed advantage of parameter efficiency. Can the authors point out scenarios where it is critical, given that eps-based action space still performs better in some cases (maybe not for MCTS)?
+ I am curious if the authors can provide more evidence that the proposed method alleviates reward hacking compared to eps-action alternatives. This will make the contribution of this work significant.

I am willing to raise my score after seeing the authors' response.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
