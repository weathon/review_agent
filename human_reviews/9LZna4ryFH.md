# A Tailored Framework for Aligning Diffusion Models with Human Preference

- Decision: Reject
- Scores: 6, 6, 1, 6, 6

## Abstract
The direct preference optimization (DPO) method has shown success in aligning text-to-image diffusion models with human preference.
Previous approaches typically assume a consistent preference label between final generated images and their corresponding noisy samples at intermediate steps, and directly apply DPO to these noisy samples for fine-tuning. However, we identify a significant issue with this consistency assumption, as directly applying DPO to noisy samples from different generation trajectories based on final preference order may disrupt the optimization process. We first demonstrate the issues inherent in previous methods from two perspectives: *gradient direction* and *preference order*, and then propose a **Tailor**ed **P**reference **O**ptimization (TailorPO) framework for aligning diffusion models with human preference, underpinned by some theoretical insights. Our approach directly ranks the preference order of intermediate noisy samples based on their step-wise reward, and effectively resolves the optimization direction issues through a simple yet efficient design. Additionally, to the best of our knowledge, we are the first to consider the distinct structure of diffusion models and leverage the gradient guidance in preference aligning to enhance the optimization effectiveness. Experimental results demonstrate that our method significantly improves the model's ability to generate aesthetically pleasing and human-preferred images.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies a key weakness in the formation of existing framework that aligns diffusion model with human preference.  The authors noted that if the winning and losing samples are in a linear subspace, it is possible for the gradient updates to take the wrong direction. To address this issue, the authors propose a novel tailored framework which ensures the correct update.  Experiments show that the proposed modification improves the performance of alignment algorithms on a variety of reward functions,

### Strengths
1. The presentation is clear and easy to follow. The problem is well motivated, and concrete derivations are provided.
2. The authors conducted extensive experiments on a wide range of prompts and reward target to highlight the effectiveness of the method
3. The authors provided user study results, which are helpful to contextualize the implications of qualitative results.

### Weaknesses
1. The details of user study are not disclosed and results are not carefully analyzed. While the author provide some overviews in Appendix B, they did not disclose how many user responses are collected from the 225 images generated. This is important because it determines the standard error and confidence interval of the user study, which is crucial to judge the significance of the results. The author also did not disclose the instructions provided to the user, and if proper mitigations are applied to reduce user biases (e.g. randomize image order of different models).

2. While the author showed that the proposed method works well in comparison with respect, to other online methods, but fails to compare against state-of-the art offline methods such as Diffusion-DPO. Online methods are costly in that they require ad-hoc sampling from the diffusion model. Such complexity should be justified.  In particular, Diffusion-DPO also uses Pick-a-Pic dataset to align for human preference.

### Questions
See weakness

Addition question that did not affect the decision:

1. What are the performance on in-distribution prompts? In particular, Table 3 shows that TailorPO-G loses to TailorPO on HPSv2, which is surprising because one would expect with direct gradient guidance from the reward model,  TailorPO-G  should perform better. Is this because TailorPO-G overfits to the training prompts?

### Soundness
2

### Presentation
3

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
This paper proposed TailorPO, a DPO framework tailored for diffusion model based on previous D3PO method. The framework includes three key improvement: 1) Turn the preference ranking on step-level instead of final image level; 2) The preference is only considered on the same condition; 3) Use gradient guidance to increase the difference on the reward between each pair.

### Strengths
- The paper is well-written and effectively organized. The authors provide a thorough analysis of the limitations in existing DPO methods, including challenges with inaccurate preference ordering and gradient direction, and subsequently propose a method that addresses these issues effectively.
- The proposed method is straightforward to understand, with clear an accesible theoretical analysis, particularly in its comparison of formulations with prior methods such as D3PO.
- The experiments examine the generalization capability of the proposed method across different prompts and reward models, a crucial aspect for fine-tuning approaches.

### Weaknesses
- The paper lacks a crucial ablation study on the contribution of each component in TailorPO to its effectiveness. Specifically, it would be valuable to evaluate the individual impact of (1) step-level preference, (2) preference indication only under the same condition, and (3) gradient guidance. This analysis is essential to substantiate the contribution of the overall framework beyond that of its individual components.
- The paper lacks verification of generalization on fine-tuning methods, such as LoRA and full fine-tuning. Additionally, the experiments appear to be conducted solely on SD 1.5; expanding evaluation to include a broader range of base models would strengthen the results.
- Figure 5 indicates an evident over-saturation issue for both TailorPO and TailorPO-G, which does not align with the caption’s description of producing "more visually pleasing images." Conducting a user study could better substantiate this claim.
- The framework may not be data-efficient, as it seems unable to leverage existing dataset including preference pairs of candidates across different conditions.

### Questions
- It may be beneficial to reflect the preference selection procedure in Eq.10, as this equation is regarded as the formulation of the TailorDPO framework.
- Could you provide insights into the differences between TailorOP and TailorOPG?
- I am concerned about the method's performance in real-world applications. Why do TailorPO and TailorPO-G lead to over-saturation issues in Figures 5 and 6?
- The baseline for comparison seems relatively weak (with timesteps set to 20 for DDIM, which is uncommon in practice). If available, could you share results using the typical choice of 50 timesteps for DDIM or 25 timesteps for other advanced schedulers?

I'm willing to increase my rating if the concerns can be addressed.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
DPO is very useful for LLMs. Its use in diffusion models is still under exploration. This paper studies how DPO can be used in the context of T2I diffusion models. The important consideration is that existing methods assign the same win/lose pair to all the intermediate steps. This is problematic. This paper thus generated two images at each step and compares the two images to obtain win/lose labels. The authors propose a way to compute such preference at later steps.

### Strengths
This paper's writing looks fine. The problems are presented in a right way.

### Weaknesses
- The proposed framework in Fig. 1(b) is almost the same as SPO (Liang et al., 2024). It is different in that SPO has a hyperparameter defining the number of images generated at each step, but this is very minor. 

- Authors claim that this is the first framework that explicitly considers the properties of diffusion models for DPO. This is very wrong, because the same has been done in SPO. 

- I understand that ArXiv papers may not be required to be cited, but it is important to acknowledge people's contribution in a proper way. It's very inappropriate to propose an identical method while claiming you are the first. 

- The only difference, if I'm correct, is enhancing the diversity of noisy samples by increasing their reward gap. However, this is not claimed as the major contribution of this work. In fact, if this is the main contribution, this paper is not as problematic as now. The only critique would be incremental novelty etc. 

- The comparison results of SPO look problematic too. There is almost no improvement of SPO over D3PO etc. This is very different from the report from SPO. Authors must carefully check the SPO paper and their open-source implementation.

### Questions
This paper should properly claim previous works.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
A. The paradigm of formulating the backward diffusion process in reinforcement learning, as contributed by DDPO, Diffusion-DPO, and D3PO, is well explored.

B. In reinforcement learning, reducing fluctuations in advantage estimation by introducing value function is unanimously critical and well-explored in GAE and its related works. A more accurate estimation of advantages benefits reinforcement learning algorithms by applying policy optimization methods like PPO.

This work, TailorPO and TailorPO-G, is an instance of A + B among all possible formulations of similar ideas. The loss function used to train the diffusion model (policy) is modified by fixing the intermediate noisy sample $x_t$ to reduce fluctuations in estimating the effect of actions. Similar to GAE, $r_t$ serves as a value function that helps judge the quality of an action (referred to as the preference order in this paper). Backpropagating through the reward function $r$ helps identify the optimal $x_t$ (best/worst) as a data point for preference training.

### Strengths
* This paper claims to be the first to combine A and B, which I regard as a factual novelty under the formulations of D3PO and DDIM.
* The paper introduces the backpropagation of the reward function to generate data points for RL training, which may be useful for generating data points for other preference-learning algorithms.
* The efficacy of TailorPO(-G) is demonstrated with fewer than 10k samples.
* I believe the derivation of $\nabla_{\theta} \mathcal{L}$ is correct in Equation (11).  The neat form of Equation (11) is inspiring to me.
* Equation (12) provides insight into estimating reward expectations. I believe the property is correct and valuable for other applications if the parameters remain intact.
* I acknowledge that the paper is worth publishing, but it falls below the standard expected for ICLR (see below).

### Weaknesses
1. In Equation (12), the fact that $r_t$ can be approximated by $\epsilon_\theta$ **during training** is questionable. It is unclear whether $r_t$ can still be accurately estimated using either $\epsilon_\theta$ or $\epsilon_{\theta'}$ after the model parameter is updated from $\theta$ to $\theta'$, as prior work is proven under the initial $\theta$. This raises concerns about whether the optimization method remains effective after the parameter update, where $\theta'$ is significantly different from $\theta$. The absence of a theoretical argument hinders the soundness.
2. The aesthetic score using DDPO has reached its best quality (>8) with 40k reward queries. Based on point 1., I doubt whether it can practically achieve this score. The authors should demonstrate how their method can achieve the best quality of DDPO/D3PO on ANY of the rewards.
3. I assessed this work as an interpolation anchored from A and B; the novelty is, therefore, upper-bounded. Even if they reach the state of the art, predictable if the technique in B is well-implemented, they bring limited new knowledge to the community. Given the paper's novelty, I have set its upper bound for evaluation marginally above the acceptance threshold. Due to unresolved soundness issues, the paper stands slightly below the acceptance threshold.


Typos I found:
- Line 144.
- Line 312.

Factual Error:
1. Line 154: The DPO method is NOT first proposed to fine-tune large language models to align with human preferences unless the "preferences" refer to preference datasets. In this case, the authors should clarify.

### Questions
1. DDPO/D3PO-related methods cannot retain the Fokker-Planck equation of the original model, where probability flow ODE relies on this property. Can we use a trained policy to generate deterministically like probability flow ODE? Although challenging, this feature could add value if developed.
2. I would consider raising the rating (even above the upper bound) if the paper includes a more thorough formulation. For example, in reinforcement learning, the contribution to the value function is a relatively small extension of GAE (2015); I recommend incorporating at least a TD(λ) formulation. Additionally, as DDIM, DDPM, and other schedulers are subclasses of EDM, reformulating your framework under EDM could enable greater adaptability to alternative schedulers. Although EDM is not yet widely adopted in this community, such a formulation may enhance the paper’s applicability.
3. See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a novel framework, TailorPO, aimed at aligning diffusion models with human preferences by directly optimizing for preference at each denoising step. This approach addresses a significant issue in existing direct preference optimization (DPO) methods, which often assume consistency between preferences for intermediate noisy samples and final generated images. The authors argue convincingly that this assumption can disrupt the optimization process, and their proposed solution is both innovative and timely, given the growing interest in controllable generative models.

### Strengths
1. Originality:
The paper introduces TailorPO, a novel framework that addresses a specific limitation in the application of direct preference optimization (DPO) to diffusion models. This represents a creative advancement in the field of generative modeling. The approach of generating noisy samples from the same input at each denoising step is innovative and directly addresses the issue of inconsistent preference ordering in intermediate steps of image generation.

2. Quality:
The paper is technically rigorous, providing a detailed theoretical analysis of the issues with existing DPO methods and a comprehensive empirical evaluation of TailorPO. The experiments are well-designed and demonstrate significant improvements over other methods in generating aesthetically pleasing images that align with human preferences.

3. Clarity:
The paper is well-organized, with a clear structure that logically progresses from problem formulation to solution proposal, followed by experimental validation.

4. Significance:
The work has high significance as it tackles a critical challenge in aligning generative models with human preferences, which is essential for the practical application of these models. The proposed TailorPO framework has the potential to influence future research in controllable generative modeling.

### Weaknesses
1. The user study, while valuable, is limited in scope with a small number of participants. A larger and more diverse user base would provide more robust evidence of the framework's effectiveness.

2. While the paper compares TailorPO with other DPO-style methods, it would be strengthened by including comparisons with the current state-of-the-art methods.

3. The paper could provide a more detailed discussion on the limitations of TailorPO, such as specific types of images or preferences that the model may struggle with, or scenarios where the framework might not be applicable.

### Questions
1. How does TailorPO perform compared to the current state-of-the-art methods in generative modeling, particularly in terms of image quality and alignment with human preferences? Benchmarking against state-of-the-art methods can provide a clearer picture of TailorPO's performance and its potential advantages.

2. Are there specific scenarios or types of images where TailorPO underperforms or fails to align with human preferences? If so, could the authors discuss these limitations and potential avenues for improvement?

### Soundness
2

### Presentation
2

### Contribution
2
