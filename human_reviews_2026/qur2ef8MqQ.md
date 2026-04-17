# Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence

- Decision: Accept (Poster)
- Scores: 6, 8, 2

## Abstract
Fine-tuning-as-a-service introduces a threat to Large Language Models' safety when service providers fine-tune their models on poisoned user-submitted datasets, a process known as harmful fine-tuning attacks. In this work, we show that by regularizing the gradient contribution of harmful samples encountered during fine-tuning, we can effectively mitigate the impact of harmful fine-tuning attacks. To this end, we introduce Antibody, a defense strategy that first ensures robust safety alignment for the model before fine-tuning, and then applies a safety-preservation learning algorithm during fine-tuning. Specifically, in the alignment stage before fine-tuning, we propose optimizing the model to be in a flat loss region with respect to harmful samples, which makes the safety alignment more resilient to subsequent harmful fine-tuning. Then, in the fine-tuning stage, we design a fine-tuning algorithm that applies a weighting scheme to all samples in each training batch to inhibit the model from learning from harmful samples while encouraging learning from benign samples. Experimental results demonstrate that Antibody successfully mitigates harmful fine-tuning attacks while boosting fine-tuning performance on the user-submitted dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a two-stage defense framework to protect FTaS models from harmful fine-tuning attacks. The first step is regularizing the model on a flat-loss region for harmful samples to make safety alignment more resilient. The second step is the finetuning with weights adjusted for samples based on how harmful they potentially are to further reduce their impacts. Experiments show that Antibody can reduce the harmful score significantly while maintaining accuracy.

### Strengths
- The problem is critical and well motivated.
- The idea of optimizing the model to stay in the flat-loss region with respect to harmful samples is interesting.
- Extensive theoretical analysis is provided.
- Experimental results are promising and the ablation studies are extensive.

### Weaknesses
- In line 196, authors state that among the models that lie in the flat region of the harmful loss L_harm, they aim to find the one that minimizes the alignment loss L_align. But the optimization objective in Equation 3 is not exactly doing this. Opposite in order, it actually finds the weights that minimize harmful loss among the weights that minimizes alignment loss. I think we need more clarity here.
- It is not clear how we should interpret Theorem 4.1 (which is the most important one in this paper) and what a_t is without looking at the appendix. More explanation is needed in the main body.
- In the second stage, it is assumed that harmful samples are promised to be in the flat-loss region and benign samples won’t be in the flat-loss region, which may not hold for benign samples.
- There is additional computation and pipeline overhead that may significantly increase the time and energy spent during the whole process. Analysis on the computational cost of the introduced first and second stage operations with detailed breakdowns are expected. 
- Fig.1 is not displayed correctly (cropped).

### Questions
- What do you mean by a test sample in Proposition 4.2? A sample used in finetuning or testing? We are not supposed to compute gradients for a “test” test sample, am I correct?
- How does the size of refusal dataset impact results?

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
5

### Summary
This paper proposes an elegant solution towards harmful fine-tuning. The solution contains two main components, an alignment stage solution and a fine-tuning stage solution. The design is elegant and with sufficient ablation study to justify each components usefulness.  Experiments show that it outperforms the SOTA solution Booster.

### Strengths
1. The studied problem is important and this papers offer a timely contribution. 
2. The solution is elegant and very fundamental. I especially like the alignment stage component of Antibody, as it enhances safety alignment by adding a few regularizer in the safety alignment stage.   
3. Amplified experimental results are given, showing the robustness of such solutions on different fine-tuning datasets, model and harmful ratios. Also, I particularly like Figure 1, **which clearly demonstrates  Antidote's superiority against existing methods over the trade-off between safety and fine-tune accuracy**. 
4. The authors conduct sufficient ablation study,  system evaluations, and quantitative results to verify the effectiveness of the proposed method.  
5. The visualization in Figure 2 is nice!


In conclusion, this is a solid paper that requires a lot of experiment and sufficient insights and understanding of the problem. I feel pleased and super excited when reading this paper.

### Weaknesses
The paper contains two components, I.e., alignment stage and fine-tuning stage solution. I need to separate them to write the review for each of them for clarity.

For alignment stage of Antibody: 

1. **Relation and contribution compared with Booster need to be clarified.**
The relation with Booster is not explicitly mentioned. Particularly, the authors should clarify that Antibody inherits components from Booster. Let's look at the update rule in  Eq. (11). 

* **Under the case $\lambda_{refusal}=0$ and $\lambda_t = \lambda$ (i.e., a constant), Antibody reduces to Booster**. 

Justification: under the above assumption  on \lambda_{refusal} and  $\lambda_t $,  then, expands $L_{sharp}(\theta)$ based on its definition given in line 222 in Page 5, Eq.  (11) becomes:  

$L_{align}(\theta_t) + \lambda L_{harm}(\theta_t) - L_{harm}(\theta_t- \rho \frac{ \nabla L_{harm}(\theta)) }{||\nabla L_{harm}(\theta)) || } ) $

**This update rule is 100\% the same with Booster's update rule. See Eq. (1) in Booster[1].**

[1] Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation. (ICLR2025)

If the authors  agree that the two methods share similarity at some level, I recommend the authors to clarify this relation at the end of Section 4.1, such that non-experts in this sub-field could better understand what the real contribution is. **Also, given that the final derived update rule of solving Equation (2) is almost identical to Booster, could you clarify what the contribution of Section 4.1 is?**  Feel free to challenge me in the rebuttal if you do not agree with what I claim.


2. **The real contribution seems to be the third term in Eq. (11), which is not clearly articulated**.  As I mentioned above, Section 3.1 (which introduces first two terms in Eq, (11)) arrives at an update rule that is almost identical to Booster. However, by looking at Eq. (11) again, actually Antibody contains an extra term that is not considered by Booster. Particularly the $\lambda_{refusal} L_{refusal}(\theta_{pert,t})$ is a new term that is not considered by existing work. To my understanding, this term is important as it makes sure that the safety alignment loss will not increase after simulating one step of harmful perturbation. That might be the key contribution that guarantee Antiboidy' superior performance against Booster. 

To verify the meaningfulness of this contribution, I suggest the authors to do a group of ablation experiments:

* Disable the fine-tuning stage solution of Antibody, and only keep the alignment part.  Then, set $\lambda_t$ to a constant, making the first two terms reduce to Booster. Finally, replicating a similar experiment settings in figure 1 by comparing $\lambda_{refusal}=0$ (reduces to Booster)  and   $\lambda_{refusal}$=an ideal number.

3. **While they are not exactly the same, a related alignment stage harmful fine-tuning defense paper also uses a similar term with the third term in Eq. (11).** Particularly, the second term in  Eq. (2) in [2] also adopt a similar term, which can be adapted to $\lambda_{collapse} L_{collpase}(\theta_{pert,t})$ based on the notation used by this paper. The difference is in the dataset used for the outer loss calculation.  Antibody is on the alignment dataset and CTRAP is on a specially constructed collapse dataset. I suggest the authors add a sentence to discuss this term in CTRAP paper to mention their similarity.  

[2] CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning
  
For fine-tuning stage of Antibody:

4. **Relation and contribution compared with SEAL[3] need to be clarified.** The idea of the fine-tuning stage solution is to assign different weights for each piece of data in the fine-tuning datasets. The high level idea is similar  to SEAL which assign a learnable weights to each data point. The authors should discuss how the assigned weights are different from SEAL and justify with experiment whether this weight allocation in Eq. (7) are better than the learnable weights in SEAL [3]. Also, there is one NeurIPS2025 accepted paper BDS[4] and two pre-print papers GradShield[5] and Ref-Teacher[6] exploring the same direction, which should also be mentioned in the paper. 

[3] SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection 

[4] Adaptive Defense against Harmful Fine-Tuning via Bayesian Data Scheduler

[5]  GradShield: Alignment Preserving Finetuning  https://openreview.net/forum?id=YYUNm7IibC

[6] Safety-Aligned Weights Are Not Enough: Refusal-Teacher-Guided Finetuning Enhances Safety and Downstream Performance under Harmful Finetuning Attacks  https://openreview.net/forum?id=OK2GR1guwv



5. I have some concerns over the  weights allocation in (7).  I think this weight allocation might incur some issues when the model is under stronger attack. If the attack is strong, the likelihood of the harmful samples of generating the harmful target completion might override that of the refusal completion. In that case. the weight allocation towards harmful samples might becomes higher and even might surpass that of the benign sample. Could you give me some insights on whether this case will happen?

6. **Some related work on harmful fine-tuning defense should be discussed.**

Detecting Adversarial Fine-tuning with Auditing Agents

Scaling Trends for Data Poisoning in LLMs

Unleashing the Unseen: Harnessing Benign Datasets for Jailbreaking Large Language Models

Virus: Harmful Fine-tuning Attack for Large Language Models Bypassing Guardrail Moderation

No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data

Benign Samples Matter! Fine-tuning On Outlier Benign Samples Severely Breaks Safety

Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents

Eliciting Harmful Capabilities by Fine-Tuning on Safeguarded Outputs

Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs

Self-Destructive Language Model

CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning

Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning

LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning

Towards Resilient Safety-driven Unlearning for Diffusion Models against Downstream Fine-tuning

SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection

Safety alignment should be made more than just a few tokens deep

Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs

Shape it Up! Restoring LLM Safety during Finetuning

Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization

Refusal-Feature-guided Teacher for Safe Finetuning via Data Filtering and Alignment Distillation

AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin

Your Task May Vary: A Systematic Understanding of Alignment and Safety Degradation when Fine-tuning LLMs

Defending MoE LLMs against Harmful Fine-Tuning via Safety Routing Alignment

GradShield: Alignment Preserving Finetuning

A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space

Detecting Instruction Fine-tuning Attack on Language Models with Influence Function

Locking Down the Finetuned LLMs Safety

Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation

Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets

Navigating the safety landscape: Measuring risks in finetuning large language models

Fundamental Safety-Capability Trade-offs in Fine-tuning Large Language Models

ESTIMATING WORST-CASE FRONTIER RISKS OF OPEN-WEIGHT LLMS

When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment 

**There may be more relevant works (I just list above some more recent work), and I suggest the authors to read and discuss all of the relevant works on harmful fine-tuning when revising the paper.**

### Questions
See Weakness. Overall I enjoy reading this paper, and I will consider to further improve my score if my concerns are addressed. 

The main concern I have is that this paper should not emphasize too much on the flatness regularization as this is exactly the same with what Booster aim to achieve (also evidenced by the almost identical update rule between Antibody and Booster), though it seems that the constraint problem Eq. (2) offers an alternative explanation of the flatness regularization and also by this transformation you seem to be able to automatically and adaptively derive the regularizer intensity $\lambda_t$, which seems to be a novel contribution. You may talk more about this if you agree to mention Booster formulation in this section.  **I also suggest to add a group of experiments to justify that the step-adaptive regularizer intensity is indeed necessary and outperform the constant intensity adopted by Booster**, if you still insist that your Section 4.1  should be the main contribution. 

The third term in (11) (i.e.,the term in Eq. (10) is more interesting and I personally think is the main contribution instead.  

Despite I do have some  complaint over the writing issue, I think this paper is a very solid work and I am happy to champion this paper in the review process.

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focused on harmful fune-tuning attacks. The proposed method regularizes the gradient contribution of harmful samples. The methodology has phases that apply precaution before fine-tuning, and also during fine-tuning with a safety-preservation learning algorithm. In the first step, it makes sure that the harmful sample gradients will be in a flat loss region. In the second stage, it applies weights to the samples during the training batch to prevent learning from harmful samples.

### Strengths
- Clear motivation and explanation
- The paper targets a significant problem.
- The proposed method shows greater performance than the SOTA methods in some datasets.

### Weaknesses
- It uses the first stage to score the samples during fine-tuning. It looks like more of a single stage, where the second stage is just an extension of the first stage.

- The paper improves the alignment stage with a refusal loss, which is already used and has shown its effectiveness in the Vaccine. There is only one ablation study given for the comparison of $L_{sharp}$ and $L_{refusal}$.

- Assuming $K^t$ to be large in benign and $K^t$ would be small in harmful is vague and not supported enough; therefore, reducing equation 4 to 5 with the given proposition is unclear.

- Solution for detecting harmful samples is ad. hoc. It shows that the model is well fit to the training data, which contains the phrase “I cannot fulfill your request” 

- The authors claim that the harmful loss stays constant while the benign loss decreases. However, the third plot in Figure 2 shows that, Antibody harmful loss value is high as the SFT-Harmful loss value.

- The proposed algorithm is not showing the best performance in all the datasets and for all the models, where it also uses previous methods in their strategy.

### Questions
- "A flat loss landscape for harmful samples makes the instilled safety behavior more difficult to remove." Can the authors define how this makes it difficult to remove? One needs more training and more samples?

- The final parameter, $\theta$, depends on the $L_{sharp}$ which depends on $L_{harm}$. To me these two losses are contradictory and their gradients could direct two opposite directions, but the methodology ends up selecting a gradient that minimizes $L_{harm}$. Overall the model is tuned towards the harmful region, but not so harmful? In other words, how do we know a gradient $\delta_t$, shown in equation 3, exists when the gradient of both losses points to exactly opposite directions? 

- How is the matrix multiplication is performed in equation 6 with the given dimensionalities?

### Soundness
2

### Presentation
3

### Contribution
2
