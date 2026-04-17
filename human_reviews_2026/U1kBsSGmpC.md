# Online learning of multidimensional distributional maps for rapid policy adaptation

- Decision: Reject
- Scores: 4, 2, 4, 6, 0

## Abstract
In order to flexibly behave in dynamic environments, agents must learn the temporal structure of causal events. Standard value-based approaches in reinforcement learning 
(RL) learn estimates of temporally discounted average future reward, leading to ambiguity about future reward timing and magnitude. Recently, midbrain dopamine neurons (DANs) have been shown to resolve this ambiguity by representing  distributional predictive maps of future reward over both time and magnitude in the encoding of reward prediction errors. However, the computational function of such time-magnitude distributions (TMD) in the brain is unknown. Here we present online learning rules for acquiring information-maximising multidimensional distributional estimates,  extending classic work in distributional RL from 1D return distributions to efficient representations of distributions of arbitrary dimensionality. In previous distributional RL approaches, the distributional information is largely used for improving representation learning. In our framework, TMDs are the direct substrates for simple policy decoders, enabling rapid risk-sensitive action selection in environments with rich probabilistic temporal reward structure, even under distributional shifts. Finally, we present cross-species neural and behavior evidence, from rodents and humans, consistent with the implementation of this theory in biological circuits. Our results advance a principled computational link between distributional RL and neural coding theory, and establish a role for multi-dimensional distributional predictive maps in rapidly generating sophisticated risk-sensitive policies in environments with complex, multi-modal, distributions of future reward.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper lies at the interface of machine learning and neuroscience. The authors present online learning rules for acquiring estimators of multidimensional return distributions in RL, which they posit can be used for efficient risk-sensitive decision-making. They also provide cross-species evidence suggesting that their proposed mechanism is consistent with biological behaviors.

### Strengths
* The proposed method is concise and intuitive.
* The authors' claims are well-supported by a comprehensive suite of experiments.

### Weaknesses
**Reviewer's Note:** I am a researcher working on reinforcement learning and do not have expertise in brain science. My comments are therefore offered from a machine learning perspective. It is possible that I do not correctly understand some claims in the paper or fully appreciate the significance of the paper's contribution to the field of brain science. Hence, I assign a low confidence score to my evaluation.

* The paper is challenging to follow for readers without specific domain knowledge in neuroscience. To improve accessibility, I strongly recommend adding concise introductions for key concepts (such as efficient coding theory, DAN, and TMD) and including brief overviews of the experimental designs directly in the main text.

* The algorithmic contribution appears limited. In fact, there are already works in distributional RL that address the multi-reward settings like [1], [2], and [3]. The authors should clarify how their approach differs from or improves upon these existing works.

* The related work section currently overlooks relevant bodies of work on generative models (e.g., VAEs, GANs, diffusion models) and other recent advances in distributional RL. I encourage the authors to conduct a more thorough review to better position their contributions within these contexts. 

[1] Distributional Reinforcement Learning with Regularized Wasserstein Loss
[2] Distributional Multivariate Policy Evaluation and Exploration with the Bellman GAN
[3] GAN Q-learning

### Questions
The authors combine a KL objective with Wasserstein proximity to update distribution representations. Could the authors please clarify the main advantages of this specific technique? A brief discussion justifying this choice over other distributional update rules would be beneficial.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Inspired by neuroscience, this paper proposes the online updating rule of the time-magnitude distributions (TMD) by using multi-dimensional distributional RL. The proposed algorithm enables rapid risk-sensitive action selection in environments. This work further connects distributional RL and neural coding and experiments are performed on neuroscience and gridworld environments.

### Strengths
1.This paper further strengthens the connection between distributional RL and neuroscience, which provides a new computational perspective in this area.

2.The illustration looks impressive, with many new environments from the neuroscience.

### Weaknesses
1. **Limited Motivation and Technical Contribution**. This paper seems to be an application of the multi-dimensional distributional RL algorithm to develop the online update rule of time-magnitude distributions (TMD) and the modifications are straightforward. From the perspective of (distributional) RL, this paper may not be well-suited to ICLR, as the focus is mainly on the neurocoding side. In particular, there are already many multi-dimensional distributional RL algorithms, such as [1, 2, 3], that present competitive performance in deep RL settings. Thus, it is less motivated to posit the contribution of this paper in the distributional RL literature from the perspective of technical contribution. It may be a useful extension in the neurocoding theory, which is beyond my expertise. 

2. **Unclear writing**. It is less clear to me about many technical words and descriptions in the context of neuroscience and neurocoding theory. For instance, what is the time-magnitude distribution in a mathematical way? Why do the authors claim that distributional RL is largely used for improvement for improving representation learning in Line 22? Any reference about that? There are also a lot of risk-sensitive RL papers that are related to this work, which has not been sufficiently investigated here.

3. **Limited experiments and insufficient explanations**. The references on the figures are provided without detailed explanations. Thus, it is easily confused about the connection between the experimental results and the claim the authors want to make. Some explanations are only given in the captions, but detailed elaborations are more important in the main content. In addition, the experiments are only performed on some neuroscience and gridworld environments in the tabular setting, which is far beyond the conventional environmental setup. Large-scale environments are almost necessary, especially in RL literature. 


## Reference
[1] Zhang, Pushi, et al. "Distributional reinforcement learning for multi-dimensional reward functions." Advances in Neural Information Processing Systems 34 (2021): 1519-1529.

[2] Sun, Ke, et al. "Distributional reinforcement learning with regularized wasserstein loss." Advances in Neural Information Processing Systems 37 (2024): 63184-63221.

[3] Wiltzer, Harley, et al. "Foundations of multivariate distributional reinforcement learning." Advances in Neural Information Processing Systems 37 (2024): 101297-101336.

### Questions
Beyond the question in the weaknesses, here are some other questions.

1. What are the benefits of adding a Wasserstein regularizer? How does it address the limitations of Zhang et al 2021 mentioned in Line 102? In my opinion, the motivation is not clear.

2. It is not clear why the authors introduce the definitions in Eq.3. The authors mentioned an entropic regularization in Eq. 5. What is the overall objective function, and is there any relationship or difference between the Sinkhorn divergence proposed by [2]?

3. The authors mentioned SM many times in the paper, but there is no explicit reference on the section, making it unclear about the detailed section in the appendix.

4. The explanation in Section 2.3 is described in a sloppy way, which is hard to follow.

5. The experimental details in the context of neuroscience should be elaborated further. For example, the patch environment used here is not standard in RL literature. For the current version, the choice of environment is too specific, which is tailored for neuroscience readers.

6. Why not follow the experimental setup in Implicit Q-learning to evaluate the risk-sensitive behavior in Section 5?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper focuses on an empirical finding of previous study (Sousa et al.,2025), that the midbrain dopamine neurons in the brain encode the time-magnitude distributions (TMD) of (multi-dimensional) rewards. 
The core goal of this paper is to learn the TMD distributions efficiently, the paper proposes an online method that minimizes the Kullback-Leibler (KL) divergence with W_2 distance regularization. 
Through extensive experiments, the paper demonstrates that the proposed method outperforms solutions based on traditional distributional reinforcement learning modeling.

### Strengths
In my view, the problem considered in this paper is valuable: how to make decisions when rewards can only be observed after a certain delay. 
The paper provides a model and an algorithm for this problem and conducts experiments across several well-designed experimental environments. 
The experimental results demonstrate the effectiveness of their method compared to traditional approaches including quantile-based distributional RL algorithms.

### Weaknesses
As a reviewer without expertise in neuroscience, I find it difficult to accurately assess the contributions of this paper. From the perspective of machine learning alone, the approach and analysis proposed in Section 2.1, using W_2 regularization to prevent collapse and maintain diversity, is a common practice. It is also predictable that this regularization would lead to smoother solutions. 

I recommend that the authors consider submitting this work to more specialized neuroscience journals (e.g., Nature or its sub-journals) to obtain more accurate evaluations .

### Questions
In some experiments, such as those presented in Figure 3, the setup appears to violate the Markov property required for RL algorithms. Given this, it seems that RL algorithms—whether distributional or traditional methods that only model value functions—should be unable to learn optimal policies. Is my understanding correct?

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
3

### Summary
This paper considers the problem of learning and using multi-dimensional reward distributions. It asks how to represent such distributions, how to learn the representations, and how to use these representations for action selection. To address these questions, the paper brings together approaches from three related disciplines: distributional RL, efficient coding, and risk-aware decision making. The authors derive a learning rule that encourages proximity to a target distribution but also efficient trajectories of approaching the target, which improves multi-dimensional distribution learning. They then show that this rule is consistent with empirical dopamine neurons, that agents that learn time-magnitude distributions outperform agents that don’t in tasks with particular reward structure, and how knowing the time-magnitude reward distribution can support risk-aware decisions.

### Strengths
This has the potential to be a good paper. It’s an interesting question, a nice approach, and the results are generally solid. The learning trajectory regularization is clever. It’s great that there’s a comparison to empirical dopamine neuron activity. The results on agents that learn time-magnitude distributions are convincing (although by design: learning a full enumerated state space must be slower than one that has part of the true reward structure baked in). The asymmetric discounting that only works for multidimensional and not for factorised discounting is a neat insight; I’m not too familiar with this literature so I’m not completely sure it’s novel, but it fits well within the paper’s story regardless. Overall, I think the experiments and results are useful though not groundbreaking. I'm leaning to accept.

### Weaknesses
As mentioned above, I think the controls are on the weak side. While this is useful to illustrate qualitatively what the new methods brings, it makes it hard to estimate the impact when evaluated against less limited models. Specifically, this applies to the factorised distribution in Fig 1b, the lack of control methods in Fig 2c, the standard/QR RL models in Fig 3b,e and the non-converged agents in Fig 3c,f. Additionally, it would be in the authors’ best interest to thoroughly proofread the manuscript before submission, as in its current form it is unnecessarily difficult to read, which risks alienating both reviewers and readers. In particular the repeated mislabelling of figures caused confusion.

### Questions
Major comments

The various writing mistakes, from grammar and typos to repeated figure panel mislabeling, contributed to making this paper difficult to read at times. I’ve included examples of such cases from just the first two pages below this review, but across the document it’s hard not to get a feeling of a lack of attention to (and proofreading of) the writing. There are multiple instances where a careful check of the text and figures could have avoided confusion. Figure panels are not referenced (e.g. Fig 1b is never mentioned in the text) or don’t exist (e.g. Fig 1d is referenced but there are only three panels in Fig 1; Fig 2d is referenced but again doesn’t exist; Fig 4d-f referenced but don’t exist (line 424); Fig 4g-l referenced but don’t exist (line 461)) or are mislabeled (e.g. Fig 1c in the main text seems to refer to the result of Fig 1a; Fig 4b in line 416 seems to refer to Fig 4c; Figure 4b in line 424 doesn’t seem to refer to Fig 4b). The fact that the references are very messy, with many items occurring multiple times in slightly different formats, doesn’t help to convey an impression of attention to detail either. 

Figure 1a is referenced once, directly before Equation 4, but the result in the panel seems unrelated to Equation 4. It took me a while to understand what was plotted in Figure 1a because the caption is very minimal, and the main text also didn’t explain the result of that panel. Then I got to section 2.3 which seems to exactly explain Figure 1a, but it refers to Figure 1c. Moreover, this section 2.3 comes after section 2.2 which explains the results of Figure 1b and 1c. The fact that 1) the reference to figure panels in the main text doesn’t match the actual figure panels and 2) the results in the main text are in a different order from the figure panels made this confusing.

I don’t think the factorized 1D distribution learning is a particularly strong control in Figure 1b. It seems obvious that a 1D distribution learning method won’t be able to learn correlations between variables. Maybe it would be useful to add another multi-dimensional method, like MMD, as a panel? That would potentially make the point that while MMD can learn the same distribution as DNL (which this new panel would show), its learning trajectory is different (as shown in current panel 1c). 

Does the gamma before vs gamma after result in Fig 2c only appear in DNL? Would other distribution learning methods, or even a factorized 1d distribution learning method, not produce the same result?

The results in Fig 3c,f are superfluous because they can be directly read from 3b,e by only looking at the final datapoint. It feels a bit unfair to only compare performance for a fixed number of timesteps – it is expected that an agent with access to a usefully structured state space learns faster than an agent that must enumerate all states, and this difference is already shown in 3b,e. The fairer comparison in 3c,f would be to compare performance after convergence of all agents.

Minor comments

147 What is \Delta_i ?

042 Unclear what TMD abbreviates as the preceding words don’t match the letters “extend the 1D reward magnitude code to a 2D time-magnitude “map” of future reward in a  distributional format (TMD)”.

047 “In particular, in the naturalistic scenario of choosing between actions leading to probabilistic rewards generated by the environments with intricate temporal structure and distributional shifts.” > Missing a verb.

098 “proposed maximum mean discrepancy (MMD) based multidimensional distributional algorithm similar to DNL” > missing “a” maximum (…) algorithm.

099 DNL acronym used without prior introduction.

102 “We address this by adding a Wasserstein regularizer that penalizing distortions in the population representation” > that “penalises”.

104 “which has three important consequences (1) extends efficient coding to higher dimensions (Schütt et al., 2024), (2) preserves the population coding when the reward distribution changes as observed in many brain regions across multiple modalities and (3) enables flexible decoding” > missing “:” after consequences, and “it” before (1) extends

211 Absorves > absorbs

456 (c) > (e)

Fig 4b No legend for blue bar (I guess it corresponds to TMRL in panel a but would be useful to be explicit)

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
Contemporary RL work mostly focus on discounted future return and there has been lack of studies focusing on the impact of reward timing and reward magnitude. In this paper, the authors addressed these limitations by introducing multi-dimensional distributional neural learning (DNL) which is based on optimal transport theory.  According to the authors, the parameters of the tuning functions are learned using DNL, which the results also align with empirical observations relating to distributions of reward times and magnitudes. After the parameters of the tuning functions have been learned, the time-magnitude distributions (TMD) can then be approximated and actions are selected based in the TMD. The authors also presented a grid world setup where the TMD can be combined with Successor Representations to learn the value function, resulting in the TMRL agent. Compared to the standard Q-learning agent and the distributional RL agent, the TMRL manage to learn more effectively, even as the number of stimuli increases. Furthermore, the authors also extend the studies to include certain and risky rewards and showed that the Multidimensional representations yield better results.

### Strengths
1.  Idea of the paper involves Reinforcement learning and provide insights into representations that align well with empirical data.
2. Most of the important equations are provided in the main paper.

### Weaknesses
1. The paper is poorly written. There are a lot of complex concepts in the paper that were not introduced gracefully. Furthermore, some concepts were not explained why they are required. For example, why was Successor Representations required for the gridworld experiment? Why was the w variable required? 

2. For readers who are not familiar with prior work, reading this paper is very challenging. The flow between some of the paragraphs were not very fluid. For example, the w variable was not defined when first introduced at line 361.

3. Some of the figures caption are labeled incorrectly or mentioned in the main text but the figure is non-existent. For example, where is Figure 1d (line 220)? Figure 4e has been mistakenly been labeled as Figure 4c in the caption. Line 345 mentioned Figure 3d as a results but Figure 3d is a figure of the grid world environment. 

4. With the above reasons and the current state of the paper, at this moment, I feel that the paper is rushed and not ready for publication.

### Questions
1. What is DNL? (Line 99). Looks like it is "Distributional neural learning". The abbreviation should be introduced much earlier, rather than at line 203. 
2. The last caption in Figure 4 should be (e) instead of (c). 
3. What is the dimension of the output of the TMD function? It is not stated clearly. 
4. It is unclear how action selection is done for the TMRL agent. After some digging, the information can be found in the appendix but the authors should have mentioned about the existence of the pseudocode in the main paper. 
5. I also listed some questions / remarks in the weakness section.

### Soundness
3

### Presentation
1

### Contribution
2
