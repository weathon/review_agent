# This State Looks Like That: Self-Interpretable Reinforcement Learning Agents using Prototype Soft Actor-Critic

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Reinforcement learning (RL) has achieved remarkable success across complex decision-making tasks, especially with the advent of deep neural networks. However, the resulting models are often opaque, making their deployment in safety-critical domains challenging. Explainable AI aims to address this issue, but most specific efforts for deep RL remain limited either to post-hoc explanation methods or to imitation learning and distillation procedures. These latter approaches rely on pre-trained black-box agents and are typically restricted to environments with discrete action spaces, limiting their scalability and interpretability. In this paper, we introduce ProtoSAC, a novel deep RL architecture that integrates a prototype-based actor into the Soft Actor-Critic (SAC) algorithm, enabling intrinsic interpretability in continuous action spaces. Our method learns a set of prototypes that represent interpretable state clusters, each associated with a Gaussian action distribution. Actions are generated as a similarity-weighted mixture over these prototypes, providing transparent decision-making without sacrificing performance. We evaluate ProtoSAC on continuous action-space environments and show that it matches the performance of the original SAC while offering enhanced interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ProtoSAC, a novel deep reinforcement learning architecture for continuous control tasks that is intrinsically interpretable. It integrates a prototype-based actor into the Soft Actor-Critic (SAC) framework. The agent learns a set of representative "prototype" states, each associated with an action distribution. Actions are then generated as a similarity-weighted mixture of these prototype policies, making the decision-making process transparent. The authors demonstrate that ProtoSAC matches the performance of the original SAC on several benchmark environments while providing clear explainability, a significant step beyond post-hoc methods or approaches limited to discrete action spaces.

### Strengths
1. Impressively matches the performance of a strong baseline (SAC) without a significant trade-off, directly addressing a key challenge in XAI.
2.   The "this state looks like that" reasoning framework is highly intuitive. Visualized prototypes and their weights offer clear, actionable insights into the agent's policy.

### Weaknesses
1.  The method introduces computational overhead compared to standard SAC due to similarity calculations, extra loss terms, and prototype management. The paper could benefit from a more detailed analysis of this overhead and its scalability.
2.  The model introduces several new hyperparameters (e.g., number of prototypes `K`, update frequency `M`, regularization coefficients `γ`). A sensitivity analysis or ablation study on these would strengthen the paper's claims and improve reproducibility.
3.  The authors note high variance in early training, which may be tied to prototype initialization. The work could be improved by exploring more robust initialization strategies to increase stability.

### Questions
1.  Have you investigated the trade-off between the number of prototypes `K`, model performance, and the granularity of the explanations? Could `K` be adapted dynamically based on environment complexity?
2. The current "hard replacement" strategy for updating prototypes might risk forgetting rare but critical edge cases. Have you considered "soft" update mechanisms, such as slowly updating a prototype's embedding with new, similar state features?
3. What is your perspective on scaling ProtoSAC to high-dimensional state spaces, such as image-based inputs (e.g., Atari)? What would be the main challenges for the encoder design and the similarity metric in such scenarios?

### Soundness
2

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
4

### Summary
The paper introduces ProtoSAC, a variant of Soft Actor-Critic which integrates a prototype framework during training time rather than posthoc. ProtoSAC enables yielding intrinsic, case-based explanations by factorizing the action space as a similarity-weighted mixture of per-prototype Gaussians. Experiments on four continuous action space environments shows the method largely matches SAC performance while providing the above mentioned interpretability benefits.

### Strengths
* Prototype based SAC is both intuitive and enables drop in modification.
* For the environments tested, performance appears to hold versus the black box, with performance generally higher then the Shared-PW-Net baseline.
* The paper is generally well written and easy to understand.

### Weaknesses
* **W1** The novelty of the word is quite limited, appearing to be a straight forward application of prototype based frameworks to SAC. As the authors acknowledge, their work is very related work PW-Net and Shared PW-Net, the functional difference rl training versus distillation / imitation learning.
* **W2** The authors argue that distilling from a black box "limits the learning capabilities to what the black-box model has already learned," which is true, but no experiments in this paper convincingly show that directly applying PW nets during RL training is more scalable then distillation.
* **W3** The experimental results in the paper are extremely limited. Pendulum, Lunar Lander, Mountain Car, and Inverted Pendulum are broadly considered toy environments. This directly ties into W2: if the novelty of the approach rests on reinforcement learning being more scalable than imitation learning /distillation for learning prototype nets, experiments on more complicated environments should validate this. I would expect at the minimum experiments on MuJoCo.

* **W4** The work would benefit from a user study considering the primary objective of introducing interpretability to SAC. Anecodes in the form of Figures 3/4 are helpful but not enough to validate the utility of the learned prototypes.

* **W5** The ablation study as shown in Figure 5 is not convincing. Removing orthogonal loss (blue) or the entropy loss (green) does not seem to have an impact on final performance. The claim that, "These findings suggest that the orthogonal loss and the negative entropy loss work in a complementary way: the orthogonal loss promotes diversity among prototypes, ensuring better coverage of the
state space, while the negative entropy loss encourages the model to rely on more than one prototype
with high similarity. Together, they help achieve more robust and generalized policies." Its unclear how Figure 5 shows either of these claims.

### Questions
1. W2/3 would be cleared if the authors are able to demonstrate the method outperforms distillation / imitation learning on more complex environments, such as MuJoCo HalfCheetah-v4 or Procgen CoinRun. 
2. Can the authors clarify what evidence supports their claims on line 645? It is unclear to me how variance in return during training implies better diversey among prototypes and better state space coverage.  
3. How does the perceived interpretability of ablation variants compare to the proposed method? A user study here appears to be necessary.

### Soundness
3

### Presentation
2

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
This paper introduces ProtoSAC, a novel self-interpretable reinforcement learning framework that integrates a prototype-based interpretation mechanism directly into the Soft Actor-Critic (SAC) framework for continuous control tasks. Its primary contribution is an intrinsically interpretable architecture where the actor's policy is defined by a similarity-weighted combination of learned prototypes, each representing an interpretable state cluster with an associated Gaussian action distribution. This design provides transparent, case-based decision-making without relying on post-hoc explanations or imitation learning from a black-box agent. The authors demonstrate that ProtoSAC achieves performance competitive with the standard SAC baseline and outperforms existing self-explainable methods like Shared-PW-Net across several benchmark environments, while also offering a prototype update mechanism and regularization losses to enhance interpretability and stability during training.

### Strengths
1. This paper presents an application of a prototype-based explanation framework (referred to as "ProtoSAC") to deep Reinforcement Learning, specifically within the Soft Actor-Critic (SAC) algorithm. The following is a broad assessment of its strengths across key dimensions.

2. Originality: The originality of the paper is moderate. The originality of this work is low. The paper simply combines two well-established techniques: the Soft Actor-Critic (SAC) algorithm and prototype-based explanation methods. Neither component is significantly modified or improved upon. The integration of these elements is straightforward and does not represent a novel algorithmic or theoretical contribution. The approach applies an existing interpretability paradigm to a new domain but fails to demonstrate substantive innovation in either the RL algorithm or the explanation framework itself.

3. Quality: The technical quality of the work is adequate but could be strengthened. The experimental validation, while demonstrating the method's basic functionality, is somewhat limited in scope, relying on a narrow set of environments. A more comprehensive evaluation, including comparisons to other explanation baselines and a deeper, more quantitative analysis of the prototype fidelity, would be necessary to robustly validate the framework's effectiveness and general applicability.

4. Clarity: The paper is generally clearly written, effectively motivating the need for explainability in RL and providing a coherent high-level overview of the ProtoSAC framework. However, the clarity is occasionally hindered, particularly by a central framework diagram that is not fully intuitive. The flow of how prototypes are generated and then utilized for explanation could be described with more precise, step-by-step detail to improve reader comprehension.

5. Significance: The significance of the work is limited. While applying prototype-based explanations to deep RL is a novel combination, the insights gained are not particularly surprising or impactful. We already have a strong understanding from other fields that prototypes can cluster behaviors and identify decision patterns. Applying this method to an RL agent merely confirms these known properties in a new context, without yielding any novel or counter-intuitive findings about the agent's learning process. The results are confirmatory rather than groundbreaking, making the overall contribution feel incremental.

### Weaknesses
1. The experimental environments used are overly limited and simplistic. The chosen benchmarks—Pendulum, MountainCar, InvertedPendulum, and LunarLander—are relatively simple and classic in contemporary deep reinforcement learning (DRL) research. These environments feature low-dimensional state vectors rather than high-dimensional pixel inputs, and the agent operates under clear dynamical models with intuitively understandable optimal policies.

2. ProtoSAC exhibits limited scalability and robustness. Even in these simple settings, ProtoSAC shows a slight performance degradation compared to the standard SAC baseline, as evidenced in Figure 2 and Table 2. The policies required in these environments are inherently straightforward—primarily involving swinging, balancing, or landing—and yet the model already relies on 30 to 60 prototypes. In more complex tasks that involve long-term planning, hierarchical decision-making, or hidden variables, a significantly larger number of prototypes would likely be needed to adequately cover the state space. This raises serious concerns regarding the method's scalability and general robustness.

3. ProtoSAC is narrowly built upon specific assumptions of the SAC algorithm. The method integrates prototypes only with SAC and cannot be directly extended to other actor-critic algorithms, such as deterministic policy-based methods like DDPG. Furthermore, it does not readily adapt to environments with non-continuous action spaces or those requiring alternative distribution representations—for instance, bounded continuous actions modeled using Beta distributions.

4. Insufficient analysis of experimental results. The interpretability claims of ProtoSAC are not convincingly supported by Figures 3 and 4 or their accompanying analysis. Although the captions state that “Each prototype is represented by its associated state and the action Gaussian distribution,” Figure 3 only visualizes the Gaussian distributions of actions (a, b, c, d), without illustrating the corresponding states associated with the four prototypes. This omission makes it unclear which specific states these prototypes represent. While the figures show how the model makes decisions—by blending prototypes—they fail to illustrate why those decisions are made, since the reader cannot see which representative states the agent considers similar to the current observation.

5. The absence of XAI/Explainable RL comparative experiments. As one of the main contributions of this article, the interpretability of this method also requires corresponding comparative experiments as support. However, although this article made a performance comparison with the explainable Shared-PW-Net, no comparison was made in terms of explainability.

### Questions
1. Expand the experimental validation to more complex environments. To rigorously assess the robustness and scalability of the proposed method, it is crucial to test it on benchmarks with high-dimensional state spaces, such as those requiring processing of image or video inputs. Demonstrating competitive performance and meaningful interpretability in these challenging settings would significantly strengthen the paper's contributions.

2. Conduct a more comprehensive evaluation of interpretability. The current qualitative analysis should be supplemented with a comparative study against other explainable DRL methods, particularly post-hoc approaches like ProtoX. A quantitative and/or human-study-based comparison would more convincingly demonstrate the advantages and unique value of the intrinsic interpretability provided by ProtoSAC.

3. Enhance the ablation studies. The paper would benefit from an additional ablation experiment that investigates the impact of the prototype update mechanism. By presenting results from a model variant where this update process is disabled, the authors could quantitatively validate its necessity for maintaining performance and prototype quality throughout training.

4. Revise the visualizations in Figures 3 and 4. To fully deliver on the promise of prototype-based explanation, these figures must be modified to visually show the actual states associated with each prototype. The current figures only display the action distributions, which makes it impossible for a reader to understand why a specific action is chosen. Illustrating the prototype states is fundamental for validating the "this state looks like that" reasoning.

5. I suggest that the author conduct a more in-depth analysis of the relationship among state, prototype and action distribution. In fact, it is very important to understand which observed variables in state/prototype affect the generation of actions.

6. The chart placement of this article needs further adjustment to make it easier to read.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an extension to current interpretable by design deep reinforcement learning algorithms by working prototypes into the training of soft actor critic. The method essential works by incorporating prototypes into the actor during training with the relevant gaussian processes. Comparisons are made against PW-Net variants and SAC vanilla algorithms, showing better performance than baseline interpretable by design methods, and almost equal performance to SAC.

### Strengths
The paper (as far as I know) is the first to propose a method to train interpretable by design deep RL agents from the ground up, which represents a significant contribution.

Appropriate baselines are chosen, and the environments tested sufficient I believe.

The ability to work in continuous action spaces is a big plus, as this represents most real-world deployment tasks of RL agents such as co-pilots etc.

### Weaknesses
I miss some qualitative analysis of the prototypes themselves, it is difficult to say how useful the final system would be for an end user, although to be fair this is a byproduct of most interpretability papers in the ML conferences.

The paper doesn’t consider pixel state spaces as far as I understand, which is a significant limitation. Would this mean the method is not applicable to deep learning problems? Or did you use pixel state spaces for these problems? If it is purely symbolic, I don't think it's fair to call this an interpretable by design deep RL algorithm, but I'm willing to hear a counter argument.

### Questions
I might have missed it in the manuscript, but did you use the symbolic or pixel state spaces for these problems?

Do you have any idea as to the qualitative properties of the prototypes? How easy would it be for a user to gain some kind of understanding of the model globally?

In the forward pass are you taking the state encoding’s similarity to all prototypes? I am wondering if the final output is a calculation traced back to all prototypes, which could be difficult to interpret.

### Soundness
3

### Presentation
3

### Contribution
3
