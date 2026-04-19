# Scaling Value Iteration Networks to 5000 Layers for Extreme Long-Term Planning

- Decision: Reject
- Scores: 5, 6, 5, 6

## Abstract
The Value Iteration Network (VIN) is an end-to-end differentiable architecture that performs value iteration on a latent Markov Decision Process (MDP) for planning in reinforcement learning (RL). However, VINs struggle to scale to long-term and large-scale planning tasks, such as navigating a $100\times 100$ maze---a task that typically requires thousands of planning steps to solve. We observe that this deficiency is due to two issues: the representation capacity of the latent MDP and the planning module's depth. We address these by augmenting the latent MDP with a dynamic transition kernel, dramatically improving its representational capacity, and, to mitigate the vanishing gradient problem, introduce an "adaptive highway loss" that constructs skip connections to improve gradient flow. We evaluate our method on 2D maze navigation environments, the ViZDoom 3D navigation benchmark, and the real-world Lunar rover navigation task. We find that our new method, named \textit{Dynamic Transition VIN (DT-VIN)}, scales to 5000 layers and solves challenging versions of the above tasks. Altogether, we believe that DT-VIN represents a concrete step forward in performing long-term large-scale planning in RL environments.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper extends the Value Iteration Networks (VIN) to handle long-term planning by making two modifications: 1. Changing the implementation of latent transition kernel in VIN from state-invariant to observation-dependent; 2. Increasing the depth of the planning module with the “Highway” connections which are defined based on the optimal planning path. The proposed algorithm, named DT-VIN, has been evaluated on 2D maze navigation tasks, a converted 3D VizDoom task and a rover navigation task with orthomosaic image as input. The empirical evaluation shows DT-VIN performs generally well, especially on large-size mazes and deep planning modules.

### Strengths
* [**Originality**] The paper considers a rather deep planning module, comprising of many value iteration modules and shows that such deep architecture is required to solve complex maze tasks. 
* [**Quality**] The proposed method has been evaluated on various benchmark tasks and the performance is generally better. 
* [**Clarity**] The paper is well written, and most parts are easily understandable. Figure 2 presents a nice overview of the main idea of the proposed method. Figures in ablation studies also clearly shows the ablation effects of each design consideration. 
* [**Significance**] The paper shows that solving a long-term planning requires a deeper planning module architecture, which can be a good contribution to the community.

### Weaknesses
### Originality: VIN can be observation-dependent
One of the main contributions claimed by this paper is the incorporation of observation into the transition kernel. However, as mentioned by the paper itself, **“original VIN paper proposes a general framework where the latent transition kernel depends on the observation”**. In other words, the VIN has incorporated the observation into the learning of transition kernel already, but its open-sourced implementation does not reflect this. So this can hardly be regarded as a contribution of this paper. **The claim of incorporation of observation may sound like a good implementation suggestion when the considered the maze is of bigger sizes**. 

### Significance: additional loss introduced by highway connections
The paper introduces the highway connections and thus incur extra loss to the final loss. Such extra loss relies on two important factors: the length of the optimal path (the provided expert path from the start to the goal) and the number of layers for the planning module. **It would be important to discuss the situations where the optimal path might not be known in advance**. For example, in the VizDoom experiment, it is unclear to me how such optimal path is obtained. Further, the number of layers has a big impact on the loss scales. Consider the extreme case of 5000 layers, if the optimal path is 1000, then the final loss will have 4000+1 terms, which can make the optimization hard as the effective backpropagation of gradients can be challenging. **The paper should discuss how this long-range backpropagation of gradients can be well handled in practise**.

### Questions
1. Since the training of highway connections requires the expert path from the start to the goal, how the training and testing mazes are constructed in the empirical evaluation? If the training and testing mazes are the same, the extra loss might leak the path info to the planning module. 
2. In Figure 9, there is no plot of 100x100 maze. Why is such plot not provided? Given that the depth of 5000 is required only for such maze, it would be crucial to see how the choice of other depth values would affect the SR and OR. 
3. The description to VizDoom experiment is vague and more details are needed. Can the authors clarify how the optimal path can be found and what it means for “3D ViZDoom mazes with size M = 35”? 
4. The evaluation on ROVER NAVIGATION is also unclear. The paper says that “elevation data are utilized solely for creating expert paths and assessing algorithm performance, not as input for the neural networks”. But the expert paths are needed to define the training loss (i.e., the additional loss), right? Then the proposed method DT-VIN still (implicitly) requires the evaluation data as the input?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the topic of long-term large-scale planning in Reinforcement Learning environments, by building on the Value Iteration Network. The authors identify two key deficiencies in VIN, and propose methods to overcome both: 1) the use of a dynamic transition kernel to improve the VIN’s representational capacity, and 2) the use of an “adaptive highway loss” to overcome problems with vanishing or exploding gradients, thus allowing the model to scale to a much greater depth. This deeper, more expressive model is able to solve problems at a scale which defeated the original VIN approach, and its more recent variants.

### Strengths
Overall, this paper is well presented, and clearly describes a step forward in the development of VIN-based RL planning. The deficiencies of the original VIN approach are well explained and demonstrated empirically, providing clear motivations for the modifications the authors suggest.
In terms of novelty, the two key innovations (the transition mapping module, and the adaptive highway loss) seem like neat iterations to the previous work, rather than being ground-breaking new directions, but I believe the authors demonstrate that there is real value to them. In particular, it is pleasing to see how their highway loss both improves on and simplifies previous work.

This paper could be a solid step forward, except for some concerns I have regarding motivation and evaluation, discussed below.

### Weaknesses
At the detailed level, the motivation is well constructed: this paper improves on several deficiencies in the established approaches. What I feel the paper lacks is a motivation from a broader perspective: why do we need an RL agent that can solve bigger mazes? Particularly when – as acknowledged by the authors – the A* algorithm has been around since 1968 (and, indeed, Dijkstra’s algorithm is used in this work to compute the shortest path lengths).

This is linked to my concerns about evaluation: all three domains (2D grid-world mazes, the 3D VizDoom, and the Rover navigation) feel like essentially the same problem. In particular, I can’t really see what value the VizDoom experiments add. As per the GPPN paper, “The maze design for the 3D mazes are generated in exactly the same manner as the 2D mazes…” – this is essentially the same environment, just reconstructed in VizDoom, but with the additional step that a network then has to translate the first-person images back into a 2D top-down representation which is then fed into the DT-VIN – with all the same hyperparameters and settings as the first domain. While I understand the value of using the same benchmarks as the previous papers, the introduction of VizDoom as a benchmark in the first place feels misleading. The suggestion is of an agent playing Doom; in reality, artificially constructing a 2D maze in the VizDoom environment, then taking a full set of N/S/E/W first-person images from each location, then turning these back into a fully-observed 2D representation, is about as far removed from the actual game dynamics as it is possible to get. Perhaps I am missing the point, but beyond training the network to reconstruct the images into the 2D map (which is not novel), this environment doesn’t demonstrate any new capabilities in the DT-VIN, and feels like a wasted opportunity. Instead, I would have loved to see how DT-VIN performed in a very different domain. For example, the original VIN paper evaluated the model in discrete and continuous path-planning, and a natural-language-based search domain. As mentioned in this paper, “VINs have been shown to perform well in… path planning, autonomous navigation, and complex decision-making in dynamic environments”, but DT-VIN was only evaluated in a very narrow set of domains.

As the introduction put it, “Planning is the problem of finding a sequence of actions that achieve a specific pre-defined goal.” – with this broad definition of planning, it is easy to see the motivation for an agent that improves on previous approaches, but the paper’s narrow focus on the already-solved problem of path-planning obscured this motivation.

I’d like to see the authors address these concerns about motivation and evaluation as a first priority.

### Questions
Some lower priority questions I had:  
1. The formulation of the highway loss requires knowledge of the shortest path l (as acknowledged by the authors in Appendix A) – where this is unknown, what is the cost of over/underestimating it? It would be interesting to see what effect this has on training if it is set too low, or too high.  
2. In the GPPN paper, three transition kernels were used (NEWS/Moore/Differential Drive) – were these considered for DT-VIN? I assume NEWS was used in this paper, but I can’t seem to find it.  
3. I have some questions about scaling. The authors highlight the “potential of our method to scale to increasingly complex planning tasks alongside the increasing availability of computing power” but I’m uncomfortable with the twin assumptions that compute will keep getting cheaper, and that the method can therefore keep scaling up. I’d like to see some consideration given to these aspects of scale:  
  a. _Compute_: GPU training times are only given for one datapoint (the 35x35 maze, with a 600 layer network). How much GPU time was needed to train the 100x100 maze, 5000 layers?  
  b. _Model size_: What is the relationship between the number of planning steps required by the problem, and the depth of the network required to solve it? From the plots it appears that success tails off as the number of steps approaches the network depth, but is it feasible to keep increasing network depth as the problems become more complicated?  
  c. _Data_: Is it necessary to keep scaling up the dataset in line with increasing the model depth, and does this limit the method’s applicability to other problem domains where training data can’t be so easily generated?  

Overall, this was a well-presented and interesting paper. I look forward to being persuaded that my misgivings about motivation and evaluation are unfounded!

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Summary: This paper presents the Dynamic Transition Value Iteration Network (DT-VIN), an improvement on the Value Iteration Network (VIN) designed to perform planning-based reasoning in reinforcement learning. The proposed method enhances VIN to handle larger-scale, long-term planning tasks by introducing a dynamic transition kernel (a learnable transition mapping module) and adaptive highway loss, a loss structure that connects hidden layers directly to the final loss. The effectiveness of DT-VIN is demonstrated empirically in several tasks, including 2D maze navigation, 3D VizDoom, and rover navigation.

### Strengths
Strengths: 

1. The results on scaling up to 5000 layers are promising, showing DT-VIN's capability to solve larger-scale environments without significant performance drops, even as trajectory length increases.

2. The paper introduces simple yet innovative modifications that effectively improve VIN.

3. Even with the scale-up, DT-VIN still has model size advantage to other RL planning algorithms

4. The writing is generally clear, with well-organized ideas that communicate the methodology and results effectively.

### Weaknesses
1. **Weaknesses:**
   One of the key aspects of the paper is the use of the adaptive highway network and the adaptive highway loss. However, more structured experiments on the structure will be beneficial to justify the new method. Below are some details about what specifically you can do:

   1. Additional ablation studies would clarify the benefits of directly connecting hidden layers to the final loss instead of intermediate layers. Furthermore, the claim that having \( n > l \) is advantageous for short-term planning tasks needs stronger empirical backing. It would be beneficial to include more controlled experiments to justify this claim. It would strengthen the claim if there is a performance gap between the two methods in 35x35 and 100x100 2D maze environments in the “shorter SPL” regime. i.e., if you do not have \( n > l \) in your loss, does the performance drop in the [1,100] regime for 35x35 and [1,600] regime for 100x100? To be clear, I am asking for the 2D maze environments to be tested under these different regimes with success rate as the metric as in the paper. 

   2. The adaptive highway network depends on an estimation of \( l \) (the trajectory/planning path length). It would be insightful to include robustness analysis of the method against different \( l \) values, exploring how sensitive DT-VIN’s performance is to the estimation of \( l \). For example, how does the success rate change in Vizdoom when you overestimate/underestimate \( l \), or when the estimate of \( l \) has high variance? The metric is the success rate. 

   3. Although adaptive highway loss was introduced to address gradient stability, the ablation study suggests that the softmax latent transition kernel plays a larger role in mitigating gradient issues. Some theoretical justifications or more explanation of this gap would be helpful. Including loss curves/gradients norm graphs on one or more of the environments from the paper would help visually illustrate the trend of gradient instability for lack of each ingredient. The specific environment should be selected by the author and I would be happy with anything from the paper. 

2. Testing on additional environments, such as continuous control domains or graph-based planning tasks, would provide a more comprehensive evaluation of DT-VIN. The original VIN paper included both types - learning policies with guided policy search with unknown dynamics and simulating in Mujoco for the continuous control domain (metric is measuring the final distance to the target); WebNav for the graph-based planning task (metric is success rates of the learned policy with DT-VIN for a query).

### Questions
None

### Soundness
2

### Presentation
3

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
This paper identifies the reasons for VIN's deficiency in long-term and large-scale planning tasks as (1) the representation capacity of the latent MDP and (2) the planning module's depth. The paper mitigates the issues with a dynamic transition kernel and adaptive highway loss. As a result, the paper scales VIN to 5000 layers and solves large-scale planning tasks.

### Strengths
The paper effectively identifies a significant limitation of the VIN method: its difficulty in scaling to long-term and large-scale planning tasks. While the proposed methods are straightforward, they are well-motivated and demonstrate substantial effectiveness in addressing this limitation. The experimental results support the claims, showing clear improvements in both scalability and performance. This combination of simplicity and efficacy highlights the practicality of the approach.

### Weaknesses
- Clarification of Motivation: While the motivations for introducing a dynamic transition kernel and an adaptive highway loss are clearly explained, the significance of investigating the Value Iteration Network (VIN) remains unclear. In the introduction, the paper notes that "Within reinforcement learning, one notable method is the Value Iteration Network (VIN)," but the paper does not expand on why VIN is particularly notable or important to study, especially in comparison to other established methods like Dreamer and MuZero. Adding a discussion that situates VIN’s relevance among these methods could strengthen the motivation and context for readers.
- Clarity of Equation 2: It would be helpful to present the overall loss function in addition to the adaptive highway loss, as showing only $\mathcal{L}$ here may introduce ambiguity. Clarifying the meaning of x and y in Equation 2 is also necessary.
- Transition from RL to Imitation Learning: The paper begins with a focus on reinforcement learning in the Introduction and Preliminaries sections, but it transitions to imitation learning (IL) in Section 3.2 and in the experimental setup without sufficient explanation. This shift from RL to IL feels abrupt and could be confusing. A more explicit discussion is needed to bridge RL and IL.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
