## Human Reviewer 1

### Summary
This paper addresses the challenge of generalizing reinforcement learning models in environments with superposed causal relationships, which are mixtures of different causal mechanisms. The authors highlight the limitations of global causal discovery techniques in such settings and propose Superposed cAusal Model (SAM) learning. SAM is an end-to-end framework utilizing a transformer-based model to dynamically recognize and adapt to encountered causal relationships. Experiments in two simulated environments demonstrate SAM's effective identification abilities and generalization to unseen states.

### Strengths
(1) The authors introduce an algorithm, i.e., SAM, that dynamically identifies causal relationships within each episode, enabling more accurate predictions and transitions. 

(2) By leveraging the Transformer architecture, the SAM can infer causal relationships from past interaction trajectories. 

(3) The effectiveness is validated by two simulated envoriments.

### Weaknesses
(1) The paper lacks clarity in distinguishing between causal and predictive aspects of the model. Further elaboration is needed on how the proposed method identifies causal relationships and how the accuracy of these causal inferences is verified. More rigorous testing and validation are required to demonstrate the correctness of the identified causal factors.

(2) The experimental results presented in the paper are not sufficient enough to fully support the claims made. To strengthen the paper's findings, it is recommended to repeat the experiments for multiple times and report the average results across multiple trials. This will provide a more reliable and consistent basis for evaluating the model's performance.

(3) The paper should more clearly summarize the contributions of the proposed method compared to existing approaches. Highlighting the distinct advantages and novel aspects of SAM learning will help to better position the paper within the existing research landscape.

### Questions
None.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
The authors tackle the problem of learning causal structure from observational data, in an environment composed of mixtures of causal graphs. The setting is challenging and relevant, since the relationship between causes and effects in the real world can change over time, or change based on which states the trajectory passes through. The authors propose a transformer-based method that conditions on an observed trajectory to predict edge in the next-step transition dynamics, then models the next-step state accordingly. Empirical proof of concept is provided on two datasets where discerning between multiple possible causal graphs is needed for success.

### Strengths
The setting is ambitious and meaningful progress along this direction could be impactful in the long run

The authors show how prior work on static causal discovery can be extended to mixtures over causal graphs

### Weaknesses
The authors place a lot of emphasis on the novelty of their approach 
[L050-051, L110-111, L523-524], but I do not agree that the idea of modeling mixtures of causal graphs is completely new. Relevant prior work is not discussed.

Clarity of presentation when it comes to the formal framework is lacking. The authors take a variational inference approach to inferring the mixtures components and local causal structure, but do not precisely define their assumptions about how the data are sampled.

### Questions
Comments/questions:
* [L 024, "powerful identify abilities"] typo
* [L050-051, "identifying mixtures of causal graphs has not been extensively explored", L110-111, "we are the first to address superimposed causal relationships", L523-524 "SAM addresses the limitations of existing causal world models, which assume a single causal structure governs the entire dataset"] These claims are misleading in my opinion. The framework of local causal models introduced in the CoDA paper seems quite related [https://proceedings.neurips.cc/paper/2020/hash/294e09f267683c7ddc6cc5134a7e68a8-Abstract.html]. This paper also discusses how transformers and dynamics models with sparsity penalties can be used to attempt to infer local causal structure.  work on learning from block MDPs is also relevant and should be cited [https://openreview.net/forum?id=fmOOI2a3tQP].  Bayesian inference over mixtures of graphs has also been attempted [https://proceedings.mlr.press/v180/deleu22a.html].
*[L123-130, L130-135] redundant information
* [L176-188] The formal presentation of the framework could be more precise. The authors describe how the masks are constructed in plain language but I found it difficult to ground the idea in the notation. Can an expression for masks be provided? The transition dynamics include a product over t factors, but it is not clear how the masks fit into this product. Strictly based on the writing it also seems ambiguous whether the environment is described by one mask [L174] or several [L176]. From the inference method described in L201 and L211 it seems clear that the masks are implicitly trajectory dependent, but I am not actually seeing this dependency described explicitly in the data generative process. Is $\mathcal{G}$ a random variable? Where is the idea of a mixture of causal mechanisms [L016, L531] introduced formally?
* [L293-295] while it's true that conditioning on an entire trajectory provides more information, it also would be a more complex posterior to approximate. Have the authors tried an ablation of their method where they control for the length of trajectory used for structure inference/identification?
* It would be interesting to see how the proposed method does when the underlying causal graph is static.

### Soundness
3

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper proposes "Superposed cAusal Model" (SAM) learning. It is a framework that learns a transformer-based model which can recognize the causal relationships that the agent is encountering. It is based on an existing work and tackles the setting where trajectories are collected from different environments.

### Strengths
The paper tackles an interesting problem.

### Weaknesses
- The writing is not strong at a point that prevents getting easily the information out of the paper. In the abstract alone there are two errors: (1) a sentence with both "in this paper" and "in this article" and (2) "identify' in the sentence "where SAM shows powerful identify abilities in environments with superposed causal relationships". Errors also happens at other places in the text: "In this section, we *introduce propose* Superposed cAusal Model (SAM) learning" in the beginning of section 4 or line 214 "we design *an* simple yet efficient". In Figure 1 "Super-post" instead of superposed?
- Some words are not clearly defined such as the word "decomposition" in line 182 "This superposed causal dataset is collected from C environments that share the same decomposition but exhibit different causal relationships $\mathcal G_i$".
- The formalization is somewhat unclear at places. For instance, the learnable parameters are $\phi$, $\theta_1$ and $\theta_2$ but only $\phi$ and $\theta$ appear in Equations 1 and 2. The reader has to guess that $\theta$ is $\\{\theta_1, \theta_2\\}$. In the formalization $\mathcal S$ represents the state space of the MDP but 
- The key contributions are not fully clear, which might be due to the fact that it is not clearly highlighted, particularly in the methodology section. In that section, it is written "We derive the optimization objective of the superposed causal world model using a similar approach to Varambally et al. (2024)" and then follows two paragraphs in the methodology and it is unclear what the key differences are.

### Questions
- What is the specific architecture of the different NN/transformers components? I do not find them in the paper nor in the appendix.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4