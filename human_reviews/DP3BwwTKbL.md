# Predicting Observation after Action in a Hierarchical Energy-based Model with Memory

- Decision: Reject
- Scores: 3, 3, 5, 6

## Abstract
Understanding the mechanisms of brain function is greatly advanced by predictive models. Recent advancements in machine learning further underscore the potency of prediction for learning optimal representation. However, there remains a gap in creating a biologically plausible model that explains how the neural system achieves prediction. In this paper, we introduce a framework employing an energy-based model (EBM) to capture the nuanced processes of predicting observation after action within the neural system, encompassing prediction, learning, and inference. We implement the EBM with a hierarchical structure and integrate a continuous attractor neural network for memory, constructing a biologically plausible model. In experimental evaluations, our model demonstrates efficacy across diverse scenarios. The range of actions includes eye movement, motion in environments, head turning, and static observation while the environment changes. Our model not only makes accurate predictions for environments it was trained on, but also provides reasonable predictions for unseen environments, matching the performances of machine learning methods in multiple tasks. We hope that this study contributes to a deep understanding of how the neural system performs prediction.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes an energy-based neural recurrent state-space model designed to predict world perception following an action. While prior works have introduced world models for predicting observations, they often rely on techniques like Backpropagation Through Time (BPTT), which lack biological plausibility. In contrast, this paper employs an energy-based state-space model, which is known to resemble biological mechanisms such as Hebbian learning and sampling-based inference, to address the sequential prediction task. The model is implemented using a hierarchical neural network for its expressiveness and biological validity, and a continuous attractor network is utilized to model temporal memory, similar to biological neural systems. Evaluation on a few vision-based tasks demonstrates that the model outperforms existing techniques in predicting observations.

### Strengths
1. The paper provides a detailed formulation of each component and effectively connects them to biological concepts, supporting its claim of biological plausibility.
2. The authors successfully integrate several neural network-based components to simulate a biological system plausibly.
3. The paper is well-written overall, with clear visualizations that enhance understanding.

### Weaknesses
1. The main technique does not appear to offer significant practical advantages beyond its biological plausibility.
2. While the paper argues that the learning rule is biologically plausible due to its resemblance to the Hebbian rule (as opposed to BP), the formulation and its properties remain quite similar to BP, which diminishes the claimed advantage of biological plausibility.
3. The evaluation is limited; the setups simulate abstract and simplistic biological behaviors, without testing the model on more realistic and practical tasks.

### Questions
I am open to adjusting my scores if the following comments are addressed:
- Please add parentheses around the citations, as the current formatting makes them difficult to read.
- While the world/observation prediction task is prevalent in biological systems, what practical applications does this have? Does it improve the performance of sequential decision-making algorithms? A more application-driven empirical evaluation would be necessary.
- It would be beneficial to include evaluations on more practical decision-making tasks that could benefit from this model, particularly by solving sub-tasks related to world prediction. Although this is mentioned in the discussion, the practical significance of the technique is unclear without actual end-to-end demonstration.
- The paper's practical motivation is not entirely convincing. Using local learning rules or neural dynamics instead of BP is argued to be advantageous, but what practical benefits does it offer? BP is known for its high memory usage and additional latency, but how does the local learning rule address these issues?
- The claim that the learning rule differs significantly from BP, or that it fully matches the Hebbian rule, is unclear. In line 281, the paper states, "Each layer will repeat this process and propagate information downward until the last layer." While it is true that gradients are not propagated backwards through the hierarchical network, latent information $s^l_t$​ is still propagated backwards. This necessitates storing all intermediate activations and backpropagating the latent information, much like conventional BP. The only aspect that aligns with the Hebbian rule is the first layer. Additionally, in Equation 16, the update rule for the sample $s^l_t$ involves the derivative $f'$ and the derivative of the Gaussian reparametrization, and this updated sample is used to update the subsequent layer. This resembles backpropagating gradient information through the sample updates. A dedicated subsection verifying that the suggested formulations are indeed biologically plausible would strengthen the paper. For example, it would be useful to mathematically verify that the learning rules adhere to the Hebbian rule and differ substantially from backpropagation. Since biological plausibility is a key advantage of the proposed technique, a thorough theoretical analysis would be necessary.
- In the imagination phase, to predict the outcome of a specific action within the observed world, must all previous sequences of actions and observations be sequentially inputted to form the memory network? Is there a component where one can input the initial observation directly?
- Why is image quality compared using MSE when there are several established image quality metrics, such as PSNR or SSIM, that could be used? MSE seems ambiguous in this context, and it would be preferable to employ commonly accepted metrics.
- The evaluation setups may need further validation to confirm that they correctly simulate biological behaviors and perceptions. For example, is it valid to simulate eye movement with datasets like CIFAR-10 or FMNIST? I would expect a more complex real-world task beyond 2D image scanning. Is there biological literature that supports the use of such datasets to simulate eye movements?
- What are the practical advantages of this energy-based model, aside from its biological plausibility? The performance does not seem to scale with the number of trained images, which raises concerns about its practicality.
- Several minor corrections:
  - Line 154: "decent" should be "descent"
  - Line 186: "deviation" should be "derivation"
  - Line 209: "employs" should be "employ"
  - Line 334: "first is step is" should be "first step is"

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors build a series of predictive neural networks following the energy-based model. What is particularly interesting about the model is the incorporation of several key ingredients: attractor network / memory, predictive coding, and a hierarchical structure. It is difficult to assess what question(s) the authors are trying to address. The abstract and introduction are a little bit all over the map, repeatedly referring to “biological plausibility”, a fashionable term that ends up being rather empty without any anchoring on actual biological data. The work also refers to eye movements, and motion in environments, both of which are only superficially discussed at a very abstract level.

### Strengths
There are many interesting ideas in here, including: (1) the energy-based models and their relationship to predictive coding, (2) the emphasis on predicting actvitiy after an action is taken, (3) neural network architectures that have a memory component, recurrent dynamics, hierarchies, and prediction.

### Weaknesses
The work would benefit a lot from being more specific throughout. Take the abstract, which continuously refer to “the neural system”. What does this mean? Is this referring to any arbitrary neural network? A fly? A human? Theories of everything often end up being theories of nothing. What are the authors trying to model here? 

It would be very useful to articulate a key question or set of questions or hypotheses. Are the authors trying to build a network that can solve a specific task? 
If so, which task, what are the benchmarks, what are other comparison models? 
If they are not trying to solve a specific task, is there a concrete hypothesis that the authors are testing? 

To the extent that this work attempts to connect with biological brains (it is not clear whether this is the case), a lot of work would be needed to establish those connections. Take the first sentence of the discussion: “We consider that the brain employs an EBM as an intrinsic generative model to predict the next observation after action.”. Is this a hypothesis, a conjecture, a conclusion, or an axiom? If it is a hypothesis, then what kind of *empirical* data would be needed to test the hypothesis? Are there any neuroscientific data that provide support for this hypothesis or evidence that contradicts the hypothesis?

### Questions
It seems that one of the interests is to model eye movements (curiously referred in the singular as eye movement throughout). There is a whole industry or real neural network models that aim to predict eye movements in images under different circumstances. Here are a few: Zhang et al Nature Communications 2018, Yang et al CVPR 2020 (these are studies during specific actions for visual search), Kummerer et al J Vision 2022 (this is without any task). In the absence of comparisons with actual eye movements, it is not clear how to evaluate the work here. But perhaps the intent is not truly to study eye movements, the actual question or goal was not clear. 

The notion of “imagination” and being able to predict observations for actions without performing the actions seems quite interesting. It would be quite interesting to actually show experimental data on this. It seems that the experiments in Fig.3 (“eye movement”) and Fig. 4 (“head turning”) are in this direction. What were the models trained on and how was cross-validation performed? Can the model take a novel image and make inferences about observations for a novel viewpoint never seen before? What is “ture” in Fig. 4? 

As a general note, to interpret results, I always find it very useful to include:
-	Error bars. 
-	Chance levels.
-	Upper bounds (not entirely sure how to define it here, but perhaps an oracle version)
-	Rigorous statistics.
This comment applies to all tables and Figs. 3. and 4.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes a hierarchical recurrent state-space model based on an energy-based model (EBM) to simulate the brain's process of predicting observations after actions. By introducing a continuous attractor neural network (CANN) for memory, the paper constructs a biologically plausible neural network model. The model is validated across various tasks, including vision, motion, and head rotation, demonstrating its predictive performance in both trained and novel environments. Based on a bio-inspired framework, this model shows comparability to existing machine learning methods.

### Strengths
1.Bio-inspired Modeling: The model leverages the EBM framework and CANN to simulate prediction, learning, and inference in the neural system, achieving biological plausibility, which is a unique attempt in machine learning.

2.Innovative Prediction and Learning Sequence: Unlike most machine learning models, this model performs learning before inference, resulting in a unique objective function. This sequence is shown to provide greater stability and robustness over traditional methods.

3.Memory Mechanism with CANN: The integration of CANN as a memory network allows the model to efficiently compress past experiences, supporting real-time prediction.

### Weaknesses
1.Biological Interpretability of Experimental Results: Although the model shows good predictive performance, its interpretability in terms of biological neural systems could be strengthened. For instance, explaining how error neurons represent biological neuron behavior or how model layers correspond to cortical structures would enhance understanding.

2.Comparison of Baselines: While the paper includes comparisons with models like TransDreamer, it would be beneficial to compare with more bio-inspired models (e.g., more complex PCN frameworks) to further highlight the model's advantages in biological plausibility and performance.

3.In-depth Analysis of Model Parameters: Model performance is likely influenced by parameters like network depth and neuron count. However, the paper lacks a detailed discussion of how these parameters impact biological plausibility. Future work could examine the model’s adherence to biological realism under various parameter settings.

4.Impact of Dynamic Environment Changes on CANN: The framework does not fully analyze how dynamic environmental changes affect the CANN structure. Although the paper mentions this as future work, adding preliminary exploration in current experiments could help validate CANN’s stability in complex environments.

### Questions
1.Future versions could provide more detail on how the model’s biological mechanisms correspond to actual neural processes, such as simulating firing patterns and memory representations of biological neurons.

2.Consider additional experiments with various parameter settings to analyze the model’s adaptability and robustness in more complex environments, especially in comparison to traditional bio-inspired models.

3.Further discussions could be added regarding the representational role of CANN in brain structures and the correspondence between the model’s hierarchical layers and cortical structure.

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
2

### Summary
The submission posted a solution an interesting topic on the neural mechanism to perform prediction of future observations given current state and memory. Authors proposed an energy based model to preform prediction, inference, and learning. A hierarchical and continuous attractor network is used to achieve these goals. Experimental results show the model can faithfully generate future observations for the environments the model is trained on and for unseen environments.

### Strengths
The introduced energy-based recurrent state space and a continuous attractor network to conceptually mimic the brain mechanism to predict next observations is interesting. The methodology introduction is clear and well organized.

### Weaknesses
The generalization ability of the proposed method is difficult to evaluate, as in Fig. 3a, the "unseen" images appear very similar to the previously "seen" images. It would be helpful to provide the mean squared error (MSE) between these unseen and seen images to clarify the degree of generalization.

What is the computation complexity for layered EBM structure and CANN memory integration. For a comprehensive comparison with baseline methods, it would be beneficial to discuss computational efficiency and scalability for larger datasets.

Additionally, the specific contribution of each component should be discussed with ablation studies. The proposed method involves several components—energy-based modeling, the continuous attractor network, and the hierarchical neural network—but it remains unclear which components contribute most significantly to prediction accuracy.

### Questions
1. The motivation for using an energy-based model is biological plausibility. However, it would be beneficial if the authors could discuss how the proposed model architecture improves prediction purely from a methodological/application perspective.

2. Table 3. how about comparison with more recent methods.

3. How the number of hierarchical layer affects the prediction accuracy.

### Soundness
3

### Presentation
3

### Contribution
3
