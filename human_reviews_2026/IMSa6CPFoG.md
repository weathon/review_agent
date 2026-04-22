# Dream2Learn: Structured Generative Dreaming for Continual Learning

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 4, 6, 0

## Abstract
Continual learning struggles with balancing plasticity and stability while mitigating catastrophic forgetting. Inspired by human sleep and dreaming mechanisms, we propose Dream2Learn (D2L), a generative approach that enables models, trained in a continual learning setting, to synthesize structured additional training signals driven by their internal knowledge. Unlike prior methods that rely on real data to simulate the dreaming process, D2L autonomously constructs semantically distinct yet structurally coherent dreamed classes, conditioning a diffusion model via soft prompt optimization. These dynamically generated samples expand the classifier’s representation space, reinforcing past knowledge while structuring features in a way that facilitates adaptation to future tasks. In particular, by integrating dreamed classes into training, D2L enables the model to self-organize its latent space, improving generalization and adaptability to new data. 
Experiments on Mini-ImageNet, FG-ImageNet, and ImageNet-R show that D2L surpasses existing methods across all evaluated metrics. Notably, it achieves positive forward transfer, confirming its ability to enhance adaptability by structuring representations for future tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Dream2Learn (D2L), a generative continual-learning framework inspired by the human dreaming process. Instead of relying on generative-replay past data, D2L uses a latent diffusion model conditioned via soft prompt optimization to synthesize dreamed classes. These dreamed samples form coherent yet distinct new concepts, supporting future task adaptation. Experiments on Mini-ImageNet, FG-ImageNet, and ImageNet-R show consistent gains across rehearsal-based baselines. Ablations validate the role of the oracle and the dynamic dream-class update mechanism.

### Strengths
- The paper is well written and easy to follow.
- The proposed method is novel - rather than retrospective replay, D2L introduces a prospective generation mechanism that structures the representation space for future tasks.
- Oracle-guided optimization is an interesting and effective solution to avoid dream collapse.
- Extensive experiments are conducted, including comparisons with SOTA methods and ablation studies. And experimental results demonstrated strong imporvements.

### Weaknesses
- The “dreaming vs. replay” distinction is interesting but not sharply formalized, and the boundary seems vague. There're OOD test, but not sufficient, as being “not old classes” doesn’t prove they’re future-oriented or structurally bridging.
- D2L adopt a pretrained diffusion backbone for generating “dreamed classes.” However, how the choice of this generator impacts results is not analyzed.

### Questions
- Can you provide quantitative or visual evidence that the dreamed samples truly occupy intermediate latent regions between past and future classes? And the generated samples indeed anticipating future?
- Would wrong anticipation harm the performance in some cases?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper develops D2L based on generative replay to address catastrophic forgetting in CL. The core idea is to enable a base model to generate structured data synthetically that are semantically distinct yet structurally coherent with previously learned knowledge to enable experience replay. Unlike past work that mainly reconstruct input data, D2L leverages a latent diffusion model. This model is conditioned on soft prompt optimization to create future-adaptive representations for experience replay. This data expands the classifier’s latent space and enable forward transfer to improve generalization on new tasks. As a result, the process does not need external supervision or additional real data. During training, D2L optimizes prompts for each learned class to synthesize new classes, guiding the generator toward distinct but consistent outputs. An oracle network monitors the optimization process to prevent collapse and overfitting. Unlike memory buffer-based methods, D2L does not store generated data but maintains an inventory of optimized prompts which are easier to store. Empirical evaluation on three benchmarks demonstrates that D2L outperforms existing baselines. The paper also provides ablative experiment to demonstrate the role of the oracle in maintaining sample diversity, and the superiority of D2L’s structured data generation over interpolation-based alternatives.

### Strengths
1. The paper is well organized and can be read straightforwardly. 

2. The method and the experiments consider forward transfer which is overlooked in many CL works, yet is very important in CL.

3. D2L introduces an oracle network that learns when to stop the soft prompt optimization. This idea is novel and to my knowledge unexplored in previous works.

4. Experimental setup is sound and demonstrate that D2L is effective.

### Weaknesses
1. Addressing catastrophic forgetting based on generative replay is a relatively old idea in CL, including several works not referenced in the paper, and hence the novelty of this work is limited. It is true that implementation of this idea is new but the core idea is not mew.

2. D2L relies on a diffusion model which is a large model augmented to the base ResNet-like classifier. This addition makes the model far more complex and given the scope of experiments, one can argue just to use several ResNets, one per task, to get even better performance results.

3. Evaluations include only 10–11 tasks, each with a small number of classes. It’s unclear how the method performs when more tasks are used.  

4. The baselines that were used for comparison are mostly old baselines. It is OK to include them but comparisons need to expand to include all methods of the past three years to demonstrate competitive performance.  

5. The benchmarks that are used are on the simpler side of CL benchmarks at the moment and a relatively old model is used in experiments. Experiments should include more recent benchmarks, e..g, CLEAR or CLAD.

6. The code is not available which makes judgement about reproducibility of the results challenging.

### Questions
1. Why backward transfer is not reported in the experiments? In CL, it is as important as forward transfer. I understand accuracy reflects that to some extend but still it is an independent yet crucial in CL.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a conditional generative replay framework to improve forward transfer in class-incremental learning (CIL). Instead of using generative replay to combat forgetting, the method aims to improve the learning of new classes. Specifically, the authors condition a latent diffusion model on a random image in the replay buffer and one of the previously seen classes to generate novel "dreamed" images. These generated samples are assigned to new pseudo-classes and are used, together with incoming and buffered images, to train the encoder/classifier. The resulting class embeddings are then used to initialize the embeddings of new real classes encountered later, which is argued to promote forward transfer. Experiments on ImageNet-based CIL benchmarks show consistent improvements over buffer replay baselines. Ablation studies show the contribution of each proposed component.

Overall, I think this is an interesting paper that rethinks the role of generative replay in continual learning. The empirical improvements are convincing, and the idea of leveraging dreamed classes for forward transfer is novel. My major concerns are related to the clarity of the presentation and additional analysis on the scaling properties, forgetting behavior, and sources of improvement.

### Strengths
- To my knowledge, the central idea of using generative replay for forward transfer rather than mitigating forgetting is novel and interesting.
- Experiments and ablations show the effectiveness of the proposed method.

### Weaknesses
1. The method involves multiple class mappings and set operations, and their description lacks clarity (see questions below). It might be helpful to point each operation in Algorithm 1 to the sections or equations where that operation is described.
2. The method seems to rely on several class embedding manipulations, and the experiments are performed using ResNet-18. It is not clear what challenges might arise when scaling up this approach to larger data or to settings where classes are not disjoint between tasks. Do the authors have any thoughts on this?
3. The method introduces several new modules, but only the buffer size is controlled when comparing different methods. A time-efficiency analysis might be helpful.
4. Forgetting is not reported. This could be informative---e.g., does the model trade off some forgetting for forward transfer?
5. It is unclear to me whether FWT comes from augmenting past classes with generated images (leading to more robust representations) or from reusing class embeddings for new classes. A helpful baseline would assign the dreamed samples to their conditioning classes without reusing class embeddings for initialization.

### Questions
1. I didn’t understand why you had a mechanism to map dreamed classes to classes in the new task (L175) to improve FWT, but then state that the dreamed classes do not reflect unseen classes (Sec. 4.3). Aren’t these statements conflicting? Why do we not want them to reflect unseen classes?
2. L285: Does "feature embeddings extracted by the classifier" refer to the features before the last layer? Is $f$ the encoder and $F$ encoder+classifier head? 
3. The Oracle uses $\mathbf{Z}_t$ as input features, but I can’t find how the stopping decisions, which are used as targets, are labeled.

### Questions/comments that did not impact the score
4. Please fix the spacing between paragraphs.
5. How did you visualize the latent space in 2D in Figure 3?
6. The paper emphasizes that prior work is not bio-plausible (e.g., L59, L107, L442), but dreams in the brain occur late in the visual cortex [Hor+13], suggesting that the proposed method, which generates pixels directly, is also not bio-plausible. I would adjust the narrative accordingly.
7. L247: Should it be "$c^{\text{out}}$ replaces the dream classes" instead?
8. The end of Sec. 4.2 could be moved to the related work section.
9. L437: This sentence should be a hypothesis rather than being directly supported by Table 5.
10. L442: Why is WSCL not listed in the table, and what is a “true dreaming process”?


[Hor+13] Neural decoding of visual imagery during sleep. Horikawa et al., 2013. Science.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper proposes Dream2Learn (D2L), a generative-replay approach for continual learning (CL) that conditions a diffusion model via soft-prompt optimization to generate semantically distinct and structurally coherent dreamed classes. The dreamed samples are interleaved with training to regularize representations to reduce forgetting. The method results in positive forward transfer on Mini-ImageNet, FG-ImageNet, and ImageNet-R.

### Strengths
1) The paper provides an interesting motivation based on human sleep based replay for CL.
2) Results on standard benchmarks demonstrate superior performance of D2L compared to other replay based methods.

### Weaknesses
1) While catastrophic forgetting is an important problem to consider in CL, the main reason to not do full replay of prior data to avoid forgetting is to reduce the amount of prior data storage, and additional training on prior data to save compute. The proposed method uses significantly larger models to generate more classes to train a small CNN on a relatively small dataset (ImageNet-100). Not only does the model need to train on a large number of generated class images, the method still relies on partial replay of prior data, leading to a much higher compute usage. Additionally, the generative models seemed to have already been trained on huge amount of data, which already covers simple datasets used for CL (e.g. ImageNet-100). In this case, why can't we just use large foundation models as zero shot classifiers on these datasets? Why can't we use full replay of prior data rather than spending compute on generating data with large diffusion models? How do you ensure no leakage from the diffusion model on the ImageNet datasets.
2) There have been many feature replay based techniques that reduce forgetting without spending a signifincant amount of compute. Have the authors considered comparing to those methods? 
3) There is only a small gain in accuracy compared to partial replay methods, such as BiC. Note that BiC does not use any extra generative or pre-trained models, and only relies on a small buffer of prior data. For a fair comparison, it is essential that the other models are provided with similar amount of compute/data. 

Given these concerns, the contribution of this paper to continual learning is minimal.

### Questions
Please see the weaknesses section for my questions.

### Soundness
2

### Presentation
2

### Contribution
1
