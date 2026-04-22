# Towards Dynamic Interleaving Optimizers

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Optimizers are critical for training deep neural networks. Existing training processes rely on a single static optimizer (e.g., SGD) or a simple hybrid of two optimizers, which miss the opportunity to exploit evolving dynamics in different training states, degrading model quality and convergence. In this paper, we propose a novel dynamic optimizer switching method called **D**ynamic **O**ptimizer **I**nterleaving **T**raining (DOIT) method, which builds surrogate models to predict different optimizers' performance from current parameter states. DOIT uses an acquisition function that combines the results from surrogate models with transferability assessments and process information to select a suitable optimizer for the subsequent training. Experiments on various models and tasks (e.g., image and text classification, machine translation, and object detection) show that DOIT effectively enhances the training, achieving faster convergence (i.e., 2\% to 10\% faster) and higher accuracy (i.e., 1\% to 3\% improvement). Additional independent experiments and case studies further validate DOIT's effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DOIT (Dynamic Optimizer Interleaving Training), a novel method that dynamically switches optimizers during training. The core hypothesis is that different optimizers are better suited for different training states. DOIT models this problem by building Gaussian Process (GP) surrogate models to predict the "short-term" performance of different optimizers based on a PCA-compressed representation of the current parameter state. A well-designed acquisition function—which considers predicted performance, uncertainty, task transferability, and training progression—is used to select the optimal optimizer for the next training cycle.

The authors provide extensive empirical validation across a wide range of tasks (image classification, text classification, machine translation, object detection), models (ResNet, ViT, RoBerta), and settings (full training, PEFT). The results consistently show that DOIT achieves higher accuracy (1-3% gains) and faster convergence (2-10% faster) compared to a large suite of static and simple-hybrid optimizers.

### Strengths
1.The motivation of this paper is direct and clear. The paper presents a principled approach to optimizer selection. Moving from static selection or simple two-stage hybrids (like SWATS) to dynamic, state-aware selection framework is a significant conceptual leap.

2.The breadth of tasks, datasets, and models is comprehensive and convincing. The inclusion of ablation studies, cost analysis, and case studies on switching behavior thoroughly validates the method's design and effectiveness.

3.Initially i questioned if the overhead of training multiple GP online would be significant. However, the authors provide a detailed cost analysis and empirical measurements suggesting that the overhead from surrogate modeling and selection is minimal (often <1%), making the method practical.

### Weaknesses
1.The most significant weakness is that the method resets optimizer states (e.g., momentum, second-moment estimates) upon every switch. This seems counter-intuitive and potentially disruptive, as the optimizer loses all accumulated history. The authors defend this in Appendix, arguing that state inheritance is complex and their empirical tests (Table 15) showed it can destabilize training. However, this should be further proven. 

2.The surrogate model's input is a PCA projection of some model layers. The paper is not perfectly clear on which layers are chosen for the different models, and how sensitive the performance is to this choice (e.g., classifier only vs. final N blocks).

### Questions
1.Some pervious work also explore the potential of dynamic hybridization of different optimizer, such as HUB (hybrid-update-based optimization strategy). They emphasized the hybridization with learned optimizers. Could you testify if DOIT could also handle the switching timing for those optimizers?     

2. How was the subset of layers for PCA compression chosen for the main experiments (e.g., for ViT and RoBerta)? How sensitive is DOIT's performance to this choice?

3.The acquisition function includes a transferability weight $\omega_t$. Does this imply the surrogate models must be re-trained from scratch for every new task, or can the GP models themselves be transferred?

### Soundness
3

### Presentation
2

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
This paper proposed a new framework called Dynamic Optimizer Interleaving Training (DOIT) for training deep neural networks. The main idea is to use a surrogate model, more specifically, a Gaussian Process (GP), to assess the performance of the optimizers, and then dynamically select the best one from the candidates. Experiments on several datasets have been conducted to demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper is well-written and easy to follow.
2. The performance of the proposed method seems good.

### Weaknesses
1. The novelty of this work is not significant enough. Specifically, the key component of the proposed framework is a surrogate model constructed by GP, which, however, is not unusual and has been widely used in hyperparameter optimization. Despite this key component, other designs are mostly engineering and heuristic, such as the performance score and the acquisition function.

2. There is no theoretical convergence guarantee for the proposed method. Since the training process by the proposed framework includes the switch of multiple optimizers and corresponding hyperparameters, the convergence behavior could be more complex than a single optimizer, and thus, a theoretical convergence guarantee can be useful to guide practice. Besides, it would also be interesting to analyze how good the surrogate function is and how it could affect the final performance.

### Questions
First, the authors should address the issues mentioned in the "Weaknesses". Despite these weaknesses, several more things should be clarified:

1. The authors only report the FLOPs of the proposed method, but it is also important to consider the memory overhead of the additional operations, such as PCA compression and GP prediction.

2. It seems that the optimizer configuration candidates lie in a discrete space, and thus, it could be tricky to construct the candidates.

### Soundness
2

### Presentation
3

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
This paper introduces a novel method called Dynamic Optimizer Interleaving Training (DOIT), which dynamically switches between optimizers during training to enhance model performance. Traditional training processes typically rely on a single static optimizer or a simple hybrid, missing opportunities to adapt to changing dynamics during training. DOIT builds surrogate models to predict optimizer performance based on the current parameter states and selects the most suitable optimizer using an acquisition function. Experiments across tasks such as classification, translation, and object detection show that DOIT improves convergence speed (2% to 10%) and accuracy (1% to 3%). Further case studies validate its effectiveness.

### Strengths
1. The paper proposes a dynamic optimizer to improve training performance.

2. The paper provides comprehensive experiments in both the vision and NLP domains, validating the proposed method's effectiveness.

3. The paper includes a comparison of training efficiency, which is quite useful.

### Weaknesses
1. The method requires retaining multiple optimizers, which may lead to high memory usage. The authors should include a direct comparison to address this concern.

2. As mentioned by the authors, the method does not support optimizers with momentum. Since momentum is a key mechanism in current mainstream optimizers, I recommend that the authors propose a feasible solution to extend their method to support momentum-based optimizers.

3. While some details still need clarification, I understand that the method retains multiple optimizers and uses strategies like performance-aware weighting after a certain number of steps (Eq. 6) for weight updates. In this process, where exactly is the "training dynamic" reflected? Additionally, doesn’t this approach resemble ensemble learning? Furthermore, the method seems related to meta-learning; it would be helpful for the authors to analyze and compare these connections.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces dynamic interleaving of optimizers (DOIT) framework that adaptively switches among multiple optimizers during training based on the training state. The method constructs and continually updates Gaussian surrogate models for each optimizer to estimate their expected performance under different conditions. At each switching step, the optimizer predicted to yield the highest improvement is selected for the next training phase.

### Strengths
- Novel approach for interleaving optimizers during training based on surrogate quantitative performance
- Experiments are conducted on a variety of datasets (6 CV, 2 NLP and 3 additional tasks) with two settings: full training and fine-tuning setups and compared against standard optimizer baselines (although a bunch of these tasks are small as per today's standards, see weaknesses below)

### Weaknesses
- The approach introduces a lot of additional hyper-parameters (see questions below). Moreover, optimal DOIT hyperparameters are different per dataset which limits the pratical utility of the approach and introduces additional tuning overhead for the practitioner. 
- The proposed approach incurs high memory and runtime overhead which are reduced by arbitrary heuristics such as PCA over parameters, additional sub-selection of layers (only last). It's unclear if these heuristics will hold for large-scale ML workloads. Similarly, acquisition function is hand-designed with heuristics. These are central components of the proposed approach and non-trivial changes to the approach may be required to make it scalable to standard ML tasks on which optimizers are usually benchmarked (GPT-2, ViT, ResNet50, AlgoPerf benchmark, etc).
- Experiment datasets are small as per today's standard (4/6 CV datasets are mnist-scale, imagenet is the only relevant one) and also heavily biased towards vision tasks (6 CV datasets, coco object detection)
- The approach requires extra surrogate model (Gaussian processes) which greatly limits scalability of the proposed approach (n^3 complexity). 
- In order to collect experience data for surrogate model, training is performed with random optimizer configurations -- wouldn't this lead to irreversible damage to the parameters of the model which may be hard to recover from at later stages in the training?
- While DOIT switching is mainly guided by the short-term benefits of each optimizer, practitioners aiming for highly efficient and performant training also consider long-term effects, taking into account the total number of iterations, final accuracy, and selecting hyperparameters such as learning rate, weight decay, and their schedules accordingly.

### Questions
Some questions on hyperparameters and heuristics: 
- how long is the experience collection phase per dataset and what % of training is dedicated to it? is it a tunable parameter?
- what are hyperparameters of gaussian processes model? how are they chosen for each dataset/task?
- how is switch cycle tau chosen? Is it an extra tuning parameter?
- which layers are sub-selected for surrogate model for each dataset? 
- for a given selected optimizer in a switch cycle based on acquisition function score, how are optimizer hyper-parameters chosen (such as learning rate, momentum, weight decay, betas if applicate etc)?

Moreover, can authors please elaborate on the key runtime/memory bottlenecks in the current approach? What are the most and least expensive parts in the approach and how do they change (if they do) based on DOIT hyperparameters?

### Soundness
2

### Presentation
3

### Contribution
2
