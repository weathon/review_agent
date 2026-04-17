# Dynamics of Learning: Generative Schedules from Latent ODEs

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
The learning rate schedule is one of the most impactful aspects of neural network optimization, yet most schedules either follow simple parametric functions or react only to short-term training signals. None of them are supported by a comprehensive temporal view of how well neural networks actually train. We present a new learning rate scheduler that models the training performance of neural networks as a dynamical system. It leverages  training runs from a hyperparameter search to learn a latent representation of the training process. Given current training metrics, it predicts the future learning rate schedule with the best long-term validation performance. Our scheduler generalizes beyond previously observed training dynamics and creates specialized schedules that deviate noticeably from common parametric functions. It achieves SOTA results for image classification with CNN and ResNet models as well as for next-token prediction with a transformer model. The trained models are located in flatter regions of the loss landscape and thus provide better generalization than those trained with other schedules. Our method is computationally efficient, optimizer-agnostic, and can easily be layered on top of ML experiment-tracking platforms. An implementation of our scheduler will be made available after acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
*Disclaimer *  I used chatgpt for better **understanding**  details of sec 3.2.  and nothing else. 

The authors propose a meta learning approach for learning learning rate/step size for first order optimization methods ( i.e. SGD and variants) typically deployment in deep learning.  The idea is to use a pretrained latent ode model that capture training dynamics and model the future trajectories include the triplet (training loss, validation metric, learning rate) from current time * to the end T.  The latent ode model is pretrained using a variety of schedulers. The proposed approach demonstrate the superior test acc on 4 benchmarks in vision + NLP.  It also shows the computation cost in wall clock is lower compared with other schedulers.

### Strengths
A few strengths include 1). The thoroughness of experiment to support major claim including the experiment accuracy and wall clock efficiency compared to baselines. 2). The topic is interesting/relevant to ml optimization especially in large scale dl systems.

### Weaknesses
A few weaknesses include 

1): approach is not novel – NODE has been use before for similar tasks such as area of learning curve exploration( e.g. Architecture-Aware Learning Curve Extrapolation via Graph Ordinary Differential Equation , ding et al. aaai 25) .  I think there are other relevant papers such as Hop, skip, jump to Convergence: Dynamics of Learning Rate Transitions for Improved Training of Large Language Models emnlp 24 

2) approach is less grounded since training loss curve is general not smooth, and neural ODEs require some Lipschitz smooth. 

3).  The last point can be either pro or con, pro is simplicity of approach that is using the only triplet but not including any weight or graph structure of neural network.  the con is loss of potential information for modeling / future trajectory.

### Questions
1.	Can the author provide some insight to address weakness 2 training loss curve is general not smooth, and neural ODEs require some Lipschitz smoothness for well postedness and stability? However the loss curves are generally not smooth, so the metric value for validation, and learning rate can also be not lip-smooth. How can you justify the well-posedness and stability for your neural odes.  


2.	Efficiency. I think the analysis of efficiency is to support one of your major claims.  I think in the proposed approach which requires some training using transformer model for your encoder and decoder which should account for/toward total wall clock times but it is not.  The authors mention it “is negligible given the cost of the hyperparameter search for any moderately large training model. “ If i understand correctly other baselines compared in the table are using some hyperparameter search.  nontheless, the pretraining should be included. Also some non wallclock or quantative analysis will support better for your claim.

3.	Accuracy. I also have a question about generalization. Since LODE is pretrained from all scheduler. Now in terms of test model generalization/accuracy, does it just memorize pretrain trajectory with certain schedulers  that best performing in terms of validation loss ?  The proposed approach may be more powerful if it is online. Some strategy like Learning to Boost Training by Periodic Nowcasting Near Future Weights paper.

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
3

### Summary
The paper considers modeling the learning rate schedule from trajectories of training metrics (loss, validation accuracy, etc.) of prior runs. The approach is evaluated on image classification and next token prediction settings and compared to standard hand-crafted learning rate schedules.

### Strengths
The paper studies an interesting problem that has been largely overlooked by the community. The initial experiments are promising.

### Weaknesses
- Experimental evaluation is overall limited. The authors include multiple datasets (FashionMNIST, CIFAR100, ImageNet) and models (ResNet-18/34) which is great. However, many of these datasets and models are fairly simple and make it hard to distinguish between different approaches (e.g., all of the tested methods perform similarly on FashionMNIST). It would be good to include evaluations with more challenging models and datasets. For example, in the case of image classification, showing results with ViT and ResNet-50 models on ImageNet would make it easier to compare against existing baselines.
- It would be good to include simple baselines that are based on the collected dataset of training runs. The paper currently compares hand-crafted schedules like constant or cosine to the proposed approach. It would be good to include a few simple baselines that e.g. learn linear weights between existing schedules. This would help understand how much of the gains are due to using prior data vs the specific method for using prior data.
- If I understand correctly from the last sentence of Section 5.3, the dataset used to model the learning rate schedule consists of 100 training runs. That sounds fairly small and it would be good to perform careful experiments to understand the level of generalization with respect to the training data. These could include nearest neighbor retrieval baselines or similar strategies to understand the level of memorization, interpolation, and generalization.
- It would be good to include much more details on the training data composition in terms of (task, model, schedule, performance). Related to my previous comment, it is a bit hard to interpret the current results without that.

### Questions
Please see the weaknesses section.

### Soundness
2

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
The paper proposes a Latent ODE (LODE)–based learning-rate scheduler that learns a continuous-time latent dynamics model from routine training traces—training loss, validation metric, and learning rate—collected during a conventional hyperparameter grid search. At test time, the scheduler encodes the most recent window of traces, samples a latent ensemble, integrates the ODE forward for each sample to predict future trajectories, and selects/averages the top-K learning-rate candidates that maximize long-horizon validation performance for the next 
𝜇
μ steps (Algorithm 1). On CNNs/ResNets (Fa-MNIST, CIFAR-100, ImageNet) and a small Transformer (TinyStories), the method consistently outperforms parametric schedules, hypergradient descent, schedule-free, and an RL baseline on final accuracy, while being ~25% slower per epoch than simple parametrics but much cheaper than hypergradient and RL. Notably, authors claim LODE inference cost does not scale with model size, unlike RL methods.

### Strengths
Long-horizon objective: The scheduler explicitly optimizes for future validation performance, enabling exploratory high-LR phases followed by late-stage annealing; greedy short-horizon variants underperform. 


Simple inputs, broad applicability: Uses only three traces routinely logged in experiments; no access to model parameters is required. 

Consistent wins across tasks: LODE beats all baselines across CNN/ResNet/Transformer settings with tight accuracy variance. 

Reasonable overhead: ~25% more wall-time than parametrics, much cheaper than hypergradient and RL; plus cost said not to scale with model size.

### Weaknesses
Heavy reliance on the quality/diversity of traces from the initial hyperparameter search. Ablations show outcomes depend on schedule diversity (notably presence of OneCycle) and benefit mildly from more random initializations; authors ultimately train each LODE from 100 runs (5 seeds × 4 schedules × 5 LRs). This creates a data-collection dependency that practitioners must meet. 

RL baseline choice and recency. The primary implemented RL controller is Xu et al. (2019); the paper mentions Xiong et al. (2022) but (per the note) does not run an official or up-to-date RL implementation and even flags that their used code may not be runtime-optimal. It also skips RL on ImageNet and the Transformer due to cost, while reporting ImageNet RL numbers from Xiong et al. (2022). Stronger comparisons should at least include Xiong-style controllers run under the same setup. 

Scaling claim vs. transfer. Authors assert LODE inference cost does not scale with model size (unlike RL). But they do not evaluate RL transfer patterns (e.g., training the RL policy on a small model/dataset and reusing it on larger ones), and they train a separate LODE per dataset–model (no cross-model transfer is demonstrated). Thus, the paper hasn’t shown that LODE transfers across model sizes/datasets, whereas RL methods in principle can be trained small and reused at inference. The scaling claim is supported by their runtime table and the decision not to run RL on larger tasks, but transferability remains untested for both, with LODE clearly not demonstrated. 

External validity limits. Training data for LODE excludes hypergradient/RL/schedule-free runs “to stay within conven

### Questions
Trace quality & minimal budget. What’s the minimal number and diversity of runs (schedules × LRs × seeds) to reach “near-paper” performance, and how does this vary by dataset/model? Current results suggest sensitivity to OneCycle presence and use 100 runs per LODE. Could you provide a practitioner recipe for “small-budget” settings? 

RL comparison. Why not include a modern RL scheduler (e.g., closer to Xiong 2022) trained/evaluated under the same hardware/runtime budget? Today’s RL controllers can often be trained on small proxies and reused—how would that alter your “RL doesn’t scale” claim?

Transfer & reuse. You train one LODE per dataset–model. Can a single LODE trained on a small proxy transfer to a larger model or a related dataset (possibly with lightweight adaptation)? If not, what prevents reuse—latent space mismatch, encoder overfitting to specific loss dynamics, or something else?

### Soundness
2

### Presentation
3

### Contribution
2
