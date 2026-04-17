# HyperFlow: Gradient-Free Emulation of Few-Shot Fine-Tuning

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
While test-time fine-tuning is beneficial in few-shot learning, the need for multiple backpropagation steps can be prohibitively expensive in resource-constrained environments or end devices. To address this limitation, we propose a computationally efficient test-time adaptation approach that emulates gradient descent without computing gradients. Specifically, we formulate gradient descent as an Euler discretization of an ordinary differential equation (ODE) and train a lightweight auxiliary network to predict the task-conditional drift using only the few-shot support set. The adaptation then reduces to a simple numerical integration (e.g., via the Euler method), which requires only a few forward passes of the auxiliary network—no gradients or forward passes of the target model are needed. In experiments on cross-domain few-shot classification using the Meta-Dataset and CD-FSL benchmarks, our method significantly improves out-of-domain performance over the non-fine-tuned baseline while incurring only 6% of peak memory and 0.1% of the FLOPs of standard fine-tuning, thus establishing a practical middle ground between direct transfer and fine-tuning approaches.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces HyperFlow, a test-time adaptation method for few-shot learning that emulates gradient descent without backpropagation. It models fine-tuning as an ODE (gradient flow) and trains a lightweight, task-conditioned drift network that takes a few-shot support set and predicts parameter updates. At inference, adaptation reduces to forward-only ODE steps, yielding strong efficiency: the authors report roughly 7% of peak memory and 0.1% of FLOPs compared to standard fine-tuning, while improving out-of-domain (OOD) accuracy over non-fine-tuned baselines. Despite compelling efficiency gains, the approach is weakened by its lack of offline cost and its inferior accuracy compared to standard (and even bias-only) fine-tuning.

### Strengths
- Casting fine-tuning as a learned task-conditioned ODE enables gradient-free adaptation using only forward passes.
- **Efficiency:** Clear compute/memory wins while beating direct transfer (and some head-tuning) on OOD benchmarks.

### Weaknesses
- **Unreported offline cost:** The method depends on generating fine-tuning trajectories and training the drift network; the compute, time, and storage for these steps are not quantified, making the total cost unclear. 
- **Accuracy gap vs. stronger baselines:** Few-shot accuracies lag full fine-tuning and bias-only fine-tuning, which questions when HyperFlow is preferable beyond strict resource constraints.
- **Scalability:** The approach appears tied to a restricted parameter subset; it’s unclear how performance and efficiency trade off when expanding beyond that (e.g., LoRA/adapters) or switching backbones.

### Questions
- What are the FLOPs, wall-clock time, and storage required for (i) collecting fine-tuning trajectories and (ii) training the conditional drift network with interpolated ODEs?
- How sensitive is performance to the number/size of ODE steps at test time?
- How does accuracy/efficiency change when expanding the adapted parameter set (e.g., LoRA/adapters)?

### Soundness
1

### Presentation
2

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
This paper reinterprets test-time few-shot adaptation as the gradient-flow ODE and trains a lightweight, support-set–conditioned drift network to predict the parameter velocity field, enabling adaptation without backpropagation via only a few steps of Euler integration. On Meta-Dataset and CD-FSL, HyperFlow improves over Direct Transfer while requiring only about 0.1% of the FLOPs and 7–10% of the peak memory of standard finetuning, thereby achieving a practical middle ground in the performance–cost trade-off.

### Strengths
1. Intuitive to understand
Framed as “fine-tuning = Euler discretization of a gradient-flow ODE” the pipeline is explained clearly: a task representation from the support set conditions a drift network whose forward passes, combined with simple numerical integration, effect adaptation without gradients. The interpolation methods—linear, piecewise-linear, and cubic —are well organized and accessible.

2. Systematic Experimental Design
Fairness Controls: Comparisons include Direct Transfer, linear-head, bias-tuning, and full fine-tuning. To ensure a fair budget, HyperFlow-L/C performs up to 50 Euler steps, aligning the number of update steps with gradient-based fine-tuning. 
By presenting results that vary the ODE objectives and the diversity of training trajectories (domains, episode counts, and initialization noise), the paper provides a factor-wise analysis of performance and generalization.

### Weaknesses
1. Backbone Generalization
The classifier is ProtoNet + ViT-Small, and the drift network’s task encoder is a frozen ResNet-18. The paper provides no empirical evaluation on alternative backbones (e.g.CLIP-ViT), leaving backbone generalization as an open limitation.

2. Implicit Gradient Dependency and Limited Generalization
While HyperFlow claims gradient-free adaptation, its drift network is trained on gradient trajectories from base tasks, meaning gradient information is still implicitly used. This raises concerns about how well the learned dynamics generalize when the loss landscape or data distribution differs from training, and since training the drift network itself requires computing gradients over simulated trajectories, the claimed efficiency advantage should be reconsidered in light of this additional training cost.(If the drift network requires additional training for distribution shifts)

3. Missing theoretical guarantees for the interpolation-based ODE objective
Relies on linear/PL/cubic interpolation and a learned drift but provides no formal guarantees on convergence, Euler-step stability, or drift–gradient approximation error. Potential issues aren’t analyzed, and no sufficient conditions are given—evidence is empirical only.

4. Under-reported offline trajectory cost
The paper does not quantify the compute/energy/storage cost of collecting trajectories as a function of episodes, initializations, and GD steps, nor provide budget–accuracy trade-offs or a break-even analysis versus fine-tuning.

### Questions
1. Would the performance gap from direct transfer still remain if the authors applied it to CLIP-ViT(B/16, L/14), or might it actually vanish?
2. If the authors tried a model like ConvNeXt, which lacks the qkv mechanism, how could this method be adapted — and would it still lead to any measurable performance improvement?
3. On the ChestX dataset, the loss decreased only gradually — is that truly due to the backbone’s limitation, or could a different backbone reveal a different trend?
4. It seems that the results in Figure 3(a) and Tables 1 and 2 are inconsistent. Specifically, the performance of HyperFlow-C in Figure 3(a) appears to be better than that of Bias-Tuning, whereas in the tables it is not. Could you clarify why there is a discrepancy between these results?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes HyperFlow, a method for gradient-free few-shot learning. A hypernetwork is trained based on few-shot training trajectories. Given a conditioning signal (the shots) and a continuous update step, this network learns to predict the gradient of the model parameters. When new shots are presented to the hypernetwork, an integration over the simulated gradient updates can be carried out to adapt a base model.

### Strengths
The paper is well written and easy to follow. The method is explained well and all experimental setups are clear. While the presented performance gains are marginal compared to bias-tuning, the method greatly reduced the memory footprint and computational cost of finetuning.

### Weaknesses
The experimental part is limited, comparing HyperFlow to 1) direct transfer, 2) head-tuning, 3) bias-tuning and 4) full-tuning. However, it is known that parameter efficient finetuning methods like LoRA (https://arxiv.org/abs/2106.09685) or sparse variants are very powerful for few-shot adaptation of pre-trained methods. 

Extending the experiments could clearly help to strenghten the paper’s contribution.

### Questions
I have a few conceptual questions:
1) Does the hyper-network generalize to increasing/decreasing number of shots?
2) Is the ODE solver always executed for a fixed number of steps? Does the gradient estimated by the hyper-network vanish after these steps?
3) For a linear flow as proposed in Eq. 5-6, the time derivative is constant. Hence, integration boils down to a simple multiplication with the time interval. So effectively, the hyper-network is trained to predict a scaled one-step update. Is this correct?

About the results:
1) In Table 2, the HyperFlow method sometimes improves upon bias-tuning. For me this is a bit strange, since HyperFlow is trained to approximate bias-tuning episodes. I would love to hear some thoughts about that.
2) Figure 4 shows the relative loss over the Euler steps. It seems that the update is not stable, because the loss increases in the end. Is this due to overfitting or because the predicted gradient does not vanish and we overshoot the optimal point?
3) Another question related to stability: Did you try if running Euler integration on training data could recover the optima from the training episodes?

And finally: Am I right that Eq.6 should be the time derivative of Eq.5?

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
This paper proposes HyperFlow, a gradient-free alternative to test-time fine-tuning for few-shot classification. The key idea is to emulate gradient descent as an ODE: train a lightweight conditional drift network that predicts task-conditioned parameter velocities from the support set, then adapt by numerical integration without computing gradients of the target model. To keep the problem tractable, HyperFlow only updates a PEFT subset of parameters, learned from offline trajectories simulated on meta-train tasks and interpolated either linearly, piecewise-linearly, or via cubic splines. On Meta-Dataset and CD-FSL, HyperFlow improves OOD performance over direct transfer and sits between direct transfer and full fine-tuning in accuracy.

### Strengths
1. Clear core idea: reinterpret gradient descent as an ODE and learn a task-conditioned drift field to perform adaptation with forward passes only. 

2. Practical efficiency: compelling compute/memory analysis showing 0.1% FLOPs vs full/bias tuning and ~7–10% of their peak memory, with only ~0.8× overhead vs direct transfer. 

3. Solid empirical coverage: results across Meta-Dataset (10 datasets) and CD-FSL (4 datasets) with 600 episodes each; multiple shots; in-domain vs OOD breakdown.

### Weaknesses
1. Dependence on offline trajectories: Training requires generating 40k fine-tuning trajectories with multiple random initializations; this offline cost and storage footprint are not quantified and may be non-negligible. 

2. Limited backbones and tasks: Experiments focus on ViT-Small ProtoNet for classification; no analysis on larger encoders or other task, such as few-shot classification for CLIP fine-tuning (coop, cocoop etc).  

3. Hyperparameter selection: Step counts/step sizes for ODE solving are selected similarly to LR tuning in fine-tuning; more detail on this selection protocol and its sensitivity would help.

### Questions
1. Trajectory cost: What are the wall-clock time / GPU hours and storage size for generating the 40k trajectories? Could a smaller set suffice without hurting performance? 

2. Backbone generality: Can the conditional drift network trained on ViT-Small generalize to larger ViTs or ConvNeXt without retraining? What fails in cross-backbone transfer?

### Soundness
3

### Presentation
3

### Contribution
3
