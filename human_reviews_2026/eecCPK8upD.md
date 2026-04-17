# Noradrenergic-inspired gain modulation attenuates the stability gap in joint training

- Decision: Reject
- Scores: 4, 4, 2, 2, 4

## Abstract
Recent work in continual learning has highlighted the *stability gap* -- a temporary performance drop on previously learned tasks when new ones are introduced. This phenomenon reflects a mismatch between rapid adaptation and strong retention at task boundaries, underscoring the need for optimization mechanisms that balance plasticity and stability over abrupt distribution changes. While optimizers such as momentum-SGD and Adam introduce implicit multi-timescale behavior, they still exhibit pronounced stability gaps. Importantly, these gaps persist even under ideal joint training, making it crucial to study them in this setting to isolate their causes from other sources of forgetting. Motivated by how noradrenergic bursts transiently increase neuronal gain under uncertainty, we introduce a dynamic gain scaling mechanism as a two-timescale optimization technique that balances adaptation and retention by modulating effective learning rates and flattening the local landscape through an effective reparameterization. Across domain- and class-incremental MNIST, CIFAR, and mini-ImageNet benchmarks under task-agnostic joint training, dynamic gain scaling effectively attenuates stability gaps while maintaining competitive accuracy, improving robustness at task transitions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The stability gap in Continual Learning is a phenomenon in which performance temporarily declines when a new task is introduced. Previous studies have shown that this occurs mainly because the strong signal from the loss leads to rapid and strong weight modifications. The authors of this study examined this phenomenon in deep learning models from the perspective of biological models. Biological models can learn new tasks through a multi-timescale dynamics that gradually allows new information to be incorporated into neurons. Inspired by this, the authors present an optimiser that accounts for two-timescale dynamics (fast and slow), enabling the balance between adaptation and stability in sequentially learned models. To study the problem in detail, the experiments focus on an environment with sequential tasks but assuming that all previous tasks are accessible (joint learning). Experiments are conducted on different variants of MNIST, CIFAR, and Mini ImageNet.

### Strengths
- Works inspired by biological models help us better understand how we can expand the capabilities of current models. The authors do a good job of linking a specific limitation (the stability gap) of current models to mechanisms that can inspire future improvements.
- The idea of adapting the optimiser with a two-timescale approach is interesting. As the authors mention, it follows the line of work that seeks fast adaptations (fast weights) and longer-term adaptations.

### Weaknesses
- The main contribution of the work is an optimiser that approximates the two-timescale gradient descent strategy to mitigate the stability gap. However, no significant benefits are seen in the results. Table 1 shows a slight improvement over SGD but introduces a level of complexity that has not been studied. Figure 2 shows that NGM-SGD helps control the loss in Task 1, but there is no significant difference compared to SGD.
- At the beginning of Section 4, it is stated that the aim is to study ‘How to optimise and not what to optimise’, but I believe the two components are closely related and difficult to study independently.
    - Could you explain in more detail what you mean?
- The paper raises two very interesting questions in the introduction (lines 62-64). However, I am not sure that they are answered in the paper. In Section 3, the method is presented, but no direct connection is made to the biological brain.

### Questions
- Can we view "g" as a value similar to the momentum of other optimisers, but which changes as described in equation 2?
- Did you conduct experiments in which you reset the optimisers for each task? As mentioned in the paper, the momentum of Adam or M-SGD can negatively affect the stability gap, which can be tackled by resetting the momentum values.
- Did you conduct experiments with different values for the number of iterations? Previous work has shown that models that are trained for more iterations achieve greater stability. It would be interesting to study this statement under this scenario.
- The motivation for proposing this new optimiser is to reduce the strong weight modifications by using the g values (which should capture most of the task change). Did you study how the changes in the weights "w" compare with those of other optimisers? Could the reduction in plasticity have a long-term effect on the model's performance?
    - Conclusions may not be difficult to obtain for simple tasks or those with few iterations. More complex tasks or those with greater changes in task distribution may have different behaviours.
- Line 303 mentions that they restrict modulation to only the last layer of the ResNet. Did you experiment with the entire model? Did you conduct experiments with pre-trained models (ResNet pre-trained with ImageNet and on datasets outside the distribution, such as CUB)?
    - The latter may be interesting, as it would give an idea of how the optimiser behaves when big variations in the weights are needed.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies and formalizes the “stability gap” in continual learning—transient performance drops at task boundaries that persist even under ideal joint training—and attributes it to optimization dynamics rather than objective choice. It proposes a biologically inspired optimizer, Noradrenergic Gain-Modulated SGD (NGM-SGD), which introduces a fast timescale, uncertainty-driven neuronal gain (triggered by output entropy) atop standard weight updates, effectively scaling the loss and flattening local curvature to curb overshoot while accelerating adaptation. The authors provide an intuitive two-timescale interpretation (fast gain × slow weights) and show that modest, head-only gain modulation suffices for deep architectures. Across class- and domain-incremental benchmarks, NGM-SGD consistently narrows the stability gap (higher min-ACC, lower avg-SG) without sacrificing final accuracy.

### Strengths
The work draws an interesting inspiration from noradrenaline-driven gain modulation, offering a principled lens on when and how a learner should adapt under distribution shifts in continual learning.

### Weaknesses
1. The paper is difficult to follow. It leans heavily on terminology from biological neural circuits without sufficient plain-language grounding or progressive intuition.
2. The method is primarily targeted at narrowing the stability gap during task transitions, but it does not directly address the central challenge of continual learning—catastrophic forgetting over long horizons. The proposed gain-modulated SGD feels like a modest variant of existing two-timescale or adaptive-step-size ideas rather than a fundamentally new optimization principle.
3. The evaluation is limited to small or mid-scale datasets. Without results on larger, more realistic settings, it is hard to assess robustness, training dynamics under real-world complexity, and computational overhead.
4. The comparisons focus on relatively dated optimizer baselines, and the reported improvements are modest. The method is not evaluated within mainstream continual learning frameworks. This raises uncertainty about extensibility and practical utility.

### Questions
see Weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the stability gap problem in continual learning. The authors propose a biologically-inspired approach based on noradrenergic gain modulation, where neuronal gains are dynamically adjusted based on prediction uncertainty. The proposed NGM-SGD implements uncertainty-driven gain boosts that create fast-slow weight dynamics and flatten the loss landscape. The approach is evaluated on domain-incremental and class-incremental benchmarks under ideal joint training conditions, demonstrating reduced stability gaps compared to standard optimizers.

### Strengths
**Solid theoretical foundation**
The mathematical framework connecting gain modulation to fast-slow weight decomposition is clear and intuitive. The analysis showing how gain boosts flatten the loss landscape provides mechanistic insight.

**Clear, implementable algorithm.**
The application of NGM-SGD seems simple, with standard SGD weight update plus a gain update driven by prediction entropy each iteration. The lack of architectural changes, replay buffers, or extra losses makes it practical, and the idea that such a minimal mechanism can shrink the stability gap in continual learning is compelling.

### Weaknesses
**Novelty with respect to biological grounding.**
The authors claim that no prior work has adopted a bio-inspired approach to mitigate the stability gap or connected it back to adaptive biological learning. However, the complementary learning systems (CLS) literature has long modeled fast/slow learning via mimicking the hippocampus–neocortex interactions of the brain, and many continual learning methods explicitly borrow this paradigm through dual-memory architectures, replay-based consolidation, etc. [1, 2] The proposed gain-modulated fast/slow decomposition closely echoes this CLS framing. The manuscript should acknowledge these approaches, explain the similarities or distinctions, and include comparisons or at least a reasoned discussion against strong CLS-style baselines. Without this positioning, the biological novelty claim feels somewhat overstated.

**Limited empirical scope.**
While the paper notes the simplicity of its benchmarks as a limitation, this is not a minor caveat. It is a necessary extension to substantiate the paper. Without evaluations on more challenging settings (e.g., longer task streams, larger-scale datasets, online/streaming protocols without task IDs, and realistic memory/compute constraints) it is difficult to conclude that the method genuinely mitigates stability gaps rather than benefiting from the specifics of the setup. Also, comparisons should be made with recent continual learning methods specifically designed to address stability gaps. Expanding the study to stronger baselines and modern architectures would further strengthen the claim.

**Hyperparameter sensitivity**
The method introduces additional hyperparameters, requiring task-specific tuning. The values vary significantly across datasets, suggesting the method may not generalize well. A guidance on how to set these parameters for new tasks would be helpful.

References

[1] Arani, Elahe, Fahad Sarfraz, and Bahram Zonooz. "Learning fast, learning slow: A general continual learning method based on complementary learning system." arXiv preprint arXiv:2201.12604 (2022).

[2] Pham, Quang, Chenghao Liu, and Steven Hoi. "Dualnet: Continual learning, fast and slow." Advances in Neural Information Processing Systems 34 (2021): 16131-16144.

### Questions
1. The performance improvements seems modest and inconsistent. Why does vanilla SGD or other baselines sometimes outperform multi-timescale optimizers? This seems counterintuitive given your multi-timescale argument.

2. As mentioned, in CNN experiments, gain modulation is applied only to the output layer. What happens if you apply gain modulation to all layers in CNNs instead of just the output layer? Is there an adaptive way to slightly modify the methodology so that gain modulation can be applied to all layers of CNN?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper addresses the stability gap in continual learning by introducing uncertainty-modulated gain dynamics as a two-timescale gradient descent adjustment  that balances adaptation and retention by adjusting learning rates and flattening the energy landscape. The authors demonstrate analytically that this neuron modulation gain induces fast-slow weight scales and flattens the local loss surface near the minima. The authors evaluate their method on MNIST, CIFAR, and mini ImageNet continual learning benchmarks against baseline optimizers: momentum-SGD, Adam, and SGD.

### Strengths
- Empirical evidences shows that NGM-SGD reduces test loss at task boundaries.
- Empirical evidence shows that NGM-SGD reduces the stability gap.

### Weaknesses
- See the first bullet point in the questions, why is it necessary to compare NGM-SGD only to other optimizers: SGD, Adam, MSGD? Could there not exist some continual learning method that outperforms MSGD in the metrics illustrated in Table 1? Given this lack of a comparison to existing continual learning methods, why do the results support the efficacy of NGM-SGD?
- Overall, the empirical results are mixed, see Table 1. For instance, the baseline optimizers attain comparable if not often better performance on many of the reported metrics, than NGM-SGD. While some clear benefits of NGM-SGD are observed, the overall mixed results and limited scale and scope of the experiments puts into question the efficacy of the method.
- While the biological motivation well motivates the proposed method, for many readers parsing this information can be difficult. It would be useful for many readers if the algorithm and contributions were distilled algorithmically, rather than solely being motivated from a neurological phenomenon, earlier in the paper.

### Questions
- Why are only SGD, Adam, and MSGD evaluated against NGM-SGD? Could there be existing CL methods that outperform NGM-SGD and would be worthwhile comparing? How should we think of NGM-SGD interfacing with other continual learning interventions?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The work introduces a method that effectively decomposes weights into two components, a slow component and a fast component. This is implemented as a neuronal gain that is multiplied to the weights of the neural network. At each step, this gain is decayed to some base value and increased based on the neural network’s uncertainty, which is biologically inspired by noradrenaline. The paper shows that their approach mitigates the “stability gap” that occurs in continual learning when switching tasks.

### Strengths
- The idea of using gain modulation as a flexible way to handle distribution shifts, and showing it can help with the stability gap is good, and it is empirically validated in the supervised continual learning experiments that are presented.
- Neuronal gain as a proxy for task complexity is interesting. The results make sense as the neuronal gain is essentially moving average of the entropy of the outputs.

### Weaknesses
- Overall, while the method does result in an optimizer that mitigates the stability gap, it does seem to do that at the expense of overall performance. 
- The proposed method has significantly more hyperparameter configurations evaluated compared to the baselines (15x more). This could very easily be the reason for any performance gains of NGM-SGD.
- I am not sure leaning so heavily into the biological framing is useful/correct. One of the contributions is listed as “We link our algorithmic gain bursts to noradrenergic neuromodulation” but there is not really much in the paper linking what happens in the biology to what’s happening in the artificial networks, other than a vague notion of uncertainty. Saying it’s biologically inspired is fine, but framing one of the contributions of your paper as “noradrenergic neuromodulation” seems like overclaiming. It’s also unclear if what is used for the uncertainty proxy in the paper (the entropy of the softmax) is valid given the networks are uncalibrated.
- I think a bit more work should be done on the dynamics of how the gain evolves with how the output evolves. Specifically, if the gain goes up, the norm of all weights increases, how does that affect the effective learning rate?

### Questions
- The flattening effect that you mention, isn’t it mitigated by the network taking effectively smaller steps? Could you give more detail into how the trajectory followed by the optimizer would look like the trajectory in a flattened minima?

### Soundness
2

### Presentation
2

### Contribution
3
