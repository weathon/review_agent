# Accelerated Predictive Coding Networks via Direct Kolen–Pollack Feedback Alignment

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Backpropagation (BP) is the cornerstone algorithm for training artificial neural networks, yet its reliance on update-locked global error propagation limits biological plausibility and hardware efficiency. Predictive coding (PC), originally proposed as a model of the visual cortex, relies on local updates that allow parallel learning across layers. However, practical implementations face two key limitations: error signals must still propagate from the output to early layers through multiple inference-phase steps, and feedback decays exponentially during this process, leading to vanishing updates in early layers. These issues restrict the efficiency and scalability of PC, undermining its theoretical advantage in parallelization over BP. We propose direct Kolen–Pollack predictive coding (DKP-PC), which simultaneously addresses both feedback delay and exponential decay, yielding a more efficient and scalable variant of PC while preserving update locality. Leveraging the direct feedback alignment and direct Kolen–Pollack algorithms, DKP-PC introduces learnable feedback connections from the output layer to all hidden layers, establishing a direct pathway for error transmission. This yields an algorithm that reduces the theoretical error propagation time complexity from $\mathcal{O}(L)$, with $L$ being the network depth, to $\mathcal{O}(1)$, enabling parallel updates of the parameters. Moreover, empirical results demonstrate that DKP-PC achieves performance at least comparable to, and often exceeding, that of standard PC, while offering improved latency and computational performance. By enhancing both scalability and efficiency of PC, DKP-PC narrows the gap between biologically-plausible learning algorithms and BP, and unlocks the potential of local learning rules for hardware-efficient implementations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work merges successfully two important learning algorithms for neural networks, that are predictive coding and direct (Kolen–Pollack) feedback alignment. The resulting algorithm is faster and better than standard PC, but a little more memory heavy, due to the feedback tensors. However, this is marginal compared to the improved performance in terms of test accuracy and running time.

### Strengths
It tackles an important problem in the field of "alternative training algorithms", that is the efficiency of the simulations. The results are quite nice. For me, the most interesting 'reading' of this work, is that it proposes a much faster PC model that is able to reach performance that are as good as the original algorithm.

### Weaknesses
Despite of the fact that the results are nice, I have some concerns on how the evaluation was performed, in the scale, and on the fact that a larger study could be implemented. I believe that such concern do not allow the work to be "complete" yet, and it would be nice to implement them. The concerns are: 

1) You compare the results of your method against the numbers presented in Pinchetti et al. 2024. While that one is a good reference, you have not reported and compared against the test accuracies computed using centered or negative nudging, that are much better. 

1b) It would be nice to have a table reporting the performance of DKP-PC with centered and negative nudging. This is the method that in Pinchetti et al. has achieved the best results, why have you decided not to include it?

2) Similarly, why don't you implement an 'iPC' version of DKP? It should be straightforward to do so, and it also has the potential of working better. 

3) How many iterations of the PC algorithm do you run? I would be interested in looking at plots that show the energy decreasing from t=1 (so, after the backward maps have updated the errors) to t=T, where T is a large, fixed number of iterations. I would be curious in confronting that against the curves we have for PC and iPC. I would also be interested in looking at the changes in test accuracy over T, which you do not report. 

I see that you have figure 2, but it still leaves multiple questions open: why is the error so large? How nicely does it decrease over time? Does it force you to use very small learning rates?

4) Despite the nice efficiency results, the test accuracies are not as good as the ones obtained via centered nudging, or more recent implementations of PC that make it work on larger and deeper models (https://arxiv.org/abs/2506.23800). We could argue whether or not this is acceptable, given the advantage in terms of running time, and I believe it is: having faster models is a good contribution. However, I feel that the authors have not placed the right effort into pushing the limits of the model they are proposing.

### Questions
I would be curious in looking at the hyperparameters you have used to achieve the results (Such as learning rates for the x's, and the number of iterations T). However, in the supplementary material such details are omitted, and referred to the github page (which is not provided). I'd like to see a section explaining what worked and what didn't, which combinations you have tried -- so, the search space --, and the best performing ones.

More importantly: I believe the paper to be based on a nice idea, that seems to work fine. However, there are many more experiments that could be performed that would dramatically improve the quality of the paper, as well as the results (See the 4 points I made above for ideas). I'd like to see them implemented before proposing acceptance.

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
4

### Summary
The paper proposes DKP-PC, a predictive-coding variant that injects learnable direct feedback connections from the output layer to every hidden layer, ensuring that error signals are present everywhere at the first inference step, thereby addressing the typical PC issues of error delay and exponential decay. This makes PC effectively depth-independent in time, and on CIFAR/VGG benchmarks, DKP-PC consistently outperforms standard PC/iPC while staying below full backpropagation.

### Strengths
- The manuscript is clearly structured and accessible: it introduces the predictive-coding bottleneck, formulates the DKP-style remedy, and presents empirical results.
- Moreover, to the best of my knowledge, it constitutes the first explicit attempt to integrate feedback-alignment (DKP) mechanisms with predictive coding for the specific purpose of eliminating PC’s time–depth coupling.

### Weaknesses
1. **Limited novelty**
The core contribution can be viewed as a relatively simple composition of two well-established ideas—predictive coding for local inference and DKP/feedback alignment for direct error delivery—without introducing a clearly new theoretical principle or a substantially different training regime. As such, part of the contribution lies more in integration than in genuine algorithmic innovation.

2. **Performance limitations**
When contrasted with recent PC-acceleration efforts (e.g., https://openreview.net/pdf?id=s3E08R4AMK), the reported results do not consistently outperform the best published performance on comparable settings, which weakens the empirical claim that the proposed mechanism is the most effective way to mitigate error delay/decay. A short, head-to-head comparison on the same architecture/dataset/protocol would be needed to substantiate the performance advantage.

3. **Insufficient experimental diversity**
The evaluation is focused on a narrow set of convolutional backbones and CIFAR-scale datasets, making it challenging to assess whether the method scales to architectures with different connectivity patterns (e.g., ResNet) or to higher-resolution, more challenging datasets, such as Tiny-ImageNet. Broader model and dataset coverage would make the generality of the proposed approach more convincing.

### Questions
1. Could you clarify why DKP was chosen over more recent state-of-the-art feedback-alignment methods, and what concrete advantages it provides in the predictive-coding setting?

2. Is there a possibility for designing a novel feedback-alignment mechanism specifically tailored to predictive coding, so that it better addresses error delay and depth-dependent attenuation and potentially closes more of the gap to backpropagation?

3.  A clearer discussion of whether the joint use of PC and DFA/DKP remains biologically plausible would be helpful.

### Soundness
3

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
4

### Summary
The authors propose combining ideas from feedback alignment in order to overcome known issues with Predictive Coding.  They demonstrate empirically their proposed algorithm performs better than both PC and DKP in terms of accuracy over a range of datasets and models.

### Strengths
Using ideas from feedback alignment to solve the signal propagation issues with PC is a novel idea and addresses important problems for potential hardware implementations.
The authors report significant peformance benefits over traditional predictive coding and KP-alignment.
The clarity of the paper is very good.

### Weaknesses
The authors give valid justifications for introducing feedback connections to improve problems with PC. However, the authors do not justify (other than benchmark) why the single inference step would be improvement over pure feedback alignment. 

This is hard to reason about due to the (prima facie strange) choice of updating the weights twice per forward pass.  Indeed it is not clear what the inference step is minimising - at worst it may just act as a damping factor on the DKP update (since the layer wise errors are zero exactly when they reflect the new weights).

Some argument as to why PC inference steps improve DKP create a much stronger paper, talking directly to both communities.

Minimally, more space in the paper should be dedicated to justifying the use of inference steps. It would be great to see some empirical analysis to this effect such as:

Checking the alignment of the weight updates with the gradient update, with and without the inference step.
Ablations showing the effect of multiple inference steps and/or the learning rate used in inference.
Or a theoretical justification of why inference should help.  Without any of this we are  left to trust the single table of numbers - in which case a more detailed description of all hyperparameter and training details should be given. 

Secondly, if the authors truly believe their formal arguments extend Webster’s results is a contribution it should be included in the main paper rather than the appendix. Perhaps, there is the potential to explicitly use this analysis to demonstrate the weakness of pure DKP and justify the inference step as above?

### Questions
Mainly Question as above:

What do the authors believe the inference step is doing?
Specific Questions:

Did the authors consider defining extra error terms in the Predictive Coding energy of the form 
ϵ' _l =(ϕ_l−Ψ_l  y)^2
  and just doing typical PC updates on this (instead of the double weight update algorithm proposed)? If so what was the conclusion? — it seems when traditional errors are small this term would dominate providing early signal, which could then be corrected by PC dynamics.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a modification to predictive coding (PC) based on direct Kolen-Pollack (DKP) learning rules. The new rule, called DKP-PC, allows for fully parallelizable PC updates. Typically, PC suffers from slow propagation of error and requires many training steps to converge stably. DKP-PC works by using learned feedback weights directly from the output to each layer. After the forward pass, the weights are updated using the feedback pathway, then standard PC takes over. In practice, the feedback weights allow error to propagate immediately to all layers and reduce the number of PC convergence steps to O(1) (instead of O(L)).

### Strengths
1. The idea of bridging conventional feedback alignment rules and PC is novel and interesting.
2. The benefits, both theoretically and practically, are immense for enabling PC beyond niche cases.
3. Theory is correct and rigorous for the most part.

### Weaknesses
Limited discussion of the role feedback alignment in convergence. See Questions section for more details.

### Questions
Q1: It seems that feedback alignment allows for quick convergence and error correction, but PC corrects approximation errors that FA schemes cannot. Is this correct? If so, would you expand on why in the discussion?

Q2: In Eq. 37, I cannot follow the final step of the derivation. It is likely I am just missing which previous result to substitute in. Can you help clarify?

Q3: The assumptions in Appendix A.1 are too much to support the claim in Lines 170-172. I would soften this language.

Q4: Do the learning rates have to remain the same between the PC and DKP portions of the algorithm? I assume they don't, but some empirical work showing their effect on convergence in the context of Q1 would be interesting.

### Soundness
3

### Presentation
4

### Contribution
3
