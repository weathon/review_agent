# Empirically Investigating the Trade-Offs in Deterministic Certified Training

- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
While there have been numerous advancements regarding the performance of deep neural networks on a broad range of supervised learning tasks, their adversarial robustness remains a major concern. To mitigate this, neural network verification aims to provide mathematically rigorous robustness guarantees at the cost of substantial computational requirements. Certified training}methods overcome this challenge by optimising for verifiable robustness during training, which, however, usually results in substantial decrease of performance on clean data. This robustness-accuracy trade-off has been extensively studied in the context of adversarial training but remains mostly unexplored for certified training. To control this trade-off, certified training techniques expose hyperparameters, which, to date, have been manually tuned to one specific configuration that compares favourable to the previous state-of-the-art. In this work, we present a novel fully-automated hyperparameter optimisation procedure for certified training that yields a Pareto front of optimal configurations with regard to the robustness-accuracy trade-off. Our approach facilitates the fair, principled and nuanced comparison of the performance of different methods. We show that most methods yield better trade-offs than previously assumed, thereby establishing a new state of the art in certified training of deep neural networks. In addition, we demonstrate that performance improvements reported over recent years are far less pronounced when all methods have been carefully tuned.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies the hyperparameter-tuning problem in certified training, which is of significance as certified training algorithms typically have much more hyperparameters than other training paradigms.

On the positive side, this paper proposes to use Bayesian optimization on the hyperparameter space instead of grid search as applied by previous works, which has been proven effective both by prior works in other domains and this work in the certified training domain. The method yields a Parento frontier rather than a single model tuned exclusively for maximizing the certified accuracy metric, thus is of potential interest to the community.

On the negative side, even with a much larger hyperparameter space, e.g., choice of optimizers, lr, decay epoch, random seeds, etc., the improvement over the grid search is marginal on SOTA. For example, on cifar 2/255, the certified accuracy was 64.54% compared to prior work which is 64.41%. The large improvement on TinyImagenet is particularly encouraging though, as 30.67% compared to 27.73%, which is of particular interest.

Overall, this paper brings the insights from autoML to certified training, including existing techniques and demonstrating similar advantages, which is interesting but might be of limited novelty.

### Strengths
This paper is clearly written and the idea of using Bayesian optimization to conduct hyperparameter search in a larger space more effectively makes sense.

The improvement on TinyImagenet is interesting, especially comparing against the minimal improvement in other settings.

### Weaknesses
The main limitation might be the novelty. AutoML, especially methods to autonomously tune hyperparameters with methods such as Bayesian optimization, is a textbook algorithm. There is no specific challenge in applying these methods to certified training, as have done by this work. As far as I am concerned, the only reason grounding this work could be an empirical illustration and evaluationg of a specific auto-tuning algorithm. If considered this way, then the contribution of this work should include more insights in applying auto-tuning, especially those novel in certified training compared to other domains.

Another major but fixable limitation is the experimental comparison. In Table 1, all comparisons with Xu et al and Zhang et al on CROWN-IBP should be removed. This is because these prior works use a very different model architecture, while the results in other works and this work uses the SOTA CNN7 architecture, which is identified independently. Further, the comparison to Mao et al on CROWN-IBP in CIFAR-10 is unfair: they use loss fusion universally, while this work did not apply loss fusion. As argued by Mao et al, this is due to practical concerns since CROWN-IBP without loss fusion does not scale to large number of classes, but not resulted from certain limitation of grid search tuning. This is supported by the vanishing improvement on CROWN-IBP in TinyImagenet, where this work applied loss fusion as well. Formatting Table 1 in its current form misleads readers to believe the relatively large gain in CIFAR comes from auto-tuning, while it is not. In addition, Table 1 should only compare against the result in Mao et al, but not the original works; as discussed by them, the original works under-tune the hyperparameter and have problematic implementations, thus comparing against those does not yield meaningful conclusions, especially with regard to the benefit of auto-tuning against grid search. When compares against Mao et al., it is worth noting that this work further tunes on random seeds, while they fix the random seed throughout CTBench (Appendix A). A comparison involving only a single random seed should be placed in the appendix to provide a fair comparison between auto-tuning and grid-search.

Some related work in certified training is missing: In Sec 3 part 1, [1,2,3] should be placed in the context of SOTA algorithms as well. Since certified training is of central study, recent works in certified training should be discussed: [4] prove that certified training does not allow single-neuron convex relaxations to yield exact bounds; [5] prove that certified training allows multi-neuron convex relaxations to exactly bound every piecewise linear continuous function; [6] empirically studies certified training in empirical robustness. This might not be a complete list.

In addition, using Bayesian optimization to auto-tune hyperparameters faces a major obstacle in parallism. While grid-search can be effectively executed in parallel and thus does not incur additional wall-clock time overhead, Bayesian optimization, despite possibly reducing the total number of trials, is bounded to execute sequentially and thus leads to larger waiting time.

Finally, the authors should remark clearly on the implications for the field. The community is interested in pursuing new frontiers, as witnessed by new SOTA. Independently, if a good set of hyperparameters can be identified, then it saves efforts in future works in hyperparameter tuning. Therefore, the author should clearly present their insights in the resulting hyperparameters, especially on how they could help the community without running 300 trials per benchmark as in this work. Further, it is **very meaningful** to discuss why SOTA (MTL-IBP) in TinyImagenet witnessed particularly large gains.

Minor comments: The authors discussed grid-search as "labour-intensive", while it is only computationally expensive. A grid search hardly requires expertise as well. The argument in excluding TAPS is problematic: as shown by CTBench, TAPS running time is usually comparable, up to 2x cost with regard to other methods, and it exceeds MTL-IBP on MNIST although it is not highlighted by this work in the main text. In addition, some hyperparameters are discrete, e.g., the choice of optimizers; the authors should clearly discuss how these discrete variables are handled by the Gaussian process.

[1] https://arxiv.org/abs/2305.04574

[2] https://arxiv.org/abs/2206.14772

[3] https://arxiv.org/abs/2403.07095

[4] https://arxiv.org/abs/2311.04015

[5] https://arxiv.org/abs/2410.06816

[6] https://arxiv.org/abs/2410.01617

### Questions
Please address questions raised in weaknesses.

Further, why does Table 1 has different numbers for the comparison against Mao et al.? What is changed in the numbers presented in this work? My guess is that the authors picked different points on the Parento frontier, is it correct?

### Soundness
2

### Presentation
3

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
The authors present a multi-objective hyperparameter optimization method for certified training methods. Their method generates a Pareto frontier of hyperparameters that address the accuracy/robustness trade-off for certified training methods. They show empirically that some certified training methods can in fact be improved via better hyperparameter tuning, and that different methods have complementary strengths when it comes to optimizing natural versus adversarial performance metrics.

### Strengths
-- The paper is well-written and mathematically sound in its descriptions

-- I find the paper easy to follow and well-structured

-- The paper makes empirical contributions that are useful for those in the field of certified training

-- Appreciate that the code is made openly available

### Weaknesses
-- My main issue with the paper is a lack of methodological contribution. Bayesian optimization is an out-of-the-box hyperparameter optimization method. The authors apply it to optimize hyperparameters in certified training, but that itself is not a methodological contribution in my opinion

-- Hyperparameter tuning *is* an issue in certified training, but I would prefer to see lighter-weight hyperparameter selection methods compared to doing a smarter grid search. For example, how might one determine the optimal the choice of $\epsilon$ a-priori, to guarantee a certain robustness level, while having good natural accuracy?

### Questions
-- I would request that the authors please clarify the novelty of the approach. I would like to see this novelty spelled out more clearly. 

-- Does the Bayesian optimization use any structure from the certified training problem? Or is it the generic method of Bayesian optimization (it seems from my reading it is).

### Soundness
3

### Presentation
3

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
This paper addresses the core challenge of deterministic certified training—balancing verifiable robustness with high clean accuracy. The authors propose an automated multi-objective hyperparameter optimization framework that simultaneously optimizes clean and certified accuracy to generate a full Pareto front. The framework leverages multi-objective Bayesian optimization (EHVI) for efficient hyperparameter search, uses IBP/CROWN-IBP as low-cost proxies, applies constraint optimization to avoid degenerate solutions, and refines results via single-link clustering and αβ-CROWN verification. Experiments on CIFAR-10 and Tiny ImageNet show that the method achieves a better robustness–accuracy trade-off than existing approaches (e.g., MTL-IBP, SABR, CROWN-IBP), offering a more comprehensive view of certified training performance.

### Strengths
1.The paper is well-format and easy to read.
2.The code has been anonymously released as open source.

### Weaknesses
1.This paper applies hyperparameter optimization to deterministic certified training. However, it contributes little technically to either hyperparameter optimization or deterministic certified training itself.
2.Regarding deterministic certified training, it is important to note that discussions about AI should first focus on usability (i.e., model performance), followed by robustness, fairness, and safety. If an AI model is not practically usable, then the rest of the discussion becomes meaningless.
3.The major issue with deterministic certified training is the significant drop in accuracy, which makes the trained models impractical for real-world use. The experiments in this paper also confirm this limitation: although the proposed approach achieves some improvement over baselines, the accuracy still drops to around 50+% on CIFAR-10 (perturbation radius 2/255) and 20+% on ImageNet (perturbation radius 1/255). Moreover, prior theoretical work has already established an upper bound for deterministic certified training, e.g., for CIFAR-10, this bound is 67.49%.
4.In terms of structure, Chapters 2 and 3 occupy disproportionately large portions of the paper and could be simpler, or even merged into a single section for conciseness.
5.In the experiments, in addition to reporting certified accuracy, it is recommended to include empirical adversarial attack success rates. Furthermore, for model training, runtime efficiency is also critical metrics, and corresponding experiments should be added.

### Questions
My main concern is that deterministic certified training has been shown to significantly degrade accuracy, making the resulting models impractical for real-world use. Therefore, I do not think I will change my opinion.

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
4

### Summary
The authors merge the concepts of robustness (via interval bound propagation) and hyperparameter optimization, in an attempt to improve performance of IBP style methods.

### Strengths
On the whole, this was a well written (there'll be a few notes on this below) paper that works in a space that is relevant. The authors align with community expectations with regards to testing datasets (typically now I'd prefer to see Imagenet rather than Tiny Imagenet, however I appreciate that this isn't viable in the IBP side of the certified robustness community). 

Fundamentally too, the idea of articulating pareto fronts (relative to single points in hyperparameter space) and fundamentally there is a need for, as the authors say, "fair, principled and nuanced comparison of the performance of different methods. We show that most methods yield better trade-offs".

### Weaknesses
Unfortunately, while it's well written, it's also relatively difficult to see impact here. The authors have merged two known techniques - hyperparameter optimization and IBP - and have measured some results. This was clearly computationally challenging, in terms of the resources required, but as far as I can see this is a paper that mainly says "parameters could be slightly improved". But doing this requires huge investments of computational resources, and is still, fundamentally, while also being inherently tied to the very same small architectures that inherently hamstring IBP style techniques. 

Fundamentally, the idea of hyperparameter optimisation of IBP style techniques feels like rearranging deckchairs on the Titanic - yielding small improvements for something that feels a tiny bit doomed. This is a bit of a harsh perspective on IBP style techniques, but their GPU and architectural limitations are significant, and performing Prateo optimisation on something that requires significant resource investments only makes the problem worse. I do not see how there would lead to any real world take away from this. Either 1) someone is following the exact same experimental setup, in which case the updated hyperparameters only form as an updated point of reference, but with limited other utility or 2) someone is using a different experimental set up, in which case they could independently use hyperparameter optimization without any need to reference the authors work. 

Results were also not contexutalised with uncertainties or error bounds, which are almost certainly non-trivial given the sample size of 300. 

The following comments are more minor, and primarily relate to writing quality. 
L12 - robustness not defined, assumed knowledge
L15 - "substantial computation requirements" - why is this true? Yes, to someone in the field this is clear, but the abstract is making strong assumptions about who the reader is. 
L15 - but don't certified training ethods also require significant computational investment? 
L38 - "map generation for autonomous driving" - how is this a safety critical task? Where are the risks? 
L39: Adversarial example sentence is a non sequitur - as presented it disguises the impact of these on safety critical problems. 
L42 How do these "play an important role in diagnosing weaknesses" - finding an adversarial example is no guarantee that no smaller / better AE could be found by an adversary? 
L46-48: Are these network wide safty properties, or sample wise properties? 
L50: Why is L_inf popular? 
L62: SOTA is not IBP. Randomised smoothing has, arguably, surpassed it, as IBP imposes significant GPU ram requirements that limit its use to small networks (and also prevents the use of some architectures), whereas RS only incurs a time penalty, with almost 0 memory penalty, meaning that it can be applied to a broader range of network architectures. 
L71: "Moreover, these methods require tuning additional hyperparameters, such as the learning rate and the number of warm-up epochs, which strongly influence training stability and final performance" - how is this any different to any neural network? What is different here besides the application of fine tuning to a new domain? 
L144: The length of this sentence causes problems. Madry's specific contribution was around the worst case loss in the l_inf norm ball, but the way this is phrased that part is 3 lines deep, and quite disconnected from the context that originally establishes that MMadry is the first to introduce...
Table 1 isn't formatted in a way that allows for easy comparisons, in that it goes clean acc, cert acc, clean acc, cert acc. 
Figure 1 is apparently a comparison between your work and Mao's, but it's not clear which is yours and which is Mao's in the majority of f  figures. , nor how things should be compared. 
Figure 2 "when prioritising natural accuracy" is a very....loose statement. One could prioritise natural accuracy but still care about certified accuracy, in which case they may choose MTL, or possibly even SABR, depending upon their threshold for natural accuracy?
Figures 3 and 4 are effectively unreadable. This style of presentation is odd, to say the least. And some words are cut off. 
Certified is a descriptor of a technique generally, and should be treated as a proper noun - including in the bibliography. 
Adam is ADAM - bibliography. 
Batch Normalization is a proper noun - bibliography. 
Deep Neural Networks and Random Forests - see above.

### Questions
How would this scale to non l_inf boundings? 

What is the value of hyperparameter optimisation in a system which is incredibly expensive from a computational perspective? What scenarios would allow this to occur?

How would performance scale with increased epsilon? 

Are the values in table 1 the values reported by these papers, or were they the results you constructed using the approaches from prior works? 

The experimental setup section doesn't make sense to me - why does C10 have different resources avaiable to it than TI/MNIST, and what does line 930 mean relative to lines 923-929. If I'm reading L930 correctly, you used one GPU (either a NVL or SXM H100 of unspecified GPU ram), 120gb of ram, and 24 CPU cores - if so, then what exactly was the point of lines 923-929, besides mentioning what resources you have available? Also why were these details only in the appendices?

### Soundness
3

### Presentation
4

### Contribution
2
