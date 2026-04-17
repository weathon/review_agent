# GradPower: Powering Gradients for Faster Language Model Pre-Training

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
We propose **GradPower**, a lightweight gradient-transformation technique for accelerating language model pre-training. Given a gradient vector $\boldsymbol{g}=(g\_{i})\_{i}$, GradPower first applies the elementwise `sign-power` transformation: $ \varphi_p(\boldsymbol{g}) = \left({\rm sign}(g\_i)|g\_i|^p\right)\_{i} $ for a fixed $p>0$, and then feeds the transformed gradient  into a base optimizer. Notably, GradPower requires only a **single-line code change** and no modifications to the base optimizer’s internal logic, including the hyperparameters. 
When applied to Adam (termed **AdamPower**), GradPower consistently achieves lower terminal loss across diverse architectures (LLaMA, Qwen2MoE), parameter scales (66M to 2B), datasets (C4, OpenWebText), and learning-rate schedules (cosine, warmup-stable-decay). 
The most pronounced gains are observed when training modern mixture-of-experts models with warmup-stable-decay schedules. GradPower also  integrates seamlessly with other state-of-the-art optimizers, such as Muon, yielding further improvements. Finally, we provide  theoretical analyses that reveal the underlying mechanism of GradPower and highlights the influence  of gradient noise.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduce the SignPower update, a fairly simple (in a good way) extension to Adam, Muon, etc. that seems to show reasonable, if modest, gains, with occasional outsized gains. The basic idea is that you can raise the gradients to a power (preserving sign) which will, when the power is > 1, accentuate larger gradients and diminish smaller ones. They do experiments with small-ish LLMs and a vision task.

The paper is generally well written. The main experiments are fairly thorough (modulo my main concern below). I'm not much of a theory person, but the proofs seem reasonable if pedestrian. I do worry a major confound is at the root of most of the gains, however. 

SignPower introduces two effects that I think need to be distinguished. Because we're clipping gradients, we know |g| < 1 (usually << 1) and so |g|^1.2 will damp gradients. So on the one hand, it decreases all updates ("global damping"); on the other it decreases smaller gradients more than larger gradients ("heavier tails"). The former amounts to an expected dampening of about 6% relative to vanilla Adam.[1] 

These effects need to be disentangled.

Footnote:

[1] Assuming ~normally distributed gradients, the adam update rule will be basically Adam(p) = E|g|^p/sqrt(E[|g|^2p]) (let's ignore beta etc) 

E[|g|^p] ∝ 2^{p/2} Gamma((p + 1)/2), ignoring constants that cancel. Plugging in we get Adam(1.2)/Adam(1) ≈ 0.94.

Same basic trends hold for the authors' preferred choice of Unif[mu-sigma, mu+sigma] gradients I think, with different numerics. (In my experience, gradients look more gaussian though!)

Disclaimer: I used chatgpt to write the code to compute the various ratios of E[|g|^p] and to check my math.

### Strengths
(See also my comments above)

The paper is generally well-written modulo a few typos/grammatical errors that probably come from a deadline rush.

The idea is a relatively small change to Adam (that is part of its appeal) and seems to work reasonably well. They have some theory and experiments to back it up. They even show that it transfers to Muon. 

They are clearly not overtuning their method relative to the baseline, which is great practice. They have a few different model classes and two domains.

I could imagine a world where we should be tuning our grad-powers and not our LR (and maybe doing p-schedules instead of LR-schedules). I suspect we aren't in that world, but it would be neat and if we were then this line of work would be very impactful!

### Weaknesses
(See also my comments above)

As I said above, I do worry a major confound is at the root of most of the gains. I also worry that, as is usually the case with optimizer papers these days, the scales aren't high enough to be trustworthy, that the gains aren't significant enough that a big lab is going to pick it up (though, maybe due to how easy it is, they will!), and that if they do, it won't scale after proper tuning. This alone shouldn't be taken to mean the paper shouldn't be published: it's the role of academia in this environment to find good ideas that seem to show some promise, and rely on big labs to show that "it matters."

More specifically:

For their main results, the authors do a coarse LR sweep that is structurally quite fair to the baseline (admirably so!). But, given the global damping behavior of the update rule, it seems like one also needs to run each experiment taking into account this effect, especially since the gains are usually relatively small. 

For instance, could the authors try running adampower with lr = 1/.94 * adam lr? This would help distinguish the "global damping effect" from the "heavier tails effect." The observation on [254] that adampower makes runs less spiky makes me suspect that the damping is a large part of what's going on. ("Effective but spiky LR" in my experience means lr is slightly too high.) Note that 6% is much less than the resolution of the LR sweep performed. 

Section 3.5:

This section also increases my suspicion: Given the implicit LR damping effect,and since the LR was  held constant across all scales other than the smallest, the optimal power would decrease as batch size increased: you want a somewhat higher LR with higher batch size (probably around sqrt(2) per power of 2; see e.g. https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/). Regardless, the differences in figure 6 seem like they could easily be within noise except for the bs=512 with low power and the bs=8192 with high power.

Again, I applaud the authors for doing an LR search for the baseline, but the search domain is fairly coarse. I'd expect the optimal batch size for powers of two batch sizes to differ by sqrt(2) per scale, which is less than the resolution of your search. This may explain why you found similar optimal LRs for those different sizes.

Section 4.1: Same basic deal. Isn't this really an argument for **varying p** (in the same way we vary the LR): following the river-valley analogy, when the SNR is high (on hill), we want lower p (and similarly, high LR), and when it's low, we want higher p (and similarly, low LR). But you aren't varying p within a run! Invoking the river-valley analogy to say that different SNR-regimes require different p's and then using fixed p for all actual runs seems wrong?

### Questions
I'd like to see experiments that disentangle the implicit LR damping effect vs the accentuation effect. if you could rerun gradpower with LR = 1/0.94 the adam LR and give me a scaling law (or maybe better, rerun the scaling suite with adam with 0.94* LR) I'd be more convinced. I'd still guess this wouldn't prove itself in the long run, but I'd easily from from a 4 to an 6.

Am I wrong that your experiments in section 4 aren't really supporting your choice of a specific p (but rather support a variable p)?

Minor points:

While gradient clipping is mentioned in passing, it's possibly worth noting the interaction with your method.

045: elemenwise -> elementwise

102: That a linear transform doesn't do anything to adaptive optimizers seems fairly trivial and doesn't merit a half page proof, even in the appendix.

113: The highlight of "one additional line of code" is a bit much

345 : Sentence isn't grammatical

360: noise-dominate -> noise-dominant

Figure 7 description isn't quite grammatical? delete the 'and'?

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
2

### Summary
This paper proposes GradPower, a simple gradient transformation that applies an element-wise sign-power function to gradients before passing them to any optimizer. Applied to Adam, it claims consistent improvements across LLM pre-training settings, including dense and MoE models, multiple datasets, and learning-rate schedules. The paper also provides theoretical insights into why different powers work in low and high-noise settings.

### Strengths
Strength:

* The proposed method is simple and practical in implementation, and can be combined with many advanced optimizer.
* Experiments across multiple LLM scales, datasets, and schedulers demonstrate the effectiveness of the method in accuracy and convergence speed.

### Weaknesses
Weaknesses and Question:

* The model scale is limited, the largest one used is 2B. As its mainly focus on pre-training process, it would be better with at least a 7B or 13B baseline to show real scalability, many performance instability issues during training tend to emerge in models larger than 7B.
* Though $p=1.2$ is effective under most settings, but its unclear how sensitive results are to $p$ across broader tasks, or can you provide some results or discussion on adaptive or layer-wise $p$ adjustment?
* The performance gain is borderline, for example in zero-shot, avg accuracy increased 0.5%, and most of increasement comes from one single dataset WINOGRANDE.
* Results limited and small-scale. Need more tasks on standard LLM evals, like including perplexity results or evaluate on benchmarks like MMLU or MT-Bench, which is closer to actual application scenarios.
* Some prior work also tries power-gradient methods, it would be great to include more discussion or quantitative comparison.
* Beside local training loss curve, could you provide more details related to convergence speed? Like GPU hour comparison.

### Questions
Please refer to the Weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper focus on accelerating large model training by developing a new optimization algorithm. Specifically, the paper studies nonlinear operation on the gradient and proposes signed-power gradient method that powers the magnitude of the gradient for each of its coordinate and keeps its sign. The paper conducts numerical experiments on multiple settings for training large language models.

### Strengths
1. Simplicity of the proposed method. The sign power method is light-weighted and is compatible with multiple base optimizers.
2. In the numerical results, the proposed method achieves consistent better performance than thier baselines, and are potentially more stable.
3. The paper provides additional theoretical insights on how the sign power method accelerates the base optimizer (Adam in the 1-d case and Adagrad in high-dimension case)

### Weaknesses
1. On the numerical experiment:
    1. Missing reporting the variance of the results. I would expected at least one result reporting the variance of the algorithms with multiple runs (>=5) to show the statistical significance. Given the difference in the loss is only ~1%, it is hard to judge if such difference is comming from the selected seed or due to the proposed algorithm. I understand that for LLM pretraining, multiple run can be time consuming, but it is expected to have at least one set experiment to report the variance.
    2. From the convergence lines, the models are still converging very fast, far before convergence. It is hard to tell how the final model performance look like when the models actually converge. 

2. On the hyper-parameter choice
    1. It is hard to see why p=1.2 is a uniformly optimal solution for training large models, given that in Table 2, p=1.2 does not achieve the optimal solution for any of the batch size on Vision model. And is only optimal for batch size 512 and 2048 when training LLaMa 0.2B on C4 (Figure 6). This might suggest that for difference batch size choice (, tasks and models), we need to pick a different p value.

3. On the theoretical result
    1. The restricted assumption: Assumptions 4.6 and 4.9 are quite restrictve, as they both asume that each coordinate of the stochastic gradient should align with the true gradient's direction. This assumption is even stronger than assuming the overall stochastic gradient is aligned with the true gradient. The author should show that under these assumptions, the normal Adagrad is still converging slower than AdagradPower. 
    2. In examples 4.7 and 4.10. It is hard to see how 0.99 and 1.01 are derived.
    3. It is inaccurate to say after Theorem 4.8 that the algorithm is 2 times acceleration. It is the rate changed from T^-1/2 to T^-1/(p+1).

### Questions
1. Does the theory suggests an optimal choice of p under differen regimes?
2, What is the stochastic gradient is not uniformly bounded, e.g., for a quadratic problem?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper propose GradPower, a new gradient-transformetion technique that can orthogonally combined with AdamW and Muon to reach faster convergence.

### Strengths
The topic of boosting optimizer performance is important. The writing is mostly clear. The experiments are sound.

### Weaknesses
The motivation is unclear, as I elaborate as follows.

### Questions
**Major concern:** I think the motivation of the proposed GradPower operator is quite unclear. 

If I understand correctly, the motivation is to "line 68: amplifies these (flat) directions". I do not understand how to achieve this goal with GradPower.  In optimization theory, flat direction usually refers to the eigenvectors of the Hessian associated with small eigenvalues.  "Amplify these (flat) directions" requires first finding the targeted Hessian eigenvectors, and then amplifying the update along them. 

In principle, in order to  "amplify these (flat) directions" or "take larger steps in the flat directions", one needs the following procedure:

Step 1. Rotate the gradient vector into a coordinate system under the eigenbasis of the Hessian, in which the Hessian will become diagonal and the eigenvectors become the standard basis.

Step 2.  Amplify the update on the small-diagonal component in Hessian. This equivalently amplifies the update along the Hessian eigenvector.

Step 3. Rotate back to the original coordinate system and update the optimization variable.

Now, how does GradPower operator ensure all these？GradPower simply takes the sign and power over the gradient component.   This indeed amplifies the small gradient components, but this does not mean we "take larger steps in the flat directions".  I do not know how GradPower helps achieve this without the rotation in Steps 1 and 3. Why is such a design reasonable?  Is there any theory supporting that we can skip the rotation steps? Is there any implicit approximation of the rotation? The current theory in Section 4.1 does not help address this concern. 

With that being said, it is possible that I misunderstood the meaning of  "amplifies these (flat) directions". Please explain what it means if this is the case.

### Soundness
3

### Presentation
3

### Contribution
2
