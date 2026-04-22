# Towards Generalized Certified Robustness with Multi-Norm Training

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2, 4

## Abstract
Existing certified training methods can only train models to be robust against a certain perturbation type (e.g. $l_\infty$ or $l_2$). However, an $l_\infty$ certifiably robust model may not be certifiably robust against $l_2$ perturbation (and vice versa) and also has low robustness against other perturbations (e.g. geometric and patch transformation). By constructing a theoretical framework to analyze and mitigate the tradeoff, we propose the first multi-norm certified training framework \textbf{CURE}, consisting of several multi-norm certified training methods, to attain better \emph{union robustness} when training from scratch or fine-tuning a pre-trained certified model. Inspired by our theoretical findings, we devise bound alignment and connect natural training with certified training for better union robustness. Compared with SOTA-certified training, \textbf{CURE} improves union robustness to $32.0\%$ on MNIST, $25.8\%$ on CIFAR-10, and $10.6\%$ on TinyImagenet across different epsilon values. It leads to better generalization on a diverse set of challenging unseen geometric and patch perturbations to $6.8\%$ and $16.0\%$ on CIFAR-10. Overall, our contributions pave a path towards \textit{generalized certified robustness}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper extends the recent work on multi-norm adversarial robustness to multi-norm certified robustness. It proposes to utilize the structure of the certified training algorithm, SABR, which first conduct an adversarial search in the input set and then compute certified bounds with regard to a small $l_\infty$ box around the adversarial point. To this end, they simply extend the adversarial search from $l_\infty$ norm to a union of $l_{1,2,\infty} $norms, and keep the rest of SABR algorithm, i.e., $l_\infty$ box is still propagated regardless of the search norm. Regarding the multi-norm adversarial search, the ingredients mostly follow the prior work on multi-norm adversarial robustness, and the main novelty is the bound alignment regularization which is tailored to certified robustness. The results show the method improves the union certified robustness compared to algorithms considering a single norm.

### Strengths
The discussion about union certified robustness is novel and interesting. The methods keep a minimal but effective design, i.e., following the practice in the multi-norm adversarial robustness literature, with only changes tailored to certified robustness. The results are promising.

### Weaknesses
The theories seem largely inrelevant for the main algorithm. In particular, Lemma 3.1 is trivial and does not provide any insight. The last term in the equation of Theorem 3.2 is unclear in meaning, and the implications are rather trivial. As far as I am concerned, fully dropping the theorems from the main text does not lose any valuable information.

Some recent development from the certified training community is missing, e.g., [1-4] should be placed in the context of SOTA single norm algorithms. 

[1] https://arxiv.org/abs/2306.10426

[2] https://arxiv.org/abs/2206.14772

[3] https://arxiv.org/abs/2403.07095

[4] https://arxiv.org/abs/2406.04848

### Questions
Please address weaknesses as well as the following technical questions.

1. In SABR, the adversarial search set is clipped with regard to the size of the small box. Did the authors do the same clipping? How did they clip a $\ell_2$ set, for example?

2. The certification of $\ell_1$ norm is not mentioned. How was this done?

3. The results suggest that none of CURE-scratch, CURE-Random and CURE-Finetune universally outperforms others. How should one decide which to choose in practice? Does the superiority depend on the setting and how?

4. In CURE-Finetune, could the authors demonstrate whether using a better pretrained model improves the final performance? They may use the public pretrained SOTA models by [4] on $\ell_\infty$ norms. If the result is not the case, why?

5. Gradient projection is an interesting idea from multi-norm adversarial training. Why did the authors choose to not use this in $\epsilon$-annealing? How did this perform during annealing? Could the authors decompose the effect by demonstrating the benefit of GP during single-norm training with SABR, to confirm whether it works universally for certified training or just for multi-norm training?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies how to train models with certified adversarial robustness under multiple norms. The proposed CURE approach is based on minimizing the worst adversarial loss under multiple norms, and has two main components:
1. Bound alignment (BA): Train the model such that the difference between the upper and lower bounds of the certified radius have similar distributions under different norms
2. Integrating natural training into certified training with gradient projection (GP) 

The authors compared their method with some baselines on commonly used image datasets.

### Strengths
1. The authors include a detailed background in adversarial and certified robustness in section 2
2. The authors describe the motivation of their proposed method

### Weaknesses
This work has two main weaknesses:
1. Writing: This paper is very long — the authors use some latex techniques to put a lot more words. However, I think the focus of this manuscript is not very clear, and the discussion on the most important part is insufficient. I believe that the paper can be made much more concise yet more compelling.
2. Empirical results: I don't think the experiment section provides strong evidence that the proposed method works, and some key analysis is missing.

Let me elaborate on these two points.

## 1. Writing
The writing of this manuscript is not concise enough. The main novelties of this paper are bound alignment (BA) and gradient project (GP) that integrates natural training into certified training. However, they are not introduced until page 5, and although there are two full pages describing them, I still feel confused after reading it. So here are two constructive suggestions:
1. Make the first 4 pages more concise. Ideally you want to shrink them into 2 and a half pages. The background is good, but you probably don't want to use more than 1 page on that. Contents on page 4 can be significantly cut, and CURE joint, max and Random can be each described with 1-2 sentences. 
2. Make the description of BA and GP clearer and more concise. For example, BA can be summarized as: Get the upper and lower bounds of the certified radius from approaches such as IBP, and then train the model so that the difference in these bounds under different norms have more similar distributions. You can use one sentence to clearly state the crux of this method. And then, you can include some background on IBP, and you can move section 2.3 here. I don't think Lemma 3.1 and Theorem 3.2 are really useful. I cannot see how they help the ordinary audience understand this paper.

## 2. Empirical evidence
The empirical evidence is quite weak. Here are some comments:
1. From Table 1, it seems that the proposed method is not always better than the baselines
2. I don't think it is necessary to have so many baselines. For example, I think CURE-Random achieves the best performance most of the time, so you can just compare to that
3. The two main novelties of this paper are BA and GP, so I expect ablation study on each of them. So one missing experiment is only GP and no BA. Also these ablation studies (in Table 2b) are only conducted on CIFAR-10 but not on the other datasets.

Overall, I lean towards rejection, though I'm willing to change my rating depending on the rebuttal.

### Questions
1. Are joint, max and random something you invented? Or are they existing methods or baselines? If they are not something you invented, I don't think it is appropriate to put CURE before them, because CURE is the method you propose in this work.
2. From Table 1, I feel that Cure-Random is the best among the three most of the time. In that case, why do you base your method on max instead of random?
3. Why don't you use GP in CURE-finetune? Is it because it does not work? Can you include an ablations study on that?
4. A more open-ended question: If a model is $\ell_p$-robust, then it is also $\ell_q$-robust for $q < p$ due to the relationship between the norms, though the bound might be loose. For example, an $\ell_{\infty}$-robust model is also $\ell_2$-robust, though the robust radius might be small. So my question is, if a model is known to be $\ell_2$ and $\ell_{\infty}$-robust, can you prove bounds for its $\ell_p$-robustness, for $2 < p < \infty$?

### Soundness
2

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
The authors attempt to generalize robustness against not one norm, but several, with the aim of trying to produce stronger certificates.

### Strengths
This work is interesting, in terms of its willingness to address multiple norms in one context (although I hold questions about this as a problem space), it is broadly well written, well presented, and is supported by decent ablation studies and supplementary materials. There's definitely an argument for this being novel, although, as I discuss below, the question would be if this novelty is useful.

### Weaknesses
The authors state that they show " that multi-norm training offers broader certified robustness than single-norm or geometric-certified models" - superficially this seems fine, however conceptually I do not see this as being a well motivated statement, because there's not clarity about what broader robustness would actually mean. How would a user interpret this? How does this relate to the risk of adversarial manipulation? The authors intent appears to be that they're improving robustness against a broad range of adversarial motivations, but their approach is still limited, as I understand it, to just a subset of norms. To me, this is still quite narrow in terms of its appreciation of risk, in terms of not covering all the things that do not align well with the \ell_p distance model (things like rotations/translations/etc), yet a non-expert reader may not understand this. There's nuance here that I'm not sure is captured by the authors framing. And that's before we even get into the question of how much of an overlap there is already in terms of the coverage of different \ell_p norms.

I think more broadly, there's a need for a stronger argument regarding why multiple-norm defences are useful, beyond "such methods are typically tailored to specific p values, leaving networks vulnerable to other perturbations."

In a way, I do wonder if the authors are also implicitly acknowledging the geometric limitations of their approach through their choice of experiments. Across the experiments, evaluations were limited to  \ell_1, \ell_2, and \ell_inf perturbations. But if the idea is that there is risk covered in different norms, that is not appropriately covered by designing around a single norm, then there exists an infinite family of uncovered \ell_p norms that could still be evaluated against. If the argument is that \ell_1, \ell_2, and \ell_inf coverage is enough to provide strong geometric coverage of the input space, then one could counter by saying that just \ell_2 and \ell_inf could also be enough with slightly larger bounds, or, similarly, just one of these.   

When it comes to the mechanics, I must say I remain unconvinced by the argument about the utility of multi-norms, and the reliance upon the choices of epsilon.  There is inherently a coverage overlap between different norms, and increasing the coverage in a single norm may cover the regions of others - whereas you're attempting to essentially blend the norms of several different regions, but still, a larger norm bound of a single \ell_p could cover all of these. It seems like there's a lot of effort and moving parts to this that would be negated by simply increasing the epsilon for a given norm, or increasing the robustness to a single \ell_p norm offered by any technique.

By limiting themselves to only evaluating in the context of IBP style robustness, I think the authors have hamstrung their ability to demonstrate relevance. On L789 you state "They apply randomized smoothing, which is expensive to compute in nature, making it impractical for real-world applications." I would argue that this statement is wildly incorrect. Randomized smoothing costs time, and that time can be non-trivial, or, effectively trivial. Take for example a system that at inference time only has to certify one input at a time, but where the allowable batch size is on the order of the amount of samples required for certification. In this context, RS costs no additional time. Crucially though, RS can certify any model, because it only requires the model to take multiple passes - and so if the model is able to evaluate a single pass, RS will produce a solution. In contrast, IBP systems cost time AND resources. I don't believe I've ever seen an IBP system scale to Imagenet style datasets, and most IBP style evaluations work on models of the size of CNN7. This is because larger datasets and models bottleneck heavily under IBP. To put it another way: how would you describe the expense of computation of IBP style mechanisms vs RS style mechanisms as the size of the model increases?   

Fundamentally, the limitations of IBP inherently limit your experimental diversity. 

When it comes to the computational elements of your work, if I understand L298-319 correctly, you have the model p, you update both using NT and CT, you compare the updates of the two, then do a cosine similarity based operation to update the weights. But, if I'm reading this right, NT and CT are going to require two backprop paths, so you're doubling the computational complexity of the training? This may not double the overall cost, as I'm assuming IBP bounding is still the dominant factor within your costs, but still, there's a question about what uses cases would justify increasing training costs like this. 

While I noted in the strengths an appreciation of its conceptual novelty, I do wonder if the reason there are no other works on multi-norms is because they may not make logistical sense (see above). And outside of that, technical novelty does appear to be quite limited. One could look at CURE-Joint and just say you're doing a weighted blend of two other robustness losses, with some additional set dressing to support this. This is not to say that all papers must be technically novel, nor does it say that simple things can't require technical novelty, but I think in the absence of that it's all the more crucial to have a clear and consistent narrative about value and need, which, as discussed in some of the earlier points I've raised, does not seem to be fully formed to the level where I would expect as a reader. 

Finally, when it comes to the experimental details, for all the work involved there's a surprising lack of specificity at some points. For example, Table 13 and 14 labels aren't sufficiently detailed - is this runtime on a per-input basis? Training runtime? Runtime is such an ambiguous concept without detail. Similarly Figures 7-9 are presented in terms of "robustness", contain somewhat surprising results (see the L2 result in Figure 8), and often fail to explore interesting dynamics (for example, Figure 9's kick up for L2 in the -> 0 region - this is unexplored, and the parameter range cuts of the region where it appears interesting dynamics are occurring). 

Other smaller comments are as follows:
- Figure 1 is visually well presented, but on the details I find it absolutely uninterruptible and positively useless as a page 2 teaser image, given how early it is within the narrative.
- L156 "Generalized certified robustness (GCR). We measure" - who the we is in this sentence is ambiguous. L159 again uses "If we have " - the we framing is idiosyncratic for academic writing within a background section.
- L199-206 run through a theoretical scenario, but justifying an approach with an extreme position is, to me, meaningless. How likely is the occurrence of the position required to meet this in real data?
- L211-213 " To effectively combine the optimization of lq and lr (q = 2, r = ∞) certified training, based on the work (Tramer & Boneh, 2019; Madaan et al., 2021; Croce & Hein, 2022) on adversarial training for multiple norms, we propose the following methods" - the we propose the following methods part should be before the "based on the work..." - grammatically at the moment it implies that the optimization of certified training for multiple norms is already a feature of Tramer, Madaan, and Croce. 
- I found the distinctions between \ell_p, \ell_q, and \ell_r to not be clearly defined, or, more to the point, that I didn't quite understand the utility of how these were presented.
- Incredibly minor, but most papers I have read would treat the subject name for S4 as "Experiment[s]", rather than Experiment.
- Bibliography capitalisation (or lack thereof) doesn't reflect the titles. If nothing else, generally a title would be include "L2" or "\ell 2" when referring to robustness, not "l2"
- Figures 7-9 does not appear to be cited in text at all? And it's labeled just as "robustness" - what do you mean by robustness in this context? And the figure label has "L1" epsilons, while the table has "l1" - same goes for the colour labels at the top of the figure as well. 
- Figure 8 seems to imply that a model trained with L2 robustness provides 0 L2 robustness for varying epilons? This, surely, is not correct?
- Figure 9 is particularly interesting, as there's suddenly a kick up for L2 in the -> 0 region. Why is this occurring? Why did you choose not to increase the resolution of the exploration of the parameter range in this region?

### Questions
Given the inherent geometric overlap between \ell_p norms, what would the impact be of simply increasing epsilon for a given norm, and then testing against \ell_p'?

Could you explain the results of Figure 8, and your choice of evaluation range for Figure 9?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a multi-norm certified training method designed to improve union robustness across different perturbation types. The approach primarily builds upon IBP-based techniques and further explores robustness against practical transformations. While the idea is interesting and relevant, the paper still falls short of meeting the ICLR publication standard in its current form. I lean toward rejecting this paper.

### Strengths
1.	This paper is well-organized and clearly written.
2.	The study addresses a practical and relevant problem in certified robustness.

### Weaknesses
1.	The overall idea and methodology lack sufficient insight. Given that Lipschitz neural networks, such as LiResNet++ (Hu, 2024), have already scaled certified robustness to ImageNet and billion-parameter models, the proposed approach appears less competitive. Therefore, I would expect either a significant improvement in performance or a more novel conceptual contribution.
2.	The method includes too many components, making it difficult for readers to assess the contribution and effectiveness of each part. Based on the experimental results, some components may have limited impact.

### Questions
1.	The motivation behind Equation (4) is unclear. If the goal is to improve robustness to q- and r-norm perturbations, why not optimize $R_r + R_q$ directly? The $R_{align}$ (alignment loss in eq (4)) is not straightforward: the paper trains samples that are weak against r, and those strong against r but weak against q. Why not simply train samples that are weak to q and those weak to r? Based on this, please compare against a very simple baseline that jointly trains with l-2 and l-inf (i.e., “jointly train l-2 and l-inf”).
2.	Table 12 needs more detail. For example, the Hu’s model is a Lipchitz model and thus can report the certified robust accuracy directly after forward pass, whereas the proposed method requires an external certification procedure. Please clarify.
3.	For each norm, results are reported at only one perturbation budget. Providing results across multiple budgets would help readers assess performance trends.
4.	The chosen budgets are quite small. Please report robustness under larger budgets to evaluate the method’s behavior in more challenging regimes.
5.	Joint, Max, and Random show notably different performance. Please report variance (or confidence intervals) for all entries in Table 1.
6.	In Table 1, the l-inf-trained model does not outperform others under evaluation on L-inf and Tiny-ImageNet. Is this expected, or due to randomness? Please explain.
7.	CURE-Scratch sometimes performs worse than Max or Random, raising concerns about the effectiveness of bound alignment and GP. Please analyze and justify. 
8.	There is a typo in the equation on line 227. Table 2 is missing its caption.

### Soundness
2

### Presentation
2

### Contribution
1
