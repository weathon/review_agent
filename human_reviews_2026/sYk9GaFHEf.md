# CERTIFIED VS. EMPIRICAL ADVERSARIAL ROBUSTNESS VIA HYBRID CONVOLUTIONS WITH ATTENTION STOCHASTICITY

- Decision: Accept (Poster)
- Scores: 8, 0, 4

## Abstract
We introduce Hybrid Convolutions with Attention Stochasticity (HyCAS), an adversarial defense that narrows the long-standing gap between provable robustness under ℓ2 certificates and empirical robustness against strong ℓ∞ attacks, while preserving strong generalization across diverse imaging benchmarks. HyCAS unifies deterministic and randomized principles by coupling 1-Lipschitz, spectrally normalized convolutions with two stochastic components—spectral normalized random-projection filters and a randomized attention-noise mechanism—to realize a randomized defense. Injecting smoothing randomness inside the architecture yields an overall ≤ 2-Lipschitz network with formal certificates. Extensive experiments on diverse imaging benchmarks—including CIFAR-10/100, ImageNet-1k, NIH Chest X-ray, HAM10000—show that HyCAS surpasses prior leading certified and empirical defenses, boosting certified accuracy by up to ≈ 7.3% (on NIH Chest X-ray) and empirical robustness by up to ≈ 3.1% (on HAM10000), without sacrificing clean accuracy. These results show that a randomized Lipschitz constrained architecture can simultaneously improve both certified ℓ2 and empirical ℓ∞ adversarial robustness, thereby supporting safer deployment of deep models in high-stakes applications.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces an architectural modification to standard CNNs that makes them more robust to (particularly $\ell_\infty$) adversarial attacks. HyCas has 3 components, each of which are 2-Lipschitz, and are fused via a convex operation, which allows them to obtain an $\ell_2$ robustness certificate. There are some works that also introduce 1-Lipschitz or spectrally normalized CNNs, but I find the internal stochasticity introduced in HyCas (via random projections, random attention noise), to be principled and interesting, and also standing out as novel in comparison to prior work. I also find the work to be refreshing from the perspective of tackling the adversarial robustness problem in a new way, as opposed to yet another method for solving the vanilla $\ell_2$/$\ell_\infty$ robustness problem. Experiments convincingly demonstrate the practical utility of the proposed method.

### Strengths
-- A new architecture, with several novel components, which are both well motivated and nicely executed

-- Bridges a key gap between certified and empirical robustness

-- Provide formal proofs that the network is 2-Lipschitz, which to my reading are convincing

-- Benchmark on both natural and medical image datasets

-- Comprehensive experiments in terms of number of defenses compared to

### Weaknesses
-- The three parallel streams plus convex gating presumably increase parameter count and compute cost. But the authors do not report FLOPs or latency.

-- There is no comparison to some new methods e.g. TRADES [1] or HR [2]. 

[1] https://arxiv.org/pdf/1901.08573
[2] https://arxiv.org/abs/2303.02251

### Questions
-- What is the parameter and FLOP overhead relative to a standard ConvNet block?

-- Are the certificates costly to obtain (i.e. via Monte-Carlo)?

-- Have the authors considered salience maps to visualize the sensitivity to input perturbations as opposed to standard l2 robustness methods. I think it would be interesting to see as an additional insight.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper explores a new adversarial defense. The main contribution of the paper is to narrow the gap between the work that has been done for certified defenses under the l2 norm and empirical defenses that use the l-inf norm. This is accomplished through the use of Hybrid Convolutions with Attention Stochasticity, a new method proposed in this paper. Experimentally the results of the new method are shown on a wide range of datasets including, CIFAR-10/100, ImageNet-1K and NIH Chest X-ray.

### Strengths
The paper has an interesting approach to adversarial robustness and the experimental results are comprehensive in terms of the datasets used.

### Weaknesses
=In the abstract the metrics mentioned don’t give any specific dataset. A 7.3% better robustness is achieved with respect to which dataset? 

=I would avoid the use of the word harmonize in the abstract. It is not clear what you mean when you say you harmonize provable and empirical adversarial robustness. Robustness is a measurement. If I said that I was going to harmonize kilometers and miles, would you understand what that meant? Obviously not. 

“the underlying randomness in these defences is static or easily inferred once seeds are fixed, rendering them vulnerable to adaptive attacks and offering no formal guarantees.”

=In the introduction this is a very bold claim and no citations are given to back up this statement. Why would seeds be fixed? Why is the underlying randomness considered static? This is not at all clear. I would need to see several citations in the literature to back up such arguments. Otherwise it is just conjecture. 

=I don’t understand why you use the term “natural” and “medical” imaging domains to describe the datasets. I know CIFAR-10, CIFAR-100, ImageNet type of datasets are images and so are chest x-rays. It sounds like you are just trying to make the empirical work more impressive when all you did is test on image datasets. You should really remove this terminology natural/medical from the paper.

=For the experiment for Figure 2, I am not convinced by the results. PGD is a very old attack. For credibility the authors at least need to update to APGD: https://arxiv.org/pdf/2003.01690

Also as far as I can tell, figure 2 is actually never referenced in the main body of the paper. Why do you have experimental results with no further explanation? 

=I think the terminology is confusing when you mention that your technique is a hybrid deterministic-stochastic defense. In theory if you add randomness to a deterministic defense, we would call that defense a randomized defense. In this case your paper is combining both deterministic and stochastic defenses, but this would then mean it is stochastic. E.g., any time a deterministic defense includes randomization, it is no longer random. I am concerned that mixing such terms as you do in your paper will really confuse readers. 

=Experimentally I don’t understand why so much space is wasted testing on PGD. In security we care about the strongest possible attacker. At this point PGD has widely been accepted as inferior to APGD. Therefore, there is NO reason to report PGD results in the main body of the paper. A huge  amount of space could be saved, simply by pushing all PGD results to the appendix. 

=I notice certain experiments are referenced in the main body but only shown in the appendix. E.g. Figure 7. I don’t think this follows the rules because it forces reviewers to look at appendix material when we should only be considering the main body. 

My overall opinion of the paper is that the experimental results offer a marginal improvement at best. However, the use of extremely convoluted writing techniques throughout the paper (natural/medical), “harmonize” and mixing up what is really a new stochastic defense that they call a nonsensical term “deterministic-stochastic”, all lead me to strong reject. I think the authors should resubmit only after major revisions are done to the writing and terminology of the paper.

### Questions
1. Why is the terminology so misused and convoluted? Can you fix the writing of the paper?  
2. Why are you referencing results that only appear in the appendix? 
3. Why did you not use APGD for all experiments instead of PGD?
4. How can you claim that your defense that has randomization is deterministic-stochastic?
5. Can you remove all PGD experiments from the paper and appendix and replace them with APGD?

### Soundness
2

### Presentation
1

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
The paper proposes HyCAS, a hybrid defense combining deterministic 1-Lipschitz convolutions with two stochastic components, namely, random-projection filters and randomized attention noise, to yield both high certified $l_2$ robustness and empirical $l_\infty$ robustness. The method claims state-of-the-art results on both natural and medical imaging benchmarks.

### Strengths
1) The method yields the state-of-the-art certified robustness, outperforming previous considered certified methods on CIFAR-10, ImageNet, and medical datasets (NIH-CXR, HAM10000), with up to $+7.3$ per cent gain at large radii.

2) By tuning the smoothing noise level, the robustness for large radii can be improved with minimal clean accuracy drop, namely, increasing $\sigma$ from $0.25$ to $0.50$ boosts the certified accuracy (specifically, from $8.5$ per cent to $12.5$ on CIFAR-10 at $r=2.0$).

### Weaknesses
1) The paper asserts HyCAS is the first method to offer both certified and empirical robustness, which is inaccurate: there were works incorporating randomization techniques and empirically robust modules, such as [1] and [2]. 

2) The main theoretical result ($\le2$-Lipschitz bound) is loose in comparison to the one in the baseline work [3]. Consequently, the robust radius is loose too; it raises the question how does HyCAS achieves higher certified robustness in comparison to RS (Table 1)? Does it happen purely because of a higher accuracy on clean, unperturbed data? If so, that limits the theoretical contribution. Overall contribution seems incremental. 

3) Noise resampling protocol is not clear: are attention masks resampled per image or per batch during inference? 

4) Computational overhead of additional modules is not reported (in terms of memory, inference time).




[1] Dong, M. and Xu, C.Adversarial robustness via random projection filters.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.  4077–4086, 2023.

[2] Yanxiang Ma, Minjing Dong, and Chang Xu. Adversarial robustness through random weight sampling. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[3] Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized
smoothing. In International Conference on Machine Learning (ICML), 2019.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
