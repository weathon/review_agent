# Best-of-N through the Smoothing Lens: KL Divergence and Regret Analysis

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
A simple yet effective method for inference-time alignment of generative models is Best-of-$N$ (BoN), where $N$ outcomes are sampled from a reference policy, evaluated using a proxy-reward model, and the highest-scoring one is selected.  
While prior work argues that BoN is almost optimal in reward vs KL tradeoffs, the effectiveness of BoN depends critically on the quality of the proxy-reward model used for selection. For this purpose, we study BoN through a smooth version known as Soft Best-of-N (SBoN) and develop a theoretical framework to address this gap. We analyze the scaling behaviour of BoN by providing bounds on the KL divergence between the SBoN policy and the reference policy, offering insights into how performance varies with the number of samples. We also study the regret gap, i.e., the gap between the expected true-reward under the optimal policy and the SBoN policy. Our theoretical and empirical findings show that smoothing helps SBoN mitigate reward overoptimization, especially when the quality of the proxy-reward is low.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies Soft Best-of-N (SBoN), a smoothed variant of Best-of-N (BoN), from a theoretical perspective. 

It proves (i) an explicit upper bound on the KL divergence between the SBoN policy and the reference policy (Lemma 4.1); (ii) an upper bound on the KL divergence between SBoN policies induced by true vs. proxy rewards as a function of (among others) the tilted error between proxy and true rewards (Lemma 4.2); (iii) an upper bound on true reward under proxy-reward SBoN (Theorem 4.3); and (iv) optimality-gap (regret) bounds for SBoN and, as a limit, BoN (Theorems 5.2, 5.3). 

Empirically, SBoN mitigates (but does not eliminate) over-optimization when the proxy is weak (Fig. 1). Overall, the paper formally shows that smoothing can be a principled mitigation against reward over-optimization.

### Strengths
S1: The study of inference-time alignment is timely and rigorous, and the presented results target practically relevant and meaningful quantities. Best-of-N in commonly used in practice, so analyzing a smoothed version that could be safer against overoptimization is valuable.  

S2: The mathematical setup is more convincing than previous related work because of the explicit reward calibration. I also appreciate that the authors considered the “tilted error” viewpoint since it gives a single knob that moves from average-case to worst-case sensitivity. It is a helpful way to reason about when smoothing should help, and I believe this setup is a great baseline for other papers in this field.

### Weaknesses
W1: The presentation can be greatly improved. In particular, I encourage the authors to motivate their setup. For example, they mention calibration in Line 49 without any context. The usefulness of calibration is not obvious until much later in Line 229. I also encourage them to be explicit in how calibration is implemented. 

W2: While the theoretical results hold for BoN as well, they seem to be much looser than if BoN had been studied directly as in previous work. I understand why the bounds are loose because of the challenges analyzing the complex SBoN policy in (3). Hence, I believe readers would benefit from a clearer motivation for studying Soft BoN. This is especially true given that SBoN is a new method that is not deployed in practice.

W3:  I am not sure of the takeaway of this paper. This is due to two issues: (1) the obvious one is the lack of a principled way to choose the right temperature, as well as the limited experimental results, and (2) more importantly, I cannot make sense of what the results tell us. I would appreciate if the authors look at simple cases: how would smoothing help when the reward model misranks near the top only, for example?

Minor:
1. There are a few typos (missing spaces, spelling mistakes) scattered in the paper (lines 36, 46, 110...). 
2. I would encourage moving the preview of the results in Tables 1 and 2 to the main text. I also liked Figure 2 so I can make sense of the different terms. I feel it is challenging to keep up with the notation in the main text. I would suggest looking at a different notation to differentiate the policies from each other.
3. At Line 35, fix the attribution of Best-of-N. It was popularized for LLMs by Stiennon et al.
4. Line 50 is confusing, what do you mean by "using some dataset"?

### Questions
Q1: Line 60: What do you mean by "scales proportionally" in line 60?

Q2: Line 273: what do you mean by KL regularized objective for SBoN? SBoN doesn't optimize that directly.

Q3: How useful are the upper bounds? How do you foresee other researchers operationalizing them to find or evaluate other strategies against reward hacking? I appreciate the toy example in Figure 4 for your first lemma and would suggest the same for your other results. Also, how would the upper bounds look like in your empirical experiment in Figure 1?

Q4: I understand finding lower bounds must be challenging, but are there settings where we could have a meaningful lower bound? 

I am convinced by the authors' results. However, I believe the paper would benefit from a stronger motivation, as well as insights on the consequences of their theoretical results.

### Soundness
4

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
This paper provides a theoretical analysis of the soft Best-of-N (SBoN) method, which smooths the output probabilities, a technique commonly used in various applications. The focus is on the calibrated proxy reward rather than the (raw) true reward, with respect to KL-divergence and regret. The authors show that in certain conditions, SBoN achieves a tighter bound on the regret gap than standard Best-of-N (BoN), while BoN attains a tighter regret upper bound when over-optimization is absent. Since the smoothing usually introduces algorithmic conservativeness, these results are consistent with general intuition.

### Strengths
* Provides finite-sample bounds on the KL divergence between SBoN and the reference policy under the true reward model, as well as between SBoN policies under the true and proxy reward models. These results generalize across different values of $\beta$ (the temperature parameter for the Shannon entropy penalty used to smooth probabilities).

* Considers a calibrated proxy-reward model, which is more realistic and relevant for practical applications.

* I carefully read the paper, including the appendices, and checked the proofs (except Appendix G). Most of the derivations appear correct, though a few technical points remain unclear (see the Questions section).

### Weaknesses
* The paper has overly complex notation, which significantly reduces readability and makes it difficult to follow the theoretical flow. Some notation should be refined or consistent for clarity.

* The main theoretical results are difficult to interpret (see the Questions section for details).

### Questions
1. In Section 3.1 the authors consider the finite set of prompts $X$. For responses this is plausible, since an LLM outputs finite-length token sequences from a finite vocabulary. For prompts, however, $X$ is provided by the human and is not naturally finite. Is finiteness of $X$ actually required for the analysis (as far as I checked, it does not seem necessary), or only for notational convenience? If it is required, can you clarify why?

2. The temperature $\beta$ is used for three different roles: (i) tilted optimal policy for the KL-regularizer, (ii) the entropy temperature in SBoN (the softmax smoothing), and (iii) the parameter in the tilted error term. Are these all meant to be the same $\beta$? If not, it would be clearer to separate them, e.g., ($\beta_{\text{KL}}, \beta_{\text{SBoN}}, \beta_{\text{err}} $).

3. What is the intended meaning of Theorem 4.3? Although the authors state that it can be viewed as an improvement relative to the reference policy, it is unclear what insight is gained from providing an **upper bound** on the expected calibrated reward. The bound can exceed $1$ in some cases, which is trivial since the calibrated reward itself is defined in $[0,1]$. While the bound is always larger than $0.5$ (the reward of the reference policy), this alone does not make the result informative. Furthermore, the authors mention that they aim  to maximize the first term and minimize the second term, but this makes the overall objective of Theorem 4.3 difficult to interpret. Although the intuition behind maximizing or minimizing each component is reasonable, the combined purpose of these directions and the motivation for presenting Theorem 4.3 remain unclear. Could the authors clarify the intended rationale for considering this theorem?

4. The regret upper bounds are more informative, as the authors provide a clear comparison between SBoN and BoN. One straightforward question: would it be possible to derive a lower bound on the optimality gap when $N$ is finite and $\beta$ is fixed? If so, such a result would offer a clearer reference point for evaluating the tightness and significance of the obtained upper bounds.

5. Could you explain why the first inequality in the proof of Lemma E.11 holds? Does the derivation implicitly assume that the reward function $r$ takes discrete values? If $r_c$ follows a uniform distribution on $[0,1]$ as the true calibrated reward function under the reference policy, then the event $\\{ R= r \\}$ has zero probability. 

6. In (44) there is an equality where terms of the form $-\beta \sum_y \cdots$ seems to disappear. Is (44) intended to be an inequality instead of equality? If not, can you provide the details?

7. Could you provide the skipped steps between equation (47) and the following inequality? Because the notation is quite dense, it is difficult to recall all the original definitions. Including the intermediate steps would make the proof much clearer and providing explicit pointers or brief reminders of key definitions would greatly help readers follow how each inequality is derived.

---

Comments 

1. The experiments use a “harmlessness score,” but there seems no definition in the main text. At least, it would be better to specify whether higher is better or lower is better.

---

Minor comments (typos)

* The reward function $r$ is first defined on $X\times Y$ with notation $r(x,y)$ in (2), but later the paper mostly uses $r(y,x)$. Please make this consistent.

* Possible typo in the definition of $Z_{N,\beta}$ in L204: the exponent $-1$ after the expectation looks like a typo.

* Possible typo in L448: the denominator shows $C\_{\beta, \hat{r}\_C, \text{ref}}$ but based on context it seems this should be $C\_{\infty, \hat{r}\_C, \text{ref}}$.

* Eisenstein et al. (2023) appears to be listed twice in the references (page 10).

* Lemma E.9: “miximizer” should be “maximizer.”

* Lemma E.11: the numerator is missing a $\delta$.

### Soundness
2

### Presentation
1

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
The point of this paper is to adapt LLM's to make their output adjust to some human evaluation metric (e.g. no harmfulness). This is typically done by a human preference function that is learned from data. By sampling multiple generations N  and choosing the best according to the learned preference function, a strategy called Best of N, can be used to generate such outputs adhering to the preference. However, for imperfect preference function, sampling more samples can lead to samples that are "overoptimized". This means that the samples obtained are not good according to the underlying human preference anymore. This work proposes Soft Best-of-N, which applies a soft-max when selecting the best sample, which regularizes the procedure. This paper studies the Soft Best-of-N theoretically, showing bounds on its performance, and analyzes when this regularization can improve performance over Best-of-N. The theoretical results are empirically verified by a small experiment.

### Strengths
- The paper is self-contained, even though I am not an expert in this particular domain I could get the main results that are quite technical
- The small experiment corroborates the theoretical findings

### Weaknesses
- Sometimes its not entirely clear what are the key novel results of this paper compared to other papers (e.g. Huang 2025), it would be good to highlight this more 
- From a rough reading of Huang 2025 it seems they also aim to solve this exact problem. How does your proposed algorithm compare to theirs? Note that I am not asking for an experiment or anything, but a theoretical interpretation would also be fine for me (both would be even stronger). 
- This paper has no Discussion section which I think should be in each quality paper; addressing: 1) are the results expected compared to related works? provide any context. 2) is it possible that there were any mistakes or weaknesses in the experiments? discuss them. 3) is there any future work? what would be most interesting to tackle next?

### Questions
Especially Questions 1-4 are particularly important for me and to decide if I will increase my score further. 

1. Why does Eq 10 measure performance? The KL term is not justified anywhere. This seems especially weak, since the SBoN method depends on Beta (while BoN does not); therefore; isn't it logical that the evaluation in terms of this loss that depends on Beta favors SBoN? Or is actually Eq 9 used? (the notation is a bit unclear here; because J is defined twice). Furthermore, since Eq 10 depends on the true r^*; I think there is no reason to stick close to the reference model, right? I don't see why KL is necessary here.
 
2. I think I got a bit confused with the bounds. For Theorem 4.3, you state around line 362 that: "We aim to minimize the second term.". But, this bound is an upperbound on the reward, right? Why would we want this term to be small? I understand why we want to minimize the upperbound on regret bounds (e.g. Theorem 5.2), but I am missing something here why we are looking at this upperbound and why we want it to be small.

3. In line 447 a quite technical condition is given under which the SBoN bound is favored. Can you explain what it means? You also allude to this in the last sentence of the conclusion; what are these conditions? 

4. How does the proposed method compare to Huang's? E.g. their inference time pessimisim method? (Given that you explicitly compare some of your results to 
theirs, isn't it fair to also compare to their strategy?) 

5. "We assume that overoptimization regime happens whenever we have \epsilon > 0"; this assumption seems strange to me. Say that the preference function is just very poorly trained (or not trained for that sake); then \epsilon > 0 right? Would this also be "overoptimization"? 
6. "Note that in (Huang et al., 2025), raw (uncalibrated)
reward models are utilized for error definition"; so? Does that in any way invalidate their result, or are their results suboptimal for that reason? Can you show this with an example? 
7. "Similarly, we can define ... as the optimal policy under the calibrated proxy-reward model"; isnt this strange, since this is not nececarily the "optimal" policy since it can have issues due to overoptimization?

Note that I did not check the mathematical proofs in the appendix, I only focused on the main body.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper provides a theoretical and empirical analysis of Best-of-N (BoN) and its smooth variant Soft Best-of-N (SBoN), two inference-time alignment methods for generative models.
The authors introduce a formal framework that connects SBoN to KL-regularized reward maximization and analyze two key quantities: (1) the KL divergence between the aligned and reference policies, and (2) the regret, i.e., the calibrated true-reward gap between the optimal and aligned policies.
They derive finite-sample upper bounds for both KL divergence and regret under general conditions, and show that SBoN can outperform BoN when proxy reward models suffer from overoptimization (reward hacking). Empirical results on text generation tasks confirm that smoothing (via temperature β) mitigates reward overoptimization when reward model quality is low.

### Strengths
- The poor calibration of reward model is a widely known problem in the community, which will largely affect the BoN (test time scaling) results as reward models are sensitive to OOD data, and one single poor-rated trajectory can influence the final BoN result. And the paper proposed SBoN seems show better performance.

### Weaknesses
- The experiments are limited, I suggest to verify the effectiveness  by conducting more experiments on recent non-verifiable agentic tasks.

### Questions
same as weakness

### Soundness
3

### Presentation
3

### Contribution
3
