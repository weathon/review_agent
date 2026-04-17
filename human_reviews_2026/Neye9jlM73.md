# Pragmatic Curiosity: Unifying Bayesian Optimization and Experimental Design via Active Inference

- Decision: Reject
- Scores: 6, 8, 2

## Abstract
Bayesian Optimization (BO) and Bayesian Experimental Design (BED) have traditionally offered separate solutions for goal-oriented and information-oriented tasks, respectively, leaving a gap in complex problems where learning and optimization are not separate phases but deeply intertwined objectives. In this paper, we provide the first unified framework of BO and BED, which is rooted in the principles of active inference (AIF). We introduce "pragmatic curiosity", a new paradigm where the classic explore-exploit dilemma is resolved by minimizing a single objective: the Expected Free Energy (EFE), which naturally balances pragmatic (goal-seeking) and epistemic (information-seeking) drives. We demonstrate the power of this approach on a suite of challenging hybrid tasks, including constrained system identification, targeted active search, and composite optimization with unknown preferences. Empirical results prove the cross-domain adaptability and effectiveness of our proposed framework: our "pragmatic curiosity" paradigm consistently outperforms standard baselines in BO and BED, demonstrating quantifiable improvements in key metrics like estimation accuracy, critical region coverage, and final solution quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper bridges Bayesian Experimental Design (BED), which focuses on learning unknown environmental parameters and Bayesian Optimization (BO), which focuses on finding optimal configurations, with the theory of Active Inference (AIF). It interprets the expected free energy as the sum of information gain (learning) and surprisal (decision-making).

Based on this insight, the authors propose a new acquisition strategy called _pragmatic curiosity_, which balances exploration and exploitation in a principled manner without manual tuning. Experiments demonstrate its effectiveness across tasks involving explicit and black-box learning and decision-making goals.

### Strengths
- The authors provided theoretical intuitions about the connection between BED and BO. They leveraged AIF, a framework common in computational neuroscience, to formulate their results. They also validated their findings in the UCB acquisition of BO. This provides new insights to the BED, BO, and AIF communities.
    
- The authors further designed a novel decision-making strategy based on the theoretical findings and validated it through experiments. This provides solid evidence for the proposed framework.

### Weaknesses
- The formulation of AIF is not clear, especially the relationship between the "true model" $p(\cdot)$ stated in Line 161 and the prior preference distribution over the outcome $y$, $p(y)$.
    
    - Usually, we assume a prior distribution of $f(x) \sim p(y|x)$. In Line 648, $p(y|x)$ is simplified as $p(y)$. Is it because it is assumed that $p(x) \propto 1 \implies p(y|x) = p(y)$?
        
- I have a concern in the technical details. While the authors assume a surrogate $q(s)$ for parameters $s$ whose real distribution is $p(s)$ in Equation (1), this assumption is **intentionally broken** in the derivation of the form of expected free energy in Line 668 by assuming $q(s) = p(s)$. Thus, the form of expected free energy is just an approximation. How could the authors justify this?
    
- There is an issue in the experimental section. The authors claim to have compared their method with standard BO in Sec. 5.1 and Sec. 5.2, but the BO results do not appear in the corresponding figures.
    
- The framework introduces temperature/curiosity parameters ($\beta, \gamma$, etc.). The paper claims that the balance **emerges** from beliefs, yet experiments still set $\beta, \gamma$ per task (Appendix lists the values). It remains unclear how sensitive performance is to these settings and whether the method truly removes manual tuning in practice. The authors should provide sensitivity plots or guidance.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes a framework called pragmatic confusion, combining the goal-oriented objectives of Bayesian Optimization (BO, where the goal is the maximize an objective), and Bayesian Experimental Design (BED, where the goal is to gain maximal information about unknown system parameters). The framework is based on active inference, which attempts to minimize the expected free energy, leading to a balance between pragmatic (goal-seeking) and epistemic (information seeking) objectives, and includes various standard BO and BED frameworks (eg GP-UCB and EI) as special cases.

### Strengths
The paper is quite well written. I found the underlying idea compelling, and as best as I can tell the mathematics is accurate. The idea matches quite well with the motivation for GP-UCB, but generalized and systematised to cover a much wider range of problems than is typical for BO alone. Experimental results are impressive for the range of problems covered.

### Weaknesses
In general, I do wonder about the computational cost required to maximize the acquisition function, but experiments appear to indicate that the approach is practical.

### Questions
None.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors offer a unifying framework for two historically different fields, Bayesian Optimization and Bayesian Experimental Design. Although the common wisdom takes these for being almost identical, being able to precisely characterize in what sense this is the case is a more difficult task. To that end, this submission introduces ``pragmatic curiosity'', a new paradigm that offers to resolve the exploration-exploitation dilemma across BO and BED, by the minimization of a sole objective called Expected Free Energy.  After introducing their theoretical framework and deriving an acquisition criterion from it, the authors apply their approach to a range of experiments from constrained system identification, targeted active search, and composite optimization with unknown preferences.

### Strengths
- The paper recasts the exploration/exploitation dilemma in active learning/Bayesian experimental design/Bayesian optimization as a tradeoff between information-seeking behavior, and goal-directed behavior. 
- On the experiments selected by the authors, the proposed approach yield satisfactory results.

### Weaknesses
This might be entirely on me, but I found the paper to be badly written. While the theoretical framework and the general intuition were understandable to me, the execution, meaning the practical computations and the experiments on test problems, is a major flaw to me. 
I believe there are simply too many things lacking, and too many things in the Appendix that should have been in the main text. 

- For each problem, I think we should be able to tell __from the main text__ its dimensionality: how many variables are we optimizing for? This should not be part of the appendix. It could be as simple as having "Source Localization 2D" in the xlabel of Figure 1 for instance. For Fig1/Sec 5.1 & Fig2/Sec 5.2, input dimensionality & hypothesis space could be retrieved from the appendix, but for Fig3/Sec 5.3, I could not find it. I assume it is well-described in the reference mentioned therein, but having to look through the text of a reference mentioned in the appendix of your paper to find something as simple as problem input dimensionality seems cumbersome.

- While I could see (e.g., through Table 1) the connection with BO, it was more difficult to pinpoint how this work relates to BO during the experiments section. For instance, the last example, section 5.3, involves composite BO. It is mentioned that "since the setting in this category does not follow the traditional BO or BED setting, there is no baseline that can be directly used". What about the paper which you mentioned then, [1]? Can you explain why this approach could not be employed here? From a more general point of view, for a paper that states in its title "unifying Bayesian optimization and experimental design", I would have expected to also see classical BO acquisition functions, like EI, UCB, or MES, being compared. 

[1] Preference Exploration for Efficient Bayesian Optimization with Multiple Outcomes, AISTATS 2022.

- The practical computation/estimation of equation 7 is lacking. 

- I believe a substantial amount of literature could have also been mentioned here:

[4] Maximizing acquisition functions for Bayesian optimization, NeurIPS 2018, shows how many classical BO acquisition functions can be written as the expectation of specific utility functions $u$ under a posterior distribution $p(y|x, \mathcal{D})$, I think this is worth mentioning, specifically how your Table 1 compares to their Table 1.

[5] Prediction-Oriented Bayesian Active Learning, AISTATS 2023, introduces a new state-of-the-art Bayesian Experimental Design criterion that allows going beyond purely reducing parameter uncertainty, which can be suboptimal from a predictive performance standpoint. This criterion has been since then adopted in many other studies since then, and in my opinion, it should be added to the baselines.

[6] Self-Correcting Bayesian Optimization through Bayesian Active Learning, NeurIPS 2023, shows how active learning and BO can be related by leveraging classical active learning criteria to reduce the uncertainty in GP surrogate hyperparameters in BO. They propose an acquisition function that balances this reduction in hyperparameter uncertainty while also seeking promising candidates; this is the very tradeoff illustrated in this submission. At the very least, this should be mentioned. 

I did not run it myself, but I assume a deep search run with any language model would have surfaced these publications, and perhaps more. Actually, it just occurred to me that the paper does not contain a related work section. The introduction is quite dense and contains related work, but only to a limited extent.

I might be wrong on several things here, and upon genuine misunderstanding from my side and clarifications from the authors on the weaknesses (and questions below), I would consider raising my score.

### Questions
- Can you provide a rationale for increasing the degree of curiosity $\beta$ in the experiments depicted in Sec 5.1 $\beta$ goes from 0.5 to 1 to 5 depending on the setting. More generally, carrying an ablation study where $\beta$ is varied would have been helpful.

- In preliminaries, Section 2.1, the acquisition function is defined as taking values in $\mathbb{R}_+$, this need not be the case, for instance with GP-UCB if the GP surrogate takes negative values?

- How is the saturation threshold $C(y)$ defined in Section 5.1 computed?

### Soundness
2

### Presentation
2

### Contribution
2
