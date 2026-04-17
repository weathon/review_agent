# PFO: Optimizing binary Preference Alignment from a Probability Flow Perspective

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
In contrast to pairwise optimization methods, binary preference alignment algorithms do not process sample pairs jointly, which may lead to suboptimal probability distributions. Under the influence of the squeezing effect, probability mass flowing out of negative samples may diffuse into neutral regions, while the mass absorbed by positive samples might originate from such neutral areas, resulting in insufficient penalty for negative responses. To address this issue, we propose the PFO (Probability Flow Optimization) algorithm. The algorithm dynamically evaluates the probability of sample generation and systematically optimizes the transfer of probability mass by reweighting samples to encourage flow from negative to positive distributions. Comprehensive experiments and analysis on the general-purpose benchmarks MT-Bench and AlpacaEval 2 demonstrate the algorithm's effectiveness. Furthermore, experiments on recommendation domain datasets show that the method effectively applies to sparse feedback scenarios, confirming the algorithm's broad applicability. Our work offers new insights into improving binary preference alignment from the perspective of probabilistic flow.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper suggests a potential squeezing effect in binary preference alignment (like KTO), where optimizing on single positive or negative samples can lead to suboptimal probability distributions. Specifically, it suggests that probability mass from penalized negative samples may flow to neutral non-target regions, and mass for rewarded positive samples may be drawn from these same neutral regions, rather than directly from the negative ones.

To address this, the authors propose Probability Flow Optimization (PFO), a new loss function that reweights samples to encourage a more direct flow of probability from negative to positive distributions. The core modification is the application of a sigmoid function to the log-probability ratio ($r_{\theta}$) before it is used in the loss. The authors claim this stabilizes training by preventing "sample drowning" and gradient explosion from unbounded $r_{\theta}$ terms (as seen in KLDO), thereby allowing the reweighting to work effectively. The paper presents PFO as a variational lower bound on the Rényi $\mathcal{D}_2$ divergence.

### Strengths
- PFO and PFO+ demonstrate consistently strong performance, outperforming previous baselines (KTO, KLDO) across multiple models (Llama3, Gemma2) and on two different task domains (dialogue and recommendation), although the margins are small.

- The final algorithm is simple to implement, essentially modifying the KLDO loss by wrapping the $r_{\theta}$ term in a sigmoid function.

- The inclusion of the PFO+ variant is useful. It helps isolate the contribution of the sigmoid function (as PFO+ is nearly identical to KLDO but with $\sigma(r_{\theta})$)

### Weaknesses
- I think the central weakness of this paper is the significant gap between its main claim (solving the squeezing effect) and its evidence. The squeezing effect is defined as probability mass flowing incorrectly to/from "neutral regions" or "non-target samples". However, the paper never measures this probability flow. While a larger margin is good, it is not direct proof that the squeezing effect (as defined) has been solved. The results demonstrate that PFO is a more effective optimizer for separating $y_w$ and $y_l$, but they fail to show why. The improved performance could be due to other factors, such as general gradient stabilization, rather than a specific correction of probability flow.

- The core argument for why the sigmoid function is the correct solution is convoluted. The logic appears to be: (1) The squeezing effect would be mitigated by reweighting samples. (2) Existing reweighting (like KLDO's $e^{r_{\theta}}$) is unstable and causes "sample drowning." (3) The sigmoid $\sigma(r_{\theta})$ stabilizes this reweighting. This is an indirect justification. Additionally, I found the paper writing to be often unclear, with some typos and inconsistencies ("off-line" rather than "offline" on lines 39 and 112, "Pairwise" capital on line 43) as well as unclear sentences (line 86 "where positive data require more protection" -> why?, "data acquisition because online feedback is often directed at a displayed reply and either likes or dislikes it" -> grammatically incorrect and I could mention more)

- The primary technical contribution is the addition of a sigmoid function to the KLDO loss. This could be interpreted as a minor stabilisation trick (i.e., a form of gradient clipping) rather than a novel alignment algorithm. Given that PFO+ (which performs very well) is only this modification, I don't see the novelty of the contribution.

### Questions
- Can the authors provide more direct evidence to support their central claim? For example, could they design an experiment to measure the probability mass assigned to a set of "neutral" or "non-target" samples and show that PFO, unlike KTO or KLDO, successfully prevents probability from flowing to this set?

- The justification for the sigmoid is to prevent "sample drowning" from the $e^{r_{\theta}}$ term in KLDO. Can the authors provide an analysis (e.g., gradient statistics, or the contribution of the top-k samples to the batch gradient) to prove that this is a significant problem in practice for KLDO, and that the sigmoid function effectively mitigates it?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Probability Flow Optimization (PFO) as an alternative objective for training on pairwise examples. The paper begins by extending the squeezing effect claims of Ren & Sutherland 2024 to additionally account for the effect of positive training (as prior work only looked at negative training). The paper then proposes PFO, motivated by the goal of reducing the squeezing effect. The paper compares the gradient of the proposed method to those of KTO and KLDO conceptually, and proves that PFO is a lower bound for the Renyi divergence ($\alpha=2$) between positive and negative samples. Experiments show that PFO increases log probability margins more than baseline methods and yields better results in a dialogue and recommendations experiment respectively.

### Strengths
1. The paper builds on impactful analysis around the squeezing effect introduced in Ren & Sutherland 2024, offering a more holistic picture the probability changes of tokens between gradient steps by adding the analysis for positive training.
2. The relationship between the objective and the Renyi divergence, $\alpha=2$, is interesting and offers useful intuition, and the stability proof builds confidence that it is not vacuous. 
3. The insight provided by the lines 216-229 and Figure 3 are a nice detail in describing the development of the method.
4. The authors test PFO in multiple experimental settings and show positive downstream performance.

### Weaknesses
1. While the motivation for the method is overcoming the squeezing effect and enabling more probability flow from negative to positive examples, the experiments still show that PFO suffers from log probabilities going down for both chosen and rejected. Indeed, even though PFO yields the largest log prob differences between chosen and rejected, Figure 2 shows that chosen and rejected log probs still go down (and in fact the most) with the proposed method. Thus, the connection between the squeezing effect analysis and the proposed PFO method seems weak. 
2. PFO makes quite a few different changes relative to existing methods, and it is hard to ascertain / isolate the influence of each. Either additional experimental ablations or a more step-by-step walkthrough of the design choices would help build confidence in the method and the insights ascertained. 
3. The gradient analysis of 3.4.1 seems a bit superficial, including claims such as the lack of an adaptive weighting may cause situations in Claim 3 and Extended Claim 3. This section would be be much more impactful with experimental evidence. 
4. The experiments would benefit from analyses showcasing what differences are significant.

### Questions
1. Could the authors clarify the design choices for the PFO objective in a more step-by-step fashion?
2. What is the significance of the log prob difference results (i.e, PFO achieving the largest) when chosen and rejected log probas still go down (and the most) for the proposed method?
3. Could the authors connect the hypotheses made in the gradient analysis with experiments?
4. Could the authors conduct significance testing for the experimental results?

### Soundness
2

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
3

### Summary
The paper identifies that the squeezing effect in binary preference alignment affects both positive and negative samples training . They propose PFO which dynamically reweights samples to encourage “probability mass flow” from negative to positive distributions. The experimental section shows performance on Alpacaeval 2 and MT Bench. Overall this is a nice extension (to some degree) to the KLDO paper but has some structural inconsistencies as below.

### Strengths
1. The paper tackles an interesting problem and provides a simple solution, backed with some theoretical underpinning

2. The paper is well written in most places. Comprehensive and easy to follow.

### Weaknesses
1. The utility of PFO+ is a bit confusing to the main premise of the paper. On one hand the authors are motivating the paper at the start by saying that both negative and positive training suffer from the squeezing effect. They even have the extended claim to explicitly account for the problem with the positive training. Then they propose PFO+ which only handles a fix to the negative samples by adding a simple sigmoid over KLDO. Ideally it should still suffer from the problem. In Table 2 they show that PFO+ outperforms KLDO, but that's just on a very narrow recommendation domain, does it generalize, it feels very forcibly included? It's as if the authors propose that the problem lies with both positive and negative training and then go on to propose a method that just ignores the premise. Further PFO+ outperforms PFO in recsys seems like fixing positive sample squeezing effect is not at all always needed. PFO+ even once in a while does better even in the MT Bench or Alpacaeval which muddies the premise further. 

2. Further in while PFO has the largest $\delta$ margin, it is unclear why or how KTO even with better margin than KLDO and PFO+ (except llama3 base, which is strange on its own) has worse results than both in the tables. It is strange. 

3. The authors use ultrafeedback dataset annotated using the armorm reward model for preference optimization, the dataset already has chosen and rejected pairs of responses and even ranked responses. If achieving great results on MT Bench and Alpaca 2 is the goal, then they should just run some obvious pairwise feedback optimization algorithm on the same dataset to show how far off they are, something like at least DPO, and then perhaps SimPO or ORPO etc. as baselines. Very specifically since the paper builds a premise around binary feedback being cheaper. This omission makes it tricky to judge the real world value of the algorithm. 
    The authors can argue why we would want to compare binary algorithms (KTO, PFO etc) with those like DPO, SimPO, ORPO etc. By using a pairwise dataset but only testing binary feedback algorithms the paper avoids answering the question: “what is the performance tradeoff for using the cheaper binary alignment?” moreover, both the baselines KTO and KLDO papers had comparisons with other pairwise preference methods that were SOTA at that time. 

4. Unclear what happens with the reasoning models with PFO, for example if your base is a reasoning model? The reasoning tokens could be very similar leading to either right or wrong answer part of the response, i would imagine the log probs may still be very close between the positive and negative examples. In this case wouldn't the scores be similar too? could this lead to inconsistent reasoning behavior of the model itself?

### Questions
there are some open questions in the weaknesses section, if addressed, I consider increasing my score.

### Soundness
3

### Presentation
3

### Contribution
3
