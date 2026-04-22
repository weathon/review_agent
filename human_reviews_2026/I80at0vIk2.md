# When Novices Teach Better: Improving Behavioral Cloning with Low-Skill Data

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Behavioral cloning (BC), which trains models to replicate behavior from offline demonstrations, is a common approach in reinforcement learning.
Several prior works argue that BC requires expert demonstrations and performs poorly when trained on low-skill or suboptimal data.
We challenge this assumption by showing that, in certain regimes, training on low-skill demonstrations can yield models that outperform those trained on high-skill data.
Since expert data is often costly and scarce, while low-skill data is cheaper and more abundant, this finding has important practical implications.
To explain the result, we introduce a measure that quantifies the \emph{resilience} of a policy—its ability to maintain reward under random perturbations—and show that resilience aligns with observed performance differences.
Building on this insight, we introduce a new skill-based training curricula—structuring the training process according to policy skill levels—and show consistently improve BC performance compared to treating all data uniformly or filtering for experts.
We validate our findings in a synthetic environment and using human data from Chess and Racing, showing consistency across domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors show that in low-data regimes, training BC models on low-skill demonstrations can outperform training on high-skill demonstrations. To explain this phenomenon, they introduce a resilience metric that quantifies a policy's ability to maintain reward under random perturbations. They test their findings across 3 domains: a synthetic tree environment where they control for fragility of states, chess using both human games from Lichess and synthetic Stockfish games, and racing using human driving data from a simulator. The results show that when low-skill policies are more resilient than high-skill policies, BC models trained on low-skill data perform better at small budgets. And they also propose a skill-based curriculum that trains from low-skill to high-skill data.

### Strengths
I liked the core observation from this submission that low-skill data can outperform expert data in low-data regimes challenges a fundamental assumption in BC. This is counterintuitive and practically important.

### Weaknesses
1. I feel that the resilience metric is computed after observing the phenomenon rather than being used to predict when low-skill data will help (seems quite ad-hoc). The authors designed the tree domain to have fragile states, computed resilience in chess after seeing the results, and didn't even compute it for racing. For this to be a methodological contribution, I need to be able to use resilience on a new domain to make decisions about data collection.
2. There's no formal analysis of when this phenomenon occurs, sample complexity comparisons, or theoretical justification for the resilience-performance connection. Even a simple toy analysis would strengthen the claims considerably.
3. I found no statistical significance testing, confidence intervals, or error bars in many places. Given these are toy datasets, I would expect the authors to run some statistical tests before reporting results and drawing conclusions over them.
4. I feel that the curriculum learning results are weak. Figs. 3 & 5 show Filtered BC-LSC mostly just averaging between the baselines. (Plus there are no error bars to meaningfully conclude something) 
5. The authors claim low-skill data is "cheaper and more abundant" but test equal-sized datasets. The real question should be: is it better to get 100 expert demos or 1000 novice demos for the same (or maybe even lesser) cost? Or an ablation where you show the performance comparison for X expert demos vs n*X low-skilled demos.

### Questions
1. Can you demonstrate that the resilience metric can predict, on a held-out domain you haven't tested yet, whether low-skill or high-skill data will perform better? This would make the contribution much stronger.
2. At what budget does high-skill start to dominate? Is there a principled way to determine this threshold based on environment properties?
3. Do you have any theoretical or empirical analysis of the sample complexity tradeoff? How many low-skill demos equal one high-skill demo? 
4. Typos and Grammatical Errors:
Line 022: "show consistently improve" → grammatical error
Line 076: "for train effectively" → missing "to" ("to train effectively")

### Soundness
2

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
4

### Summary
The paper studies when training behavioral cloning (BC) purely on low-skill demonstrations can outperform training on high-skill demonstrations, especially at small data budgets. The authors introduce resilience: the expected return of a policy if exactly one action in a rollout is replaced by a random action (a one-step deviation), after which the policy resumes. They show that when low-skill data is more resilient than high-skill data, BC trained on low-skill can win at small budgets; with enough data, high-skill tends to retake the lead. They also propose a simple skill-ordered training schedule (Filtered BC-LSC): train on all skill levels, then progressively drop lower-skill slices and finish on experts.

### Strengths
- Clear, cross-domain empirical pattern: small-budget BC can benefit more from mistake-tolerant (high-resilience) trajectories than from brittle expert ones. The resilience metric offers a practical signal for which skill bucket to prioritize.
- Practical scheduling idea (Filtered BC-LSC) that operationalizes "start broad, finish sharp", consistent with the resilience story.
-  Separation of “which data” from "how much": matched-budget low-only vs. high-only comparisons isolate composition/resilience from sample size; fixed acquisition-budget analyses make the filtering trade-off explicit

### Weaknesses
- Narrow perturbation model for resilience: exactly one uniformly timed one-step deviation. BC errors can be state-dependent or clustered; sensitivity to non-uniform or multi-deviation (k>1) settings isn’t explored.
- Oracle dependence: resilience relies on an approximate value function (e.g., a chess engine). Robustness of the resilience ordering to the oracle choice and settings isn’t established.
- Racing details are partly indirect: constraints limit direct resilience computation; some conclusions rely on proxies and qualitative alignment rather than a fully shared evaluator.
- Training-compute accounting for curricula isn’t fully disentangled from the schedule itself (e.g., equalizing optimizer steps across schedules).

### Questions
- Composition vs. quantity: can you centralize, in the main text, the matched-budget low-only vs. high-only results and the fixed acquisition-budget schedule results to make the separation between “which data” and “how much data” unmistakable? Any sensitivity to alternative budget splits across skill bins?

- Oracle sensitivity: how stable is the resilience ranking under different value oracles (depths/engines) and reward scalings? A rank-correlation study would help.

- Perturbation model: what happens if the random action is sampled non-uniformly in time, state-dependently, or if you allow multiple random actions (k one-step deviations)? Do the main conclusions persist?

- Continuous control: for racing, how exactly are random actions sampled, and what proxies were used where direct resilience was infeasible? Please document the evaluator and sampling range.

- Compute-matched curricula: do curriculum gains remain when training steps/epochs are matched to the strongest baseline? If so, please add those numbers.


- Mixture selection: with a fixed budget and K skill bins, what mixed proportion of low/high skill maximizes performance? Even a small grid over mixtures in the synthetic domain would be informative.

- Distribution shift: do the resilience performance links hold under test-time shifts (e.g., different chess openings or track conditions)?

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
Behavior cloning typically requires high-quality expert data, which restricts learning to such datasets. However, curating this data is difficult, whereas low-skill (worse-performing) data is more abundant. Is there a way we can use such data? This project explores this question with two main focuses: (1) what are the different qualities of high and low-skill data? (2) how can we use low-skill data in conjunction with high-skill data? To address (1), the authors define “resilence,” and show that for select datasets, low-skill data has higher resilience. Higher resilience means that the underlying policy generating the data is more robust to noise (e.g. the policy is able to recover after taking a noised action, whether this is due to a better policy overall, or the state coverage of the policy leading to less harmful/extreme states). To address (2), the authors explore curriculum using both high and low quality data. The authors show results on 3 domains: a synthetic toy experiment, chess, and racing.

### Strengths
1. The idea of resilience is interesting. 
2. The intro and method section is fairly easy to read.

### Weaknesses
1. The experimental domains are a bit limited. The toy domain is useful for some analysis, but it is too synthetic. I’m not convinced by the chess or racing results.
2. The results aren’t too convincing. The gap between high and low skill is not too different. 
3. Missing prior works in data quality and curation (DemoScore, DemInf, etc.)
4. The curriculum is not particularly novel. 
5. Missing baseline where the policy is conditioned on the optimality of the data.

### Questions
1. I’m a bit confused by the training budget size. For example, for a given budget, if there are only two skill levels, the high-skill policy is trained on all high quality data, and the low-skill policy is trained on half high-quality and half low-quality, where the total amount of total is the same? (line 234)
2. How exactly is Eqn 1 calculated? Does the dataset include examples where different actions are taken from the same state? If so, I believe this is a strong assumption for many domains (although for chess, toy envs, etc., this is doable)
3. Could you calculate resilience with Monte Carlo rollouts in the environment, with added noise?
4. For Fig 4, is human data (fragile) a different opponent / environment altogether, or is it just a different training dataset.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper challenges the conventional wisdom in behavioral cloning (BC) that only expert demonstrations are useful. The authors show that in low-data regimes, training on low-skill demonstrations can outperform training on high-skill data when the low-skill policies are more resilient. They introduce a resilience metric to quantify this property and propose a skill-based curriculum (Filtered BC-LSC) that progressively incorporates higher-skill data. Experiments in synthetic tree environments, chess, and racing simulators validate their claims, demonstrating consistent improvements over standard and filtered BC baselines.

### Strengths
1. The authors provide a counterintuitive insight: Under certain conditions, specifically when the data-collecting policy exhibits higher resilience and the data budget is constrained, BC policies trained on low-skill data can outperform those trained on high-skill data.
2. Experimental results from multiple tasks across different domains demonstrate the validity of the theory.

### Weaknesses
1. The skill-based curriculum is heuristic and not theoretically justified; it would benefit from a more formal analysis or comparison to other curriculum strategies.
2. Lack of analysis on training time.

### Questions
1. In the evaluation of the chess task, the trained model is compared against a randomly-playing opponent. Is this a reasonable evaluation method? Can this approach genuinely reflect the performance quality of the trained policy?
2. Is the success rate in the racing experiment, obtained from only 5 attempts, statistically insufficient in terms of sample size?
2. Have you considered evaluating the approach in more complex or high-dimensional domains (e.g., robotics)?
3. Does the resilience of a policy relate to the diversity of states visited in low- vs. high-skill demonstrations?

### Soundness
2

### Presentation
3

### Contribution
2
