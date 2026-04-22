# Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 2, 8

## Abstract
Conventional preference learning methods often prioritize opinions held more widely when aggregating preferences from multiple evaluators. This may result in policies that are biased in favor of some types of opinions or groups and susceptible to strategic manipulation. To address this issue, we develop a novel preference learning framework capable of aligning aggregate opinions and policies proportionally with the true population distribution of evaluator preferences. Grounded in social choice theory, our approach infers the feasible set of evaluator population distributions directly from pairwise comparison data. Using these estimates, the algorithm constructs a policy that satisfies foundational axioms from social choice theory, namely monotonicity and Pareto efficiency, as well as our newly-introduced axioms of population-proportional alignment and population-bounded manipulability. Moreover, we propose a soft-max relaxation method that smoothly trades off population-proportional alignment with the selection of the Condorcet winner (which beats all other options in pairwise comparisons). Finally, we validate the effectiveness and scalability of our approach through experiments on both tabular recommendation tasks and large language model alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes an approach to better represent human preferences from a diverse population, ie reflecting the mix of views not just the majority, in a trained policy. It comes with guarantees on its proportionality to the actual population (population‑proportional alignment, or PPA) and its manipulation by a group (population‑bounded manipulability, PBM). And with a parameter knob to trade off proportionality against picking the Condorcet winner (the one option that beats all the others, pairwise) Their experiments show that they are able to get a good PPA with a win rate on par with RLHF, offering no proportionality.

### Strengths
- RLHF/NLHF tend to pick a single preference even when the "camps" are close; their approach better handles mix views from a diverse population of evaluators and align a policy according to its distribution ;
- it offers axiomatic guarantees, first monotonicity (if an option pairwise standing improves its probability doesn't go down) and Pareto efficiency (if everyone prefers $y$ to $y’$, then $y$ gets at least as much weight as $y’$), and their proposed PPA (every option gets at least a fraction of its true population share) and PBM (the influence of a group is bounded by a function of its share) ;
- the $\beta$ parameter adds flexibility between converging to Condorcet (higher $\beta$) or aiming for proportionality (lower $\beta$) ;
- they confirm their predicted results (with RHLF, NHLF, and their algorithm), observing the  $\beta$-induced trade-off.

### Weaknesses
- this is not an easy read: the topic has inherent complexity but the intuition for the algorithm is not easy to grasp from the math-heavy notation —it also means that, under time constraint, I cannot guarantee that sections 3 and 4 are fully sound ;
- despite this machinery, the true share of the population remains elusive, and a conservative strategy has to be picked ;
- it would have been nice to better illustrate the theory with use cases clearly showing how the $\beta$ parameter can be adjusted in situation ;
- the second, large-scale, experiment, shows a muted trade‑off on Alpaca‑GPT4, so the central PPA claim is hard to appreciate —the suggestion that "this is because the group distributions of outputs are estimated using an annotation model" is not satisfying regarding that claim.

### Questions
- out of curiosity, how would the approach be adapted to handle more than pairwise comparisons, e.g. respect the ranking of multiple options by a diverse population ?
- the influence of a single group is bounded, but isn't there a risk of collusion amongst groups?

### Soundness
4

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
The paper targets distributional pluralism without group labels. It introduces PPA (probability mass at least proportional to each group's top choice share) and PBM (single-group manipulation is bounded by its true share) and gives a simple rule by considering the worst pairwise win it has against any other option and then set each option's probability to that worst win score, rescaled. There's a temperature knob shifting probabilities into the strongest options. The authors offer some experimental validation of this approach.

### Strengths
It clearly exposits what RLHF/NLHF does not guarentee. It provides theoretical analysis to back the proposed rule, and the rule itself is simple with a tunable $\beta$ term, and bridges social choice theory and preference learning.

### Weaknesses
There are a few issues I have with the paper.
- First of all, even though the author motivates with examples, the proposed method exhibits a clear trade-off as shown in the experiment: as beta increases, there's less pluralism but the performance increases. The paper shows the trade-off but offers little operational guidance for picking beta in real systems where “the population” is dynamic and contested. This pushes heavy governance choices onto practitioners.

- Perhaps most concerning to me is a lack of evidence that this method works. The only LLM study fine-tunes Qwen2.5-3B-Instruct; NLHF isn’t included, and group proportions in Alpaca are inferred by GPT-4.1 classifiers (no human labels). The authors themselves say differences across beta are "less distinct" due to annotator noise and that measuring PPA for LLMs are still an open challenge. This cast doubt on practical employability. The clearest empirical story is the tabular MovieLens task (1,297 rankings, 20 movies), where PPA/PBM can be computed exactly and the trade-off is clean; but this is not an LLM domain, so generalization to instruction tuning is unclear.

- PBM is weaker than strategy-proofness and difficult to measure in practice. Because it relies on unknown group labels, its guarantees aren’t verifiable. It limits—but does not eliminate—misreporting incentives; when manipulation is cheap, the remaining incentives can still be significant.

- As the authors pointed out in sec 4.3, the proposed rule is not condorcet consistent - even with a majority winner, it deliberately allocates mass to minority options, which comes at odds with desirable notions in many application which wants to pick the majority winner.

- The method only observes head-to-head outcomes and not true group sizes. It falls back on conservative upper bounds but when many matchups are close, the resulting policies become nearly flat even if one group is actually much larger.

### Questions
- Is there a principled way to choose or adapt beta across contexts and over time (under distribution shift), especially when the two desiderata are hard to measure reliably and given the trade-offs noted in the paper?

- Can the authors validate on more standard RLHF-style settings (preferably with human or auditor-verified cohort labels, etc.), given that the current Alpaca example relies on classifier-inferred groups and model-generated outputs and seems too noisy to support firm conclusions?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors attempt to expand beyond RLHF and NLHF in the area of AI alignment. Existing strategies often prioritize majority views at the expense of the minority; this not only excludes these views but can be brittle when majorities are slim. The authors use axioms of social choice to propose a method that better represents the proportion of disagreement in minority-view groups, using the top preference. They also introduce 2 axioms of their own. 

They later propose a way of trading off between dueling objectives and validate their proposals empirically.

### Strengths
* The authors tackle a key problem in alignment; under representation of minority class preferences.

* I did not rigorously check the math, but I was able to follow the equations and to my eye they seemed correct.

* The tradeoffs of the method are clear, the authors also include a parameter to tradeoff between PPA (a metric they introduce) and Condorcet consistency (more on that in weaknesses)

* Ultimately this is a well-written, well-justified paper tackling a key problem in alignment. The social choice framing fits very well.

### Weaknesses
* I felt that the experiments section had 3 major deficits:

** The movielens results seems somewhat underdeveloped. In many ways this seems to be a fairly ideal case, with potentially large disagreement. The results provided seem to back this, but they are provided briefly in sentence form, and not included with tables, more specific intermediate values, making it hard to get a sense for how robust or reliable those values are.

** The baselines seem relatively reasonable, but given the framing of the paper as "Beyond RLHF And NLHF", I am puzzled by the lack of comparison between these methods. It seems fairly paramount to understand how the method performs relative to those.

** The use of $\beta$ is underdiscussed prior to the experiments, in my view. While it is true that the prupose of $\beta$ is discussed, the authors show a wide variety of beta values. It is unclear to me how one would pick a specific value ina  principled manner.

* I felt 4.3 could have used a better exploration of use cases for $\beta$. I applaud the authors for including such a lever, but it seems to me that how one would decide on usage of $\beta$ is underdeveloped.

* I feel also that the discussion of manipulation in 5.1 is underdeveloped. While the authors do have bounds on manipulation by the minority class, a more detailed set of empirical results would be appreciated. I feel this would also tie into a discussion on when to use $\beta$.

### Questions
1. Can you elaborate on the movielens experiments? What did the full "path" of beta values look like there?

2. What guidance would you give practictioners on the choice of $\beta$?

3. Do you have comparisons between RLHF/NLHF on the experimental sections? If not could you attempt to provide some?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes “population-proportional alignment” where a model’s output distribution is meant to mirror how many people put each option at the top of their ranking, i.e., options that are the first choice for larger groups should get more probability mass. Because the true share of “ranked first by voters” isn’t directly observable from pairwise comparisons because the authors assume annotator identities are not available, the authors estimate very simple per-option bounds from the weakest head-to-head results and then allocate probability in proportion to those bounds. They pair this with a loose “manipulability” control that limits over-representation by any one group, and they add a softmax variant that can shift mass toward majority winners. The experiments illustrate a tunable trade-off between win rate and this top-choice proportionality notion, with weaker effects when the group shares have to be estimated. Experiments on MovieLens and small LLM fine-tunes show the intended trade-off between plurality and top-choice proportionality (and lower PBM in the simulator), but the effect is weak on Alpaca when group shares are estimated by a classifier; the two-phase training budget is comparable to RLHF and higher than DPO.

### Strengths
The main strength is the attempt to do dsitributional/proportional alignment based on preference structure instead of based on pre-assigned group labels.

### Weaknesses
This is a weak paper as is. The main “contributions” rely on a narrow, first-choice notion of proportionality and a misnamed “strategyproofness” that is really an over-representation cap. Core claims are overstated, proofs are straightforward given the simplified setup, and the paper overlooks major, directly relevant social-choice literature.

1. If “alignment” means matching users’ top choices, supervised fine-tuning with annotators providing preferred options (either in free form text or by choosing their top option) already does this. That setup is well known and not novel. It's perfectly reasonable to assume that we could get annotator ids and seems like an unnecessary constraint in this paper. (The necessary/ important constraint is indeed not having access to group labels)

2. Your proportionality tracks only top ranks. Focusing on first choices wastes richer preference information that could support consensus-oriented outcomes. A rule like plurality (which returns the winner with the most top votes) is the rule within social choice that I would say is at a similar level of sophistication. However, it is well known that it cannot capture compromises or broad consensus (e.g., the “everyone’s #2” candidate that increases average welfare). With such a weak target, the results do not speak to meaningful proportional representation.

3. The paper largely ignores decades of work on proportional representation and budgeting (e.g., JR/EJR, proportionality for solid coalitions (PSC) and fairness in budgeting/ portioning) etc that directly addresses proportionality using full rankings or other preference formats. 

Some references:
Portioning using ordinal preferences: Fairness and efficiency by Airiau et al.
Proportional Participatory Budgeting with Additive Utilities by Peters et al.
Justified representation in approval-based committee
voting by Aziz et Al.
Voting procedures (book by Dummett)

4. Because proportionality is defined at the top-choice level and feasibility bounds are simple, the proofs are immediate. The most involved argument repeats a known result (Proposition C.1 that the rewards output when optimizing the BT objective are in the same order as the Borda ranking of candidates). There is no technical depth. 

5. Consider e.g. Lemma 4.7 and 4.9. The latter has a lower bound that is sometimes 1, but often seems to be of the order of constant /m. Interpretations of how good this simple bound is in a larger class of examples is missing.

6. Claiming RLHF “fails proportionality” in Example 3.1 is misleading. n the two-candidate case, the reward/DPO model can encode the correct split (e.g., 1/2+ε), it just that during expected reward maximization this information gets lost.

7. The manipulability condition simply bounds over-representation. It is not strategyproofness in the social-choice sense (unilateral profitable deviations). In particular, with γ=1/2, an arbitrarily small minority can still drive up to half of the probability mass. It's way too weak to be meaningful. The discussion around lines 246–248 is unclear and mixes group and unilateral deviations.

8. Section 4.3 is conceptually confused. If the output is a distribution over candidates, Condorcet consistency and proportionality are fundamentally at odds (and as a technical claim this is trivial); the paper’s own motivation - distributional alignment/Overton pluralism implies we should favor proportional mixtures, not a winner-takes-all outcome. Dragging Condorcet back in (e.g., via large-β limits) contradicts that goal and reads like a reversion to single-winner logic. Arguably, a Condorcet winner matters mainly in settings with a ground-truth label, which is not the setup here. If you insist on accommodating Condorcet, do it only as a secondary priority within the most proportional distributions: when a Condorcet option exists, give it the highest probability within the proportional mix, without collapsing the mixture. As written, the section is misleading and undermines the paper’s stated objectives.

9. I looked at the experiments in the main body, but did not look into the appendix in detail. The experiments confirm the theory/approach prediction in some sense, but I am not convinced by the theory/approach and its novelty in the first place.


Minor comment: In the experiments you seem to use "RLHF" and "DPO" interchangably? At least in the first experiment you say that you compare to RLHF not to DPO, but then Table 2 contains a comparison to DPO.

### Questions
I don't have any questions.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies preference optimization algorithms under social choice theory.
Specifically, any profile induces a preference function, and preference optimization algorithms learn a policy from the preference function, which together form a PSCF.
PSCF is desired to satisfy four axioms, *i.e.*, monotonicity, Pareto efficiency, PPA, and PBM.
The paper shows that popular preference learning algorithms, *e.g.*, RLHF and NLHF, do not satisfy PPA and PBM.
While a typical PSCF, *i.e.*, random dictatorship, satisfies all axioms, it cannot be implemented by any preference algorithms from preference function.
To bridge the gap, this paper proposes a preference learning algorithm that satisfies all axioms.
This paper also proposes a softmax relaxation of the proposed algorithm to achieve trade-off between two conflicted axioms, *i.e.*, PPA and Condorcet consistency.
The proposed algorithm is evaluated empirically under the tabular and function approximation settings, validating the theoretical analyses.

### Strengths
* This paper studies a fundamental and important problem, reveals the shortcomings of existing algorithms, and proposes desired method through solid theoretical analyses, which is insightful and meaningful.
* This paper has clear structure and writing, making it easy to understand.
I enjoyed reading the paper.

### Weaknesses
I found no obvious flaw of this paper.
My only concern is that this paper mostly studies social choice theory in the context of preference learning, and therefore may not be interest of most audiences in the ML community.

### Questions
I only have an undergraduate-level understanding of social choice theory and am not sure whether the proposed algorithm is trivial within the field.
Maybe the authors could add literature review to help the audiences from ML community to better understand the background of the problem being addressed.

### Soundness
4

### Presentation
4

### Contribution
3
