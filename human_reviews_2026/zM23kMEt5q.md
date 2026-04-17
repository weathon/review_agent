# Variance-Dependent Regret Lower Bounds for Contextual Bandits

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
Variance-dependent regret bounds for linear contextual bandits, which improve upon the classical $\tilde{O}(d\sqrt{K})$ regret bound to $\tilde{O}(d\sqrt{\sum_{k=1}^K\sigma_k^2})$, where $d$ is the context dimension, $K$ is the number of rounds, and $\sigma^2_k$ is the noise variance in round $k$, has been widely studied in recent years. However, most existing works focus on the regret upper bounds instead of lower bounds. To our knowledge, the only lower bound is from Jia et al. (2024), which proved that for any eluder dimension $d_{\textbf{elu}}$ and total variance budget $\Lambda$, there exists an instance with $\sum_{k=1}^K\sigma_k^2\leq \Lambda$ for which  any algorithm incurs a variance-dependent lower bound of $\Omega(\sqrt{d_{\textbf{elu}}\Lambda})$. However, this lower bound has a $\sqrt{d}$ gap with existing upper bounds. Moreover, it only considers a fixed total variance budget $\Lambda$ and does not apply to a general variance sequence $\{\sigma_1^2,\ldots,\sigma_K^2\}$.
In this paper, to overcome the limitations of Jia et al. (2024), we consider the general variance sequence under two settings. For a prefixed sequence, where the entire variance sequence is revealed to the learner at the beginning of the learning process, we establish a variance-dependent lower bound of $\Omega(d \sqrt{\sum_{k=1}^K\sigma_k^2 }/\log K)$ for linear contextual bandits. For an adaptive sequence, where an adversary can generate the variance $\sigma_k^2$ in each round $k$ based on historical observations, we show that when the adversary must generate $\sigma_k^2$ before observing the decision set $D_k$, a similar lower bound of $\Omega(d\sqrt{ \sum_{k=1}^K\sigma_k^2} /\log^6(dK))$ holds. In both settings, our results match the upper bounds of the SAVE algorithm (Zhao et al. 2023) up to logarithmic factors. Furthermore, if the adversary can generate the variance $\sigma_k$ after observing the decision set $D_k$, we construct a counter-example showing that it is impossible to construct a variance-dependent lower bound if the adversary properly selects variances in collaboration with the learner.
Our lower bound proofs use a novel peeling technique that groups rounds by variance magnitude. For each group, we construct separate instances and assign the learner distinct decision sets. We believe this proof technique may be of independent interest.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the variance-dependent lower bound for the linear contextual bandits.
While the previously known lower bound required the specific choice of variance-sequences $\{\sigma^2_k\}$ and only considered the total fixed variance budget,  this paper extends the analysis to more general variance sequences.

 For the prefixed sequence, where the variance sequence is revealed to the learner at the beginning of the learning process, the paper establishes an expected lower bound of　$\tilde{\Omega}( d \sqrt{\sum_{k=1}^K \sigma^2_k })$．


For the adaptive sequence: for the weak adversary (who must generate the $\sigma_k$ before observing the decision set $D_k$), 
they provide  the high probability lower bound of $\tilde{\Omega}( d \sqrt{\sum_{k=1}^K \sigma^2_k })$.


For the strong adversary who can generate $\sigma_k$ after observing the decision set $\mathcal{D}_k$, they show that the counter algorithm can collaborate with the adversary and achieve $O(d)$ regret even when $\sum \sigma^2_k =\Omega(K)$. This is due to the construction where the learner can effectively observe noise-free rewards in certain rounds depending on $\mathcal{D}_k$ and action set, obtaining full exploration in just $d$ rounds. In the remaining rounds, even when the variance is one, the regret remains  $O(d)$ when $K >2d$.

### Strengths
- The heteroscedastic linear contextual bandit  is a general model and it is important to investigate the lower bound of the cumulative regret. The previous result  only considered the total fixed variance budget.  This paper extends the analysis to handle more general variance sequences.

- The lower bounds match the upper bound in Zhao et al. (2023) up to logarithmic factors.
- A peeling technique for prefixed variance sequence and the orthogonal construction of decision sets across different groups are novel techniques to provide novel hard-to-learn instances.
Theorem 5.2 is the first high-probability lower bound for linear contextual bandit.

### Weaknesses
- Theorem 5.2 has additional dependency of $1/\log^6(dK)$, which is logarithmic but still somewhat loose compared to the upper bound.
- See also the question below regarding the assumptions in Theorem 5.4.

### Questions
For the strong adversary who can generate $\sigma_k$ after observing the decision set $\mathcal{D}_k$, the reason why a counter algorithm cooperating with the adversary incurs only $O(d)$ regret is that the construction allows the adversary to assign zero variance in certain rounds. What if we restrict the strong adversary to generate strictly positive (non-zero) variance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper provides the first high probability lower bounds on variance dependent regret for linear contextual bandits. These bounds match the known upper bounds up to logarithmic factors under varied assumptions on the adversarial strength in deciding and revealing variance sequences. In proving these novel results, the authors devise a "peeling" proof machinery which has broader implications beyond the studied problem instance.

### Strengths
The derivation of variance-dependent lower bounds is an important and challenging problem in bandits literature. Achieving near-tightness (up to logarithmic terms) under a generalized heteroscedastic setting is a significant theoretical advance.

The proposed "peeling" or "grouping" technique appears to be novel and essential in obtaining these sharper bounds, suggesting its potential applicability to other structured problems (see my questions for more discussion on the implications of this machinery).

The analysis under varied assumptions on adversarial strength adds depth and robustness to the result.

### Weaknesses
The bibliography contains several critical errors that must be corrected before publication:
 - The first entry appears to be duplicated.
- The citation for Dani (2008) is reportedly incorrect and should instead refer to the CoLT (Contextual Linear Threshold) paper.
- Several cited preprints have since been formally published (journal articles, updated conference proceedings) and must be updated to their final, official citations.

The statement of Theorem 4.1 (and many such statements) are imprecise and somewhat confusing. The authors should rigorously re-examine this statement for technical correctness and clarity.

The paper introduces a notion of "collaboration" or a specific form of an adversary. Claiming "near-tight" bounds is only meaningful if the lower bound is tight against the existing upper bounds (like SAVE or similar) under the same set of assumptions. If the adversarial definition is fundamentally different from the standard setting used for upper bounds, the claim of near-tightness seems misleading? Feel free to clarify this point for me, I am unfamiliar with this literature. 

The writing quality is generally acceptable, but the material can be opaque at times. Given that most detailed proofs are deferred to the appendix, the main body should dedicate more space to building intuitive understanding of the key proof ideas. Specifically, the high-level intuition behind the "peeling" technique needs to be expanded in the main text for the uninitiated reader.

### Questions
Proof Rigor (Lemma A.2): Prior to Lemma A.2, the argument mentions considering a specific cyclic instance where the learner visits instances in a cyclic manner. However, the resulting lower bound is claimed to hold for any bandit algorithm. Please explicitly include or reference the standard information-theoretic reduction that proves the bound derived from this specific instance generalizes to all valid bandit algorithms?

Heteroscedastic Variance Utility: Can you provide an expanded explanation of the utility of leveraging "heteroscedastic variance information". Furthermore, what key technical obstacles generally prevent obtaining variance-dependent lower bounds in bandit problems, and what was the unique aspect of the LCB problem that allowed the authors to overcome this limitation here?

Peeling Technique and Extensions: Please highlight the novelty and essential nature of this grouping/peeling proof technique. What is the fundamental mechanism that allows it to achieve tighter, variance-dependent results compared to prior methods? Can the authors speculate on how this technique might be useful for other problems or domains, particularly in terms of extending the ideas to more complex bandit style problems beyond the linear contextual case?

Broader Context: Please situate this result within the broader literature of lower bounds for bandit type problems (i.e., not just linear contextual). Does this result have broader implications based on the established hierarchy of these problems (e.g., standard MAB, stochastic contextual, LCB)?

Adversarial Bound (Collaboration): Regarding the "collaboration" notion, please clarify the specific regret bound on SAVE (or the best known algorithm in this setup) that your lower bound is being compared against under your definition of the adversary.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies linear contextual bandits with round-varying noise and asks whether the variance-aware upper bounds $\tilde{O}(d\sqrt{\sum_t \sigma_t^2})$ are actually tight. It shows that when the whole variance sequence is fixed in advance, for any sequence one can build a contextual instance on which every algorithm incurs $\tilde{\Omega}(d\sqrt{\sum_t \sigma_t^2})$ regret matching known upper bounds. For an adaptive but weak adversary it further gets a high-probability lower bound with more high-order polylog factors. Finally, it shows that for a strong adaptive adversary, such variance-dependent lower bounds are impossible, giving a comprehensive understanding of second-order bound for linear contextual bandit.

### Strengths
1. This paper closes a theoretical gap by deriving a new variance-dependent lower bound that matches previous upper bounds. This improves previous lower bound that depends on total-variance budget and suboptimal $d$ factor for linear contextual bandit.

2. This paper is technically neat, the “variance-level peeling + orthogonal subproblems” construction is clean and reusable; the high-probability version for adaptive variance is nontrivial.

3. The paper gives a comprehensive discussion for the separation of weak vs. strong adaptive adversaries and shows why the strong adversary cannot admit such lower bounds.

### Weaknesses
I did not see major weakness. The current lower bound relies on being able to assign (almost) orthogonal decision sets to different variance groups. That’s fine for an information-theoretic contextual lower bound with adversarially chosen context, but less reflective of i.i.d. contexts. Is it possible to improve the lower bound for linear contextual bandit with stochastic context?

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work studies linear contextual bandits with changing decision sets and aims to characterize optimal variance-based regret rates, depending on the sequence of reward variances. Previous lower bounds were shown in terms of a fixed total variance budget, and had a mismatch in the dimension dependence. This paper strengthens the lower bound in these regards for various modeling assumptions on the adversary.

### Strengths
* The paper refines known lower bounds in linear contextual bandits in terms of dimension dependence and generalized variance sequences, which requires novelties in the analysis such as a refined peelilng technique and separate analyses of different variance values.

### Weaknesses
* The presentation seems lacking in motivation. am not sure what the real world applicability of the various adversary (prefixed vs. weak vs. strong) scenarios are and there could also be more discussion of whether such refined variance-sequence based lower or upper regret bounds are known for the simpler finite-armed bandit case.
* There are also no experiments empirically demonstrating the regret bounds exhibit such tight dependences.
* I also find the presentation a bit confusing or lacking rigor in that, at times, especially Theorem 5.4. The writing at times (e.g., Remark 3.1) suggests the adversary may cooperate with the learner, but I'm not sure how to interpret this as "adversary" in bandits typically represents a worst-case environment or environment-generating process. I think this may be due to lack of clarity on the order of quantifiers in theorem statements. Perhaps a diagram or table would make it easier to understand. The utility of Theorem 5.4 is unclear as statements of the kind "there exists an adversary" implies such that there is an algorithm that can get small regret seem unsurprising. It is more useful to understand what is the worst-case behavior here, or is the strong adversary somehow weaker than the prefixed one? 
* Because of the wording of Theorem 5.4, it's unclear whether the statement "it's impossible to derive a variance-dependent lower bound if the adversary can determine the variance after observing the decision set" really follows as Theorem 5.4 concerns a _specific_ adversary and not the worst-case one.
* Furthermore, the problem is quite complex as the decision sets are also being generated adversarailly (as it adaptive or oblivious?) and the adversary may decide the variance before or after the decision set which seems to affect the bounds. In particular, some intuition on why the weak and strong adversary leads to different rates would be helpful.
* The scope of technical novelties of this work also seem limited, and it's a bit unsurprising that general variance sequence-based bounds can be derived using a grouping/peeling argment. Can the authors elaborate on difficulties of the general function approximation setting or discuss wider applicability of techniques introduced here?

### Questions
Please see weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2
