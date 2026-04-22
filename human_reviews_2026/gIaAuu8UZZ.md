# Skirting Additive Error Barriers for Private Turnstile Streams

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
We study differentially private continual release of the number of distinct items in a turnstile stream, where items may be both inserted and deleted. A recent work of Jain, Kalemaj, Raskhodnikova, Sivakumar, and Smith (NeurIPS '23) shows that for streams of length $T$, polynomial additive error of $\Omega(T^{1/4})$ is necessary, even without any space restrictions. We show that this additive error lower bound can be circumvented if the algorithm is allowed to output estimates with both additive \emph{and multiplicative} error. We give an algorithm for the continual release of the number of distinct elements with $\text{polylog} (T)$ multiplicative and  $\text{polylog}(T)$ additive error. We also show a qualitatively similar phenomenon for estimating the $F_2$ moment of a turnstile stream, where we can obtain $1+o(1)$ multiplicative and $\text{polylog} (T)$ additive error. Both results can be achieved using polylogarithmic space whereas prior approaches use polynomial space. In the sublinear space regime, some multiplicative error is necessary even if privacy is not a consideration. We raise several open questions aimed at better understanding trade-offs between multiplicative and additive error in private continual release.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies (approximately) differentially private (DP) release of the number of distinct items in a turnstile stream, where at each step an item can be either added or removed from the active set. The stream has length $T$, the universe of items is of size $n$, and the notion of neighboring is *event-level*, meaning that the input at a single step can change. Past work showed a lower bound of $\Omega(T^{1/4})$ (Jain et al., 2023a) (event-level) and gave an algorithm running in $\tilde{O}(T^{1/3})$ (Jain et al., 2023b) (item-level), both on the additive error achieved by the algorithm. This paper circumvents the lower bound by considering mixed error guarantees which are also *multiplicative*, showing that $\mathrm{polylog}(T)$ additive error is possible if $\mathrm{polylog}(T)$ multiplicative error is tolerated. This can be contrasted with other lines of work (Jain et al., 2023a; Henzinger et al., 2024a; Cummings et al., 2025) that instead pursue guarantees parameterized in the input.

The paper also investigates the $F_2$ moment on turnstile streams, showing that any purely additive error guarantee necessarily is $\Omega(T)$, but that if constant multiplicative error is allowed then $\mathrm{polylog}(T)$ additive error is possible.

The techniques in the paper are based on sketching, where the counting problem on the turnstile stream is reduced to running many parallel instances of differentially private continual counting. Algorithm 2 builds on MinHash, Algorithm 3 (conceptually in my view, see later question) on CountSketch and Algorithm 4 on the AMS sketch for $F_2$ estimation.

### Strengths
1. The problem of (DP) counting distinct elements on a turnstile stream is easy to motivate, and a fundamental problem in DP. Given the existence of a polynomial lower bound on the additive error, studying the problem under a multiplicative guarantee is natural and well-motivated.
2. The paper is overall well-written, with the main part of the paper giving enough intuition for the results while still giving broader context.
3. The techniques/ideas used are natural, and the proofs in the appendix that I checked seemed correct (up to constants, see questions).
4. The message that polynomial additive errors can be replaced by polylogarithmic additive errors at the expense of (potentially non-trivial) multiplicative error, and sometimes in low space, is relatively clean and nice. I think this message, in combination with the Open Problems section, will lead to additional future work in this direction.
5. The subject fits the scope of ICLR.

### Weaknesses
1. A $\mathrm{polylog}(T)$ multiplicative guarantee will not always be competitive with the pure additive error guarantees parametrized in e.g., flippancy (Jain et al. (2023a)) or total flippancy (Henzinger et al. (2024a)). Many natural streams are prone to exhibit low flippancy (e.g., tracking occupancy in a store). The multiplicative guarantee can potentially offer an improvement in pathological cases where the flippancy is very high, e.g., when updates are concentrated in a few items, but it is not clear to me how realistic/interesting a setting this is.
2. The paper only considers event-level DP, rather than item-level DP. It is not clear how (or if) the techniques could be extended to work for item-level DP.
3. The techniques used in the paper, and how they compare to techniques used in past work on DP counting distinct elements on turnstile streams, are insufficiently discussed. This paper is not the first to use sketching in the context of DP more broadly, see e.g., [1,2,3,4]. [1] for example appears to solve the same problem in the setting where only the final number of distinct elements has to be released (i.e., not under continual release), and does so with a multiplicative/additive guarantee. More related still, the key building block in Epasto et al. (2023) is an implementation of CountSketch under continual release. It would strengthen the paper to discuss the extent to which its techniques differ from past work. This could mostly still be done in the appendix by extending Section A.

References:
- [1] The Flajolet-Martin Sketch Itself Preserves Differential Privacy: Private Counting with Minimal Space, Smith et al., NeurIPS‘20.
- [2] Differentially Private Linear Sketches: Efficient Implementations and Applications, Zhao et al., NeurIPS’22.
- [3] Improved Utility Analysis of Private CountSketch, Pagh & Thorup, NeurIPS’22.
- [4] Better Differentially Private Approximate Histograms and Heavy Hitters using the Misra-Gries Sketch, Lebeda & Tetek, ACM Trans. Datab. Syst.’25.

### Questions
Overall, I find the paper well-written and interesting with a meaningful contribution. I also think it invites clear follow-up work. I recommend it for acceptance, but I have some concerns regarding the novelty of the techniques used (hence my confidence score). I think the paper would benefit from, at least a brief, discussion of how the techniques employed compare to past work.

1. How novel are your techniques over past work for counting distinct elements on turnstile streams specifically? How do your techniques differ?
2. Can Algorithm 3 be viewed as continual release of CountSketch, in a “high-collision regime”? If so, how does it compare to the continual release of CountSketch employed in Epasto et al. (2023)?
3. Do you believe your techniques could be extended to work for item-level DP?
4. Not a severe issue, but I think the inequality you state on line 856 in the proof of Lemma C.1 is wrong. My understanding of the argument is that you enlarge the unit-sized interval to have size $\sqrt{l}/500$, and then union bound over ~this number of unit-sized intervals, but then the first factor on the right-hand-side should be divided by 500 rather than 50000, right? If this is an error, I think it should only impact constants in Lemma C.1, nothing in the main paper.

*Typos/comments:*
- General comment: The exact assumption on the hash/random functions involved are not stated clearly. E.g., in Algorithm 1, Step 2 it is a “random hash function” from $[n]$ to $[n]$, Lemma C.1 only specifies that $f$ and $g$ are “random functions”. From the proof of the space usage in Theorem 3.1, you state that pairwise independent hash functions are enough for the analysis in that case, and necessary for the claimed space usage, but it is not stated in the main theorem. It would benefit the clarity of the paper if the requirement on the hash functions was more explicit. Perhaps it could be stated once that e.g., "all hash functions are assumed pairwise independent and can be stored in space...", if the requirement is consistent throughout the paper.
- Line 043-044: References appear twice?
- Line 150-151: remove “to”
- Line 318-319: The interpretation of $f_t[k]$ seems wrong to me, that it is the number of elements at time $t$ whose value hash into $[2^k, 2^{k+1})$. I think this would only be true if it tracked the *most* significant bit.
- Line 393:  Step 3 of Algorithm 3 missing a “for”
- Line 444: If it *is* possible
- Line 485: Remove “for” in “algorithm for behind”. Stylistic preference “that it satisfies” -> “it satisfying”.
- Line 820: *form* the foundation..
- Line 843: [k]now
- Line 851 and 855: I think it would not hurt to add a bit of motivation for why these inequalities hold. The first seems like a loose invocation of a bound by Erdos and the second a union bound.
- Line 847: Again, it is not just any random function.
- Line 872: Large[r]

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
4

### Summary
This paper considers the differentially private estimation of the fundamental stream quantity of distinct elements (and F2 norm). The privacy (and utility) guarantee is with respect to all intermediate updates of the number of distinct elements. This is kknown as the continual release model of differential privacy. 

Prior work shows an $T^{1/3}$-additive error algorithm, where $T$ is the stream length, for continually counting the number of disinct elements in turnstile streams, where stream updates consist of elements insertions and removals. They also show a lower bound of $T^{1/4}$ on the additive error. (The problem is much simpler for insertion-only streams, where polylog T error is achievable). The authors of this paper show that by allowing mulitplicative  error, we can bypass the polynomial lower bound on the additive error. They show an algorithm with polylogT additive and mulitplicative error that also has polylogT space (the prior work used polynomial in T space). 

An open question on the optimality of the multiplicative error remains, in particular can there be algorithms with constant multiplicative error for this private estimation problem?

### Strengths
- This paper opens an interesting research direction of considering multiplicative error algorithms in the space of continual release algorithms, where strong additive error lower bounds are known. 

- Allowing for multiplicative error gives rise to algorithms with polylog space, which has been a relevant question of prior works (Jain et al 2023, Cummings et al 25).

### Weaknesses
- The algorithmic techniques are not very novel, the main results are achieved through a combination of common streaming algorithms and the differentially private technique of continual counting. 

- Without lower bounds, it is hard to say how close/far from optimal the approximation bound is. It is very open whether an algorithm with constant multiplicative error exists.

### Questions
The algorithm in Section 4 was hard to grasp without much explanation in words. Could you please provide an overview of the technical novelty for Algorithm 4 compared to existing work in the non-private literature?

### Soundness
3

### Presentation
3

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
The paper studies the problem of estimating statistics on a data stream under a general turnstile model. In this model, there is a strong lower bound that essentially rules out any algorithm with additive error that is logarithmic in the stream length. The problem is also studied in other models of streams, specifically insertion-only models, where one can achieve a polylog additive error with a small multiplicative approximation. 

The paper gives algorithms for distinct element counting and F_2 moment estimation with additive error that scales with poly-log(T) at a cost of a large multiplicative error.

### Strengths
The paper's strength lies in being the first private algorithm for distinct counting and F_2 moment estimation under the general turnstile update model. Their result also raises a general question as to what the tradeoff is between multiplicative and additive error. I also like the fact that the authors mention some open problems with proper discussion. 

In terms of idea, I do not see something new that comes up and that is alright. I do not consider it a weakness and actually consider a strength with a small bias (so mentioning it in the strengths section).

### Weaknesses
My biggest worry is that the multiplicative error and the additive error have the same scale. In particular, what the authors end up showing in the upper bound side is that they can approximate distinct elements with polylog(T) multiplicative approximation (the additive term can get subsumed if we consider that there are at least $O(1)$ distinct elements). The same goes for $F_2$ estimation. This makes me a little worried about the strength of the result, especially because in both problems, one can achieve a significantly smaller multiplicative approximation (without privacy constraints). Since the extra $\log^2(T)$ factor in space comes mainly because of continual release, we can take $\rho = O(\log^4(T))$ and get a constant approximation, but larger space than non-private variants. Ideally, one sanity check I like to make is to let the privacy parameter go to $\infty$, and we should recover the result in a non-private setting. That does not seem to happen here. Can the authors shed some light here?

### Questions
Please look at my weakness point above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper provides new algorithms for differentially-private, low-space estimators of statistics on a **turnstile** data stream, where elements may be added and removed in the stream. The paper gives algorithms with a mixed multiplicative $polylog(T)$ and additive $poly(log(T)/\epsilon)$ guarantee for counting distinct elements and $F_2$ estimation. These algorithm build on a pair of recent results, one that uses linear space to get an additive-only guarantee, and another that uses sublinear but still polynomial in T (about $T^{1/3}$) space. 

On a technical level, the paper modifies existing sketching algorithms. They obtain significantly stronger results in the `strict turnstile' setting (where deletion requests only occur for items that actually exist in the current logical dataset) than in the general setting.

### Strengths
* The paper provides a nice contribution to the literature on space-bounded private computations, addressing basic problems ($F_0$ and $F_2$ estimation). 

* The paper an open question explicitly asked by previous work of JKRSS (Jain et al, ICML 2023); it improves on [CEMMOZ] (Cummings et al., ICML 25), which explicitly addressed the same question.

* The paper appears to be technically sound, with clear high-level overviews of the algorithms and proof ideas.

### Weaknesses
* The polylog multiplicative guarantee is quite weak. The previous work of CEMMOZ (as well as a combination of JKRSS and KNW'10) achieves a $(1+\eta)$-multiplicative guarantee, and already demonstrates that sublinear additive error and space guarantees are possible. Is there any evidence that one cannot get a constant-factor multiplicative approximation along with a polylogarithmic additive error guarantee?


Significant, but not major, drawbacks: 

* The techniques in the paper are standard—they mostly consist of combining known tools in the right way. (There's nothing wrong with that, but it means that new technical tools are not a big contribution here.)

* The fit for ICLR is a bit weird. Although cardinality estimation is a basic algorithmic and statistical topic, it's unclear how relevant it is to most of the ICLR audience. Of course, ICLR is quite broad at this point.

* The comparison to previous work in Table 1 is missing a discussion of the work of [CEMMOZ]. That work provides the first sublinear-space algorithm for the problem (with a much tighter multiplicative guarantee and a much looser additive error and space guarantee than the ones in this paper). If I understood correctly, this submission points out that one could achieve a better result (that is, $(1+\eta, \tilde O(\sqrt[3]{T}))$ multiplicative/additive error in polylog space) by directly combining the JKRSS approach with an algorithm due to Kane et al. It's good to point that out, but I would still keep CEMMOZ in the discussion. (Also, it would be good to spell out the combination of JKRSS with Kane et al in a bit more detail—perhaps in the appendix—and clarify the attribution in Table 1, since the result does not appear in JKRSS.)


Minor comments: 

* The exact (event-level) privacy definition is unclear until page 3 of the paper. Claiming "differential privacy" doesn't make sense without an adjacency notion. Given that several have been studied for this problem, it makes sense to clarify the point early. 

* The theorem statements claim 1/polylog(T) probability of error. Presumably one could amplify this by running several parallel copies of the algorithm (and increasing epsilon). Why the weaker statement?

### Questions
* What is possible with item-level guarantees? (These are natural for cardinality, which is 1-sensitive to item-level changes at every time step.)

### Soundness
4

### Presentation
3

### Contribution
3
