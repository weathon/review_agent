# Learning-Augmented Moment Estimation on Time-Decay Models

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Motivated by the prevalence and success of machine learning, a line of recent work has studied learning-augmented algorithms in the streaming model. These results have shown that for natural and practical oracles implemented with machine learning models, we can obtain streaming algorithms with improved space efficiency that are otherwise provably impossible. On the other hand, our understanding is much more limited for the sliding window model, which captures applications where either recent data leads to better or older data must be expunged from the dataset, e.g., by privacy regulation laws. In this paper, we utilize an oracle for the heavy-hitters of datasets to give learning-augmented algorithms for a number of fundamental problems in the sliding window model, such as norm/moment estimation, frequency estimation, cascaded norms, and rectangular moment estimation. We complement our theoretical results with a number of empirical evaluations that demonstrate the practical efficiency of our algorithms on real and synthetic datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies learning-augmented streaming algorithms under time-decay weighting, including polynomial decay, exponential decay, and sliding-window streams. In this model, the streaming algorithm is allowed to access a suffix-compatible heavy-hitter oracle which can predict the heavy hitters of suffix streams.

The paper extends the framework of Jiang et al. (2020), which used a learned heavy-hitter oracle to improve space bounds for frequency moment estimation, to the more general time-decay setting. The authors propose a smoothness-based reduction that converts learning-augmented streaming algorithm into a learning-augmented time-decay algorithm with nearly the same asymptotic space complexity. Then they obtain: 

1. $F_p$ estimation: using $\tilde{O}(n^{1/2-1/p}/\epsilon^{4+p})$ space, matching known lower bound in the exponent $n$.
2. Rectangular $F_p$ frequency and cascaded norms: analogous extensions of Jiang et al. (2020) to time-decay models.

### Strengths
1. It is well-motivated to study the learning-augmented streaming under time-decay weighting.
2. The paper proposes a unified reduction framework from streaming to time-decay models, and obtains near optimal algorithm for fundamental frequency estimation problems.

### Weaknesses
1. Major concern: the paper assumes the existence of a suffix-compatible heavy-hitter oracle but provides few details in the main context on how such an oracle is learned or implemented in practice. It is unclear whether the oracle requires prior knowledge of the stream length $m$, whether it learns online or offline, and what access it has to the stream data.
2. Minor issues: 
   1. The citation command \citep and \citet are used inpropritory at times, e.g. line 90.
   2. In the equation of line 197, "$i_{t}=i$" => "$i_{t'}=i$".

### Questions
Can the suffix-compatible oracle assumption be relaxed, e.g. learned online during streaming (without access to full suffixes and without prior knowledge of the stream length)?

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
*I reviewed a previous version of this paper for NeurIPS'25 where I had some relatively big concerns about the technical novelty of the paper, as well as some concerns about the proof of the main theorem (which is four lines in the appendix and very sketchy, and not entirely easy to verify). Unfortunately, it seems to me that these concerns are still largely unaddressed. In addition, my (more minor) comments on parts of the paper that were mathematically unclear or imprecise have not been addressed either.  Below is my previous review of the paper with minor updates:*

The paper falls in the line of research of algorithms with predictions where algorithms are equipped with advice or predictions. Given recent developments in machine learning it is often too pessimistic to run a worst-case algorithm and not leverage predictable structures of the input. The line of research seeks to bridge algorithms and ML by obtaining provably improved algorithms with accurate predictions while also maintaining worst case guarantees.

Specifically, the paper considers $p$-moment estimation as well as sketching rectangle $F_p$ frequencies and cascaded norms in time-decay models, e.g., the sliding window model. There is an arriving stream of elements over a universe of size $n$ and a corresponding frequency vector $x$, and the goal is to maintain a small sketch of $x$ which captures important statistics of $x$ -- in this paper it's $\ell_p$ moments for $p\geq 2$. In the general time-decay model, there is a weight function $w$ that at any point of time $t$ weights all arrivals of each element $i$ based on how far back these arrivals occurred. The frequency of $i$ at time $t$ is the sum of these weights, and the idea is that elements that arrived very far back in time are less relevant and can get weighted less.
As a special case, in the sliding window model, there is a window of the last previous $W$ updates, and instead of maintaining a sketch of the full stream, we at any point of time want to be able to approximate the moments of the frequency vector corresponding to the last $W$ updates. The paper considers the well studied prediction model introduced in [HIKV'19] where the algorithm has access to a heavy-hitter oracle which can predict whether an arriving element has frequency above a certain threshold  (which depend on the problem at hand). For moment estimation, it is well-known that standard learning augmented algorithms can achieve $\tilde O(n^{1/2-1/p})$ space while approximating  the frequency moments, which improved upon the bound of $\tilde O(n^{1-1/2p})$ in the classic model. The paper proves similar bounds in time-decay, e.g., the sliding window model (but with worse exponential dependencies on $p$ and the approximation parameter $\varepsilon$).

One update that has been made to the paper since I last reviewed it is a framework for general time-decay models  which requires a generalization of the idea of $(\alpha,\beta)$-smooth functions from Braverman and Ostrovsky [BOO7].  Albeit the proofs in the appendix are somewhat sketchy, this is a nice addition to the paper since my last review.

### Strengths
Moment estimation is an important problem sketching problem and I find it well-motivated in the learning augmented-framework. It is well-studied in several past works. The sliding window model is also natural since it captures the idea that we may often be more interested in statistics of the most recent data. I like specifically about the paper, that they generalize the framework of Braverman and Ostrovsky [BOO7] to more general time-decay functions.

### Weaknesses
(1) I found that the paper lacked technical novelty. It heavily relies on the smooth histogram algorithm from Braverman and Ostrovsky [BOO7] which turns a classic sketching algorithm into an algorithm in the sliding window model with little overhead. The paper shows that the learning augmented algorithm from [JLL+20] fits the framework from [BO07] in a white-box fashion, they argue that the sliding-window algorithm from [BOO7] retains two properties captured in Proposition 1 and 2 even for the learning-augmented versions. The proof of this fact is 4 lines in the appendix (and it is quite sketchy, so additionally it was unclear to me why it is actually true). It seems to me that the rest of the paper just applies known bounds and techniques from past work. This main issue with the paper seems to be similar for $F_p$ frequencies and cascaded norms estimation, although I haven't read those parts of the appendix in detail.
(2) While the paper does obtain a good polynomial dependence on $n$, the dependence on $\varepsilon,p$ seem to become quite bad.
(3) Parts of the paper is poorly written and it is hard to follow precisely what the authors are claiming. I will give details below.

### Questions
*Mainly copied from my NeurIPS review.*

l98-l99: Can such formal guarantees be given? What are the challenges in proving such formal guarantees. This seems relevant.

l267-l269: The notation $x_A$ is not properly defined.

l278-l279:  This property is not clear when frequencies of stream elements can be negative. I would have liked a more precise discussion of which model you consider.

Algorithm 1: Notation like $ALG^k\geq something$ is not really meaningful, and I suppose it refers to the output of the algorithm. Also, the algorithm doesn't describe how it returns estimates, e.g. of a moment.

l781-786: This is (still) very sketchy and I don't know what sentences like "as long as we could maintain the smooth histogram" and "... every copy of the algorithm can get the oracle advice it needs,..." means in a formal sense. While you mention it, it is also (still) not clear to me how suffix-compatibility is actually used. This is essentially the proof of the main result of the paper, and needs to be written in a way that can be verified.

Theorem 3 is for general decay models, but when you prove the following many theorems on page 20, it is unclear how you use polynomial and exponential decay.

### Soundness
3

### Presentation
2

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
This paper presents a learning-augmented framework for frequency moment estimation, applicable to both sliding-window and time-decay models. The authors design and employ a suffix-compatible heavy-hitter oracle tailored for time-decay settings. In the sliding-window case, the proposed algorithm achieves a space complexity that even meets the lower bound for streaming algorithms. Beyond sliding windows, the paper studies the more general time-decay model and provides rigorous space complexity guarantees. Experiments on both real-world and synthetic datasets demonstrate the practical efficiency of the proposed approach.

### Strengths
- The extension of learning-augmented frequency moment estimation to general time-decay settings is novel and appealing.
- The paper provides space complexity bounds for several common estimation tasks—including $F_p$ moments, rectangular moments, and cascaded norms—under two decay models, filling an existing theoretical gap.
- The experiments convincingly show that the learning-augmented approach significantly improves estimation accuracy under the sliding-window model.

### Weaknesses
- The experimental evaluation does not include direct comparisons with other learning-augmented sliding-window algorithms, such as Shahout et al. (2024), and lacks empirical validation in more general time-decay environments. 
- The discussion on training suffix-compatible heavy-hitter oracles is insufficient. Although the PAC learning framework and Theorem 14 address the learnability of general heavy-hitter oracles, the paper does not examine specific training strategies or conditions that guarantee the suffix-compatible property, which is crucial for ensuring correctness in time-decay models.

### Questions
- The proposed algorithms depend on the suffix-compatible property of the oracle. In practice, if the oracle's predictions are inaccurate, could this undermine the suffix-compatible property and compromise the guarantees on space complexity?
- As shown in Figs. 2(c) and 4(c), increasing the sampling probability of SSA yields little improvement in accuracy. What could explain this limited effect?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper brings machine-learning hints to streaming algorithms that forget old data. It targets sliding windows and decay models where recent items count more. With a predictor that flags large coordinates, the authors obtain accurate frequency and norm estimates using memory that grows slowly with data size and error tolerance. Theory shows the method is close to the best possible, and tests on internet traffic traces cut error by twenty to fifty percent versus standard sketches.

### Strengths
1) Novel and timely problem formulation: First learning-augmented guarantees for time-decay streams. Captures privacy-driven data deletion (GDPR) and recency-weighted analytics, motivating practical relevance.

2) Theoretically strong results: General reduction shows any ($\alpha$, $\beta$)-smooth function enjoys a black-box sliding-window lift while preserving approximation and randomised guarantees; rectangle and cascaded norms benefit for free

3) Experimental validation on real data: CAIDA traces (30 M IPs) show AMSA (augmented AMS) keeps relative error ≤1.2× across window sizes while AMS drifts up to 2.3×. Distribution-shift experiment on synthetic data demonstrates >2× error gap between SSA and SS when stream distribution changes, confirming robustness claim. Code released aiding reproducibility.

### Weaknesses
1) Limited empirical scope: Only l2 and l3 norms are evaluated; rectangle and cascaded-norm algorithms lack any implementation or micro-benchmark, leaving practical impact uncertain.

2) Expand empirical coverage: Include CPU-time and peak RAM tables for each dataset to confirm the claimed overhead.

3) Strengthen statistical reporting: Release full parameter files (hash seeds, repetition counts, bucket sizes) to facilitate exact reproduction.

### Questions
See above weakness.

### Soundness
3

### Presentation
3

### Contribution
3
