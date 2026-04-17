# Primal Optimism in Online Optimization

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
We consider the classic online convex optimization problem in which an algorithm outputs vectors $z_t$ in response to vectors $g_t$. Our algorithm seeks to improve the regret when it has access to a sequence of "hint" vectors $v_t$ that estimate the location of the final optimal parameter value $u$. Specifically, we provide an online linear optimization algorithm that guarantees regret $R_T(u)=\sum_{t=1}^T \langle g_t, z_t -u\rangle \le \sqrt{\sum_{t=1}^T \|g_t\|^2\|v_t-u\|^2}$ for any comparison point $u$ and any sequence of vectors $v_t$, so long as $v_t$ is available before we commit to $z_{t+1}$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents an approach for low regret when the optimal point for an online convex optimisation can be accurately predicted.  The main contribution is the theoretical worst-case analysis of this algorithm.

### Strengths
The paper is well written, and as far as I can tell (I did not check every part of the maths) the analysis is sound.

### Weaknesses
The justification by reference to ML training made in the introduction is largely spurious - neural net training is non-convex, so none of the analysis presented applies.   The focus on worst-case analysis also weakens the contribution.  More generally, analysis of online convex learning algorithms is a mature area and the present paper feels like only a relatively minor addition to that literature, in my view not really enough for a flagship conference like ICLR.

### Questions
See comments re weaknesses above.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents an algorithm for online convex optimization with hints for the decision vector. This differs from most existing work that output hints for the gradient, i.e. optimism. This develops an algorithm for this setting that has adaptive regret bounds in the sense that the regret is bounded by $\sqrt{\sum_{t=1}^T || g_t ||^2 || v_t - u ||}$, where $g_t$ are the cost vectors, $v_t$ are the hints for the decision vector, and $u$ is the comparator.

### Strengths
1. It appears to be a new approach to consider hints for the decision vector ("primal optimism"). This problem seems to be a nice counterpart to hints on the gradients. Furthermore, it is well-motivated by its relevance for the analysis of strongly-convex stochastic optimization.
2. The paper presents a very clear exposition of the proposed approach and its analysis. I found it insightful and quite easy to follow.
3. The paper clearly discusses how the work relates to existing approaches in the online optimization and stochastic optimization literatures.

### Weaknesses
1. The approach appears to be a fairly straightforward application of existing techniques, although I do think that it is valuable to bring these techniques together for a new problem setting.

### Questions
1. In the first line of the abstract, it is not quite clear to say: "classic online convex optimization problem in which an algorithm outputs vectors $z_t$ in response to vectors $g_t$." Its somewhat confusing because the $g_t$ are not revealed until after $z_t$ are chosen so its not really "in response to".
2. On line 122, there is the notation $H^\star$, which is not defined. It seems that it's supposed to be $H$.

### Soundness
3

### Presentation
4

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
The paper studies the topic of online convex optimization and theoretically analyzes the algorithm's regret in this setting. They consider a black-box unconstrained optimizer with known regret bounds and utilize that to achieve primal optimistic regret guarantees, where hints are given for the optimal parameter and are possibly changing with time.

### Strengths
Originality:
The paper studies optimistic optimization from a new perspective.

Quality:
Many parts of the submission appear technically correct.

Clarity:
The general structure of the text is clear.

Significance:
Theoretical novel findings are present in the form of regret guarantees, which are improved with respect to the existing literature.

### Weaknesses
I am leaning towards rejection. Below are my reasons.

1. Experiments are missing as a whole.
2. The results are promising. However, the challenges involved are not clear. The direction taken seems rather straightforward in this field of research. Please clarify exactly why primal optimism has not been studied like this before. Is it a motivation issue? If not, what is the exact novelty in your approach that helped you achieve primal optimism unlike the others before.

3. Certain parts of the paper have skipped some explanations, resulting in a lack of rigor, see "Questions" for examples.
4. A substantial number of writing mistakes (such as typos) are present, see "Questions" for examples.

### Questions
Questions:

Page 6 Corollary 1: it is not justified how the simple $D$ terms are eliminated in the guarantee as opposed to Theorem 2. How exactly?

Page 7 Line 327: where did this objective come from?

Page 7 Line 371: should $K$ be $K_t$ instead? and the argument of regret in RHS should be additive inverse?

Page 8 Equation 8: it is not clear how (8) is used. How exactly?


Suggestions:

Page 3 Line 125: clarify if $v_t$ is revealed before $z_t$ or $z_{t+1}$.

Page 6 Line 323: sum should start at index $0$.

Page 8 Line 400: explicitly explain how the log terms are generated.


Minor Comments:

Page 3 Line 110: correct the grammar in this paragraph.

Page 6 Algorithm 1: correct typos, e.g., $v_9$ in Line 272.

Page 6 Theorem 2: correct typos.

### Soundness
2

### Presentation
2

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
This paper studies the online convex optimization problem and proposes methods that incorporate optimism with respect to the comparator. Based on existing parameter-free online learning methods, the paper provides a reduction to achieve regret bounds that can be tight when the gap between the comparator and the optimistic sequence is small.

### Strengths
- The idea of introducing optimism in the primal space is interesting. The authors demonstrate its usefulness in stochastic optimization with strongly convex functions, showing that the proposed method can adapt to both convex and Lipschitz as well as strongly convex and Lipschitz settings.

- The proposed approach benefits from a gradient-based update and avoids the need to run multiple algorithms in parallel, as required by methods such as MetaGrad (Van Erven & Koolen, 2016) and Müller’s algorithm (Wang et al., 2020).

### Weaknesses
- My main concern lies in the technical novelty of the paper. The proposed method and analysis (e.g., the decomposition in Section 3.1) appear quite similar to Algorithm 6 in Cutkosky and Orabona (2018). The primary difference seems to be that the paper chooses $v_t$ as the learner’s prediction and generalizes it to a more flexible form. However, it is not clear how challenging this generalization actually is, or how much additional insight it brings beyond the previous work.


- Hidden logarithmic factors: The use of big-$O$ and $\leq$ notation obscures logarithmic dependencies, which makes comparisons with prior work potentially misleading. For example, in stochastic strongly convex optimization, the optimal convergence rate is $O(1/T)$ without logarithmic factors (Cutkosky 2019). However, since the proposed method builds upon a parameter-free base algorithm, additional logarithmic factors are introduced, resulting in a rate that is not optimal. I strongly recommend that the authors make the dependence on $\log T$ explicit in the paper.


- Inaccurate claims and unclear statements:
  - Line 58: In Rakhlin and Sridharan (2013), the proposed bound does not adapt to the comparator; rather, it scales with the diameter of the comparator space.
  - Line 86: In Van Erven & Koolen (2016), when $v_t = z_t$, the regret bound they obtain is actually stronger than the one in this paper, as it takes the form $\sqrt{\sum\_{t=1}^T (g_t^\top (z_t - u))^2}.$

### Questions
- Could the authors clearly highlight the main technical challenges and contributions beyond Cutkosky and Orabona (2018).

- What is the significance of introducing the primal hint? In particular, for stochastic strongly convex optimization, the obtained results are not optimal up to logarithmic factors in  $T$. It appears difficult to eliminate these $\log T$ terms due to the reliance on a parameter-free base algorithm.

### Soundness
2

### Presentation
2

### Contribution
2
