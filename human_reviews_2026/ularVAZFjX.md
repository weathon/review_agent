# A Pitfall in Conformal Prediction:  When Shorter Intervals Are Not Better

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 0, 4, 2

## Abstract
Conformal prediction has become a cornerstone of distribution-free uncertainty
quantification, conventionally evaluated by its coverage and interval length. This
work critically examines the sufficiency of these standard metrics. We demon-
strate that the interval length might be deceptively improved through a counter-
intuitive approach termed Prejudicial Trick (PT), while the coverage remains
valid. Specifically, for any given test sample, PT probabilistically returns an inter-
val, which is either null or constructed using an adjusted confidence level, thereby
preserving marginal coverage. While PT potentially yields a deceptively lower
interval length, it introduces practical vulnerabilities: the same input can yield
completely different prediction intervals across repeated runs of the algorithm.
We formally derive the conditions under which PT achieves these misleading improvements and provide extensive empirical evidence across various regression
and classification tasks. Furthermore, we introduce a new metric interval stability which helps detect whether a new conformal prediction method implicitly
improves the length based on such PT-like techniques.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper is clearly written and easy to follow. 

Problem context: Conventionally, CP methods are evaluated by its coverage and interval length. This work argues the sufficiency of these standard metrics.

Paper's proposal: This paper introduces a mechanism called the "Prejudicial Trick" (PT) to demonstrate a supposed "pitfall" in the standard evaluation of conformal prediction (CP) methods. The authors claim that PT can "hack" the conventional coverage-length metric by probabilistically returning a null set (length 0) or a wider interval, thereby preserving marginal coverage while deceptively reducing the average interval length. The authors then argue this reveals a flaw in the standard metrics, as PT introduces practical instability (a random output for a fixed input). They propose a new metric, "Interval Stability", defined as the expected variance of the interval length, to detect this "vacuous randomness." 

Empirical results across numerous regression and classification datasets, using different base CP algorithms (VCP and CQR) , confirm that PT can deceptively improve interval length while the proposed "Interval Stability" metric successfully identifies the trick.

### Strengths
1. This paper makes a conceptual contribution. The CP community relies heavily on the coverage-length trade-off as the primary method for evaluation. This work demonstrates that these two metrics are insufficient for capturing the practical utility of a CP method. 

2. The Prejudicial Trick (PT) is simple and elegant. 

3. The theoretical analysis, such as the proofs that PT preserves marginal coverage (Theorem 4) and the conditions under which it reduces length (Theorems 7, 8, 11), appear to be mathematically sound.

4. The proposed Interval Stability metric is intuitive, simple to compute, and (as shown empirically) diagnoses the issue of "vacuous randomness" introduced by PT.

### Weaknesses
1. The title “A Pitfall in Conformal Prediction” strongly implies that the authors have identified a fundamental weakness in the conformal prediction framework. In reality, the pitfall lies entirely in the choice of evaluation metric, not in conformal prediction itself. The proposed “PT”  predictor is not a conformal method; it is an external randomization layer applied after conformal intervals have been constructed. Therefore, the phenomenon described is not a failure of conformal prediction, but a property of an artificially randomized post-processing step.

2. The randomization mechanism used in PT is well-known probability argument. 

3. For all standard, deterministic CP methods (VCP, CQR, etc.), the metric "Interval Stability" will be identically zero. Thus, the metric is only useful in detecting authors' PT trick. The work does not identify a scenario where such behavior might arise in standard CP outputs.

4. The paper mentions (in Remark 6) that "as methods become increasingly complex, they may implicitly utilize similar randomness to improve the length". The authors do not provide any legitimate, complex, published CP method suffers from this supposed "implicit randomness". 

While the paper is clearly written and motivated by an interesting observation about the variability of conformal intervals, the contribution remains conceptually and practically limited. The reported instability arises entirely from an artificial, externally randomized post-processing step (the “PT” predictor), which is not itself conformal. The randomization mechanism is a standard probability trick and does not constitute methodological innovation. Moreover, the proposed “interval stability” metric is only nontrivial for such randomized constructions and is identically zero for all standard deterministic CP methods (e.g., VCP, CQR, split CP). Therefore, the metric lacks general practical value. These limitations of the paper lead to a low score.

### Questions
1. The title suggests a fundamental ``pitfall'' in conformal prediction, yet the instability arises only from an externally randomized post-processing (PT). Can you clarify why this should be viewed as a limitation of conformal prediction rather than of the PT randomization itself?

2. For standard deterministic conformal methods (e.g., split CP, CQR, VCP), the interval stability metric is identically zero. In what realistic settings do you expect nonzero instability to occur without deliberately injecting randomness?

3. With additional interval stability metric, what should practitioners aim for: target coverage, with small length, and zero interval stability? What is the advantage of this new metric in practice?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper identifies a weakness in the standard evaluation of CP, arguing that the two most common metrics (marginal coverage and average interval length) are insufficient for a robust evaluation. 
The authors introduce a pathological algorithm PT, which, by construction, maintains (or exceeds) the desired marginal coverage. 
PT strategically assigns either a null set or a wider-than-necessary interval to different data points based on their underlying properties, allowing the algorithm to achieve deceptively short interval lengths, even though it fails to provide meaningful uncertainty quantification for a subset of the data. 
The authors argue that this pitfall underscores the need to move beyond marginal coverage and evaluate methods based on conditional coverage.

### Strengths
1. The paper is clear and easy to follow, and PT is a simple, well-designed, and intuitive counterexample. 
2. The authors identify a weakness in the common practice of optimizing for average interval length, demonstrating how this objective can be tricked.

### Weaknesses
1. The paper's primary conclusion, that marginal coverage is insufficient and conditional coverage is the more desirable property, is not a new insight. Conditional coverage has been a big area of research in CP for many years. Prior work has extensively discussed the limitations of marginal coverage and proposed numerous methods towards better conditional coverage.
2. The paper does not offer a practical, novel method or a new solution to a practical problem. Specifically, the paper argues against using common CP evaluation metrics, but it fails to propose a concrete alternative. It neither offers a new method to achieve it nor proposes a new, practical evaluation metric that could replace average length for comparing methods.
3. The authors do not provide any evidence that widely used methods and datasets suffer from this pitfall - the paper's significance is primarily pedagogical. The experiments are essentially synthetic. An empirical study where conventional CP methods have this pitfall would significantly enhance the persuasiveness of the work.
4. The efficiency gains from using PT-VCP are quite minimal and not convincing (table 2)

### Questions
1. Can you elaborate on your core contribution in the context of the existing literature that already advocates for conditional coverage? What does your paper add for someone who is already convinced that marginal coverage is insufficient and that we should move towards conditional coverage?
2. Are there any existing, non-adversarial CP methods (like CQR or split) or other adaptive methods that fall into this trap on any real datasets?

Addressing these concerns in-text will significantly enhance the paper's impact and quality.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a theoretically and empirically supported perspective that interval width and coverage alone are insufficient criteria for comparing prediction intervals. The authors argue that an additional metric---\emph{interval stability}, defined as the variance of the interval length---should also be considered, as it captures how much the interval fluctuates across repeated samples. This is an important point, given that interval width and coverage are typically the primary metrics used to evaluate predictive intervals.

To motivate this claim, the authors introduce a simple construction---the ``Prejudicial Trick''---which can be applied to any conformal predictor to produce intervals that are, in many cases, narrower while still preserving marginal coverage. The method randomizes the output: with probability $p$, it returns the empty (null) set, and with probability $1-p$, it returns an enlarged conformal interval constructed at a higher coverage level than usual. The authors show, under mild assumptions, that the expected interval width of this randomized procedure---averaged over draws of the calibration data---is strictly smaller than that of the standard conformal interval. Intuitively, the zero width of the null set reduces the average more than the enlarged interval increases it.

The paper further provides several sets of sufficient conditions under which the proposed trick yields smaller expected interval widths, as well as a counterexample illustrating when the method fails to improve width.

The theoretical results are supported by several experimental studies

### Strengths
The Prejudicial Trick is a simple, deliberately pathological construction that reduces the average length of any prediction interval without changing its marginal coverage (under suitable conditions). It demonstrates that interval length can be made artificially smaller in a misleading way: the resulting intervals are unstable and, with a fixed probability, collapse to a degenerate null set that provides no information. Although mathematically valid, the construction yields intervals that are clearly undesirable in practice. The authors substantiate this point with both theoretical analysis and empirical results.


The paper is clearly written and well structured, and the exposition effectively conveys the construction and its implications.

### Weaknesses
1. Full conformal prediction and split conformal prediction are non-randomized procedures: once the calibration data, the conformity score, and (in the split setting) the train--calibration split are fixed, the resulting prediction set is fully deterministic. By contrast, the Prejudicial Trick (PT) proposed in the paper is a \emph{randomized} construction: with probability $1-p$ it outputs a valid conformal interval, and with probability $p$ it outputs a degenerate null interval. As a result, rerunning the procedure on the same data may produce different outputs. In this sense, PT is not a conformal prediction method in the usual sense. (While there exist randomized variants of conformal prediction, the standard framework and the vast majority of methods are deterministic.)

The authors argue that PT demonstrates a fundamental limitation of evaluating intervals solely by coverage and average length. However, this claim only holds if one allows \emph{randomized} algorithms. If we restrict attention to conformal prediction methods---or, more generally, deterministic procedures---PT no longer serves as a counterexample. In that setting, it is not clear that coverage and interval length are insufficient metrics. This distinction matters, because the motivating question on page~1 asks:

 
``Can a conformal prediction method maintain valid coverage and deceptively improve interval length metrics through counter-intuitive constructions, while introducing practical risks?''
 

PT is then presented as evidence that the answer is ``yes,'' but PT is not actually a conformal prediction method, so it does not address the stated question.

 

2. More broadly, the paper would benefit from a decision-theoretic perspective. The situation is reminiscent of Hodges' superefficient estimator: it improves a standard performance metric in a pathological way, yet is inadmissible under any reasonable risk criterion. The paper argues that PT is undesirable because it introduces ``instability,'' but randomization is not inherently problematic. What is missing is a principled notion under which PT is formally suboptimal---for example, a proper scoring rule for set-valued predictions, a loss function that penalizes degenerate intervals, or a stability constraint that prevents algorithms from exploiting randomness to game marginal performance metrics. For instance, Section~6.2 of Gneiting and Raftery (2007) introduces the interval score as a proper scoring rule for prediction intervals. PT would perform very poorly under this score, since the null interval incurs a large penalty whenever the true value lies outside it.

https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf


3. On this note, the paper would benefit from a broader discussion of existing evaluation frameworks for predictive intervals. The forecasting literature is extensive, and similar issues regarding interval quality, proper scoring rules, and pathologies of evaluation metrics have been studied in depth. In particular, see  Gneiting and Raftery (2007) and references therein.


4. Finally, the proposed stability metric is always zero for deterministic methods, and therefore functions only as a measure of randomization. Since most conformal prediction algorithms are deterministic, it is unclear how actionable this metric is in practice. Moreover, if a method appears unstable solely because it is randomized, how should this be interpreted? Randomization does not automatically imply deficiency, so a low stability score does not itself diagnose a problem. The paper would be stronger if it provided guidance on how such a metric should inform methodological choice: when does instability constitute a meaningful failure, and when is it merely a benign algorithmic feature?

### Questions
The guarantees in the paper are derived marginally over draws of the calibration data. This raises a natural question: does the Prejudicial Trick retain its properties when we condition on a fixed calibration set? In particular, does it still reduce the calibration-conditional average width while preserving calibration-conditional coverage? 

As discussed in the weaknesses, can PT be shown to be suboptimal from a decision-theoretic perspective? E.g., in a minimax sense?

### Soundness
3

### Presentation
4

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
This paper focuses on the sufficiency of the standard approach in evaluating conformal prediction (CP) intervals involving two metrics: coverage and interval length. The authors introduce an adversarial approach called the "Prejudicial Trick" (PT), which yields CP intervals with deceptively lower interval lengths, but for which one input can yield significantly different prediction intervals across repeated runs of the algorithm. The main idea behind PT is to return a null prediction interval with some fixed probability and return confidence intervals with lower miscoverage rates in remaining cases. The paper derives the conditions under which PT achieves these misleading improvements and provide experimental evaluations for both regression and classification tasks.

### Strengths
- The paper is quite well-written and easy to understand.

- I like the simplicity of the PT adversarial device, which is easy to understand and explore theoretically. After demonstrating the construction of the PT trick, the authors present several necessary theoretical directions including coverage guarantees and sufficient conditions under which PT improves interval lengths. 

- In addition to devising PT, the paper also suggests a metric to counteract PT ("interval stability") which can flag the type of vacuous randomness of PT in proposed CP methods.

### Weaknesses
- The contribution here (PT) has limited practical utility nor does it seem to expose a fundamental insight about CP pushing the development of CP methods forward. Perhaps the practical implication of PT is as a warning to researchers constructing CP methods on the perils of only relying on interval length and coverage. However, if this is the case, I find the construction of a CP approach with a probabilistic component assigning null intervals to be an artificial one not grounded in practice. 

- The current experiments seem to be essentially synthetic, demonstrating a potential corner-case that in theory could happen. The paper would benefit from showing even one convincing real application where an already proposed CP approach in the literature may have to deal with this PT danger.

### Questions
1. Can you please answer how PT has practical relevance to researchers in CP?

2. Is there an actual instance (using existing CP methods) where this problem arises, or could reasonably potentially arise?

### Soundness
3

### Presentation
3

### Contribution
2
