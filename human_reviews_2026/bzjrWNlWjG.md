# The Alignment Bottleneck

- Decision: Reject
- Scores: 2, 6, 6, 2, 4

## Abstract
We study feedback alignment for large language models under a finite information budget. The feedback loop is modeled as a two-stage channel $U \to H \to Y$ given context $S$, where $U$ is the target, $H$ is the bounded judgment, and $Y$ is the label. The average capacity $\bar C_{\mathrm{tot}\mid S}$ of this channel constitutes an alignment bottleneck. By applying Fano's inequality to separable codebooks, we derive a minimax lower bound on alignment error that depends on value complexity $\log M$ and capacity but is independent of dataset size. This implies that scaling data cannot eliminate error when the feedback channel is structurally deficient. We further show that the same capacity term controls the environmental budget in a PAC-Bayes generalization bound. These results define a performance interval where optimization beyond the channel capacity fits rater artifacts such as sycophancy. Experiments with Qwen confirm that low-capacity feedback leads to saturation and degradation even as data scales. Our framework suggests that improving alignment requires increasing the channel capacity through richer interfaces or clearer constitutions rather than just collecting more data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper analyzes the AI alignment problem from an information-bottleneck perspective. The process is modeled as a two-stage cascade, modeling the users’ true preferences ($U$), the noisy information they emit ($Y$) through and intermediate stage ($H$), and the context ($S$). The authors present a Fano risk lower bound under mixture assumptions, a PAC-Bayes upper bound, and discuss the theoretical implications of the results.

### Strengths
* Interesting perspective on a timely and well-motivated topic.
* Analysis provides both lower and upper bounds.

### Weaknesses
* Significance of results is unclear, and practical implications are not explicitly discussed.
* Presentation is unclear, and the paper was slightly hard to follow.
* No empirical validation of findings.

### Questions
* What are the channel capacities expected to appear in practice?
* What are the practical implications of the presented results, and how may they inform the design of AI systems in practice?
* What are the limitations of the analysis? When is it expected to hold "tightly", and when are assumptions expected to break?

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
The paper models human--AI feedback as a two-stage, capacity-limited channel (U - H - Y) conditioned on context \(S\). It derives (i) a data-size--independent Fano-style lower bound on true risk using separable codebooks, and (ii) a PAC--Bayes upper bound whose KL term is explicitly controlled by the same channel capacity via $C_{tot|S}$ Under a canonical observable loss and matched codebook mixture, these yield a two-sided ``Alignment Performance Interval,'' implying that more labels alone cannot beat the lower wall; required capacity scales with value complexity ${\log}M$; and over-optimization fits residual channel regularities (e.g., sycophancy or reward hacking).

### Strengths
1 Clear, simple formalization of the human loop. The cascade $U-H-Y$ with cognitive and articulation capacities is crisply defined and linked to information-bottleneck/rate--distortion intuitions; a central proposition bounds  $I(U;Y|S)$ by the average total capacity.

2. The key novelty is turning the KL complexity in PAC--Bayes into an environmental budget, aligning the ceiling with the same capacity that drives the Fano floor---an elegant, unifying perspective.

3. The consequences of the paper are (i) a lower bound independent of dataset size, (ii) necessary capacity scaling with  ${log}M$, and (iii) a mechanism for channel overfitting---translate theory into design levers (measure/allocate capacity, manage value complexity, regularize residual information).

### Weaknesses
No empirical validation: The theory is compelling, but there is no empirical study (even toy) quantifying $C_{tot|s}$ or demonstrating the predicted saturation/overfitting behavior under controlled capacity budgets across alignment protocols.

### Questions
My only question is regarding the empirical evaluations. Do the authors have any plans to get any empirical results?

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
This paper argues that aligning LLMs with human values is limited by human cognitive capacity. The authors treat human feedback as a bounded-information channel and show that there is a hard lower bound on how well alignment can work when human judgment is limited. They combine classical information-theoretic tools (Fano bounds) and PAC-Bayes theory to prove that even with more data, alignment performance cannot surpass this capacity bottleneck.

### Strengths
The paper is mathematically rigorous and formally couple a Fano lower bound with a PAC-Bayes upper bound using the same human-feedback capacity term, yielding an alignment performance interval that clarifies when and why increasing data does not improve alignment.

### Weaknesses
-Related prior work around the connection between bounded rationality and alignment is missing. For instance, see Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time (arXiv:2505.23729), which appears to have explicitly connect alignment and bounded rationality. Incorporating this reference would help contextualize your contribution and clarify how your information-theoretic perspective differs from and extends that prior framing.

- But as such, the connection made between bounded rationality and information theory seems forced, without much discussion or evidence. While the theoretical analysis is well done, the need for it and the key rationale behind studying this problem is not clear. 

- The authors immediately went into formulating feedback loop as two stage cascade framework, but how it would help or what problem it is aiming to solve exactly is unclear. 

- In problem setup, what is the exact bottleneck of alignment is not defined, or clear from the discussion. 

- Is the analysis presented in the paper is for parametrized settings or it does not matter, a discussion would help? 

- Would the analysis of this work could shed some light or guidance on the type of feedback one should use for alignment, such as preference feedback?  Is it optimal? 

- There are no empirical results in the paper. I understand that the work in theoretical in nature, but is it possible to provide some basic experiments to may be just connect what is the lower bound, and how we are doing currently with the available methods to motivate the importance of these lower bounds. 

Note: I will rely on other reviewers to comment on the mathematical novelty of this work, since I am not an expert in information theory.

### Questions
Please refer to the discussion in weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper argues that the primary alignment problems in large language models come from a fundamental information bottleneck between humans and models treating human feedback as passing through a limited channel with a fixed capacity.  The paper builds this argument using simple information-theoretic tools (like Fano’s and PAC–Bayes bounds) to show both lower and upper performance limits that depend on this same “human capacity.”

### Strengths
- The paper provides an interesting way to connect cognitive science and information theory to alignment in a clean, mathematical way.
- Gives a natural explanation for problems like reward hacking which happens when models overfit beyond what the human feedback can represent.
- Highlights that just collecting more data alone won’t fix alignment unless human feedback capacity increases.

### Weaknesses
- The analysis and the conception is interesting however is not directly measurable or testable in real LLM pipelines.
- The analysis builds on standard tools of information theory primarily under the constant “capacity” for humans assumption. However, the connection is weak and is not clear what are the key new aspects coming out from the analysis? This makes it hard to evaluate the paper
- Can the authors provide empirical demonstration - even a toy example of the hypothesis? Also, explaining what are the key-terms that are new or novel to the LLM/Agent Alignment paradigm? Even connecting with relevant papers and showing which term in the bound is new or provide some insights is crucial.

Note : I am open to update my view after understanding the key new aspects of this understanding.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
In this paper, the author presents a theoretical framework explaining why feedback-based alignment methods for LLMs lead to failures like sycophancy and reward hacking, despite scaling. The central idea is that the human feedback loop is a resource-limited information channel, imposing a fundamental alignment bottleneck. The paper models this process as a two-stage cascade: $U \rightarrow H \rightarrow Y \text{ given } S$ (Value $\rightarrow$ Judgment $\rightarrow$ Feedback). This channel is limited by a finite cognitive capacity ($\overline{C}_{\text{tot}|S}$) due to bounded rationality. The authors argue that this single-capacity coupling leads to key implications: 1. Scaling dataset size alone is insufficient to overcome the bottleneck. 2. Aligning on more complex values requires a corresponding increase in channel capacity. 3. Once this capacity is saturated, a powerful optimizer will fit the rater's biases, providing a theoretical explanation for reward hacking and sycophancy.

### Strengths
1. Motivated by concepts from bounded rationality, the author provides a principled framework that reframes alignment failure as an information-channel limit.

2. In the Alignment Performance Interval bound, the coupling of the Fano lower bound and the PAC-Bayes upper bound using the single channel capacity term ($\overline{C}_{\text{tot}|S}$) is interesting.

### Weaknesses
1. The paper is motivated entirely by theoretical analysis and does not present experiments to validate the theoretical claims. A synthetic experiment to support the claims would make the paper stronger.

2. The framework primarily relies on the average total capacity, $\overline{C}_{\text{tot}|S}$. However, the paper provides no discussion on how one might estimate or measure this quantity in a real-world setting.

### Questions
Please refer weaknesses

### Soundness
3

### Presentation
2

### Contribution
3
