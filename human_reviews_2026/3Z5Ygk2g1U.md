# Auditable Early Stopping for Agentic Routing: Ledger-Verified Run-Wise Certificates under Local DP

- Decision: Reject
- Scores: 2, 0, 0, 4, 2

## Abstract
\begin{abstract}
We study early stopping for best-first routing in tool-use agents under local differential privacy (LDP), with an auditable, validator-replayable ledger. Our key idea is a \emph{run-wise certificate}: we couple each node's key to the \emph{same} exponential race that realizes leaf perturbations, so the standard halting rule---stop when $\max_{v\in\mathcal{F}}\Key(v)\le B^{\ast}$, where $B^{\ast}$ is the incumbent realized leaf value---soundly certifies the realized run. We provide two certified modes on context-indexed prefix--DAGs whose children partition the leaf set. \emph{Exact} mode (known counts) implements lazy offset propagation with winner reuse; \emph{Surrogate} mode (upper bounds only) disables winner reuse and uses a parent-anchored surrogate race; keys are conservative online and can be tightened ex post by a validator via $\kappa=\log(N/\Nub)$. A small compiler enforces the partition property, and an admissible, race-independent $M_\tau$ keeps keys sound. A replayable ledger records uniforms, counts, and tie handling; privacy follows from post-processing. Experiments on synthetic graphs and a small real tool-use pipeline show tight stopping, deterministic replay, and low overhead.
\end{abstract}

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper appears to present something, but it is not entirely clear what. There is no real introduction, no motivation for the work, and little to go on for meaning unless the reader happens to have worked on exactly the same thing and uses the same terminology. Since tghe paper is written only for a very specialized small audience that is deeply enmeshed in everything described, this is not suitable for conference review in it's present form. If other reviewers feel differently, that will surface through the review process and the paper can be evaluated on its merits.

### Strengths
This paper appears to be written in a precise mathematical style.

### Weaknesses
Not understandable to a general audience.

### Questions
What is this about? In particular, many of the references are to work that predates agentic AI, so it is not clear whether technical terms used here refer to modern agentic systems or to some older computation model or algorithms.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
2

### Summary
The paper presents a way to use local differential privacy in best-first routing with early stopping and certification. The paper also aims to create a repayable ledger or log.

### Strengths
The paper wants to create an auditable system where the routing decisions both satisfy differential private and are auditable. The use of post-processing property of DP seems like the right approach. The evaluation with respect to different baselines shows that the method outperforms them.

### Weaknesses
The paper is incomprehensible to this reviewer in its present form. It begins without defining a problem statement and instead presents a very specific running example with undefined terms. There is no context given for the problem that a general reader can understand. 

The technical claims are not possible to check. There are so many terms used which have no definitions before they are used (e.g. certificates, context-indexed, PRF, etc.). Citations to the prior algorithms are omitted so one cannot follow them to fill in the gaps. Barring these syntactic issues, the deeper concern is that there are formal claims throughout which lack a proof. For example, one page 3 Lemma 1 doesn't have a proof, it has a sketch, and Lemma 2 has merely a justification. 

Given the state of the manuscript, a technical evaluation or acceptance of the presented ideas is very difficult.

### Questions
None.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
2

### Summary
The paper is very difficult to read. I could not go beyond the first few pages.

The paper does not clearly explain the motivation behind the problem it aims to solve. Many technical terms are introduced without adequate explanation. A few examples are listed below:

Line 013: The term “local” in local differential privacy is not defined.

Line 015: The term “leaf” is used without a prior definition of a DAG. Why does a node contain a key?

Line 019: Exact mode and Surrogate mode are mentioned with little explanation of their meaning.

Line 024: The analysis refers to DAGs, but the experiments are conducted on graphs—this inconsistency is unclear.

Line 032: The paper mentions a “grammar” without introducing its purpose or explaining how it relates to local differential privacy.

I may be an outsider to this specific field, but a well-written paper should still make its goals and motivations clear without requiring additional background reading.

### Strengths
I could not finish reading the paper, so I am unable to assess its strengths.

### Weaknesses
The paper is difficult to read for people outside the field.

### Questions
What is the motivation behind your work?

### Soundness
2

### Presentation
1

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
The authors explore the easy stopping for best-first routing agents under local differential privacy (LDP) and develop a run-wise certificate mechanism by coupling node pruning keys to the same exponential races. The discussed issue is significant, and the authors conduct several experiments to verify the effectiveness of the work.

### Strengths
1. The discussed issue in terms of privacy and auditability is significant.
2. Several experiments are performed.

### Weaknesses
1. The presentation of the work should be improved. This paper does not fully illustrate the research background and motivations. 
2. The work lacks a formal and detailed definition of the studied problem;
3. Some symbols are not predefined before being used. For example, in the first paragraph of the Introduction, $s_{det}$, $C(P)$ are not explained.
4. A literature review is not provided. Are there any similar latest works on the same topic?
5. The comparison seems a little insufficient.

### Questions
Refer to the weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method for applying local differential privacy to best-first routing with early stopping and certification, and further seeks to construct a verifiable, replayable ledger or log.

### Strengths
This paper addresses an important problem: ensuring that the auditable system satisfies differential privacy and allows routing decisions to be audited. They have conducted experiments against selected baselines to demonstrate how their method has improved upon them and to justify their claims.

### Weaknesses
- The issue from the paper is that it is very difficult to decipher the motivation behind the work, and the concepts discussed are not introduced in a logical way.
- The formal definition of the problem is missing, and most of their definitions are lacking mathematical proof. This is a dangerous trend for the paper, making me hard to trust the paper and its claims.
- Given the current state of the manuscript, a thorough technical assessment or validation of the ideas is challenging.

### Questions
1. For "Run-wise certificate" the statement itself does not make sense. Can you please expand upon it?

### Soundness
2

### Presentation
1

### Contribution
1
