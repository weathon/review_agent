# Limitations on Safe, Trusted, Artificial General Intelligence

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Safety, trust and Artificial General Intelligence (AGI) are aspirational goals in artificial intelligence (AI) systems, and there are several informal interpretations  of these notions. In this paper, we propose strict, mathematical definitions of safety, trust, and AGI, and demonstrate a fundamental incompatibility between them. We define safety of a system as the property that it never makes any false claims, trust as the assumption that the system is safe, and AGI as the property of an AI system always matching or exceeding human capability. Our core finding is that---for our formal definitions of these notions---a safe and trusted AI system cannot be an AGI system: for such a safe, trusted system there are task instances which are easily and provably solvable by a human but not by the system. We note that we consider strict mathematical definitions of safety and trust, and it is possible for real-world deployments to instead rely on alternate, practical interpretations of these notions. We show our results for program verification, planning, and graph reachability. Our proofs draw parallels to Gödel's incompleteness theorems and Turing's proof of the undecidability of the halting problem, and can be regarded as interpretations of Gödel's and Turing's results.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper gives strict, mathematical definitions of safety (“never makes a false claim”), trust (“we assume the system is safe”), and AGI (“solves every instance a human can provably solve with non-zero probability”). With these definitions, it reaches the conclusion that an AI system that is both safe and trusted cannot be an AGI. In other words, among {safety, trust, AGI}, any system can have at most two. The proofs adapt Gödel-style diagonalization and Turing’s halting arguments to concrete AI tasks: program verification, planning, and graph reachability, constructing instances humans can certify but any safe, trusted system must abstain on.  They also analyze a relaxed notion, calibration-safety, and prove a parallel impossibility: even well-behaved, calibration-safe, trusted systems must abstain on some provably halting cases.  Discussion emphasizes the worst-case nature and that “adding axioms” only pushes the problem one step further due to self-reference.

### Strengths
1. Very clear.  It gives crisp, operational definitions of safety (“does not make any false claims… either answers… correctly or abstains”), trust (“the assumption that the system is safe”), and AGI (“solve the instance with some non-zero probability”), which anchors all results.

2. The paper has a clear take home message: The paper's premise is that a system cannot be safe, trusted, and AGI simultaneously. It uses clear examples: spells this out for program verification, planning, and graph reachability. 

3. Discusses other approaches with relaxed safety. It introduces calibration-safety and still derives an impossibility (Theorem 4.2): even a well-behaved, calibration-safe, calibration-trusted system must abstain on some instances that provably halt with high probability.

4. The proofs are presented in a way that doesn't require deep background in formal logic or axiomatic systems, making them accessible to the broader AI research community while maintaining mathematical rigor.

### Weaknesses
Artificial Trust Assumption: The "trust" assumption (that we scientifically accept the system is safe) is somewhat circular in the construction. In practice, we can empirically test AI systems on specific instances without assuming global safety, which could allow us to solve the constructed adversarial examples.

Questionable AGI Definition: The definition of AGI (matching humans on all provably solvable instances) is narrow and controversial. It doesn't capture common intuitions about AGI involving general reasoning, learning, and adaptation across diverse domains. Many would argue a system could be "AGI" while failing on specific adversarial instances.

Overly Restrictive Definitions: The definition of safety (zero false claims, must be correct or abstain) is extremely stringent and may not reflect how "safe" AI systems are conceptualized in practice. Real-world safety often involves bounded error rates, not perfect correctness.

Limited practical impact: The proofs rely on self-referential constructions; the paper does not estimate their prevalence in real deployments or offer empirical benchmarks, leaving the practical incidence of the impossibility unmeasured.

### Questions
Can you characterize trade-offs (e.g., bounds on abstention rate vs. safety) rather than all-or-nothing impossibility?

Which weaker safety/trust notions (e.g., probabilistic safety, local trust, distributional assumptions) still yield impossibility—and which permit meaningful AGI capability?

Any empirical proxies (benchmarks) that approximate your constructions to test how real systems fail/abstain?

### Soundness
3

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
This paper proposes a formal impossibility result showing that safety, trust, and Artificial General Intelligence (AGI) cannot be simultaneously achieved. The authors formalize safety as the system never making false claims, trust as the assumption that the system is safe, and AGI as always matching or exceeding human capability across all provably solvable task instances. Using Gödel’s incompleteness theorem and Turing’s halting problem as theoretical foundations, they prove that any AI system that is both safe and trusted cannot be an AGI, illustrating this through program verification, planning, and graph reachability tasks. The paper aims to reveal a fundamental trade-off between correctness guarantees and generality.

### Strengths
- The paper attempts a rigorous formal treatment of concepts - safety, trust, and AGI - that are often discussed only qualitatively.

- The argument is structured across multiple canonical AI problem settings, enhancing clarity and accessibility.

- The connection to Gödel’s and Turing’s classical results is elegant and provides theoretical depth.

### Weaknesses
- The definitions of safety and trust are overly strong and conceptually narrow, assuming perfect correctness and universal acceptance of safety: conditions far removed from realistic AI deployments.

- The reduction of trust to “assumption of safety” is philosophically and technically weak, ignoring the social, epistemic, and empirical bases of how trust in AI is actually formed.

- The core conclusion depends entirely on these rigid definitions, so the incompatibility may not hold under more plausible, probabilistic, or bounded interpretations of safety and trust.

- The paper implicitly equates truth with safety, yet not all true claims are safe (e.g., truthful but uncalibrated or context-insensitive outputs can still lead to unsafe outcomes). This undermines the claim that correctness alone guarantees safety.

- While the logical result is internally consistent, the external validity and practical relevance of the theorem are insufficiently supported.

### Questions
- How would the impossibility result change if safety were defined probabilistically or contextually (e.g., bounded-risk safety instead of absolute correctness)?

- Given that not all true statements are safe, could the authors refine the framework to distinguish epistemic correctness from operational safety?

- Would redefining trust as empirical confidence (based on verification or transparency) rather than assumption preserve or dissolve the incompatibility?

- Could these results extend to learning-based systems where reasoning is inductive and not explicitly logical, and if so, how would "provable correctness" be replaced?

### Soundness
2

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
This paper presents some impossibility results on concrete definitions of safe, trusted, and general artificial intelligence (AI).

An AI system is formalized as a program that receives some input and produces an output that can be "yes", "no", or "don't know". An AI system is safe if its outputs are always correct or "don't know", trusted if it is generally assumed to be safe (by whom?), and general if it can solve any task that a human can solve.

The paper shows several such impossibility results, following classical arguments to show the undecidability of fundamental problems such as the halting problem.

### Strengths
**S1.** The results and proofs are sound.

**S2.** Once you buy the premise of the language being used, the paper is easy to follow.

**S3.** The paper provides a fun perspective on the theoretical limits of computer programs compared to human reasoning, that will be appealing for researchers acquainted with classical undecidability arguments, and may bring attention to them to unfamiliar readers in the AGI space.

### Weaknesses
**W1. Eye-catching language, that overpromises and underdelivers in the results**.
The language of safe, trust, and AGI is not a good choice. These concepts are subject to a wide study, and this paper does not engage properly with them. 

In particular, sentences like "an AI system that is safe and trusted cannot be an AGI" are very eye-catching; however, when you properly understand them, they are not surprising at all. 

**W2. The treatment of "trust" is not very solid.** 
The notion of "trust" is treated informally and conflated with scientific acceptability, which has well-understood epistemic and community-dependent aspects. The authors should either formalize trust as a belief layer and avoid claims that invoke vague phrases such as "scientifically accepted proof." A better approach would be to treat "trust" as a belief layer, so it is different to say "A is safe", than "A is believed to be safe". This also leads to different concepts of AGI: a program that can solve any problem a human can, or a problem that can solve any problem that a human believes they can solve. 

*The problem with the "science" talk*. Expressions like "scientifically accepted that the system is safe", or "scientifically acceptable proof" are meaningless in essence. To go a step deeper, in most contexts a scientific truth or a scientifically acceptable statement is understood to be a statement that is supported by the available evidence and part of the consensus of the scientific community for which it is relevant. There are two key notions to treat here: data and community. Data is very important because these statements typically refer to empirical sciences, where absolute (or formal) truths cannot exist and are always subject to fallibility: for any scientifically accepted statement, there exists an experiment and an adequate result that turn the statement to unaccepted. Computer science (and more generally, mathematics) does not depend on data because the statements are true or false independently of the data. Community is also important because a scientifically acceptable statement is always meant in reference to a community. The same fact may be acceptable for a community and unacceptable for another one, so saying that a fact is scientifically accepted is meaningless without mentioning to which community it refers.

Because of all these arguments, I would strongly recommend the authors to explicitly include a formal way to reason about beliefs in their paper, instead of relying on this vague and imprecise language of something being "scientifically acceptable".

**W3. Modelling of human reasoning**. By definition (the definition adopted in this paper, not necessarily the generally accepted definition), the human's problem-solving capability subsumes that of the AI (since the human can incorporate the AI as a tool). Thus, a claim like "humans can always solve at least as much as a safe, trusted AI" would be tautological. This is not clear from an initial reading of the paper, and it would be helpful if the authors explicitly discussed this asymmetry and its implications for the notion of AGI they adopt.

**W4. Technical novelty**. The results are reformulations of classical diagonalization arguments known for years in the computability literature. The novelty is therefore primarily rhetorical rather than technical.

### Questions
**Q1**. In Algorithm 1, it is never defined what a valid input to a program is, and whether it is something that is reasonable to check. Would it be possible that checking whether an input is valid is itself an undecidable problem, and therefore $\mathtt{Goedel-program}$ is not well defined?

**Q2**. Since the graph reachability problem cannot be solved in general faster than in the time it takes to read the whole graph, any program that solves it faster and in a "safe" way is forced to leave some instances "unanswered". A human could arbitrarily select one of these instances, add it to its solving capabilities in constant time, and thus claim to be able to solve more instances than the AI program in a given time, concluding then that it cannot be an AGI. Is this a fair restatement of theorem 3.10?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper formally defines safety (no false claims, either correct answers or abstention), trust (assumption of safety), and AGI (matching or exceeding human capability on all provably solvable task instances) for AI systems. Its core finding is the fundamental incompatibility of these three properties: a safe and trusted AI system cannot be an AGI system, as there exist task instances (in program verification, planning, and graph reachability) easily solvable by humans but unsolvable by the system. The proofs draw parallels to Gödel’s incompleteness theorems and Turing’s halting problem undecidability, and the result extends to a relaxed "calibration-safety" notion. The paper also discusses potential workarounds (e.g., adding axioms) and clarifies that the result is worst-case rather than universal.

### Strengths
+ The paper provides precise mathematical definitions for vague core concepts (safety, trust, AGI), addressing a longstanding gap in AGI and AI safety research.

+ By linking classic computability results (Gödel, Turing) to modern AI challenges, it reveals a fundamental constraint on AGI development that is independent of specific technologies or architectures.

+ The result is validated across three critical AI tasks (program verification, planning, graph reachability), enhancing its generalizability and practical relevance.

+ Exploring calibration-safety (a relaxed safety notion) shows the incompatibility is not limited to strict safety, strengthening the core argument.

### Weaknesses
- The AGI definition focuses on "matching human capability on provably solvable instances", which excludes aspects like adaptability, common sense, or learning from unstructured data—key parts of mainstream AGI discourse.
- The strict mathematical assumptions (e.g., absolute safety, perfect trust) may not align with practical AI deployments, where "sufficiently safe" or "empirically trusted" systems are more common. Note that even the traditional software system could have bugs or other trustworthiness issues. (However, I understand that the author aims to present their argument from a universal and fully safe perspective.) 
- The paper assumes humans can consistently produce "scientifically acceptable proofs" for task instances, which I think might be a little bit overlooking human cognitive limitations (e.g., bias, resource constraints) highlighted in cognitive science.
- While identifying the incompatibility, the paper offers little guidance on tradeoffs or implementation strategies for balancing safety, trust, and capability in real-world AI systems.

### Questions
1. How might the paper’s safety-trust-AGI incompatibility shift if AGI prioritizes "most" (not "all") human-solvable tasks (matching real-world needs)? If so, which task subsets could work for safe/trusted setups? Do the authors share insights on this?

2. Could hybrid systems (e.g., AI augmented with human oversight for edge cases) circumvent the incompatibility, and what formal guarantees might apply to such configurations?

3. The paper notes relaxed safety/trust interpretations could enable partial compatibility—what specific thresholds for "acceptable error" or "empirical trust" would make AGI feasible while retaining meaningful safety?

4. How do the results extend to modern foundation models (e.g., LLMs) that often abstain implicitly (via vague outputs) rather than explicitly saying "don’t know", and how would calibration-safety apply to their probabilistic outputs? 

5. Given the self-referential nature of the constructed task instances, are these edge cases relevant to practical AGI applications, or do they remain theoretical constraints with limited real-world impact?

### Soundness
3

### Presentation
4

### Contribution
2
