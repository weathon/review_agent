# NeuroBack: Improving CDCL SAT Solving using Graph Neural Networks

- Decision: Accept (poster)
- Scores: 5, 8, 5, 6

## Abstract
Propositional satisfiability (SAT) is an NP-complete problem that impacts many
research fields, such as planning, verification, and security. Mainstream modern
SAT solvers are based on the Conflict-Driven Clause Learning (CDCL) algorithm.
Recent work aimed to enhance CDCL SAT solvers using Graph Neural Networks
(GNNs). However, so far this approach either has not made solving more effective,
or required substantial GPU resources for frequent online model inferences. Aiming
to make GNN improvements practical, this paper proposes an approach called
NeuroBack, which builds on two insights: (1) predicting phases (i.e., values) of
variables appearing in the majority (or even all) of the satisfying assignments are
essential for CDCL SAT solving, and (2) it is sufficient to query the neural model
only once for the predictions before the SAT solving starts. Once trained, the
offline model inference allows NeuroBack to execute exclusively on the CPU,
removing its reliance on GPU resources. To train NeuroBack, a new dataset called
DataBack containing 120,286 data samples is created. Finally, NeuroBack is implemented
as an enhancement to a state-of-the-art SAT solver called Kissat. As a result,
it allowed Kissat to solve 5.2% more problems on the recent SAT competition
problem set, SATCOMP-2022. NeuroBack therefore shows how machine learning
can be harnessed to improve SAT solving in an effective and practical manner.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the proposition satisfiability (SAT) problem, which is an NP-complete problem. Recent algorithms focus on enhancing CDCL, the mainstream algorithm for solving SAT problems, using Graph Neural Networks (GNNs). While this can make the  solving process more effective, these methods require online model inferences, which consumes substantial GPU resources. In this paper, the authors design an approach called NeuroBack, which uses the trained model and can be executed on the CPU, avoiding the dependence on GPU resources. The authors claim that enhancing the state-of-the-art SAT solver Kissat with NeuroBack can achieve better results than Kissat itself.

### Strengths
1. NeuroBack gets rid of the GPU resource dependence of the solving process and improves the practicality of using GNN to enhance SAT solving.

2. The authors conduct experiments to evaluate the performance of their proposed method.

### Weaknesses
I have several comments regarding the experiment part, which are listed below.

1. Regarding the baseline competitor. As described in the experiment part, the baseline competitor is kissat. Although kissat can be regarded as the latest breakthrough in the community of SAT solving, lots of kissat’s variants have been proposed since the introduction of kissat (as can be observed in the recent editions of SAT competitions). However, the authors only compare their solver with vanilla kissat. As a submission to a top-tier conference, this is not that thorough.

2. Regarding the evaluation on SAT Competition 2022’s datasets. Actually, as a submission in SAT solving, the tradition is to evaluate their proposed solver on the dataset from the latest edition of SAT Competition. In fact, as of the submission deadline of ICLR, the latest edition of SAT Competition is SAT Competition 2023. According to the official website of SAT Competition 2023 (https://satcompetition.github.io/2023/), the dataset of SAT Competition 2023 was published in July, 2023. Since the submission deadline of ICLR is due on September 28. 2023, there left two months for the authors to conduct the comparative experiments on the dataset of SAT Competition 2023. Hence, such lack of experiments is indeed a minus.

### Questions
Please see my comments listed in "Weaknesses".

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the challenge of improving propositional satisfiability (SAT) problem-solving, a significant task in various research domains like planning, verification, and security. It introduces a novel approach called NeuroBack, which enhances Conflict-Driven Clause Learning (CDCL) SAT solvers using Graph Neural Networks (GNNs. The innovation lies in two key insights: predicting variable phases that frequently appear in satisfying assignments and requiring just one query to the neural model before starting the SAT solving process. Once trained, NeuroBack can operate entirely on the CPU, reducing the reliance on GPU resources.

The authors developed a new dataset, DataBack, comprising 120,286 data samples, to train NeuroBack and implemented it as an enhancement to the state-of-the-art SAT solver, Kissat. By incorporating NeuroBack into Kissat, the SAT solver exhibited a 5.2% improvement in problem-solving effectiveness, as demonstrated in the SATCOMP-2022 competition dataset. It also improved solving efficiency, resulting in an average time saving of 117 seconds per problem. This research introduces the first practical neural approach to enhance CDCL SAT solving without requiring GPU resources, provides a new dataset, and offers public access to the NeuroBack model and NeuroBack-Kissat source code.

### Strengths
Providing a dataset
Competing with SAT 2022

### Weaknesses
I think some citations are missing

https://cs.stanford.edu/~jure/pubs/g2sat-neurips19.pdf
Neurogift: Using a machine learning-based sat solver for cryptanalysis
Role of Machine Learning for Solving Satisfiability Problems and its Applications in Cryptanalysis

It would be nice to have a use case in Cryptanalysis (solving AES or small AES instance)

### Questions
-

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper is devoted to improving propositional satisfiability (SAT)
solving by means of utilizing modern machine learning (ML) technology.
Namely, the paper proposes to make use of a GNN architecture for
representing CNF formulas, which is trained to determine the variable
polarity / phase following the ideas behind backbone literals.
Backbone literals of a CNF formula are literals that must necessarily
be satisfied by every satisfying assignment of that formula. The
authors argue that the GNN architecture they propose can efficiently
determine the polarities of all variables: including those appearing
in backbone literals but also any other variables in the formula.
Afterwards, when the predicted polarities are obtained, a
state-of-the-art SAT solver can be bootstrapped with these variable
phases with the hope that such initialization can boost the solver's
performance. Experimental results shown in the paper demonstrate the
effectiveness of this idea if implemented on top of a modern SAT
solver called Kissat.

### Strengths
- The paper is clearly written and easy to follow. Normally, papers on
  applying ML for improving combinatorial problem solving are written
  in Greek, if seen from the perspective of a researcher with
  expertise in the combinatorial problem of interest. This paper
  serves as a nice exception.

- The idea is reasonable. It is simple to implement and it can be used
  with any SAT solver.

- The experimental results reported although not amazing look solid.

### Weaknesses
- Although everything the paper describes is described well, there are
  bits that aren't detailed sufficiently - some ML people may find it
  to be a minus.

- The paper says nothing about the usability of this heuristic for
  unsatisfiable instances. There are no backbones in those but I
  presume some variable phases may still be more useful in practice
  than the others.

- Despite the claimed experimental results, nothing is shown for the
  unsatisfiable instances. This and the point above can be joined
  together.

- Although Neuroback+Kissat solves 10 more instances out of 308, there
  are 92 more (308 + 92 = 400) where Neuroback fails to do anything
  useful. Hence, the proposed solver configuration must clearly lose
  to Kissat on those 92 instances as it spends additioanl time with no
  effect. If this understanding of mine is correct then the results
  aren't so positive.

- Minor1: in CDCL description, the algorithm undoes not only
  *wrong decisions* but also propagated literals.

- Minor2: in CDCL description, the conventional algorithm backtracks
  to the *latest* decision level where the conflict is resolved - not
  *earliest*.

### Questions
- Can you comment on the use of this heuristic for unsatisfiable
  instances? Have you tried this? If yes, what is the performance
  compared to Kissat? If not, why?

- Can you comment on losing on 82 instances to Kissat if we consider
  all the 400 instances in the benchmark set? Am I missing anything?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The NeuroBack paper proposes a new approach to improve CDCL SAT solving using Graph Neural Networks. The proposed approach, called NeuroBack, gives an initial assignment to all variables and queries the neural model only once, allowing it to execute exclusively on the CPU and making GNN improvements practical. The paper also introduces a new dataset called DataBack and implements NeuroBack as an enhancement to a state-of-the-art SAT solver called Kissat. The authors evaluate NeuroBack on a variety of benchmarks and show that it outperforms Kissat and other state-of-the-art solvers on many instances.

### Strengths
Using ML to enhance CDCL SAT solver is promising research, which is more likely to yield improvement in SAT solving than end-to-end learning on SAT. I like the idea in this paper of using ML to give an initial assignment for the CDCL solvers by training on predicting the value of backdoor variables. The paper mentions the importance of backdoor valuables and the intuition of why training on backdoor variables can help predict the values for all variables that appear in the majority of the variables.

It is also promising to see that NeuroBack is actually able to improve Kissat, as state-of-the-art SAT solvers are well-engineered and very hard to optimize.

### Weaknesses
While training on backdoor variables yields a good predictor for the value of all variables, why this approach works is still a bit mysterious. It would be helpful to have more experiments showing that initialization is actually better than a random assignment. Therefore I would like to see some behavior studies of NeuroBack on different kinds of benchmarks. Possible ways include comparing the distance from this initial assignment given by NeuroBack with the nearest solution. Or evaluating the prediction accuracy of the value of a specific variable given by NeuroBack with the majority value of this variable in all solutions (this requires listing all solutions of a formula, which is only doable on small formulas).  

The experiment section also lacks a comparison with other initialization methods for SAT solving, as Neuro-back is in essence an initialization method. 

I am not quite convinced by the claimed advantage that NeuroBack only needs to be called once. This is a trivial property for any initialization approach for SAT solving. I believe that a more interactive collaboration between GNN and SAT solvers can further improve the performance of STA solvers, even if GNN needs to be called multiple times (correct me if I am wrong). I would be happy to change the evaluation if my concerns can be addressed.

Minor comments:
I would use "value" instead of "phase" and "initial assignment" instead of "initialization" to make it clear.

### Questions
See the above comments.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
