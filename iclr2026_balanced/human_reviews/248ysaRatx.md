## Human Reviewer 1

### Summary
This paper investigates the expressive power of recurrent quantum neural networks (RQNNs) in processing temporal data. The authors address a significant gap in quantum machine learning theory – namely, whether quantum recurrent models (a form of quantum reservoir computing with feedback) can universally approximate sequences and dynamical systems, and if so, under what resource requirements. They develop a rigorous theoretical framework combining quantum neural network function approximation results with classical reservoir computing theory. The main contributions include quantitative approximation error bounds and universality theorems for RQNNs with simple linear output layers.

### Strengths
1. IMHO, this work is the first to establish quantitative universal approximation bounds for recurrent quantum neural networks. Prior to this, the literature lacked error guarantees for quantum recurrent models. The paper fills that gap by providing rigorous theorems (with proofs) that demonstrate RQNNs’ ability to approximate a broad class of time-dependent functions to arbitrary accuracy. 
2. Also, it shows that RQNNs can achieve universality without the need for high-degree polynomial readout functions. 
3. to reach a desired approximation error, the required number of qubits grows only logarithmically with $1/\varepsilon$. In other words, exponentially increasing the accuracy only adds a linear number of additional qubits. This is a remarkable claim as it suggests no curse of dimensionality in qubit resources.

### Weaknesses
1. Thm 4.6 relies on the requirement that the state transition function lies in the Barron function class and has bounded first derivatives (plus contractivity $\lambda<1$), which means the results apply primarily to “well-behaved” systems (smooth, band-limited, and not too chaotic). Real-world temporal processes might violate these conditions (e.g., non-smooth or highly non-contractive dynamics). 
2. Minor weakness: the paper does not include any experimental or numerical simulation results to complement the theory. All results are analytical. This is a weakness in the context of a machine learning conference. Still, what I said is merely a comment, rather than a criticism.
3. The paper assumes the existence of optimal parameters $\theta$ for the RQNN (since it’s a universal approximation argument), but does not discuss how one might find these parameters in practice. Training a quantum model with many parameters is non-trivial, e.g., you might face barren plateaus, circuit noises, etc. You can argue that this is beyond the scope of this paper. Still, I think it is quite an important aspect to at least discuss them in the conclusion.

### Questions
1. The paper asserts that RQNNs have approximation capabilities “as competitive as” classical reservoir families like echo state networks or state-affine systems. However, it doesn’t provide a direct comparison or quantification of any potential advantage. Hmm.. IMHO, most ppl in QML would ask about the potential for quantum advantage here, for instance, do there exist learning instances such that RQNNs and their classical counterparts exhibit a learning separation in terms of either time or sample complexity?
2. Have the authors considered the practical side of how one would train or set the parameters $\theta$ for an RQNN to approximate a given system? Would one use gradient-based variational quantum circuits training to fit observed data from the target system?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
10

### Confidence
4

---

## Human Reviewer 2

### Summary
Summary

The manuscript rigorously derives approximation error bounds and a
proof for universality of recurrent quantum neural networks (RQNNs).
The main novelty is the inclusion of recurrency/feedback, as similar
results have been recently presented for feedforward quantum neural
networks (Gonon & Jacquier 2025). Universal approximation theorems are
an important topic and well researched for classical neural networks,
as they provide important insights on the power of different neural
network architectures. Given the importance of recurrent networks in
processing time series, the results of this study are highly relevant.


Soundness

The study is purely theoretical. The obtained results, however, look
sound as detailed proofs are presented in the appendices. I admit that
I did not dive deeply into all aspects of the proofs, but many aspects
seem to rely on closely related previous studies on feedforward QNNs.


Presentation

The general structure of the study is very good. The main text
focuses on the main results, and detailed proofs are being relegated
to appendix sections. Also the appendix on previous techniques in
quantum neural networks is very helpful.

The authors further attempt to provide simple introductions to the
individual sections to guide the reader through the logic of the
construction of the quantum circuits and the proofs for their
approximation capabilities. While this helped me to a certain extent,
in my opinion, the manuscript is still hard to follow for non-experts
on quantum neural networks. Given that the ICLR community has very
broad backgrounds, publication at ICLR therefore requires some
improvements in terms of presentation. The main issue I have is that
the manuscript is not self-contained enough. See detailed points in
weaknesses below.


Contribution

The authors present a systematic way to construct unitary operators
from a suitable combination of rotation and Hadamard gates that upon
measurement after application to initial states define a set of
functions that can approximate target activation functions of neurons
in recurrent networks to arbitrary accuracy. This is done by tuning
the parameters of the gates. Mathematically rigorous proofs are
presented for these construction steps and the resulting approximation
power. The analysis largely relies on a recently published study on
similar error bounds and universality properties of quantum
feedforward neural networks. For non-expert readers, the novelty, i.e.
the specific differences in the derivations that feedback and recurrency
introduce, are hard to detect and not clearly presented enough.

### Strengths
The derivations in the study seem mathematically rigorous. The authors
also provide useful background on filters and functionals, and a
helpful review of existing literature in the related works and
appendix sections.

### Weaknesses
The study is not self-contained enough:
references in many places to previous work Gonon and Jacquier (2025).
This makes it harder to follow. In particular, this issue appears when
the authors aim to introduce the unitary matrix V, which seems to be
important for the recurrent quantum neural network architecture. But
all details are relayed to the previous publication.

The study needs to better work out the novelty:
As often mentioned many times throughout the manuscript, the work
largely follows the techniques in Gonon & Jacquier 2025 and others on
feedforward QNNs. The differences between the feedforward and the
recurrent case of QNNs, however, needs to be highlighted more strongly
so that the novelty of the results also becomes more apparent to
readers that are not experts in the field and familar with the
previous studies. In particular concerning the constructions in
Section 3: how do they differ explicitly from the case of feedforward
networks? This is not obvious, but crucial to judge the advances
compared to Gonon & Jacquier 2025.

Proposition 4.1:
This proposition seems to be central for the understanding of the
procedure. The proof of proposition 4.1 is only relayed to Gonon &
Jacquier 2025. For a more self-contained presentation that targets the
broader community of ICLR, it would be better to also present the
proof of proposition 4.1 in the appendix of the current study.

The "curse of dimensionality" is emphasized in the abstract,
introduction and the summary of contributions, but never mentioned in
the results sections. It would be helpful to point to the results on
dimensionality more specifically throughout the manuscript and
emphasize the importance on the log scaling there.

### Questions
Reservoir computing typically optimizes only the readout layer. The
authors mention in Section 1.2. that their results are for networks
where all parameters are trainable, but they claim that they are also
generalizable to random parameters in the recurrent layer. This
generalization does not become clear from the current presentation.
This point is only mentioned again in the conclusion section, but it
is not clear how the analyses of the current study support this claim.
This point needs to be elaborated much further.


Minor points:
- Clarify why it is necessary to extend previous results on functions to the first derivatives.
- The abbreviation SAS is not defined.
- Please spell out Barron-type conditions somewhere. 
- Please elaborate more on the function of control and target qubits for non-expert readers.
- Please make clearer around line 266 that one needs N parallel circuits to have one for approximating each component of F in eq. (1)?

### Soundness
4

### Presentation
2

### Contribution
4

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper derives approximation bounds and universality statements for recurrent quantum neural networks. The proposed approach is based on a uniformly controlled quantum gate to apply multicontrolled rotations to a set of control and target qubits, and it has been recently shown that it can be efficiently implemented.

The authors first prove that RQNNs are able to uniformly approximate the filters induced by any contracting Barrontype state-space system.
Second, they extend this universality property to the much larger category of arbitrary fading memory, causal, and time-invariant filters. In this last result, neither Barron-type integrability nor contractivity conditions are needed for the target filter.

The paper is a strong theoretical contribution to the field. One of its major limitations for ICLR is that it is only theoretical.

### Strengths
- Relevant theoretetical contribution
- Excellent technical depth and rigor
- Clear positioning in literature and in particular w.r.t recent literature

### Weaknesses
- Lack of empirical validation. Theoretical findings are strong. However, no numerical or experimental results are a limitation for this paper.
- Some assumption (e.g., Barron-type integrability) may restrict practical applicability
- A comparison between the proposed approach and classical RNNs or RC models for large n

### Questions
Could you discuss the fact that error rates do not suffer from the curse of dimensionality? I think you refer to d, but  I'm not sure.

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
This theoretical paper discusses the universal approximation bound for RQNNs. The authors extend approximation error bounds from previous QNN work and introduce a time factor as a state-space system to derive the approximation bound for RQNNs.

### Strengths
1. The paper is well-written and logically structured. 
2. The subject is important for the quantum reservoir computing community, providing theoretical foundations for a field dominated by empirical studies.

### Weaknesses
The presentation could be more reader-friendly with graphical illustrations of the problem statement and derivation direction (nice to have but minor).

### Questions
1. In equation (12), where N = 2^{qubit count} relates to qubit size and n relates to parameter count, the role of λ (the gradient bound) in the denominator appears extremely important. How would λ be determined in practice? For example, does it relate to quantum noise? 
2. Can the measurement shot requirement for extracting quantum information from the QRNN play an important role in the approximation bound? For example, the empirical estimation of expectation value typically has a distance error of the form O(1/√N_{shot}) (ref. eq. (19) in [1] and also eq. (19) in [2]).

[1] Qi, J., Yang, CH.H., Chen, PY. et al. Theoretical error performance analysis for variational quantum circuit based functional regression. npj Quantum Inf 9, 4 (2023)

[2] Liu, CY., Kuo, EJ., Abraham Lin, CH. et al. Quantum-Train: rethinking hybrid quantum-classical machine learning in the model compression perspective. Quantum Mach. Intell. 7, 80 (2025).

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4