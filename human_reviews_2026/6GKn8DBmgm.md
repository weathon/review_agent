# AudiFair: Privacy-Preserving Framework for Auditing Fairness

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Ensuring fairness in AI is challenging, especially when privacy concerns prevent access to proprietary models and training data. 
We propose a cryptographic framework for auditing fairness without requiring model disclosure. 

Unlike existing solutions---which either do not capture attack vectors enabling dishonest model providers to manipulate a dataset to pass audits unfairly, or require involving real-world model users to protect against dishonest behaviors---our framework realizes the following properties simultaneously for the first time: 
1) Model Privacy: Proprietary model details remain hidden from verifiers.
2) Dishonest Provider Robustness: Even if model providers are dishonest, a verifier can statistically attest to the fairness of the model without involving real-world users.
3) Test Data Transparency: Test data for auditing is generated in a transparent and accountable way, preventing dishonest parties from manipulating it.
We achieve these goals by carefully orchestrating cryptographic commitments, coin tossing, and zero knowledge proofs, and we report the empirical performance for auditing private decision tree models.
Our solution is highly communication-efficient, delivering a significant improvement (~200,000x for a 30k-sized dataset) over the current state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The submission describes a system for auditing/evaluating the fairness of a machine learning model such that the model itself is not leaked to the verifying party, no potentially private data is needed in the process, and neither model provider nor verifier need to be *honest*, i.e. follow the protocol. To achieve these properties, a number of cryptographic primitives are instantiated: the model provider has to provide a formal *commitment* to ensure that they actually compute the evaluation of the promised model on the provided data, not a potentially manipulated version or manipulated output. A zero-knowledge proof scheme allows the model provider to prove to the verifier that a desired fairness proper holds, without leaking any other information. Finally, agreed on synthetic data is used as input to the system to avoid privacy concerns. 
Most of the manuscript is used explains these cryptographic concepts, and their application to the specific task at hand. Afterwards, results of experiments are reported for decision tree models in reasonably large/real-world datasets. One observation here is that the proposed implementation is much faster than prior work that furthermore provided weaker guarantees.

### Strengths
+ cryptographic safety for ML methods is an intersting and relevant topic
+ the described method makes sense for the task at hand
+ the presentation does a good job at trying to explain the method also to non-experts readers
+ the reported speedups are impressive

### Weaknesses
The main weakness of the manuscript is its unclear scientific contribution. In general, I could see multiple axes of potential innovation: new crypto, new machine learning, new contributions to  systems/implementation, new insights at the interface of two or more of these.

On the crypto side, the manuscript does not seem to propose new methods, but exloit on existing primitives. Essentially, it follows the common pattern of replacing a trusted party (that would get the model and the data and compute the fairness certificate) by cryptography. The fact that existing component are used is a good thing for the credibility to the actual safety of the proposed system and the possibility of practical implementation, but it does not constitute much of scientific novelty, besides proof that the combination of the component still fulfills the desired privacy criteria.
On the ML side, the manuscript also does not propose new methods. It deals with the question how to evaluate a known quantity (fairness) for a known and already trained ML model. Again, this is good, because it allows widespread applicability, but there is no scientific novelty on the ML side.
One novel aspect is on the systems side. Running crypto in practice requires a substantial amount of system design and implementational effort, e.g. to achieve efficiency. However, the manuscript is not written as a systems paper. Aspects of implementation are largely skipped, only the underlying libraries are mentioned. Even the specific choice of cryptographic compenet is not justified, except SWIFFT/Groth16/zkSNARK for trees, which was introduced in prior work Zhang et al. (2020). For example, why Groth16? That leads to small size for the proof, but in this context, efficiency seems more of the focus. Instead of the SWIFFT hash, why not e.g. Poseidon? 
Ultimately, the main contribution seems to be of connecting the fields, demonstrating to the machine learning community how (existing) cryptography allows solving problems without a trusted party. I agree that the majority of the machine learning community is not aware of the possibilities of modern crypto. However, such a demonstration might be better placed in a position paper or a tutorial, not in a paper submission to the core technical track. 

Besides the above, I find it a shortcoming that the procedure requires synthetic data, but the limitations of this are not discussed. Fairness is a distribution dependent quantity, and it is not discussed under which conditions, synthetic data will be a suitable proxy for real data. Furthermore, practical synthetic data generators often have subtle artifacts, which might offer a path to break the proposed protocol: if the model provider can detect such artifacts, they might commit to a model that returns fair (e.g. random) answers on synthetic data, but otherwise behaves unfairly.

### Questions
* I would like to understand your reasoning why you think ICLR is the right publication venue for your work. Please clarify what you consider the main scientific contribution of your work? Is it on the crypto, ML or systems side, or does it lie a new combination of existing tools, or increasing the awareness of the ML community to the existence of certain cryptographic tools? Who do you think will benefit from your work, if it is published at ICLR?

* Is the proposed scheme still secure if synthetic data is distiguishable from real data? 

* Please explain your choices of SWIFFT and Groth16 zkSNARK.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper, AudiFair, presents a cryptographic framework for auditing AI model fairness while preserving the model's privacy. The main idea is to address issues with existing methods, such as model providers cheating or real-world users having to be involved in the audit process. The paper claims to be the first to combine three features: model privacy, robustness against a dishonest provider, and transparent test data generation.

The protocol works in three phases: the provider "commits" to their model, then the provider and verifier jointly generate a synthetic test dataset, and finally, the provider uses a zero-knowledge proof (ZKP) to demonstrate that their model is fair on that dataset. The authors implemented and tested this for decision trees. They report a substantial improvement in communication size over a prior work called C-PROFITT.

### Strengths
* The paper tackles an important practical problem: how to verify a model's fairness when the model owner cannot or will not reveal it. The goal of achieving this without involving real-world users is a good one and seems to be a key advantage over some related work.

* he paper is clearly written. The introduction does a good job of explaining the problem, and Table 1 was particularly helpful for understanding how this work claims to be different from its competitors.

* Experimental Validation: The authors use synthetic data for the audit, which could be a concern. However, they sensibly check that the fairness scores (like Demographic Parity) on the synthetic data are close to the scores on the real data, which provides some validation for this approach.

* Communication Efficiency: The reduction in communication bandwidth (the final proof size) is a notable practical improvement.

### Weaknesses
* Computational Cost: A major drawback is the prover's computation time. The paper states this is ~10x higher than the C-PROFITT baseline. This seems like a very high price to pay for the added security. This high overhead might make the system impractical for many real-world scenarios where audits need to be run quickly or frequently.

* Limited Model Scope: The evaluation is only for decision trees. This is a very simple model class. Most modern AI systems with serious fairness concerns use much more complex models like deep neural networks or large gradient-boosted ensembles (e.g., XGBoost). It's not at all clear how or if this framework could ever be efficient enough for those models, and the paper doesn't provide a path forward.

* Unclear DataGen Security: I was a bit confused about the security of the DataGen step. The protocol secures the random seed used to generate the data, but it seems to assume the synthetic data generator algorithm itself is honest. What stops a provider from training and committing to a generator that, even with a random seed, only produces "easy" or non-representative data, making the fairness check trivial to pass? This seems like a potential gap.

### Questions
* How realistic is the "future work" of applying this to models like XGBoost or neural networks? My understanding is that ZK-proving for these models is extremely expensive, so it's hard for me to see how this could be practical.

* Could you please clarify the assumption about the DataGen algorithm ? Does the protocol assume both parties have access to and agree on a trusted, public version of the generator? What happens if the provider is the one who supplies the generator?

### Soundness
3

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
3

### Summary
The paper proposes defense against malicious auditors when verifying fairness of a model in a privacy-preserving fashion. It claims to do so by having the prover and verifier agree on a randomness, which is used to generate synthetic data for fairness auditing.

### Strengths
The idea that there needs to be protection against malicious verifiers seems interesting.

### Weaknesses
1. I think exactly how the verifier can manipulate fairness scores and therefore the corresponding threat model the algorithm is trying to protect against is not clear. As far as I can understand, it could be because verifier gives model A an easy audit dataset vs. model B. Authors claim the issue could be prevented for giving a harder dataset to B (since in the protocol both verifier and prover B in this case will agree to a randomness). However, then wouldn't everyone want an easier audit dataset to pass? I think the point should be to come up with a scheme that leads to true audits. To see exactly what your defense protects against, please add more clarity in the threat model.

2. What does it mean to "fail the audit despite holding a fair model."?? -- all auditing datasets are valid, some can successfully find the flaws in the model others cannot. I am failing to understand how the model not succeeding on an audit dataset is a problem -- it just means the audit dataset did its job.

3. The abstract is really badly written. Major part of the abstract should talk about your contribution. From the abstract it seems like this is the first paper proposing ZKPs for fairness audits. Fig1 also completely misses the point of the paper. The focus of the paper is verifier, not prover. Preventing against dishonest provers has been seen in the literature.

4. How do you get lower numbers than CP? I couldn't understand your contributions which reduced the time overhead.

### Questions
See above

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
This paper presents a privacy-preserving framework for evaluating fairness in machine learning models. The proposed solution leverages zero-knowledge proofs to maintain model privacy, ensure robustness against dishonest providers, and preserve the transparency of test data while evaluating fairness. The authors validate their approach using a decision tree model on the ACS dataset and compare its performance with C-PROFITT. Experimental results indicate that the proposed method achieves higher efficiency than the evaluated baselines.

### Strengths
1.	Developing privacy-preserving frameworks is an important and meaningful research direction for the machine learning community.
2.	The reported results appear to outperform the evaluated baselines, and the authors support their claims with empirical evidence rather than purely theoretical analysis.
3.	The paper is clearly written and well-structured, making it easy to follow.

### Weaknesses
1.	Similar approaches have been introduced in prior work, making the added value and contribution of the proposed framework unclear.
2.	The applicability of the proposed method is demonstrated solely using a decision tree model, which limits the scope of the evaluation.

### Questions
[1] While the proposed method is interesting, its novelty relative to prior work remains unclear - particularly in comparison to [1], which also defines fairness using commitment schemes, zero-knowledge proofs, and data augmentation techniques. The authors are encouraged to elaborate on how their approach differs from this prior work and what unique contributions it provides.

[2] Although I am not a cryptography expert, I recognize that constructing circuits for machine learning models, especially neural networks, is a non-trivial task. Could the authors provide more details on this aspect? Doing so would help clarify the practicality and potential applicability of the proposed framework for readers.


[1] Segal, Shahar, et al. "Fairness in the eyes of the data: Certifying machine-learning models." Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society. 2021.

### Soundness
3

### Presentation
3

### Contribution
2
