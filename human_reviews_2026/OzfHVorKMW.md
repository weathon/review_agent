# Privacy-Preserving Mechanisms Enable Cheap Verifiable Inference of LLMs

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
As large language models (LLMs) continue to grow in size, fewer users are able to host and run models locally. This has led to increased use of third-party hosting services. However, in this setting, there is a lack of guarantees on the computation performed by the inference provider. For example, a dishonest provider may replace an expensive large model with a cheaper-to-run weaker model and return the results from the weaker model to the user. Existing tools to verify inference typically rely on methods from cryptography such as zero-knowledge proofs (ZKPs), but these add significant computational overhead, and remain infeasible for use for large models. In this work, we develop a new insight -- that given a method for performing \emph{private} LLM inference, one can obtain forms of \emph{verified} inference at marginal extra cost. Specifically, we propose two new protocols, each of which leverage privacy-preserving LLM inference in order to provide different guarantees over the inference that was carried out. Our approaches are cheap, requiring the addition of a few extra tokens of computation, and have little to no downstream impact. As the fastest privacy-preserving inference methods are typically faster than ZK methods, the proposed protocols also improve verification runtime. Our work provides novel insights into the connections between privacy and verifiability in LLM inference. We open source our code at https://anonymous.4open.science/r/priveri/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper focus on the problem of verifiability of hosted LLM inference services. Different from prior works that use heavy ZKP technologies, the authors propose to verify the LLM output by inserting random tokens to the user prompt or adding random noises to the token embeddings, where the output logits can be significantly different if a different (typically smaller) model is used. The user-side perturbation is made oblivious to the server owing to the assumed underlying MPC-based inference. The proposed methods exhibit both theoretical and statistical resilience to attacks.

### Strengths
- The paper is well-written and easy to follow.
- The problem of verifying MLaaS is an increasingly important direction.
- The idea of build verification upon MPC-based secure inference is interesting.

### Weaknesses
- Table 2 only shows experiments for one single forward pass. However, the proposed approaches  require continual user interaction at every decoding step. Runtime for a full generation is need to showcase the introduced additional cost.
- Protocol 1 and Protocol 2 both inherently require user interaction for every step of token decoding. This is not reasonable in real-world deployment,
- Protocol 3, which is claimed to be non-interactive, also has a significant limitation. For any models with instruction-following, the key appending does not work.

### Questions
Please refer to the weakness. I belive the current limitations for each protocol is significant and not applicable to real-world deployment.

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
1

### Summary
This paper addresses a critical practical challenge in remote large language model (LLM) inference: untrusted third-party providers may substitute expensive, high-capacity models with cheaper, weaker alternatives (e.g., replacing LLaMA-70B with LLaMA-7B) to cut costs, while users lack mechanisms to verify the authenticity of the computation. Existing verification tools (e.g., zero-knowledge proofs, ZKPs) are infeasible for LLMs due to prohibitive computational overhead.  The authors propose three distinct protocols that leverage privacy-preserving inference to provide targeted verification guarantees. The work aims to advance both practical verification tools for remote LLM inference and theoretical understanding of privacy-verifiability links in AI systems.

### Strengths
1. This paper addresses a high-priority problem in LLM deployment with a novel, practical insight: leveraging private inference to enable low-overhead verification. 
2. The link between private and verified inference is a key innovation.

### Weaknesses
1. The presentation of this paper is not good enough, and it is a little hard to understand how it works. Maybe a schematic graph helps
2. What are connections between three protocols? How does this paper relate to existing works?

### Questions
see weaknesses

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper explores the intersection between privacy-preserving computation and verifiable inference for large language models (LLMs). It argues that when privacy-preserving techniques such as Secure Multi-Party Computation (SMPC) or Fully Homomorphic Encryption (FHE) are already used for model inference, these mechanisms can be extended to provide verifiable computation almost for free. 

To support this idea, the paper proposes three novel protocols: Logit Fingerprinting, Logit Fingerprinting with Noise, and Key Appending. Experimental results show that these methods can detect dishonest inference efficiently and run much faster than zero-knowledge proof (ZK)–based approaches.

Since I am not an expert in security or cryptographic verification, I may not be able to fully assess the depth or novelty of the proposed mechanisms. However, from a general machine learning perspective, the idea of connecting privacy and verification in LLM inference seems potentially impactful.

### Strengths
1. This work presents a creative and timely exploration of a relatively unstudied relationship between privacy and verifiability in large model inference. 

2. The motivation is clear, as the growing reliance on third-party model hosting introduces both privacy and integrity risks.

3. The three proposed protocols provide a spectrum of practical trade-offs between interaction cost, computational efficiency, and verification strength.

### Weaknesses
1. The paper seems to rely mostly on empirical results rather than formal proofs, and it is not clear how strong the guarantees are compared with existing cryptographic approaches.

2. The evaluation, while comprehensive in experiments, could include more discussion of practical deployment aspects such as latency, communication overhead, and integration into existing systems. It is also not entirely clear how well the proposed methods scale to long or interactive LLM sessions.

3. For readers without a strong security background, it would be helpful to have clearer explanations of key assumptions (for example, what it means for a provider to be “honest but curious,” or how the privacy mechanism ensures non-collusion).

### Questions
- Could the authors clarify, in more intuitive terms, what kind of verification guarantee each protocol provides compared with standard cryptographic verification? For instance, are these probabilistic or statistical guarantees?
- How do the proposed methods perform in more realistic, multi-turn generation settings where verification cannot be repeated at every step?
- To what extent can these protocols be combined with lighter-weight security assumptions such as trusted execution environments (TEEs)?
- From an engineering perspective, how difficult would it be to integrate these protocols into existing inference frameworks used in industry?
- Could the authors discuss more concretely what kind of real-world scenarios or users would most benefit from these methods?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
he submission proposes several mechanisms to ensure the verifiability
of privacy-preserving LLM inferences. By "verifiability," the user who
supplies private input x can verify that the output y is legitimately
obtained by computing M(x). In the FHE scenario, the user may send a
ciphertext ct = Enc(x); the inference provider homomorphically evaluates
M on ct, and the user decrypts the resulting ciphertext to obtain y.
While one could naively achieve verifiability by using zero-knowledge
proofs, this solution tends to incur significant overhead.

The submission explores alternative and more lightweight methods:

- In Protocol 1, the user inserts randomly selected sentinel tokens into
the tokenized prompt at random positions. The user then observes the
output logit vector at all token positions and compares it against the
precomputed cached logits for model M.

- Protocol 2 is a variant of Protocol 1, which adds randomly sampled
noise to the token embeddings.

- In Protocol 3, the user appends a randomly generated key to the prompt
with explicit instructions asking to repeat the key in the response. The
user then checks that the response includes the same key.

### Strengths
The submission explores alternative solutions to verifiable LLM
inference. Since zero-knowledge proofs for verifiable LLM are
prohibitively expensive due to the underlying cryptographic operations,
proposing a new paradigm is a promising direction.

### Weaknesses
Unfortunately, the submission does not adequately justify the security
of the proposed methods.

- Protocol 3 does not provide a meaningful guarantee. Since the user
prompt always follows the same format, in the FHE setting, any malicious
inference provider can identify the location of the key in the text. By
applying suitable homomorphic operations, an adversary can single out
the ciphertext containing the key and append it to an arbitrary output
text. Thus, this methodology barely meets the requirement of verifiable
computation.

- The authors present some experiments in the SMPC setting, where a
prompt is secret-shared among multiple parties and at least one behaves
honestly. However, if the remaining parties are dishonest, one should
simply use SMPC with **active security** to achieve verifiability, as
the protocol can then tolerate any misbehavior. This means that as soon
as any dishonest party deviates from the protocol by trying to evaluate
secret shares of the private prompt on an incorrect circuit (i.e., wrong
model), the honest party can detect such cheating behavior.

- In general, the paper does not formally define what kind of
"verifiability" is guaranteed by the proposed solutions. I recommend
that the authors make the security notion mathematically precise. For
example, in the context of ZKP, verifiability can be formulated in a
game-based manner: for any input $x$ and model $M$, and any
(computationally bounded) adversary $A$ outputting a proof $\pi$ and
(encryption of) $y$, the probability $\Pr[M(x)\neq y \land V \text{
accepts } \pi]$ is negligible. SMPC with active security also formally
guarantees that either $M(x)=y$ is correctly computed (or otherwise the
protocol simply aborts if it does not have the guaranteed output
delivery property), even in the presence of malicious parties. Both
paradigms provide formal security proofs against _any_ adversarial
strategy. In contrast, the submission only checks security with respect
to _specific attack strategies_, which is generally not a sound
methodology for analyzing security.

### Questions
- Although the main goal of the paper is verifiability, the paper does
not provide any formal security notion as opposed to ZKP and SMPC. Could
the authors define a formal security definition similar to (knowledge)
soundness of ZKP or active security of SMPC?

- If one were to use SMPC to achieve privacy-preserving LLM inference,
wouldn't verifiability be trivially ensured by employing actively secure
SMPC? Note that actively secure SMPC does not necessarily require
zero-knowledge proofs.

### Soundness
1

### Presentation
2

### Contribution
1
