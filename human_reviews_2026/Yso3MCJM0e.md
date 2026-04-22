# Statistically Undetectable Backdoors in Deep Neural Networks

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
We show how an adversarial model trainer can plant backdoors in a large class of deep, feedforward neural networks. These backdoors are statistically undetectable in the white-box setting, meaning that the backdoored and honestly trained models are close in total variation distance, even given the full descriptions of the models (e.g., all of the weights). The backdoor provides access to invariance-based adversarial examples for every input, mapping distant inputs to unusually close outputs. However, without the backdoor, it is provably impossible (under standard cryptographic assumptions) to generate any such adversarial examples in polynomial time. Our theoretical and preliminary empirical findings demonstrate a fundamental power asymmetry between model trainers and model users.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper discusses a method for planting a backdoor into neural networks that is statistically undetectable. The main idea involves a random compression embedding layer right after the input. This layer is created by rejection sampling in such a way that a given fixed perturbation vector z, when added to any input x, will result in a very small difference in the output. At the same time, this vector cannot be efficiently identified if only the weights of the embedding layer are known. The construction is supported by an extensive theoretical analysis that establishes strong cryptographic properties.

### Strengths
The paper provides thorough theoretical support regarding the cryptographic properties and the indistinguishability of the backdoored random compression layer, which might potentially help establish cryptographic applications in neural networks.

### Weaknesses
The paper heavily builds on the reference Bogdanov et al (2025), where most of the key ideas have already been established, most importantly, the secure hash-like properties of the Gaussian embedding layer included in the proposed construction. The paper does offer novelty, though, regarding the total variation distance of the plain and backdoored layer and the backdoor strength.

However, the paper poses assumptions that, in the present form, almost prevent any meaningful applications of the idea. A random compression layer at the beginning is highly unusual. While one can argue that it is a theoretically justified technique for feature extraction, in current architectures it is  not applied, and if its sole purpose is to carry backdoors, then users will simply avoid such networks.  The Lipschitz assumption in the remaining part of the network is extremely strong, and it is certainly not practical.

The theoretical results are very interesting but in general, the proposed application is not necessarily the best for them. In other cryptographic tasks, like a steganographic signature of the model weights to prevent model theft, they would work much better. In that case, even the Lipschitz assumption is unnecessary. The "backdoors" that are proposed are not really backdoors in the usual sense, as they are unsuitable to controllably change model outputs. The vector z is more like a signature of the model.

### Questions
Can you summarize the contributions relative to Bogdanov et al 2025?

Can you use this technique to "backdoor" neural networks in the usual sense of maliciously and controllably changing the network outcome by manipulating the input?

While the vector z cannot be recovered, one can get suspicious by looking at the unusual model architecture. Can you hide the technique in existing architectures?

What is your idea about other applications, like watermarking? Wouldn't that be much more suitable? If not, why? How about related work in that area?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present a strategy to train DNNs with statistically undetectable backdoors, empowering model trainers with more control to tamper with the training process without model users' knowledge. Both theoretical justifications and empirical studies are presented.

### Strengths
The presented study is very interesting. 

Both theoretical and empirical justifications are convincing.

### Weaknesses
The nature of the backdoors needs future clarification.

### Questions
The computed backdoors appear to be model-dependent. Given a set of data D, 
1. An algorithm A and its backdoored version B output DNNs M_A and M_B, computing a backdoor z for a given x. Suppose another algorithm A', trained on a dataset following the same distribution as D, outputs M_A'. Will z be able to fool M_A' ? 
2. If a random noise dz is added to z, will M_A and its backdoored counterpart M_B output indistinguishable results with  x+z+dz  as the input? 

Both questions are concerned with the generality of the computed backdoors. If their adversity is dependent on a specific algorithm, it's unclear how much practical use this discovery can offer. 
 
Minor comments:

On page 3, "than" ---> then

On page 7, "a the same pair " ---> remove "a"

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
This paper proposes a method for planting statistically undetectable backdoors into certain deep neural networks, showing that there's a significant power imbalance between the model's creator and its users: even with full white-box access, a user cannot detect the existence of a backdoor, which allows the creator to generate invariance-based adversarial examples, nor can it create such examples itself. This means the creator can use a secret key ($z$) to make any input $x$ and a distant, altered input $x'$ (e.g., $x+z$) produce nearly identical outputs. Under standard cryptographic assumptions, it is provably impossible for a user without the key to find such colliding pairs. The technique applies to feedforward DNNs where the first layer is a frozen, compressing Gaussian matrix and all subsequent layers are bi-Lipschitz. The backdoor is embedded by specially generating this first-layer matrix, conditioned on the secret vector $z$ (which is also drawn at random first).

### Strengths
- The paper studies an important topic both for theoreticians and practitioners.

- The guarantees are rather strong, since they hold even when one has white-box access to the model.

- The technical constructions are using some interesting ideas.

- The exposition is pretty clean and nice to read.

### Weaknesses
- Set of models is a bit restrictive and makes the technical result less interesting than I originally thought after reading the abstract; essentially all the difficulty of handling the DNN comes from the first layer because of the Lipschitzness condition of the subsequent layers. Could you comment on if you see a technical path towards extending the results to more general families of DNNs?

- I'm a bit confused about the notion of backdoor here: prior work asks for the ability of the model creator to map input examples to ones that are close, and their labels are far. This paper adopts a dual notion: it asks for input examples that are far but their labels are close. Moreover, the formal definition of this notion of backdoors requires that the model creator can produce a single pair of "collisions."

Why did use this notion over the more well-studied one? Intuitively, the former requirement of the backdoor (the one studied in prior work) sounds more reasonable. Could you explain why you chose this notion of backdoors, how it compares with the one I'm suggesting here, and if you can handle this notion?

Despite these concerns, I think that the result is interesting in its own right, so I'm on the positive side. However, it is important to clarify why these colliding inputs should be thought of as backdoors (given the extensive prior work), and if there isn't a good justification to do so it would be good to phrase the results differently.

### Questions
Please see weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This mostly theoretical paper's goal is to prove that it is possible (under cryptographic & other assumptions) to train a neural network in such a way that
- there is a vector $z$ known to the trainer such that adding $z$ to any input $x$ results in an input $x+z$ where the final layer embedding (and in fact any layer embedding) of $x+z$ is very close to that of $x$. 
- for someone with white-box access, it is
	1. statistically infeasible to tell that the training process was "tampered with"
	2. computationally infeasible (w.r.t cryptographic assumptions and polynomial-time complexity) to compute any pair of inputs $(x, x')$ that result in nearly as close final-layer embeddings ("collisions") as for $(x, x+z)$. 

	This is proposed as a way to "backdoor" the model. The proof makes some assumptions on the neural network (importantly, bi-Lipschitzness of the composition of layers after the 1st), which essentially reduce it to showing that the 1st layer can be set up to have the main property we want. The proof heavily relies on the prior work [1], which shows among other things that finding collisions in a matrix $A$ drawn from a certain random ensemble is computationally intractable. This random ensemble is used as a basis for the construction, which proceeds by first choosing ~any $z$, and then rejection sampling the rows of $A$ so that they ensure $z$ has the collision property. The main difficulty in the paper is to show that the rejection sampling does not move the distribution of the matrix $A$ too much away from the random ensemble the result in [1] is based on. This also ensures the statistical infeasibility of distinguishing the trained network from a "normally" trained one. 
	
[1] Andrej Bogdanov, Alon Rosen, Neekon Vafa, and Vinod Vaikuntanathan. Adaptive robustness of hypergrid johnson-lindenstrauss.

### Strengths
- Writing is clear and motivates and describes the math well, providing useful intuition and context in prior work
- Strong results on infeasibility to know that the training process was modified, and to be able to produce collisions without the hidden attacker vector $z$
- Some empirical validation

### Weaknesses
- The core of the result and all the work happens in the first layer of the network; the bi-Lipschitzness assumption quite easily takes care of the rest of the network. As such, it's a major assumption, and it's uncler the extent to which we can expect it to hold in practice. For instance, the existence of adversarial examples suggests that bi-Lipschitzness can't be too strong
- There is important reliance in the result on the degree of compression happening in the first layer. It's important to note that modern practical neural networks do not have arbitrarily high compression ratios; 1/10 seems a reasonable lower bound.
- it's unclear why this should be considered a "backdoor". I don't see a practically relevant scenario where an attacker could derive some benefit from the existence of these collisions.

### Questions
- why should we be worried about backdoors like that? to me it seems much more natural to use this as a watermarking tool.
- in practice, how bad is the breakdown in the bi-Lipschitz property due to adversarial examples

### Soundness
4

### Presentation
4

### Contribution
3
