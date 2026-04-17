# Selective Rotary Position Embedding

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Positional information is essential for language modeling. Softmax Transformers with Rotary Position Embeddings (RoPE) encode it with fixed-angle rotations, while linear Transformers rely on input-dependent gates that only decay past key-value norms. We provide a theoretical argument for the necessity of a rotation and decay component in well-performing sequence models, and observe that the missing ingredient in linear models is precisely the rotation that softmax attention performs implicitly. We introduce Selective Rotary Position Embedding (*Selective RoPE*), an input-dependent, learnable rotary embedding that generalizes RoPE to arbitrary angles and composes seamlessly with decay gates. Equipping gated linear attention with *Selective RoPE* yields a complex-valued recurrent layer that can be implemented efficiently with the “RoPE trick”. On synthetic benchmarks (MQAR, copying, state tracking) and 370M-parameter language-model pre-training, the method improves recall, downstream accuracy, and expressivity while adding minimal architectural overhead. We open-source our implementation [here](https://github.com/timurcarstensen/selective-rope).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors generalise RoPE to a mechanism allowing it to choose angles in a way that is input dependent. The authors perform analysis mainly on linear-attention models and show that selective rope seems to improve performance over baselines such as RoPE or NoPE.

### Strengths
The connection between SSMs, linear attention, and RoPE is interesting. I particularly liked the presentation in Table 1.

### Weaknesses
My main area of research is in Transformers and not linear attention although I have some experience with linear attention and RFF.s 

I do not think I quite understand the point of "Softmax attention implicitly applies a selective rotation, to encode relative positional information between tokens." Are you arguing that the rotations come from the relationship between the softmax kernel and RFF? So you can view the softmax kernel as applying RoPE but where the angles are sampled IID from a Gaussian. This however would really only be true if your angle samples tend to infinity of course. Is this how you are connecting RoPE with a "NoPE" softmax?

I found the notation slightly hard to follow especially as someone not coming from SMMs. I mainly found confusing that RoPE is really a method used in Transformers, but the paper seems to only implemented the selective mechanism for linear attention and was not implemented for normal quadratic Transformers? Is there something stopping you from implementing for a normal Transformer? 

Minor
Typo in abstract: rotation in all angels -> rotation in all angles

### Questions
Please see questions in the weaknesses

### Soundness
2

### Presentation
2

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
This paper presents a version of RoPE with learned, input-dependent arbitrary rotations. Theoretical analysis provided indicates that softmax attention implicitly performs selective rotations, motiving the proposed architecture. Selective RoPE uses a learned linear projection and cumulative sum to produce input-dependent rotations. An analysis of diagonal SSMs is provided which shows distinct roles for the real and imaginary parts of the state matrix, motiving the incorporation of Selective RoPE with GLA to provide better memory. Experiments on language modeling and show improvements with Selective RoPE compared to RoPE and softmax attention in GLA models.

### Strengths
The paper gives a detailed theoretical justification for the architecture design. The analyses of softmax attention as implicit rotation and spectral leakage in SSMs may be useful to future work. Experimental evidence is provided to support claims.

### Weaknesses
The conclusion that softmax attention applies implicit selective rotation is based on the RFF approximation and additional normalization assumptions. The paper does not prove that the resultant normalized approximation converges in the limit to softmax attention, and so this analysis may be overstating the connection. 


The real-data language modeling results in table 3 omit the RoPE condition.

### Questions
Did you compute RoPE setting for Table 3?

### Soundness
3

### Presentation
3

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
The paper proposes a selective Rotary Position Embedding (RoPE) that uses an input-dependent rotation to enhance the performance of models with Gated Linear Attention. The paper provides a theoretical analysis of how softmax attention performs a hidden form of rotation and further proposes to determine the rotation angle via a linear projection of the query. Experiments are conducted on GLA showing that selective RoPE achieves better performance than NoPE and RoPE.

### Strengths
* The paper is clear with summarized insights and clear figures.

* The analyses on the implicit rotation of softmax attention are interesting.

* The paper provides an in-depth analysis from the RFF perspective.

### Weaknesses
* While the paper takes a lot of effort in the derivation of implicit selective rotation in softmax attention, the proposed method is applied to gated linear attention. Given that the derivation heavily relies on the Random Fourier Features (RFF) approximation, the practical impact of the proposed method has not been validated.

* Limited experiments. The experiments are conducted on small-scale models, which raises my concern about the stability and scalability of the proposed method. The leaned rotation angle may lead to unstable training.

### Questions
* It would be appreciated if the authors could provide additional experimental results of applying selective RoPE to softmax attention with more details on the experiments.

* Since rotations are composable. Can the proposed method be equivalently viewed as applying a rotation to the queries and keys before the RoPE operation? What is the significance of doing so?

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
2

### Summary
The paper introduces Selective Rotary Position Embedding (Selective RoPE), an input-dependent mechanism designed to generalize standard Rotary Position Embeddings (RoPE) by performing rotations at arbitrary, selective frequencies.

### Strengths
Interesting theoretical insights, such as:
- Softmax attention implicitly applies a selective rotation, to encode relative positional information between tokens
- Linear Transformers can be enhanced by using both forgetting via real decay and rotation via imaginary gate.

### Weaknesses
Very limited evaluation for language modeling.  Would be good to have at least RoPE as baseline (Table 3) and evaluate Selective RoPE in different settings (i.e context length)

### Questions
Do you expect GLA  and Softmax Transformers to benefit equally from Selective RoPE?

### Soundness
2

### Presentation
3

### Contribution
2
