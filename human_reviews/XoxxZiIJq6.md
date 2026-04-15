# Comet: A Communication-efficient and Performant Approximation for Private Transformer Inference

- Decision: Reject
- Scores: 5, 6, 6, 5

## Abstract
The prevalent use of Transformer-like models, exemplified by ChatGPT in modern language processing applications, underscores the critical need for enabling private inference essential for many cloud-based services reliant on such models. However, current privacy-preserving frameworks impose significant communication burden, especially for non-linear computation in Transformer model. In this paper, we introduce a novel plug-in method Comet to effectively reduce the communication cost without compromising the inference performance. We second introduce an efficient approximation method to eliminate the heavy communication in finding good initial approximation. We evaluate our Comet on Bert and RoBERTa models with GLUE benchmark datasets, showing up to 3.9 less communication and 3.5 speedups while keep competitive model performance compared to the prior art.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a secure Multi-Party Computing (MPC)-based private inference framework, called Comet. Comet aims to address the challenge of high communication overhead, particularly for non-linear computations in MPC-based Transformer model inferences. Existing solutions, such as Look-Up Table (LUT) methods and aggressive polynomial approximations, often result in either high communication costs or reduced model performance. To tackle this, Comet harmonizes non-linear functions like GeLU and Softmax into a single approximation function, eliminating the need for heavy communication in finding initial approximations for Newton iterations. A “share flooding” technique is also proposed to further optimize secure two-party computations. Comet was evaluated on encoder-based Transformer models like BERT and RoBERTa (as well as on the evaluation benchmark GLUE) to demonstrate that it is both communication-efficient (i.e., fast in inference) and more accurate over the benchmark compared to existing MPC-based private Transformer inference frameworks.

### Strengths
- This paper is well-motivated. Studying methods that enable private inference over large language models has great practical potential for privacy-constrained use cases, such as in financial and medical institutions. Private inference methods (e.g., MPC-based approaches) can serve as potential alternatives to pure local computing.
- The paper is well-written, and the techniques proposed in the Comet framework appear sound.
- The experimental results look promising, especially regarding accuracy on the GLUE benchmark.

### Weaknesses
- The paper was motivated by using ChatGPT as an example, but all experiments were conducted on encoder-based Transformer models like BERT and RoBERTa. I understand that, architecturally, encoder models and decoder models differ mainly in language masking. However, models like GPT are autoregressive, which involves completely different computational intensity. For instance, autoregressive models generate tokens one by one, making it very difficult to batch inputs.
- Although MPC-based frameworks ensure that neither the client nor the server discloses information to each other, in Comet’s setting, the private information includes both the model weights and the users’ data. However, both parties still see the same (non-encrypted) output in these frameworks. While this may be acceptable for classification tasks conducted by BERT, it may not be suitable for generative models like ChatGPT (e.g., generated content could contain private user information).
- The absolute inference speed remains quite slow, e.g., around 30 seconds per batch, as shown in Table 3. Speeds like this make the proposed framework less practical.

### Questions
- As discussed in the “Weaknesses” section, there is concern about whether the proposed Comet framework can be applied to generative language models. Can the authors provide a feasible approach to extend the proposed framework to generative LLMs like ChatGPT and Llama?
- What if, in a practical application, the server model is not open-source (e.g., users can only call their API, like GPT and Claude)? Is there a way to still conduct private inference?
- The experiments in this paper are conducted on CPUs. However, Crypten supports GPU computing [1]. I wonder how the performance of Comet compares to Crypten on GPUs.
- It seems the major weakness of MPCFormer [2] is that it requires fine-tuning using knowledge distillation (KD). However, the idea makes sense for pure inference. I wonder if Comet could also benefit from KD.
- Encoder models are generally easy to batch. I wonder how different batch sizes affect the computation speed of Comet.

[1] https://github.com/facebookresearch/CrypTen?tab=readme-ov-file#installing-crypten  
[2] https://openreview.net/forum?id=CWmvjOEhgH-

### Soundness
3

### Presentation
3

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
The paper introduces Comet, a framework aimed at efficient, privacy-preserving inference for Transformer models. Comet achieves this by unifying complex non-linear functions, like GeLU and Softmax, under a single protocol based on inverse square root calculations. It introduces a "double approximation" method to eliminate the need for extensive communication when finding initial approximations, further improving communication efficiency with a technique called "share flooding." Experimental results show that Comet reduces communication costs while maintaining good performance on models like BERT and RoBERTa on the GLUE benchmark

### Strengths
1. Designing an improved approximation for non-linear functions in private Transformer inference is critical, and this paper's adoption of the "smu" function is both innovative and effective, demonstrating strong accuracy after fine-tuning.
2. The paper presents an effective solution for obtaining a good initial approximation in the context of 2PC-based secret sharing, which is a key contribution to reducing communication overhead.

### Weaknesses
1. The properties of the "smu" function are not sufficiently explained. While experiments indicate that it achieves 1-2% higher accuracy than the original GeLU and Softmax functions, the underlying mechanism remains unclear. Including a graph of the approximated function would help clarify this.
2. The explanation of the double approximation method is confusing. Since MPC typically operates on fixed-point numbers, it is unclear whether the double approximation method is designed for floating-point operations, which are costly in MPC.
3. It is worth noting that other approximation methods like Bolt, Bumblebee, and Puma do not require fine-tuning, and this difference should be mentioned for context.

### Questions
Do you have code to reproduce your work since the "smu" function is rarely used?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Comet to tackle to slow private inference for transformer based model. It is a plug-and-play method with an efficient approximation method to reduce the cost in non-linear computation. It achieves 3.9x less communication and 3.5x speed up compared to previous SOTA.

### Strengths
1. The methods are well motivated by observations.
2. The approximations are quite novel and interesting.

### Weaknesses
1. Figures are (in my opinion, not a major weakness) not good enough: (1) Figure 1 seems like hand written, with small fonts and large empty spaces. (2) Figure 2 should be put in appendix or wrapfigure (since they are not the main point of the paper).
2. The 2-relu approximations have been proposed in literature, e.g. MPCFormer. Also, since approximation has been heavily studied, can the author possible make a small table to present what is in literature and what is not (to improve presentation)?

### Questions
Please see the weakness section.

One other comment I have is that the experiment testbed is quite outdated. The author should consider more benchmark and larger scale model (This is out-of scope of the paper I think, but would be useful for future research), e.g. Vicuna and MMLU. ChatGPT is not Bert-like models.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper tackles the communication (and latency) overhead introduced by nonlinear operations (GELU, LayerNorm, and Softmax) in cryptographically secure private inference of transformer-based models. The authors replace GELU and Softmax with a smoothed maximum unit (SMU) function, effectively transforming nonlinearities into inverse square root operations, and developed the methods such as double approximation and share flooding to efficiently compute them in privacy-preserving settings.  

By mitigating the communication overhead associated with these nonlinear operations, this work tackles a major bottleneck for the practical deployment of private LLM services.

### Strengths
$\bullet$ Comprehensive evaluation of post-training performance across a variety of downstream tasks, demonstrating robustness, applicability, and generalizability of their proposed approaches. 

$\bullet$ The integration of SMU with trainable parameters as an alternative to standard nonlinearities like GELU and ReLU-approximated Softmax is an interesting approach to simplifying the conventional nonlinearities and reducing communication and latency overheads in private inference. 

$\bullet$ A thorough comparison with previous methods for private inference in transformer-based models.

### Weaknesses
$\bullet$ The contributions in this paper are largely engineering tweaks rather than substantive algorithmic advancements. While incorporating SMU (Biswas et al., CVPR'22) as a replacement for traditional nonlinearities like GELU and ReLU-approximated Softmax in a transformer-based model is an interesting approach, it lacks sufficient originality and novelty. Additionally, the use of Network Raphson methods for inverse square root calculations, as previously implemented in CryPTen but with an improved initialization, which reduces the number of iterations to converge, does not offer a particularly compelling approach. These techniques are often highly model- and task-dependent, requiring tuning when applied to other tasks or domains. 

Furthermore, the authors’ proposed share flooding method, which is based on the observation  "that the absolute magnitude of tensor values is closely surrounded around zero (Line#317)," is rather intuitive and lacks deeper insight. Nonetheless, the authors will definitely get the points for the double approximation approach which effectively reduces the communication overhead to constant time complexity for lookup tables. 

Overall, this paper lacks significant research insights or novel observations addressing the challenges of efficient private inference for large language models.

$\bullet$ I also find the comparison with CrypTen unclear, as CrypTen is a 3PC method, whereas the authors use a 2PC method. At one point (Line#482-483), the authors mention excluding PUMA due to its 3PC computation, which raises questions about the consistency of the comparison.

$\bullet$ The authors should consider comparing their approximation methods with the polynomial approximations of LayerNorm, GELU, and Softmax used in *Zimerman et al., Converting Transformers to Polynomial Form for Secure Inference Over Homomorphic Encryption* (**ICML'24**). It’s essential to at least qualitatively contrast and position the merits (efficiency and predictive performance) of their approach with these recent polynomial approximations. 

$\bullet$ I find it quite challenging to follow the draft because the readability suffers from overly long paragraphs. Breaking up the content into shorter, clearer paragraphs would really help make the information easier to digest.

## Correction in the draft

$\bullet$ Figure 1 depicts the 2PC threat model for Post-LN configuration, however, in the caption the authors have mentioned CrypTen (which is 3PC). Also, in the FFN block diagram, one linear layer (after GELU layer) is missing. 

$\bullet$ Line#180: GELU necessitates the Gaussian error function---> BERT and other recent transformer-based models adopt the $Tanh$ approximated GELU implementation, which is significantly faster, even in cryptographic settings (e.g., Bumblebee), than the original GELU   (`torch.nn.GELU()`). See [1] and their implementation in hugging face activation libraries [2]

1. https://paperswithcode.com/method/gelu


2. https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L49

### Questions
$\bullet$ Why the authors have compared their 2PC method to 3PC method CrypTen? Is that the comparison with Crypten's Newton Raphson's method and other approximations, and not exactly with their 3PC settings? 

$\bullet$ Why do the client and server share, which is an integer in the field arithmetic, converted to floating point numbers (Eq 1 and Eq 2)?

### Soundness
3

### Presentation
1

### Contribution
2
