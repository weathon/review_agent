# TivTok: Broadcasting Time-Invariant Tokens for Scalable Video Tokenization

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Video tokenization is a critical bottleneck for learned video compression and generation. Existing methods often fail to adapt to the uneven information density of videos, underutilize temporal redundancy, and overlook the reusability of shared content. We present \textbf{TivTok} (\emph{Time-Invariant Tokenizer}), a transformer-based tokenizer that explicitly decouples videos into \textbf{time-invariant (TIV) tokens}, which capture global information shared across frames, and \textbf{time-variant (TV) tokens}, which encode frame-specific residual details. The encoder is designed with tailored attention masking to enforce this factorization, enabling the invariant component to capture not only static elements but also temporally coherent patterns such as consistent motion trajectories. In decoding, a broadcast mechanism reuses TIV tokens across frames, reducing complexity from quadratic to linear in video length. We further extend this approach to long videos through cross-chunk reuse, enabling scalable compression. Experiments show that TivTok improves reconstruction quality with FVD of 12.65 in the traditional $16 \times 256 \times 256$ setting and achieves a $2.91\times$ gain in compression efficiency for $128 \times 256 \times 256$ videos compared to state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
TIVTok provides a way to take advantage of redundancy in video tokenization (for generative models). Typically prior works consider pixel-level redundancy; however, TIVTok instead proposes a notion of semantic redundancy and lets the model learn exactly what to persist across each frame. It does so by only allowing the TIV tokens to attend to every frame, while letting other tokens attend to others only within the same frame. This forces the model to use the TIV tokens for higher-level concepts that persist across the entire video, and the per-frame TV tokens to be used for lower-level details. The results show that this method is able to perform on par with other video tokenizers while compressing the input into significantly less tokens.

### Strengths
1. I think the idea is very clever. Redundancy has generally been limited to pixel-level (spatial / temporal redundancy) rather than semantic, and the TIV tokens are a nice way to incorporate this. 
2. I think the mechanism of using the invariant tokens and keeping time-variant tokens separate per frame is a nice way to parallelize the decoding and potentially decrease the memory and time usage while also making generation potentially faster. 
3. I think the paper is clearly written and it is very easy to grasp the novelty of the method and what the core contribution is.

### Weaknesses
1. The paper never measures the reduction in wall-clock encoding and decoding time, or the memory savings that come from using the time invariant tokens, and only reports TFLOPS instead. I think memory / WC time are much more important and useful metrics to report, so I would want to see those.
2. A key design decision in this paper is deciding how much time invariant tokens to use and understanding exactly what they capture. As far as I can tell, there is no systematic evaluation of the tradeoff between the number of TIV tokens and TV tokens, and the analysis that measures the effect of the TIV tokens is quite confusing. For example, it's not obvious at all from Figure 4 that the ice skater is covered by the TIV tokens - wouldn't you have to do something like masking out the TV tokens, or vary them in some way to understand the the effect of these? What does "intersection across frames" mean?
3. The whole entropy discussion in the methods section seems very unnecessary. It's obvious that if there is a lot of repeated information content that the entropy of the video will be less - this math is not providing any new insight and it would be better served to have some additional analysis from the experiments instead.
4. The generation experiment seems to be somewhat of an afterthought, and it's not clear to me why TIVTok would necessarily lead to a better representation - there are very few baselines here and the experiment is not explained in any detail.  I think the generation aspect should be explored more, and if it were very clear that TIVTok led to a generation quality that was no worse and also more efficient (due to less tokens + parallelizable per frame attention) then it would be a stronger contribution.

### Questions
1. My main suggestion is that i think seeing the decrease in memory and WC time would be very helpful to understand the impact of this method.
2. I would want a clearer explanation of the analysis on what the TIV tokens actually are doing, not necessarily a new experiment but just a clarification on why it's obvious that the TIV tokens actually learn semantic invariants. If so, I would gladly raise my rating.

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
5

### Summary
This paper presents a new tokenization method for videos, called TivTok. The main idea is to decompose video tokens into (1) time-invariant tokens that are shared across all video frames and (2) time-variant tokens that are allocated to each frame. Intuitively, the time-invariant tokens encode static features in videos (e.g., background), while the time-variant tokens capture motion information. To achieve this, TivTok extends TiTok, an efficient image tokenization method that uses two different types of tokens. The authors also modify the encoder’s attention mask: the time-invariant tokens attend to all tokens across all frames, whereas the time-variant tokens attend only to the current frame and the time-invariant tokens. For the decoder, they introduce parallel decoding, where each frame is decoded independently using the corresponding time-invariant and time-variant tokens. By doing so, the proposed method achieves better reconstruction quality than other video tokenizers under the same compression ratio.

### Strengths
- The paper is generally well-written and easy to follow. 
- I like the idea of decomposing videos into time-invariant parts and time-variant parts. There have been similar approaches, but I think this approach tries to do a more explicit decomposition than prior works, as well as showing better reconstruction quality.
- The paper seems to have a great compression ratio compared with existing video compression methods.

### Weaknesses
Although I appreciate the idea introduced in this paper, I have several concerns:
- **Scalability of the tokenizer.** The paper only conducts experiments on UCF-101 with a resolution of 256×256. While I understand that training a video tokenizer requires significant computational resources, it is still crucial to demonstrate scalability in terms of dataset size or resolution. Could the authors provide additional results by training the tokenizer on larger and higher-resolution datasets, such as VidProM [1]? If memory is the bottleneck, one possible workaround is to first encode each frame into latent vectors using SD-VAE or other image VAEs, and then reconstruct it by training TiVTok in that latent space. The framework could follow the structure: SD-VAE Encoder → TiVTok → SD-VAE Decoder.
- **Evaluation beyond reconstruction.** Showing only reconstruction quality is insufficient, as this paper focuses on tokenization rather than compression. While the authors do provide generation results on UCF-101, many implementation details are missing. For example, is the generation class-conditional? What is the video length used for training the generative model? Moreover, can this tokenizer be applied to other downstream tasks, such as video question answering (VQA)?
- **Demonstrating decomposition.** To validate the claimed decomposition property, it would be valuable to include generation results where the time-invariant tokens are fixed and only the time-variant tokens are varied. This would help visualize the separation between static and dynamic information, similar to the experiment presented in Appendix F of the CMD paper [2].

[1] https://vidprom.github.io/   
[2] Yu et al., Efficient Video Diffusion Models via Content-Frame Motion-Latent Decomposition, ICLR 2024

### Questions
- Could the authors provide more details about the video generation experiments?
- The proposed trick for extending to “long videos” might have a practical limit on the maximum number of frames it can handle. For example, it could be difficult to process 1000 frames, since in that case there may be no time-invariant component shared across all frames. Could the authors provide any insights or empirical results regarding this limitation?

### Soundness
2

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
4

### Summary
In this paper, a transformer-based tokenizer called TivTok is proposed for more efficient video tokenization. Tivtok decomposes a video into time-invariant tokens and time-variant tokens. To this end, the encoder uses masked attention for time-variant tokens to enforce the factorization. In decoding, the time-invariant tokens are re-used for all frames to reduce the computation complexity. Also, the proposed algorithm is extended to the chunked compression scheme for the long video processing. Experimental results show that the proposed algorithm achieves comparable performances with the conventional methods with lower computation budget.

### Strengths
- This paper clearly describes the proposed algorithm and it is easy to follow.
- The motivation is solid and the proposed algorithm seems technically sound.
- The proposed algorithm shows decent video construction performance with lower computation budget.

### Weaknesses
- This paper does not provide sufficient details to reproduce the proposed method. For example, there is no detailed description of the encoder and decoder architecture. It would be helpful if such information were included, at least in the appendix.
- From Equations (5) and (6), it seems that the time-variant token may not encode only the frame-local residuals, as stated in L259. Ideally, that would be the case, but in practice, it might redundantly encode information that is already captured by the time-invariant tokens. While minimizing this redundancy would be more efficient and help reduce the loss, the claim might be somewhat overstated. Do the authors have any additional analysis related to this?
- It would be helpful to include a discussion on the performance differences among the S, M, and L models in Table 1.
- In Table 1, what is the difference between two LARP models in L337 and L338?
- Is there a reason why the performances of all S, M, and L models are not compared in Tables 2 and 3?
- How would the performance change if multiple time-invariant tokens were used — for example, by dividing the video into chunks without averaging the shared tokens?
- Typo
  - L149: complexities.In -> complexities. In
  - L185: , Producing -> , producing
  - L195 Eq1: Patchfy -> Patchify
  - L229: time-invarian -> time-invariant
  - L348: we report -> We report
  - L397: proposed techniques -> proposed techniques.

### Questions
Please find my concerns in the weakness section. I believe this paper has some strengths, and I am willing to increase my score if the concerns are properly addressed in the rebuttal.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes TivTok, a video tokenization method that decouples videos into time-invariant (TIV) tokens and time-variant (TV) tokens to achieve better compression efficiency. The TIV tokens capture shared content across frames, while the TV tokens encode frame-specific details. The approach uses a transformer-based architecture with dual-range attention masks and a TIV Token Broadcasting mechanism to isolate shared versus frame-specific content, facilitating reusability of TIV tokens. The method shows a 2.91× improvement in compression efficiency, while maintaining comparable reconstruction quality

### Strengths
Scalability for video length: The proposed method efficiently scales to long videos by reusing TIV tokens across chunks, reducing tokenization complexity from quadratic to linear.

Clear methodology: The transformer-based architecture, attention masking, learned common content and the broadcasting mechanism for token reuse are well-explained.

### Weaknesses
Lack of innovation and comparison to previous methods: The core idea of TivTok is very close to the CMD and HiVAE method, which also uses a form of content decomposition. The claim of novelty is unsubstantiated, and the authors do not convincingly distinguish their method from CMD and HiVAE. Although these methods are mentioned and described in the related work section, the direct comparison of these baselines is absent in the experiments.

Meaningless and trivial derivation of Eq. (2)-(4): Equations (2)-(4) provide a trivial proof that adds little value to the paper. The argument can be summarized in one sentence: encoding frames independently consumes significantly more tokens than encoding shared content and frame-specific residuals.

Scalability concerns with TivTok-L: TivTok-L, despite being larger, appears to be inferior to the smaller TivTok-M in short video reconstruction. The related explanation is lacking.

Insufficient ablation studies: The ablation study is too simplistic and lacks critical insights. Specifically, there is no exploration of alternative ways to generate TIV and TV tokens. For example, could using 3D convolutions for generating TIV tokens provide better performance? Furthermore, is there any potential improvement in performance if TV tokens are generated by leveraging adjacent frames, or would this approach merely add computational overhead without significant benefits? The impact of different loss functions is not presented. These aspects should be explored to understand the full impact of different design choices and to validate the effectiveness of the proposed method.

Weak qualitative results: The qualitative results presented in Figure 3 do not demonstrate significant improvements over the baseline. For example, the results in Figure 3(a) for TiVTok are less clear than those from Omni-DV, and the hand reflection on the piano surface in Figure 3(d) is closer to the ground truth in CV-VAE. Furthermore, the description of shoe details in the sumo scene is inaccurate, as the sumo wrestler is not wearing shoes in the competition.

### Questions
1. The core idea of TivTok is similar to CMD and HiVAE. Clarify the specific innovations that distinguish TivTok from CMD, and add the comparison in the experiments.

2. Clarify why TivTok-L, despite being larger, appears inferior to TivTok-M in some cases. Explain how the increased computational cost and model size of TivTok-L are justified in terms of performance improvements.

3. Expand the ablation study to include the impact of various design choices, hyperparameters and loss functions.

4. Provide clearer visual examples or more detailed comparisons that highlight TivTok's advantages over existing methods.

### Soundness
2

### Presentation
3

### Contribution
2
