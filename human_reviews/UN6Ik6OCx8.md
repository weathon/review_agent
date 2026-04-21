# Exploring the Design Space of Visual Context Representation in Video MLLMs

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Video Multimodal Large Language Models~(MLLMs) have shown remarkable capability of understanding the video semantics on various downstream tasks. Despite the advancements, there is still a lack of systematic research on visual context representation, which refers to the scheme to select frames from a video and further select the tokens from a frame. In this paper, we explore the design space for visual context representation, and aim to improve the performance of video MLLMs by finding more effective representation schemes. Firstly, we formulate the task of visual context representation as a constrained optimization problem, and model the language modeling loss as a function of the number of frames and the number of embeddings (or tokens) per frame, given the maximum visual context window size. Then, we explore the scaling effects in frame selection and token selection respectively, and fit the corresponding function curve by conducting extensive empirical experiments. We examine the effectiveness of typical selection strategies and present empirical findings to determine the two factors. Furthermore, we study the joint effect of frame selection and token selection, and derive the optimal formula for determining the two factors. We demonstrate that the derived optimal settings show alignment with the best-performed results of empirical experiments. The data and code are available at: https://github.com/RUCAIBox/Opt-Visor.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper analyses the input design space of Video Multimodal Large Language Models (MLLMs) by scaling two components: the number of frames and the number of tokens per frame.
Given the quadratic computational complexity of the attention mechanism in the number of input tokens, MLLMs are limited in the size of their input. This paper studies the impact of varying the number of frames and the number of tokens per frame, as well as the sampling and compression strategies to obtain the final tokens.

### Strengths
The paper offers a valuable analysis of visual context representation design in video MLLMs. The approach is straightforward, with clearly explained experimental settings and well-presented results, and the conclusions are well-supported by experiments.

It is interesting to see the contribution of mean pooling versus uniform sampling, both for selecting frames and selecting tokens.

The authors promise to release code and data for reproducing their results.

### Weaknesses
Comparison to SOTA results on the benchmarks.
It is understood that the goal of this paper is not to obtain SOTA results on the evaluated benchmarks, but rather to perform an analysis on the design space of visual context representation in video MLLMs. Nevertheless, it would be informative to the reader to have the information of how far the experimental recipes presented in this paper are from the SOTA.

### Questions
Figure 2b and 3b: In addition to the performance curves on each benchmark, it would be valuable to also plot the average over all of them.

Section 3: At the other end of the scaling spectrum, it would be interesting to also report the performance without any visual token at all (M=0) to see what would be the comparative performance of a "blind" MLLM.

Minor typo:
L325: "increasing the frames"

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Given a fixed Video-LLM window size, this paper examines the optimal selection of the number of video frames, the number of video tokens, and the compression/sampling methods used for frames and embeddings. The authors include experiments on both loss and benchmark evaluations.

### Strengths
1. The paper is clearly written and follows a logical structure.  

2. The paper includes extensive experiments and offers a thorough analysis. Notably, the experiments that consider fixed frames, fixed embeddings per frame, and the trade-off between embeddings and frames are particularly useful for following research.

### Weaknesses
1. Is the design space for embedding and frame operations too narrow? The findings within this design space may be biased towards different LLM backbones and vision encoders. We suggest that the authors conduct experiments across a variety of LLM backbones and vision encoders to ascertain the generalizability of their findings. Or at least one more different LLM backbone and vision encoders.

2. In the experiments depicted in Figure 2 and Tables 1 & 2, the authors utilize the same T = 32 (L168) for all experiments, which results in video tokens of varying lengths within the same LLM context window size. It is generally assumed that longer tokens within a pretrained LLM context window size lead to better performance. How can it be determined whether longer visual embeddings after the vision encoder or longer visual tokens in the LLM contribute more significantly to performance improvements? 

3. Since this paper aims to study the scaling effect, conducting a sufficient number of experiments with training models is crucial to mitigate the impact of randomness bias. The authors should clarify the number of models trained for each data point presented in experiments.

### Questions
My major concerns lie in the soundness of the experiments. Please find more details in the 'Weaknesses' section

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
4

### Summary
This paper mainly discusses the video frame and visual token per frame selection strategies for video MLLMs. Based on LLaVA-like architecture, this paper conducts several experiments, e.g., varying frame number while fixing visual token per frame and altering visual token per frame while fixing total frame number, to fit the loss curves. Finally, it proposes a fitting function of the two factors and draw an optimal combination conclusion.

### Strengths
1. The motivation and idea of this paper is straightforward.
2. The writing and structure organization of this paper is clear and easy to follow.
3. The experimental results are clearly clarified, and the statistics are persuasive.

### Weaknesses
1. My main concern about this paper is the generalization of the final conclusion. As we all know, video MLLM is very large and complex. This paper uses only one model structure (SigLip + Qwen2-7B) to conduct experiments on several benchmarks. Is the final best setting ($T_{opt} \approx 118$ and $M_{opt} \approx 51$) generalizable to other models and benchmarks? If not, must we conduct several experiments to fit a new curve to find the best factor combination?
2. Based on the above question, if this paper is just an experiment report, readers cannot obtain enough inspiration from this paper.

### Questions
Besides the question in Weakness, how can you determine the function formation of Eqs. 3, 4, 5, and 6? Just observe from figures 2a, 3a, and 4?

### Soundness
2

### Presentation
3

### Contribution
3
