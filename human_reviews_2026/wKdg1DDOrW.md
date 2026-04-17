# SAIL: Self-Amplified Iterative Learning for Diffusion Model Alignment with Minimal Human Feedback

- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
Aligning diffusion models with human preferences remains challenging, particularly when reward models are unavailable or impractical to obtain, and collecting large-scale preference datasets is prohibitively expensive. This raises a fundamental question: can we achieve effective alignment using only minimal human feedback, without auxiliary reward models, by unlocking the latent capabilities within diffusion models themselves? In this paper, we propose SAIL (Self-Amplified Iterative Learning), a novel framework that enables diffusion models to act as their own teachers through iterative self-improvement. Starting from a minimal seed set of human-annotated preference pairs, SAIL operates in a closed-loop manner where the model progressively generates diverse samples, self-annotates preferences based on its evolving understanding, and refines itself using this self-augmented dataset. To ensure robust learning and prevent catastrophic forgetting, we introduce a ranked preference mixup strategy that carefully balances exploration with adherence to initial human priors. Extensive experiments demonstrate that SAIL consistently outperforms state-of-the-art methods across multiple benchmarks while using merely 6\% of the preference data required by existing approaches, revealing that diffusion models possess remarkable self-improvement capabilities that, when properly harnessed, can effectively replace both large-scale human annotation and external reward models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper deals with the limitations of existing approaches to preference alignment for diffusion models: (1) DiffusionDPO requires large-scale human-annotated preference data; (2) Auxiliary reward modeling approaches introduce more biases, are vulnerable to reward hacking, and struggle with distributional shifts from training data. To address these issues, the authors propose their methodology with a novel argument for diffusion model.

The authors argue that diffusion models' potential hasn't been fully exploited through supervised fine-tuning on human-labeled datasets alone. Instead, they propose iterative self-improvement by cold-starting with only a small seed set of human preference data. Based on the under-exploitation assumption, they propose SAIL, an iterative self-improvement closed-loop learning process and also introduce a ranked preference mixup strategy to prevent distribution collapse.

Experimental results show that SAIL achieves comparable performance to state-of-the-art methods while using only 6% of the human preference data, highlighting the sample efficiency of the proposed algorithm.

### Strengths
1. The idea of fully exploiting the base model's potential is innovative. By eliciting this potential through the proposed iterative self-improvement method SAIL, the authors achieve comparable or better preference performance using only 6% of the human preference data.

2. The consistent improvement across multiple iterations further demonstrates the effectiveness of the proposed iterative self-improvement paradigm.

### Weaknesses
1. The tables lack multiple trials and confidence intervals, which are necessary to demonstrate the statistical significance of performance improvements and validate the effectiveness of the algorithm design in ablation studies.

2. It would be valuable for the authors to include results over a larger range of iterations to illustrate the performance trajectory and reveal how the improvement trend evolves as the number of iterations increases.

### Questions
See above.

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
This paper presents a method for aligning diffusion models with human preference using limited preference data. It utilizes the implicit reward function from DPO to create online preference data. To address overfitting, it mixes online preference data with the initial preference data. Experiments show that the proposed method can achieve comparable or better metrics than some previous work, using less data.

### Strengths
- Using self-rewarding to rank online data is relatively new in text-to-image generation. 
- The proposed method works well with limited data.

### Weaknesses
- Implicit reward is adopted from previous work in LLM.
- The mixup of online and initial preference data is straightforward.
- Some generated images seem to have a color saturation problem.
- There are a lot of problems in writing, e.g.
  - Line 144-145: grammar error
  - Line 145: some -> Some
  - Line 344: Pic-a-Pic -> Pick-a-Pic
  - Line 357: use -> uses
  - Line 362: bringing challenge -> brings challenges
  - Line 373: fix reference
  - Line 374-375: Thus, …, so … 
  - Line 454: reveals -> reveal
  - Line 457: suffers -> suffers from

### Questions
- What are the drawbacks when performing more iterations of SAIL?

### Soundness
2

### Presentation
2

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
This paper proposes SAIL, a novel framework for aligning diffusion models with human preferences without reward models or the use of large-scale human-annotated data. The idea is to bootstrap alignment through self-amplified iterative learning using a minimal set of preference data. Experiments demonstrate strong performance while using only a small part of the preference data.

### Strengths
1. The paper proposes a self-improving framework to align diffusion models with human preferences without large-scale annotated datasets, which is novel. 
2. Thorough empirical evaluation results show that the proposed method is effective, outperforming existing alignment methods using only 6% of the annotations.

### Weaknesses
1. Lack of Theoretical Guarantees. While the reward formulation (Eq. 8–9) is mathematically correct in the DiffusionDPO framework, the paper does not provide theoretical analysis of what distribution SAIL converges to. What is the target distribution of this method? Is it the same as DPO? If so, what explains the performance with fewer annotations? It's unclear where the observed gains come from. Whether and why the self-reward metric aligns with true human preference distributions?  A thorough analysis would benefit the paper.

2. Even though the method needs fewer human annotations, it introduces additional cost to generate samples in the training loop, which is not efficient. Could the author provide some discussion on the trade-off between annotation efficiency and training efficiency? 

3. The central reward estimation strategy, which computes preference scores using differences in squared denoising errors between current and reference models (Eq. 6–9), closely follows the DiffusionDPO formulation and thus is not novel.

### Questions
see Weaknesses 1. 2 

1. How sensitive is the proposed method to the choice of the initial seed dataset? What happens if the seed preferences are noisy or biased?

### Soundness
2

### Presentation
3

### Contribution
3
