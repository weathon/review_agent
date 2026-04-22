# Online Black-Box Prompt Optimization with Regret Guarantees under Noisy Feedback

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Generative AI excels in various tasks through advanced language modeling techniques, with its performance heavily influenced by input prompts. This has driven significant research into prompt optimization, particularly in commercial generative AI platforms, where prompt optimization is treated as a black-box optimization problem. Most existing research on black-box prompt optimization primarily focuses on offline learning and overlooks the randomness in outputs. However, in real-world applications, black-box prompt optimization typically operates in an online learning setting, which remains largely unexplored, especially given the noisy outputs. To address these challenges, we propose an \textbf{A}daptive \textbf{O}nline \textbf{Z}eroth-order \textbf{P}rompt \textbf{T}uning (AOZPT) approach which integrates zeroth-order optimization with online learning in the non-convex setting. Specifically, we developed an uncertainty-scale-adjustment mechanism to mitigate the noise inherent in generative AI and the high variance associated with zeroth-order estimates. We conducted a comprehensive regret analysis of the AOZPT approach, and the results indicate that sublinear regret convergence is achievable. Extensive generative experiments demonstrate that AOZPT outperforms existing black-box prompt tuning methods, particularly in terms of stability in online scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper considers black-box prompt optimization problem. It proposes AOZPT, an online algorithm, to update the prompt as the streaming data comes in and provides sublinear regret guarantees. The method regards the soft prompt as optimization variables and update them according to the feedback of the generated results in an Adam-style optimization process, in order to reduce the noise. Experiments on CNN/DailyMai, GSM8k and Anime, Paining tasks on Llama-3.1-8B, GPT-3.5-Turbo, and Stable Diffusion are carried out to demonstrate its performance.

### Strengths
- The paper explores an interesting intersection between online learning and black-box prompt tuning, extending zeroth-order optimization methods to an emerging and practically relevant domain.
- The proposed adaptive uncertainty-scaling mechanism—inspired by Adam—offers a principled way to stabilize noisy gradient-free updates.
- The paper provides theoretical guarantees on local regret under noisy feedback.

### Weaknesses
- The noise model (Assumption 3.5) assumes uniform boundedness rather than a stochastic model (e.g., sub-Gaussian), which restricts theoretical generality.
- The anonymous GitHub link has expired, preventing code verification and reproducibility assessment.
- The prompt optimization is motivated by ongoing interactions with users (Lines 65-74). However, the online nature is simulated rather than true sequential feedback. And it is a bit unclear to me how it is simulated in the paper. It would be great if real streaming or non-stationary benchmarks would better justify “online”.
- The computational overhead and latency introduced by the two function calls per iteration are not evaluated.
- The empirical study require more results. For instance, (1) the prompt generating model is fixed for each task,  (2) important configuration details are not reported, e.g., the temperature and decoding strategy, which can greatly influence the final outcome. (3) the tested tasks are limited. (4) the original performance of the models should be included. According to the official report, Llama 3.1 8B (8 shots) achieve 84.5 on GSM8K. (5) Additional ablation studies on text-to-text tasks is appreciated.

### Questions
- Under Assumption 3.5, since the regret bound depends linearly on the noise level $\Delta$, can the authors quantify or empirically estimate the typical scale of $\Delta$ in practice? Given the autoregressive nature of LLMs, small perturbations in input can amplify downstream variance.
- Given similar computational/latency budgets, could a larger model (e.g., GPT-4) achieve comparable or superior performance without AOZPT? For instance, in Table 1, Qwen2.5-14B (MP) outperforms Llama-3.1-8B (AOZPT) on GSM8K—can the authors clarify this discrepancy?
- As the instruct models usually have better instruction following abilities, it is appreciated that the performance on these models (e.g., Qwen2.5-14B-Instruct) can be included to further demonstrate the necessity and benefit of the proposed method. For instance, Qwen2.5-14B-Instruct can achieve 94.8 on GSM8K, better than the results reported in Table 1.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes AOZPT, a method that integrates black-box prompt optimization with online learning. It introduces an adaptive uncertainty scaling mechanism to handle noise from generative models and high variance in zeroth-order gradient estimation. Theoretical analysis shows sublinear regret convergence, and experiments demonstrate superior performance and stability over existing methods.

### Strengths
1. Novel and timely topic that bridges black-box prompt optimization and online learning, addressing an underexplored area

2. The adaptive uncertainty scaling mechanism is innovative and effectively mitigates output noise and gradient variance, improving robustness in practice

3. The paper combines theoretical rigor with empirical validation, providing regret guarantees and solid experimental results across multiple tasks

### Weaknesses
1. While the methodological presentation is generally clear, the description of the adaptive scaling mechanism could benefit from additional implementation details or algorithmic steps to enhance reproducibility.

2. The experimental evaluation would be strengthened by including a wider range of baselines, particularly reinforcement learning–based prompt optimization methods.

3. The theoretical analysis is solid, though a deeper discussion of hyperparameter sensitivity and computational complexity in practical deployments could provide valuable insights into the method’s applicability.

### Questions
see weakness.

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
4

### Summary
This paper proposes an adaptive online zeroth-order prompt tuning, a novel approach integrating zeroth-order optimization with online learning for non-convex settings. It can be applied to real-world scenarios where generative AI operates on streaming data and requires dynamic prompt adjustment.  This approach fills the gap of online learning for black-box prompt optimization, enabling dynamic adaptation to streaming data.

### Strengths
This paper addresses a timely problem of great importance.  It can effectively tackle two core uncertainties in online black-box scenarios: noise from generative AI outputs and high variance in zeroth-order gradient estimates. This paper comes with rigorous theoretical foundations such as formal regret analysis for non-convex settings and comprehensive experiment results.

### Weaknesses
While the proposed method claims efficiency for online scenarios, it lacks details on inference latency and scalability. The framework involves two point gradient estimation and sliding-window gradient averaging but no data is provided on how these steps may affect runtime. In addition,  this paper compares AOZPT to offline baselines and a basic online method but it seems authors omit recent online prompt optimization and adaptive zeroth-order methods for LLMs. Therefore, it is difficult to assess AOZPT’s novelty against state-of-the-art online approaches.  Finally, the proposed method depends on frozen open-source LLMs to convert soft prompts to discrete prompts. Experiments show that removing this component may causes a significant drop in performance. This creates a dependency on high-quality open-source LLMs, which limits its deployment in scenarios where such models are unavailable.

### Questions
The paper uses frozen open-source LLMs to generate discrete prompts, but what if such LLMs are unavailable ?   Can authors explain why there is no evaluation of the proposed method under adversarial noise in generative AI outputs ?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces AOZPT, an online black-box prompt optimization method thta uses a two-point zeroth-order gradient estimator with an adaptive uncertainty scaling to counter both LLM output noise and ZO variance. It proves a sublinear local regret bound under noise and shows consistent empirical improvements across text2text and text2image tasks.

### Strengths
1) It is an interesting idea to move black-box prompt optimization from offline to online learning, this might be useful in training agentic systems that interacts with real environment.
2) The proposed method is clearly presented and very practical.
3) A mostly sound theory (maybe with some assumptions being a little bit too ideal) is built to support the effectiveness of the method.

### Weaknesses
1) Some assumptions, for example the Lipschitzness of \nabla f_t in z, can be too optimistic. Is it possible if the authors can further verify how much we can expect these assumptions to hold in practice, and when these assumptions fail to hold, how much impact does this have on the effectiveness of the method in practices?

2) I would recommend the authors to consider more baselines besides ZO-OGD. This can help isolating gains from the adaptive scaling agaisnt the two-point estimator.

### Questions
Please refer to Weakness).

### Soundness
3

### Presentation
4

### Contribution
3
