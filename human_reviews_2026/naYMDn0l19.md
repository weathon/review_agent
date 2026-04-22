# Bidding for Influence: Auction-Driven Diffusion Image Generation

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Motivated by online auctions for banner ads, we propose auctions that fractionally allocate the creation of a banner to bidders according to their preferences. Our mechanism elicits bids and textual prompts from the advertisers, and composes them into a score function that drives a reverse diffusion process that generates the banner. Then, it implements Monte Carlo sampling to calculate approximate VCG-based payments to incentivize high-welfare images. Extensive experiments on a diverse 20-prompt dataset with up to 3 agents demonstrate key economic properties. Our mechanism achieves: (1) bid monotonicity; (2) efficiency improvement of up to 20.7% higher welfare than a single-winner VCG baseline; and (3) approximate incentive compatibility, with average regret as low as 7% when deviating from truthful bidding. These benefits are achieved while preserving high image quality. Our study establishes a principled and scalable bridge between auction theory and controllable image diffusion, laying a foundation for economically aligned, multi-stakeholder image generation in advertising and beyond.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a Vickrey-Clarke-Groves (VCG)-based bidding mechanism for diffusion-based image generation. The framework allows multiple agents to bid for fractional influence over the generated image via classifier-free guidance. Experiments show higher social welfare than a winner-takes-all baseline.

### Strengths
* The idea of combining auction theory with diffusion image generation is conceptually novel and potentially impactful for applications such as multi-stakeholder content generation or advertising. 
* The paper is well written, and the empirical results demonstrate consistent welfare improvement and approximate incentive compatibility.

### Weaknesses
* The framework mainly integrates existing methods (VCG and classifier-free guidance) and lacks a clear novel algorithmic contribution in either modeling or mechanism design.
* Since the proposed approach relies on Monte Carlo estimation for counterfactual reruns, the computational cost grows roughly as $O(nk)$ . It remains unclear how scalable this framework is in practice. The experiments are limited to at most three agents and $k=20$ , which raises concerns about the feasibility of extending the mechanism to larger, real-world settings.
* The baseline comparison is limited, only a single-winner VCG. 
* Since the proposed framework relies heavily on classifier-free guidance, the authors should include ablation studies varying the guidance scale to assess its effect on image quality, welfare, and alignment.
* The experiments are conducted using only a single diffusion model (FLUX.1-schnell). Evaluating the framework on additional backbones (e.g., SDXL, Stable Diffusion) would help demonstrate robustness and model-agnostic applicability.

### Questions
* How sensitive are the results to the choice of alignment metric? The proposed framework assumes that each agent’s semantic component can be cleanly separated in the embedding space (both text and image), e.g., bags vs. shoes. However, when prompts become semantically similar or compositional, such as a bag vs. a red bag, the distinction in embedding space may be weak. In such cases, how does the proposed value function ensure reliable welfare estimation?
* Why is only joint conditioning considered, rather than an additive form such as $w_1 * s_t(x|c, c_1)  + w_2 * s_t(x|c, c_2)$ 
* The authors mention using a guidance scale of 10, but its interpretation is unclear. Since the proposed method composes multiple conditional scores weighted by bids, does this guidance scale apply globally to the composed score, or do the individual bid-based weights sum to 10? 
* Is it possible to compare the proposed framework with other allocation or pricing mechanisms, e.g., Shapley value?
* How accurate is the Monte Carlo estimation with respect to the number of samples $k$?

### Soundness
3

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
4

### Summary
This paper proposes a method for ad image generation. Leveraging the VCG mechanism from ad auctions and diffusion models, this method allows multiple advertisers to bid to influence the generation of a image. Experiments show that this approach maintains image quality while achieving higher social welfare and economic rationality compared to traditional methods.

### Strengths
The integration of auctions with diffusion models is a novel idea. Traditional internet ad slots display content from a single advertiser. This paper's method of merging multiple ads into one generated image has significant potential for a new type of advertising.

### Weaknesses
My major concerns of the paper are its theoretic depth and computational complexity. I understand that the diffusion process is difficult to analyze. But maybe it is possible to provide some structural results about the distribution of the final image? Furhtermore, the VCG-based mechanism also suffers from computational issues, and the issues are magnified when combined with the diffusion process. The experiments only involves 3 agents, which is insufficient.

### Questions
Please see my comments above.

### Soundness
3

### Presentation
4

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
This paper proposes to apply the Vickrey-Clarke-Groves (VCG) mechanism to the setting where multiple agents bid to influence the generation outcome of a vision diffusion model. An allocation rule is implemented based on classifier-free guidance in diffusion models, while a payment rule is implemented based on Monte Carlo sampling. Empirically, the implemented mechanism achieves welfare improvement over the winner-takes-all baseline, incentive compatibility, and bid monotonicity, all while maintaining image aesthetic quality.

### Strengths
- To my knowledge, this paper is the first to propose applying a bidding mechanism to the setting of vision diffusion models. It is an original idea to use score composition to implement multi-bidder influence.
- As noted by the authors, the real-world implication in online advertising can be significant.
- The exposition is well written. As a person without a strong economics background, I can understand the exposition.

### Weaknesses
- The authors argue that their method can improve total welfare compared to the single-winner baseline. However, the agent prompts tested in this paper (Table 3) all bid for different objects in the generated image. For example, in the first setting, agent 1 bids for showing their brand on a mug, agent 2 bids for showing their brand on a laptop, and agent 3 bids for showing their brand on a newspaper. It is unclear whether the proposed method would collapse to the single-winner baseline when all agents bid for the same object in a generated image.
- In Section 4.4, it seems more reasonable to visualize regret vs. truthfulness deviation in one scatter plot instead of two plots--in order to assess incentive compatibility.
- In Section 4.5, the LAION aesthetic predictor can also be used to assess image aesthetic quality. Also, the baseline quality should be from images generated from the base prompt $c$.
- Lines 415-431 contain two very similar paragraphs. Please trim down to one paragraph.

### Questions
- Does your method still outperform the single-winner baseline in terms of total welfare, when all agents bid for the same object in a generated image?
- Does your framework result in higher regret when the truthfulness deviation is higher?
- In terms of image quality, does your framework produces similarly quality images compared to images generated with only the base prompt?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper  introduces the auction mechanism designed for diffusion-based image generation, enabling multiple agents to bid for and share influence over a single generated image. Motivated by the limitations of traditional winner-take-all online ad auctions, this work allows fractional allocation of image content creation according to agents' bids and preferences.
Key contributions include:
A generative auction framework where agent bids dynamically control the composition of a diffusion model's score function.
An allocation and pricing mechanism inspired by Vickrey-Clarke-Groves (VCG) auctions.
Experiments on a dataset of 20 prompts with up to 3 agents, demonstrating bid monotonicity, welfare improvement.
Preservation of image quality when blending multiple agents' inputs, validated via CLIP alignment scores.

### Strengths
It pioneers the task of a generative auction specifically for diffusion-based image generation, bridging auction theory and controllable image synthesis in a new domain. While it builds on known diffusion and VCG auction concepts, their combination to enable multi-agent fractional influence over a continuous visual output is creative.
The problem is timely and important, responding to the technological and economic shift caused by generative AI in advertising and content creation. The results demonstrate meaningful incentives for adopting multi-agent auctions in visual media, potentially impacting online ad platforms and extending to dynamic media applications.

### Weaknesses
The paper exhibits several weaknesses and areas for improvement:
The core auction mechanism primarily adapts existing concepts from classifier-free guidance in diffusion models and classical VCG auction theory without substantial original algorithmic contributions. 
The evaluation compares against only a single-winner VCG baseline, which is a minimal comparative standard. Metrics rely heavily on CLIP-based alignment scores as proxies for semantic accuracy and image quality, which cannot fully capture compositional quality. There is no use of stronger quantitative metrics like FID, human preference score.
Experimental validation uses just 20 base prompts with up to 3 agents, limiting claims about generalizability. The dataset represents a narrow range of scenarios without stress tests on complex or larger-scale settings.
No Human Validation: Given the intended application in advertising, a lack of human studies or user feedback evaluation reduces the practical impact and reliability of semantic alignment and image quality claims.

### Questions
Could the authors please provide a detailed explanation of the single-winner VCG baseline used in the experiments? Specifically, how is the baseline implemented in terms of image generation sampling, prompt conditioning, and image selection? Additionally, how does this baseline relate theoretically and practically to classical single-winner VCG auctions that allocate a discrete good to one highest bidder? Lastly, what hypotheses or advantages is this baseline intended to demonstrate with respect to the multi-agent generative auction, and why is it considered a strong or relevant comparator in this context?
Is it feasible to add standard image quality metrics (e.g., FID, IS,Pickscore) to complement CLIP alignment?


Can we try a simpler baseline or alternative approach where the bidding mechanism only changes the prompt given to the diffusion model according to the bid weights, without modifying the internal diffusion score composition or guidance.
Human-Centric Validation: Are there plans to conduct human studies or ad platform experiments to validate semantic fidelity and economic incentives in real-world contexts?

### Soundness
2

### Presentation
1

### Contribution
2
