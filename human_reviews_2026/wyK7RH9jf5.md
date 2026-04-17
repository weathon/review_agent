# Flow Connecting Actions and Reactions: A Condition-Free Framework for Human Action-Reaction Synthesis

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Human action-reaction synthesis, a fundamental challenge in modeling causal human interactions, plays a critical role in applications ranging from virtual reality to social robotics. While diffusion-based models have demonstrated promising performance, they exhibit two key limitations for interaction synthesis: reliance on complex noise-to-reaction generators with intricate conditional mechanisms, thus limiting to unidirectional generation, and frequent physical violations in generated motions. To address these issues, we propose Action-Reaction Flow Matching (ARFlow), a novel paradigm that establishes direct action-to-reaction mappings, eliminating the need for complex conditional mechanisms and supporting bi-directional generation. Directly applying traditional guidance algorithms tends to undermine the quality of generated reaction motion. We analyze the sampling of flow matching in depth and reveal an issue (Initial Point Deviation) which causes the sampling trajectory to ever farther from the initial action motion. Thus, we propose a reprojection guidance method, RE-GUID, to correct this deviation to enable better interaction. To further enhance the reaction diversity, we incorporate randomness into the sampling process. Extensive experiments on NTU120, Chi3D and InterHuman datasets demonstrate that ARFlow not only outperforms existing methods in terms of Fréchet Inception Distance and motion diversity but also significantly reduces body collisions, as measured by our introduced Intersection Volume and Intersection Frequency metrics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper present a new method for human reaction motion generation by using flow matching adapted for action to reaction synthesis. Additionaly, a guidance module is build to correct error that can appear from using the flow matching, correcting direction and body penetration. Experiments are performed on 3 commonly used datasets and higly the stenghts of the methods especially with regard to avoiding physical impossibilities.

### Strengths
- The use of flow matching for reaction generation is interesting to avoid issue encountered by previous method while being  faster than diffusion based methods.
- The proposed guidance module seems to improve performance while being faster than traditional guidance.
- The proposed methods outperforms the other approachs on the three datasets proposed in the study quantitatively and qualitatively.
- Methods and experiments are very detailed and well explained
- The new metrics to check for body penetration are interesting and should be of interest to the community

### Weaknesses
1. Some more recent are missing for the comparison [1,2], while I am not sure if they are useable in the online setting proposed by the authors it would have been interesting to have the comparison at least for the offline setting.
2. There are two claims about the ability of the methods to deal with challenges but they are never investigated in the experiments. i.e. boxing with recursively generating the motion of both humans ([1] propose a boxing dataset), the condition signal (action motion) is stated to be replaceable by text or audio. Showing results in both scenari would really strengthen the paper.
3. The three datasets contain only relatively simple and short interactions. It would have been interesting to see results on more complex and longer motions.
4. The IF and IV values are much lower for the proposed method than for the GT but this is never explained
5. Table B1 and B2 are missing the IV and IF values
6. It would have been interesting to see the results of reverse generation for all method to highlight their limitations.


[1] Ready-to-React: Online Reaction Policy for Two-Character Interaction Generation, ICLR 2025
[2] REMOS: 3D Motion-Conditioned Reaction Synthesis for Two-Person Interactions, ECCV 2024

### Questions
questions:
- see weaknesses. Particularly answers to 1 and 2 could make me increase my rating

suggestions:
- Interhuman is mentionned several times in the paper but the results are only in the appendix, this should be made more clear before referencing to table B1.
- Table 3 show the methods speed but the gpu used is only mentioned in the appendix. It should be mentionned in the main paper instead.

### Soundness
2

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
This paper proposes ARFlow, a human action-reaction synthesis framework that establishes direct mappings between action and reaction distributions through flow matching. ARFlow eliminates complex conditional mechanisms in traditional diffusion models and enables bidirectional motion generation while maintaining real-time capability. The framework incorporates a reprojection guidance method  that corrects initial point deviation during sampling and significantly reduces bodily inter-penetration between characters. Extensive experiments demonstrate that ARFlow achieves superior performance in motion quality, diversity and physical plausibility, outperforming existing methods in both online and unconstrained settings.

### Strengths
1. The proposal presents a novel application of flow matching to action-reaction synthesis, establishing direct mappings between action and reaction distributions. This eliminates the need for complex conditional mechanisms in diffusion-based methods and enables bidirectional generation.

2. The paper accurately identifies the Initial Point Deviation issue in flow matching sampling and designs the RE-GUID reprojection guidance method. This method corrects the deviation without requiring differentiation of the neural network, balancing physical plausibility and motion generation quality while effectively reducing body penetration.

3. Through extensive experiments, the proposed method is shown to outperform existing baselines on common motion metrics and drastically reduce body collisions, while also achieving simpler training and faster inference.

### Weaknesses
1. The penetration loss function might force characters to separate in close interactions, failing to balance physical plausibility and natural interaction dynamics.

2. Although the paper visualizes a failure case in Figure J.2, it lacks a systematic analysis of failure patterns, weakening the completeness of the method’s limitation discussion

### Questions
1. Are there specific motion categories or interaction patterns where the model exhibits significant performance degradation?

2. Regarding the physical constraint guidance, could its formulation be extended to address a wider range of physical plausibility aspects, such as foot-skating? Furthermore, I am interested in seeing ARFlow's performance evaluated on more physical plausibility metrics to better understand its capabilities and limitations.

3. Would the physical constraint guidance struggle to maintain consistent physical plausibility across long sequences, resulting in occasional penetration or unnatural motions in later frames?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ARFlow, a flow-matching framework for human action-reaction synthesis, claiming to replace noise-conditioned diffusion models with a direct mapping between action and reaction motions. It introduces a reprojection guidance (RE-GUID) method to reduce body penetration. Experiments on NTU120-AS, Chi3D-AS, and InterHuman-AS report lower FID and fewer penetrations compared with diffusion baselines.

### Strengths
Flow matching is an interesting alternative to diffusion and has potential for faster inference. This paper introduces Flow matching for reaction synthesis for two person interaction

The proposed RE-GUID physically-guided correction is simple and computationally efficient.

### Weaknesses
The claimed “condition-free” direct mapping from action to reaction is still conditioned on the action itself. The insight over conventional conditional diffusion models is unclear. 

The demo video show mostly low-dynamic, stiff interactions with limited physical realism. Motions appear collision-free but lack natural dynamics and responsiveness expected in animation paper. This undermines the claimed advantage. I think the contribution is not enough if it just shows clear depenetration between humans, a post-hoc optimization applied after generation can achieve this easily as well.

Important baselines like ReMoS [1] are omitted. The reverse-generation results are minimal and not compelling.

[1] Ghosh, Anindita, et al. "Remos: 3d motion-conditioned reaction synthesis for two-person interactions." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

### Questions
The proposed RE-GUID method adjusts the predicted clean pose (`x1_hat`) using a physical gradient and feeds the corrected result into the next sampling step, avoiding backpropagation through the network. 

However, in principle, one could already do something similar in diffusion models:  `x1_hat' = x1_hat - λ * ∇ L_pene(x1_hat)` followed by `x_{t-1} = g(x_t, x1_hat')` where `g` is the usual reverse sampling update (e.g., DDPM or DDIM). 

Could the authors clarify why RE-GUID is novel beyond this straightforward “detached guidance” idea? Is its contribution mainly the empirical integration into flow matching, or is there a theoretical reason why direct guidance on `x1_hat` would not work equivalently in diffusion models

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of human action-reaction synthesis, which aims to generate physically plausible human reactions in response to given actions. The authors identify two key limitations in existing method of 1. Complex and unidirectional generation; 2. Physical violations. To solve these problems, this paper proposes Action-Reaction Flow matching (ARFlow), a novel framework built on flow matching, to directly model the mapping from the action distribution to the reaction distribution. The simple architecture supports bi-directional generation for reaction-to-action generation. Besides, the authors identify the issue of “Initial Point Deviation” for physical violations of human-human interactions. Then, RE-GUID is proposed as a reprojection guidance to prevent penetration. Extensive experiments on NTU120, Chi3D and InterHuman show the superiority of ARFlow.

### Strengths
1. This work is the first work to apply a flow matching framework to the human action-reaction synthesis task. This "condition-free" approach, which directly models the mapping from the action distribution to the reaction distribution, is an elegant departure from existing diffusion models that rely on complex conditional mechanisms. Critically, this design choice directly enables bi-directional generation (both action-to-reaction and reaction-to-action), solving a key limitation of prior unidirectional methods.
2. The paper's method for handling body inter-penetration is highly insightful. The proposed RE-GUID method is an effective and efficient solution that corrects this deviation by re-projecting the interpolation endpoint . This method, validated by the paper's newly introduced IF and IV metrics, is shown to dramatically reduce body collisions far below the levels of prior work.
3. The authors provide thorough experimental validation across multiple challenging datasets (NTU120-AS, Chi3D-AS, and InterHuman-AS). The results are compelling, demonstrating that ARFlow outperforms existing state-of-the-art methods on key metrics like Fréchet Inception Distance (FID) and motion diversity. Furthermore, the proposed model is significantly more efficient, with fewer parameters, faster training convergence (e.g., half the time of ReGenNet), and much lower inference latency.
4. The paper is well-written and organized. The authors clearly motivate the problem, articulate the limitations of existing work, and present their technical contributions in a logical, easy-to-follow manner.

### Weaknesses
1. Limited Extensibility to Conditional Generation: The paper's primary innovation is its "condition-free" design. The paper provides no experiments or results for the conditional setting, making its flexibility and practical utility in constrained scenarios unproven.
2. Unclear Bi-Directionality for Dynamic Interactions: The claim of "bi-directional generation" is a key strength, but its practical application seems limited. Besides, for the “boxing” example, it is unclear how the current framework could support such a continuous, auto-regressive switching of roles, as this would require a different model setup not demonstrated in the paper.
3. Insufficient Evidence for Reaction Diversity: The model claims to support diverse reaction motions for the same action by incorporating stochasticity during sampling. The paper lacks clear visualization examples for diverse reaction generation. Without this, it is difficult to assess if the model is generating truly meaningful variations or just minor, low-impact perturbations of the same motion pattern. 
4. Limited Generalizability of Physical Constraints: The RE-GUID method, while effective at reducing penetration, may not generalize well. From the visualization, current generated results still physically implausible.

### Questions
The authors are asked to clarify:
1. **Conditional Generation:** How can the "condition-free" model be extended to handle conditional inputs like text, since this was not demonstrated?
2. **Dynamic Bi-Directionality:** How does the *static* "reaction-to-action" experiment support the claim of handling *dynamic*, role-switching interactions (e.g., "boxing") ?
3. **Visual Evidence:** Please provide more visual results to prove:
    - **Diversity:** Show multiple, *distinct* reactions to the *same* action.
    - **Physical Realism:** Show that the model works for *close-contact* interactions (like handshakes) and doesn't just "unnaturally avoid" contact .

### Soundness
3

### Presentation
3

### Contribution
3
