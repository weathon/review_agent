## Human Reviewer 1

### Summary
This paper introduces a novel framework that formulate the alignment problem as a two-player zero-sum game. This framework involves an adversarial agent and a defensive agent that iteratively interact to improve the LLM’s performance. The adversarial agent generates prompts to reveal weaknesses in the defensive agent’s responses, while the defensive agent seeks to adapt and strengthen its performance based on these prompts.

### Strengths
The main strength of this paper are:
1. The overall writing is well-organized and easy to follow, making the ideas presented clear and understandable. 
2. The experimental results appear solid, especially in safety-related tasks. The proposed framework shows improvements compared to traditional RLHF methods, particularly in handling harmful inputs and jailbreak scenarios, which suggests that the approach is effective in these contexts.

### Weaknesses
1. Lack of Novelty and Insight. While the overall idea is well-executed, it seems relatively straightforward and lacks significant novelty. The two-player game framework, while effective in this context, feels more like an incremental improvement rather than a significant innovation. 
2. Triviality of the Additional Diversity Reward. The additional diversity reward also feels somewhat trivial, as it is a common technique in multi-agent settings. It appears more as a practical trick rather than a meaningful contribution or innovation to the overall methodology.
3. Technical Flaw: The paper’s analysis relies on mirror descent, which guarantees convergence only for the average strategy[1]. However, the final round strategy tends to cycle around the Nash equilibrium rather than converge to it [1][2]. As a result, using only the final strategy in place of the average strategy is not theoretically justified in this context.
4. Computational Cost: The approach requires maintaining two policies for alternating updates, with each policy being optimized using PPO. This results in substantial storage and computational costs, particularly in the context of RLHF. Furthermore, as highlighted in the third point, the use of mirror descent mandates tracking the average policy over time, making it insufficient to rely solely on the final policy. Storing all the historical policies or learning an averagy policy further exacerbates the computational burden, complicating practical implementation at scale.

[1] Mertikopoulos, P., Lecouat, B., Zenati, H., Foo, C. S., Chandrasekhar, V., & Piliouras, G. (2018). Optimistic mirror descent in saddle-point problems: Going the extra (gradient) mile. arXiv preprint arXiv:1807.02629.
[2] Perolat, J., Munos, R., Lespiau, J. B., Omidshafiei, S., Rowland, M., Ortega, P., ... & Tuyls, K. (2021, July). From poincaré recurrence to convergence in imperfect information games: Finding equilibrium via regularization. In International Conference on Machine Learning (pp. 8525-8535). PMLR.

### Questions
1. How much GPU memory was required to run the llama-2-7B experiments for alternative updating?
2. How many total iterations were performed during the experiments? Was the performance consistently improving throughout the iterations？

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 2

### Summary
The authors in this paper address the limitations of traditional LLM alignment methods, which often rely on static prompt sets pre-collected by human labelers. Current methods need more adaptability to identify areas where LLMs require dynamic improvement. 
The authors propose a two-player game involving an adversarial (tutor) and a defensive (student) LLM to overcome these issues. The adversarial LLM automatically generates challenging prompts designed to expose the defensive LLM's weaknesses, pushing it to adapt and improve iteratively. The iterative adversarial alignment process is shown converge to Nash equilibrium between the adversarial and defensive agents. Moreover, they have also given an algorithm that finds the $O(1/\sqrt{T})$-approximate Nash equilibrium in T iterations.

### Strengths
The problem addressed in this paper is both exciting and novel, offering a fresh approach to LLM alignment. The analysis appears sound, and the proofs seem correct at first glance. However, I have some questions I would like to clarify, as highlighted below.

### Weaknesses
Some things need to be appropriately motivated; for example, $R_{div}(x)$ in eqn 3.1 is defined, but I need to figure out how to obtain this. Only in some sections is it defined, but that is also very restrictive, and how this will be defined or obtained in general needs to be clarified. Look at some more questions regarding this below. 

Different variants of the same algorithms are also hard to parse, and there needs to be a discussion about which algorithm is finally used in the theoretical analysis and why.

Though the paper addresses a good problem, it still lacks some details, and I would like to see more clarity in the revised versions.

My primary concern lies with the motivation and problem formulation. The central motivation here relies on an attack prompt, assuming that an adversarial player controls the prompt distribution. This assumption seems overly strong. In practice, unless prompt optimization techniques or another language model are employed, we have limited control over the prompts users provide.

### Questions
1. What is the motivation behind using the $R_{\text{div}}$​ term in Equation 3.1? Specifically, could you
a. explain how this diversity reward relates to or enhances existing alignment objectives.
b. discuss the advantages of this approach over traditional alignment methods.
c. clarify the general definition or derivation of $R_{\text{div}}$​, as its current form seems restrictive in certain sections.

2. I would like further explanation on including the KL divergence term in Equation 3.3, which is absent in Equation 3.1.

a. Could you introduce the KL divergence term when it first appears in Equation 3.3 and discuss its implications for the overall optimization process?

3. The paper presents at least three algorithm variants, making it unclear which ones are used in the theoretical analysis and implementation and the rationale for this choice. Additionally, given these variants, what are the discrepancies observed between the theoretical and implemented versions?

4. The existence of a Nash equilibrium (NE) is asserted based on the linearity of $J(\pi, \mu)$ in Equation 3.1. However, Algorithm 1 introduces KL terms in its updates, which contradicts this claim. Even without the KL term, could you explain why the $R_{\text{div}}$​ term in Equation 3.1 would be linear?

5.Outside of the attack prompting scenario, it needs to be clarified why minimization over the prompt distribution is necessary. Please clarify specific use cases where the user has the flexibility to control the prompt distribution.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
3

---

## Human Reviewer 3

### Summary
This work proposes to use two-player zero-sum games to perform LLM alignment in safety-critical scenarios. This method iteratively trains a defensive agent and an adversarial agent in turn to adaptively generate progressively harder prompts. The authors provide a theoretical analysis of the method's convergence to Nash equilibrium and perform experiments to show the effectiveness of the method.

### Strengths
1. The idea of using self-play in a two-player zero-sum game to improve LLM alignment is novel and intuitive.
2. The manuscript is well-organized and easy to follow. The main idea of using self-play and introducing diversity is well-explained.
3. The authors provide theoretical analysis as well as experiment results to show the effectiveness of their method.

### Weaknesses
1. More comprehensive evaluation. 
    1. The theoretical result claims that the proposed method can converge to a Nash equilibrium, but there are no experiment results validating this claim. I would suggest the author use metrics like exploitability or NashConv to evaluate how far the current agents are from Nash equilibrium.
    2. The main results in Table 1, and 2 only show performance under a certain amount of training. A more comprehensive evaluation is to show the method's performance curve w.r.t training amount, e.g., the performance curve w.r.t. GPO iteration. This could better compare GPO with baselines like RLHF to show the effect brought by self-play training and show the progressive improvement process of GPO.
2. Need for ethics and social impact statement: this method trains a defensive agent as well as an adversarial agent. Although the authors discuss that the adversarial agent can be utilized for red teaming, it can also be potentially used to make attacks and induce harmful behaviors. However, the authors claim "this work does not involve potential malicious or unintended uses ..." in the ethics statement. I would suggest the authors add necessary discussions on how to prevent potentially harmful use of their method.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
- This paper proposes a 2 player adversarial zero-sum game (GPO) to develop a more robust and less toxic LLM. It consists of a defensive model that generates high quality and safe responses to the prompts generated by the adversarial agent. The adversarial agent generates prompts to try and make the defensive model generate bad or unsafe responses. 
 - As a side effect, the adversarial agent serves as a good red-teaming partner.
 - A diversity parameter in adversarial agent’s reward ensures that a diverse set of prompts are covered during the training process
 - GPO hows strong improvements in safety alignment.

### Strengths
- The authors showcase the effectiveness of GPO and the diversity reward across different safety datasets and attacks. 
- GPO does not seem to harm general quality despite only being used to align the model for safety scenarios. 
- The paper is well written and method is clearly detailed.

### Weaknesses
- A pretrained Llama 2 7B model is used as a base, which then goes through SFT and RLHF. The data used for this isn't specified and it is unclear how the quality of the post-SFT model affect alignment. For example, [Vicuna 7B has a score of 6.00 on MT-Bench](https://lmsys.org/blog/2023-06-22-leaderboard/), which is comparable to the score post GPO. 
 - The paper largely focuses on safety alignment and it is not clear how much GPO would benefit general alignment. 
 - It is not clear how this method generalizes to larger models.

### Questions
- The typical RLHF objective anchors to the intial reference policy. It is not clear why the GPO objective anchors to the policy from the previous step and how this affects this. 
 - Given that the anchor is updated at every step, this would result in a larger policy shift for both the defensive and adversarial agents. How does the RM perform when the prompts generated by the adversarial agent is OOD?

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
4