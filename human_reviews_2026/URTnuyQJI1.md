# A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Diffusion large language models (dLLMs) enable any-order generation, but this flexibility enlarges the attack surface: harmful spans may appear at arbitrary positions, and template-based prefilling attacks such as DIJA bypass response-level refusals.
We introduce A2D (Any-Order, Any-Step Defense), a token-level alignment method that aligns dLLMs to emit an [EOS] refusal signal whenever harmful content arises. By aligning safety directly at the token-level under randomized masking, A2D achieves robustness to both any-decoding-order and any-step prefilling attacks under various conditions. It also enables real-time monitoring: dLLMs may begin a response but automatically terminate if unsafe continuation emerges. On safety benchmarks, A2D consistently prevents the generation of harmful outputs, slashing DIJA success rates from over 80\% to near-zero (1.3\% on LLaDA-8B-Instruct, 0.0\% on Dream-v0-Instruct-7B), and thresholded [EOS] probabilities allow early rejection, yielding up to 19.3× faster safe termination.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes A2D (Any-Order, Any-Step Defense), a novel token-level safety alignment method for diffusion large language models (dLLMs).
Unlike autoregressive (AR) models, dLLMs generate tokens in any order, which enlarges the attack surface and makes them susceptible to new types of jailbreaks such as Prefilling and DIJA attacks.
A2D trains dLLMs to emit an [EOS] refusal signal whenever harmful spans appear, by replacing masked harmful tokens with [EOS] during alignment.
This ensures the model can halt generation upon encountering unsafe continuations—at any position and at any decoding step.
Empirically, A2D drastically reduces DIJA success rates (from >80% to ≈0%) across multiple dLLMs (LLaDA, LLaDA-1.5, Dream) while preserving task capability.
Moreover, A2D introduces a real-time safety signal: the probability of [EOS] acts as an online indicator of harmfulness, enabling early rejection (up to 19.3× faster safe termination).
Overall, this work identifies a core safety vulnerability in dLLMs and proposes a simple yet effective solution through token-level alignment.

### Strengths
- Introduces the first token-level alignment strategy tailored to dLLMs’ flexible decoding process.

- The method is theoretically consistent with masked diffusion objectives and empirically verified through KL divergence analyses.

- Reduces attack success rates to nearly zero across diverse settings while maintaining comparable MMLU, Math, and Coding performance.

- [EOS] probabilities provide a clear, real-time safety signal enabling early refusal.

- Strong performance across decoding orders, generation lengths, and extreme attacks (FITS).

- The paper explicitly verifies that using [EOS] as the refusal token does not harm normal task performance (see Table 3); in fact, it yields the highest capability scores among alternatives while preserving safety.

- The writing and figures are excellent.

### Weaknesses
- While intuitively motivated, the paper lacks a more formal explanation of why [EOS]-based token-level suppression leads to stable refusal boundaries.
- Additional discussion on the training or inference cost overhead of A2D would be helpful.
- Evaluation mainly uses BeaverTails; broader validation across more diverse safety datasets would further strengthen the claims.

### Questions
- Could the [EOS]-based refusal mechanism lead to premature refusals in complex, long-form tasks? A quantitative analysis of false rejections beyond XSTest would be valuable.
- Is the early-rejection threshold τ transferable across models, or does it require per-model calibration?
- Does A2D affect long-context coherence or cause early truncation in benign generations?

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
This paper introduces A2D (Any-Order, Any-Step Defense), a novel token-level safety alignment method specifically designed for diffusion large language models (dLLMs). The authors first identify a critical vulnerability in dLLMs: standard alignment methods are "shallow" and only provide refusal signals at the beginning of a response, making them highly susceptible to template-based prefilling attacks (like DIJA) that inject harmful content at arbitrary positions and steps. A2D addresses this by training the model to predict an [EOS] token (as a universal refusal signal) at any masked position that corresponds to harmful content. This token-level supervision provides robust safety alignment that persists at "any-step" and in "any-order" of the diffusion decoding process.

### Strengths
**Novelty and Significance:** The paper tackles a highly significant and novel problem at the frontier of LLM safety. It clearly diagnoses a critical vulnerability specific to the emerging dLLM paradigm (vulnerability to any-order, any-step attacks) and proposes a solution that is directly tailored to that paradigm.

**Effectiveness and Simplicity:** The A2D method is simple and elegant, cleverly repurposing the existing [EOS] token as a distributed refusal signal. The experimental results are extremely strong, showing a near-complete neutralization of powerful, model-specific attacks (DIJA, FITS) against which other alignment methods (RT, SFT, VRPO) fail.

### Weaknesses
**Potential for Alignment Tax:** While the paper claims model capability is preserved (Tables 1 & 8), the results are not uniformly dominant. For instance, on LLaDA-1.5, A2D scores slightly lower on the "General" capability mean than RT and lower on "Math" than VRPO. A more in-depth analysis of this potential "alignment tax" on more nuanced or complex benign tasks would be beneficial.

**Limited Scope of Harmful Content:** The alignment dataset is based on BeaverTails. It is unclear how A2D would perform against a wider, more diverse range of harmful content categories (e.g., subtle misinformation, complex reasoning-based harms) that may not be well-represented in this dataset.

**Scalability of Training:** The A2D method requires fine-tuning on a dataset of harmful prompts paired with unsafe responses, where harmful spans are identified and masked. It's not clear how this data-generation and training process scales to more complex, "in-the-wild" harmful requests that may not follow clear templates.

### Questions
A2D trains the model to output **[EOS]** for any masked token within a harmful span. How does this "all-or-nothing" token-level refusal handle prompts with mixed intent (i.e., requests that are partially harmful and partially benign)? Does it ever lead to over-refusal, or does it successfully "censor" only the harmful parts while completing the rest?

The "early rejection" mechanism (Section 7) relies on the **[EOS]** probability at the leftmost masked position in the first decoding step. Why was the leftmost position chosen? Given the "any-order" nature of dLLMs, wouldn't an aggregate signal (e.g., the mean or max [EOS] probability across all masked tokens) be a more robust heuristic for refusal?

### Soundness
3

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
This paper introduces a critical safety vulnerability inherent to Diffusion Large Language Models (dLLMs), specifically the fact that their any-order, any-step generation capability greatly expands the attack surface. To address this, the authors propose A2D, a token-level alignment method that trains dLLMs to emit the [EOS] token at masked positions whenever harmful content is encountered. Experiments on three masked dLLMs show A2D outperforms traditional baselines: it reduces attack success rates, preserves core capabilities, and improves monitoring efficiency.

### Strengths
- The paper introduces the first token-level safety alignment method tailored to dLLMs. The proposed A2D method is simple, effective, and addresses two unique vulnerabilities—any-order decoding and any-step adversarial attacks.
- The method strikes a balance between safety and general utility.
- The real-time [EOS] probability monitoring and early rejection mechanism introduces a measurable efficiency benefit.

### Weaknesses
- Lack of Generalization Discussion: The paper focuses exclusively on masked diffusion dLLMs (LLaDA, LLaDA-1.5, Dream). While this is a prominent class, it would strengthen the paper to discuss the theoretical or practical challenges of extending A2D to other LLM architectures.
- Potential for Refusal Token Attack: The use of the single [EOS] token as the universal suppression signal is clever but introduces a single point of failure. A minor discussion on the robustness of the [EOS] signal itself—i.e., the risk of an adversary explicitly targeting or corrupting the refusal behavior of this specific token—would complete the safety analysis.
- Inadequate Assessment: The main experiment is only conducted on the Harmbench, making the assessment of A2D’s generalization to the full spectrum of real-world threats insufficient.
- Limited Practical Value: With few existing landing scenarios for dLLMs, A2D’s practical impact is currently constrained.

### Questions
- Could A2D be adapted to other LLM architectures? 
- A2D relies on [EOS] as the refusal token, but attackers might attempt to induce dLLMs to reduce the generation probability of the targeted token. Have the authors tested A2D’s robustness against such [EOS]-interference attacks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
Recently, the safety alignment of diffusion LLMs (dLLMs) was shown to be easily circumvented by a decoding exploit called DiJA, a generalization of the prefilling attack against autoregressive LLMs to dLLMs. This work proposes a simple fine-tuning defense against DiJA called A2D, where the model learns to output EOS tokens when encountering a DiJA attack, rather than harmful content. Through extensive experiments, A2D is shown to significantly reduce the success of DiJA across multiple models, safety benchmarks and decoding strategies. Various other attacks (e.g., PAIR) are also evaluated, and their success is also shown to significantly degrade under A2D.

### Strengths
1. A KL-divergence-based study to explore the shallowness of the safety alignment of dLLMs is conducted, following the approach of [1]. The study reveals a novel insight specific to dLLMs: that shallow safety aligmment is exhibited under differing orders of mask token denoising (“any order”).
2. The design of the A2D defense is simple to implement and intuitive. Experimental results show that A2D is highly effective at reducing the success of DiJA and prefilling attacks across various decoding strategies, and a followup KL-divergence study shows that deep safety alignment has been effectively learned both in terms of decoding step and order.
3. A “worst-case” version of DiJA called FITS is evaluated to test the scenario where everything in the assistant response except one (masked) sentence contains harmful content. Even under this challenging scenario, A2D effectively reduces the attack success to near zero.

[1] Qi, Xiangyu, et al. "Safety alignment should be made more than just a few tokens deep." arXiv preprint arXiv:2406.05946 (2024).

### Weaknesses
1. During evaluation, there are two types of randomness at play: randomness from sampling from the model’s output probability distribution, and randomness in the decoding strategy. Hence, it would be good to evaluate over multiple runs and report the mean and standard devations.
2. Compliance on XSTest is reported in Figure 5b, yet compliance does not necessarily indicate the outputs are useful under the challenging XSTest setting. Can some evaluation of response quality also be reported for XSTest?
3. Notational discrepancies:
- Line 152-153: “For each selected index i” is mentioned, but then “for all i” is mentioned in the equation. The equation’s “for all i” statement should be modified to indicate only the selected subset of mask tokens.

### Questions
1. Are the data points plotted in Figure 3 averages across the 150 prompts? It would be good to also depict a measure of spread (e.g., standard deviation).
2. In Table 1, can results for the original models (i.e., without additional fine-tuning) also be reported?

### Soundness
4

### Presentation
3

### Contribution
3
