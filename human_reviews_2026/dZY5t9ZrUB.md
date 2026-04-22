# LLM Watermark Evasion via Bias Inversion

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Watermarking for large language models (LLMs) embeds a statistical signal during generation to enable detection of model-produced text. While watermarking has proven effective in benign settings, its robustness under adversarial evasion remains contested. To advance a rigorous understanding and evaluation of such vulnerabilities, we propose the *Bias-Inversion Rewriting Attack* (BIRA), which is theoretically motivated and model-agnostic. BIRA weakens the watermark signal by suppressing the logits of likely watermarked tokens during LLM-based rewriting, without any knowledge of the underlying watermarking scheme. Across recent watermarking methods, BIRA achieves over 99\% evasion while preserving the semantic content of the original text. Beyond demonstrating an attack, our results reveal a systematic vulnerability, emphasizing the need for stress testing and robust defenses.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present a theoretical and empirical study of vulnerabilities in current watermarking schemes for LLMs. They introduce BIRA, a query-free, model-agnostic method that systematically removes watermarks by applying a negative logit bias to tokens with high self-information, thereby suppressing those most likely to carry the watermark signal. They further provided theoretical analysis showing that reducing the average probability of green tokens by a small margin $\delta$ causes the watermark detection probability to decay exponentially as $\Pr[D(\tilde{y}, W_k)=1] \le e^{-N\delta^2/2}$. They ran extensive experiments across seven watermarking algorithms and multiple LLMs (Llama-3.1, GPT-4o-mini) and show $>99%$ evasion success while maintaining high semantic fidelity and fluency.

### Strengths
- They showed that watermark detection probability decays exponentially with even small reductions in green-token bias, certifying their novelty.
- The proposed approach is both query-free and model-agnostic, requiring no secret knowledge, fine-tuning, or training. 
- Their comprehensive empirical validation across seven watermarking methods and multiple LLM families (Llama, GPT-4o) is enough to drive home this contribution.
- The writing and presentation are clear, structured, and polished.

### Weaknesses
- It is well-known in literature that paraphrasing attacks mostly succeeds in evading detection. So, proposing yet another attack seems like confirming what has already been confirmed.
- Although the self-information heuristic for selecting proxy tokens is intuitive, it remains heuristic; providing a theoretical or empirical justification (e.g., correlation with actual green-token distributions) would reinforce confidence in its generality.
- No defenses were proposed nor evaluated in this case.
- The paper would benefit from an included discussion and evaluation of adaptive paraphrasing attacks [1, 2], which currently achieve the best evasion–fidelity trade-off. Explicitly contrasting BIRA’s query-free design with these adaptive attackers would greatly strengthen the motivation and highlight its advantage of achieving similar success without watermark or detector knowledge.

**References**

[1] Jovanović N, Staab R, Vechev M. Watermark stealing in large language models. arXiv preprint arXiv:2402.19361. 2024 Feb 29.

[2] Diaa A, Aremu T, Lukas N. Optimizing Adaptive Attacks against Watermarks for Language Models. arXiv preprint arXiv:2410.02440. 2024 Oct 3.

### Questions
Please see weaknesses above

### Soundness
3

### Presentation
4

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
This paper presents BIRA, a method for evading LLM watermarks through adversarial paraphrasing. The authors first establish theoretical foundations showing that watermark detection reduces to a threshold test on green token rates, and that suppressing green token probabilities leads to exponential decay in detection probability. Building on this insight, BIRA identifies likely watermarked tokens using self-information, then applies negative logit biases during LLM-based paraphrasing to systematically reduce green token generation. Experiments across seven watermarking schemes demonstrate that BIRA achieves over 99% attack success rates while maintaining semantic fidelity, significantly outperforming existing query-free attacks like DIPPER and SIRA.

### Strengths
1. Unlike previous methods that rely on coarse masking-and-rewriting strategies, BIRA proposes a more fine-grained paraphrasing mechanism by applying token-level negative biases at every generation step. 
2. The method demonstrates consistently high attack success rates across multiple watermarking schemes and different language models (Llama-3.1-8B/70B, GPT-4o-mini). 
3. The method is straightforward to understand and implement. The two-phase approach (proxy green set construction + bias-inversion rewriting) is intuitive, and the use of self-information for identifying likely watermarked tokens is well-motivated.

### Weaknesses
- While the paper claims to preserve semantic fidelity, Table 2 and Table 11 reveal that perplexity increases significantly with stronger attacks (from 9.17 at β=0 to 14.10 at β=-9.0 for Llama models, and notably higher for GPT-4o-mini at 15.5-17.1). The authors acknowledge in Appendix F.3 that GPT-4o-mini produces "stiff text" with "unconventional vocabulary," but this critical limitation receives minimal discussion in the main paper. The trade-off between attack efficacy and text naturalness deserves deeper analysis: At what perplexity threshold does the attacked text become noticeably unnatural to human readers? Could this quality degradation itself serve as a detection signal? The paper would benefit from human evaluation studies assessing whether the perplexity increase translates to perceptible quality loss in real-world scenarios.

- The paper exclusively evaluates BIRA against token-level watermarking schemes where the watermark signal is embedded through biased token sampling. However, recent work has explored alternative watermarking paradigms that operate at different granularities, such as sentence-level watermarks, semantic watermarks that constrain meaning rather than lexical choices, or structural watermarks embedded in discourse patterns. Since BIRA's core mechanism relies on suppressing token-level green tokens identified through self-information, it remains unclear whether this approach generalizes to non-token-based watermarking methods. The authors should discuss these limitations and provide theoretical or empirical analysis on whether BIRA's principles (suppressing watermark signals during rewriting) extend to other watermarking paradigms, or if fundamentally different attack strategies would be required.

### Questions
- Could you provide more detailed analysis on the perplexity increase? Specifically: (a) Have you conducted human evaluation to assess whether the perplexity degradation (especially for GPT-4o-mini with PPL ~15-17) results in noticeably unnatural text that human readers can detect? (b) Could detectors exploit this quality degradation as a secondary signal? For instance, could a two-stage detector first check for watermark absence, then flag texts with suspicious perplexity patterns as potentially attacked? (c) What is the perplexity distribution of genuinely human-written text versus BIRA-attacked text, and is there sufficient overlap to avoid detection?

- How would BIRA perform against watermarking schemes that don't rely on token-level biases? For example: (a) Sentence-level watermarks that embed signals across entire sentences rather than individual tokens? (b) Semantic watermarks that constrain the semantic space rather than lexical choices (c) Have you considered evaluating against or discussing these alternative watermarking paradigms to clarify the scope of BIRA's applicability?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the authors present a bias inversion method for attacking LM watermarking. The method focuses on red-green list styled watermark, in which the watermark is embedded through adding bias to the green list tokens. In the proposed method, we first identify a proxy green token set, then add a negative bias on the logits of the green tokens to erase the watermark.

### Strengths
1. The experiments on various LM watermarking show the effectiveness of the proposed method.

### Weaknesses
The authors missed an important line of watermarking removal works including [1,2], which are very related to their proposed method. The watermarking attacks introduced in [1,2] are almost the same as the proposed methods, they both focus on a) identify the green list and then b) remove the watermark by adding a negative delta to the green list token logits. Moreover, [2] introduced an unbiased green list and watermarking strength estimating algorithm, which can remove the watermark without hurting the generation quality of the language model. Given the existing work [1,2], the proposed method is not novel and there is no methodology contributions.

[1] Jovanovic et al. Watermark Stealing in Large Language Models. ICML 2024.

[2] Chen et al. De-mark: Watermark Removal in Large Language Models. ICML 2025.

### Questions
Is the proposed method superior than [1,2]?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studied the robustness of watermarking in adversarial attack settings. To address the problem and understand it deeply, the authors introduced a method called the bias-inversion rewrite attack, whose application is agnostic to specific models. The experiments show that the method can achieve a high rate of evasion while preserving the original semantic content.

### Strengths
1, The problem setting is clearly presented;

2, The method seems to address the problem well from the experiments

### Weaknesses
The rewriting attack is common in recent literature. It would be great if the authors could show how their algorithm differs from others. The experiments also show that their method is indeed effective. It would also be great to link their theoretical analysis with the experiments.

### Questions
Seems that only 1 dataset is involved in your experiments. Will it be more convincing to conduct experiments on much more datasets?

### Soundness
3

### Presentation
3

### Contribution
2
