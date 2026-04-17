# Review

## Summary
This paper studies the role of verifiers in the context of RL with LLMs. Specifically, the authors study rule-based and model-based verifiers and their limitations. They find that rule-based verifiers have high precision but low recall, while model-based verifiers have higher recall but are susceptible to reward hacking. The authors suggest a hybrid approach combining both types of verifiers to balance precision and recall.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
- The paper is well written and easy to follow
- The authors study an important problem, namely the role of verifiers in RL
- The authors conduct extensive experiments

## Weaknesses
- The main weakness of the paper is that the main findings are not novel. It is well known that rule-based verifiers have high precision and low recall, while model-based verifiers have higher recall but are susceptible to reward hacking (e.g., [1]). The authors themselves acknowledge that these findings are concurrent with other papers. The main contribution of the paper is the hybrid approach, which seems ad-hoc and not well justified.

[1] Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, and Dong Yu. Expanding rl with verifiable rewards across diverse domains. arXiv preprint arXiv:2503.23829, 2025.

## Questions
- What is the motivation behind the hybrid verifier? Why is it a good idea to combine two verifiers that have such different properties?
- How does the hybrid verifier compare to a model-based verifier that is trained specifically to have high precision and recall?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4