# Review

## Summary
The paper conducts a systematic analysis of Group Relative Policy Optimization (GRPO) and identifies two key findings: (1) incorporating negative feedback is essential—training solely on actions above a baseline limits learning; and (2) PPO-style constraints, such as policy ratio clipping, are not required to improve mathematical reasoning or performance. Building on these insights, the authors propose REINFORCE with Group Relative Advantage (RGRA), a simplified variant that retains group-relative advantage estimation but removes PPO-style clipping and policy ratio terms. Experiments across standard mathematical benchmarks indicate that RGRA has the potential to achieve stronger performance than GRPO. Our results suggest that simpler REINFORCE-based approaches can effectively enhance reasoning in LLMs, offering a more transparent and efficient alternative to GRPO.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
The paper is well-written and easy to follow.

The authors propose a simplified variant of GRPO, which achieves better performance than GRPO on some benchmarks.

The paper provides a detailed analysis of the results.

## Weaknesses
1. The paper's novelty is limited. The authors propose REINFORCE with Group Relative Advantage (RGRA), which is a simplified variant of GRPO. The paper claims that incorporating negative feedback is essential and PPO-style constraints are not required. However, these findings are not new and have been discussed in previous papers.

2. The paper does not provide a clear explanation of why RGRA outperforms GRPO. The paper claims that PPO-style constraints are not required and RGRA simplifies training and can lead to improved performance. However, the paper does not provide a clear explanation of why this is the case. Additionally, the paper does not provide a detailed analysis of the differences between GRPO and RGRA, making it difficult to understand why RGRA outperforms GRPO.

3. The paper lacks a detailed analysis of the limitations of RGRA. The paper claims that RGRA offers a more transparent and efficient alternative to GRPO. However, the paper does not provide a detailed analysis of the limitations of RGRA, such as its potential lack of robustness or scalability to larger models or more complex tasks.

## Questions
See the weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4