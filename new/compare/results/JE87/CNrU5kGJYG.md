# Review

## Summary
This paper presents TrojanTO, the first action-level backdoor attack against trajectory optimization (TO) models commonly used in offline reinforcement learning. The paper makes the following contributions: 1) it demonstrates that existing RL backdoor attack techniques are ineffective against TO models, 2) it identifies the key factors influencing TO model security, and 3) it proposes TrojanTO, an alternating training framework for injecting action-level backdoors into TO models with low attack budgets. The paper evaluates TrojanTO's performance across diverse tasks and model architectures, showing its effectiveness, stealthiness, and broad applicability.

## Soundness
3

## Presentation
4

## Contribution
3

## Strengths
1. The paper presents the first systematic study of action-level backdoors in offline RL and introduces a novel post-training attack paradigm against trajectory optimization models, revealing an underexplored threat vector.
2. The authors conduct extensive experiments to evaluate TrojanTO's performance across diverse tasks and model architectures, demonstrating its effectiveness, efficiency, and stealthiness.
3. The authors provide a comprehensive analysis of the key factors influencing TO model security, including action and state selection, trigger design, and reward manipulation, offering valuable insights for future research.
4. The proposed TrojanTO framework utilizes a consistency poisoning strategy and employs trajectory filtering, batch poisoning, and alternating training to inject potent backdoors with a low attack budget, maintaining the agent's normal performance.

## Weaknesses
1. The paper does not provide sufficient details on the implementation of the proposed attack method, making it difficult to reproduce the results.
2. The paper does not thoroughly discuss the potential limitations and challenges of the proposed attack method, such as the difficulty in maintaining the stealthiness of the attack over longer periods of time or across different environments.
3. The paper does not explore the potential countermeasures that can be employed to mitigate the impact of the proposed attack method.

## Questions
1. Could you provide more detailed implementation details of TrojanTO, including the specific algorithms used for trajectory filtering, batch poisoning, and alternating training? How were the trigger values optimized using MI-FGSM?
2. How does TrojanTO maintain the stealthiness of the backdoor over longer periods of time or across different environments? Have you conducted experiments to evaluate the persistence of the backdoor over time?
3. What are the potential countermeasures that can be employed to mitigate the impact of TrojanTO? Have you tested any defense methods against TrojanTO?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4