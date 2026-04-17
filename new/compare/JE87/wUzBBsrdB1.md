# Review

## Summary
The authors study the effect of $L_0$ on SAEs, and show that if $L_0$ is not set correctly, the SAE fails to disentangle the underlying features of the LLM. If $L_0$ is too low, the SAE will mix correlated features to improve reconstruction. If $L_0$ is too high, the SAE finds degenerate solutions that also mix features. Further, the authors present a proxy metric that can help guide the search for the correct $L_0$ for an SAE on a given training distribution.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
The paper is well-written and easy to follow. The authors conduct extensive experiments to demonstrate the effect of $L_0$ on SAEs. The authors also provide a proxy metric that can help guide the search for the correct $L_0$.

## Weaknesses
- The experiments are only conducted on one toy model and one LLM, which may limit the generalizability of the conclusions. It would be better to conduct experiments on more toy models and LLMs.
- The authors claim that the proxy metric can help guide the search for the correct $L_0$, but it may be challenging to use in practice. The metric requires sweeping over different $L_0$ values to find the optimal one, and it is unclear how to efficiently search for the correct $L_0$.
- The authors do not provide a theoretical analysis of the problem, which could provide insights into why the incorrect $L_0$ leads to incorrect features and how to choose the correct $L_0$.

## Questions
- How to efficiently search for the correct $L_0$ using the proxy metric?
- Are there any theoretical results that can support the experimental findings?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4