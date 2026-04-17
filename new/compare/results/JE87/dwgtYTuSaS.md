# Review

## Summary
The paper introduces a new task of continuous online action detection from egocentric videos, where the model needs to continuously learn and adapt from streaming videos for action detection in a resource-constrained environment. The authors curate a new benchmark, Ego-OAD, based on the Ego4D dataset, for this task and develop training strategies that enhance adaptation to individual users and generalization to unseen environments. The experimental results on Ego-OAD demonstrate the effectiveness of the proposed method.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper is well-written and easy to follow. 
2. The proposed task is interesting and has practical applications.
3. The curated benchmark is valuable for future research in this area.

## Weaknesses
1. The technical contribution of the proposed method is limited. The proposed COAD training is a simple extension of the method proposed by Carreira et al. (2024a) to the OAD task.
2. The experiment is not comprehensive. The authors only evaluate the proposed method on the Ego-OAD benchmark, which may not be sufficient to demonstrate its effectiveness. The authors should consider adding more experiments on other egocentric video benchmarks, such as EPIC-KITCHENS.
3. The comparison with existing methods is insufficient. The authors only compare the proposed method with two baselines, which may not be sufficient to demonstrate its effectiveness. The authors should consider adding more comparisons with existing OAD methods, such as LSTR and TeSTra mentioned in the related work section.

## Questions
1. Could the authors provide more details on the difference between the proposed COAD training and the method proposed by Carreira et al. (2024a)?
2. Could the authors provide more experimental results on other egocentric video benchmarks, such as EPIC-KITCHENS?
3. Could the authors provide more comparisons with existing OAD methods?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4