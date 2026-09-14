# Physics Route Lab: a concrete invitation to contribute

Prepared September 15, 2026. These are draft communications, not evidence that outreach has been sent or that collaborators have joined.

Live demo: https://physics-route-lab-minnatullah.ayan14.chatgpt.site

## Positioning

One quantity. More than one way to find it.

The interactive companion calculates routes from known equations. It does not run the trained GNN or claim to reveal the network's internal reasoning. Its purpose is to make the research question tangible and invite reproducible criticism.

## Launch post

I see physics as a graph of quantities and relationships.

If one measurement is missing, can we still reach the answer another way? For mass, force and acceleration are one route. Momentum and speed are another. Kinetic energy gives us more possibilities.

I'm Md Minnatullah, an independent researcher from Supaul, Bihar. I started this project without access to a laboratory or research funding. My ambition is to teach AI to reason across physics—and to require evidence before claiming anything new.

I've built a small interactive companion where you can hide measurements and inspect the equation routes that remain. This demo uses a symbolic solver, not the trained GNN. The research code and preprint are public separately.

Can you help with one thing: reproduce a result, find a wrong assumption, or contribute a carefully measured experiment?

Try the demo: https://physics-route-lab-minnatullah.ayan14.chatgpt.site

Code and contribution options: https://github.com/mi99at/physics-gnn-symbolic-discovery

Paper: https://zenodo.org/records/22091010

## 60–90 second video outline

1. Speak naturally: "I see physics as a graph. If one connection is unavailable, can we find another route?"
2. Show the mass example, hide force and acceleration, and expand a remaining calculation.
3. Change one available measurement so the answers disagree. Explain why a warning is more useful than hiding the failure.
4. Say clearly: "This page calculates known equations. My GNN research is a separate experiment; the next challenge is to compare it fairly against these routes."
5. Briefly tell your own Bihar/no-lab story in Hindi, English, or both. Add accurate subtitles rather than a manufactured accent or persona.
6. Ask for one contribution: one reproduction, one physics correction, or one independent dataset.

## Small, targeted outreach

Contact at most ten relevant people in the first round, individually. Read their relevant work before choosing a specific request. Do not invent familiarity, imply endorsement, request upvotes, or send repeated generic messages.

Suggested message structure:

"Hello [name], your work on [specific, verified topic] relates to a question I'm testing: can a model recover a physical quantity when one valid observation route is removed? I built an open project and a small symbolic demonstration. Would you be willing to check [one equation assumption / one reproduction command / one proposed measurement protocol]? This is not a claim of new physics. A correction would be genuinely useful. Thank you, Md Minnatullah."

Do not send the placeholder text. Choose recipients and replace the bracketed parts with verified, relevant information first. One follow-up after at least a week is enough; respect nonresponse and opt-outs.

## First-month goals, not predictions

- 10 people try the demonstration and offer substantive feedback.
- 3 independent reports distinguish file validation, checkpoint evaluation, and retraining.
- 1 real-measurement protocol is reviewed and executed with uncertainty and provenance.
- Track substantive issues, reproducibility reports, and actual contributions; do not equate pageviews or downloads with endorsement.

## Scientific work before a broader research launch

1. Separate physical quantity instances: object identity, time, units, and reference frame. Record applicability conditions on equation edges.
2. Audit the known sign-handling issue in discover_v1.py before producing new mining results. Preserve existing published artifacts; document any corrected rerun separately.
3. Benchmark the GNN and a symbolic solver on identical observation masks, including the same known information and physically valid alternatives.
4. Repeat GNN training with independent seeds and matched baseline training budgets. Test retrained graph ablations, not only feature removal from a fixed checkpoint.
5. Publish real-data uncertainty and held-out-condition results, including failures. The current synthetic results do not establish new laws or superior-to-human physics ability.

GitHub Actions account billing remains an account issue, not a code defect; adding workflow files cannot resolve it.
