# Recruitment Evidence Gap Analysis v0.6

Document status: `0.6-design-draft`

## 1. Method and discipline

This document uses repository materials only to describe present Evidence
readiness. A broad legacy chunk or matching filename is a review lead, not an
approved legal proposition.

Current architecture includes:

- `config/legal_sources.json`;
- legacy `vector_store.json`;
- committed metadata-v2 packs for AI Act Article 6(1) product safety and Data
  Act relevance;
- manifests, stable Evidence IDs, exact citation matching, and legacy fallback.

Recruitment AI Act and GDPR Evidence still comes mainly from the legacy corpus.
It is insufficient for the new formal consequence chain.

## 2. Evidence approval standard

Every implemented proposition requires:

- instrument ID and official authoritative source;
- document and legal-source baseline version;
- atomic canonical citation;
- proposition-specific excerpt;
- authority level and jurisdiction;
- stable source record and Evidence IDs;
- Evidence-pack version;
- manifest entry and hash;
- reviewer mapping to rule, predicate branch, reason code, and permitted
  conclusion.

Guidance must be distinguishable from binding regulation text. No Evidence is
translated or rewritten for presentation.

## 3. Minimum AI Act recruitment pack

Priority pack:

`eu_ai_act_recruitment_metadata_v2.json`

Required target review:

| Target | Proposition to approve before use | Atomic-record requirement | Affected rule | Status |
| --- | --- | --- | --- | --- |
| Article 2 | territorial/material scope proposition selected for later applicability work | each relied-on paragraph separately | temporal/territorial module, deferred | blocked |
| relevant Article 3 provider/deployer/operator definitions | exact definition elements used by narrow deployer relevance | one record per defined term and element | `AI_ACT_DEPLOYER_ROLE_RELEVANCE` | blocked |
| Article 6(2) | exact employment high-risk route | paragraph-level record | existing rule only if future logic changes | partially blocked |
| Article 6(3) | exact exception or profiling consequence actually used | paragraph/subparagraph records | future employment-rule version | blocked |
| Annex III point 4(a) | exact recruitment/selection proposition | atomic point record | employment screen Evidence modernization | partially blocked |
| Article 26(1) | exact use-instructions proposition selected by approved predicate | atomic duty and condition records | `AI_ACT_DEPLOYER_USE_INSTRUCTIONS_RELEVANCE` | blocked |
| Article 26(2) | exact oversight proposition selected by approved predicate | atomic duty and condition records | `AI_ACT_DEPLOYER_HUMAN_OVERSIGHT_RELEVANCE` | blocked |
| Article 14, only if relied upon | system oversight proposition distinct from deployer duty | separate atomic records | oversight rule | blocked |
| Article 113 | temporal application proposition selected for future module | each relied-on paragraph separately | temporal module, deferred | blocked |

The pack includes only propositions approved for the exact implemented scope.
Candidate provisions are not automatically included merely because they may be
relevant.

## 4. Minimum GDPR recruitment pack

Priority pack:

`gdpr_recruitment_metadata_v2.json`

Required target review:

| Target | Proposition to approve before use | Atomic-record requirement | Affected output/rule | Status |
| --- | --- | --- | --- | --- |
| Article 3 | territorial scope propositions | one record per relied-on paragraph | territorial module, deferred | blocked |
| Article 4(4) | profiling definition elements | atomic definition record | future refined Article 22 | blocked |
| Article 4(7) | controller definition elements | atomic definition record | informational hypothesis now; formal role deferred | blocked for formal rule |
| Article 4(8) | processor definition elements | atomic definition record | informational hypothesis now; formal role deferred | blocked for formal rule |
| Article 22(1)–(4) | separate decision, automation, effect, exception, and safeguard propositions | one record per paragraph/subparagraph used | future refined Article 22 and safeguard rules | blocked |
| selected authoritative controller/processor guidance | concrete purpose/means/instruction interpretation approved for projection | proposition-specific passage with authority/version | informational role projection; formal role deferred | blocked for formal rule |
| selected authoritative automated-decision guidance | meaningful human involvement and effect interpretation | proposition-specific passage with authority/version | future refined Article 22 | blocked |

The existing `GDPR_ARTICLE22_RELEVANCE / 2026.1` continues to use the released
legacy Evidence behavior. That compatibility does not approve a complete
Article 22 proposition set.

## 5. Formal-rule gap register

| Proposed rule | Missing provision or source | Intended proposition | Required excerpt | Implementation status |
| --- | --- | --- | --- | --- |
| `AI_ACT_DEPLOYER_ROLE_RELEVANCE` | relevant Article 3 definitions and any exact linked provision | narrow deployer-role relevance for scoped use | atomic definition elements and any linked condition | blocked |
| `AI_ACT_DEPLOYER_USE_INSTRUCTIONS_RELEVANCE` | Article 26(1) | relevance of selected instructions-for-use conditions | separate duty/condition excerpts | blocked |
| `AI_ACT_DEPLOYER_HUMAN_OVERSIGHT_RELEVANCE` | Article 26(2) and Article 14 only where relied upon | relevance of scoped deployer oversight controls | deployer duty separated from system requirement | blocked |
| future refined `GDPR_ARTICLE22_RELEVANCE` | Articles 4(4), 22(1)–(4), selected guidance | complete approved automated-decision predicate | separate statutory and guidance propositions | blocked and deferred |
| `GDPR_ACTOR_ROLE_RELEVANCE` | Articles 4(7), 4(8), selected role guidance, and other approved linked provisions | formal controller/processor/joint-control analysis | proposition-specific records | blocked and deferred |
| `GDPR_ARTICLE22_SAFEGUARD_RELEVANCE` | Article 22 safeguard proposition set | safeguard relevance | one record per safeguard/condition | blocked and deferred |
| `GDPR_DPIA_RELEVANCE` | Article 35 plus approved authoritative criteria | DPIA trigger | statutory trigger and each selected criterion | blocked and deferred |
| temporal/territorial modules | AI Act Articles 2 and 113; GDPR Article 3 | applicability by place/date | each relied-on paragraph | blocked and deferred |

## 6. Informational outputs and Evidence

Minimum v0.6 may show controller-like, processor-like, joint-control-analysis,
and independent-reuse hypotheses as informational projections. They must:

- identify the underlying concrete facts;
- state that legal role classification is not performed;
- avoid Finding status and authoritative styling;
- avoid authorizing formal GDPR consequence rules;
- identify missing reviewed authority where interpretation is material.

Selected guidance may improve explanations only after version and authority
review. It does not convert an informational hypothesis into a formal Finding.

## 7. Deferred Evidence acquisition

The following remain outside minimum formal v0.6:

- complete GDPR role classification;
- Article 22 exception and safeguard rule;
- DPIA rule;
- AI Act logging/monitoring;
- AI Act affected-person information;
- FRIA;
- document-content sufficiency;
- criterion lawfulness, discrimination, and national employment/equality law.

Their sources may be catalogued, but no runtime rule is enabled until its full
proposition set and truth table are approved.

## 8. Temporal and territorial limitation

Facts will collect dates, establishments, use/output locations, affected-person
locations, operation context, and legal-source baseline. No v0.6 minimum rule
evaluates them.

The report therefore displays “Territorial and temporal applicability not
assessed” when those modules are outside scope or facts are incomplete.
Articles 2 and 113 of the AI Act and Article 3 GDPR remain explicit Evidence
gaps for future rules.

## 9. Legal Evidence versus compliance artefact

| Legal Evidence | Compliance artefact |
| --- | --- |
| supports a legal proposition | supports a fact about organizational conduct or records |
| authoritative legal source | assessed organization or counterparty source |
| canonical citation and authority | custodian, version, scope, and review metadata |
| stable metadata-v2 Evidence ID | case-local artefact ID |
| bound to a Finding | linked to facts, requirements, or informational gaps |

Artefact absence does not establish non-compliance. Minimum v0.6 stores only
artefact metadata and an opaque reference; no file content is accepted,
extracted, persisted, or hashed.

## 10. Baseline and fingerprint requirements

The Legal Evidence baseline includes:

- ordered Evidence-pack IDs and versions;
- manifest hashes;
- legal-source baseline ID;
- instrument/document versions;
- expected citation set.

Changing a pack version, manifest hash, source baseline, citation, excerpt, or
stable Evidence ID invalidates affected reports. Evidence baseline validation
occurs before rule execution; unavailable approved Evidence produces
`blocked_by_evidence`, not a substantive Finding.

## 11. Acquisition priorities

1. Approve the exact AI Act employment and deployer-chain propositions.
2. Build and manifest the AI Act recruitment metadata-v2 pack.
3. Atomize GDPR Articles 3, 4(4), 4(7), 4(8), and 22(1)–(4).
4. Select and version authoritative controller/processor and automated-decision
   guidance.
5. Build the GDPR recruitment metadata-v2 pack.
6. Keep formal GDPR role, safeguard, and DPIA rules deferred until their
   complete proposition sets are approved.

## 12. Clean-checkout gate

An Evidence-ready rule requires:

- committed pack and manifest;
- deterministic stable IDs;
- exact citation regression;
- manifest/hash validation;
- clean-checkout availability;
- framework-correct Evidence binding;
- unchanged IDs across English and Simplified Chinese;
- explicit baseline inclusion in the report fingerprint.
