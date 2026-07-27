# EU AI Act Article 6(1) Product-Safety Extension Plan

## 1. Purpose and design decision

This extension adds an independent, preliminary EU AI Act assessment module for the product-safety route in Article 6(1) of Regulation (EU) 2024/1689. It must answer a narrow question:

> Do the confirmed facts indicate that the AI system may fall within the Article 6(1) high-risk classification route because it is a regulated product or a safety component of one, and the relevant product must undergo third-party conformity assessment?

The module is not a product certification, conformity assessment, or definitive legal classification. It does not infer Article 6(1) merely from labels such as “industrial AI”, “regulated product”, “medical AI”, or “connected machinery”.

The recommended implementation has four important properties:

1. it introduces a dedicated `product_regulation` fact namespace instead of further overloading the existing coarse `high_risk` fields;
2. it adds a versioned Annex I reference catalogue, with plain-language product categories linked to exact Annex I entries;
3. it preserves the existing rule, questionnaire, evidence, and report contracts, adding only a backward-compatible conditional-requirement hook;
4. it keeps the new Finding independent from the existing employment, GDPR, and Data Act Findings.

Legal baseline: the official text of [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng), in particular Article 3(14), Article 6(1)(a), Article 6(1)(b), Recitals 50–51, and Annex I. The catalogue must pin its source version and must not silently absorb later amendments.

## 2. Normative structure

Article 6(1) contains a conjunction between points (a) and (b). Point (a) itself contains an alternative route.

### 2.1 Atomic predicates

| ID | Atomic predicate | Legal source | Evaluation rule |
|---|---|---|---|
| `P1A` | The AI system is intended to be used as a safety component of a product. | Article 6(1)(a), read with Article 3(14) | Confirmed `YES` only. A productivity, monitoring, or optimisation function is not automatically a safety function. |
| `P1B` | The AI system is itself a product. | Article 6(1)(a) | Confirmed `YES` only. Being software used with a product does not by itself satisfy this route. |
| `P2` | The relevant product is covered by Union harmonisation legislation listed in Annex I. | Article 6(1)(a), Annex I | Requires an explicit catalogue-backed confirmation, not a free-text assertion that the product is “regulated”. |
| `P3` | That product must undergo third-party conformity assessment before being placed on the market or put into service under the relevant Annex I legislation. | Article 6(1)(b) | Confirmed `YES` only. The existence of a conformity regime or CE marking alone is insufficient. |

The positive predicate is:

```text
(P1A OR P1B) AND P2 AND P3
```

The following are routing or documentation facts, not substitute legal predicates:

- product category;
- industrial or connected-product context;
- the existence of generated product data;
- a generic statement that product law applies;
- the presence of an internal compliance review.

### 2.2 Safety-component interpretation

Article 3(14) should be used as the definition evidence for the safety-component branch. The questionnaire must distinguish:

- a component that performs a safety function or whose failure or malfunction endangers health, safety, or property; from
- analytics, efficiency, maintenance scheduling, quality control, or optimisation that does not itself perform the safety function.

The module records the user's confirmed characterization. It does not use keyword matching to convert “monitoring”, “anomaly detection”, or “predictive maintenance” into `ai_is_safety_component = YES`.

## 3. Proposed fact schema

### 3.1 Namespace

Add a new `ProductRegulationFacts` section to `AssessmentFacts`:

```python
@dataclass(slots=True)
class ProductRegulationFacts(SerializableModel):
    ai_is_product: TriState = TriState.UNKNOWN
    ai_is_safety_component: TriState = TriState.UNKNOWN
    product_type: str | None = None
    annex_i_instrument: str | None = None
    annex_i_instrument_confirmed: TriState = TriState.UNKNOWN
    third_party_conformity_required: TriState = TriState.UNKNOWN
```

The existing `high_risk.is_safety_component_or_product`, `high_risk.product_covered_by_annex_i`, and `high_risk.requires_third_party_conformity_assessment` remain readable for backward compatibility. They must not be silently copied into the new facts, because the first legacy field combines two legally distinct branches and the second does not identify the applicable instrument. Any migration should require explicit user confirmation and record provenance.

### 3.2 Fact definitions

| Fact path | Type and allowed values | Unknown behavior | Dependencies and invalidation | English label and help | Simplified Chinese label and help | Documentary confirmation |
|---|---|---|---|---|---|---|
| `product_regulation.ai_is_product` | `TriState`: `yes`, `no`, `unknown` | If the safety-component branch is not `YES`, `UNKNOWN` blocks a negative conclusion. If the safety-component branch is `YES`, this field need not block the positive route. | No prerequisite. A material change clears product type, Annex I selection/confirmation, third-party conformity, stale report, and this module's provenance. | **Is the AI system itself a regulated product?** Help: Select Yes only where the AI system itself is placed on the market or put into service as a product governed by applicable EU product legislation. | **该 AI 系统本身是否属于受监管产品？** 帮助：仅当该 AI 系统本身作为受相关欧盟产品法规约束的产品投放市场或投入使用时选择“是”。 | Advisable: product classification, technical file, declaration, or legal/product-compliance review. |
| `product_regulation.ai_is_safety_component` | `TriState` | If the product branch is not `YES`, `UNKNOWN` prevents a negative conclusion. If the product branch is `YES`, it does not block the positive route. | No prerequisite. A material change invalidates all downstream product-law facts and reports for this module. | **Is the AI used to perform a safety function within a product?** Help: Consider whether it performs a safety function, or whether its failure or malfunction could endanger people or property. Monitoring or optimisation alone is not enough. | **该 AI 是否在产品中承担安全功能？** 帮助：请判断其是否直接承担安全功能，或其失效、故障是否可能危及人员健康、安全或财产。仅用于监测或优化并不足够。 | Strongly advisable: hazard analysis, safety architecture, intended-purpose statement, FMEA or equivalent. |
| `product_regulation.product_type` | Controlled catalogue ID or normalized string; `None` means unknown | Missing only after either Article 6(1)(a) branch is confirmed. It is a routing/documentation fact, not an independent positive predicate. | Depends on either `ai_is_product = YES` or `ai_is_safety_component = YES`. A change clears Annex I instrument, confirmation, third-party conformity, and stale report. | **What type of product is involved?** Help: Choose the closest business-facing category, such as machinery, medical device, lift, radio equipment, vehicle, or pressure equipment. | **涉及哪类产品？** 帮助：请选择最接近的业务类别，例如机械、医疗器械、电梯、无线电设备、车辆或压力设备。 | Advisable: product specification and regulatory classification. |
| `product_regulation.annex_i_instrument` | Stable Annex I catalogue `instrument_id`; `None` means none selected or unknown | Required only where coverage is confirmed `YES`. Free text is not accepted as the canonical value. | Depends on product type and an Article 6(1)(a) branch. A change clears confirmation, third-party conformity, and report. | **Which listed EU product law covers this product?** Help: Select the specific law after reviewing the product category. The interface shows title and product context; users do not need to know Annex I numbering. | **哪一项所列欧盟产品法规适用于该产品？** 帮助：请结合产品类别选择具体法规。界面会显示法规名称和产品场景，无需用户掌握附件 I 编号。 | Required for a positive result: applicable legislation, declaration, certificate, or counsel/product-compliance confirmation. |
| `product_regulation.annex_i_instrument_confirmed` | `TriState` | `UNKNOWN` means coverage has not been verified and blocks evaluation beyond this point. `YES` requires a selected catalogue instrument. `NO` means a documented review concluded that no Annex I instrument covers the product; it must be stored with `annex_i_instrument = None`. | Depends on product type and at least one Article 6(1)(a) branch. Changing away from `YES` clears third-party conformity. `YES` without an instrument and `NO` with an instrument are inconsistent. | **Has Annex I coverage been confirmed?** Help: Confirm only after identifying and checking the applicable listed product law. Select No only after a review concludes that none of the listed laws covers the product. | **是否已确认该产品属于附件 I 所列法规范围？** 帮助：仅在识别并核对具体产品法规后确认。“否”仅用于经审查确认没有任何所列法规适用的情形。 | Strongly advisable; the source and reviewer should be captured in `fact_metadata`. |
| `product_regulation.third_party_conformity_required` | `TriState` | Required only after Annex I coverage is confirmed `YES`. `UNKNOWN` prevents a Finding. | Depends on confirmed Annex I instrument. Instrument or coverage changes clear this fact and report. | **Must an independent third party assess conformity before market placement or putting into service?** Help: Answer for this product and its applicable conformity route. Do not answer Yes merely because CE marking or a conformity regime exists. | **该产品在投放市场或投入使用前是否必须由独立第三方进行合格评定？** 帮助：请针对该产品适用的具体合格评定路径作答。仅存在 CE 标志或合格评定制度并不当然意味着“是”。 | Required for a positive result: notified-body route, certificate, conformity module, or documented product-law analysis. |

### 3.3 Provenance

The existing `fact_metadata` mechanism is sufficient for source, question ID, and recording time. The questionnaire's `FactProvenance` should additionally retain module ownership and dependencies. The UI should label documentary confirmation as advisable but must not imply that uploading a document has been implemented.

### 3.4 Conditional required facts

Article 6(1) cannot be modeled cleanly as one unconditional tuple of six required paths. Requiring downstream product-law answers after both Article 6(1)(a) branches are `NO` would create unnecessary questions; treating missing downstream values as `NO` would create legally misleading facts.

Add a backward-compatible hook:

```python
class AssessmentRule:
    required_fact_paths: tuple[str, ...]

    def required_fact_paths_for(
        self, facts: AssessmentFacts
    ) -> tuple[str, ...]:
        return self.required_fact_paths
```

`FactRequirementValidator` should call this method. Existing rules inherit unchanged behavior. The new rule returns the following deterministic requirement path:

1. require enough of `ai_is_product` and `ai_is_safety_component` to resolve their OR condition;
2. if either is `YES`, require `product_type` and `annex_i_instrument_confirmed`;
3. if coverage is `YES`, require `annex_i_instrument` and `third_party_conformity_required`;
4. if both Article 6(1)(a) branches are `NO`, no downstream fact is required;
5. if coverage is `NO`, third-party conformity is not required.

The static `required_fact_paths` remains the complete superset for registry validation, questionnaire ownership, documentation, and report metadata.

## 4. Rule definition

### 4.1 Metadata

```text
class: AIActHighRiskProductSafetyRule
rule_id: AI_ACT_HIGH_RISK_PRODUCT_SAFETY
framework: EU_AI_ACT
category: HIGH_RISK_ARTICLE_6_1
version: 2026.1
issue_code: AIA_HIGH_RISK_ARTICLE_6_1_PRELIMINARY
requires_legal_review on positive/undetermined: true
```

Static required-fact superset:

```text
product_regulation.ai_is_product
product_regulation.ai_is_safety_component
product_regulation.product_type
product_regulation.annex_i_instrument
product_regulation.annex_i_instrument_confirmed
product_regulation.third_party_conformity_required
```

### 4.2 Outcomes

| Situation | Output | Reason codes |
|---|---|---|
| `(AI is product OR AI is safety component) AND confirmed Annex I coverage AND third-party conformity required` | `potentially_applies` | `AI_IS_PRODUCT` and/or `AI_IS_SAFETY_COMPONENT`; `ANNEX_I_COVERAGE_CONFIRMED`; `THIRD_PARTY_CONFORMITY_REQUIRED` |
| Both product and safety-component branches are confirmed `NO` | `does_not_apply` | `NEITHER_AI_PRODUCT_NOR_SAFETY_COMPONENT` |
| At least one Article 6(1)(a) branch is `YES`, but review confirms no Annex I instrument covers the product | `does_not_apply` | `NO_ANNEX_I_INSTRUMENT_CONFIRMED` |
| Article 6(1)(a) and Annex I coverage are satisfied, but third-party conformity is confirmed `NO` | `does_not_apply` | `NO_THIRD_PARTY_CONFORMITY_REQUIREMENT` |
| Required fact is `UNKNOWN`, absent, or blank | No legal Finding; existing structured missing-information output | Missing path and `MissingFactReason` from the requirement layer |
| Structurally inconsistent complete facts | `undetermined` | One or more of `INCONSISTENT_ANNEX_I_CONFIRMATION`, `INCONSISTENT_DOWNSTREAM_PRODUCT_FACTS`, `INCONSISTENT_PRODUCT_RELATION` |

Negative wording must state only that this Article 6(1) product-safety route is not met on the confirmed facts. It must not exclude Article 6(2), another Annex III category, or another applicable law.

### 4.3 Finding trace

The trace should contain one entry per legal predicate, not one entry per questionnaire control:

1. **Product relationship** — `ai_is_product` and `ai_is_safety_component`; result records which OR branch was satisfied.
2. **Annex I coverage** — product type, catalogue instrument, and confirmation.
3. **Third-party conformity** — the confirmed product-specific requirement.

The selected catalogue ID and catalogue version belong in Technical details. The readable product category and legal title belong in the main decision path.

### 4.4 Human-readable presentation

Positive:

- EN title: **Product-safety high-risk classification potentially applies**
- EN summary: **The confirmed facts indicate that the AI system is itself a covered product or performs a safety function within one, the product is covered by a listed Annex I instrument, and third-party conformity assessment is required. Article 6(1) may therefore classify the AI system as high-risk. This is a preliminary screening result and requires product-law and legal review.**
- zh-CN title: **产品安全路径的高风险分类可能适用**
- zh-CN summary: **现有确认事实表明，该 AI 系统本身可能属于相关产品，或在该产品中承担安全功能；该产品属于附件 I 所列法规范围，且需要第三方合格评定。因此，Article 6(1) 的高风险分类路径可能适用。本结论仅为初步筛查，仍需产品法规及法律复核。**

Negative:

- EN title: **Article 6(1) product-safety screening criteria not met**
- zh-CN title: **Article 6(1) 产品安全路径初筛条件未满足**

Undetermined:

- EN title: **Article 6(1) assessment requires fact reconciliation**
- zh-CN title: **Article 6(1) 评估需要核对不一致事实**

## 5. Annex I catalogue

### 5.1 Recommendation: catalogue and atomic Evidence records

Use both:

1. a structured catalogue for deterministic selection, aliases, product labels, status, and validation; and
2. one metadata-v2 Evidence record for every individual Annex I entry.

The catalogue answers “which entry does this business fact refer to?” Evidence answers “what authoritative text supports that entry?” Neither should replace the other.

### 5.2 Catalogue structure

Create `config/ai_act_annex_i_instruments.json` with a versioned schema:

```json
{
  "catalog_schema_version": "1.0.0",
  "ai_act_document_version": "Regulation (EU) 2024/1689, OJ 12 July 2024",
  "effective_as_of": "2026-07-14",
  "entries": [
    {
      "instrument_id": "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
      "annex_section": "A",
      "annex_point": 1,
      "official_title": "Directive 2006/42/EC on machinery",
      "instrument_number": "Directive 2006/42/EC",
      "product_category": "machinery",
      "status": "listed",
      "official_citation": "Annex I, Section A, point 1",
      "official_uri": "https://eur-lex.europa.eu/eli/dir/2006/42/oj",
      "aliases": ["machinery", "industrial machinery", "machine safety"],
      "display_label_en": "Machinery — Directive 2006/42/EC",
      "display_label_zh_cn": "机械 — Directive 2006/42/EC"
    }
  ]
}
```

Recommended `status` values are `listed`, `superseded`, `removed`, and `pending_review`. Status changes require an authored catalogue migration and tests; they must never be inferred from aliases. Add `effective_from` and `effective_to` when an Annex amendment or product-law transition makes temporal selection material.

### 5.3 Initial catalogue coverage

The first catalogue version should contain all 20 entries in the enacted Annex I snapshot, not only the demo categories:

| Annex entry | Product context | Instrument |
|---|---|---|
| A/1 | Machinery | Directive 2006/42/EC |
| A/2 | Toys | Directive 2009/48/EC |
| A/3 | Recreational craft and personal watercraft | Directive 2013/53/EU |
| A/4 | Lifts and lift safety components | Directive 2014/33/EU |
| A/5 | Equipment for potentially explosive atmospheres | Directive 2014/34/EU |
| A/6 | Radio equipment | Directive 2014/53/EU |
| A/7 | Pressure equipment | Directive 2014/68/EU |
| A/8 | Cableway installations | Regulation (EU) 2016/424 |
| A/9 | Personal protective equipment | Regulation (EU) 2016/425 |
| A/10 | Appliances burning gaseous fuels | Regulation (EU) 2016/426 |
| A/11 | Medical devices | Regulation (EU) 2017/745 |
| A/12 | In vitro diagnostic medical devices | Regulation (EU) 2017/746 |
| B/13 | Civil aviation security | Regulation (EC) No 300/2008 |
| B/14 | Two- or three-wheel vehicles and quadricycles | Regulation (EU) No 168/2013 |
| B/15 | Agricultural and forestry vehicles | Regulation (EU) No 167/2013 |
| B/16 | Marine equipment | Directive 2014/90/EU |
| B/17 | Rail-system interoperability | Directive (EU) 2016/797 |
| B/18 | Motor-vehicle type approval | Regulation (EU) 2018/858 |
| B/19 | Motor-vehicle general safety | Regulation (EU) 2019/2144 |
| B/20 | Civil aviation, including the Annex I limitation for covered unmanned aircraft contexts | Regulation (EU) 2018/1139 |

The product labels are navigation aids only. Selecting a label does not confirm legal coverage or third-party conformity.

## 6. Legal basis and Evidence design

### 6.1 Atomic references

The rule should use only `instrument="EU_AI_ACT"` and exact canonical citations:

- `Article 3(14)` — only when the safety-component branch is relevant;
- `Article 6(1)(a)`;
- `Article 6(1)(b)`;
- the selected entry, for example `Annex I, Section A, point 1`.

Recitals 50 and 51 may be supplemental interpretive Evidence but should not replace the operative provisions. Avoid `Article 6`, `Article 6(1)(a)-(b)`, or generic `Annex I` where atomic records are available.

### 6.2 Metadata-v2 corpus requirement

The current legacy AI Act store contains a combined `Article 6(1)` record and broad `Annex I` chunks. That is sufficient for legacy retrieval but not for this extension's atomic traceability requirement. Build a separate, isolated metadata-v2 AI Act product-safety candidate corpus containing:

- Article 3(14);
- Article 6(1)(a);
- Article 6(1)(b);
- Recitals 50 and 51;
- 20 atomic Annex I entry records.

Each record must contain `instrument_id=EU_AI_ACT`, document version, exact canonical citation, binding authority for Articles/Annex entries, source record ID, and stable Evidence ID. Do not rewrite `vector_store.json`; configure the existing multi-corpus retriever with the new candidate path after the legacy store.

### 6.3 Rule-authored evidence isolation

The static rule metadata declares Article 6(1)(a) and Article 6(1)(b). The Finding adds Article 3(14) only for the safety-component branch and adds the exact Annex I citation resolved from the selected catalogue ID.

Because the current demo factory preloads evidence from static rule metadata before a case is assessed, Phase D should add a small rule-evidence manifest or preload the active Annex I catalogue citations specifically for `AI_ACT_HIGH_RISK_PRODUCT_SAFETY`. Exact Finding legal bases still control binding. The following safeguards are mandatory:

- only `EU_AI_ACT` records are eligible;
- the selected catalogue entry determines the Annex citation;
- Evidence is matched by exact canonical citation and source;
- no record from GDPR, Data Act, or an external sectoral-law corpus can satisfy the reference;
- the Finding keeps the requested citation and atomic Evidence keeps its metadata-v2 stable ID;
- evidence declared for another rule is not added merely because it shares a keyword.

The catalogue may link to the official sectoral instrument for user verification, but this extension does not ingest or make determinations under those external instruments. The fact that third-party assessment is required remains a confirmed case fact requiring legal/product-compliance review.

## 7. Dynamic questionnaire integration

### 7.1 Companion definition

Add a `RuleQuestionnaireDefinition` keyed by `AI_ACT_HIGH_RISK_PRODUCT_SAFETY`:

```text
framework: EU_AI_ACT
display_module_key: module.ai_act.product_safety
confirmation_question_id: CONFIRM-MODULE::AI_ACT_HIGH_RISK_PRODUCT_SAFETY
supported_domains: PRODUCT_SAFETY, OTHER
required_fact_paths: complete static superset from section 4
```

Stable question IDs:

| Question ID | Fact path |
|---|---|
| `AI-ACT-6-1-AI-IS-PRODUCT` | `product_regulation.ai_is_product` |
| `AI-ACT-6-1-AI-SAFETY-COMPONENT` | `product_regulation.ai_is_safety_component` |
| `AI-ACT-6-1-PRODUCT-TYPE` | `product_regulation.product_type` |
| `AI-ACT-6-1-ANNEX-I-INSTRUMENT` | `product_regulation.annex_i_instrument` |
| `AI-ACT-6-1-ANNEX-I-CONFIRMATION` | `product_regulation.annex_i_instrument_confirmed` |
| `AI-ACT-6-1-THIRD-PARTY-CONFORMITY` | `product_regulation.third_party_conformity_required` |

### 7.2 Routing hints

Controlled hints:

```text
ai_act.regulated_product
ai_act.product_safety_component
ai_act.annex_i_product_context
ai_act.third_party_conformity
product.machinery
product.medical_device
product.vehicle
```

The existing `ai_act.product_safety_component` hint can be retained. Any additional hints must originate from explicit structured intake or audited deterministic normalization mappings. Raw keyword matches must not activate the module.

Suggested eligibility groups:

1. `use_context.domain = PRODUCT_SAFETY` plus a confirmed product-safety or regulated-product hint;
2. `ai_is_product = YES`;
3. `ai_is_safety_component = YES`;
4. a catalogue-backed product category plus an explicit Annex I/product-conformity hint.

Industrial machinery, medical devices, and similar terms may suggest the module, but suggestion creates only a confirmation step. It must not create a Finding or set any Article 6(1) fact to `YES`.

### 7.3 Dependencies and invalidation

```text
AI product / safety-component relation
  -> product type
  -> Annex I instrument selection
  -> Annex I coverage confirmation
  -> third-party conformity requirement
```

Invalidation rules:

- changing either product-relation fact invalidates downstream product facts, this module's run/report, and dependent provenance;
- changing product type clears instrument, confirmation, and third-party facts;
- changing instrument clears confirmation and third-party facts;
- changing coverage away from `YES` clears third-party facts;
- removing or declining the module removes it from route completion but need not delete otherwise valid case facts;
- language changes affect labels only and preserve canonical facts, route, progress, report, citations, and Evidence IDs.

If the module is recognized but the catalogue lacks the user's product category, show an unsupported/ambiguous path with a neutral message and allow manual legal review. Do not guess the nearest instrument.

### 7.4 Module confirmation copy

- EN: **Assess the AI Act product-safety high-risk route?** Help: This module checks whether the AI is itself a listed regulated product or performs a safety function within one, and whether third-party conformity assessment is required.
- zh-CN: **是否评估《欧盟人工智能法案》的产品安全高风险路径？** 帮助：本模块将核查 AI 是否本身属于所列受监管产品，或是否在该产品中承担安全功能，以及是否需要第三方合格评定。

## 8. Business-facing questionnaire copy

The questionnaire must not ask only “Is Annex I applicable?”

| Order | English | Simplified Chinese |
|---|---|---|
| 1 | Is the AI system itself placed on the market or put into service as a product governed by EU product-safety legislation? | 该 AI 系统本身是否作为受欧盟产品安全法规约束的产品投放市场或投入使用？ |
| 2 | Is the AI system intended to perform a safety function within another product, or could its failure or malfunction endanger people or property? | 该 AI 系统是否拟在另一产品中承担安全功能，或其失效、故障是否可能危及人员或财产？ |
| 3 | What type of product is involved? | 涉及哪类产品？ |
| 4 | Based on that product category, which listed EU product law may apply? | 根据该产品类别，哪一项所列欧盟产品法规可能适用？ |
| 5 | Has a product-compliance owner or legal reviewer confirmed that this product is covered by that listed law? | 产品合规负责人或法律审核人员是否已确认该产品属于该项所列法规范围？ |
| 6 | Under the applicable conformity route, must an independent third party assess the product before it is placed on the market or put into service? | 根据适用的合格评定路径，该产品在投放市场或投入使用前是否必须由独立第三方进行评定？ |

Every question includes `Unknown / Not yet confirmed`. The UI should recommend reviewing the product technical file, applicable legislation, notified-body route, and conformity documentation.

## 9. Industrial demo extension

The current Industrial AI demo remains Data Act-positive only. Its existing coarse high-risk facts are `UNKNOWN`; they must not be interpreted as an Article 6(1) result.

### 9.1 Independent modules

| Module | Decisive facts | Finding isolation |
|---|---|---|
| EU Data Act relevance | connected product or related service, plus generated data | `EU_DATA_ACT_RELEVANCE`; Data Act legal bases and Evidence only |
| AI Act Article 6(1) | AI product/safety-component relation, confirmed Annex I coverage, third-party conformity | `AI_ACT_HIGH_RISK_PRODUCT_SAFETY`; Article 6(1)/Annex I Evidence only |

The report may contain both Findings, but each keeps its own rule ID, framework, issue code, reason codes, legal basis, Evidence binding, and missing-information scope.

### 9.2 Fixture variants

Do not silently change the existing fixture into a positive Article 6(1) case. Add explicit variants or a versioned extension block:

**Unresolved Article 6(1)**

```json
{
  "ai_is_product": "unknown",
  "ai_is_safety_component": "unknown",
  "product_type": "industrial_machinery",
  "annex_i_instrument": null,
  "annex_i_instrument_confirmed": "unknown",
  "third_party_conformity_required": "unknown"
}
```

Expected: Data Act Finding remains positive; Article 6(1) produces structured missing information and no legal Finding.

**Negative Article 6(1)**

```json
{
  "ai_is_product": "no",
  "ai_is_safety_component": "no",
  "product_type": "industrial_machinery",
  "annex_i_instrument": null,
  "annex_i_instrument_confirmed": "unknown",
  "third_party_conformity_required": "unknown"
}
```

Expected: Article 6(1) `does_not_apply` because the first predicate is resolved negatively; downstream unknown values do not block that conclusion.

**Potentially applicable Article 6(1)**

```json
{
  "ai_is_product": "no",
  "ai_is_safety_component": "yes",
  "product_type": "industrial_machinery",
  "annex_i_instrument": "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
  "annex_i_instrument_confirmed": "yes",
  "third_party_conformity_required": "yes"
}
```

Expected: separate Data Act and Article 6(1) `potentially_applies` Findings. The machinery instrument and third-party route must be supported by case-specific documentary review; the fixture must state that this is a demonstration assumption, not a general proposition about all machinery.

## 10. Report and Evidence Trace presentation

### 10.1 Primary conclusion

Display the preliminary conclusion first, scoped to “AI Act — Article 6(1) product-safety route”. Avoid “certified”, “compliant”, “conformity approved”, or an unqualified “is high-risk”.

### 10.2 Decision path

```text
Product relationship
  AI is product: No
  AI is safety component: Yes
        ↓
Annex I coverage
  Machinery — Directive 2006/42/EC
  Coverage confirmed: Yes
        ↓
Third-party conformity
  Required before market placement / putting into service: Yes
        ↓
Preliminary result
  Article 6(1) potentially applies — legal review required
```

### 10.3 Evidence Trace

Preserve the existing four-stage hierarchy:

1. **Facts** — actual values and provenance;
2. **Rule application** — `(product OR safety component) AND Annex I AND third-party`, with an overall predicate count and result;
3. **Legal basis** — Article 3(14) where relevant, Article 6(1)(a), Article 6(1)(b), selected Annex I entry;
4. **Source Evidence** — atomic metadata-v2 excerpts and stable IDs.

Condition-to-fact mapping remains collapsed progressive disclosure. Technical details contain raw paths, canonical catalogue ID, catalogue version, question IDs, provenance IDs, rule ID/version, issue code, and reason codes.

### 10.4 Other framework screens

GDPR and Data Act screens remain visible for audit but independent. Missing Article 6(1) facts appear only in the Article 6(1) screen and do not make a completed Data Act module appear incomplete. Conversely, Data Act connected-product facts must not satisfy Article 6(1).

## 11. Boundary behavior

| Scenario | Expected Article 6(1) behavior |
|---|---|
| Regulated product + AI safety component + Annex I coverage confirmed + third-party conformity required | `potentially_applies`; legal review required. |
| Regulated product, but AI is not itself the product and is not a safety component | `does_not_apply`; reason `NEITHER_AI_PRODUCT_NOR_SAFETY_COMPONENT`. |
| AI is a safety component, but applicable legislation is unknown | No legal Finding; request product type, Annex I instrument, and confirmation. |
| Annex I legislation applies, but no third-party conformity assessment is required for the applicable route | `does_not_apply`; reason `NO_THIRD_PARTY_CONFORMITY_REQUIREMENT`. |
| AI performs only productivity, efficiency, analytics, or optimisation functions | Do not infer safety-component status. If both Article 6(1)(a) branches are confirmed `NO`, return `does_not_apply`; otherwise request confirmation. |
| Connected machinery is Data Act-relevant, but Article 6(1) facts are unresolved | Data Act Finding remains; Article 6(1) shows scoped missing information and no Finding. |
| Connected machinery satisfies both modules | Two independent `potentially_applies` Findings with separate legal bases and Evidence. |
| Coverage is `YES` but no Annex I instrument is selected | Missing/inconsistent input; no positive conclusion. |
| Coverage is `NO` while an instrument remains selected | `undetermined` due to inconsistent facts; require reconciliation. |
| Third-party conformity is `YES` while Annex I coverage is unconfirmed | `undetermined` or blocked by consistency validation; never positive. |

## 12. Test matrix

| Area | Test | Expected assertion |
|---|---|---|
| Positive rule | Safety component + confirmed Annex I instrument + third-party required | One Article 6(1) `potentially_applies` Finding with exact reasons and preliminary language. |
| Alternative positive | AI itself is product + confirmed Annex I instrument + third-party required | Positive without requiring safety-component `YES`. |
| Negative relation | Both Article 6(1)(a) branches `NO` | `does_not_apply` without downstream questions. |
| Negative coverage | At least one branch `YES`, coverage confirmed `NO` | `does_not_apply`; third-party fact not required. |
| Negative conformity | Coverage confirmed `YES`, third-party requirement `NO` | `does_not_apply`. |
| Unknown OR branch | Product `NO`, safety component `UNKNOWN` | No Finding; missing safety-component fact. |
| Short-circuit OR | Product `YES`, safety component `UNKNOWN` | Continue to downstream facts; unknown alternative does not block. |
| Unknown instrument | Safety component `YES`, coverage `YES`, instrument absent | No Finding; instrument identified as missing. |
| Inconsistent facts | Coverage `NO` with instrument selected, or third-party `YES` before coverage | `undetermined` or explicit consistency error; never positive. |
| Product-category invalidation | Change machinery to medical device | Clear instrument, coverage confirmation, third-party fact, report, and dependent provenance. |
| Instrument invalidation | Change Annex I selection | Clear confirmation, third-party fact, and report. |
| Bilingual equivalence | Equivalent EN and zh-CN structured answers | Identical canonical facts, route, Finding, citations, and Evidence IDs. |
| Routing | Product-safety hint or confirmed branch | Module suggested but no Finding before confirmation/run. |
| Unsupported category | Product category absent from catalogue | Neutral unsupported/ambiguous route; no guessed instrument. |
| Requirement isolation | Article 6(1) incomplete; Data Act complete | Data Act progress remains complete; Article 6(1) missing facts remain scoped. |
| Evidence source isolation | Same citation-like text exists in another corpus | Only `EU_AI_ACT` Evidence binds. |
| Atomic citations | Positive safety-component case | Article 3(14), Article 6(1)(a), Article 6(1)(b), and selected Annex I entry bind separately. |
| Stable Evidence IDs | Rebuild unchanged product-safety candidate corpus | Identical metadata-v2 IDs. |
| Industrial unresolved | Existing Industrial fixture plus unconfirmed module | Data Act result/evidence unchanged; no Article 6(1) positive Finding. |
| Industrial dual-framework | Explicit positive Article 6(1) variant | Data Act and AI Act Findings and bindings remain independent. |
| Recruitment regression | Existing recruitment fixture | Employment Finding and seven legacy Evidence records unchanged. |
| GDPR regression | Existing complete GDPR scenario | Article 22 Finding and Evidence unchanged. |

## 13. Implementation plan and expected files

### Phase A — facts and Annex I catalogue foundation

Create:

- `config/ai_act_annex_i_instruments.json`
- `src/assessment/product_regulation/__init__.py`
- `src/assessment/product_regulation/catalog.py`
- `src/assessment/product_regulation/models.py`
- `tests/assessment/product_regulation/__init__.py`
- `tests/assessment/product_regulation/test_annex_i_catalog.py`

Modify:

- `src/assessment/facts.py` — add `ProductRegulationFacts` and the new namespace with unknown defaults;
- `src/assessment/__init__.py` — export only public fact/catalog types if consistent with current exports;
- `tests/assessment/test_facts.py` — serialization, unknown defaults, provenance, existing fixture compatibility;
- `src/assessment/rules/base.py` — add default `required_fact_paths_for()`;
- `src/assessment/requirements.py` — validate the conditional path set;
- relevant requirement tests — existing rules must produce identical requirement results.

Acceptance: all existing fixtures and rules remain unchanged; catalogue IDs, aliases, ordering, and version are deterministic.

### Phase B — Article 6(1) rule and tests

Create:

- `src/assessment/rules/ai_act_product_safety.py`
- `tests/assessment/rules/test_ai_act_product_safety.py`

Modify:

- `src/assessment/rules/__init__.py`
- `src/assessment/demo/factory.py` — register the new rule in deterministic order;
- `tests/assessment/demo/test_factory.py`
- report tests as needed for a second independent AI Act Finding.

Acceptance: positive, negative, unknown, short-circuit, and inconsistent tests pass; no other rule output changes.

### Phase C — questionnaire registry and routing

Modify:

- `src/assessment/questionnaire/definitions.py` — stable questions, hints, companion definition, dependencies, invalidations;
- `src/assessment/questionnaire/router.py` — register the new rule; no routing algorithm change should otherwise be needed;
- `src/assessment/questionnaire/invalidation.py` only if conditional invalidation cannot be represented by existing metadata;
- `src/assessment/questionnaire/__init__.py` — public constants if required;
- `src/ui/questionnaire.py` — catalogue-backed options and canonical coercion;
- `src/ui/normalization.py` — only audited controlled product-context mappings;
- `src/ui/i18n.py` — bilingual questions, module copy, hints, Finding, trace, recommendations;
- `assessment_app.py` — render catalogue options and documentary-review guidance using existing components.

Test:

- `tests/assessment/questionnaire/test_router.py`
- `tests/assessment/questionnaire/test_invalidation.py`
- `tests/ui/test_questionnaire.py`
- `tests/ui/test_normalization.py`
- `tests/ui/test_i18n.py`
- `tests/ui/test_assessment_app.py`

Acceptance: suggestions remain non-conclusive; confirmation controls execution; dependency changes remove stale facts and reports; language changes are state-neutral.

### Phase D — atomic Evidence binding

Create:

- `src/ai_act_product_safety_corpus.py`
- `scripts/build_ai_act_product_safety_candidate_corpus.py`
- `tests/test_ai_act_product_safety_corpus.py`

Modify:

- `src/corpus_enrichment.py` — recognize `Annex I, Section X, point N` if not already supported;
- `src/assessment/evidence/citations.py` — only if exact canonical validation/normalization is needed, without changing existing citation behavior;
- `src/assessment/demo/factory.py` — add the isolated candidate path and rule-specific Annex I preload/manifest;
- `tests/assessment/evidence/test_multi_corpus_retriever.py`
- `tests/assessment/evidence/test_metadata_conversion.py`
- `tests/assessment/evidence/test_citation_resolution.py`

Generated but not committed:

- `corpus_builds/EU_AI_ACT/ai_act_product_safety_candidate.json`

Acceptance: exact EU AI Act citations resolve to metadata-v2 Evidence; no cross-instrument leakage; existing legacy IDs and retrieval counts remain unchanged.

### Phase E — Industrial demo and UI regression

Create either separate pure-data fixtures or explicit variants:

- `tests/fixtures/industrial_ai_article_6_1_unresolved_case.json`
- `tests/fixtures/industrial_ai_article_6_1_negative_case.json`
- `tests/fixtures/industrial_ai_article_6_1_positive_case.json`

Create/modify tests:

- `tests/assessment/test_industrial_article_6_1_fixtures.py`
- `tests/ui/test_assessment_app.py`
- `tests/ui/test_components.py`
- `tests/assessment/report/test_builder.py`

The existing `tests/fixtures/industrial_ai_case.json` should remain unchanged unless a later migration explicitly versions it. Acceptance requires a Data Act-only regression and a dual-framework demonstration with separate Findings and Evidence bindings.

## 14. Scope exclusions

This extension does not implement:

- Article 6(2) categories beyond the existing employment rule;
- Article 6(3) exceptions;
- provider, deployer, importer, or distributor obligation mapping;
- a full product conformity-assessment workflow;
- notified-body selection or certificate validation;
- Article 5 prohibited practices;
- legal determinations under the sectoral instruments listed in Annex I;
- new regulatory frameworks outside the existing AI Act, GDPR, and Data Act scope;
- LLM-based product classification, legal coverage selection, or conformity conclusions.

## 15. Completion criteria

The extension is complete only when:

1. Article 6(1)(a)'s two alternative branches and Article 6(1)(b) are represented separately;
2. unknown and downstream facts follow conditional requirement semantics;
3. no product category or routing hint produces a legal Finding by itself;
4. every positive Finding includes an exact Annex I catalogue entry and third-party confirmation;
5. atomic metadata-v2 EU AI Act Evidence binds without cross-instrument leakage;
6. Data Act, employment AI Act, and GDPR results remain independently reproducible;
7. English and Simplified Chinese inputs produce identical canonical facts and legal outcomes;
8. the report uses preliminary language and does not imply certification or conformity approval.
