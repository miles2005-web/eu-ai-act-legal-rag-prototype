"""Centralized UI localization for the compliance workspace."""

from __future__ import annotations

from typing import Any


DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ("en", "zh-CN")
LANGUAGE_LABELS = {"en": "EN", "zh-CN": "中文"}


TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "language.selector": "Language",
        "app.page_title": "EU Digital Regulation Assessment Platform",
        "product.eyebrow": "Legal engineering",
        "product.name": "EU Digital Regulation",
        "product.meta": "Assessment Platform · Prototype",
        "navigation.workspace": "Workspace",
        "navigation.assessment": "Assessment",
        "navigation.demo_cases": "Demo cases",
        "navigation.findings": "Findings",
        "navigation.evidence_trace": "Evidence trace",
        "navigation.reference": "Reference",
        "navigation.frameworks": "Regulatory frameworks",
        "navigation.technical_details": "Technical details",
        "navigation.new_assessment": "Start a new assessment",
        "reference.frameworks.copy": (
            "Deterministic screening with versioned legal authority."
        ),
        "reference.case_required": (
            "Create or open a case to inspect technical details."
        ),
        "progress.title": "Assessment progress",
        "progress.case": "Case created",
        "progress.facts": "Required facts provided",
        "progress.assessment": "Assessment completed",
        "progress.assessment.incomplete": (
            "Run completed without a substantive finding"
        ),
        "progress.complete": "Complete",
        "progress.pending": "Pending",
        "landing.eyebrow": "EU compliance intelligence",
        "landing.title": "EU Digital Regulation<br>Assessment Platform",
        "landing.subtitle": (
            "Turn system facts into preliminary, evidence-grounded assessments "
            "across Europe's core digital regulatory frameworks."
        ),
        "landing.disclaimer": (
            "Prototype only — this output does not constitute legal advice."
        ),
        "capability.facts.title": "Establish the facts",
        "capability.facts.copy": (
            "Capture legally relevant system, use-case, and data facts without "
            "inference."
        ),
        "capability.rules.title": "Run versioned rules",
        "capability.rules.copy": (
            "Apply deterministic regulatory screening with explicit rule metadata."
        ),
        "capability.evidence.title": "Ground every finding",
        "capability.evidence.copy": (
            "Resolve authoritative provisions from instrument-aware legal corpora."
        ),
        "capability.trace.title": "Review the trace",
        "capability.trace.copy": (
            "Connect facts, findings, legal basis, and stable evidence identities."
        ),
        "landing.current_case": "Current case",
        "landing.continue": "Continue assessment",
        "landing.open_workspace": "Open workspace",
        "landing.no_description": "No case description provided.",
        "demo.section.eyebrow": "Guided assessment",
        "demo.section.title": "Choose a demonstration scenario",
        "demo.section.copy": (
            "Start with a prepared factual record to explore the compliance "
            "workspace and report experience."
        ),
        "demo.recruitment.label": "Employment · EU AI Act",
        "demo.recruitment.title": "Recruitment AI Screening",
        "demo.recruitment.copy": (
            "Assess an AI system that screens CVs, ranks candidates, and "
            "materially influences access to employment."
        ),
        "demo.recruitment.meta": (
            "Prepared facts · High-risk classification screening"
        ),
        "demo.recruitment.open": "Open recruitment demo",
        "demo.industrial.label": "Industrial data · EU Data Act",
        "demo.industrial.title": "Industrial AI Monitoring",
        "demo.industrial.copy": (
            "Assess connected machinery monitoring that generates operational "
            "data requested by an external maintenance provider."
        ),
        "demo.industrial.meta": (
            "Prepared facts · Connected-product relevance screening"
        ),
        "demo.industrial.open": "Open industrial demo",
        "demo.industrial_multi_framework.label": (
            "Industrial safety and data · EU AI Act + EU Data Act"
        ),
        "demo.industrial_multi_framework.title": (
            "Industrial Robot Safety and Data Access"
        ),
        "demo.industrial_multi_framework.copy": (
            "Assess embedded AI that performs an industrial-robot safety function "
            "while connected product and related-service data is requested."
        ),
        "demo.industrial_multi_framework.meta": (
            "Prepared facts · Two independent regulatory screens"
        ),
        "demo.industrial_multi_framework.open": "Open multi-framework demo",
        "scenario.recruitment-ai-ranking-candidates.case_name": (
            "Recruitment AI candidate screening and ranking"
        ),
        "scenario.recruitment-ai-ranking-candidates.description": (
            "A fictional EU-based company uses a third-party AI system to screen "
            "job applications, score candidates against role criteria, and rank "
            "candidates for recruiter review. Recruiters use the ranking to decide "
            "which candidates progress to interview."
        ),
        "scenario.recruitment-ai-ranking-candidates.purpose": (
            "Support recruitment teams by screening and ranking job candidates "
            "for open positions."
        ),
        "scenario.recruitment-ai-ranking-candidates.task": (
            "Recruitment screening and candidate ranking for interview selection"
        ),
        "scenario.industrial-ai-connected-machinery-data-access.case_name": (
            "Industrial AI connected machinery monitoring"
        ),
        "scenario.industrial-ai-connected-machinery-data-access.description": (
            "A European manufacturer operates an AI-enabled industrial monitoring "
            "system connected to production machinery. The system collects and "
            "analyses operational data from connected equipment, and an external "
            "maintenance provider requests access to relevant machine data."
        ),
        "scenario.industrial-ai-connected-machinery-data-access.purpose": (
            "Monitor connected production equipment and support preventive "
            "maintenance using operational machine data."
        ),
        "scenario.industrial-ai-connected-machinery-data-access.task": (
            "Connected machinery monitoring through a related service using "
            "product operational data"
        ),
        "scenario.industrial-robot-safety-data-access.case_name": (
            "Industrial Robot Safety and Data Access"
        ),
        "scenario.industrial-robot-safety-data-access.description": (
            "A European manufacturer operates an industrial robot with embedded "
            "AI that automatically slows, stops, or triggers emergency braking. "
            "The connected robot and related service generate operational data "
            "requested by an external maintenance provider."
        ),
        "scenario.industrial-robot-safety-data-access.purpose": (
            "Perform a protective robot safety function and support connected "
            "monitoring and maintenance data access."
        ),
        "scenario.industrial-robot-safety-data-access.task": (
            "Industrial robot protective control and emergency braking with "
            "connected product and related-service data access"
        ),
        "case.new.eyebrow": "New case",
        "case.new.title": "Create a blank assessment",
        "case.new.copy": "Start with an empty factual record for your own AI system.",
        "case.name": "Case name",
        "case.name.placeholder": "Example: Recruitment screening system",
        "case.description": "Description",
        "case.description.placeholder": (
            "Briefly describe the AI system and assessment purpose."
        ),
        "case.create": "Create case",
        "case.name.required": "Enter a case name before creating the case.",
        "case.active": "Active assessment case",
        "case.system_summary_aria": "System summary",
        "case.system": "System",
        "case.system.unnamed": "Unnamed AI system",
        "case.purpose.missing": "Purpose not yet provided.",
        "case.use_context": "Use context",
        "case.task.missing": "Task not yet provided.",
        "case.assessment": "Assessment",
        "case.report_available": "Report available",
        "case.facts_in_progress": "Facts in progress",
        "case.custom": "Custom case",
        "case.demo": "{name} demo",
        "technical.case_id": "Case ID",
        "technical.case_schema": "Case schema",
        "technical.facts_schema": "Facts schema",
        "technical.report": "Report {version} · Engine {engine}",
        "facts.eyebrow": "Step 2",
        "facts.title": "Provide assessment facts",
        "facts.copy": (
            "Unknown answers remain visible as missing information and do not "
            "create a legal finding."
        ),
        "facts.demo_loaded": (
            "{name} demo loaded. Review or edit the populated facts before "
            "running the assessment."
        ),
        "facts.domain.question": "In which context is the AI system used?",
        "facts.task.question": "What task does the AI system perform?",
        "facts.task.placeholder": (
            "Example: Screens CVs and ranks candidates for recruiter review."
        ),
        "facts.influence.question": (
            "Does the output materially influence a decision or operational outcome?"
        ),
        "facts.save": "Save facts",
        "facts.saved": "Facts saved.",
        "normalization.recognized": (
            "Recognized controlled mappings: {mappings}"
        ),
        "normalization.ambiguous": (
            "No controlled scenario was automatically identified from this "
            "description. The text has been saved as case context; assessment "
            "modules can still be selected through structured inputs or explicit "
            "confirmation."
        ),
        "normalization.confirm_original": (
            "Confirm the original text as the task description without legal "
            "classification"
        ),
        "normalization.saved_unknown": (
            "No controlled scenario was automatically identified from this "
            "description. The text has been saved as case context; assessment "
            "modules can still be selected through structured inputs or explicit "
            "confirmation."
        ),
        "assessment.eyebrow": "Step 3",
        "assessment.title": "Run assessment",
        "assessment.copy": (
            "Apply the configured rules, resolve legal evidence, and build a "
            "traceable preliminary report."
        ),
        "assessment.run": "Run assessment",
        "assessment.run_with_gaps": "Run with information gaps",
        "assessment.running": "Running assessment and resolving legal evidence...",
        "assessment.status.generated": "Report generated",
        "assessment.status.with_gaps": "Assessment run with information gaps",
        "assessment.required": (
            "Run the assessment before opening results or evidence trace."
        ),
        "assessment.return": "Return to assessment workspace",
        "app.initialization_error": (
            "The assessment workflow could not be initialized: {error}"
        ),
        "report.eyebrow": "Preliminary result",
        "report.title": "Assessment report",
        "report.copy": (
            "Review framework findings, reasoning, and supporting legal authority."
        ),
        "report.findings.one": "{count} finding",
        "report.findings.other": "{count} findings",
        "report.evidence.one": "{count} evidence record",
        "report.evidence.other": "{count} evidence records",
        "report.gaps.zero": "No information gaps",
        "report.gaps.one": "{count} information gap",
        "report.gaps.other": "{count} information gaps",
        "report.findings.title": "Findings",
        "report.no_finding": (
            "No substantive assessment was produced. Additional facts or an "
            "implemented assessment module are required."
        ),
        "incomplete.eyebrow": "Preliminary assessment status",
        "incomplete.title": "Assessment incomplete",
        "incomplete.summary": (
            "No substantive legal finding was produced because the confirmed "
            "assessment module still contains unresolved facts."
        ),
        "incomplete.unresolved.title": "Unresolved required facts",
        "incomplete.blocked.title": (
            "Downstream questions blocked by the unresolved fact"
        ),
        "incomplete.recorded_answer": "Recorded answer: Unknown",
        "incomplete.awaiting_answer": "Answer not yet provided",
        "incomplete.blocked.copy": (
            "These questions were not counted as unanswered omissions because "
            "their prerequisite facts remain unresolved."
        ),
        "incomplete.evidence": (
            "Evidence binding has not been reached because no substantive "
            "Finding exists yet. Authored legal references define the module "
            "scope only; they do not support a nonexistent Finding."
        ),
        "incomplete.edit": "Edit unresolved answer",
        "incomplete.return": "Return to Article 6(1) questions",
        "incomplete.later": "Continue assessment later",
        "report.screened.one": "Other screened issue ({count})",
        "report.screened.other": "Other screened issues ({count})",
        "report.screened.copy": (
            "These rules were executed and retained for audit, but their "
            "screening conditions were not met."
        ),
        "report.no_framework": "No framework recorded",
        "finding.evidence_count.one": "{count} supporting record",
        "finding.evidence_count.other": "{count} supporting records",
        "finding.kicker": "Preliminary conclusion",
        "finding.review_warning": (
            "Further legal review is required before treating this preliminary "
            "result as a final classification."
        ),
        "finding.AI_ACT_HIGH_RISK_EMPLOYMENT.potentially_applies.title": (
            "Employment-related high-risk classification potentially applies"
        ),
        "finding.AI_ACT_HIGH_RISK_EMPLOYMENT.potentially_applies.summary": (
            "The available facts match the preliminary employment-related "
            "screening conditions under Article 6 and Annex III point 4(a). "
            "This is not a definitive high-risk classification and requires "
            "further legal assessment."
        ),
        "finding.AI_ACT_HIGH_RISK_EMPLOYMENT.does_not_apply.title": (
            "Employment-related high-risk screening criteria not met"
        ),
        "finding.AI_ACT_HIGH_RISK_EMPLOYMENT.does_not_apply.summary": (
            "The available facts do not match the employment-related conditions "
            "screened by this preliminary rule. This does not determine whether "
            "another EU AI Act high-risk category applies."
        ),
        "finding.GDPR_ARTICLE22_RELEVANCE.potentially_applies.title": (
            "GDPR Article 22 assessment may be relevant"
        ),
        "finding.GDPR_ARTICLE22_RELEVANCE.potentially_applies.summary": (
            "The available facts indicate personal-data processing, an automated "
            "individual decision, and material influence on an individual. A fuller "
            "GDPR Article 22 assessment is warranted. This trigger does not determine "
            "that Article 22 applies or that the processing is non-compliant."
        ),
        "finding.GDPR_ARTICLE22_RELEVANCE.does_not_apply.title": (
            "GDPR Article 22 relevance trigger not met"
        ),
        "finding.GDPR_ARTICLE22_RELEVANCE.does_not_apply.summary": (
            "The known facts do not satisfy all conditions for this preliminary "
            "Article 22 relevance trigger. This is not a general GDPR compliance "
            "conclusion."
        ),
        "finding.EU_DATA_ACT_RELEVANCE.potentially_applies.title": (
            "Data Act relevance potentially applies"
        ),
        "finding.EU_DATA_ACT_RELEVANCE.potentially_applies.summary": (
            "The known facts indicate a connected product or related service that "
            "generates data. Further EU Data Act assessment is warranted. This "
            "preliminary trigger does not determine final scope, obligations, "
            "exemptions, or compliance."
        ),
        "finding.EU_DATA_ACT_RELEVANCE.does_not_apply.title": (
            "Data Act relevance trigger not met"
        ),
        "finding.EU_DATA_ACT_RELEVANCE.does_not_apply.summary": (
            "The complete facts do not satisfy this preliminary Data Act relevance "
            "trigger. This result does not determine other Data Act contexts or "
            "general regulatory compliance."
        ),
        "trace.description.use_context.domain": (
            "Checks whether the system is used in the relevant employment context."
        ),
        "trace.description.use_context.task": (
            "Checks whether the AI performs a covered recruitment or worker-related function."
        ),
        "trace.description.use_context.materially_influences_decision": (
            "Checks whether the AI output materially influences the relevant decision."
        ),
        "trace.description.data_protection.personal_data_processed": (
            "Checks whether personal data are processed."
        ),
        "trace.description.data_protection.automated_individual_decision": (
            "Checks whether an automated individual decision is involved."
        ),
        "trace.description.data_act.connected_product": (
            "Checks whether a connected product is involved."
        ),
        "trace.description.data_act.related_service": (
            "Checks whether a related service is involved."
        ),
        "trace.description.data_act.data_generated": (
            "Checks whether the product or service generates data."
        ),
        "decision.title": "Decision path",
        "decision.no_trace": "No reasoning stages were recorded for this finding.",
        "decision.condition": "Condition {number}",
        "state.matched": "Matched",
        "state.not_matched": "Not matched",
        "state.unknown": "Unknown",
        "legal.title": "Legal authority",
        "legal.none": "No legal basis recorded for this finding.",
        "evidence.summary.title": "Evidence summary",
        "evidence.none": "No supporting evidence is currently bound to this finding.",
        "evidence.pending_binding": (
            "Legal authorities have been identified for this finding, but atomic "
            "official source Evidence has not yet been bound for this rule."
        ),
        "evidence.supporting.one": "{count} supporting record",
        "evidence.supporting.other": "{count} supporting records",
        "evidence.citations.one": "{count} citation represented",
        "evidence.citations.other": "{count} citations represented",
        "evidence.excerpts.one": "{count} supporting excerpt",
        "evidence.excerpts.other": "{count} supporting excerpts",
        "evidence.view_trace": "View full evidence trace",
        "technical.rule_id": "Rule ID",
        "technical.rule_version": "Rule version",
        "technical.issue_code": "Issue code",
        "technical.raw_facts": "Raw fact keys",
        "technical.raw_reasons": "Raw reason codes",
        "report.missing.title": "Missing information",
        "report.missing.none": (
            "No additional information is required for the primary assessment."
        ),
        "missing_reason.not_provided": "Not provided",
        "missing_reason.unknown": "Unknown",
        "missing_reason.path_not_found": "Fact path not found",
        "report.recommendations": "Recommendations",
        "report.multiple_findings.summary": (
            "Two independent regulatory screens produced substantive findings."
        ),
        "report.framework_actions.title": "Framework-specific next steps",
        "recommendation.requirement": (
            "Confirm {fact} before running the {framework} assessment."
        ),
        "recommendation.review.AI_ACT_HIGH_RISK_EMPLOYMENT": (
            "Obtain legal review before relying on this preliminary employment "
            "high-risk classification."
        ),
        "recommendation.review.GDPR_ARTICLE22_RELEVANCE": (
            "Obtain legal review before relying on this preliminary GDPR "
            "Article 22 relevance assessment."
        ),
        "recommendation.review.AI_ACT_HIGH_RISK_PRODUCT_SAFETY": (
            "Verify the applicable Annex I product legislation, retain the "
            "safety-component classification record, confirm the relevant "
            "conformity-assessment route, and conduct further legal and product-"
            "compliance review."
        ),
        "recommendation.review.EU_DATA_ACT_RELEVANCE": (
            "Identify the relevant user and data-holder relationships, map "
            "generated product and related-service data, review access, sharing "
            "and contractual arrangements, and conduct further Data Act assessment."
        ),
        "recommendation.supporting_authority_readable": (
            "Obtain supporting legal authority before relying on this preliminary "
            "finding."
        ),
        "recommendation.execution_failure_readable": (
            "The {rule} could not be completed. Review its technical details "
            "before finalizing the assessment."
        ),
        "recommendation.missing_fact": (
            "Provide or confirm the missing fact '{fact}' before rerunning the "
            "affected assessment rules."
        ),
        "recommendation.legal_review": (
            "Obtain legal review for finding '{issue}' before relying on the "
            "preliminary result."
        ),
        "recommendation.supporting_authority": (
            "Obtain supporting authority for finding '{issue}'."
        ),
        "recommendation.execution_failure": (
            "Review the execution failure for rule '{rule_id}' before finalizing "
            "the assessment."
        ),
        "report.technical": "Report technical details",
        "technical.report_id": "Report ID",
        "technical.report_meta": (
            "Report version {version} · Engine {engine} · Generated {generated}"
        ),
        "technical.raw_missing": "Raw missing fact keys",
        "technical.raw_recommendations": "Raw report recommendations",
        "framework_screens.title": "Other framework screens",
        "framework_screens.copy": (
            "These screens are retained for audit but do not define the primary "
            "assessment conclusion."
        ),
        "framework_screen.completed": "Completed",
        "framework_screen.potentially_relevant": "Potentially relevant",
        "framework_screen.additional_facts": (
            "Not assessed — additional facts required"
        ),
        "framework_screen.screened_out": "Screened out",
        "framework_screen.failure": "Not assessed — execution review required",
        "framework_screen.missing": "Information needed for this framework",
        "evidence.trace.eyebrow": "Legal authority",
        "evidence.trace.title": "Evidence trace",
        "evidence.trace.copy": (
            "Inspect the relationship between findings, versioned rules, legal "
            "basis references, and atomic corpus evidence."
        ),
        "evidence.trace.no_finding": "No finding is available for evidence tracing.",
        "evidence.trace.select": "Select finding",
        "evidence.trace.selected": "Selected finding",
        "trace.facts.title": "Facts",
        "trace.facts.copy": "The factual inputs used by the assessment rule.",
        "trace.rule.title": "Rule evaluation",
        "trace.rule.copy": (
            "The versioned assessment logic and recorded reasoning sequence."
        ),
        "trace.rule.none": "No reasoning trace was recorded.",
        "trace.rule.technical": "Rule technical details",
        "trace.rule.id_version": "Rule ID and version",
        "trace.legal.title": "Legal basis",
        "trace.legal.copy": "The authored legal references supporting the finding.",
        "trace.legal.none": "No legal basis was recorded.",
        "trace.evidence.title": "Source evidence",
        "trace.evidence.copy": (
            "Atomic, versioned legal excerpts bound to this finding."
        ),
        "trace.evidence.none": "No supporting evidence is bound to this finding.",
        "trace.evidence.pending_binding": (
            "Legal authorities have been identified for this finding, but atomic "
            "official source Evidence has not yet been bound for this rule."
        ),
        "evidence.official_item": "Official excerpt {number}",
        "evidence.authority": "Authority level",
        "evidence.version": "Document version",
        "evidence.source": "Legal source",
        "evidence.citation": "Citation",
        "evidence.original": "Official source text",
        "evidence.stable_id": "Stable Evidence ID",
        "evidence.not_recorded": "Not recorded",
        "evidence.technical": "Evidence technical details",
        "evidence.all_ids": "All bound stable Evidence IDs",
        "evidence.excerpt_hash": "Authoritative excerpt SHA-256",
        "evidence.raw_source": "Raw legal source value",
        "evidence.source_version": "Official source version",
        "authority.binding_legislation": "Binding legislation",
        "authority.case_law": "Case law",
        "authority.official_guidance": "Official guidance",
        "authority.non_binding_official_material": "Non-binding official material",
        "authority.secondary_source": "Secondary source",
        "authority.unknown": "Unknown authority",
        "fact.use_context.domain": "Employment context",
        "fact.use_context.task": "AI system function",
        "fact.use_context.materially_influences_decision": (
            "Material influence on decisions"
        ),
        "fact.data_protection.personal_data_processed": "Personal data processing",
        "fact.data_protection.automated_individual_decision": (
            "Automated individual decision-making"
        ),
        "fact.data_protection.special_category_data_processed": (
            "Special-category data processing"
        ),
        "fact.data_act.connected_product": "Connected product",
        "fact.data_act.related_service": "Related service",
        "fact.data_act.data_generated": (
            "Product or related-service data generation"
        ),
        "fact.data_act.data_holder_identified": "Data holder identified",
        "fact.data_act.user_or_third_party_access_request": (
            "User or third-party data access request"
        ),
        "rule.AI_ACT_HIGH_RISK_EMPLOYMENT": "Employment high-risk screening",
        "rule.GDPR_ARTICLE22_RELEVANCE": (
            "GDPR automated-decision relevance screening"
        ),
        "rule.EU_DATA_ACT_RELEVANCE": "EU Data Act relevance screening",
        "status.applies": "Applies",
        "status.does_not_apply": "Does Not Apply",
        "status.potentially_applies": "Potentially Applies",
        "status.undetermined": "Undetermined",
        "status.not_assessed": "Not Assessed",
        "framework.EU_AI_ACT": "EU AI Act",
        "framework.GDPR": "GDPR",
        "framework.EU_DATA_ACT": "EU Data Act",
        "framework.UNKNOWN": "Other framework",
        "value.yes": "Yes",
        "value.no": "No",
        "value.unknown": "Unknown",
        "value.not_answered": "Not answered",
        "value.not_recorded": "Not recorded",
        "value.none_recorded": "None recorded",
        "domain.unknown": "Unknown / not yet provided",
        "domain.employment": "Employment",
        "domain.biometrics": "Biometrics",
        "domain.critical_infrastructure": "Critical infrastructure",
        "domain.education": "Education",
        "domain.essential_services": "Essential services",
        "domain.law_enforcement": "Law enforcement",
        "domain.migration_asylum_border_control": (
            "Migration, asylum, or border control"
        ),
        "domain.justice_democratic_processes": (
            "Justice or democratic processes"
        ),
        "domain.product_safety": "Product safety",
        "domain.other": "Other",
        "test.english_only": "English fallback",
        "component.overview": "Assessment overview",
        "component.frameworks_assessed": "Frameworks assessed",
        "component.findings": "Findings",
        "component.evidence_records": "Evidence records",
        "component.missing_information": "Missing information",
        "component.framework_summary": (
            "{findings} finding(s) · {reviews} requiring legal review"
        ),
        "component.no_findings": "No findings recorded for this framework.",
        "component.issue_code": "Issue code",
        "component.facts_referenced": "Facts referenced",
        "component.bound_evidence": "Bound evidence",
        "component.legal_basis": "Legal basis",
        "component.no_legal_basis": "No legal basis recorded.",
        "component.review_required": (
            "Further legal review is required. This is a preliminary assessment, "
            "not a definitive legal conclusion."
        ),
        "component.reasoning_trace": "Reasoning trace",
        "component.no_trace": "No reasoning trace recorded.",
        "component.result": "Result",
        "component.facts": "Facts",
        "component.requested_basis": "Requested legal basis",
        "component.resolved_citation": "Resolved citation",
        "component.rule": "Rule",
        "component.version": "Version",
    },
    "zh-CN": {
        "language.selector": "语言",
        "app.page_title": "欧盟数字监管评估平台",
        "product.eyebrow": "法律工程",
        "product.name": "欧盟数字监管",
        "product.meta": "评估平台 · 原型",
        "navigation.workspace": "工作区",
        "navigation.assessment": "评估",
        "navigation.demo_cases": "演示案例",
        "navigation.findings": "评估结论",
        "navigation.evidence_trace": "证据链",
        "navigation.reference": "参考",
        "navigation.frameworks": "监管框架",
        "navigation.technical_details": "技术详情",
        "navigation.new_assessment": "开始新评估",
        "reference.frameworks.copy": "基于确定性规则与版本化法律依据进行筛查。",
        "reference.case_required": "创建或打开案例后可查看技术详情。",
        "progress.title": "评估进度",
        "progress.case": "已创建案例",
        "progress.facts": "已提供必要事实",
        "progress.assessment": "已完成评估",
        "progress.assessment.incomplete": "本次运行已完成，但未形成实质性法律结论",
        "progress.complete": "完成",
        "progress.pending": "待完成",
        "landing.eyebrow": "欧盟合规智能",
        "landing.title": "欧盟数字监管<br>评估平台",
        "landing.subtitle": "将系统事实转化为有法律证据支持的初步监管评估。",
        "landing.disclaimer": "本项目仅为原型，输出不构成法律意见。",
        "capability.facts.title": "建立事实基础",
        "capability.facts.copy": "记录与系统、使用场景及数据相关的法律事实，不作推测。",
        "capability.rules.title": "运行版本化规则",
        "capability.rules.copy": "采用确定性监管筛查，并保留明确的规则元数据。",
        "capability.evidence.title": "为结论绑定依据",
        "capability.evidence.copy": "从按法律文件识别的语料中解析权威条款。",
        "capability.trace.title": "审阅完整链路",
        "capability.trace.copy": "连接事实、结论、法律依据与稳定证据标识。",
        "landing.current_case": "当前案例",
        "landing.continue": "继续评估",
        "landing.open_workspace": "打开工作区",
        "landing.no_description": "未提供案例说明。",
        "demo.section.eyebrow": "引导式评估",
        "demo.section.title": "选择演示场景",
        "demo.section.copy": "使用预设事实记录体验合规工作区和评估报告。",
        "demo.recruitment.label": "就业 · 《欧盟人工智能法案》",
        "demo.recruitment.title": "招聘 AI 候选人筛选与排序",
        "demo.recruitment.copy": "评估用于筛选简历、排列候选人并实质影响就业机会的 AI 系统。",
        "demo.recruitment.meta": "预设事实 · 高风险分类初步筛查",
        "demo.recruitment.open": "打开招聘 AI 演示",
        "demo.industrial.label": "工业数据 · 《欧盟数据法案》",
        "demo.industrial.title": "工业 AI 互联机械监测",
        "demo.industrial.copy": "评估产生运行数据且外部维护方请求数据访问的联网机械监测系统。",
        "demo.industrial.meta": "预设事实 · 联网产品相关性筛查",
        "demo.industrial.open": "打开工业 AI 演示",
        "demo.industrial_multi_framework.label": (
            "工业安全与数据 · 《欧盟人工智能法案》+《欧盟数据法案》"
        ),
        "demo.industrial_multi_framework.title": "工业机器人安全与数据访问联合评估",
        "demo.industrial_multi_framework.copy": (
            "评估承担工业机器人安全功能的嵌入式 AI，以及被请求访问的互联产品和相关服务数据。"
        ),
        "demo.industrial_multi_framework.meta": "预设事实 · 两项相互独立的监管筛查",
        "demo.industrial_multi_framework.open": "打开多框架联合评估",
        "scenario.recruitment-ai-ranking-candidates.case_name": (
            "招聘 AI 候选人筛选与排序"
        ),
        "scenario.recruitment-ai-ranking-candidates.description": (
            "一家虚构的欧盟公司使用第三方 AI 系统筛选求职申请，按职位标准为候选人"
            "评分并排序，供招聘人员审核。招聘人员使用该排序决定哪些候选人进入面试。"
        ),
        "scenario.recruitment-ai-ranking-candidates.purpose": (
            "通过筛选并排序开放职位的候选人，为招聘团队提供支持。"
        ),
        "scenario.recruitment-ai-ranking-candidates.task": (
            "招聘筛选与候选人排序，用于面试遴选"
        ),
        "scenario.industrial-ai-connected-machinery-data-access.case_name": (
            "工业 AI 互联机械监测"
        ),
        "scenario.industrial-ai-connected-machinery-data-access.description": (
            "一家欧洲制造商运行一套连接生产机械的 AI 工业监测系统。该系统收集并分析"
            "互联设备产生的运行数据，外部维护服务商请求访问相关设备数据。"
        ),
        "scenario.industrial-ai-connected-machinery-data-access.purpose": (
            "利用设备运行数据监测互联生产设备并支持预防性维护。"
        ),
        "scenario.industrial-ai-connected-machinery-data-access.task": (
            "通过相关服务监测联网机械并处理产品运行数据"
        ),
        "scenario.industrial-robot-safety-data-access.case_name": (
            "工业机器人安全与数据访问联合评估"
        ),
        "scenario.industrial-robot-safety-data-access.description": (
            "一家欧洲制造商运行一台配有嵌入式 AI 的工业机器人。该 AI 会自动减速、"
            "停止或触发紧急制动；联网机器人及相关服务产生运行数据，外部维护服务商"
            "请求访问有关数据。"
        ),
        "scenario.industrial-robot-safety-data-access.purpose": (
            "执行机器人保护性安全功能，并支持联网监测及维护数据访问。"
        ),
        "scenario.industrial-robot-safety-data-access.task": (
            "工业机器人保护性控制与紧急制动，以及互联产品和相关服务数据访问"
        ),
        "case.new.eyebrow": "新案例",
        "case.new.title": "创建空白评估",
        "case.new.copy": "为您的 AI 系统创建一份空白事实记录。",
        "case.name": "案例名称",
        "case.name.placeholder": "例如：招聘筛选系统",
        "case.description": "案例说明",
        "case.description.placeholder": "简要说明 AI 系统及本次评估目的。",
        "case.create": "创建案例",
        "case.name.required": "请先填写案例名称。",
        "case.active": "当前评估案例",
        "case.system_summary_aria": "系统摘要",
        "case.system": "系统",
        "case.system.unnamed": "未命名的 AI 系统",
        "case.purpose.missing": "尚未提供用途。",
        "case.use_context": "使用场景",
        "case.task.missing": "尚未提供任务说明。",
        "case.assessment": "评估状态",
        "case.report_available": "报告已生成",
        "case.facts_in_progress": "事实收集中",
        "case.custom": "自定义案例",
        "case.demo": "{name}演示",
        "technical.case_id": "案例 ID",
        "technical.case_schema": "案例 Schema",
        "technical.facts_schema": "事实 Schema",
        "technical.report": "报告 {version} · 引擎 {engine}",
        "facts.eyebrow": "第 2 步",
        "facts.title": "提供评估事实",
        "facts.copy": "未知答案将显示为信息缺口，不会生成法律结论。",
        "facts.demo_loaded": "已载入{name}演示。运行评估前可检查或修改预设事实。",
        "facts.domain.question": "AI 系统用于什么场景？",
        "facts.task.question": "AI 系统执行什么任务？",
        "facts.task.placeholder": "例如：筛选简历并排列候选人，供招聘人员审阅。",
        "facts.influence.question": "系统输出是否会实质影响个人决定或运营结果？",
        "facts.save": "保存事实",
        "facts.saved": "事实已保存。",
        "normalization.recognized": "已识别受控映射：{mappings}",
        "normalization.ambiguous": (
            "未从该描述中自动识别受控场景。描述已作为案例信息保存；你仍可通过"
            "结构化问题或手动确认选择评估模块。"
        ),
        "normalization.confirm_original": (
            "确认仅将原文作为任务描述，不进行法律分类"
        ),
        "normalization.saved_unknown": (
            "未从该描述中自动识别受控场景。描述已作为案例信息保存；你仍可通过"
            "结构化问题或手动确认选择评估模块。"
        ),
        "assessment.eyebrow": "第 3 步",
        "assessment.title": "运行评估",
        "assessment.copy": "应用已配置规则、解析法律证据并生成可追溯的初步报告。",
        "assessment.run": "运行评估",
        "assessment.run_with_gaps": "带信息缺口运行",
        "assessment.running": "正在运行评估并解析法律证据……",
        "assessment.status.generated": "报告已生成",
        "assessment.status.with_gaps": "评估已运行，存在信息缺口",
        "assessment.required": "请先运行评估，再打开评估结论或证据链。",
        "assessment.return": "返回评估工作区",
        "app.initialization_error": "无法初始化评估工作流：{error}",
        "report.eyebrow": "初步结果",
        "report.title": "评估报告",
        "report.copy": "审阅监管结论、推理路径及支持性法律依据。",
        "report.findings.one": "{count} 项结论",
        "report.findings.other": "{count} 项结论",
        "report.evidence.one": "{count} 条证据记录",
        "report.evidence.other": "{count} 条证据记录",
        "report.gaps.zero": "无信息缺口",
        "report.gaps.one": "{count} 项信息缺口",
        "report.gaps.other": "{count} 项信息缺口",
        "report.findings.title": "评估结论",
        "report.no_finding": (
            "尚未形成实质性评估结论。需要补充事实，或当前场景尚需相应的评估模块支持。"
        ),
        "incomplete.eyebrow": "初步评估状态",
        "incomplete.title": "评估尚未完成",
        "incomplete.summary": (
            "由于已确认的评估模块仍存在未决事实，当前未形成实质性法律结论。"
        ),
        "incomplete.unresolved.title": "未决的必要事实",
        "incomplete.blocked.title": "因未决事实而暂时阻断的后续问题",
        "incomplete.recorded_answer": "已记录回答：未知",
        "incomplete.awaiting_answer": "尚未提供回答",
        "incomplete.blocked.copy": (
            "这些问题的前置事实仍未解决，因此不会被计为用户遗漏回答。"
        ),
        "incomplete.evidence": (
            "由于尚未形成实质性 Finding，当前尚未进入证据绑定阶段。"
            "已编写的法律依据仅用于说明模块范围，并非对不存在 Finding 的支持。"
        ),
        "incomplete.edit": "编辑未决回答",
        "incomplete.return": "返回 Article 6(1) 补充问题",
        "incomplete.later": "稍后继续评估",
        "report.screened.one": "其他已筛查问题（{count}）",
        "report.screened.other": "其他已筛查问题（{count}）",
        "report.screened.copy": "这些规则已执行并保留用于审计，但其筛查条件未满足。",
        "report.no_framework": "未记录监管框架",
        "finding.evidence_count.one": "{count} 条支持记录",
        "finding.evidence_count.other": "{count} 条支持记录",
        "finding.kicker": "初步结论",
        "finding.review_warning": "将该初步结果作为最终分类前，需要进一步法律审查。",
        "finding.AI_ACT_HIGH_RISK_EMPLOYMENT.potentially_applies.title": (
            "可能属于就业相关高风险 AI 系统"
        ),
        "finding.AI_ACT_HIGH_RISK_EMPLOYMENT.potentially_applies.summary": (
            "现有事实符合《欧盟人工智能法案》第 6 条及附件三第 4(a) 点的就业相关初步筛查条件。"
            "该结果并非最终高风险分类，仍需进一步法律评估。"
        ),
        "finding.AI_ACT_HIGH_RISK_EMPLOYMENT.does_not_apply.title": (
            "未满足就业相关高风险初步筛查条件"
        ),
        "finding.AI_ACT_HIGH_RISK_EMPLOYMENT.does_not_apply.summary": (
            "现有事实不符合本初步规则筛查的就业相关条件。"
            "该结果不能排除《欧盟人工智能法案》下的其他高风险类别。"
        ),
        "finding.GDPR_ARTICLE22_RELEVANCE.potentially_applies.title": (
            "可能需要评估 GDPR 第 22 条"
        ),
        "finding.GDPR_ARTICLE22_RELEVANCE.potentially_applies.summary": (
            "现有事实表明涉及个人数据处理、自动化个人决策，且对个人产生实质性影响。"
            "因此需要进一步评估 GDPR 第 22 条。本触发规则不代表第 22 条已确定适用，"
            "也不代表相关处理不合规。"
        ),
        "finding.GDPR_ARTICLE22_RELEVANCE.does_not_apply.title": (
            "未满足 GDPR 第 22 条相关性触发条件"
        ),
        "finding.GDPR_ARTICLE22_RELEVANCE.does_not_apply.summary": (
            "已知事实未满足本初步第 22 条相关性触发规则的全部条件。"
            "该结果不是对 GDPR 整体合规性的结论。"
        ),
        "finding.EU_DATA_ACT_RELEVANCE.potentially_applies.title": (
            "可能涉及《欧盟数据法案》"
        ),
        "finding.EU_DATA_ACT_RELEVANCE.potentially_applies.summary": (
            "已知事实表明存在生成数据的联网产品或相关服务，因此需要进一步评估"
            "《欧盟数据法案》。本初步触发规则不确定最终适用范围、义务、例外或合规性。"
        ),
        "finding.EU_DATA_ACT_RELEVANCE.does_not_apply.title": (
            "未满足《欧盟数据法案》相关性触发条件"
        ),
        "finding.EU_DATA_ACT_RELEVANCE.does_not_apply.summary": (
            "完整事实未满足本初步《欧盟数据法案》相关性触发规则。"
            "该结果不确定其他《欧盟数据法案》场景或整体合规性。"
        ),
        "trace.description.use_context.domain": "检查系统是否用于相关就业场景。",
        "trace.description.use_context.task": "检查 AI 是否执行受规制的招聘或劳动者相关功能。",
        "trace.description.use_context.materially_influences_decision": (
            "检查 AI 输出是否对相关决策产生实质性影响。"
        ),
        "trace.description.data_protection.personal_data_processed": (
            "检查是否处理个人数据。"
        ),
        "trace.description.data_protection.automated_individual_decision": (
            "检查是否涉及自动化个人决策。"
        ),
        "trace.description.data_act.connected_product": "检查是否涉及联网产品。",
        "trace.description.data_act.related_service": "检查是否涉及相关服务。",
        "trace.description.data_act.data_generated": "检查产品或服务是否生成数据。",
        "decision.title": "判断路径",
        "decision.no_trace": "该结论未记录判断阶段。",
        "decision.condition": "条件 {number}",
        "state.matched": "已满足",
        "state.not_matched": "未满足",
        "state.unknown": "未知",
        "legal.title": "法律依据",
        "legal.none": "该结论未记录法律依据。",
        "evidence.summary.title": "证据概览",
        "evidence.none": "当前没有与该结论绑定的支持证据。",
        "evidence.pending_binding": (
            "本结论已生成法律依据，但当前版本尚未为该规则绑定原子官方原文证据。"
        ),
        "evidence.supporting.one": "{count} 条支持记录",
        "evidence.supporting.other": "{count} 条支持记录",
        "evidence.citations.one": "涉及 {count} 项条款",
        "evidence.citations.other": "涉及 {count} 项条款",
        "evidence.excerpts.one": "{count} 条支持性摘录",
        "evidence.excerpts.other": "{count} 条支持性摘录",
        "evidence.view_trace": "查看完整证据链",
        "technical.rule_id": "规则 ID",
        "technical.rule_version": "规则版本",
        "technical.issue_code": "问题代码",
        "technical.raw_facts": "原始事实字段",
        "technical.raw_reasons": "原始原因代码",
        "report.missing.title": "缺失信息",
        "report.missing.none": "当前主要评估无需补充信息。",
        "missing_reason.not_provided": "未提供",
        "missing_reason.unknown": "未知",
        "missing_reason.path_not_found": "未找到事实字段",
        "report.recommendations": "建议",
        "report.multiple_findings.summary": "两项相互独立的监管筛查形成了实质性结论。",
        "report.framework_actions.title": "按监管框架区分的后续行动",
        "recommendation.requirement": (
            "请确认“{fact}”，然后重新运行{framework}评估。"
        ),
        "recommendation.review.AI_ACT_HIGH_RISK_EMPLOYMENT": (
            "在依赖该就业领域高风险初步分类结论前，请进行进一步法律审查。"
        ),
        "recommendation.review.GDPR_ARTICLE22_RELEVANCE": (
            "在依赖该 GDPR 第 22 条相关性初步评估前，请进行进一步法律审查。"
        ),
        "recommendation.review.AI_ACT_HIGH_RISK_PRODUCT_SAFETY": (
            "请核实适用的附件 I 产品法规，保留支持安全部件分类的记录，核实适用的"
            "合格评定路径，并进一步开展法律及产品合规审查。"
        ),
        "recommendation.review.EU_DATA_ACT_RELEVANCE": (
            "请识别相关用户和数据持有人关系，梳理生成的产品及相关服务数据，审查"
            "访问、共享和合同安排，并进一步开展《欧盟数据法案》评估。"
        ),
        "recommendation.supporting_authority_readable": (
            "在依赖该初步结论前，请补充支持性法律依据。"
        ),
        "recommendation.execution_failure_readable": (
            "{rule}未能完成。最终确定评估前，请审阅其技术详情。"
        ),
        "recommendation.missing_fact": (
            "请提供或确认缺失事实“{fact}”，然后重新运行受影响的评估规则。"
        ),
        "recommendation.legal_review": (
            "依赖初步结果前，请就结论“{issue}”取得法律审查。"
        ),
        "recommendation.supporting_authority": (
            "请为结论“{issue}”补充支持性法律依据。"
        ),
        "recommendation.execution_failure": (
            "最终确定评估前，请审查规则“{rule_id}”的执行失败。"
        ),
        "report.technical": "报告技术详情",
        "technical.report_id": "报告 ID",
        "technical.report_meta": "报告版本 {version} · 引擎 {engine} · 生成时间 {generated}",
        "technical.raw_missing": "原始缺失事实字段",
        "technical.raw_recommendations": "原始报告建议",
        "framework_screens.title": "其他框架筛查",
        "framework_screens.copy": "这些筛查记录保留用于审计，但不构成当前主要评估结论。",
        "framework_screen.completed": "已完成",
        "framework_screen.potentially_relevant": "可能相关",
        "framework_screen.additional_facts": "未评估 — 需要补充事实",
        "framework_screen.screened_out": "已排除",
        "framework_screen.failure": "未评估 — 需要审查执行问题",
        "framework_screen.missing": "该框架所需信息",
        "evidence.trace.eyebrow": "法律依据",
        "evidence.trace.title": "证据链",
        "evidence.trace.copy": "查看结论、版本化规则、法律依据与原子语料证据之间的关系。",
        "evidence.trace.no_finding": "当前没有可供追踪的评估结论。",
        "evidence.trace.select": "选择评估结论",
        "evidence.trace.selected": "已选择的评估结论",
        "trace.facts.title": "事实",
        "trace.facts.copy": "评估规则使用的事实输入。",
        "trace.rule.title": "规则判断",
        "trace.rule.copy": "版本化评估逻辑及记录的判断过程。",
        "trace.rule.none": "未记录判断过程。",
        "trace.rule.technical": "规则技术详情",
        "trace.rule.id_version": "规则 ID 与版本",
        "trace.legal.title": "法律依据",
        "trace.legal.copy": "支持该结论的规则所引用法律条款。",
        "trace.legal.none": "未记录法律依据。",
        "trace.evidence.title": "来源证据",
        "trace.evidence.copy": "与该结论绑定的原子化、版本化法律原文摘录。",
        "trace.evidence.none": "该结论未绑定支持证据。",
        "trace.evidence.pending_binding": (
            "本结论已生成法律依据，但当前版本尚未为该规则绑定原子官方原文证据。"
        ),
        "evidence.official_item": "官方原文摘录 {number}",
        "evidence.authority": "权威层级",
        "evidence.version": "文件版本",
        "evidence.source": "法律来源",
        "evidence.citation": "条款引证",
        "evidence.original": "官方原文（英语）",
        "evidence.stable_id": "稳定证据 ID",
        "evidence.not_recorded": "未记录",
        "evidence.technical": "证据技术详情",
        "evidence.all_ids": "全部绑定的稳定证据 ID",
        "evidence.excerpt_hash": "权威原文 SHA-256",
        "evidence.raw_source": "原始法律来源值",
        "evidence.source_version": "官方来源版本",
        "authority.binding_legislation": "具有约束力的立法",
        "authority.case_law": "判例法",
        "authority.official_guidance": "官方指引",
        "authority.non_binding_official_material": "不具约束力的官方材料",
        "authority.secondary_source": "二手资料",
        "authority.unknown": "未知权威层级",
        "fact.use_context.domain": "就业场景",
        "fact.use_context.task": "AI 系统功能",
        "fact.use_context.materially_influences_decision": "对决策产生实质性影响",
        "fact.data_protection.personal_data_processed": "个人数据处理",
        "fact.data_protection.automated_individual_decision": "自动化个人决策",
        "fact.data_protection.special_category_data_processed": "特殊类别数据处理",
        "fact.data_act.connected_product": "互联产品",
        "fact.data_act.related_service": "相关服务",
        "fact.data_act.data_generated": "产品或相关服务生成数据",
        "fact.data_act.data_holder_identified": "已识别数据持有者",
        "fact.data_act.user_or_third_party_access_request": "用户或第三方数据访问请求",
        "rule.AI_ACT_HIGH_RISK_EMPLOYMENT": "就业高风险初步筛查",
        "rule.GDPR_ARTICLE22_RELEVANCE": "GDPR 自动化决策相关性筛查",
        "rule.EU_DATA_ACT_RELEVANCE": "《欧盟数据法案》相关性筛查",
        "status.applies": "适用",
        "status.does_not_apply": "不适用",
        "status.potentially_applies": "可能适用",
        "status.undetermined": "无法确定",
        "status.not_assessed": "未评估",
        "framework.EU_AI_ACT": "《欧盟人工智能法案》",
        "framework.GDPR": "《通用数据保护条例》",
        "framework.EU_DATA_ACT": "《欧盟数据法案》",
        "framework.UNKNOWN": "其他监管框架",
        "value.yes": "是",
        "value.no": "否",
        "value.unknown": "未知",
        "value.not_answered": "尚未回答",
        "value.not_recorded": "未记录",
        "value.none_recorded": "未记录任何项目",
        "domain.unknown": "未知／尚未提供",
        "domain.employment": "就业",
        "domain.biometrics": "生物识别",
        "domain.critical_infrastructure": "关键基础设施",
        "domain.education": "教育",
        "domain.essential_services": "基本公共及私人服务",
        "domain.law_enforcement": "执法",
        "domain.migration_asylum_border_control": "移民、庇护或边境管理",
        "domain.justice_democratic_processes": "司法或民主程序",
        "domain.product_safety": "产品安全",
        "domain.other": "其他",
        "component.overview": "评估概览",
        "component.frameworks_assessed": "已评估框架",
        "component.findings": "评估结论",
        "component.evidence_records": "证据记录",
        "component.missing_information": "缺失信息",
        "component.framework_summary": "{findings} 项结论 · {reviews} 项需要法律审查",
        "component.no_findings": "该监管框架下未记录评估结论。",
        "component.issue_code": "问题代码",
        "component.facts_referenced": "引用事实",
        "component.bound_evidence": "绑定证据",
        "component.legal_basis": "法律依据",
        "component.no_legal_basis": "未记录法律依据。",
        "component.review_required": "需要进一步法律审查。本结果仅为初步评估，并非最终法律结论。",
        "component.reasoning_trace": "判断过程",
        "component.no_trace": "未记录判断过程。",
        "component.result": "结果",
        "component.facts": "事实",
        "component.requested_basis": "请求的法律依据",
        "component.resolved_citation": "解析后的条款引证",
        "component.rule": "规则",
        "component.version": "版本",
    },
}


_ROUTED_QUESTION_COPY = {
    "system_name": {
        "en": ("What is the AI system called?", "Use a stable internal or product name."),
        "zh-CN": ("该 AI 系统叫什么？", "请填写稳定的内部名称或产品名称。"),
    },
    "system_purpose": {
        "en": ("What is the system's intended purpose?", "Describe the purpose, not a legal conclusion."),
        "zh-CN": ("该系统的预期用途是什么？", "请描述用途，不要填写法律结论。"),
    },
    "use_domain": {
        "en": ("In which context is the system used?", "Select the closest controlled use domain."),
        "zh-CN": ("该系统用于什么场景？", "请选择最接近的受控使用领域。"),
    },
    "use_task": {
        "en": ("What task does the system perform?", "Free text is retained, but does not by itself establish a legal fact."),
        "zh-CN": ("该系统执行什么任务？", "系统会保留原始文本，但自由文本本身不会形成正式法律事实。"),
    },
    "affected_persons": {
        "en": ("Who may be affected by the system?", "Select every known group."),
        "zh-CN": ("该系统可能影响哪些人员？", "请选择所有已知群体。"),
    },
    "decision_impact": {
        "en": ("Does the output materially influence a decision or operational outcome?", "Use Unknown when the effect has not been confirmed."),
        "zh-CN": ("系统输出是否会实质影响个人决定或运营结果？", "尚未确认影响时请选择“未知”。"),
    },
    "human_review_before_effect": {
        "en": ("Is there human review before the output takes effect?", "Record the actual workflow, not the intended policy."),
        "zh-CN": ("输出产生效果前是否经过人工复核？", "请按实际流程回答，而非仅按制度设计回答。"),
    },
    "personal_data_processed": {
        "en": ("Does the system process personal data?", "This is a factual screening input, not a GDPR conclusion."),
        "zh-CN": ("该系统是否处理个人数据？", "这是事实筛查输入，不是 GDPR 法律结论。"),
    },
    "automated_individual_decision": {
        "en": ("Is an automated decision about an individual involved?", "Include decisions produced or materially driven by automated processing."),
        "zh-CN": ("是否涉及针对个人的自动化决定？", "包括由自动化处理作出或实质推动的个人决定。"),
    },
    "connected_product": {
        "en": ("Is a connected product involved?", "Examples include connected machinery or equipment."),
        "zh-CN": ("是否涉及互联产品？", "例如联网机械或互联设备。"),
    },
    "related_service": {
        "en": ("Is a related service involved?", "Answer independently from the connected-product field."),
        "zh-CN": ("是否涉及相关服务？", "请与互联产品字段分别判断。"),
    },
    "data_generated": {
        "en": ("Does the product or related service generate data?", "Record whether operational or use data is generated."),
        "zh-CN": ("产品或相关服务是否生成数据？", "请记录是否生成运行数据或使用数据。"),
    },
    "ai_is_product": {
        "en": (
            "Is the AI system itself a product placed on the market or put into service?",
            "Answer Yes only where the AI system itself is treated as the relevant regulated product.",
        ),
        "zh-CN": (
            "该 AI 系统本身是否作为产品投放市场或投入使用？",
            "仅当该 AI 系统本身被视为相关受监管产品时选择“是”。",
        ),
    },
    "ai_is_safety_component": {
        "en": (
            "Is the AI system intended to perform a safety function as part of a product?",
            "Monitoring or optimisation alone is insufficient; consider whether it performs a safety function or its failure could endanger people or property.",
        ),
        "zh-CN": (
            "该 AI 系统是否拟作为产品的一部分承担安全功能？",
            "仅用于监测或优化并不足够；请判断其是否承担安全功能，或其失效是否可能危及人员或财产。",
        ),
    },
    "product_type": {
        "en": (
            "What type of product is involved?",
            "Use a plain-language category such as machinery, medical device, lift, radio equipment or pressure equipment. This does not establish Annex I coverage.",
        ),
        "zh-CN": (
            "涉及哪类产品？",
            "请使用机械、医疗器械、电梯、无线电设备或压力设备等通俗类别。该回答本身不能确认附件 I 覆盖。",
        ),
    },
    "annex_i_instrument": {
        "en": (
            "Which Annex I product legislation may cover the product?",
            "Select the specific legislation or Unknown. Selection identifies a candidate instrument and does not confirm that it applies.",
        ),
        "zh-CN": (
            "哪一项附件 I 产品法规可能涵盖该产品？",
            "请选择具体法规或“未知”。选择仅用于识别候选法规，并不确认该法规适用。",
        ),
    },
    "annex_i_instrument_confirmed": {
        "en": (
            "Has the selected Annex I legislation been confirmed as applying to this product?",
            "Confirm coverage separately after reviewing the product law. Selecting an instrument is not confirmation.",
        ),
        "zh-CN": (
            "是否已确认所选附件 I 法规适用于该产品？",
            "请在核对产品法规后单独确认覆盖范围。选择法规并不等于确认适用。",
        ),
    },
    "third_party_conformity_required": {
        "en": (
            "Must an independent third party assess conformity before market placement or putting into service?",
            "Verify the product-specific route against applicable product law and technical documentation. The catalogue does not answer this question.",
        ),
        "zh-CN": (
            "该产品在投放市场或投入使用前是否必须由独立第三方进行合格评定？",
            "请结合适用产品法规和技术文件核实具体路径。目录不会自动回答此问题。",
        ),
    },
    "confirm_ai_act_product_safety": {
        "en": (
            "Confirm Article 6(1) product-safety screening",
            "Confirmation enables the implemented module but does not set any legal predicate.",
        ),
        "zh-CN": (
            "确认第 6 条第 1 款产品安全筛查",
            "确认只会启用已实现模块，不会自动设置任何法律判断事实。",
        ),
    },
}

for _question_key, _localized in _ROUTED_QUESTION_COPY.items():
    for _language, (_label, _help) in _localized.items():
        TRANSLATIONS[_language][f"question.{_question_key}.label.en"] = _label
        TRANSLATIONS[_language][f"question.{_question_key}.label.zh_cn"] = _label
        TRANSLATIONS[_language][f"question.{_question_key}.help.en"] = _help
        TRANSLATIONS[_language][f"question.{_question_key}.help.zh_cn"] = _help

TRANSLATIONS["en"].update(
    {
        "questionnaire.hints.label": "Controlled use-case descriptors",
        "questionnaire.hints.help": "Select only descriptors you can explicitly confirm. These are routing inputs, not legal conclusions.",
        "questionnaire.modules.eyebrow": "Deterministic routing",
        "questionnaire.modules.title": "Assessment modules",
        "questionnaire.modules.copy": "Review suggested modules before any formal screening is run.",
        "questionnaire.suggested.title": "Suggested assessment modules",
        "questionnaire.suggested.none": "No implemented module is currently suggested from confirmed facts and routing inputs.",
        "questionnaire.confirmed.title": "Confirmed assessment modules",
        "questionnaire.confirmed.none": "No assessment module has been confirmed.",
        "questionnaire.unsupported.title": "Potentially relevant routes not yet supported",
        "questionnaire.unsupported.none": "No unsupported route was identified from the current controlled inputs.",
        "questionnaire.screened.title": "Screened-out routes",
        "questionnaire.screened.none": "No implemented route is currently screened out.",
        "questionnaire.routing_audit.title": "Technical details / Routing audit",
        "questionnaire.routing_audit.screened": (
            "Implemented modules excluded by current facts"
        ),
        "evidence.navigation.title.AI_ACT_HIGH_RISK_PRODUCT_SAFETY": (
            "EU AI Act Article 6(1) product-safety route"
        ),
        "evidence.navigation.title.EU_DATA_ACT_RELEVANCE": (
            "EU Data Act relevance screening"
        ),
        "evidence.navigation.action.AI_ACT_HIGH_RISK_PRODUCT_SAFETY": (
            "View AI Act Evidence Trace"
        ),
        "evidence.navigation.action.EU_DATA_ACT_RELEVANCE": (
            "View Data Act Evidence Trace"
        ),
        "evidence.navigation.count.one": "{count} Evidence record",
        "evidence.navigation.count.other": "{count} Evidence records",
        "evidence.trace.currently_viewing": "Currently viewing: {finding}",
        "questionnaire.why_suggested": "Why this was suggested",
        "questionnaire.confirm": "Confirm module",
        "questionnaire.decline": "Decline",
        "questionnaire.remove": "Remove",
        "questionnaire.manual.title": "Manually select an implemented module",
        "questionnaire.manual.label": "Modules available for screening",
        "questionnaire.manual.confirm": "Confirm selected modules",
        "questionnaire.followups.title": "Module follow-up questions",
        "questionnaire.followups.copy": "Only missing facts required by confirmed modules are shown.",
        "questionnaire.followups.none": "Confirmed modules have no unanswered required questions.",
        "questionnaire.followups.save": "Save follow-up answers",
        "questionnaire.followups.saved": "Follow-up facts saved.",
        "questionnaire.followups.no_changes": "No answer was selected to save.",
        "questionnaire.followups.unresolved_unknown": (
            "This module remains unresolved because one or more required answers "
            "are recorded as Unknown. You may run with information gaps or edit "
            "the recorded answer."
        ),
        "questionnaire.response.recorded_unknown": "Recorded: Unknown",
        "questionnaire.response.edit": "Edit answer",
        "assessment.confirm_module_first": "Confirm at least one implemented assessment module before running the assessment.",
        "trace.fact_source.dynamic_questionnaire": "Dynamic questionnaire",
        "trace.fact_source.demo_fixture": "Demo fixture",
        "trace.fact_source.user_confirmed": "User-confirmed structured input",
        "trace.fact_source.normalization": "Deterministic normalization",
        "trace.fact_source.case_record": "Current case record",
        "trace.rule.mapping": "View condition-to-fact mapping",
        "trace.rule.conditions": "Conditions satisfied: {matched} of {total}",
        "trace.rule.overall_result": "Overall result",
        "trace.rule.name.AI_ACT_HIGH_RISK_EMPLOYMENT": "AI Act employment high-risk test",
        "trace.rule.name.AI_ACT_HIGH_RISK_PRODUCT_SAFETY": "AI Act Article 6(1) product-safety test",
        "trace.rule.name.GDPR_ARTICLE22_RELEVANCE": "GDPR Article 22 relevance test",
        "trace.rule.name.EU_DATA_ACT_RELEVANCE": "EU Data Act relevance test",
        "trace.rule.explanation.GDPR_ARTICLE22_RELEVANCE.potentially_applies": (
            "The confirmed facts satisfy all predicates of the relevance-screening "
            "rule. GDPR Article 22 may therefore be relevant and further legal "
            "review is required."
        ),
        "module.ai_act.employment": "AI Act employment high-risk screening",
        "module.gdpr.article22": "GDPR Article 22 relevance screening",
        "module.data_act.relevance": "EU Data Act relevance screening",
        "module.ai_act.credit_essential_services": "AI Act credit and essential-services assessment",
        "module.ai_act.judicial": "AI Act judicial-system assessment",
        "module.ai_act.product_safety": "AI Act product-safety assessment",
        "module.ai_act.product_safety.boundary": "This module assesses only the Article 6(1) product and safety-component route. Other AI Act high-risk routes may require separate assessment.",
        "question.annex_i_instrument.option.unknown": "Unknown / not yet identified",
        "question.unsupported_ai_act_credit.label.en": "This legal assessment module is not yet implemented.",
        "question.unsupported_ai_act_credit.help.en": "No applicability or compliance conclusion is produced for this route.",
        "question.unsupported_ai_act_judicial.label.en": "This legal assessment module is not yet implemented.",
        "question.unsupported_ai_act_judicial.help.en": "No applicability or compliance conclusion is produced for this route.",
        "question.unsupported_ai_act_product_safety.label.en": "This legal assessment module is not yet implemented.",
        "question.unsupported_ai_act_product_safety.help.en": "No applicability or compliance conclusion is produced for this route.",
        "routing_hint.employment.recruitment": "Recruitment",
        "routing_hint.employment.selection": "Candidate or worker selection",
        "routing_hint.employment.candidate_ranking": "Candidate ranking",
        "routing_hint.employment.worker_management": "Worker management",
        "routing_hint.decision.individual_significant": "Individual decision with legal or significant effect",
        "routing_hint.decision.credit": "Credit or loan decision",
        "routing_hint.data_act.industrial_connected_equipment": "Industrial connected equipment",
        "routing_hint.ai_act.product_safety_component": "AI safety component of a regulated product",
        "routing_hint.ai_act.regulated_ai_product": "AI system itself is a regulated product",
        "routing_hint.ai_act.product_safety_context": "Machinery or industrial product-safety context",
        "routing_hint.ai_act.medical_device_context": "Medical-device context",
        "routing_hint.ai_act.regulated_equipment_context": "Regulated-equipment context",
        "routing_hint.ai_act.conformity_assessment": "Third-party conformity-assessment relevance",
        "routing_reason.EMPLOYMENT_CONTEXT_AND_CONFIRMED_FUNCTION": "Employment domain and a confirmed employment function.",
        "routing_reason.PERSONAL_DATA_AND_AUTOMATED_SIGNIFICANT_DECISION": "Confirmed personal-data processing and automated significant decision facts.",
        "routing_reason.PERSONAL_DATA_AND_CONFIRMED_INDIVIDUAL_DECISION_CONTEXT": "Personal data and a confirmed individual-decision context.",
        "routing_reason.CONNECTED_PRODUCT_OR_SERVICE_GENERATES_DATA": "A connected product or related service generates data.",
        "routing_reason.CONFIRMED_CONNECTED_EQUIPMENT_CONTEXT": "A connected-equipment context was explicitly confirmed.",
        "routing_reason.CONFIRMED_PRODUCT_RELATIONSHIP": "A product or safety-component relationship was explicitly confirmed.",
        "routing_reason.CONTROLLED_PRODUCT_SAFETY_CONTEXT": "A controlled product-safety routing descriptor was confirmed.",
        "routing_reason.PRODUCT_SAFETY_DOMAIN": "The controlled use domain is product safety.",
        "routing_reason.ANNEX_I_INSTRUMENT_SELECTED": "A stable Annex I catalogue instrument was selected; this does not confirm coverage.",
        "routing_reason.USER_CONFIRMED_MODULE": "The user manually confirmed this module.",
        "rule.AI_ACT_HIGH_RISK_PRODUCT_SAFETY": "Article 6(1) product-safety screening",
        "fact.product_regulation.ai_is_product": "AI system is itself the product",
        "fact.product_regulation.ai_is_safety_component": "AI system is a safety component",
        "fact.product_regulation.product_type": "Product category",
        "fact.product_regulation.annex_i_instrument": "Annex I legislation",
        "fact.product_regulation.annex_i_instrument_confirmed": "Annex I coverage confirmation",
        "fact.product_regulation.third_party_conformity_required": "Third-party conformity assessment",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.potentially_applies.title": "Product-safety high-risk classification potentially applies",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.potentially_applies.summary": "The confirmed facts satisfy the preliminary Article 6(1) product and safety-component route. Product-law and legal review remain required.",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.does_not_apply.title": "Article 6(1) product-safety screening criteria not met",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.does_not_apply.summary": "The confirmed facts do not meet this Article 6(1) product-safety route. This does not exclude Article 6(2), another Annex III category, or another applicable law.",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.undetermined.title": "Article 6(1) assessment requires fact reconciliation",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.undetermined.summary": "The product-regulation facts must be reconciled before an Article 6(1) conclusion can be drawn.",
        "affected_person.worker": "Worker",
        "affected_person.job_candidate": "Job candidate",
        "affected_person.student": "Student",
        "affected_person.consumer": "Consumer",
        "affected_person.patient": "Patient",
        "affected_person.child": "Child",
        "affected_person.other": "Other person",
    }
)

TRANSLATIONS["zh-CN"].update(
    {
        "questionnaire.hints.label": "受控使用场景描述",
        "questionnaire.hints.help": "仅选择可以明确确认的描述。这些内容只用于路由，不构成法律结论。",
        "questionnaire.modules.eyebrow": "确定性路由",
        "questionnaire.modules.title": "评估模块",
        "questionnaire.modules.copy": "运行正式筛查前，请先审阅并确认建议模块。",
        "questionnaire.suggested.title": "建议的评估模块",
        "questionnaire.suggested.none": "根据当前已确认事实和路由输入，暂未建议已实现模块。",
        "questionnaire.confirmed.title": "已确认的评估模块",
        "questionnaire.confirmed.none": "尚未确认任何评估模块。",
        "questionnaire.unsupported.title": "可能相关但尚未支持的路径",
        "questionnaire.unsupported.none": "当前受控输入未识别出尚未支持的路径。",
        "questionnaire.screened.title": "已筛除的路径",
        "questionnaire.screened.none": "当前没有已筛除的已实现路径。",
        "questionnaire.routing_audit.title": "技术详情 / 路由审计",
        "questionnaire.routing_audit.screened": "根据当前事实已排除的已实现模块",
        "evidence.navigation.title.AI_ACT_HIGH_RISK_PRODUCT_SAFETY": (
            "《欧盟人工智能法案》第6条第1款产品安全路径"
        ),
        "evidence.navigation.title.EU_DATA_ACT_RELEVANCE": (
            "《欧盟数据法案》相关性筛查"
        ),
        "evidence.navigation.action.AI_ACT_HIGH_RISK_PRODUCT_SAFETY": (
            "查看《欧盟人工智能法案》证据链"
        ),
        "evidence.navigation.action.EU_DATA_ACT_RELEVANCE": (
            "查看《欧盟数据法案》证据链"
        ),
        "evidence.navigation.count.one": "{count} 条证据",
        "evidence.navigation.count.other": "{count} 条证据",
        "evidence.trace.currently_viewing": "当前查看：{finding}",
        "questionnaire.why_suggested": "建议原因",
        "questionnaire.confirm": "确认模块",
        "questionnaire.decline": "暂不评估",
        "questionnaire.remove": "移除",
        "questionnaire.manual.title": "手动选择已实现模块",
        "questionnaire.manual.label": "可供筛查的模块",
        "questionnaire.manual.confirm": "确认所选模块",
        "questionnaire.followups.title": "模块补充问题",
        "questionnaire.followups.copy": "仅显示已确认模块尚缺失的必要事实。",
        "questionnaire.followups.none": "已确认模块不存在尚未回答的必要问题。",
        "questionnaire.followups.save": "保存补充回答",
        "questionnaire.followups.saved": "补充事实已保存。",
        "questionnaire.followups.no_changes": "尚未选择需要保存的回答。",
        "questionnaire.followups.unresolved_unknown": (
            "一个或多个必要回答已记录为未知，因此本模块仍未解决。你可以在保留"
            "信息缺口的情况下运行评估，或编辑已记录的回答。"
        ),
        "questionnaire.response.recorded_unknown": "已记录：未知",
        "questionnaire.response.edit": "编辑回答",
        "assessment.confirm_module_first": "运行评估前，请至少确认一个已实现的评估模块。",
        "trace.fact_source.dynamic_questionnaire": "动态问卷",
        "trace.fact_source.demo_fixture": "演示案例预设事实",
        "trace.fact_source.user_confirmed": "用户确认的结构化输入",
        "trace.fact_source.normalization": "确定性文本规范化",
        "trace.fact_source.case_record": "当前案例事实记录",
        "trace.rule.mapping": "查看规则条件与事实映射",
        "trace.rule.conditions": "已满足条件：{matched}/{total}",
        "trace.rule.overall_result": "总体判断结果",
        "trace.rule.name.AI_ACT_HIGH_RISK_EMPLOYMENT": "《欧盟人工智能法案》就业高风险测试",
        "trace.rule.name.AI_ACT_HIGH_RISK_PRODUCT_SAFETY": "《欧盟人工智能法案》第 6 条第 1 款产品安全测试",
        "trace.rule.name.GDPR_ARTICLE22_RELEVANCE": "GDPR 第22条相关性测试",
        "trace.rule.name.EU_DATA_ACT_RELEVANCE": "《欧盟数据法案》相关性测试",
        "trace.rule.explanation.GDPR_ARTICLE22_RELEVANCE.potentially_applies": (
            "现有事实满足该相关性筛查规则的全部判断条件，因此可能涉及 GDPR 第22条，"
            "需要进一步法律审查。"
        ),
        "module.ai_act.employment": "《欧盟人工智能法案》就业高风险筛查",
        "module.gdpr.article22": "GDPR 第 22 条相关性筛查",
        "module.data_act.relevance": "《欧盟数据法案》相关性筛查",
        "module.ai_act.credit_essential_services": "《欧盟人工智能法案》信贷与基本服务评估",
        "module.ai_act.judicial": "《欧盟人工智能法案》司法系统评估",
        "module.ai_act.product_safety": "《欧盟人工智能法案》产品安全评估",
        "module.ai_act.product_safety.boundary": "本模块仅评估《人工智能法案》第6条第1款的产品及安全部件路径。其他高风险分类路径仍可能需要单独评估。",
        "question.annex_i_instrument.option.unknown": "未知／尚未识别",
        "question.unsupported_ai_act_credit.label.zh_cn": "当前版本尚未实现该法律评估模块。",
        "question.unsupported_ai_act_credit.help.zh_cn": "本路径不会生成适用、不适用或合规结论。",
        "question.unsupported_ai_act_judicial.label.zh_cn": "当前版本尚未实现该法律评估模块。",
        "question.unsupported_ai_act_judicial.help.zh_cn": "本路径不会生成适用、不适用或合规结论。",
        "question.unsupported_ai_act_product_safety.label.zh_cn": "当前版本尚未实现该法律评估模块。",
        "question.unsupported_ai_act_product_safety.help.zh_cn": "本路径不会生成适用、不适用或合规结论。",
        "routing_hint.employment.recruitment": "招聘",
        "routing_hint.employment.selection": "候选人或劳动者甄选",
        "routing_hint.employment.candidate_ranking": "候选人排序",
        "routing_hint.employment.worker_management": "劳动者管理",
        "routing_hint.decision.individual_significant": "对个人产生法律效果或类似重大影响的决定",
        "routing_hint.decision.credit": "信贷或贷款决定",
        "routing_hint.data_act.industrial_connected_equipment": "工业互联设备",
        "routing_hint.ai_act.product_safety_component": "受监管产品的 AI 安全部件",
        "routing_hint.ai_act.regulated_ai_product": "AI 系统本身属于受监管产品",
        "routing_hint.ai_act.product_safety_context": "机械或工业产品安全场景",
        "routing_hint.ai_act.medical_device_context": "医疗器械场景",
        "routing_hint.ai_act.regulated_equipment_context": "受监管设备场景",
        "routing_hint.ai_act.conformity_assessment": "第三方合格评定相关性",
        "routing_reason.EMPLOYMENT_CONTEXT_AND_CONFIRMED_FUNCTION": "就业领域与已确认的就业功能同时存在。",
        "routing_reason.PERSONAL_DATA_AND_AUTOMATED_SIGNIFICANT_DECISION": "已确认个人数据处理及自动化重大决定事实。",
        "routing_reason.PERSONAL_DATA_AND_CONFIRMED_INDIVIDUAL_DECISION_CONTEXT": "涉及个人数据，且已确认个人决定场景。",
        "routing_reason.CONNECTED_PRODUCT_OR_SERVICE_GENERATES_DATA": "互联产品或相关服务会生成数据。",
        "routing_reason.CONFIRMED_CONNECTED_EQUIPMENT_CONTEXT": "已明确确认互联设备场景。",
        "routing_reason.CONFIRMED_PRODUCT_RELATIONSHIP": "已明确确认产品或安全部件关系。",
        "routing_reason.CONTROLLED_PRODUCT_SAFETY_CONTEXT": "已确认受控的产品安全路由描述。",
        "routing_reason.PRODUCT_SAFETY_DOMAIN": "受控使用领域为产品安全。",
        "routing_reason.ANNEX_I_INSTRUMENT_SELECTED": "已选择稳定的附件 I 目录法规；该选择不构成覆盖确认。",
        "routing_reason.USER_CONFIRMED_MODULE": "用户已手动确认该模块。",
        "rule.AI_ACT_HIGH_RISK_PRODUCT_SAFETY": "第 6 条第 1 款产品安全筛查",
        "fact.product_regulation.ai_is_product": "AI 系统本身属于产品",
        "fact.product_regulation.ai_is_safety_component": "AI 系统属于安全部件",
        "fact.product_regulation.product_type": "产品类别",
        "fact.product_regulation.annex_i_instrument": "附件 I 法规",
        "fact.product_regulation.annex_i_instrument_confirmed": "附件 I 覆盖确认",
        "fact.product_regulation.third_party_conformity_required": "第三方合格评定",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.potentially_applies.title": "产品安全路径的高风险分类可能适用",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.potentially_applies.summary": "现有确认事实满足第 6 条第 1 款产品及安全部件路径的初步条件，仍需产品法规及法律复核。",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.does_not_apply.title": "第 6 条第 1 款产品安全路径初筛条件未满足",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.does_not_apply.summary": "现有确认事实不满足第 6 条第 1 款产品安全路径；本结论不排除第 6 条第 2 款、附件 III 其他类别或其他适用法律。",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.undetermined.title": "第 6 条第 1 款评估需要核对事实",
        "finding.AI_ACT_HIGH_RISK_PRODUCT_SAFETY.undetermined.summary": "在形成第 6 条第 1 款结论前，需要先核对产品监管事实。",
        "affected_person.worker": "劳动者",
        "affected_person.job_candidate": "求职者",
        "affected_person.student": "学生",
        "affected_person.consumer": "消费者",
        "affected_person.patient": "患者",
        "affected_person.child": "儿童",
        "affected_person.other": "其他人员",
    }
)


def normalize_language(language: str | None) -> str:
    """Return a supported language code, defaulting safely to English."""

    return language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def t(key: str, language: str = DEFAULT_LANGUAGE, **values: Any) -> str:
    """Translate one UI key with safe English fallback and formatting."""

    normalized = normalize_language(language)
    template = TRANSLATIONS.get(normalized, {}).get(key)
    if template is None:
        template = TRANSLATIONS[DEFAULT_LANGUAGE].get(key, key)
    return template.format(**values) if values else template


def t_or(
    key: str,
    fallback: str,
    language: str = DEFAULT_LANGUAGE,
    **values: Any,
) -> str:
    """Translate a key, returning caller-provided legal copy if unavailable."""

    normalized = normalize_language(language)
    template = TRANSLATIONS.get(normalized, {}).get(key)
    if template is None:
        template = TRANSLATIONS[DEFAULT_LANGUAGE].get(key, fallback)
    return template.format(**values) if values else template


def count_text(key: str, count: int, language: str = DEFAULT_LANGUAGE) -> str:
    """Return a localized count with English singular/plural handling."""

    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError("count must be an integer")
    suffix = "zero" if count == 0 and f"{key}.zero" in TRANSLATIONS["en"] else (
        "one" if count == 1 else "other"
    )
    return t(f"{key}.{suffix}", language, count=count)
