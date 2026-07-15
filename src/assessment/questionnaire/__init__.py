"""Question models, registration, and missing-fact planning."""

from src.assessment.questionnaire.engine import QuestionnaireEngine
from src.assessment.questionnaire.definitions import (
    QUESTION_DEFINITIONS,
    RULE_QUESTIONNAIRE_DEFINITIONS,
    UNSUPPORTED_PATH_DEFINITIONS,
    RuleQuestionnaireRegistry,
    build_question_registry,
    build_rule_questionnaire_registry,
)
from src.assessment.questionnaire.invalidation import calculate_invalidations
from src.assessment.questionnaire.models import (
    AnswerType,
    Question,
    QuestionnairePlan,
    QuestionOption,
)
from src.assessment.questionnaire.registry import (
    DuplicateQuestionError,
    DuplicateQuestionFactPathError,
    InvalidQuestionFactPathError,
    QuestionNotFoundError,
    QuestionRegistry,
)
from src.assessment.questionnaire.router import (
    QuestionnaireRouter,
    build_default_questionnaire_router,
)
from src.assessment.questionnaire.routing_models import (
    EligibilityHintGroup,
    FactCondition,
    FactConditionOperator,
    FactProvenance,
    InvalidationResult,
    LocalizedTextKeys,
    QuestionnaireRoute,
    QuestionDependency,
    QuestionInvalidation,
    QuestionResponseState,
    RoutingQuestionDefinition,
    RuleQuestionnaireDefinition,
    UnsupportedPathDefinition,
    UnsupportedPathRoute,
)

__all__ = [
    "AnswerType",
    "DuplicateQuestionError",
    "DuplicateQuestionFactPathError",
    "EligibilityHintGroup",
    "FactCondition",
    "FactConditionOperator",
    "FactProvenance",
    "InvalidationResult",
    "InvalidQuestionFactPathError",
    "LocalizedTextKeys",
    "QUESTION_DEFINITIONS",
    "Question",
    "QuestionDependency",
    "QuestionInvalidation",
    "QuestionResponseState",
    "QuestionnaireEngine",
    "QuestionnairePlan",
    "QuestionnaireRoute",
    "QuestionnaireRouter",
    "QuestionNotFoundError",
    "QuestionOption",
    "QuestionRegistry",
    "RULE_QUESTIONNAIRE_DEFINITIONS",
    "RoutingQuestionDefinition",
    "RuleQuestionnaireDefinition",
    "RuleQuestionnaireRegistry",
    "UNSUPPORTED_PATH_DEFINITIONS",
    "UnsupportedPathDefinition",
    "UnsupportedPathRoute",
    "build_default_questionnaire_router",
    "build_question_registry",
    "build_rule_questionnaire_registry",
    "calculate_invalidations",
]
