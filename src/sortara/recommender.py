from dataclasses import dataclass

from sortara.config import (
    CATEGORY_RULES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    LOCAL_RULE_DISCLAIMER,
    LOW_CONFIDENCE_MESSAGE,
)


@dataclass(frozen=True)
class RecommendationOutput:
    is_confident: bool
    category: str | None
    confidence_score: float
    threshold_used: float
    icon: str
    badge_color: str
    examples: str
    handling_advice: str
    preparation_steps: tuple[str, ...]
    bin_color_guide: str
    bin_badge_color: str
    decomposition_time: str
    littering_impact: str
    recycling_benefit: str
    upcycling_ideas: tuple[str, ...]
    low_confidence_message: str
    local_rule_disclaimer: str


def build_recommendation(
    predicted_category: str,
    confidence_score: float,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> RecommendationOutput:
    """Membuat rekomendasi penanganan, edukasi lingkungan, & panduan tong sampah (FR-09, FR-10, FR-12)."""
    rule = CATEGORY_RULES.get(predicted_category)
    is_confident = confidence_score >= threshold and rule is not None

    if not is_confident or rule is None:
        return RecommendationOutput(
            is_confident=False,
            category=None,
            confidence_score=confidence_score,
            threshold_used=threshold,
            icon=":material/help_center:",
            badge_color="orange",
            examples="",
            handling_advice="",
            preparation_steps=(
                "Pastikan hanya ada satu benda sampah utama di dalam bingkai foto.",
                "Gunakan pencahayaan yang cukup dan hindari bayangan gelap atau silau berlebih.",
                "Dekatkan kamera agar bentuk dan tekstur material benda terlihat jelas.",
            ),
            bin_color_guide="",
            bin_badge_color="gray",
            decomposition_time="",
            littering_impact="",
            recycling_benefit="",
            upcycling_ideas=(),
            low_confidence_message=LOW_CONFIDENCE_MESSAGE,
            local_rule_disclaimer=LOCAL_RULE_DISCLAIMER,
        )

    return RecommendationOutput(
        is_confident=True,
        category=rule.category,
        confidence_score=confidence_score,
        threshold_used=threshold,
        icon=rule.icon,
        badge_color=rule.badge_color,
        examples=rule.examples,
        handling_advice=rule.handling_advice,
        preparation_steps=rule.preparation_steps,
        bin_color_guide=rule.bin_color_guide,
        bin_badge_color=rule.bin_badge_color,
        decomposition_time=rule.decomposition_time,
        littering_impact=rule.littering_impact,
        recycling_benefit=rule.recycling_benefit,
        upcycling_ideas=rule.upcycling_ideas,
        low_confidence_message="",
        local_rule_disclaimer=LOCAL_RULE_DISCLAIMER,
    )
