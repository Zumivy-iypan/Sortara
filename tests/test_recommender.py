from sortara.config import CLASS_LABELS, DEFAULT_CONFIDENCE_THRESHOLD
from sortara.recommender import build_recommendation


def test_build_recommendation_returns_valid_guidance_for_all_six_categories_fr09():
    """FR-09 & FR-12 + Fitur Edukasi: Setiap kategori memiliki rekomendasi, waktu terurai, sebab-akibat, upcycling, & warna tong."""
    for category in CLASS_LABELS:
        rec = build_recommendation(
            predicted_category=category,
            confidence_score=0.88,
            threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        assert rec.is_confident is True
        assert rec.category == category
        assert len(rec.examples) > 0
        assert len(rec.handling_advice) > 10
        assert len(rec.decomposition_time) > 3
        assert len(rec.littering_impact) > 10
        assert len(rec.recycling_benefit) > 10
        assert len(rec.bin_color_guide) > 5
        assert len(rec.upcycling_ideas) >= 2
        assert "setempat" in rec.local_rule_disclaimer.lower() or "lokal" in rec.local_rule_disclaimer.lower()


def test_build_recommendation_handles_low_confidence_prediction_fr10_at06():
    """FR-10 & AT-06: Jika skor di bawah ambang, tampilkan pesan belum dapat dikenali dan saran foto ulang."""
    rec = build_recommendation(
        predicted_category="Plastik",
        confidence_score=0.42,
        threshold=0.60,
    )
    assert rec.is_confident is False
    assert "belum dapat mengenali" in rec.low_confidence_message.lower()
    assert "satu benda" in rec.low_confidence_message.lower()
