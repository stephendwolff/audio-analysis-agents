"""Rule-based descriptor generation from analysis dimensions."""


def generate_descriptors(dimensions: dict) -> list[str]:
    """
    Generate human-readable rhythm descriptors from dimension values.

    Args:
        dimensions: Dict with keys bpm, swing, steadiness, upbeat

    Returns:
        List of descriptor strings
    """
    descriptors = []
    bpm = dimensions.get("bpm", 0)
    swing = dimensions.get("swing", 0)
    steadiness = dimensions.get("steadiness", 0)
    upbeat = dimensions.get("upbeat", False)

    # Tempo
    if bpm > 140:
        descriptors.append("driving")
    elif bpm < 90:
        descriptors.append("laid-back")
    else:
        descriptors.append("moderate-tempo")

    # Swing
    if swing > 0.4:
        descriptors.append("swung")
    elif swing < 0.1:
        descriptors.append("straight")

    # Steadiness
    if steadiness > 0.8:
        descriptors.append("steady")
    elif steadiness < 0.4:
        descriptors.append("loose")

    # Upbeat
    if upbeat:
        descriptors.append("upbeat-start")

    return descriptors
