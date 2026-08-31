"""Normalization and compatibility summaries for multi-foul model output."""

from typing import Optional


VALID_TEAMS = {"offense", "defense", "kicking", "receiving", "unknown"}


def _confidence(value, default=0.0):
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def normalize_findings_payload(
    data: dict,
    summary: str,
    confidence_threshold: float,
    play_duration: Optional[float] = None,
) -> dict:
    raw_findings = data.get("findings")
    if not isinstance(raw_findings, list):
        raw_findings = []
        legacy_result = str(data.get("result") or "").strip().upper()
        if legacy_result in {"FOUL", "UNCERTAIN"}:
            raw_findings.append({
                "disposition": legacy_result,
                "foul_label": data.get("label"),
                "confidence": data.get("confidence"),
                "evidence_summary": data.get("evidence_summary"),
                "reasoning": data.get("why_foul_or_no_foul"),
                "rule_citations": data.get("rule_citations"),
                "uncertainty_notes": data.get("uncertainty_notes"),
            })

    findings = []
    seen = set()
    for raw in raw_findings[:8]:
        if not isinstance(raw, dict):
            continue
        disposition = str(raw.get("disposition") or "UNCERTAIN").strip().upper()
        if disposition == "NO FOUL":
            continue
        if disposition not in {"FOUL", "UNCERTAIN"}:
            disposition = "UNCERTAIN"
        confidence = _confidence(raw.get("confidence"))
        if disposition == "FOUL" and confidence < confidence_threshold:
            disposition = "UNCERTAIN"
        label = str(raw.get("foul_label") or raw.get("label") or "Uncertain").strip()
        if label.lower() in {"", "none", "no foul"}:
            label = "Uncertain"
            disposition = "UNCERTAIN"
        team = str(raw.get("committing_team") or "unknown").strip().lower()
        if team not in VALID_TEAMS:
            team = "unknown"
        player_number = str(raw.get("player_number") or "").strip() or None
        try:
            event_time = float(raw.get("event_time"))
            event_time = max(0.0, event_time)
            if play_duration is not None:
                event_time = min(play_duration, event_time)
            event_time = round(event_time, 2)
        except (TypeError, ValueError):
            event_time = None
        citations = raw.get("rule_citations") or []
        if not isinstance(citations, list):
            citations = [citations]
        dedupe_key = (label.lower(), event_time, team, player_number)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        findings.append({
            "disposition": disposition,
            "foul_label": label,
            "committing_team": team,
            "player_number": player_number,
            "event_time": event_time,
            "confidence": confidence,
            "evidence_summary": str(raw.get("evidence_summary") or summary).strip(),
            "reasoning": str(raw.get("reasoning") or raw.get("why_foul_or_no_foul") or "").strip(),
            "rule_citations": [str(citation) for citation in citations if citation],
            "uncertainty_notes": str(raw.get("uncertainty_notes") or "").strip(),
        })

    foul_findings = [finding for finding in findings if finding["disposition"] == "FOUL"]
    uncertain_findings = [finding for finding in findings if finding["disposition"] == "UNCERTAIN"]
    if foul_findings:
        result = "FOUL"
        primary = max(foul_findings, key=lambda finding: finding["confidence"])
    elif uncertain_findings:
        result = "UNCERTAIN"
        primary = max(uncertain_findings, key=lambda finding: finding["confidence"])
    else:
        result = "NO FOUL"
        primary = None

    confidence = (
        primary["confidence"] if primary else _confidence(data.get("confidence"), 0.0)
    )
    no_foul_reason = str(data.get("no_foul_reason") or "").strip()
    return {
        "result": result,
        "label": primary["foul_label"] if primary else "None",
        "confidence": confidence,
        "findings": findings,
        "evidence_summary": str(data.get("play_summary") or data.get("evidence_summary") or summary).strip(),
        "why_foul_or_no_foul": (
            primary["reasoning"] if primary else no_foul_reason
        ),
        "rule_citations": primary["rule_citations"] if primary else [],
        "no_foul_reason": no_foul_reason,
        "uncertainty_notes": str(data.get("uncertainty_notes") or "").strip(),
    }
