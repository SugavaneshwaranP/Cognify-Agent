"""
CognifyX – Reflection Agent
Self-correcting agent that audits LLM scoring against hard facts.
Detects score inflation/deflation, inconsistencies, and factual mismatches.
"""
import re
import json


class ReflectionAgent:
    """
    Cross-validates LLM scores against hard facts extracted from profiles.
    Detects anomalies and triggers re-scoring when needed.
    """

    # Thresholds for anomaly detection
    SCORE_DEVIATION_THRESHOLD = 25   # Max allowed gap between keyword score and LLM score
    MIN_EXPERIENCE_FOR_SENIOR = 4    # Minimum years for "senior" roles
    INFLATION_CEILING = 90           # Scores above this with low keyword match are suspect

    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent
        self.reflection_logs = []

    def log(self, message):
        self.reflection_logs.append(message)

    def _extract_jd_requirements(self, job_description):
        """
        Extract hard requirements from JD text.
        Returns dict with experience_required, required_skills, seniority_level, etc.
        """
        jd_lower = (job_description or "").lower()

        # Extract minimum experience
        exp_match = re.search(r'(\d+)\s*\+?\s*(?:years|yrs)', jd_lower)
        min_experience = int(exp_match.group(1)) if exp_match else 0

        # Detect seniority level
        seniority = "any"
        if any(w in jd_lower for w in ["senior", "sr.", "lead", "principal", "staff"]):
            seniority = "senior"
        elif any(w in jd_lower for w in ["junior", "jr.", "entry", "fresher", "intern"]):
            seniority = "junior"
        elif any(w in jd_lower for w in ["mid", "intermediate"]):
            seniority = "mid"

        # Extract must-have skills (words near "required", "must", "mandatory")
        required_skills = []
        must_sections = re.findall(
            r'(?:required|must have|mandatory|essential)[:\s]*(.+?)(?:\n|$)',
            jd_lower, re.IGNORECASE
        )
        for section in must_sections:
            skills = re.findall(r'\b([a-z]+(?:\s[a-z]+)?)\b', section)
            required_skills.extend(skills)

        # Extract degree requirement
        degree_required = False
        if re.search(r"(?:bachelor|master|degree|b\.?tech|m\.?tech|b\.?s\.?|m\.?s\.?)", jd_lower):
            degree_required = True

        return {
            "min_experience": min_experience,
            "seniority": seniority,
            "required_skills": list(set(required_skills)),
            "degree_required": degree_required
        }

    def _detect_anomalies(self, profile, jd_requirements):
        """
        Detect scoring anomalies by cross-referencing profile facts against scores.
        Returns list of anomaly dicts with type, severity, description, and suggested_adjustment.
        """
        anomalies = []
        llm_score = profile.get('llm_score', 50)
        keyword_score = profile.get('keyword_score', 0)
        ats_score = profile.get('ats_score', 0)
        experience = profile.get('experience_years', 0)
        skills = [s.lower() for s in profile.get('skills', [])]
        name = profile.get('name', profile.get('filename', 'Unknown'))

        # ── Anomaly 1: Score Inflation ──────────────────────────────────
        # High LLM score but low keyword match suggests the LLM was too generous
        score_gap = llm_score - keyword_score
        if score_gap > self.SCORE_DEVIATION_THRESHOLD and keyword_score < 40:
            anomalies.append({
                "type": "SCORE_INFLATION",
                "severity": "HIGH",
                "description": (
                    f"LLM score ({llm_score}) is {score_gap} points higher than keyword match "
                    f"({keyword_score}%). The candidate may lack core JD requirements."
                ),
                "suggested_adjustment": max(10, min(score_gap // 2, 25))
            })

        # ── Anomaly 2: Experience Mismatch ─────────────────────────────
        # JD requires X years but candidate has less
        min_exp = jd_requirements.get('min_experience', 0)
        if min_exp > 0 and experience < min_exp and llm_score > 60:
            exp_gap = min_exp - experience
            anomalies.append({
                "type": "EXPERIENCE_SHORTFALL",
                "severity": "HIGH" if exp_gap >= 3 else "MEDIUM",
                "description": (
                    f"JD requires {min_exp}+ years experience but candidate has only "
                    f"{experience} years (gap: {exp_gap} years). LLM score of {llm_score} "
                    f"may not account for this."
                ),
                "suggested_adjustment": min(20, exp_gap * 5)
            })

        # ── Anomaly 3: Seniority Mismatch ──────────────────────────────
        seniority = jd_requirements.get('seniority', 'any')
        if seniority == "senior" and experience < self.MIN_EXPERIENCE_FOR_SENIOR and llm_score > 55:
            anomalies.append({
                "type": "SENIORITY_MISMATCH",
                "severity": "MEDIUM",
                "description": (
                    f"JD requires a '{seniority}' level candidate, but this candidate has "
                    f"only {experience} years of experience. May not meet seniority expectations."
                ),
                "suggested_adjustment": 10
            })

        # ── Anomaly 4: Score Deflation ─────────────────────────────────
        # High keyword match but very low LLM score → LLM may be too harsh
        if keyword_score > 70 and llm_score < 40:
            anomalies.append({
                "type": "SCORE_DEFLATION",
                "severity": "MEDIUM",
                "description": (
                    f"Keyword match is strong ({keyword_score}%) but LLM score is only "
                    f"{llm_score}. The LLM may have been too conservative."
                ),
                "suggested_adjustment": -min(15, (keyword_score - llm_score) // 3)
            })

        # ── Anomaly 5: Suspiciously High Score ─────────────────────────
        # Perfect or near-perfect scores with mediocre supporting data
        if llm_score >= self.INFLATION_CEILING and keyword_score < 50 and ats_score < 50:
            anomalies.append({
                "type": "SUSPICIOUS_HIGH_SCORE",
                "severity": "HIGH",
                "description": (
                    f"LLM score ({llm_score}) is near-perfect but both ATS ({ats_score}) "
                    f"and keyword ({keyword_score}%) scores are below average. "
                    f"Likely a hallucinated or inflated score."
                ),
                "suggested_adjustment": 20
            })

        # ── Anomaly 6: Missing Critical Skills ────────────────────────
        # Candidate is missing most keywords but still scored highly
        missed = profile.get('keywords_missed', [])
        matched = profile.get('keywords_matched', [])
        total_kw = len(missed) + len(matched)
        if total_kw > 0 and len(missed) > len(matched) and llm_score > 65:
            miss_ratio = len(missed) / total_kw
            anomalies.append({
                "type": "CRITICAL_SKILLS_MISSING",
                "severity": "MEDIUM" if miss_ratio < 0.7 else "HIGH",
                "description": (
                    f"Candidate is missing {len(missed)}/{total_kw} keywords "
                    f"({', '.join(missed[:5])}) but received LLM score of {llm_score}."
                ),
                "suggested_adjustment": min(15, len(missed) * 3)
            })

        return anomalies

    def _apply_corrections(self, profile, anomalies):
        """
        Apply score corrections based on detected anomalies.
        Returns corrected scores and a detailed correction report.
        """
        original_llm = profile.get('llm_score', 50)
        total_adjustment = 0
        corrections = []

        for anomaly in anomalies:
            adj = anomaly.get('suggested_adjustment', 0)
            total_adjustment += adj
            corrections.append({
                "type": anomaly['type'],
                "severity": anomaly['severity'],
                "description": anomaly['description'],
                "adjustment": f"-{adj}" if adj > 0 else f"+{abs(adj)}"
            })

        # Apply adjustment (cap it to avoid extreme swings)
        total_adjustment = min(total_adjustment, 35)  # Max 35-point correction
        corrected_llm = max(5, min(100, original_llm - total_adjustment))

        return {
            "original_score": original_llm,
            "corrected_score": corrected_llm,
            "total_adjustment": total_adjustment,
            "corrections": corrections,
            "was_corrected": total_adjustment != 0
        }

    def _generate_llm_reflection(self, profile, anomalies, jd):
        """
        Use LLM to generate a detailed reflection on the score (when Ollama is available).
        Returns the LLM's reflective analysis.
        """
        if not self.llm_agent:
            return None

        anomaly_text = "\n".join([
            f"- [{a['type']}] {a['description']}" for a in anomalies
        ])

        profile_brief = {k: v for k, v in profile.items()
                         if k not in ('full_text', 'llm_analysis')}

        prompt = f"""You are a senior quality auditor for an AI recruitment system. 
Your job is to REVIEW and VALIDATE the initial scoring of a candidate.

The initial LLM scoring gave this candidate a score of {profile.get('llm_score', 50)}/100.

However, our automated anomaly detector found these issues:
{anomaly_text if anomaly_text else "No anomalies detected."}

Candidate Profile:
{json.dumps(profile_brief, indent=2, default=str)[:2000]}

Job Description:
{jd[:1500]}

INSTRUCTIONS:
1. Review the anomalies above and determine if they are valid concerns.
2. Provide a CORRECTED score (0-100) that accounts for these issues.
3. Start your response with "Corrected Score: XX/100" on the first line.
4. Then explain your reasoning in 2-3 bullet points.
5. End with a CONFIDENCE level: HIGH, MEDIUM, or LOW.
"""
        response, elapsed = self.llm_agent.call_llm(
            self.llm_agent.MODELS.get("insights", "llama3:latest"),
            prompt,
            timeout=30
        )

        return response if response else None

    def reflect(self, profile, job_description, use_llm=True):
        """
        Main reflection method. Runs full audit on a single candidate's scores.

        Args:
            profile: dict with all candidate data including scores
            job_description: the JD text
            use_llm: whether to use LLM for deeper reflection (default True)

        Returns:
            dict with reflection results including corrections and report
        """
        name = profile.get('name', profile.get('filename', 'Unknown'))
        self.log(f"🔍 Reflecting on: {name}")

        # Step 1: Extract JD requirements
        jd_reqs = self._extract_jd_requirements(job_description)

        # Step 2: Detect anomalies
        anomalies = self._detect_anomalies(profile, jd_reqs)

        if not anomalies:
            self.log(f"   ✅ {name}: No anomalies detected. Score validated.")
            return {
                "candidate": name,
                "filename": profile.get('filename', ''),
                "status": "VALIDATED",
                "original_score": profile.get('llm_score', 50),
                "corrected_score": profile.get('llm_score', 50),
                "anomalies": [],
                "corrections": [],
                "was_corrected": False,
                "llm_reflection": None,
                "confidence": "HIGH"
            }

        # Step 3: Apply heuristic corrections
        correction_result = self._apply_corrections(profile, anomalies)

        # Step 4: LLM-powered deep reflection (if available)
        llm_reflection = None
        llm_corrected_score = None

        if use_llm and self.llm_agent and anomalies:
            llm_reflection = self._generate_llm_reflection(
                profile, anomalies, job_description
            )

            # Extract LLM's corrected score
            if llm_reflection:
                score_match = re.search(r'[Cc]orrected\s*[Ss]core[:\s]*(\d{1,3})', llm_reflection)
                if score_match:
                    llm_corrected_score = min(100, int(score_match.group(1)))

        # Step 5: Determine final corrected score
        # If LLM gave a reflection, average the heuristic and LLM corrections
        if llm_corrected_score is not None:
            final_score = round((correction_result['corrected_score'] + llm_corrected_score) / 2)
        else:
            final_score = correction_result['corrected_score']

        status = "CORRECTED" if correction_result['was_corrected'] else "VALIDATED"
        severity_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        max_severity = max((severity_map.get(a['severity'], 1) for a in anomalies), default=1)
        confidence = {3: "LOW", 2: "MEDIUM", 1: "HIGH"}.get(max_severity, "MEDIUM")

        self.log(
            f"   {'⚠️' if status == 'CORRECTED' else '✅'} {name}: "
            f"{correction_result['original_score']} → {final_score} "
            f"({len(anomalies)} anomalies, confidence: {confidence})"
        )

        return {
            "candidate": name,
            "filename": profile.get('filename', ''),
            "status": status,
            "original_score": correction_result['original_score'],
            "corrected_score": final_score,
            "total_adjustment": correction_result['original_score'] - final_score,
            "anomalies": anomalies,
            "corrections": correction_result['corrections'],
            "was_corrected": correction_result['was_corrected'],
            "llm_reflection": llm_reflection,
            "llm_corrected_score": llm_corrected_score,
            "confidence": confidence
        }

    def reflect_batch(self, profiles, job_description, use_llm=True):
        """
        Run reflection on a batch of candidate profiles.

        Args:
            profiles: list of candidate profile dicts
            job_description: the JD text
            use_llm: whether to use LLM for reflection

        Returns:
            list of reflection results, one per candidate
        """
        self.reflection_logs = []
        results = []

        for profile in profiles:
            result = self.reflect(profile, job_description, use_llm=use_llm)
            results.append(result)

        # Summary stats
        corrected_count = sum(1 for r in results if r['was_corrected'])
        validated_count = len(results) - corrected_count

        self.log(f"\n📊 Reflection Summary: {corrected_count} corrected, {validated_count} validated out of {len(results)} candidates")

        return results
