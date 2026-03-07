"""
CognifyX – Experience Evaluator Agent
Assesses career trajectory, experience relevance, and professional growth.
Uses Mistral model for experience evaluation.
"""
import re
import json


class ExperienceAgent:
    """
    Specialized agent focused on evaluating professional experience.
    Assesses career trajectory, role relevance, and growth indicators.
    """

    # Seniority level mappings
    SENIORITY_MAP = {
        "intern": 0, "trainee": 0, "fresher": 0,
        "junior": 1, "entry": 1,
        "mid": 3, "intermediate": 3,
        "senior": 5, "lead": 6,
        "principal": 8, "staff": 8,
        "architect": 8, "director": 10,
        "vp": 12, "cto": 12, "ceo": 15,
    }

    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent

    def _extract_jd_experience_req(self, job_description):
        """Extract experience requirements from JD."""
        jd_lower = job_description.lower()

        # Min years
        exp_match = re.search(r'(\d+)\s*\+?\s*(?:years|yrs)', jd_lower)
        min_years = int(exp_match.group(1)) if exp_match else 0

        # Seniority
        seniority = "any"
        for level, years in sorted(self.SENIORITY_MAP.items(), key=lambda x: -x[1]):
            if level in jd_lower:
                seniority = level
                if min_years == 0:
                    min_years = years
                break

        return {"min_years": min_years, "seniority": seniority}

    def _heuristic_evaluate(self, profile, job_description):
        """Evaluate experience using heuristic analysis."""
        exp_years = profile.get('experience_years', 0) or 0
        jd_req = self._extract_jd_experience_req(job_description)
        min_years = jd_req['min_years']
        req_seniority = jd_req['seniority']

        # Experience score components
        # 1. Years match (0-40 points)
        if min_years > 0:
            if exp_years >= min_years:
                years_score = 40
            elif exp_years >= min_years * 0.7:
                years_score = 25
            elif exp_years >= min_years * 0.5:
                years_score = 15
            else:
                years_score = 5
        else:
            years_score = min(40, exp_years * 5)

        # 2. Seniority match (0-25 points)
        if req_seniority != "any":
            req_min = self.SENIORITY_MAP.get(req_seniority, 3)
            if exp_years >= req_min:
                seniority_score = 25
            elif exp_years >= req_min * 0.6:
                seniority_score = 15
            else:
                seniority_score = 5
        else:
            seniority_score = 15

        # 3. Domain relevance (0-20 points)
        domain = (profile.get('domain', '') or '').lower()
        jd_lower = job_description.lower()
        domain_keywords = domain.split('/') + domain.split() if domain else []
        domain_match = sum(1 for kw in domain_keywords if kw.strip() in jd_lower)
        domain_score = min(20, domain_match * 10) if domain_keywords else 10

        # 4. Projects & growth indicators (0-15 points)
        projects = profile.get('projects', [])
        project_score = min(15, len(projects) * 3) if projects else 5

        score = min(95, max(5, years_score + seniority_score + domain_score + project_score))

        # Experience gap assessment
        gap = ""
        if min_years > 0 and exp_years < min_years:
            gap = f"⚠️ Experience gap: requires {min_years}+ years, has {exp_years}"
        elif min_years > 0 and exp_years >= min_years:
            gap = f"✅ Meets experience requirement ({exp_years} ≥ {min_years} years)"

        argument = (
            f"**Experience Evaluation Score: {score}/100**\n"
            f"• Experience: {exp_years} years (JD requires: {min_years}+)\n"
            f"• Target Seniority: {req_seniority} | Candidate Level: "
            f"{'Senior' if exp_years >= 5 else 'Mid' if exp_years >= 2 else 'Junior'}\n"
            f"• Domain: {profile.get('domain', 'General')}\n"
            f"• Projects: {len(projects)} listed\n"
            f"• {gap}"
        )

        return {
            "score": score,
            "argument": argument,
            "experience_years": exp_years,
            "required_years": min_years,
            "seniority_match": exp_years >= self.SENIORITY_MAP.get(req_seniority, 0),
            "experience_gap": gap,
            "agent": "Experience Evaluator",
            "model": "Mistral"
        }

    def evaluate(self, profile, job_description):
        """Evaluate a candidate's professional experience."""
        heuristic = self._heuristic_evaluate(profile, job_description)

        if not self.llm_agent:
            return heuristic

        profile_brief = {k: v for k, v in profile.items() if k not in ('full_text', 'llm_analysis')}
        prompt = f"""You are a Career Experience Evaluator. Assess this candidate's experience against the job requirements.

Job Description:
{job_description[:1500]}

Candidate Profile:
{json.dumps(profile_brief, indent=2, default=str)[:1500]}

EVALUATE:
1. Does the candidate meet the minimum experience requirement?
2. Is their career trajectory relevant to this role?
3. Do their projects demonstrate applicable experience?
4. Are there any career growth red flags?

Start with "Experience Score: XX/100" on the first line.
Then provide 2-3 bullet points of analysis."""

        response, elapsed = self.llm_agent.call_llm(
            self.llm_agent.MODELS.get("scoring", "mistral:latest"),
            prompt, timeout=25
        )

        if response:
            score_match = re.search(r'[Ee]xperience\s*[Ss]core[:\s]*(\d{1,3})', response)
            if score_match:
                llm_score = min(100, int(score_match.group(1)))
                final_score = int(llm_score * 0.6 + heuristic['score'] * 0.4)
                heuristic['score'] = final_score
                heuristic['argument'] = response
                heuristic['llm_enhanced'] = True

        return heuristic
