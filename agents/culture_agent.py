"""
CognifyX – Culture Fit Assessor Agent
Evaluates soft skills, communication quality, and team alignment potential.
Uses LLaMA model for holistic cultural assessment.
"""
import re
import json


class CultureAgent:
    """
    Specialized agent focused on evaluating soft skills and cultural fit.
    Assesses communication quality, teamwork indicators, and alignment potential.
    """

    # Soft skill indicators in resume text
    SOFT_SKILL_INDICATORS = {
        "leadership": ["led", "managed", "supervised", "directed", "mentored", "coordinated", "headed"],
        "teamwork": ["collaborated", "team", "cross-functional", "partnered", "contributed", "worked with"],
        "communication": ["presented", "communicated", "wrote", "documented", "reported", "stakeholder"],
        "problem_solving": ["solved", "resolved", "debugged", "optimized", "improved", "reduced", "increased"],
        "initiative": ["initiated", "launched", "proposed", "created", "built", "designed", "pioneered"],
        "adaptability": ["adapted", "learned", "transitioned", "migrated", "upgraded", "diverse"],
        "ownership": ["owned", "responsible", "accountability", "end-to-end", "full-stack", "independently"],
    }

    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent

    def _heuristic_evaluate(self, profile, job_description):
        """Evaluate culture fit using text analysis of resume."""
        resume_text = (profile.get('full_text', '') or '').lower()

        # Detect soft skill indicators
        detected_skills = {}
        total_indicators = 0
        for skill, keywords in self.SOFT_SKILL_INDICATORS.items():
            count = sum(1 for kw in keywords if kw in resume_text)
            if count > 0:
                detected_skills[skill] = count
                total_indicators += count

        # Education quality (proxy for communication/rigor)
        education = (profile.get('education', '') or '').lower()
        edu_score = 0
        if 'doctorate' in education or 'phd' in education:
            edu_score = 15
        elif 'master' in education:
            edu_score = 12
        elif 'bachelor' in education:
            edu_score = 8
        else:
            edu_score = 3

        # Projects (proxy for initiative and ownership)
        projects = profile.get('projects', [])
        project_score = min(15, len(projects) * 4)

        # Communication quality - resume length and structure (well-written resumes are longer and structured)
        text_length = len(resume_text)
        structure_score = 0
        if text_length > 2000:
            structure_score = 15
        elif text_length > 1000:
            structure_score = 10
        elif text_length > 500:
            structure_score = 5

        # Soft skills breadth
        soft_breadth = min(25, len(detected_skills) * 5)

        # Action-oriented language (shows proactive culture)
        action_score = min(15, total_indicators * 2)

        score = min(95, max(10, edu_score + project_score + structure_score + soft_breadth + action_score))

        top_skills = sorted(detected_skills.items(), key=lambda x: -x[1])[:4]
        top_skill_text = ", ".join(f"{s[0].title()} ({s[1]})" for s in top_skills) or "Limited indicators"

        argument = (
            f"**Culture Fit Assessment Score: {score}/100**\n"
            f"• Soft Skills Detected: {len(detected_skills)}/7 categories\n"
            f"• Top Strengths: {top_skill_text}\n"
            f"• Education: {profile.get('education', 'Not specified')}\n"
            f"• Projects/Initiative: {len(projects)} projects\n"
            f"• Communication Indicators: {'Strong' if total_indicators >= 8 else 'Moderate' if total_indicators >= 4 else 'Limited'}"
        )

        return {
            "score": score,
            "argument": argument,
            "soft_skills": detected_skills,
            "top_strengths": [s[0] for s in top_skills],
            "agent": "Culture Fit Assessor",
            "model": "LLaMA"
        }

    def evaluate(self, profile, job_description):
        """Evaluate a candidate's culture fit and soft skills."""
        heuristic = self._heuristic_evaluate(profile, job_description)

        if not self.llm_agent:
            return heuristic

        profile_brief = {k: v for k, v in profile.items() if k not in ('full_text', 'llm_analysis')}
        prompt = f"""You are a Culture Fit and Soft Skills Assessor. Evaluate this candidate's potential cultural alignment.

Job Description:
{job_description[:1500]}

Candidate Profile:
{json.dumps(profile_brief, indent=2, default=str)[:1500]}

EVALUATE:
1. Does the candidate show leadership and teamwork indicators?
2. How strong are their communication and documentation skills?
3. Do they demonstrate initiative and problem-solving ability?
4. Would they likely fit in a collaborative engineering team?

Start with "Culture Score: XX/100" on the first line.
Then provide 2-3 bullet points of analysis."""

        response, elapsed = self.llm_agent.call_llm(
            self.llm_agent.MODELS.get("insights", "llama3:latest"),
            prompt, timeout=25
        )

        if response:
            score_match = re.search(r'[Cc]ulture\s*[Ss]core[:\s]*(\d{1,3})', response)
            if score_match:
                llm_score = min(100, int(score_match.group(1)))
                final_score = int(llm_score * 0.6 + heuristic['score'] * 0.4)
                heuristic['score'] = final_score
                heuristic['argument'] = response
                heuristic['llm_enhanced'] = True

        return heuristic
