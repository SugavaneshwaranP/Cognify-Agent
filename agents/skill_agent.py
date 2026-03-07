"""
CognifyX – Skill Analyst Agent
Evaluates technical skills depth and breadth against job requirements.
Uses Qwen model for detailed technical assessment.
"""
import re
import json


class SkillAgent:
    """
    Specialized agent focused on evaluating technical skills.
    Assesses skill depth, breadth, relevance, and recency.
    """

    # Skill categories for structured evaluation
    SKILL_CATEGORIES = {
        "languages": ["python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "r"],
        "frontend": ["react", "angular", "vue", "html", "css", "sass", "tailwind", "next.js", "svelte"],
        "backend": ["node", "express", "django", "flask", "fastapi", "spring", "rails", ".net"],
        "data": ["sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "cassandra"],
        "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible"],
        "ai_ml": ["machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn"],
        "devops": ["ci/cd", "jenkins", "github actions", "gitlab", "docker", "kubernetes", "monitoring"],
        "tools": ["git", "jira", "figma", "tableau", "power bi", "excel", "postman"],
    }

    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent

    def _extract_jd_skills(self, job_description):
        """Extract required skills from the job description."""
        jd_lower = job_description.lower()
        required = []
        for category, skills in self.SKILL_CATEGORIES.items():
            for skill in skills:
                if skill in jd_lower:
                    required.append({"skill": skill, "category": category})
        return required

    def _heuristic_evaluate(self, profile, job_description):
        """Evaluate skills using heuristic analysis."""
        candidate_skills = [s.lower() for s in profile.get('skills', [])]
        jd_skills = self._extract_jd_skills(job_description)
        jd_skill_names = [s['skill'] for s in jd_skills]

        # Skill match ratio
        matched = [s for s in jd_skill_names if s in candidate_skills or any(s in cs for cs in candidate_skills)]
        match_ratio = len(matched) / max(len(jd_skill_names), 1)

        # Breadth: how many skill categories does the candidate cover?
        categories_covered = set()
        for skill in candidate_skills:
            for cat, cat_skills in self.SKILL_CATEGORIES.items():
                if skill in cat_skills or any(skill in cs for cs in cat_skills):
                    categories_covered.add(cat)

        breadth_score = min(100, len(categories_covered) * 15)

        # Depth: total number of skills
        depth_score = min(100, len(candidate_skills) * 8)

        # Relevance: JD match weight
        relevance_score = match_ratio * 100

        # Composite skill score
        score = int(relevance_score * 0.50 + depth_score * 0.25 + breadth_score * 0.25)
        score = min(95, max(5, score))

        argument = (
            f"**Skill Analysis Score: {score}/100**\n"
            f"• JD Skill Match: {len(matched)}/{len(jd_skill_names)} required skills ({match_ratio:.0%})\n"
            f"• Matched: {', '.join(matched[:6]) or 'None'}\n"
            f"• Missing: {', '.join(set(jd_skill_names) - set(matched))[:80] or 'None'}\n"
            f"• Skill Breadth: {len(categories_covered)} categories ({', '.join(categories_covered)})\n"
            f"• Total Skills: {len(candidate_skills)}"
        )

        return {
            "score": score,
            "argument": argument,
            "matched_skills": matched,
            "missing_skills": list(set(jd_skill_names) - set(matched)),
            "categories_covered": list(categories_covered),
            "agent": "Skill Analyst",
            "model": "Qwen"
        }

    def evaluate(self, profile, job_description):
        """
        Evaluate a candidate's technical skills.
        Uses LLM when available, heuristic fallback otherwise.
        """
        # Always run heuristic for factual grounding
        heuristic = self._heuristic_evaluate(profile, job_description)

        if not self.llm_agent:
            return heuristic

        # Try LLM-powered deep analysis
        profile_brief = {k: v for k, v in profile.items() if k not in ('full_text', 'llm_analysis')}
        prompt = f"""You are a Technical Skills Analyst. Evaluate this candidate's technical skills against the job requirements.

Job Description:
{job_description[:1500]}

Candidate Profile:
{json.dumps(profile_brief, indent=2, default=str)[:1500]}

EVALUATE:
1. How well do the candidate's skills match the JD requirements? (skill relevance)
2. How deep is their expertise? (skill depth)
3. How broad is their skill set? (skill breadth)
4. Are there critical skill gaps?

Start with "Skill Score: XX/100" on the first line.
Then provide 2-3 bullet points of analysis.
End with your verdict: STRONG MATCH, MODERATE MATCH, or WEAK MATCH."""

        response, elapsed = self.llm_agent.call_llm(
            self.llm_agent.MODELS.get("extraction", "qwen2.5:latest"),
            prompt, timeout=25
        )

        if response:
            # Extract LLM score
            score_match = re.search(r'[Ss](?:kill)?\s*[Ss]core[:\s]*(\d{1,3})', response)
            if score_match:
                llm_score = min(100, int(score_match.group(1)))
                # Blend with heuristic
                final_score = int(llm_score * 0.6 + heuristic['score'] * 0.4)
                heuristic['score'] = final_score
                heuristic['argument'] = response
                heuristic['llm_enhanced'] = True

        return heuristic
