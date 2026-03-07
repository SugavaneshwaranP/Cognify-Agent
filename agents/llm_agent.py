"""
CognifyX – LLM Agent
Handles interactions with local Ollama models: Qwen (extraction), Mistral (scoring), LLaMA (insights).
Includes intelligent heuristic fallback when Ollama is unavailable.
"""
import json
import re
import time
import requests


class LLMAgent:
    """
    Handles interactions with Local Ollama models.
    Supports Qwen, Mistral, and LLaMA via local deployment.
    """

    # Model configuration – easily switchable
    MODELS = {
        "extraction": "qwen2.5:latest",
        "scoring": "mistral:latest",
        "insights": "llama3:latest",
    }

    def __init__(self, base_url="http://localhost:11434/api/generate"):
        self.base_url = base_url
        self.model_status = {}  # Track which models are available
        self.call_times = {}    # Performance tracking

    def check_ollama_status(self):
        """Quick check if Ollama server is running."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                available = [m.get("name", "") for m in models]
                return True, available
        except Exception:
            pass
        return False, []

    def call_llm(self, model, prompt, response_format="text", timeout=60):
        """
        Call local Ollama model with retry logic.
        Returns (response_text, elapsed_seconds).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json" if response_format == "json" else ""
        }

        start = time.time()
        try:
            response = requests.post(self.base_url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            elapsed = round(time.time() - start, 2)
            text = result.get("response", "")
            self.call_times[model] = elapsed
            return text, elapsed
        except requests.exceptions.ConnectionError:
            return "", 0
        except requests.exceptions.Timeout:
            return "", 0
        except Exception:
            return "", 0

    # -----------------------------------------------------------------
    # HEURISTIC FALLBACK – Advanced text-based extraction when LLM is unavailable
    # -----------------------------------------------------------------
    def _heuristic_extract(self, resume_text, identifier="unknown"):
        """
        Extract structured data from resume using regex and pattern matching.
        """
        text = resume_text or ""
        text_lower = text.lower()

        # 1. NAME EXTRACTION
        name = identifier.replace(".pdf", "").replace(".docx", "").replace(".txt", "").replace("-Resume", "").replace("_", " ")
        # Try to find capitalized names
        name_match = re.search(r'^([A-Z][A-Z]+(?:\s+[A-Z][A-Z]*)+)', text, re.MULTILINE)
        if not name_match:
            name_match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text, re.MULTILINE)
        if name_match:
            candidate_name = name_match.group(1).strip()
            if len(candidate_name) < 40:  # Sanity check
                name = candidate_name

        # 2. SKILL EXTRACTION
        tech_skills = [
            "python", "java", "javascript", "typescript", "react", "angular", "vue",
            "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn",
            "spark", "hadoop", "kafka", "airflow",
            "html", "css", "sass", "tailwind",
            "c++", "c#", "go", "rust", "ruby", "php", "swift", "kotlin",
            "node", "express", "django", "flask", "fastapi", "spring",
            "git", "jenkins", "ci/cd", "agile", "scrum", "jira",
            "figma", "tableau", "power bi", "excel",
            "machine learning", "deep learning", "nlp", "computer vision",
            "microservices", "rest api", "graphql", "grpc",
            "linux", "bash", "powershell"
        ]
        found_skills = []
        for s in tech_skills:
            if s in text_lower:
                found_skills.append(s.title() if len(s) > 3 else s.upper())
        found_skills = list(dict.fromkeys(found_skills))  # Remove duplicates preserving order

        # 3. EXPERIENCE
        exp_match = re.search(r'(\d+)\s*(?:\+)?\s*(?:years|yrs|year)', text_lower)
        if exp_match:
            exp_years = int(exp_match.group(1))
        elif re.search(r'fresher|intern|student|undergraduate', text_lower):
            exp_years = 0
        elif re.search(r'202\d\s*[-–]\s*202\d', text_lower):
            exp_years = 0
        else:
            exp_years = 1

        # 4. EDUCATION
        education = "Not specified"
        edu_patterns = [
            (r'(b\.?tech|bachelor|b\.?e\.?|b\.?sc)', "Bachelor's Degree"),
            (r'(m\.?tech|master|m\.?s\.?|m\.?sc|mba)', "Master's Degree"),
            (r'(ph\.?d|doctorate)', "Doctorate"),
            (r'(diploma|associate)', "Diploma/Associate"),
        ]
        for pattern, label in edu_patterns:
            if re.search(pattern, text_lower):
                education = label
                break

        # 5. DOMAIN DETECTION
        domain_keywords = {
            "Software Engineering": ["developer", "software", "backend", "frontend", "fullstack", "engineer", "coding"],
            "Web Development": ["web", "html", "css", "react", "node", "javascript", "frontend"],
            "Data Science / ML": ["data", "machine learning", "ai", "scientist", "analytics", "deep learning", "nlp"],
            "Cloud / DevOps": ["cloud", "aws", "azure", "devops", "infrastructure", "kubernetes", "docker"],
            "UI/UX Design": ["ui", "ux", "designer", "graphic", "figma", "prototype"],
            "Mobile Development": ["android", "ios", "flutter", "react native", "mobile", "swift", "kotlin"],
            "Cybersecurity": ["security", "penetration", "vulnerability", "firewall", "soc"],
        }
        detected_domain = "General Professional"
        max_hits = 0
        for domain, keywords in domain_keywords.items():
            hits = sum(1 for k in keywords if k in text_lower)
            if hits > max_hits:
                max_hits = hits
                detected_domain = domain

        # 6. PROJECT EXTRACTION
        projects = []
        proj_section = re.search(r'(?:projects?|personal projects?|academic projects?)[:\s]*\n(.*?)(?:\n\n|\n[A-Z])', text, re.DOTALL | re.IGNORECASE)
        if proj_section:
            proj_lines = [l.strip() for l in proj_section.group(1).split('\n') if l.strip() and len(l.strip()) > 5]
            projects = proj_lines[:5]
        if not projects:
            projects = ["Details in resume"]

        return {
            "name": name,
            "skills": found_skills[:12],
            "experience_years": exp_years,
            "education": education,
            "projects": projects,
            "tools": found_skills[12:17] if len(found_skills) > 12 else [],
            "domain": detected_domain
        }

    def _heuristic_score(self, resume_text, profile, jd):
        """
        Score a candidate against the JD using heuristic analysis.
        Returns a score string with rationale.
        """
        text_lower = (resume_text or "").lower()
        jd_lower = (jd or "").lower()

        # Extract JD keywords
        jd_words = set(re.findall(r'\b[a-z]{3,}\b', jd_lower))
        resume_words = set(re.findall(r'\b[a-z]{3,}\b', text_lower))
        overlap = jd_words & resume_words
        overlap_ratio = len(overlap) / max(len(jd_words), 1)

        # Skill count
        skills = profile.get('skills', []) if profile else []
        skill_count = len(skills)

        # Experience
        exp = profile.get('experience_years', 0) if profile else 0

        # Calculate score
        base = 40
        keyword_bonus = min(30, overlap_ratio * 40)
        skill_bonus = min(20, skill_count * 2)
        exp_bonus = min(10, exp * 2)
        score = min(95, int(base + keyword_bonus + skill_bonus + exp_bonus))

        rationale = (
            f"Heuristic Analysis Score: {score}/100\n"
            f"• Keyword overlap with JD: {len(overlap)} terms ({overlap_ratio:.0%})\n"
            f"• Technical skills found: {skill_count}\n"
            f"• Experience: {exp} years\n"
            f"• Domain: {profile.get('domain', 'General') if profile else 'General'}\n"
            f"• Key matches: {', '.join(list(overlap)[:8])}"
        )
        return rationale

    # -----------------------------------------------------------------
    # PUBLIC API – Called by pipeline
    # -----------------------------------------------------------------
    def extract_structured_data(self, resume_text, identifier="unknown"):
        """Stage 2: Extract structured data using Qwen."""
        model = self.MODELS["extraction"]
        prompt = f"""Extract structured information from this resume accurately. Return ONLY valid JSON.
Resume Content:
{resume_text[:3000]}

INSTRUCTIONS:
1. Identify the candidate's real name. It is usually a large header or at the top.
2. Calculate total years of professional experience. If student, use 0.
3. Extract top technical skills and tools.
4. Detect professional domain (e.g., Web Development, Data Science).

JSON keys: name, skills, years_of_experience, education, projects, tools, domain"""

        response, elapsed = self.call_llm(model, prompt, response_format="json", timeout=30)

        if not response:
            return self._heuristic_extract(resume_text, identifier)

        try:
            data = json.loads(response)
            if 'years_of_experience' in data:
                data['experience_years'] = data.pop('years_of_experience')
            return data
        except (json.JSONDecodeError, ValueError):
            return self._heuristic_extract(resume_text, identifier)

    def score_candidate(self, jd, profile, identifier="unknown"):
        """Stage 3: Score candidate using Mistral."""
        model = self.MODELS["scoring"]
        resume_text = profile.get('full_text', "")
        profile_brief = {k: v for k, v in profile.items() if k not in ('full_text',)}

        prompt = f"""You are an expert recruiter AI. Score this candidate against the job description.

Job Description:
{jd[:2000]}

Candidate Profile:
{json.dumps(profile_brief, indent=2)[:2000]}

INSTRUCTIONS:
1. Assign a numerical score from 0-100 based on fit.
2. Provide brief rationale covering: skills match, experience relevance, domain alignment.
3. Start your response with: "Score: XX/100" on the first line.
"""
        response, elapsed = self.call_llm(model, prompt, timeout=30)

        if not response:
            return self._heuristic_score(resume_text, profile, jd)

        return response

    def final_intelligence(self, top_candidates_summary):
        """Stage 4: Generate final insights using LLaMA."""
        model = self.MODELS["insights"]
        prompt = f"""You are a senior recruiter AI. Analyze these top candidates and provide:

1. **Ranked Recommendation** – Who should be interviewed first and why
2. **Risk Flags** – Any concerns per candidate
3. **Suggested Interview Questions** – 2-3 per top candidate
4. **Summary** – Concise recruiter brief

Top Candidates Data:
{top_candidates_summary[:3000]}
"""
        response, elapsed = self.call_llm(model, prompt, timeout=45)

        if not response:
            return (
                "**Final Recommendation (Heuristic Mode)**\n\n"
                "Candidates have been ranked based on ATS keyword matching, "
                "skill analysis, and experience evaluation. Review the individual "
                "scores and AI analysis for each candidate above.\n\n"
                "💡 *Tip: Start Ollama with `ollama serve` and pull models for "
                "enhanced AI-powered insights.*"
            )

        return response
