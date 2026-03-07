"""
CognifyX – Report Agent
Generates structured hiring report content including:
  - Executive Summary
  - Candidate Comparison Matrix
  - Skill Gap Analysis
  - Interview Question Bank (personalized per candidate)
  - Diversity & Inclusion Metrics
  - Salary Benchmarking
"""
import re
import json
from datetime import datetime


class ReportAgent:
    """
    Generates rich, structured hiring report data from pipeline results.
    Content is consumed by ReportGenerator to produce PDF/DOCX files.
    """

    # ----- approximate salary bands (USD, annual) by domain + experience -----
    SALARY_BANDS = {
        "software": {
            "junior": (60000, 90000), "mid": (90000, 130000),
            "senior": (130000, 180000), "lead": (170000, 220000),
        },
        "data": {
            "junior": (65000, 95000), "mid": (95000, 140000),
            "senior": (140000, 190000), "lead": (180000, 240000),
        },
        "devops": {
            "junior": (70000, 100000), "mid": (100000, 145000),
            "senior": (145000, 195000), "lead": (185000, 250000),
        },
        "default": {
            "junior": (55000, 85000), "mid": (85000, 120000),
            "senior": (120000, 170000), "lead": (160000, 210000),
        },
    }

    # ----- behavioural / technical question templates -----
    QUESTION_TEMPLATES = {
        "skills_gap": "You lack experience with {skill}. Can you describe how you'd approach learning it?",
        "experience_match": "Tell me about a project where you applied {skill} in a production environment.",
        "leadership": "Describe a time you led a team through a challenging deadline.",
        "problem_solving": "Walk me through how you debugged a complex production issue.",
        "culture_fit": "How do you handle disagreements with teammates?",
        "career_growth": "Where do you see yourself in 3 years, and how does this role fit?",
        "domain_depth": "Explain a recent advancement in {domain} that excites you.",
    }

    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent

    # ══════════════════════════════════════════════════════════════════
    # 1. EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════
    def generate_executive_summary(self, results, job_description=""):
        """One-paragraph executive summary of the entire screening."""
        shortlisted = results.get('shortlisted', [])
        total = results.get('total_parsed', 0)
        filtered = results.get('total_filtered', 0)
        n_short = len(shortlisted)

        if shortlisted:
            top = shortlisted[0]
            top_name = top.get('name', 'Unknown')
            top_score = top.get('composite_score', 0)
            avg_score = round(sum(c.get('composite_score', 0) for c in shortlisted) / max(n_short, 1), 1)
        else:
            top_name, top_score, avg_score = "N/A", 0, 0

        plan = results.get('plan')
        strategy = plan.strategy_label if plan else "Standard"

        summary = (
            f"CognifyX screened {total} resumes using the {strategy} strategy, "
            f"filtering to {filtered} via ATS scoring and shortlisting the top {n_short} candidates. "
            f"The highest-ranked candidate is **{top_name}** with a composite score of "
            f"**{top_score}/100**. The shortlisted average is **{avg_score}/100**."
        )

        insights = results.get('insights', '')
        if insights:
            # Take first sentence of LLM insights
            first_sentence = insights.split('.')[0] + '.' if '.' in insights else insights[:200]
            summary += f" AI assessment: {first_sentence}"

        return {
            "title": "Executive Summary",
            "content": summary,
            "stats": {
                "total_screened": total,
                "ats_filtered": filtered,
                "shortlisted": n_short,
                "top_candidate": top_name,
                "top_score": top_score,
                "avg_score": avg_score,
                "strategy": strategy,
            },
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    # ══════════════════════════════════════════════════════════════════
    # 2. CANDIDATE COMPARISON MATRIX
    # ══════════════════════════════════════════════════════════════════
    def generate_comparison_matrix(self, results):
        """Side-by-side comparison of all shortlisted candidates."""
        shortlisted = results.get('shortlisted', [])
        matrix = []
        for c in shortlisted:
            debate = c.get('debate', {})
            matrix.append({
                "rank": c.get('final_rank', '?'),
                "name": c.get('name', c.get('filename', 'Unknown')),
                "domain": c.get('domain', 'General'),
                "experience_years": c.get('experience_years', 0) or 0,
                "education": c.get('education', 'Not specified'),
                "skills": c.get('skills', [])[:8],
                "ats_score": c.get('ats_score', 0),
                "keyword_score": c.get('keyword_score', 0),
                "llm_score": c.get('llm_score', 0),
                "composite_score": c.get('composite_score', 0),
                "skill_debate_score": debate.get('skill_score', '-'),
                "exp_debate_score": debate.get('experience_score', '-'),
                "culture_debate_score": debate.get('culture_score', '-'),
                "consensus_score": debate.get('consensus_score', '-'),
                "keywords_matched": c.get('keywords_matched', []),
                "keywords_missed": c.get('keywords_missed', []),
            })

        # Column headers
        headers = [
            "Rank", "Name", "Domain", "Exp (yrs)", "Education",
            "ATS", "Keyword %", "LLM Score", "Composite",
            "Skill Debate", "Exp Debate", "Culture Debate", "Consensus",
        ]

        return {
            "title": "Candidate Comparison Matrix",
            "headers": headers,
            "rows": matrix,
            "count": len(matrix),
        }

    # ══════════════════════════════════════════════════════════════════
    # 3. SKILL GAP ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    def generate_skill_gap_analysis(self, results):
        """Analyse what skills are missing from the entire talent pool."""
        shortlisted = results.get('shortlisted', [])
        keywords = results.get('keywords', [])

        # Aggregate all matched / missed keywords
        all_matched = {}
        all_missed = {}
        all_skills = {}

        for c in shortlisted:
            for kw in c.get('keywords_matched', []):
                all_matched[kw.lower()] = all_matched.get(kw.lower(), 0) + 1
            for kw in c.get('keywords_missed', []):
                all_missed[kw.lower()] = all_missed.get(kw.lower(), 0) + 1
            for sk in c.get('skills', []):
                all_skills[sk.lower()] = all_skills.get(sk.lower(), 0) + 1

        n = max(len(shortlisted), 1)

        # Skills coverage
        skill_coverage = []
        for kw in keywords:
            kw_lower = kw.strip().lower()
            matched_count = all_matched.get(kw_lower, 0)
            coverage_pct = round((matched_count / n) * 100)
            skill_coverage.append({
                "keyword": kw.strip(),
                "candidates_with": matched_count,
                "coverage_pct": coverage_pct,
                "status": "Strong" if coverage_pct >= 70 else "Moderate" if coverage_pct >= 40 else "Gap",
            })

        # Top missing skills (most frequently missed)
        top_gaps = sorted(all_missed.items(), key=lambda x: -x[1])[:10]

        # Most common skills across pool
        top_skills = sorted(all_skills.items(), key=lambda x: -x[1])[:15]

        return {
            "title": "Skill Gap Analysis",
            "coverage": skill_coverage,
            "top_gaps": [{"skill": s, "missed_by": c} for s, c in top_gaps],
            "top_skills_in_pool": [{"skill": s, "count": c} for s, c in top_skills],
            "total_keywords": len(keywords),
            "strong_coverage": len([s for s in skill_coverage if s["status"] == "Strong"]),
            "gaps_found": len([s for s in skill_coverage if s["status"] == "Gap"]),
        }

    # ══════════════════════════════════════════════════════════════════
    # 4. INTERVIEW QUESTION BANK
    # ══════════════════════════════════════════════════════════════════
    def generate_interview_questions(self, results, job_description=""):
        """Personalised interview questions per candidate."""
        shortlisted = results.get('shortlisted', [])
        question_bank = []

        for c in shortlisted:
            name = c.get('name', 'Unknown')
            skills = c.get('skills', [])
            missed = c.get('keywords_missed', [])
            domain = c.get('domain', 'Software')
            exp_years = c.get('experience_years', 0) or 0

            questions = []

            # Gap-based questions
            for gap_skill in missed[:3]:
                questions.append({
                    "type": "Skill Gap",
                    "question": self.QUESTION_TEMPLATES["skills_gap"].format(skill=gap_skill),
                    "reason": f"Missing required skill: {gap_skill}",
                })

            # Strength-based questions
            for strong_skill in skills[:2]:
                questions.append({
                    "type": "Technical Depth",
                    "question": self.QUESTION_TEMPLATES["experience_match"].format(skill=strong_skill),
                    "reason": f"Verify depth in claimed skill: {strong_skill}",
                })

            # Experience-appropriate questions
            if exp_years >= 5:
                questions.append({
                    "type": "Leadership",
                    "question": self.QUESTION_TEMPLATES["leadership"],
                    "reason": f"Senior-level candidate ({exp_years} years)",
                })
            else:
                questions.append({
                    "type": "Growth Potential",
                    "question": self.QUESTION_TEMPLATES["career_growth"],
                    "reason": f"Junior/mid candidate ({exp_years} years)",
                })

            # Universal questions
            questions.append({
                "type": "Problem Solving",
                "question": self.QUESTION_TEMPLATES["problem_solving"],
                "reason": "Assess analytical thinking",
            })
            questions.append({
                "type": "Culture Fit",
                "question": self.QUESTION_TEMPLATES["culture_fit"],
                "reason": "Team compatibility assessment",
            })

            # Domain question
            if domain:
                questions.append({
                    "type": "Domain Knowledge",
                    "question": self.QUESTION_TEMPLATES["domain_depth"].format(domain=domain),
                    "reason": f"Domain expertise: {domain}",
                })

            question_bank.append({
                "candidate": name,
                "rank": c.get('final_rank', '?'),
                "questions": questions,
            })

        return {
            "title": "Interview Question Bank",
            "candidates": question_bank,
            "total_questions": sum(len(cb["questions"]) for cb in question_bank),
        }

    # ══════════════════════════════════════════════════════════════════
    # 5. DIVERSITY & INCLUSION METRICS
    # ══════════════════════════════════════════════════════════════════
    def generate_diversity_metrics(self, results):
        """Analyse diversity indicators across the candidate pool."""
        shortlisted = results.get('shortlisted', [])
        all_candidates = results.get('candidates', shortlisted)

        # Domain diversity
        domains = {}
        for c in all_candidates:
            d = (c.get('domain', '') or 'General').strip()
            domains[d] = domains.get(d, 0) + 1

        # Education level distribution
        edu_levels = {"Doctorate/PhD": 0, "Master's": 0, "Bachelor's": 0, "Other": 0}
        for c in all_candidates:
            edu = (c.get('education', '') or '').lower()
            if 'phd' in edu or 'doctorate' in edu:
                edu_levels["Doctorate/PhD"] += 1
            elif 'master' in edu or 'm.sc' in edu or 'm.tech' in edu or "m.s" in edu:
                edu_levels["Master's"] += 1
            elif 'bachelor' in edu or 'b.sc' in edu or 'b.tech' in edu or 'b.s' in edu or 'b.e' in edu:
                edu_levels["Bachelor's"] += 1
            else:
                edu_levels["Other"] += 1

        # Experience distribution
        exp_buckets = {"0-2 yrs": 0, "3-5 yrs": 0, "6-10 yrs": 0, "10+ yrs": 0}
        for c in all_candidates:
            yrs = c.get('experience_years', 0) or 0
            if yrs <= 2:
                exp_buckets["0-2 yrs"] += 1
            elif yrs <= 5:
                exp_buckets["3-5 yrs"] += 1
            elif yrs <= 10:
                exp_buckets["6-10 yrs"] += 1
            else:
                exp_buckets["10+ yrs"] += 1

        # Score distribution for shortlisted
        if shortlisted:
            scores = [c.get('composite_score', 0) for c in shortlisted]
            score_stats = {
                "min": round(min(scores), 1),
                "max": round(max(scores), 1),
                "avg": round(sum(scores) / len(scores), 1),
                "spread": round(max(scores) - min(scores), 1),
            }
        else:
            score_stats = {"min": 0, "max": 0, "avg": 0, "spread": 0}

        # Domain diversity index (simple: unique domains / total)
        unique_domains = len(domains)
        total_candidates = len(all_candidates)
        diversity_index = round((unique_domains / max(total_candidates, 1)) * 100, 1)

        return {
            "title": "Diversity & Inclusion Metrics",
            "domain_distribution": domains,
            "education_distribution": edu_levels,
            "experience_distribution": exp_buckets,
            "score_statistics": score_stats,
            "diversity_index": diversity_index,
            "unique_domains": unique_domains,
            "total_pool": total_candidates,
            "shortlisted_count": len(shortlisted),
        }

    # ══════════════════════════════════════════════════════════════════
    # 6. SALARY BENCHMARKING
    # ══════════════════════════════════════════════════════════════════
    def generate_salary_benchmarks(self, results):
        """Estimate salary ranges based on domain + experience level."""
        shortlisted = results.get('shortlisted', [])
        benchmarks = []

        for c in shortlisted:
            domain_raw = (c.get('domain', '') or '').lower()
            exp_years = c.get('experience_years', 0) or 0

            # Map domain to salary category
            domain_key = "default"
            if any(kw in domain_raw for kw in ["software", "web", "full-stack", "backend", "frontend"]):
                domain_key = "software"
            elif any(kw in domain_raw for kw in ["data", "machine learning", "ai", "analytics"]):
                domain_key = "data"
            elif any(kw in domain_raw for kw in ["devops", "cloud", "infrastructure", "sre"]):
                domain_key = "devops"

            # Map experience to seniority
            if exp_years <= 2:
                level = "junior"
            elif exp_years <= 5:
                level = "mid"
            elif exp_years <= 8:
                level = "senior"
            else:
                level = "lead"

            band = self.SALARY_BANDS.get(domain_key, self.SALARY_BANDS["default"]).get(level, (60000, 120000))
            midpoint = (band[0] + band[1]) // 2

            benchmarks.append({
                "rank": c.get('final_rank', '?'),
                "name": c.get('name', 'Unknown'),
                "domain": c.get('domain', 'General'),
                "experience": exp_years,
                "level": level.title(),
                "salary_min": f"${band[0]:,}",
                "salary_max": f"${band[1]:,}",
                "midpoint": f"${midpoint:,}",
                "composite_score": c.get('composite_score', 0),
            })

        return {
            "title": "Salary Benchmarking",
            "benchmarks": benchmarks,
            "note": "Estimates based on 2024 US market data for similar roles.",
        }

    # ══════════════════════════════════════════════════════════════════
    # FULL REPORT
    # ══════════════════════════════════════════════════════════════════
    def generate_full_report(self, results, job_description=""):
        """Generate all report sections."""
        return {
            "executive_summary": self.generate_executive_summary(results, job_description),
            "comparison_matrix": self.generate_comparison_matrix(results),
            "skill_gap_analysis": self.generate_skill_gap_analysis(results),
            "interview_questions": self.generate_interview_questions(results, job_description),
            "diversity_metrics": self.generate_diversity_metrics(results),
            "salary_benchmarks": self.generate_salary_benchmarks(results),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report_title": "CognifyX Hiring Intelligence Report",
        }
