"""
CognifyX – Multi-Agent Debate Moderator
Orchestrates structured debates between specialized agents (Skill, Experience, Culture).
Includes a Devil's Advocate that challenges other agents' conclusions.
Synthesizes debate into a consensus score with full transcript.
"""
import re
import json


class DebateModerator:
    """
    Moderates multi-agent debates for candidate evaluation.
    
    Debate Flow:
    1. Each specialist agent presents their evaluation
    2. Devil's Advocate challenges the evaluations
    3. Agents can respond to challenges
    4. Moderator synthesizes a consensus score
    """

    # Debate weight configuration
    WEIGHTS = {
        "skill": 0.40,       # Technical skills matter most
        "experience": 0.35,  # Experience is very important
        "culture": 0.25,     # Culture fit is a tiebreaker
    }

    def __init__(self, skill_agent, experience_agent, culture_agent, llm_agent=None):
        self.skill_agent = skill_agent
        self.experience_agent = experience_agent
        self.culture_agent = culture_agent
        self.llm_agent = llm_agent

    def _devils_advocate(self, evaluations, profile, job_description):
        """
        Devil's Advocate: challenges the other agents' conclusions.
        Looks for inconsistencies and overconfidence.
        """
        challenges = []
        scores = {e['agent']: e['score'] for e in evaluations}

        # Challenge 1: Score spread too tight (groupthink)
        score_values = list(scores.values())
        spread = max(score_values) - min(score_values)
        if spread < 10 and len(score_values) >= 3:
            challenges.append({
                "type": "GROUPTHINK_WARNING",
                "severity": "MEDIUM",
                "challenge": (
                    f"All agents scored within {spread} points of each other "
                    f"({', '.join(f'{k}: {v}' for k, v in scores.items())}). "
                    f"This suspicious agreement may indicate shallow analysis."
                )
            })

        # Challenge 2: High skill score but missing key JD requirements
        skill_eval = next((e for e in evaluations if e['agent'] == 'Skill Analyst'), None)
        if skill_eval and skill_eval['score'] >= 70:
            missing = skill_eval.get('missing_skills', [])
            if len(missing) >= 3:
                challenges.append({
                    "type": "SKILLS_GAP_IGNORED",
                    "severity": "HIGH",
                    "challenge": (
                        f"Skill Analyst gave {skill_eval['score']}/100 but candidate is "
                        f"missing {len(missing)} required skills: {', '.join(missing[:4])}. "
                        f"The score may be inflated."
                    )
                })

        # Challenge 3: Experience mismatch not reflected in score
        exp_eval = next((e for e in evaluations if e['agent'] == 'Experience Evaluator'), None)
        if exp_eval:
            exp_years = exp_eval.get('experience_years', 0)
            req_years = exp_eval.get('required_years', 0)
            if req_years > 0 and exp_years < req_years and exp_eval['score'] > 60:
                challenges.append({
                    "type": "EXPERIENCE_OVERSCORED",
                    "severity": "HIGH",
                    "challenge": (
                        f"Experience Evaluator gave {exp_eval['score']}/100 but candidate "
                        f"has {exp_years} years vs. required {req_years}+. "
                        f"A {req_years - exp_years}-year gap shouldn't yield a passing score."
                    )
                })

        # Challenge 4: High culture score with low skills (style over substance)
        culture_eval = next((e for e in evaluations if e['agent'] == 'Culture Fit Assessor'), None)
        if culture_eval and skill_eval:
            if culture_eval['score'] > skill_eval['score'] + 20:
                challenges.append({
                    "type": "STYLE_OVER_SUBSTANCE",
                    "severity": "MEDIUM",
                    "challenge": (
                        f"Culture score ({culture_eval['score']}) is significantly higher "
                        f"than Skill score ({skill_eval['score']}). For a technical role, "
                        f"soft skills shouldn't compensate for skill gaps."
                    )
                })

        # Challenge 5: Very low experience but high other scores
        if exp_eval and exp_eval.get('experience_years', 0) <= 1:
            avg_other = sum(e['score'] for e in evaluations if e['agent'] != 'Experience Evaluator')
            avg_other = avg_other / max(1, len(evaluations) - 1)
            if avg_other > 70:
                challenges.append({
                    "type": "FRESHSER_OVERESTIMATE",
                    "severity": "MEDIUM",
                    "challenge": (
                        f"Candidate has ≤1 year experience but other agents averaged "
                        f"{avg_other:.0f}/100. Fresh candidates carry higher risk "
                        f"regardless of academic credentials."
                    )
                })

        # Generate Devil's Advocate summary
        if not challenges:
            challenges.append({
                "type": "NO_OBJECTIONS",
                "severity": "LOW",
                "challenge": "No significant inconsistencies found. The evaluations appear reasonable."
            })

        return {
            "agent": "Devil's Advocate",
            "challenges": challenges,
            "challenge_count": len([c for c in challenges if c['type'] != 'NO_OBJECTIONS']),
            "argument": "\n".join(f"⚡ [{c['severity']}] {c['challenge']}" for c in challenges)
        }

    def _compute_consensus(self, evaluations, devils_advocacy):
        """
        Synthesize debate into a consensus score.
        Adjusts weights based on Devil's Advocate challenges.
        """
        # Start with weighted average
        weighted_sum = 0
        agent_to_weight_key = {
            "Skill Analyst": "skill",
            "Experience Evaluator": "experience",
            "Culture Fit Assessor": "culture",
        }

        for eval_result in evaluations:
            agent = eval_result['agent']
            weight_key = agent_to_weight_key.get(agent, "skill")
            weight = self.WEIGHTS.get(weight_key, 0.33)
            weighted_sum += eval_result['score'] * weight

        consensus_score = int(weighted_sum)

        # Apply Devil's Advocate penalties
        penalty = 0
        high_challenges = [c for c in devils_advocacy['challenges'] if c['severity'] == 'HIGH']
        medium_challenges = [c for c in devils_advocacy['challenges'] if c['severity'] == 'MEDIUM']

        penalty += len(high_challenges) * 8
        penalty += len(medium_challenges) * 3

        # Cap penalty at 25 points
        penalty = min(25, penalty)
        consensus_score = max(5, consensus_score - penalty)

        return consensus_score, penalty

    def _generate_llm_moderation(self, evaluations, devils_advocacy, profile, job_description):
        """Use LLM to generate a nuanced moderator summary."""
        if not self.llm_agent:
            return None

        debate_text = ""
        for eval_r in evaluations:
            debate_text += f"\n{eval_r['agent']}:\n{eval_r['argument']}\n"
        debate_text += f"\nDevil's Advocate:\n{devils_advocacy['argument']}\n"

        name = profile.get('name', profile.get('filename', 'Unknown'))
        prompt = f"""You are the Moderator of a multi-agent candidate evaluation debate.
Three specialist agents evaluated {name}, and the Devil's Advocate raised challenges.

DEBATE TRANSCRIPT:
{debate_text[:2500]}

YOUR TASK:
1. Synthesize the debate into a final verdict.
2. State which agent's argument was most compelling and why.
3. Address each Devil's Advocate challenge briefly.
4. Provide a final consensus score (0-100).
5. Start with "Consensus Score: XX/100" on the first line.

Keep your response under 200 words. Be decisive."""

        response, elapsed = self.llm_agent.call_llm(
            self.llm_agent.MODELS.get("insights", "llama3:latest"),
            prompt, timeout=30
        )
        return response if response else None

    def debate(self, profile, job_description):
        """
        Run a full multi-agent debate for a single candidate.
        
        Returns:
            dict with debate_transcript, consensus_score, individual evaluations,
            devils_advocacy, moderator_summary, and all debate metadata.
        """
        name = profile.get('name', profile.get('filename', 'Unknown'))

        # Step 1: Each agent presents their evaluation
        skill_eval = self.skill_agent.evaluate(profile, job_description)
        exp_eval = self.experience_agent.evaluate(profile, job_description)
        culture_eval = self.culture_agent.evaluate(profile, job_description)

        evaluations = [skill_eval, exp_eval, culture_eval]

        # Step 2: Devil's Advocate challenges
        devils = self._devils_advocate(evaluations, profile, job_description)

        # Step 3: Compute consensus
        consensus_score, penalty = self._compute_consensus(evaluations, devils)

        # Step 4: LLM moderation (if available)
        moderator_summary = self._generate_llm_moderation(
            evaluations, devils, profile, job_description
        )

        # Extract LLM consensus score if provided
        if moderator_summary:
            score_match = re.search(r'[Cc]onsensus\s*[Ss]core[:\s]*(\d{1,3})', moderator_summary)
            if score_match:
                llm_consensus = min(100, int(score_match.group(1)))
                consensus_score = int(consensus_score * 0.5 + llm_consensus * 0.5)

        # Build debate transcript
        transcript = f"## 🏛️ Debate: {name}\n\n"
        
        for ev in evaluations:
            icon = {"Skill Analyst": "🔧", "Experience Evaluator": "💼", "Culture Fit Assessor": "🤝"}.get(ev['agent'], "📋")
            transcript += f"### {icon} {ev['agent']} ({ev['model']})\n"
            transcript += f"**Score: {ev['score']}/100**\n\n{ev['argument']}\n\n"

        transcript += f"### ⚡ Devil's Advocate\n"
        transcript += f"{devils['argument']}\n\n"

        transcript += f"### 🏆 Moderator Verdict\n"
        transcript += f"**Consensus Score: {consensus_score}/100**"
        if penalty > 0:
            transcript += f" (includes -{penalty}pt Devil's Advocate penalty)"
        transcript += "\n"

        if moderator_summary:
            transcript += f"\n{moderator_summary}\n"

        return {
            "candidate": name,
            "filename": profile.get('filename', ''),
            "consensus_score": consensus_score,
            "skill_score": skill_eval['score'],
            "experience_score": exp_eval['score'],
            "culture_score": culture_eval['score'],
            "evaluations": evaluations,
            "devils_advocacy": devils,
            "penalty": penalty,
            "moderator_summary": moderator_summary,
            "transcript": transcript,
        }

    def debate_batch(self, profiles, job_description):
        """
        Run debates for a batch of candidates.
        
        Returns:
            list of debate results, one per candidate.
        """
        results = []
        for profile in profiles:
            result = self.debate(profile, job_description)
            results.append(result)
        return results
