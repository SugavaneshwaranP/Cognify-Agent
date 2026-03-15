
    # ═══════════════════════════════════════════════════════════════════
    # TABBED RESULTS DASHBOARD
    # ═══════════════════════════════════════════════════════════════════
    tab_strategy, tab_shortlisted, tab_rankings, tab_insights, tab_report, tab_chat = st.tabs([
        "🧠 AI Strategy & Autonomous Reasoning",
        "🏆 Shortlisted Candidates",
        "📋 Full Candidate Rankings",
        "🤖 Final AI Insights & Recommendation",
        "📝 Automated Hiring Report",
        "💬 CognifyX AI Assistant",
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1: AI Strategy & Autonomous Reasoning
    # ══════════════════════════════════════════════════════════════
    with tab_strategy:
        plan = res.get('plan')
        if plan:
            st.markdown("## 🧠 AI Strategy & Autonomous Reasoning")

            st.markdown(f"""
            <div class='plan-card'>
                <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;'>
                    <div>
                        <span style='font-size:1.6rem; font-weight:800; color:#e0e7ff;'>{plan.strategy_label}</span>
                        <br><span style='font-size:0.85rem; color:#a5b4fc;'>
                            Complexity: <b>{plan.jd_complexity.title()}</b> &nbsp;|&nbsp;
                            Est. Time: <b>{plan.estimated_time}</b> &nbsp;|&nbsp;
                            ATS Top-N: <b>{plan.ats_top_n}</b>
                        </span>
                    </div>
                    <div style='text-align:right;'>
                        <span style='font-size:0.75rem; background:rgba(99,102,241,0.3); padding:4px 12px;
                            border-radius:8px; color:#c7d2fe;'>🧠 ReAct Reasoning</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            plan_col1, plan_col2 = st.columns([1, 1])

            with plan_col1:
                st.markdown("#### 💡 Strategic Recommendations")
                for rec in plan.recommendations:
                    st.markdown(f"- {rec}")

                st.markdown("#### ⚖️ Dynamic Weight Configuration")
                st.markdown(f"""
                <table class='plan-weight-table'>
                    <tr><th>Component</th><th>Weight</th></tr>
                    <tr><td>ATS Score</td><td><b>{plan.composite_weights['ats']:.0%}</b></td></tr>
                    <tr><td>Keyword Match</td><td><b>{plan.composite_weights['keyword']:.0%}</b></td></tr>
                    <tr><td>LLM Score</td><td><b>{plan.composite_weights['llm']:.0%}</b></td></tr>
                    <tr><td>Debate: Skill</td><td><b>{plan.debate_weights.get('skill', 0.40):.0%}</b></td></tr>
                    <tr><td>Debate: Experience</td><td><b>{plan.debate_weights.get('experience', 0.35):.0%}</b></td></tr>
                    <tr><td>Debate: Culture</td><td><b>{plan.debate_weights.get('culture', 0.25):.0%}</b></td></tr>
                </table>
                """, unsafe_allow_html=True)

            with plan_col2:
                st.markdown("#### 🔄 ReAct Reasoning Trace")
                for step in plan.reasoning_steps:
                    phase = step.get('phase', '')
                    if phase == 'THINK':
                        st.markdown(f"<div class='react-step react-think'><b>💭 THINK:</b> {step.get('observation', '')}<br><span style='color:#a5b4fc;'>→ {step.get('thought', '')}</span></div>", unsafe_allow_html=True)
                    elif phase == 'ACT':
                        st.markdown(f"<div class='react-step react-act'><b>⚡ ACT:</b> <code>{step.get('action', '')}</code><br><span style='color:#6ee7b7;'>→ {step.get('detail', '')}</span></div>", unsafe_allow_html=True)
                    elif phase == 'OBSERVE':
                        st.markdown(f"<div class='react-step react-observe'><b>👁️ OBSERVE:</b> {step.get('result', '')}</div>", unsafe_allow_html=True)
                    elif phase == 'LLM_REASONING':
                        st.markdown(f"<div class='react-step react-llm'><b>🤖 LLM Validation:</b> {step.get('output', '')[:400]}</div>", unsafe_allow_html=True)

                st.markdown("#### 🔧 Stage Configuration")
                stages_status = [
                    ("ATS Filter", plan.run_ats_filter, f"top_n={plan.ats_top_n}"),
                    ("LLM Extraction", plan.run_extraction, "Qwen"),
                    ("LLM Scoring", plan.run_scoring, "Mistral"),
                    ("Self-Correction", plan.run_reflection, "Reflection"),
                    ("Multi-Agent Debate", plan.run_debate, "3 Agents"),
                    ("Final Insights", plan.run_insights, "LLaMA"),
                ]
                for label, enabled, detail in stages_status:
                    icon = "✅" if enabled else "⏭️"
                    status = "Enabled" if enabled else "**Skipped**"
                    st.markdown(f"- {icon} **{label}** — {status} ({detail})")
        else:
            st.info("No planning data available. Run the pipeline to see AI strategy details.")

    # ══════════════════════════════════════════════════════════════
    # TAB 2: Shortlisted Candidates
    # ══════════════════════════════════════════════════════════════
    with tab_shortlisted:
        st.markdown("## 🏆 Shortlisted Candidates")

        for i, c in enumerate(res['shortlisted']):
            rank = c.get('final_rank', i + 1)
            medal = get_medal(rank)
            card_class = {1: "gold", 2: "silver", 3: "bronze"}.get(rank, "shortlisted")
            composite = c.get('composite_score', 0)
            score_class = get_score_class(composite)

            st.markdown(f"<div class='candidate-card {card_class}'>", unsafe_allow_html=True)
            col_info, col_scores, col_btn = st.columns([3, 2, 1])

            with col_info:
                st.markdown(f"### {medal} {c.get('name', c.get('filename', 'Unknown'))}")
                st.caption(f"📄 {c.get('filename', '')}  •  🏷️ {c.get('domain', 'General')}")
                skills = c.get('skills', [])
                if skills:
                    skill_html = " ".join([f"<span class='skill-tag'>{s}</span>" for s in skills[:8]])
                    st.markdown(f"**Skills:** {skill_html}", unsafe_allow_html=True)
                matched = c.get('keywords_matched', [])
                missed = c.get('keywords_missed', [])
                if matched:
                    match_html = " ".join([f"<span class='skill-tag matched'>✓ {k}</span>" for k in matched[:6]])
                    st.markdown(f"**Keywords Matched:** {match_html}", unsafe_allow_html=True)
                if missed:
                    miss_html = " ".join([f"<span class='skill-tag missed'>✗ {k}</span>" for k in missed[:4]])
                    st.markdown(f"**Missing:** {miss_html}", unsafe_allow_html=True)

            with col_scores:
                st.markdown(f"""
                <div style='text-align: center; padding-top: 10px;'>
                    <div class='score-badge {score_class}' style='font-size: 1.4rem; padding: 10px 24px;'>{composite}</div>
                    <div style='color: #64748b; font-size: 0.75rem; margin-top: 4px;'>Composite Score</div>
                </div>
                """, unsafe_allow_html=True)
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("ATS", f"{c.get('ats_score', 0)}")
                sc2.metric("Keywords", f"{c.get('keyword_score', 0)}")
                sc3.metric("LLM", f"{c.get('llm_score', 0)}")

            with col_btn:
                file_path = os.path.join(TEMP_DIR, c['filename'])
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        st.download_button(label="📄 Download", data=f, file_name=c['filename'], mime="application/octet-stream", key=f"dl_short_{i}")

            with st.expander("🔍 View AI Analysis"):
                st.write(c.get('llm_analysis', 'No analysis available.'))
            st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 3: Full Candidate Rankings
    # ══════════════════════════════════════════════════════════════
    with tab_rankings:
        st.markdown("## 📋 Full Candidate Rankings")
        show_all = st.checkbox("Show all candidates (not just shortlisted)", value=False)
        display_list = res['candidates'] if show_all else res['shortlisted']
        table_data = []
        for c in display_list:
            table_data.append({
                "Rank": c.get('final_rank', '-'), "Name": c.get('name', c.get('filename', '')),
                "File": c.get('filename', ''), "Domain": c.get('domain', 'General'),
                "Experience": f"{c.get('experience_years', '?')} yrs",
                "ATS Score": c.get('ats_score', 0), "Keyword Match": f"{c.get('keyword_score', 0)}%",
                "LLM Score": c.get('llm_score', 0), "Composite": c.get('composite_score', 0),
                "Shortlisted": "✅" if c.get('shortlisted') else "—",
            })
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=600)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Full Report (CSV)", data=csv_data, file_name=f'cognifyx_report_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', mime='text/csv', key="global_csv")

        # Self-Correction Report
        reflection_data = res.get('reflection_results', [])
        if reflection_data:
            st.markdown("---")
            st.markdown("### 🔄 AI Self-Correction Report")
            st.markdown("*The Reflection Agent audited each candidate's LLM score against hard facts from the JD and profile, correcting inflated or deflated scores.*")
            total_r = len(reflection_data)
            corrected_r = sum(1 for r in reflection_data if r.get('was_corrected'))
            validated_r = total_r - corrected_r
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #f59e0b;'>{corrected_r}</div><div class='metric-label'>Scores Corrected</div></div>", unsafe_allow_html=True)
            with rc2:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #10b981;'>{validated_r}</div><div class='metric-label'>Scores Validated</div></div>", unsafe_allow_html=True)
            with rc3:
                avg_adj = 0
                if corrected_r > 0:
                    avg_adj = round(sum(r.get('total_adjustment', 0) for r in reflection_data if r.get('was_corrected')) / corrected_r, 1)
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #ef4444;'>-{avg_adj}</div><div class='metric-label'>Avg Score Adjustment</div></div>", unsafe_allow_html=True)

            for ref in reflection_data:
                card_class = "corrected" if ref.get('was_corrected') else "validated"
                status_icon = "⚠️" if ref.get('was_corrected') else "✅"
                confidence = ref.get('confidence', 'MEDIUM')
                st.markdown(f"<div class='reflection-card {card_class}'>", unsafe_allow_html=True)
                ref_col1, ref_col2 = st.columns([3, 1])
                with ref_col1:
                    st.markdown(f"#### {status_icon} {ref.get('candidate', 'Unknown')}")
                    if ref.get('was_corrected'):
                        st.markdown(f"<div class='score-correction'><span class='score-original'>{ref.get('original_score', '?')}</span><span class='score-arrow'>→</span><span class='score-corrected'>{ref.get('corrected_score', '?')}</span><span style='font-size:0.8rem; color:#64748b;'>/100 (LLM Score)</span></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"Score **{ref.get('original_score', '?')}/100** — No issues found.")
                with ref_col2:
                    st.markdown(f"<div style='text-align:right; padding-top:10px;'><span class='confidence-badge confidence-{confidence}'>Confidence: {confidence}</span></div>", unsafe_allow_html=True)
                anomalies = ref.get('anomalies', [])
                if anomalies:
                    with st.expander(f"🔍 View {len(anomalies)} Anomalies Detected"):
                        for a in anomalies:
                            severity = a.get('severity', 'LOW')
                            st.markdown(f"<span class='anomaly-tag anomaly-{severity.lower()}'>{severity}</span> **{a.get('type', 'UNKNOWN').replace('_', ' ')}**", unsafe_allow_html=True)
                            st.write(f"  {a.get('description', '')}")
                llm_reflection = ref.get('llm_reflection')
                if llm_reflection:
                    with st.expander("🤖 View AI Reflection Analysis"):
                        st.write(llm_reflection)
                st.markdown("</div>", unsafe_allow_html=True)

        # Multi-Agent Debate Transcripts
        debate_data = res.get('debate_results', [])
        if debate_data:
            st.markdown("---")
            st.markdown("### 🏛️ Multi-Agent Debate Transcripts")
            st.markdown("*Three specialized agents debated each candidate. A Devil's Advocate challenged their conclusions, and the Moderator synthesized the final consensus score.*")
            for di, d in enumerate(debate_data):
                with st.expander(f"🏛️ {d.get('candidate', 'Unknown')} — Consensus: {d.get('consensus_score', '?')}/100 | Skill: {d.get('skill_score', '?')} | Exp: {d.get('experience_score', '?')} | Culture: {d.get('culture_score', '?')}{'  ⚡' + str(d.get('penalty', 0)) + 'pt penalty' if d.get('penalty', 0) > 0 else ''}", expanded=(di == 0)):
                    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                    with dcol1:
                        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #06b6d4;'>{d.get('skill_score', 0)}</div><div class='metric-label'>🔧 Skill</div></div>", unsafe_allow_html=True)
                    with dcol2:
                        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #8b5cf6;'>{d.get('experience_score', 0)}</div><div class='metric-label'>💼 Experience</div></div>", unsafe_allow_html=True)
                    with dcol3:
                        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #f59e0b;'>{d.get('culture_score', 0)}</div><div class='metric-label'>🤝 Culture</div></div>", unsafe_allow_html=True)
                    with dcol4:
                        penalty = d.get('penalty', 0)
                        penalty_text = f" (-{penalty})" if penalty > 0 else ""
                        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color: #10b981;'>{d.get('consensus_score', 0)}</div><div class='metric-label'>🏆 Consensus{penalty_text}</div></div>", unsafe_allow_html=True)
                    evaluations = d.get('evaluations', [])
                    for ev in evaluations:
                        agent = ev.get('agent', 'Unknown')
                        if 'Skill' in agent: css_class, icon = 'skill', '🔧'
                        elif 'Experience' in agent: css_class, icon = 'experience', '💼'
                        else: css_class, icon = 'culture', '🤝'
                        st.markdown(f"<div class='debate-agent {css_class}'><div class='debate-agent-header'><span class='debate-agent-name'>{icon} {agent} ({ev.get('model', '')})</span><span class='debate-score-pill'>{ev.get('score', 0)}/100</span></div><div style='font-size:0.83rem; color:#475569; white-space: pre-wrap;'>{ev.get('argument', '')[:500]}</div></div>", unsafe_allow_html=True)
                    devils = d.get('devils_advocacy', {})
                    challenges = devils.get('challenges', [])
                    if challenges:
                        challenge_text = ""
                        for ch in challenges:
                            sev = ch.get('severity', 'LOW')
                            sev_color = {'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#64748b'}.get(sev, '#64748b')
                            challenge_text += f"<span style='color:{sev_color}; font-weight:700;'>[{sev}]</span> {ch.get('challenge', '')}<br>"
                        st.markdown(f"<div class='debate-agent devil'><div class='debate-agent-header'><span class='debate-agent-name'>⚡ Devil's Advocate</span><span class='debate-score-pill' style='background:#fee2e2; color:#991b1b;'>{devils.get('challenge_count', 0)} challenges</span></div><div style='font-size:0.83rem; color:#475569;'>{challenge_text}</div></div>", unsafe_allow_html=True)
                    mod_summary = d.get('moderator_summary', '')
                    if mod_summary:
                        st.markdown(f"<div class='debate-agent moderator'><div class='debate-agent-header'><span class='debate-agent-name'>🏆 Moderator Verdict</span><span class='debate-score-pill' style='background:#d1fae5; color:#065f46;'>Consensus: {d.get('consensus_score', 0)}/100</span></div><div style='font-size:0.83rem; color:#475569; white-space: pre-wrap;'>{mod_summary[:600]}</div></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 4: Final AI Insights & Recommendation
    # ══════════════════════════════════════════════════════════════
    with tab_insights:
        st.markdown("## 🤖 Final AI Insights & Recommendation")
        st.markdown(f"<div class='insights-panel'>{res['insights']}</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 5: Automated Hiring Report
    # ══════════════════════════════════════════════════════════════
    with tab_report:
        st.markdown("## 📝 Automated Hiring Report")
        st.markdown("*Generate a polished, professional hiring report with visualizations, comparisons, and actionable recommendations.*")
        if 'report_data' not in st.session_state:
            st.session_state.report_data = None
        gen_col1, gen_col2, gen_col3 = st.columns([2, 1, 1])
        with gen_col1:
            if st.button("🧠 Generate Full Report", key="gen_report", use_container_width=True):
                with st.spinner("Generating report sections..."):
                    report_agent = ReportAgent()
                    st.session_state.report_data = report_agent.generate_full_report(res, job_description=st.session_state.get('last_jd', ''))
                st.success("✅ Report generated!")
                st.rerun()
        report = st.session_state.report_data
        if report:
            with gen_col2:
                try:
                    pdf_gen = PDFReportGenerator()
                    pdf_bytes = pdf_gen.generate(report)
                    st.download_button("📄 Download PDF", data=pdf_bytes, file_name=f"CognifyX_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf", key="dl_pdf_report", use_container_width=True)
                except Exception as e:
                    st.warning(f"PDF unavailable: {e}")
            with gen_col3:
                try:
                    docx_gen = DOCXReportGenerator()
                    docx_bytes = docx_gen.generate(report)
                    st.download_button("📝 Download DOCX", data=docx_bytes, file_name=f"CognifyX_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="dl_docx_report", use_container_width=True)
                except Exception as e:
                    st.warning(f"DOCX unavailable: {e}")
            exec_sum = report.get('executive_summary', {})
            if exec_sum:
                with st.expander("📋 Executive Summary", expanded=True):
                    st.markdown(exec_sum.get('content', ''))
                    stats = exec_sum.get('stats', {})
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Screened", stats.get('total_screened', 0))
                    sc2.metric("Filtered", stats.get('ats_filtered', 0))
                    sc3.metric("Shortlisted", stats.get('shortlisted', 0))
                    sc4.metric("Top Score", f"{stats.get('top_score', 0)}/100")
            matrix = report.get('comparison_matrix', {})
            matrix_rows = matrix.get('rows', [])
            if matrix_rows:
                with st.expander("📊 Candidate Comparison Matrix"):
                    import pandas as pd
                    df_matrix = pd.DataFrame(matrix_rows)
                    display_cols = ['rank', 'name', 'domain', 'experience_years', 'ats_score', 'keyword_score', 'llm_score', 'composite_score', 'consensus_score']
                    available = [c for c in display_cols if c in df_matrix.columns]
                    st.dataframe(df_matrix[available], use_container_width=True, hide_index=True)
            skill_gap = report.get('skill_gap_analysis', {})
            if skill_gap:
                with st.expander("🔍 Skill Gap Analysis"):
                    gap_col1, gap_col2 = st.columns([2, 1])
                    with gap_col1:
                        coverage = skill_gap.get('coverage', [])
                        if coverage:
                            for c in coverage:
                                status = c['status']
                                if status == 'Strong': icon = '🟢'
                                elif status == 'Moderate': icon = '🟡'
                                else: icon = '🔴'
                                st.markdown(f"{icon} **{c['keyword']}** — {c['coverage_pct']}% coverage ({c['candidates_with']} candidates)")
                    with gap_col2:
                        st.metric("Total Keywords", skill_gap.get('total_keywords', 0))
                        st.metric("Strong Coverage", skill_gap.get('strong_coverage', 0))
                        st.metric("Gaps Found", skill_gap.get('gaps_found', 0))
            iq = report.get('interview_questions', {})
            if iq:
                with st.expander("💬 Interview Question Bank"):
                    st.markdown(f"*{iq.get('total_questions', 0)} personalized questions generated*")
                    for cand in iq.get('candidates', []):
                        st.markdown(f"#### #{cand['rank']} – {cand['candidate']}")
                        for q in cand['questions']:
                            st.markdown(f"**[{q['type']}]** {q['question']}  \n<small style='color:#64748b;'>📌 {q['reason']}</small>", unsafe_allow_html=True)
                        st.markdown("---")
            diversity = report.get('diversity_metrics', {})
            if diversity:
                with st.expander("🌍 Diversity & Inclusion Metrics"):
                    div_col1, div_col2, div_col3 = st.columns(3)
                    div_col1.metric("Diversity Index", f"{diversity.get('diversity_index', 0)}%")
                    div_col2.metric("Unique Domains", diversity.get('unique_domains', 0))
                    div_col3.metric("Total Pool", diversity.get('total_pool', 0))
                    st.markdown("**Domain Distribution:**")
                    for domain, count in diversity.get('domain_distribution', {}).items():
                        st.markdown(f"- **{domain}**: {count} candidates")
                    st.markdown("**Education Distribution:**")
                    for edu, count in diversity.get('education_distribution', {}).items():
                        if count > 0: st.markdown(f"- **{edu}**: {count}")
                    st.markdown("**Experience Distribution:**")
                    for bucket, count in diversity.get('experience_distribution', {}).items():
                        if count > 0: st.markdown(f"- **{bucket}**: {count}")
            salary = report.get('salary_benchmarks', {})
            benchmarks = salary.get('benchmarks', [])
            if benchmarks:
                with st.expander("💰 Salary Benchmarking"):
                    import pandas as pd
                    df_salary = pd.DataFrame(benchmarks)
                    display_cols = ['rank', 'name', 'level', 'salary_min', 'salary_max', 'midpoint', 'composite_score']
                    available = [c for c in display_cols if c in df_salary.columns]
                    st.dataframe(df_salary[available], use_container_width=True, hide_index=True)
                    st.caption(salary.get('note', ''))

    # ══════════════════════════════════════════════════════════════
    # TAB 6: CognifyX AI Assistant
    # ══════════════════════════════════════════════════════════════
    with tab_chat:
        st.markdown("""
        <div class='chat-container'>
            <div class='chat-header'>
                <span class='chat-header-icon'>💬</span>
                <span class='chat-header-title'>CognifyX AI Assistant</span>
                <span class='chat-header-badge'>Agentic AI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        proactive_qs = st.session_state.proactive_questions
        if proactive_qs and not any(m.get('role') == 'user' for m in st.session_state.chat_messages):
            st.markdown("**🤔 I have some questions before you proceed:**")
            for i, q in enumerate(proactive_qs):
                if st.button(q, key=f"proactive_{i}", use_container_width=True):
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    response = st.session_state.chat_agent.chat(q, st.session_state.results)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    st.session_state.proactive_questions = None
                    st.rerun()
        for msg in st.session_state.chat_messages:
            if msg['role'] == 'user':
                st.markdown(f"<div class='chat-msg user'><div class='chat-bubble user-bubble'>{msg['content']}</div><div class='chat-avatar user-av'>👤</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-msg assistant'><div class='chat-avatar ai-av'>🤖</div><div class='chat-bubble ai-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
        chat_col1, chat_col2 = st.columns([5, 1])
        with chat_col1:
            user_input = st.text_input("Ask CognifyX AI...", placeholder="Compare #1 and #2 | Why is #3 ranked here? | Prioritize experience | Summary", key=f"chat_input_{st.session_state.chat_input_key}", label_visibility="collapsed")
        with chat_col2:
            send_clicked = st.button("🚀 Send", use_container_width=True, key="chat_send_btn")
        if (send_clicked or user_input) and user_input and user_input.strip():
            st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})
            response = st.session_state.chat_agent.chat(user_input.strip(), st.session_state.results)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.session_state.proactive_questions = None
            st.session_state.chat_input_key += 1
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)
        with qa_col1:
            if st.button("📊 Summary", key="qa_summary", use_container_width=True):
                msg = "Give me a summary"
                st.session_state.chat_messages.append({"role": "user", "content": msg})
                response = st.session_state.chat_agent.chat(msg, st.session_state.results)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
        with qa_col2:
            if st.button("🔍 Compare Top 2", key="qa_compare", use_container_width=True):
                msg = "Compare #1 and #2"
                st.session_state.chat_messages.append({"role": "user", "content": msg})
                response = st.session_state.chat_agent.chat(msg, st.session_state.results)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
        with qa_col3:
            if st.button("💡 Explain #1", key="qa_explain", use_container_width=True):
                msg = "Explain why #1 is ranked first"
                st.session_state.chat_messages.append({"role": "user", "content": msg})
                response = st.session_state.chat_agent.chat(msg, st.session_state.results)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
        with qa_col4:
            if st.button("🗑️ Clear Chat", key="qa_clear", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.chat_agent.conversation_history = []
                st.session_state.chat_agent.preferences = {}
                st.session_state.proactive_questions = st.session_state.chat_agent.get_proactive_questions(st.session_state.results)
                st.rerun()

elif not st.session_state.results:
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 20px;
        padding: 40px; text-align: center; border: 1px solid #334155;'>
        <span style='font-size: 3rem;'>💬</span>
        <h3 style='color: #f1f5f9 !important; margin-top: 12px;'>CognifyX AI Assistant</h3>
        <p style='color: #94a3b8 !important; max-width: 500px; margin: 0 auto;'>
            Run the screening pipeline to activate the AI Assistant.
            Ask questions, compare candidates, refine rankings, and get personalized recommendations.
        </p>
        <div style='margin-top: 16px;'>
            <span class='chat-header-badge'>Agentic AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
