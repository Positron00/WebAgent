permit (principal, action, resource);

@id("untrusted-injection-watch") @outcome("verify")
forbid (principal, action == Action::"converse", resource)
when { ["web_result", "retrieved_doc", "document", "tool_output", "unlabelled"].contains(context.source)
       && context.injection >= context.injection_watch_at && context.injection < context.injection_drop_at };

@id("untrusted-injection-drop") @outcome("decline")
@reason("This content looked like instructions aimed at the assistant.")
forbid (principal, action == Action::"converse", resource)
when { ["web_result", "retrieved_doc", "document", "tool_output", "unlabelled"].contains(context.source)
       && (context.injection >= context.injection_drop_at || context.pattern_hits.containsAny(["injection_phrase", "role_marker", "hidden_text"])) };

@id("user-harm-decline") @outcome("decline")
@reason("This request looks harmful.")
forbid (principal, action == Action::"converse", resource)
when { ["user_turn", "history"].contains(context.source) && context.harm >= context.harm_decline_at };

// Below the decline and consent scores, the classifier can't tell look-alike questions from the real
// thing on its own; these send the turn to the LLM judge (hop_guard), which declines, asks for
// consent or allows. Without the judge, verify continues with the task tainted.
@id("user-harm-verify") @outcome("verify")
@reason("This request needs a closer look.")
forbid (principal, action == Action::"converse", resource)
when { ["user_turn", "history"].contains(context.source)
       && context.harm >= context.harm_verify_at && context.harm < context.harm_decline_at };

@id("user-jailbreak-verify") @outcome("verify")
@reason("This may try to change how the assistant behaves.")
forbid (principal, action == Action::"converse", resource)
when { context.source == "user_turn" && context.jailbreak >= context.jailbreak_verify_at
       && context.jailbreak < context.jailbreak_consent_at && context.harm < context.harm_decline_at };

@id("user-jailbreak-consent") @outcome("consent")
@reason("This may try to change how the assistant behaves.")
forbid (principal, action == Action::"converse", resource)
when { context.source == "user_turn" && context.jailbreak >= context.jailbreak_consent_at
       && context.harm < context.harm_decline_at && !context.consented };

@id("search-secret-decline") @outcome("decline")
@reason("The search query may expose secrets.")
forbid (principal, action == Action::"tool::web_search", resource)
when { context.pattern_hits.containsAny(["api_key", "private_key", "password"]) };

@id("tainted-search-verify") @outcome("verify")
forbid (principal, action == Action::"tool::web_search", resource)
when { context.tainted && context.alignment < 60 };

@id("tainted-code-consent") @outcome("consent")
forbid (principal, action == Action::"tool::execute_code", resource)
when { (context.tainted || context.code_findings.contains("network"))
       && !context.approved_tools.contains("tool::execute_code") };

@id("repeated-declines-escalate") @outcome("escalate")
forbid (principal, action, resource)
when { context.declined_actions_in_task >= 3 };

@id("classifier-down-untrusted") @outcome("verify")
forbid (principal, action == Action::"converse", resource)
when { !context.scores_available && context.source != "user_turn" };

@id("document-provenance-decline") @outcome("decline")
forbid (principal, action == Action::"tool::document_extract", resource)
when { context.pattern_hits.contains("document_provenance") };

@id("search-personal-data-verify") @outcome("verify")
forbid (principal, action in [Action::"tool::web_search", Action::"tool::internal_retrieve"], resource)
when { context.pattern_hits.containsAny(["email", "phone", "long_token"]) };

@id("search-url-verify") @outcome("verify")
forbid (principal, action in [Action::"tool::web_search", Action::"tool::internal_retrieve"], resource)
when { context.pattern_hits.containsAny(["url", "site_operator", "query_too_long"]) };

@id("code-dangerous-decline") @outcome("decline")
forbid (principal, action == Action::"tool::execute_code", resource)
when { context.code_findings.containsAny(["process", "native", "env", "unparseable"]) };

@id("code-disabled-decline") @outcome("decline")
forbid (principal, action == Action::"tool::execute_code", resource)
when { !context.code_execution_enabled };

// Checkpoint 2 against the user's intent (checkpoint2-intent spec SEC-109). All shadow-only: matches are
// recorded (webagent_guard_shadow_policy_total, collection) and never change an outcome until a rule's
// threshold is measured (SEC-130). off-intent-search-verify replaces untainted-search-misaligned
// (SEC-044a): it compares with the user's own turns, not the supervisor's rewrite, tainted or not.
@id("rewrite-drift-verify") @outcome("verify") @shadow("true")
forbid (principal, action == Action::"plan::rewrite", resource)
when { context.signals_available && context.rewrite_alignment < context.rewrite_at };

@id("subgoal-harm-decline") @outcome("decline") @shadow("true")
forbid (principal, action == Action::"plan::subgoal", resource)
when { context.signals_available && context.subgoal_harm >= context.harm_decline_at
       && context.intent_harm < context.harm_verify_at };

@id("subgoal-off-intent-decline") @outcome("decline") @shadow("true")
forbid (principal, action == Action::"plan::subgoal", resource)
when { context.signals_available && context.intent_alignment < context.subgoal_align_at };

@id("call-risk-above-intent-decline") @outcome("decline") @shadow("true")
forbid (principal, action in [Action::"tool::web_search", Action::"tool::internal_retrieve", Action::"tool::execute_code"], resource)
when { context.signals_available && context.risk_above_intent >= context.risk_decline_at };

@id("call-risk-above-intent-verify") @outcome("verify") @shadow("true")
forbid (principal, action in [Action::"tool::web_search", Action::"tool::internal_retrieve", Action::"tool::execute_code"], resource)
when { context.signals_available && context.risk_above_intent >= context.risk_verify_at };

@id("off-intent-search-verify") @outcome("verify") @shadow("true")
forbid (principal, action in [Action::"tool::web_search", Action::"tool::internal_retrieve"], resource)
when { context.signals_available && context.intent_alignment < context.search_align_at };

@id("private-egress-decline") @outcome("decline") @shadow("true")
forbid (principal, action == Action::"tool::web_search", resource)
when { context.egress_overlap >= context.egress_words_at };

@id("undeclared-tool-verify") @outcome("verify") @shadow("true")
forbid (principal, action in [Action::"tool::web_search", Action::"tool::internal_retrieve", Action::"tool::document_extract", Action::"tool::execute_code"], resource)
when { !context.declared };

@id("drift-escalate") @outcome("escalate") @shadow("true")
forbid (principal, action in [Action::"tool::web_search", Action::"tool::internal_retrieve", Action::"plan::subgoal"], resource)
when { context.low_alignment_calls >= context.drift_calls_at };
