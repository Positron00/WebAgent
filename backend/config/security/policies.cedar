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

// Shadow-only (2026-10-12): a search far from the task in a task checkpoint 1 didn't taint. InjecAgent
// injections that checkpoint 1 misses produce exactly these (alignment median 9-16); how often real
// follow-up searches fall below 30 isn't known yet, so matches are recorded, never applied (SEC-044a).
@id("untainted-search-misaligned") @outcome("verify") @shadow("true")
forbid (principal, action in [Action::"tool::web_search", Action::"tool::internal_retrieve"], resource)
when { !context.tainted && context.alignment < 30 };

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
