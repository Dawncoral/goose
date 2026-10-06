//! Ends the turn when the agent has used its autonomous turn budget.

use crate::agents::state_machine::effects::GooseEffect;
use crate::agents::state_machine::ops_stop_hook::DENIED;
use crate::agents::state_machine::{
    messages_since_kickoff, not_applicable, yielded_with, Emitter, Operation, OperationResult,
};
use crate::conversation::message::Message;
use crate::conversation::Conversation;
use crate::session::Session;
use anyhow::Result;
use async_trait::async_trait;
use rmcp::model::Role;

pub const MAX_TURNS_MESSAGE: &str = "I've reached the maximum number of actions I can do without user input. Would you like me to continue?";

pub struct MaxTurnsOperation {
    max_turns: u32,
}

fn turn_budget(messages: &[Message]) -> (u32, bool) {
    let mut counted_turns = 0;
    let mut next_reply_is_stop_hook_retry = false;
    for message_group in messages.chunk_by(|previous, next| previous.role == next.role) {
        if message_group.iter().any(|message| {
            message
                .metadata
                .operation_note("stop_hook", DENIED)
                .is_some()
        }) {
            next_reply_is_stop_hook_retry = true;
        }

        if message_group[0].role == Role::Assistant
            && message_group
                .iter()
                .any(|message| message.metadata.inference.is_some())
        {
            if !next_reply_is_stop_hook_retry {
                counted_turns += 1;
            }
            next_reply_is_stop_hook_retry = false;
        }
    }
    (counted_turns, next_reply_is_stop_hook_retry)
}

fn turn_budget_part(counted_turns: u32, max_turns: u32) -> Option<String> {
    if max_turns == 0 || counted_turns.saturating_mul(2) < max_turns {
        return None;
    }

    Some(format!(
        "<turn-budget>{counted_turns}/{max_turns} used</turn-budget>"
    ))
}

impl MaxTurnsOperation {
    pub fn new(max_turns: u32) -> Self {
        Self { max_turns }
    }
}

#[async_trait]
impl Operation<Session, GooseEffect> for MaxTurnsOperation {
    fn name(&self) -> &'static str {
        "max_turns"
    }

    async fn moim_parts(
        &self,
        _session: &Session,
        conversation: &Conversation,
    ) -> Result<Vec<String>> {
        let (counted_turns, _) = turn_budget(messages_since_kickoff(conversation)?);
        Ok(turn_budget_part(counted_turns, self.max_turns)
            .into_iter()
            .collect())
    }

    async fn run(
        &self,
        _session: &Session,
        conversation: &Conversation,
        emit: &Emitter,
    ) -> Result<OperationResult<GooseEffect>> {
        let messages = messages_since_kickoff(conversation)?;
        let (counted_turns, next_reply_is_stop_hook_retry) = turn_budget(messages);
        if next_reply_is_stop_hook_retry || counted_turns < self.max_turns {
            return not_applicable();
        }

        let message = Message::assistant().with_text(MAX_TURNS_MESSAGE);
        let message = emit.message(message).await;
        yielded_with([message.into()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::message::InferenceMetadata;

    fn model_reply() -> Message {
        Message::assistant().with_inference(InferenceMetadata {
            provider: "test".to_string(),
            requested_model: "test".to_string(),
            resolved_model: None,
            provider_session_id: None,
        })
    }

    fn stop_hook_denial() -> Message {
        let mut message = Message::user().with_visibility(false, true);
        message
            .metadata
            .set_operation_note("stop_hook", DENIED, serde_json::json!(true));
        message
    }

    #[test]
    fn turn_budget_counts_model_replies_once() {
        assert_eq!(turn_budget(&[]), (0, false));

        let messages = [
            Message::user(),
            model_reply().with_text("first chunk"),
            model_reply().with_text("second chunk"),
            Message::user(),
            Message::assistant().with_text("Goose-generated output"),
            Message::user(),
            model_reply().with_text("next reply"),
        ];

        assert_eq!(turn_budget(&messages), (2, false));
    }

    #[test]
    fn turn_budget_exempts_only_next_reply_after_denial() {
        let mut messages = vec![Message::user(), model_reply(), stop_hook_denial()];
        assert_eq!(turn_budget(&messages), (1, true));

        messages.push(Message::assistant().with_text("Goose-generated output"));
        assert_eq!(turn_budget(&messages), (1, true));

        messages.extend([model_reply(), model_reply()]);
        assert_eq!(turn_budget(&messages), (1, false));

        messages.extend([stop_hook_denial(), model_reply()]);
        assert_eq!(turn_budget(&messages), (1, false));

        messages.extend([Message::user(), model_reply()]);
        assert_eq!(turn_budget(&messages), (2, false));
    }
}
