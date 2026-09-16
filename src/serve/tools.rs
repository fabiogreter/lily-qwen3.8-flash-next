//! Tool definitions and the model's `<tool_call>` XML.
//!
//! Qwen3.8 emits calls as
//!
//! ```text
//! <tool_call>
//! <function=name>
//! <parameter=key>
//! value
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Parameter values are plain text; the tool's JSON schema decides whether a
//! value is kept as a string or parsed as JSON (numbers, booleans, objects,
//! arrays), the way the reference parser in Qwen's own server does.

use anyhow::{Context as _, Result, bail};
use serde_json::{Map, Value};

/// One function the request offered, reduced to what parsing needs.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: String,
    /// `parameters.properties`, if the schema has any.
    pub properties: Map<String, Value>,
}

impl ToolSchema {
    /// Reads OpenAI `{"type": "function", "function": {...}}` tool entries.
    pub fn from_request(tools: &[Value]) -> Result<Vec<Self>> {
        tools
            .iter()
            .map(|tool| {
                let function = tool
                    .get("function")
                    .filter(|f| f.is_object())
                    .context("each tool needs a `function` object")?;
                let name = function
                    .get("name")
                    .and_then(Value::as_str)
                    .context("each tool function needs a `name`")?
                    .to_string();
                let properties = function
                    .get("parameters")
                    .and_then(|p| p.get("properties"))
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                Ok(Self { name, properties })
            })
            .collect()
    }

    fn param_is_string(&self, key: &str) -> bool {
        let Some(schema) = self.properties.get(key) else {
            // Unknown parameter: keep the text.
            return true;
        };
        match schema.get("type") {
            Some(Value::String(t)) => t == "string",
            Some(Value::Array(ts)) => {
                // Nullable strings are common: ["string", "null"].
                ts.iter().any(|t| t == "string") && ts.len() <= 2
            }
            // No type: enum-of-strings or anything else; text is the safe read.
            _ => schema.get("enum").is_some() || schema.get("anyOf").is_none(),
        }
    }
}

/// A parsed call: the function name and its arguments as a JSON object text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

/// Parses the text between `<tool_call>` and `</tool_call>`.
pub fn parse_tool_call(block: &str, tools: &[ToolSchema]) -> Result<ParsedToolCall> {
    let block = block.trim();
    let rest = block
        .strip_prefix("<function=")
        .context("tool call does not start with <function=")?;
    let (name, rest) = rest.split_once('>').context("unterminated <function= tag")?;
    let name = name.trim();
    let body = rest
        .strip_suffix("</function>")
        .or_else(|| rest.trim_end().strip_suffix("</function>"))
        .context("tool call does not end with </function>")?;
    let schema = tools.iter().find(|t| t.name == name);
    let mut arguments = Map::new();
    let mut cursor = body;
    while let Some(start) = cursor.find("<parameter=") {
        let after = &cursor[start + "<parameter=".len()..];
        let (key, after) =
            after.split_once('>').context("unterminated <parameter= tag")?;
        let end = after.find("</parameter>").context("unterminated parameter")?;
        let raw = &after[..end];
        // The template writes one newline around the value; strip exactly that.
        let value = raw.strip_prefix('\n').unwrap_or(raw);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let typed = match schema {
            Some(s) if s.param_is_string(key) => Value::String(value.to_string()),
            Some(_) => serde_json::from_str(value)
                .unwrap_or_else(|_| Value::String(value.to_string())),
            None => Value::String(value.to_string()),
        };
        arguments.insert(key.trim().to_string(), typed);
        cursor = &after[end + "</parameter>".len()..];
    }
    if cursor.trim().contains("<parameter") {
        bail!("malformed parameter block");
    }
    Ok(ParsedToolCall {
        name: name.to_string(),
        arguments: serde_json::to_string(&Value::Object(arguments))?,
    })
}

/// Parses OpenAI-style `tool_calls` on an incoming assistant message into
/// the shape the chat template wants (`arguments` as an object).
pub fn template_tool_calls(calls: &[Value]) -> Result<Vec<Value>> {
    calls
        .iter()
        .map(|call| {
            let function = call
                .get("function")
                .filter(|f| f.is_object())
                .context("assistant tool_calls entries need a `function` object")?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .context("tool call function needs a `name`")?;
            let arguments = match function.get("arguments") {
                None | Some(Value::Null) => Value::Object(Map::new()),
                Some(Value::String(text)) => {
                    if text.trim().is_empty() {
                        Value::Object(Map::new())
                    } else {
                        serde_json::from_str::<Value>(text).with_context(|| {
                            format!("tool call {name}: arguments are not valid JSON")
                        })?
                    }
                }
                Some(other) => other.clone(),
            };
            anyhow::ensure!(
                arguments.is_object(),
                "tool call {name}: arguments must be a JSON object"
            );
            Ok(serde_json::json!({
                "id": call.get("id").cloned().unwrap_or(Value::Null),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }))
        })
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/serve/tools.rs"]
mod tests;
