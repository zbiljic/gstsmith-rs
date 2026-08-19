#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Role {
    System,
    User,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "reserved neutral role is exercised by adapter tests before literal mode emits it"
        )
    )]
    Model,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Part {
    Text(String),
    Media { mime_type: String, data_url: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Message {
    pub(crate) role: Role,
    pub(crate) parts: Vec<Part>,
}

pub(crate) fn literal_messages(
    system: Option<String>,
    user: String,
    jpeg_data_urls: Vec<String>,
) -> Vec<Message> {
    let mut messages = Vec::with_capacity(2);
    if let Some(text) = system {
        messages.push(Message {
            role: Role::System,
            parts: vec![Part::Text(text)],
        });
    }
    let mut parts = Vec::with_capacity(jpeg_data_urls.len().saturating_add(1));
    parts.push(Part::Text(user));
    parts.extend(jpeg_data_urls.into_iter().map(|data_url| Part::Media {
        mime_type: "image/jpeg".to_owned(),
        data_url,
    }));
    messages.push(Message {
        role: Role::User,
        parts,
    });
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_absent_system_is_omitted_and_parts_are_ordered() {
        let messages = literal_messages(None, "{{literal}}".into(), vec!["data:one".into()]);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].parts[0], Part::Text("{{literal}}".into()));
        assert_eq!(
            messages[0].parts[1],
            Part::Media {
                mime_type: "image/jpeg".into(),
                data_url: "data:one".into()
            }
        );
    }

    #[test]
    fn prompt_empty_system_is_preserved() {
        let messages = literal_messages(Some(String::new()), "user".into(), Vec::new());
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].parts, vec![Part::Text(String::new())]);
    }
}
