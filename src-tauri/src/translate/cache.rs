//! The last translations, kept in memory: the same text through the same
//! model and prompt style is answered at once instead of asking the model
//! again. Nothing is written to disk; the popup's "Translate again" skips the
//! lookup and replaces the entry.

use std::sync::Mutex;

/// How many translations are remembered (the most recent ones).
const CAPACITY: usize = 8;

/// Most recently used last.
pub struct Cache {
    entries: Vec<(String, String)>,
}

impl Cache {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// The remembered translation; the entry becomes the most recent one.
    pub fn get(&mut self, key: &str) -> Option<String> {
        let at = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(at);
        let translation = entry.1.clone();
        self.entries.push(entry);
        Some(translation)
    }

    /// Remembers a translation (an empty one is not worth keeping); the same
    /// key is replaced, the oldest entry goes when the cache is full.
    pub fn put(&mut self, key: String, translation: String) {
        if translation.trim().is_empty() {
            return;
        }
        self.entries.retain(|(k, _)| *k != key);
        if self.entries.len() == CAPACITY {
            self.entries.remove(0);
        }
        self.entries.push((key, translation));
    }
}

/// What a translation depends on: the model file, the prompt style (with a
/// custom prompt), the source and target languages (both can be picked in the
/// popup) and the text.
pub fn key(text: &str, model: &str, style: &str, src: &str, tgt: &str) -> String {
    format!(
        "{}:{model}{}:{style}{}:{src}{}:{tgt}{text}",
        model.len(),
        style.len(),
        src.len(),
        tgt.len()
    )
}

static CACHE: Mutex<Cache> = Mutex::new(Cache::new());

pub fn lookup(key: &str) -> Option<String> {
    CACHE.lock().unwrap_or_else(|e| e.into_inner()).get(key)
}

pub fn remember(key: String, translation: String) {
    CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .put(key, translation);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_remembered_text_comes_back() {
        let mut cache = Cache::new();
        let k = key("Привет", "fast.gguf", "Standard", "ru", "en");
        assert_eq!(cache.get(&k), None);
        cache.put(k.clone(), "Hello".into());
        assert_eq!(cache.get(&k).as_deref(), Some("Hello"));
    }

    #[test]
    fn another_model_style_or_text_is_another_entry() {
        let base = key("Привет", "fast.gguf", "Standard", "ru", "en");
        assert_ne!(base, key("Привет", "big.gguf", "Standard", "ru", "en"));
        assert_ne!(base, key("Привет", "fast.gguf", "Natural", "ru", "en"));
        assert_ne!(base, key("Привет!", "fast.gguf", "Standard", "ru", "en"));
        assert_ne!(base, key("Привет", "fast.gguf", "Standard", "ru", "de"));
        assert_ne!(base, key("Привет", "fast.gguf", "Standard", "bg", "en"));
        // Where the model ends and the style begins cannot be faked.
        assert_ne!(key("a", "ms", "x", "ru", "en"), key("a", "m", "sx", "ru", "en"));
        assert_ne!(key("1:b", "a", "", "ru", "en"), key("", "a", "1:b", "ru", "en"));
    }

    #[test]
    fn the_same_key_is_replaced() {
        let mut cache = Cache::new();
        let k = key("Привет", "fast.gguf", "Standard", "ru", "en");
        cache.put(k.clone(), "Hello".into());
        cache.put(k.clone(), "Hi".into());
        assert_eq!(cache.get(&k).as_deref(), Some("Hi"));
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn the_oldest_goes_and_a_read_keeps_an_entry_alive() {
        let mut cache = Cache::new();
        for i in 0..CAPACITY {
            cache.put(format!("k{i}"), format!("t{i}"));
        }
        // k0 is the oldest; reading it makes k1 the oldest.
        assert!(cache.get("k0").is_some());
        cache.put("new".into(), "t".into());
        assert_eq!(cache.entries.len(), CAPACITY);
        assert!(cache.get("k0").is_some());
        assert!(cache.get("k1").is_none());
    }

    #[test]
    fn an_empty_translation_is_not_kept() {
        let mut cache = Cache::new();
        cache.put("k".into(), "  \n".into());
        assert_eq!(cache.get("k"), None);
    }
}
