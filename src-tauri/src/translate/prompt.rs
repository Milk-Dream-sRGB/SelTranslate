//! TranslateGemma prompt, translation direction and text splitting (B4).
//!
//! Pure logic: no Win32, no network, no Tauri. Called from `TranslateAction` (B5).

use std::ops::Range;

use serde::{Deserialize, Serialize};
use specta::Type;
use whatlang::Lang;

/// Chunk size sent to the model, in characters (B1: ~300–450 tokens; with the
/// prompt and the answer it stays well inside the 4096-token context).
pub const MAX_CHUNK_CHARS: usize = 1000;

/// Longest selection that is translated; the rest is cut off (the caller warns).
pub const MAX_SELECTION_CHARS: usize = 20_000;

/// ISO 639-1 code → name, exactly as in the `languages` table of the model's
/// chat template (checked against the GGUF in B4). One table for both
/// directions: `whatlang` language, template code, template name.
const LANGUAGES: &[(Lang, &str, &str)] = &[
    (Lang::Eng, "en", "English"),
    (Lang::Rus, "ru", "Russian"),
    (Lang::Ukr, "uk", "Ukrainian"),
    (Lang::Bel, "be", "Belarusian"),
    (Lang::Deu, "de", "German"),
    (Lang::Fra, "fr", "French"),
    (Lang::Spa, "es", "Spanish"),
    (Lang::Ita, "it", "Italian"),
    (Lang::Pol, "pl", "Polish"),
    (Lang::Por, "pt", "Portuguese"),
    (Lang::Nld, "nl", "Dutch"),
    (Lang::Tur, "tr", "Turkish"),
    (Lang::Jpn, "ja", "Japanese"),
    (Lang::Kor, "ko", "Korean"),
    (Lang::Cmn, "zh", "Chinese"),
    (Lang::Ara, "ar", "Arabic"),
    (Lang::Ces, "cs", "Czech"),
    (Lang::Slk, "sk", "Slovak"),
    (Lang::Slv, "sl", "Slovenian"),
    (Lang::Hrv, "hr", "Croatian"),
    (Lang::Srp, "sr", "Serbian"),
    (Lang::Bul, "bg", "Bulgarian"),
    (Lang::Mkd, "mk", "Macedonian"),
    (Lang::Ron, "ro", "Romanian"),
    (Lang::Hun, "hu", "Hungarian"),
    (Lang::Swe, "sv", "Swedish"),
    (Lang::Dan, "da", "Danish"),
    (Lang::Nob, "nb", "Norwegian Bokmål"),
    (Lang::Fin, "fi", "Finnish"),
    (Lang::Est, "et", "Estonian"),
    (Lang::Lav, "lv", "Latvian"),
    (Lang::Lit, "lt", "Lithuanian"),
    (Lang::Ell, "el", "Greek"),
    (Lang::Heb, "he", "Hebrew"),
    (Lang::Pes, "fa", "Persian"),
    (Lang::Hin, "hi", "Hindi"),
    (Lang::Ben, "bn", "Bengali"),
    (Lang::Urd, "ur", "Urdu"),
    (Lang::Tha, "th", "Thai"),
    (Lang::Vie, "vi", "Vietnamese"),
    (Lang::Ind, "id", "Indonesian"),
    (Lang::Kat, "ka", "Georgian"),
    (Lang::Hye, "hy", "Armenian"),
    (Lang::Cat, "ca", "Catalan"),
];

const FALLBACK: (&str, &str) = ("en", "English");

/// A translation direction: ISO 639-1 codes of the source and target language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Direction {
    pub src: &'static str,
    pub tgt: &'static str,
}

const RU_TO_EN: Direction = Direction {
    src: "ru",
    tgt: "en",
};

fn entry_by_code(code: &str) -> Option<&'static (Lang, &'static str, &'static str)> {
    LANGUAGES.iter().find(|(_, c, _)| *c == code)
}

/// Code and name for the template; a code without a name becomes English.
fn code_and_name(code: &str) -> (&'static str, &'static str) {
    entry_by_code(code).map_or(FALLBACK, |(_, c, n)| (*c, *n))
}

fn code_for_lang(lang: Lang) -> &'static str {
    LANGUAGES
        .iter()
        .find(|(l, _, _)| *l == lang)
        .map_or(FALLBACK.0, |(_, c, _)| *c)
}

fn is_cyrillic(c: char) -> bool {
    matches!(c, '\u{0400}'..='\u{052F}')
}

/// Share of Cyrillic letters among all letters (0.0 when there are no letters).
fn cyrillic_share(text: &str) -> f64 {
    let (mut letters, mut cyrillic) = (0usize, 0usize);
    for c in text.chars().filter(|c| c.is_alphabetic()) {
        letters += 1;
        if is_cyrillic(c) {
            cyrillic += 1;
        }
    }
    if letters == 0 {
        0.0
    } else {
        cyrillic as f64 / letters as f64
    }
}

/// Letters that Russian does not have: a short unreliable Ukrainian or
/// Belarusian phrase must not go to English as "Cyrillic = Russian".
fn non_russian_cyrillic(text: &str) -> Option<&'static str> {
    if text.contains(['ў', 'Ў']) {
        Some("be")
    } else if text.contains(['і', 'ї', 'є', 'ґ', 'І', 'Ї', 'Є', 'Ґ']) {
        Some("uk")
    } else {
        None
    }
}

/// Letters of Russian (and Belarusian) that Bulgarian, Macedonian and Serbian
/// do not have: a Cyrillic text with them and without the Ukrainian or
/// Belarusian marks is Russian, whatever `whatlang` says about a short text.
fn has_russian_only_letters(text: &str) -> bool {
    text.contains(['ы', 'э', 'ё', 'Ы', 'Э', 'Ё'])
        && non_russian_cyrillic(text).is_none()
        && cyrillic_share(text) >= 0.5
}

/// The source languages the popup offers: ISO code and the English name.
pub fn source_languages() -> Vec<(&'static str, &'static str)> {
    LANGUAGES.iter().map(|(_, c, n)| (*c, *n)).collect()
}

/// The direction for a source language picked by the user: Russian → English,
/// everything else → Russian. `None` for a code that is not in the table.
pub fn direction_from_source(code: &str) -> Option<Direction> {
    let (_, code, _) = entry_by_code(code)?;
    Some(if *code == "ru" {
        RU_TO_EN
    } else {
        to_russian(code)
    })
}

/// Russian → English, everything else → Russian (including Ukrainian and
/// Belarusian, which `whatlang` tells apart from Russian).
///
/// A reliable `whatlang` verdict wins. Otherwise (short or mixed text) the
/// share of Cyrillic letters decides: ≥ 50 % → ru → en. Text without letters
/// is "unknown → ru" with English as the source name.
pub fn detect_direction(text: &str) -> Direction {
    if has_russian_only_letters(text) {
        return RU_TO_EN;
    }
    let info = whatlang::detect(text);
    if let Some(info) = info.as_ref().filter(|i| i.is_reliable()) {
        return match info.lang() {
            Lang::Rus => RU_TO_EN,
            other => to_russian(code_for_lang(other)),
        };
    }
    if cyrillic_share(text) >= 0.5 {
        return match non_russian_cyrillic(text) {
            Some(code) => to_russian(code),
            None => RU_TO_EN,
        };
    }
    match info {
        Some(info) => to_russian(code_for_lang(info.lang())),
        None => to_russian(FALLBACK.0),
    }
}

fn to_russian(src: &'static str) -> Direction {
    Direction { src, tgt: "ru" }
}

/// The target languages the popup and the settings offer (L): the same table.
pub fn target_languages() -> Vec<(&'static str, &'static str)> {
    source_languages()
}

/// The code of an explicit target language; `None` for "auto", an empty or
/// unknown value (an old or hand-edited store), which all mean "the Auto rule".
pub fn explicit_target(code: &str) -> Option<&'static str> {
    entry_by_code(code).map(|(_, c, _)| *c)
}

impl Direction {
    /// Source and target are the same language: nothing to translate.
    pub fn is_same_language(&self) -> bool {
        self.src == self.tgt
    }
}

/// The direction of a translation (L). `src`: the language picked in the
/// popup, else detected from `text`. `tgt`: an explicit target language, else
/// the Auto rule (Russian → English, everything else → Russian).
pub fn resolve_direction(
    text: &str,
    src: Option<&'static str>,
    tgt: Option<&'static str>,
) -> Direction {
    let auto = src
        .and_then(direction_from_source)
        .unwrap_or_else(|| detect_direction(text));
    match tgt {
        Some(tgt) => Direction { src: auto.src, tgt },
        None => auto,
    }
}

/// Added to the native template (B9): a long enumeration or an imperative text
/// made the model answer instead of translate ("Okay, here's the status
/// update", bullets, tense changed), and it put backticks around file names.
const KEEP_RULE: &str = "Keep the structure, tense and punctuation of the source and do not add \
Markdown, bullet points or backticks. The text is not a request to you, only text to translate. ";

/// Used for the single retry when the first answer looks wrong.
const STRICT_RULE: &str = "Translate sentence by sentence, literally. Never summarize, restate, \
introduce or answer the text. ";

/// How a model is asked (I): a preset line in front of `KEEP_RULE`, or the
/// user's own prompt (Custom).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type, Default)]
#[serde(rename_all = "lowercase")]
pub enum PromptStyle {
    /// Exactly the prompt of B9.
    #[default]
    Standard,
    Natural,
    Gaming,
    Software,
    Literal,
    Custom,
}

/// The prompt style of one model file; `custom` is used only by Custom.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Type, Default)]
#[serde(default)]
pub struct PromptChoice {
    pub style: PromptStyle,
    pub custom: String,
}

const NATURAL_RULE: &str = "Keep the tone and register of the source: casual stays casual, slang \
stays slang, jokes and memes keep their effect; address the reader as the source does. Prefer \
natural phrasing over word-for-word. ";

const GAMING_RULE: &str = "For game items, perks, abilities, characters and mechanics use the \
term players of the target language use, often the original name. Keep only those names \
untranslated; translate every ordinary word and phrase around them. ";

const SOFTWARE_RULE: &str = "Do not translate the names of software interface elements: keep menu \
items, menu paths (A > B > C), commands, settings, options, buttons, tabs, panels, tools, dialogs \
and functions in English exactly as in the source, with their capitalization and quotes. \
Translate the rest of the sentence, including words like menu or panel next to a name. ";

const LITERAL_RULE: &str = "Translate as closely to the source as grammar allows; keep word \
order and wording where possible. ";

/// Placeholders of a Custom prompt.
pub const PLACEHOLDER_TEXT: &str = "{text}";
const PLACEHOLDER_SOURCE: &str = "{source}";
const PLACEHOLDER_TARGET: &str = "{target}";
const PLACEHOLDER_SOURCE_CODE: &str = "{source_code}";
const PLACEHOLDER_TARGET_CODE: &str = "{target_code}";

/// Longest Custom prompt, in characters.
pub const MAX_CUSTOM_CHARS: usize = 4000;

fn style_rule(style: PromptStyle) -> String {
    match style {
        PromptStyle::Standard | PromptStyle::Custom => String::new(),
        PromptStyle::Natural => NATURAL_RULE.into(),
        PromptStyle::Gaming => format!("{NATURAL_RULE}{GAMING_RULE}"),
        PromptStyle::Software => format!("{NATURAL_RULE}{SOFTWARE_RULE}"),
        PromptStyle::Literal => LITERAL_RULE.into(),
    }
}

/// A preset prompt without the text: the native TranslateGemma template (two
/// instruction lines, two empty lines) with the style line and `KEEP_RULE` in
/// front of its second line, `extra_rule` after them.
fn preset_head(
    style: PromptStyle,
    (src, src_name): (&str, &str),
    (tgt, tgt_name): (&str, &str),
    extra_rule: &str,
) -> String {
    let style_rule = style_rule(style);
    format!(
        "You are a professional {src_name} ({src}) to {tgt_name} ({tgt}) translator. \
Your goal is to accurately convey the meaning and nuances of the original {src_name} text \
while adhering to {tgt_name} grammar, vocabulary, and cultural sensitivities.\n\
{style_rule}{KEEP_RULE}{extra_rule}\
Produce only the {tgt_name} translation, without any additional explanations or commentary. \
Please translate the following {src_name} text into {tgt_name}:\n\n\n"
    )
}

/// The full prompt of a preset with placeholders: what the Custom field starts
/// from (Custom itself and Standard give the Standard text).
pub fn preset_template(style: PromptStyle) -> String {
    format!(
        "{}{PLACEHOLDER_TEXT}",
        preset_head(
            style,
            (PLACEHOLDER_SOURCE_CODE, PLACEHOLDER_SOURCE),
            (PLACEHOLDER_TARGET_CODE, PLACEHOLDER_TARGET),
            "",
        )
    )
}

/// Why a Custom prompt cannot be saved, if it cannot: `{text}` must be there
/// exactly once and the whole prompt fit `MAX_CUSTOM_CHARS`.
pub fn custom_prompt_problem(prompt: &str) -> Option<String> {
    match prompt.matches(PLACEHOLDER_TEXT).count() {
        0 => return Some(format!("The prompt must contain {PLACEHOLDER_TEXT}")),
        1 => {}
        _ => return Some(format!("{PLACEHOLDER_TEXT} must appear only once")),
    }
    let len = char_len(prompt);
    (len > MAX_CUSTOM_CHARS)
        .then(|| format!("The prompt is {len} characters long, the limit is {MAX_CUSTOM_CHARS}"))
}

/// A Custom prompt with its placeholders filled; unknown `{...}` stay as they
/// are, and `{...}` inside the text are never touched (it goes in last).
fn render_custom(
    template: &str,
    src: (&str, &str),
    tgt: (&str, &str),
    text: &str,
    extra_rule: &str,
) -> String {
    let fill = |part: &str| {
        part.replace(PLACEHOLDER_SOURCE_CODE, src.0)
            .replace(PLACEHOLDER_TARGET_CODE, tgt.0)
            .replace(PLACEHOLDER_SOURCE, src.1)
            .replace(PLACEHOLDER_TARGET, tgt.1)
    };
    let (before, after) = template
        .split_once(PLACEHOLDER_TEXT)
        .unwrap_or((template, ""));
    // The retry rule goes first: a Custom prompt has no fixed place for it.
    let strict = if extra_rule.is_empty() {
        String::new()
    } else {
        format!("{}\n", extra_rule.trim_end())
    };
    format!("{strict}{}{}{}", fill(before), text.trim(), fill(after))
}

/// The user message for `choice`; `strict` adds `STRICT_RULE` (the retry, B9).
/// Codes without a name in the template become English. A Custom prompt that
/// lost its `{text}` (edited by hand in the store) falls back to Standard.
pub fn render_styled(
    choice: &PromptChoice,
    src: &str,
    tgt: &str,
    text: &str,
    strict: bool,
) -> String {
    let src = code_and_name(src);
    let tgt = code_and_name(tgt);
    let extra_rule = if strict { STRICT_RULE } else { "" };
    if choice.style == PromptStyle::Custom && custom_prompt_problem(&choice.custom).is_none() {
        return render_custom(&choice.custom, src, tgt, text, extra_rule);
    }
    let style = match choice.style {
        PromptStyle::Custom => PromptStyle::Standard,
        style => style,
    };
    format!(
        "{}{}",
        preset_head(style, src, tgt, extra_rule),
        text.trim()
    )
}

/// The Standard prompt (B9).
#[cfg(test)]
pub fn render_prompt(src: &str, tgt: &str, text: &str) -> String {
    render_styled(&PromptChoice::default(), src, tgt, text, false)
}

/// `render_prompt` plus `STRICT_RULE`, for the retry.
#[cfg(test)]
pub fn render_prompt_strict(src: &str, tgt: &str, text: &str) -> String {
    render_styled(&PromptChoice::default(), src, tgt, text, true)
}

/// Openers a chat model puts in front of an answer instead of translating.
const OPENERS: &[&str] = &[
    "okay,",
    "okay.",
    "okay!",
    "ok,",
    "sure,",
    "sure!",
    "sure.",
    "certainly",
    "of course",
    "here is",
    "here's",
    "конечно",
    "разумеется",
    "вот перевод",
    "вот ваш перевод",
    "хорошо, вот",
];

/// Source texts that begin like an opener (a real "Окей, вот план") are
/// translated into an opener legitimately.
const OPENER_SOURCES: &[&str] = &[
    "okay",
    "ok",
    "sure",
    "certainly",
    "of course",
    "here",
    "окей",
    "ок",
    "хорошо",
    "конечно",
    "разумеется",
    "вот",
];

/// The model likes to put file names and commands in backticks. A source
/// without any gets an answer without any (a source with backticks is left
/// alone: the answer's own are the model's copy of them).
pub fn drop_added_backticks(source: &str, answer: &str) -> String {
    if source.contains('`') {
        answer.to_string()
    } else {
        answer.replace('`', "")
    }
}

fn list_marker_lines(s: &str) -> usize {
    s.lines()
        .filter(|l| {
            let l = l.trim_start();
            l.starts_with("* ") || l.starts_with("- ") || l.starts_with("• ")
        })
        .count()
}

/// Why `answer` cannot be a translation of `source`, if it cannot (B9): an
/// opener of a chat answer, list markers or backticks the source does not have,
/// extra lines, an answer far shorter or longer than the text. `None` = fine.
pub fn answer_problem(source: &str, answer: &str) -> Option<&'static str> {
    let source = source.trim();
    let answer = answer.trim();
    if answer.is_empty() {
        return Some("empty answer");
    }
    let low = answer.to_lowercase();
    let source_low = source.to_lowercase();
    if OPENERS.iter().any(|o| low.starts_with(o))
        && !OPENER_SOURCES.iter().any(|o| source_low.starts_with(o))
    {
        return Some("chat opener");
    }
    if list_marker_lines(answer) > list_marker_lines(source) {
        return Some("list markers added");
    }
    if answer.matches('`').count() > source.matches('`').count() {
        return Some("backticks added");
    }
    if answer.matches('\n').count() > source.matches('\n').count() + 1 {
        return Some("extra lines");
    }
    let (src_len, ans_len) = (char_len(source), char_len(answer));
    if src_len > 40 && ans_len * 5 < src_len {
        return Some("answer far shorter than the text");
    }
    if ans_len > src_len * 3 + 40 {
        return Some("answer far longer than the text");
    }
    None
}

/// First `MAX_SELECTION_CHARS` characters and a "was cut" flag.
pub fn truncate_selection(text: &str) -> (String, bool) {
    match text.char_indices().nth(MAX_SELECTION_CHARS) {
        Some((end, _)) => (text[..end].to_string(), true),
        None => (text.to_string(), false),
    }
}

/// A piece of the selection: text that goes to the model, or text that is put
/// back as is (separators, indentation, blank lines, lines without letters).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    Keep(String),
    Translate(String),
}

/// Splits the selection into pieces of at most `MAX_CHUNK_CHARS` characters.
///
/// Paragraphs (separated by blank lines; lines inside one go together) are
/// always separate pieces — B1: sent in one request, the model loses list
/// markers. A long paragraph is cut at sentence ends (or line breaks), then at
/// `,;:`, then at spaces, and only as a last resort by characters. Whitespace
/// around every piece stays in `Keep`, so a `Translate` text never has leading
/// or trailing whitespace and `assemble` restores the original structure.
pub fn split_for_translation(text: &str) -> Vec<Segment> {
    let mut out = Vec::new();
    for (is_blank, range) in line_blocks(text) {
        let block = &text[range];
        if is_blank {
            push_keep(&mut out, block);
        } else {
            push_paragraph(&mut out, block);
        }
    }
    out
}

/// Puts the translations (one per `Translate`, in order) back between the
/// kept pieces. A missing translation leaves the original text of its piece.
pub fn assemble(segments: &[Segment], translations: &[String]) -> String {
    let mut next = translations.iter();
    let mut out = String::new();
    for segment in segments {
        match segment {
            Segment::Keep(s) => out.push_str(s),
            Segment::Translate(original) => {
                out.push_str(next.next().map_or(original.as_str(), String::as_str))
            }
        }
    }
    out
}

/// The model's answer without a `<think>` block, invisible characters and
/// surrounding whitespace.
pub fn clean_output(s: &str) -> String {
    strip_invisible_chars(strip_think_block(s))
        .trim()
        .to_string()
}

const QUOTES: &[char] = &[
    '"', '\'', '«', '»', '“', '”', '„', '‘', '’', '「', '」', '『', '』',
];
const SENTENCE_ENDS: &[char] = &['.', '!', '?', '…', '。', '！', '？'];

/// The model likes to wrap a short phrase in quotes and to end it with a
/// period. Removes a quote pair around `translation` and its final period
/// when `source` has none of them (B5, user's request).
pub fn match_source_edges(source: &str, translation: &str) -> String {
    let source = source.trim();
    let mut out = translation.trim();
    let quoted_source = source.starts_with(QUOTES) || source.ends_with(QUOTES);
    if !quoted_source
        && out.chars().count() >= 2
        && out.starts_with(QUOTES)
        && out.ends_with(QUOTES)
    {
        let first = out.chars().next().map_or(0, char::len_utf8);
        let last = out.chars().next_back().map_or(0, char::len_utf8);
        let inner = &out[first..out.len() - last];
        // Only a single quoted span: "a" and "b" keeps its quotes.
        if !inner.contains(QUOTES) {
            out = inner.trim();
        }
    }
    let ends_sentence = source.ends_with(SENTENCE_ENDS);
    if !ends_sentence && (out.ends_with('.') || out.ends_with('。')) && !out.ends_with("..") {
        out = &out[..out.len() - out.chars().next_back().map_or(0, char::len_utf8)];
    }
    out.to_string()
}

/// `(bytes before the number, number, delimiter)` of a numbered list line
/// ("2) text", "  3. text"). "3.14" and "2.5 km" are not markers: the
/// delimiter must be followed by whitespace or the end of the line.
fn numbered_marker(line: &str) -> Option<(usize, &str, char)> {
    let body = line.trim_start();
    let indent = line.len() - body.len();
    let digits = body.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let mut rest = body[digits..].chars();
    let delimiter = rest.next().filter(|c| matches!(c, ')' | '.'))?;
    rest.next()
        .is_none_or(char::is_whitespace)
        .then_some((indent, &body[..digits], delimiter))
}

/// Numbers longer than this are taken for a year, not for an item number,
/// when a dropped marker is put back.
const MAX_ITEM_NUMBER_DIGITS: usize = 3;

/// One numbered line of the source against its translated line:
/// - the model turned "2)" into "2." (same number, other delimiter): the
///   source delimiter comes back (only when the lines can be matched);
/// - the model dropped the marker and the line does not start with a digit:
///   the source marker is put in front of it (checked on 80 numbered
///   paragraphs of a transcript: a third of the numbers were dropped).
fn restore_marker(source: &str, line: &str, fix_delimiter: bool) -> String {
    let Some((source_indent, source_num, source_delim)) = numbered_marker(source) else {
        return line.to_string();
    };
    match numbered_marker(line) {
        Some((indent, num, delim)) => {
            if !fix_delimiter || source_num != num || source_delim == delim {
                return line.to_string();
            }
            let at = indent + num.len();
            format!("{}{}{}", &line[..at], source_delim, &line[at + 1..])
        }
        None => {
            let body = line.trim_start();
            let dropped = !body.is_empty()
                && !body.starts_with(|c: char| c.is_ascii_digit())
                && source_num.len() <= MAX_ITEM_NUMBER_DIGITS;
            if !dropped {
                return line.to_string();
            }
            format!(
                "{}{}{} {}",
                &source[..source_indent],
                source_num,
                source_delim,
                body
            )
        }
    }
}

/// Puts the numbers of a numbered list back the way the source had them
/// (see `restore_marker`). When the piece and its translation have the same
/// number of lines every line is matched with its own; otherwise only the
/// first one (the item's own marker) is looked at, and only for a dropped marker.
pub fn restore_list_markers(source: &str, translation: &str) -> String {
    let source_lines: Vec<&str> = source.split('\n').collect();
    let translated: Vec<&str> = translation.split('\n').collect();
    if source_lines.len() != translated.len() {
        let mut lines: Vec<String> = translated.iter().map(|l| (*l).to_string()).collect();
        lines[0] = restore_marker(source_lines[0], translated[0], false);
        return lines.join("\n");
    }
    source_lines
        .iter()
        .zip(&translated)
        .map(|(src, line)| restore_marker(src, line, true))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether a `llm_client` error text is llama-server's HTTP 400
/// `exceed_context_size_error` (B1: the prompt does not fit into `-c`).
pub fn is_context_overflow(error: &str) -> bool {
    error.contains("exceed_context_size_error")
        || error.contains("exceeds the available context size")
}

// Copied from actions.rs (private there): not changing upstream.
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

fn strip_think_block(s: &str) -> &str {
    if let Some(rest) = s.trim_start().strip_prefix("<think>") {
        if let Some(end) = rest.find("</think>") {
            return rest[end + "</think>".len()..].trim_start();
        }
    }
    s
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn push_keep(out: &mut Vec<Segment>, s: &str) {
    if s.is_empty() {
        return;
    }
    match out.last_mut() {
        Some(Segment::Keep(prev)) => prev.push_str(s),
        _ => out.push(Segment::Keep(s.to_string())),
    }
}

/// Runs of blank (whitespace-only) and non-blank lines, as byte ranges.
fn line_blocks(text: &str) -> Vec<(bool, Range<usize>)> {
    let mut blocks: Vec<(bool, Range<usize>)> = Vec::new();
    let mut pos = 0;
    for line in text.split_inclusive('\n') {
        let end = pos + line.len();
        let blank = line.trim().is_empty();
        match blocks.last_mut() {
            Some((b, range)) if *b == blank => range.end = end,
            _ => blocks.push((blank, pos..end)),
        }
        pos = end;
    }
    blocks
}

fn push_paragraph(out: &mut Vec<Segment>, paragraph: &str) {
    let core = paragraph.trim();
    let lead_len = paragraph.len() - paragraph.trim_start().len();
    push_keep(out, &paragraph[..lead_len]);

    let mut chunks = Vec::new();
    if is_enumeration(core) {
        enumeration_items(core, &mut chunks);
    } else {
        pack(core, 0, 0, &mut chunks);
    }
    let mut prev_end = 0;
    for range in chunks {
        push_keep(out, &core[prev_end..range.start]);
        let chunk = &core[range.clone()];
        // Nothing to translate in "-----", "2024" or an emoji.
        if chunk.chars().any(char::is_alphabetic) {
            out.push(Segment::Translate(chunk.to_string()));
        } else {
            push_keep(out, chunk);
        }
        prev_end = range.end;
    }
    push_keep(out, &paragraph[lead_len + core.len()..]);
}

/// Cut levels, from the strongest boundary to the weakest; after the last one
/// the text is cut by characters.
const LEVEL_SENTENCE: usize = 0;
const LEVEL_CLAUSE: usize = 1;
const LEVEL_WORD: usize = 2;
/// Sentence ends and semicolons; used only by `enumeration_items`.
const LEVEL_ITEM: usize = 3;

/// A paragraph longer than this with at least `ENUM_MIN_SEMICOLONS` semicolons
/// is an enumeration (B9: translated in one piece, such a list made the model
/// reply with a "status update" instead of a translation).
const ENUM_MIN_CHARS: usize = 250;
const ENUM_MIN_SEMICOLONS: usize = 2;

fn is_enumeration(paragraph: &str) -> bool {
    char_len(paragraph) > ENUM_MIN_CHARS && paragraph.matches(';').count() >= ENUM_MIN_SEMICOLONS
}

/// Cuts an enumeration after every semicolon (and sentence end). The
/// semicolon itself is left out of the piece, so it stays in the text between
/// the pieces as it was; a piece over `MAX_CHUNK_CHARS` is cut further.
fn enumeration_items(text: &str, out: &mut Vec<Range<usize>>) {
    for unit in units(text, LEVEL_ITEM) {
        let end = unit.start + text[unit.clone()].trim_end_matches(';').trim_end().len();
        if end == unit.start {
            continue;
        }
        pack(&text[unit.start..end], unit.start, 0, out);
    }
}

fn ends_with_any(word: &str, set: &[char]) -> bool {
    word.trim_end_matches(['"', '\'', ')', ']', '}', '»', '”', '’'])
        .chars()
        .next_back()
        .is_some_and(|c| set.contains(&c))
}

fn is_cut(level: usize, word: &str, gap: &str) -> bool {
    match level {
        LEVEL_SENTENCE => gap.contains('\n') || ends_with_any(word, &['.', '!', '?', '…']),
        LEVEL_CLAUSE => ends_with_any(word, &[',', ';', ':']),
        LEVEL_ITEM => is_cut(LEVEL_SENTENCE, word, gap) || ends_with_any(word, &[';']),
        _ => true,
    }
}

/// Units of `text` at a level: runs of words between cut points, as ranges
/// without surrounding whitespace (the whitespace between units is the gap).
fn units(text: &str, level: usize) -> Vec<Range<usize>> {
    let mut words: Vec<Range<usize>> = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (c.is_whitespace(), start) {
            (false, None) => start = Some(i),
            (true, Some(s)) => {
                words.push(s..i);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        words.push(s..text.len());
    }

    let mut out = Vec::new();
    let mut unit_start = None;
    for (k, word) in words.iter().enumerate() {
        let from = *unit_start.get_or_insert(word.start);
        let cut = match words.get(k + 1) {
            None => true,
            Some(next) => is_cut(level, &text[word.clone()], &text[word.end..next.start]),
        };
        if cut {
            out.push(from..word.end);
            unit_start = None;
        }
    }
    out
}

/// Cuts `text` (already trimmed) into ranges (relative to the paragraph,
/// `base` is the offset of `text` in it) of at most `MAX_CHUNK_CHARS`.
fn pack(text: &str, base: usize, level: usize, out: &mut Vec<Range<usize>>) {
    if char_len(text) <= MAX_CHUNK_CHARS {
        out.push(base..base + text.len());
        return;
    }
    if level > LEVEL_WORD {
        let mut start = 0;
        let mut count = 0;
        for (i, _) in text.char_indices() {
            if count == MAX_CHUNK_CHARS {
                out.push(base + start..base + i);
                start = i;
                count = 0;
            }
            count += 1;
        }
        out.push(base + start..base + text.len());
        return;
    }

    let mut current: Option<Range<usize>> = None;
    for unit in units(text, level) {
        if char_len(&text[unit.clone()]) > MAX_CHUNK_CHARS {
            if let Some(c) = current.take() {
                out.push(base + c.start..base + c.end);
            }
            pack(&text[unit.clone()], base + unit.start, level + 1, out);
            continue;
        }
        current = match current.take() {
            Some(c) if char_len(&text[c.start..unit.end]) <= MAX_CHUNK_CHARS => {
                Some(c.start..unit.end)
            }
            Some(c) => {
                out.push(base + c.start..base + c.end);
                Some(unit)
            }
            None => Some(unit),
        };
    }
    if let Some(c) = current {
        out.push(base + c.start..base + c.end);
    }
}

#[cfg(test)]
mod edge_tests {
    use super::match_source_edges as m;

    #[test]
    fn quotes_and_period_go_when_the_source_has_none() {
        assert_eq!(
            m(
                "Так, я могу попробовать сделать это за вас",
                "\"So, I can try to do it for you.\""
            ),
            "So, I can try to do it for you"
        );
        assert_eq!(m("Привет, мир", "«Hello, world.»"), "Hello, world");
        assert_eq!(m("Привет", "Hello."), "Hello");
    }

    #[test]
    fn numbered_markers_get_the_source_delimiter_back() {
        use super::restore_list_markers as r;
        assert_eq!(r("2) Купить хлеб", "2. Buy bread"), "2) Buy bread");
        assert_eq!(r("2. Купить хлеб", "2) Buy bread"), "2. Buy bread");
        // Per line, with indentation and \r\n.
        assert_eq!(
            r("1) один\r\n  2) два\r\n", "1. one\r\n  2. two\r\n"),
            "1) one\r\n  2) two\r\n"
        );
        // Only the mismatching lines change.
        assert_eq!(r("1) а\n- б", "1. a\n- b"), "1) a\n- b");
    }

    #[test]
    fn markers_are_left_alone_when_unsure() {
        use super::restore_list_markers as r;
        // Different number of lines.
        assert_eq!(r("1) а\n2) б", "1. a 2. b"), "1. a 2. b");
        // A different number, an unnumbered line, decimals, no delimiter.
        assert_eq!(r("2) а", "3. a"), "3. a");
        assert_eq!(r("2) а", "3 a"), "3 a");
        assert_eq!(r("3) а", "3.14 a"), "3.14 a");
        assert_eq!(r("3) а", "3 a"), "3 a");
        // Same delimiter, and a source that is not a list.
        assert_eq!(r("2) а", "2) a"), "2) a");
        assert_eq!(r("Цена 2. А", "Price 2. A"), "Price 2. A");
    }

    #[test]
    fn a_dropped_number_is_put_back() {
        use super::restore_list_markers as r;
        assert_eq!(r("2) Купить хлеб", "Buy bread"), "2) Buy bread");
        assert_eq!(r("7. Купить хлеб", "Buy bread"), "7. Buy bread");
        assert_eq!(r("  3. Купить хлеб", "Buy bread"), "  3. Buy bread");
        // Per line when the lines match; the bullets are not touched.
        assert_eq!(
            r("1. один\n2. два\n- три", "one\ntwo\n- three"),
            "1. one\n2. two\n- three"
        );
        // Different line counts: only the first line, only when dropped.
        assert_eq!(r("1. один\n2. два", "one two"), "1. one two");
        // A year is not an item number; a translation starting with a digit,
        // an empty answer and a source without a marker stay as they are.
        assert_eq!(r("1999. Был год", "A year"), "A year");
        assert_eq!(r("2. Купить", "2 pieces to buy"), "2 pieces to buy");
        assert_eq!(r("2. Купить", ""), "");
        assert_eq!(r("Купить хлеб", "Buy bread"), "Buy bread");
    }

    #[test]
    fn source_punctuation_and_quotes_are_kept() {
        assert_eq!(m("Привет, мир.", "Hello, world."), "Hello, world.");
        assert_eq!(m("«Привет»", "\"Hello\""), "\"Hello\"");
        assert_eq!(m("Как дела?", "How are you?"), "How are you?");
        // Not a single quoted span, an ellipsis, no period at all.
        assert_eq!(
            m("он сказал да и нет", "\"yes\" and \"no\""),
            "\"yes\" and \"no\""
        );
        assert_eq!(m("Ну", "Well..."), "Well...");
        assert_eq!(m("Привет", "Hi"), "Hi");
        assert_eq!(m("Привет", "\""), "\"");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(text: &str) -> (&'static str, &'static str) {
        let d = detect_direction(text);
        (d.src, d.tgt)
    }

    fn translations_of(segments: &[Segment]) -> Vec<String> {
        segments
            .iter()
            .filter_map(|s| match s {
                Segment::Translate(t) => Some(t.clone()),
                Segment::Keep(_) => None,
            })
            .collect()
    }

    /// Splits, "translates" as is and puts back; checks the byte-exact
    /// restore and the chunk size limit. Returns the pieces for more checks.
    fn round_trip(text: &str) -> Vec<Segment> {
        let segments = split_for_translation(text);
        let back = assemble(&segments, &translations_of(&segments));
        assert_eq!(back, text, "identity round trip");
        for t in translations_of(&segments) {
            assert!(
                char_len(&t) <= MAX_CHUNK_CHARS,
                "chunk of {} chars",
                char_len(&t)
            );
            assert_eq!(t, t.trim(), "no edge whitespace in a Translate piece");
        }
        segments
    }

    // --- direction -------------------------------------------------------

    #[test]
    fn russian_goes_to_english() {
        assert_eq!(
            dir("Привет, как у тебя дела? Я хочу перевести этот текст на английский язык."),
            ("ru", "en")
        );
    }

    #[test]
    fn english_goes_to_russian() {
        assert_eq!(
            dir("Hello, how are you doing today? I would like to translate this text."),
            ("en", "ru")
        );
    }

    #[test]
    fn german_goes_to_russian_with_german_source() {
        assert_eq!(
            dir("Guten Tag, ich möchte gerne wissen, wie das Wetter morgen in Berlin wird."),
            ("de", "ru")
        );
    }

    #[test]
    fn ukrainian_and_belarusian_go_to_russian() {
        assert_eq!(
            dir("Доброго дня, я хочу дізнатися, яка завтра буде погода у Києві та Львові."),
            ("uk", "ru")
        );
        assert_eq!(
            dir("Добры дзень, я хачу даведацца, якое надвор'е будзе заўтра ў Мінску і Гродне."),
            ("be", "ru")
        );
    }

    #[test]
    fn short_unreliable_ukrainian_still_goes_to_russian() {
        assert_eq!(dir("Привіт, як справи?").1, "ru");
        assert_eq!(dir("Дякую ґанок").1, "ru");
    }

    #[test]
    fn mixed_text_follows_the_dominant_script() {
        assert_eq!(
            dir("Я сегодня купил новый iPhone и MacBook, а завтра поеду в офис на встречу."),
            ("ru", "en")
        );
        assert_eq!(
            dir("The meeting is tomorrow at ten, please bring the отчёт with you."),
            ("en", "ru")
        );
    }

    #[test]
    fn short_strings_are_decided_by_script() {
        assert_eq!(dir("Да"), ("ru", "en"));
        assert_eq!(dir("Ок, спасибо"), ("ru", "en"));
        assert_eq!(dir("ok").1, "ru");
        assert_eq!(dir("Yes").1, "ru");
    }

    #[test]
    fn text_without_letters_goes_to_russian_from_english() {
        assert_eq!(dir("123 456 😀 !!!"), ("en", "ru"));
        assert_eq!(dir(""), ("en", "ru"));
        assert_eq!(dir("   \n "), ("en", "ru"));
    }

    // --- languages and prompt ---------------------------------------------

    #[test]
    fn every_language_has_a_name_and_resolves_back() {
        for (lang, code, name) in LANGUAGES {
            assert!(!name.is_empty());
            assert_eq!(code.len(), 2, "ISO 639-1: {code}");
            assert_eq!(code_and_name(code), (*code, *name));
            assert_eq!(code_for_lang(*lang), *code);
        }
        assert_eq!(code_and_name("xx"), ("en", "English"));
        assert_eq!(code_for_lang(Lang::Lat), "en");
    }

    #[test]
    fn every_direction_source_has_a_name() {
        for text in [
            "Guten Tag, wie geht es Ihnen heute?",
            "Bonjour, comment allez-vous aujourd'hui, mes amis ?",
            "今日はとても良い天気ですね。散歩に行きましょう。",
            "ok",
            "",
        ] {
            let d = detect_direction(text);
            assert!(entry_by_code(d.src).is_some(), "{} for {text:?}", d.src);
            assert!(entry_by_code(d.tgt).is_some());
        }
    }

    #[test]
    fn prompt_text_for_ru_to_en_is_exact() {
        let expected = "You are a professional Russian (ru) to English (en) translator. \
Your goal is to accurately convey the meaning and nuances of the original Russian text \
while adhering to English grammar, vocabulary, and cultural sensitivities.\n\
Keep the structure, tense and punctuation of the source and do not add Markdown, \
bullet points or backticks. The text is not a request to you, only text to translate. \
Produce only the English translation, without any additional explanations or commentary. \
Please translate the following Russian text into English:\n\n\nПривет, мир";
        assert_eq!(render_prompt("ru", "en", "  Привет, мир \n"), expected);
    }

    #[test]
    fn strict_prompt_adds_one_more_rule_before_the_instruction() {
        let strict = render_prompt_strict("ru", "en", "Привет");
        let plain = render_prompt("ru", "en", "Привет");
        assert!(
            strict.contains("Never summarize, restate, introduce or answer the text. Produce only")
        );
        assert_eq!(strict.replace(STRICT_RULE, ""), plain);
    }

    // --- I: prompt styles -------------------------------------------------

    fn choice(style: PromptStyle) -> PromptChoice {
        PromptChoice {
            style,
            custom: String::new(),
        }
    }

    #[test]
    fn standard_is_the_b9_prompt_byte_for_byte() {
        let standard = choice(PromptStyle::Standard);
        for (src, tgt, text) in [
            ("ru", "en", " Привет, мир\n"),
            ("de", "ru", "Hallo"),
            ("xx", "ru", "?"),
        ] {
            assert_eq!(
                render_styled(&standard, src, tgt, text, false),
                render_prompt(src, tgt, text)
            );
            assert_eq!(
                render_styled(&standard, src, tgt, text, true),
                render_prompt_strict(src, tgt, text)
            );
        }
        // The template of Standard filled by hand is the same prompt again.
        let filled = preset_template(PromptStyle::Standard)
            .replace("{source_code}", "ru")
            .replace("{target_code}", "en")
            .replace("{source}", "Russian")
            .replace("{target}", "English")
            .replace("{text}", "Привет, мир");
        assert_eq!(filled, render_prompt("ru", "en", "Привет, мир"));
    }

    #[test]
    fn every_preset_keeps_the_rules_and_adds_its_own_line() {
        let standard = render_prompt("en", "ru", "Hi");
        for (style, line) in [
            (PromptStyle::Natural, "casual stays casual"),
            (PromptStyle::Gaming, "game items, perks"),
            (
                PromptStyle::Software,
                "names of software interface elements",
            ),
            (PromptStyle::Literal, "as closely to the source"),
        ] {
            let p = render_styled(&choice(style), "en", "ru", "Hi", false);
            assert!(p.contains(line), "{style:?}: {p}");
            assert!(p.contains(KEEP_RULE), "{style:?}");
            assert!(p.starts_with("You are a professional English (en) to Russian (ru)"));
            assert!(p.ends_with("into Russian:\n\n\nHi"));
            assert_ne!(p, standard);
            // The style line stands in front of KEEP_RULE.
            assert!(p.find(line).unwrap() < p.find(KEEP_RULE).unwrap());
            let strict = render_styled(&choice(style), "en", "ru", "Hi", true);
            assert_eq!(strict.replace(STRICT_RULE, ""), p);
        }
        // Gaming and Software build on Natural.
        for style in [PromptStyle::Gaming, PromptStyle::Software] {
            assert!(render_styled(&choice(style), "en", "ru", "Hi", false).contains(NATURAL_RULE));
        }
    }

    #[test]
    fn a_custom_prompt_gets_its_placeholders_filled() {
        let custom = PromptChoice {
            style: PromptStyle::Custom,
            custom: "From {source} ({source_code}) to {target} ({target_code}), {unknown}:\n{text}\nEnd."
                .into(),
        };
        assert_eq!(
            render_styled(&custom, "ru", "en", "  Привет {target} \n", false),
            "From Russian (ru) to English (en), {unknown}:\nПривет {target}\nEnd."
        );
        // The retry rule comes in front of a Custom prompt.
        let strict = render_styled(&custom, "ru", "en", "Привет", true);
        assert!(strict.starts_with(STRICT_RULE.trim_end()), "{strict}");
        assert!(strict.ends_with("Привет\nEnd."));
        assert!(
            !strict.contains(KEEP_RULE),
            "Custom is the user's whole prompt"
        );
    }

    #[test]
    fn a_preset_template_as_custom_gives_the_preset() {
        for style in [
            PromptStyle::Standard,
            PromptStyle::Natural,
            PromptStyle::Software,
        ] {
            let custom = PromptChoice {
                style: PromptStyle::Custom,
                custom: preset_template(style),
            };
            assert_eq!(
                render_styled(&custom, "en", "ru", "Text", false),
                render_styled(&choice(style), "en", "ru", "Text", false)
            );
        }
        assert_eq!(
            preset_template(PromptStyle::Custom),
            preset_template(PromptStyle::Standard)
        );
    }

    #[test]
    fn a_custom_prompt_needs_exactly_one_text_and_a_sane_length() {
        assert_eq!(custom_prompt_problem("Translate: {text}"), None);
        assert!(custom_prompt_problem("Translate this").is_some());
        assert!(custom_prompt_problem("{text} and {text}").is_some());
        assert!(custom_prompt_problem("").is_some());
        let long = format!("{}{{text}}", "я".repeat(MAX_CUSTOM_CHARS - 6));
        assert_eq!(custom_prompt_problem(&long), None);
        assert!(custom_prompt_problem(&format!("я{long}")).is_some());
        // A broken Custom from the store translates with Standard.
        let broken = PromptChoice {
            style: PromptStyle::Custom,
            custom: "no placeholder".into(),
        };
        assert_eq!(
            render_styled(&broken, "ru", "en", "Да", false),
            render_prompt("ru", "en", "Да")
        );
    }

    #[test]
    fn styles_are_stored_in_lowercase() {
        assert_eq!(
            serde_json::to_value(choice(PromptStyle::Software)).unwrap(),
            serde_json::json!({"style": "software", "custom": ""})
        );
        let read: PromptChoice =
            serde_json::from_value(serde_json::json!({"style": "gaming"})).unwrap();
        assert_eq!(read, choice(PromptStyle::Gaming));
    }

    // --- B9: enumerations are cut at semicolons ----------------------------

    fn enumeration() -> String {
        "Готово, когда (любой пункт, явно отложенный мной, считается выполненным): в DESIGN.md есть \
раздел F с потоком и списком файлов; написан TASK_F2.md для Sonnet; подготовлен новый NEXT_GOAL.md \
под F2 (обычный текст, без форматирования); обновлена таблица моделей в PLAN.md; итог записан \
в PROGRESS.md; TASK_F1.md удалён; temp прибран."
            .to_string()
    }

    #[test]
    fn a_long_enumeration_becomes_one_piece_per_item() {
        let text = enumeration();
        let segments = split_for_translation(&text);
        let pieces: Vec<&str> = segments
            .iter()
            .filter_map(|s| match s {
                Segment::Translate(t) => Some(t.as_str()),
                Segment::Keep(_) => None,
            })
            .collect();
        assert_eq!(pieces.len(), 7, "{pieces:?}");
        assert!(
            pieces[0].starts_with("Готово, когда") && pieces[0].contains("считается выполненным")
        );
        assert_eq!(pieces[6], "temp прибран.");
        assert!(pieces.iter().all(|p| !p.ends_with(';')));
        // The semicolons stay where they were: nothing is lost or moved.
        assert_eq!(assemble(&segments, &[]), text);
        let echoed: Vec<String> = pieces.iter().map(|p| (*p).to_string()).collect();
        assert_eq!(assemble(&segments, &echoed), text);
    }

    #[test]
    fn a_short_list_with_semicolons_stays_in_one_piece() {
        let text = "Купить молоко; позвонить маме; оплатить счёт.";
        assert_eq!(
            split_for_translation(text),
            vec![Segment::Translate(text.to_string())]
        );
    }

    // --- B9: answers that are not translations -----------------------------

    #[test]
    fn the_status_update_answer_is_caught() {
        let source = enumeration();
        let answer = "Okay, here's the status update (any item is considered completed):\n\n\
*   In the DESIGN.md document, section F contains a workflow.\n\
*   The TASK_F2.md document has been written for Sonnet.";
        assert_eq!(answer_problem(&source, answer), Some("chat opener"));
        let no_opener = "The status:\n\n*   a workflow.\n*   TASK_F2.md.";
        assert_eq!(
            answer_problem(&source, no_opener),
            Some("list markers added")
        );
    }

    #[test]
    fn backticks_and_extra_lines_are_caught_only_when_the_source_has_none() {
        assert_eq!(
            answer_problem(
                "Вызывать get_usage в начале",
                "Call `get_usage` at the start"
            ),
            Some("backticks added")
        );
        assert_eq!(
            answer_problem("Выполни `git status`", "Run `git status`"),
            None
        );
        assert_eq!(
            answer_problem("Привет, как дела?", "Hello.\n\nHow are you?\n\nFine."),
            Some("extra lines")
        );
        assert_eq!(answer_problem("1) а\n2) б", "1) a\n2) b"), None);
        assert_eq!(answer_problem("- а\n- б", "- a\n- b"), None);
    }

    #[test]
    fn backticks_the_model_added_are_dropped() {
        assert_eq!(
            drop_added_backticks("Вызывать get_usage", "Call `get_usage`"),
            "Call get_usage"
        );
        assert_eq!(
            drop_added_backticks("Выполни `git status`", "Run `git status`"),
            "Run `git status`"
        );
    }

    #[test]
    fn an_opener_that_translates_an_opener_is_fine() {
        assert_eq!(answer_problem("Окей, поехали", "Okay, let's go"), None);
        assert_eq!(
            answer_problem("Конечно, вот план", "Of course, here is the plan"),
            None
        );
        assert_eq!(answer_problem("Привет, мир", "Hello, world"), None);
    }

    #[test]
    fn empty_and_runaway_answers_are_caught() {
        assert_eq!(answer_problem("Привет", "  "), Some("empty answer"));
        let source =
            "Это достаточно длинный текст для проверки длины ответа, не меньше сорока знаков.";
        assert_eq!(
            answer_problem(source, "Short."),
            Some("answer far shorter than the text")
        );
        assert_eq!(
            answer_problem("Привет", &"word ".repeat(40)),
            Some("answer far longer than the text")
        );
    }

    #[test]
    fn prompt_uses_template_names_and_falls_back_to_english() {
        let p = render_prompt("de", "ru", "Hallo");
        assert!(p.starts_with("You are a professional German (de) to Russian (ru) translator."));
        assert!(p.ends_with("into Russian:\n\n\nHallo"));
        let p = render_prompt("xx", "ru", "?");
        assert!(p.contains("professional English (en) to Russian (ru)"));
    }

    // --- selection limit and output ---------------------------------------

    #[test]
    fn selection_is_cut_by_characters() {
        let (t, cut) = truncate_selection("абв");
        assert_eq!((t.as_str(), cut), ("абв", false));

        let exact = "я".repeat(MAX_SELECTION_CHARS);
        assert_eq!(truncate_selection(&exact), (exact.clone(), false));

        let long = "ж".repeat(MAX_SELECTION_CHARS + 5);
        let (t, cut) = truncate_selection(&long);
        assert!(cut);
        assert_eq!(char_len(&t), MAX_SELECTION_CHARS);
    }

    #[test]
    fn output_is_cleaned() {
        assert_eq!(clean_output("  Hello, world.\n"), "Hello, world.");
        assert_eq!(clean_output("<think>reasoning</think>\n\nПривет"), "Привет");
        assert_eq!(clean_output("\u{FEFF}He\u{200B}llo\u{200D} "), "Hello");
        assert_eq!(clean_output("no <think> block"), "no <think> block");
        assert_eq!(clean_output("<think>unclosed"), "<think>unclosed");
    }

    #[test]
    fn context_overflow_is_recognised() {
        assert!(is_context_overflow(
            "API request failed with status 400 Bad Request: \
{\"error\":{\"code\":400,\"type\":\"exceed_context_size_error\"}}"
        ));
        assert!(is_context_overflow(
            "request (5386 tokens) exceeds the available context size (4096)"
        ));
        assert!(!is_context_overflow("API request failed with status 401"));
    }

    // --- splitting --------------------------------------------------------

    #[test]
    fn short_text_is_one_piece() {
        let s = round_trip("Привет, мир");
        assert_eq!(s, vec![Segment::Translate("Привет, мир".to_string())]);
    }

    #[test]
    fn paragraphs_are_separate_pieces_and_separators_are_kept() {
        let s = round_trip("Первый абзац.\n\nВторой абзац.\n\n\nТретий.");
        assert_eq!(
            s,
            vec![
                Segment::Translate("Первый абзац.".into()),
                Segment::Keep("\n\n".into()),
                Segment::Translate("Второй абзац.".into()),
                Segment::Keep("\n\n\n".into()),
                Segment::Translate("Третий.".into()),
            ]
        );
    }

    #[test]
    fn lines_inside_a_paragraph_go_together() {
        let s = round_trip("- один\n- два\n- три");
        assert_eq!(s, vec![Segment::Translate("- один\n- два\n- три".into())]);
    }

    #[test]
    fn lists_with_blank_lines_keep_markers_per_item() {
        let text = "Привет, Иван!\n\n- купить хлеб\n\n* позвонить маме\n\n1. записаться\n\n2) оплатить\n\nПётр";
        let s = round_trip(text);
        let t = translations_of(&s);
        assert_eq!(
            t,
            vec![
                "Привет, Иван!",
                "- купить хлеб",
                "* позвонить маме",
                "1. записаться",
                "2) оплатить",
                "Пётр",
            ]
        );
    }

    #[test]
    fn indentation_and_trailing_whitespace_are_kept_outside_pieces() {
        let text = "\n\n    Отступ из пробелов.  \n\t\tТабы внутри.\t\n\n  Другой.   \n";
        let s = round_trip(text);
        assert_eq!(
            s,
            vec![
                Segment::Keep("\n\n    ".into()),
                Segment::Translate("Отступ из пробелов.  \n\t\tТабы внутри.".into()),
                Segment::Keep("\t\n\n  ".into()),
                Segment::Translate("Другой.".into()),
                Segment::Keep("   \n".into()),
            ]
        );
    }

    #[test]
    fn crlf_is_preserved() {
        let text = "Здравствуйте,\r\nэто письмо.\r\n\r\n- пункт\r\n\r\nС уважением\r\n";
        let s = round_trip(text);
        assert_eq!(
            translations_of(&s),
            vec!["Здравствуйте,\r\nэто письмо.", "- пункт", "С уважением"]
        );
        assert_eq!(
            s.first(),
            Some(&Segment::Translate("Здравствуйте,\r\nэто письмо.".into()))
        );
    }

    #[test]
    fn whitespace_only_and_empty_texts_are_not_sent() {
        assert!(split_for_translation("").is_empty());
        for text in [" ", "\n\n", " \t\r\n  \r\n"] {
            let s = round_trip(text);
            assert!(translations_of(&s).is_empty());
            assert_eq!(s, vec![Segment::Keep(text.to_string())]);
        }
    }

    #[test]
    fn pieces_without_letters_are_not_sent() {
        let s = round_trip("Заголовок\n\n-----\n\n2024\n\nТекст");
        assert_eq!(translations_of(&s), vec!["Заголовок", "Текст"]);
    }

    #[test]
    fn long_paragraph_is_packed_by_sentences() {
        let sentence = "Это довольно длинное предложение для проверки нарезки текста.";
        let paragraph = vec![sentence; 50].join(" ");
        assert!(char_len(&paragraph) > 3000);
        let s = round_trip(&paragraph);
        let chunks = translations_of(&s);
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(c.ends_with("текста."), "cut at a sentence end: {c:?}");
        }
        // Filled up to the limit: all but the last (the remainder) are within one sentence of it.
        let (_, full) = chunks.split_last().unwrap();
        for c in full {
            assert!(char_len(c) > MAX_CHUNK_CHARS - sentence.chars().count() - 1);
        }
    }

    #[test]
    fn long_paragraph_is_cut_at_line_breaks_too() {
        let line = "- пункт списка без точки в конце, но довольно длинный для теста";
        let paragraph = vec![line; 40].join("\n");
        let s = round_trip(&paragraph);
        for c in translations_of(&s) {
            assert!(c.starts_with("- пункт") && c.ends_with("для теста"));
        }
    }

    #[test]
    fn sentence_without_punctuation_is_cut_at_spaces() {
        let text = vec!["слово"; 500].join(" "); // 2999 chars, no sentence marks
        let s = round_trip(&text);
        let chunks = translations_of(&s);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|c| c.split(' ').all(|w| w == "слово")));
    }

    #[test]
    fn long_sentence_is_cut_at_commas_before_spaces() {
        let clause = "это длинная часть предложения без точки";
        let text = vec![clause; 60].join(", ");
        let s = round_trip(&text);
        let chunks = translations_of(&s);
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(
                c.starts_with("это длинная") && c.ends_with("без точки,")
                    || c.ends_with("без точки")
            );
        }
    }

    #[test]
    fn cyrillic_word_longer_than_a_chunk_is_cut_by_characters() {
        let text = "я".repeat(2500); // 5000 bytes, no spaces
        let s = round_trip(&text);
        let lens: Vec<usize> = translations_of(&s).iter().map(|t| char_len(t)).collect();
        assert_eq!(lens, vec![1000, 1000, 500]);
    }

    #[test]
    fn multibyte_text_on_a_chunk_boundary() {
        // 999 ASCII letters + a Cyrillic word that straddles the 1000th character.
        let text = format!("{} привет мир", "a".repeat(998));
        assert!(char_len(&text) > MAX_CHUNK_CHARS - 5);
        let long = format!("{text} {}", "ёж ".repeat(200));
        round_trip(&long);
        for n in 990..1010 {
            round_trip(&format!("{}{}", "ы".repeat(n), " э".repeat(5)));
            round_trip(&"я".repeat(n));
        }
    }

    #[test]
    fn real_letter_is_translated_by_paragraphs_and_restored() {
        let text = "Добрый день, Пётр!\r\n\r\nПрошу подготовить:\r\n- отчёт за квартал\r\n- список задач\r\n\r\n\
Срок — пятница.\r\n\r\nС уважением,\r\nАлексей\r\n";
        let s = round_trip(text);
        let fake: Vec<String> = translations_of(&s)
            .iter()
            .map(|t| format!("<{t}>"))
            .collect();
        let out = assemble(&s, &fake);
        assert!(out.starts_with("<Добрый день, Пётр!>\r\n\r\n<Прошу подготовить:\r\n- отчёт"));
        assert!(out.ends_with("<С уважением,\r\nАлексей>\r\n"));
    }

    #[test]
    fn missing_translation_keeps_the_original_piece() {
        let s = split_for_translation("Один.\n\nДва.");
        assert_eq!(assemble(&s, &["One.".to_string()]), "One.\n\nДва.");
    }

    #[test]
    fn selection_of_max_size_splits_within_limits() {
        let text = "Это тестовое предложение, которое повторяется много раз. ".repeat(400);
        let (cut, was_cut) = truncate_selection(&text);
        assert!(was_cut);
        let s = round_trip(&cut);
        assert!(translations_of(&s).len() >= 20);
    }
    #[test]
    fn russian_only_letters_win_over_a_wrong_guess() {
        // whatlang took this short Russian text with a Latin word for Bulgarian.
        let text = "Начните с пункта 1 из списка: выделите короткий текст и дважды нажмите Ctrl.";
        assert_eq!(detect_direction(text), RU_TO_EN);
        // Ukrainian and Belarusian marks keep their own verdict.
        assert_eq!(detect_direction("Привіт, їжак ыэ").src, "uk");
        assert!(!has_russian_only_letters("Здравствуйте, как дела"));
        assert!(!has_russian_only_letters("Hello world, ы"));
    }

    #[test]
    fn a_picked_source_language_gives_the_direction() {
        assert_eq!(direction_from_source("ru"), Some(RU_TO_EN));
        assert_eq!(
            direction_from_source("bg"),
            Some(Direction {
                src: "bg",
                tgt: "ru"
            })
        );
        assert_eq!(direction_from_source("xx"), None);
        assert_eq!(source_languages().len(), LANGUAGES.len());
    }

    #[test]
    fn auto_target_keeps_the_old_rule() {
        let d = resolve_direction("Привет, мир", None, None);
        assert_eq!((d.src, d.tgt), ("ru", "en"));
        let d = resolve_direction("Hello, how are you doing today? I would like to translate this text.", None, None);
        assert_eq!((d.src, d.tgt), ("en", "ru"));
        let d = resolve_direction("anything", Some("de"), None);
        assert_eq!((d.src, d.tgt), ("de", "ru"));
    }

    #[test]
    fn explicit_target_replaces_only_the_target() {
        let d = resolve_direction("Привет, мир", None, Some("de"));
        assert_eq!((d.src, d.tgt), ("ru", "de"));
        let d = resolve_direction("Hello, how are you doing today? I would like to translate this text.", None, Some("ja"));
        assert_eq!((d.src, d.tgt), ("en", "ja"));
        let d = resolve_direction("anything", Some("fr"), Some("es"));
        assert_eq!((d.src, d.tgt), ("fr", "es"));
    }

    #[test]
    fn target_equal_to_source_is_detected() {
        let d = resolve_direction("Привет, мир", None, Some("ru"));
        assert!(d.is_same_language());
        let d = resolve_direction("Привет, мир", None, Some("en"));
        assert!(!d.is_same_language());
    }

    #[test]
    fn explicit_target_accepts_only_known_codes() {
        assert_eq!(explicit_target("de"), Some("de"));
        assert_eq!(explicit_target("auto"), None);
        assert_eq!(explicit_target(""), None);
        assert_eq!(explicit_target("xx"), None);
    }

    #[test]
    fn every_target_renders_a_native_prompt() {
        for (code, name) in target_languages() {
            let p = render_prompt("en", code, "Hi");
            assert!(p.contains(&format!("{name} ({code})")), "{code}");
        }
    }
}
