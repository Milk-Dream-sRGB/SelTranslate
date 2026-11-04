// SelTranslate: the text of a Speakers result, for the panel, Copy and Save.
import type { TurnText } from "./historySpeakersStore";

/** `[01:02:03]`: always with hours, so a long file's lines line up. */
export const timestamp = (ms: number): string => {
  const total = Math.max(0, Math.floor(ms / 1000));
  const pad = (n: number) => String(n).padStart(2, "0");
  return `[${pad(Math.floor(total / 3600))}:${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}]`;
};

/** The file name without its extension; `null` for dictation. */
export const titleOf = (sourceName: string | null): string | null =>
  sourceName ? sourceName.replace(/\.[^.]+$/, "") || sourceName : null;

/** `# title`, a blank line, then one `[time] Speaker N: text` per turn. */
export const formatTranscript = (
  turns: TurnText[],
  title: string | null,
  speakerLabel: (n: number) => string,
): string => {
  const lines = turns.map(
    (turn) =>
      `${timestamp(turn.start_ms)} ${speakerLabel(turn.speaker)}: ${turn.text}`,
  );
  return (title ? [`# ${title}`, "", ...lines] : lines).join("\n");
};
