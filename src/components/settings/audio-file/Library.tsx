// SelTranslate: transcript library under the Audio file result (stage H).
// The backend saves every finished transcript as a .md file while Library is
// on; this lists the folder and renames, opens, reveals and deletes files.
import React, { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { listen } from "@tauri-apps/api/event";
import { ask, open } from "@tauri-apps/plugin-dialog";
import { FolderOpen, Pencil, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { commands, type LibraryEntry, type LibraryInfo } from "@/bindings";
import { Button } from "../../ui/Button";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { ToggleSwitch } from "../../ui/ToggleSwitch";
import { useAudioFileStore } from "./audioFileStore";

// `YYYY-MM-DD HH-mm Title.md` → `Title` (library::file_name in the backend).
const titleOf = (fileName: string) =>
  fileName.replace(/^\d{4}-\d{2}-\d{2} \d{2}-\d{2} /, "").replace(/\.md$/i, "");

const IconButton: React.FC<{
  onClick: () => void;
  title: string;
  children: React.ReactNode;
}> = ({ onClick, title, children }) => (
  <button
    type="button"
    onClick={onClick}
    title={title}
    aria-label={title}
    className="p-1.5 rounded-md text-text/50 hover:text-logo-primary transition-colors cursor-pointer"
  >
    {children}
  </button>
);

const Row: React.FC<{ entry: LibraryEntry }> = ({ entry }) => {
  const { t, i18n } = useTranslation();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(entry.title);

  const date = new Date(entry.date).toLocaleString(i18n.language, {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  const report = (result: { status: string; error?: string }) => {
    if (result.status === "error")
      toast.error(t("audioFile.library.failed", { error: result.error }));
  };

  const commit = async () => {
    const title = draft.trim();
    if (!title) return;
    setEditing(false);
    if (title !== entry.title)
      report(await commands.libraryRename(entry.file_name, title));
  };

  const handleDelete = async () => {
    const confirmed = await ask(
      t("audioFile.library.deleteConfirm", { title: entry.title }),
      { title: t("audioFile.library.deleteTitle"), kind: "warning" },
    );
    if (confirmed) report(await commands.libraryDelete(entry.file_name));
  };

  return (
    <div className="flex items-center gap-2 px-4 py-1.5">
      {editing ? (
        <input
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void commit();
            else if (e.key === "Escape") setEditing(false);
          }}
          onBlur={() => setEditing(false)}
          className={`flex-1 min-w-0 px-2 py-1 text-sm bg-mid-gray/10 border rounded-md focus:outline-none ${
            draft.trim() ? "border-logo-primary" : "border-red-400"
          }`}
        />
      ) : (
        <button
          type="button"
          onClick={async () =>
            report(await commands.libraryOpen(entry.file_name))
          }
          title={entry.file_name}
          className="flex-1 min-w-0 text-left text-sm truncate cursor-pointer hover:text-logo-primary"
        >
          <span className="text-mid-gray">{`${date} · `}</span>
          {entry.title}
        </button>
      )}
      <IconButton
        title={t("audioFile.library.rename")}
        onClick={() => {
          setDraft(entry.title);
          setEditing(true);
        }}
      >
        <Pencil width={15} height={15} />
      </IconButton>
      <IconButton
        title={t("audioFile.library.showInFolder")}
        onClick={async () =>
          report(await commands.libraryReveal(entry.file_name))
        }
      >
        <FolderOpen width={15} height={15} />
      </IconButton>
      <IconButton title={t("audioFile.library.delete")} onClick={handleDelete}>
        <Trash2 width={15} height={15} />
      </IconButton>
    </div>
  );
};

export const Library: React.FC = () => {
  const { t } = useTranslation();
  const { status, savedAs, saveError } = useAudioFileStore();
  const [info, setInfo] = useState<LibraryInfo | null>(null);
  const [entries, setEntries] = useState<LibraryEntry[]>([]);

  const refresh = useCallback(async () => {
    const next = await commands.libraryGetSettings();
    setInfo(next);
    if (!next.enabled) return;
    const result = await commands.libraryList();
    if (result.status === "ok") setEntries(result.data);
  }, []);

  useEffect(() => {
    void refresh();
    const unlisten = listen("library-changed", () => void refresh());
    return () => {
      void unlisten.then((f) => f());
    };
  }, [refresh]);

  const update = async (enabled: boolean, dir: string) => {
    const result = await commands.librarySetSettings(enabled, dir);
    if (result.status === "ok") setInfo(result.data);
    else toast.error(t("audioFile.library.failed", { error: result.error }));
  };

  const handleChange = async () => {
    if (!info) return;
    const picked = await open({
      directory: true,
      multiple: false,
      defaultPath: info.resolved_dir || undefined,
    });
    if (typeof picked === "string") await update(info.enabled, picked);
  };

  if (!info) return null;

  return (
    <SettingsGroup title={t("audioFile.library.title")}>
      {status === "done" && info.enabled && (savedAs || saveError) && (
        <div className="px-4 py-2">
          {savedAs ? (
            <p className="text-sm text-mid-gray truncate">
              {t("audioFile.library.savedTo", { title: titleOf(savedAs) })}
            </p>
          ) : (
            <p className="text-sm text-warning break-words">
              {t("audioFile.library.saveFailed", { error: saveError })}
            </p>
          )}
        </div>
      )}
      <ToggleSwitch
        checked={info.enabled}
        onChange={(enabled) => void update(enabled, info.dir)}
        label={t("audioFile.library.enable")}
        description={t("audioFile.library.enableDescription")}
        descriptionMode="inline"
        grouped
      />
      <div className="px-4 py-3 space-y-1">
        <div className="flex items-center gap-2">
          <span
            className="flex-1 min-w-0 text-xs font-mono truncate"
            title={info.resolved_dir}
          >
            {info.resolved_dir}
          </span>
          {info.dir && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => update(info.enabled, "")}
            >
              {t("audioFile.library.useDefault")}
            </Button>
          )}
          <Button variant="secondary" size="sm" onClick={handleChange}>
            {t("audioFile.library.change")}
          </Button>
        </div>
        <p className="text-xs text-mid-gray">
          {t("audioFile.library.changeHint")}
        </p>
      </div>
      {info.enabled &&
        (entries.length === 0 ? (
          <p className="px-4 py-3 text-sm text-mid-gray">
            {t("audioFile.library.empty")}
          </p>
        ) : (
          <div className="py-1">
            {entries.map((entry) => (
              <Row key={entry.file_name} entry={entry} />
            ))}
          </div>
        ))}
    </SettingsGroup>
  );
};
