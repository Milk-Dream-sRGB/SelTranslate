// SelTranslate (J.11): macOS lets an app press keys (Cmd+C to copy the
// selection, Cmd+V for Replace) and watch the keyboard (Double Ctrl) only with
// the Accessibility permission. This row says whether it is there and leads to
// the System Settings page when it is not. Shown on a Mac only.
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  checkAccessibilityPermission,
  requestAccessibilityPermission,
} from "tauri-plugin-macos-permissions-api";
import { Button } from "../../ui/Button";
import { SettingContainer } from "../../ui/SettingContainer";
import { SettingsGroup } from "../../ui/SettingsGroup";

const CHECK_EVERY_MS = 2000;

export const AccessibilityRow: React.FC = () => {
  const { t } = useTranslation();
  const [allowed, setAllowed] = useState<boolean | null>(null);

  // The user flips the switch in System Settings while this page is open.
  useEffect(() => {
    let alive = true;
    const check = () =>
      void checkAccessibilityPermission()
        .then((ok) => alive && setAllowed(ok))
        .catch(() => alive && setAllowed(null));
    check();
    const timer = window.setInterval(check, CHECK_EVERY_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  return (
    <SettingsGroup title={t("translateSettings.accessibility.title")}>
      <SettingContainer
        title={t("translateSettings.accessibility.label")}
        description={t("translateSettings.accessibility.description")}
        descriptionMode="inline"
        grouped={true}
      >
        {allowed ? (
          <span className="text-sm">
            {t("translateSettings.accessibility.allowed")}
          </span>
        ) : (
          <Button
            variant="primary"
            onClick={() => void requestAccessibilityPermission()}
          >
            {t("translateSettings.accessibility.open")}
          </Button>
        )}
      </SettingContainer>
    </SettingsGroup>
  );
};
