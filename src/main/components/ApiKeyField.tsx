// Write-only API key input. Keys go straight to the OS keychain through the
// backend and are never read back; only whether one is configured (and from
// where) is shown.

import { useEffect, useId, useState, type ReactNode } from "react";
import type { SecretProvider } from "@/bindings";
import { errorMessage } from "@/lib/ipc";
import { Button, Field, StatusDot, TextInput } from "../ui";
import { SECRET_LABELS, useApiKeys } from "./apiKeys";
import { validateApiKey } from "./validation";

export function ApiKeyField({
  provider,
  optional = false,
}: {
  provider: SecretProvider;
  optional?: boolean;
}) {
  const id = useId();
  const statusId = `${id}-status`;
  const status = useApiKeys((s) => s.statuses[provider]);
  const loaded = useApiKeys((s) => s.loaded);
  const loadError = useApiKeys((s) => s.error);
  const save = useApiKeys((s) => s.save);
  const clear = useApiKeys((s) => s.clear);
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState<"save" | "clear" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    const s = useApiKeys.getState();
    if (!s.loaded && !s.loading) void s.load();
  }, []);

  const onSave = async () => {
    const invalid = validateApiKey(key);
    if (invalid) {
      setError(invalid);
      return;
    }
    setBusy("save");
    setError(null);
    setMessage(null);
    try {
      await save(provider, key);
      setKey("");
      setMessage("Key saved to the OS keychain.");
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(null);
    }
  };

  const onClear = async () => {
    setBusy("clear");
    setError(null);
    setMessage(null);
    try {
      await clear(provider);
      setMessage("Key removed from the keychain.");
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(null);
    }
  };

  const label = `${SECRET_LABELS[provider]} API key${optional ? " (optional)" : ""}`;

  let statusLine: ReactNode;
  if (!loaded) statusLine = <span className="text-muted">Checking…</span>;
  else if (loadError && !status)
    statusLine = <span className="text-danger">Key status unavailable: {loadError}</span>;
  else if (status?.source === "keychain")
    statusLine = (
      <span className="inline-flex items-center gap-1.5 text-fg-soft">
        <StatusDot tone="ok" /> Saved in the OS keychain
      </span>
    );
  else if (status?.source === "environment")
    statusLine = (
      <span className="inline-flex items-center gap-1.5 text-fg-soft">
        <StatusDot tone="ok" /> Read from the environment variable{" "}
        <code className="font-mono">{status.envVar}</code>
      </span>
    );
  else
    statusLine = (
      <span className="inline-flex items-center gap-1.5 text-muted">
        <StatusDot tone={optional ? "neutral" : "warn"} /> No key set
        {status?.envVar ? (
          <>
            {" "}
            — paste one below or set <code className="font-mono">{status.envVar}</code>
          </>
        ) : null}
      </span>
    );

  return (
    <Field
      label={label}
      htmlFor={id}
      error={error}
      description={
        <span id={statusId} role="status" className="block">
          {statusLine}
          {message && <span className="mt-0.5 block text-ok">{message}</span>}
        </span>
      }
    >
      <form
        className="flex items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void onSave();
        }}
      >
        <TextInput
          id={id}
          type="password"
          value={key}
          autoComplete="off"
          spellCheck={false}
          placeholder={status?.configured ? "Paste a new key to replace it" : "Paste API key"}
          aria-describedby={statusId}
          aria-invalid={error ? true : undefined}
          onChange={(e) => {
            setKey(e.target.value);
            setError(null);
          }}
          className="font-mono"
        />
        <Button
          type="submit"
          variant="secondary"
          busy={busy === "save"}
          disabled={key.trim() === "" || busy !== null}
        >
          Save key
        </Button>
        {status?.source === "keychain" && (
          <Button
            variant="ghost"
            busy={busy === "clear"}
            disabled={busy !== null}
            onClick={() => void onClear()}
          >
            Remove
          </Button>
        )}
      </form>
    </Field>
  );
}
