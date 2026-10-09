// Small form controls used by the meeting and history sections (native
// inputs underneath, so keyboard and screen-reader behaviour is standard).

import { useId, type ReactNode } from "react";
import { cx } from "@/main/ui";
import { Icon } from "./icons";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
}

/** Radio group rendered as a compact segmented control (arrow keys move the selection). */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled,
}: {
  label: string;
  value: T;
  options: ReadonlyArray<SegmentedOption<T>>;
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  const name = useId();
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex rounded-md border border-border bg-surface-2 p-0.5"
    >
      {options.map((o) => (
        <label key={o.value} className="relative">
          <input
            type="radio"
            name={name}
            value={o.value}
            checked={value === o.value}
            disabled={disabled || o.disabled}
            onChange={() => onChange(o.value)}
            className="peer sr-only"
          />
          <span
            className={cx(
              "block cursor-pointer rounded px-2.5 py-1 text-[0.8rem] font-medium transition-colors select-none",
              "peer-focus-visible:outline-2 peer-focus-visible:outline-offset-1 peer-focus-visible:outline-accent",
              "peer-disabled:cursor-not-allowed peer-disabled:opacity-50",
              value === o.value ? "bg-raised text-fg shadow-sm" : "text-muted hover:text-fg-soft",
            )}
          >
            {o.label}
          </span>
        </label>
      ))}
    </div>
  );
}

/** Native checkbox with a label and optional description. */
export function Checkbox({
  label,
  description,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  description?: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const descId = useId();
  return (
    <div className="flex items-start gap-2.5 py-1.5">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        aria-describedby={description ? descId : undefined}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 size-4 shrink-0 cursor-pointer accent-[var(--om-accent)] disabled:cursor-not-allowed"
      />
      <div className="min-w-0">
        <label htmlFor={id} className="block cursor-pointer text-[0.86rem] text-fg">
          {label}
        </label>
        {description && (
          <div id={descId} className="text-[0.78rem] leading-snug text-muted">
            {description}
          </div>
        )}
      </div>
    </div>
  );
}

/** Inline error line (role="alert"), used next to the control that failed. */
export function InlineError({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p role="alert" className={cx("flex items-start gap-1.5 text-[0.8rem] text-danger", className)}>
      <Icon name="info" className="mt-0.5 size-3.5" />
      <span className="min-w-0">{children}</span>
    </p>
  );
}

/** Text-styled button for secondary navigation ("Privacy settings →"). */
export function LinkButton({
  children,
  onClick,
  className,
  disabled,
}: {
  children: ReactNode;
  onClick: () => void;
  className?: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cx(
        "inline-flex items-center gap-1 rounded text-[0.8rem] font-medium text-accent underline-offset-2 hover:underline",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
    >
      {children}
    </button>
  );
}

/** A level meter (0..1) for audio sources. */
export function LevelMeter({ level, label }: { level: number; label: string }) {
  const pct = Math.round(Math.min(1, Math.max(0, level)) * 100);
  return (
    <span
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      className="inline-block h-1 w-14 overflow-hidden rounded-full bg-surface-2"
    >
      <span className="block h-full rounded-full bg-ok transition-[width] duration-100" style={{ width: `${pct}%` }} />
    </span>
  );
}
