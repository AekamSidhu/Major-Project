// Settings form controls: labelled inputs that validate inline and save
// through the store (debounced for typing, immediate for choices).

import { useId, type ReactNode } from "react";
import { Badge, Field, Select, Slider, TextArea, TextInput, cx, type SelectOption } from "../ui";
import { useDraft } from "./useDraft";
import { parseIntegerInRange } from "./validation";
import type { ChoiceLocation } from "./settingsMeta";

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

export interface TextSettingProps {
  label: string;
  description?: ReactNode;
  value: string;
  onCommit: (value: string) => Promise<boolean>;
  validate?: (value: string) => string | null;
  placeholder?: string;
  /** id of a <datalist> with suggestions. */
  list?: string;
  monospace?: boolean;
  type?: "text" | "url";
  disabled?: boolean;
  /** Extra controls rendered next to the input (e.g. "Load models"). */
  addon?: ReactNode;
  autoFocus?: boolean;
}

export function TextSetting({
  label,
  description,
  value,
  onCommit,
  validate,
  placeholder,
  list,
  monospace,
  type = "text",
  disabled,
  addon,
  autoFocus,
}: TextSettingProps) {
  const id = useId();
  const draft = useDraft({ value, commit: onCommit, validate });
  return (
    <Field label={label} description={description} htmlFor={id} error={draft.error}>
      <div className="flex items-center gap-2">
        <TextInput
          id={id}
          type={type}
          value={draft.value}
          list={list}
          placeholder={placeholder}
          disabled={disabled}
          autoFocus={autoFocus}
          spellCheck={false}
          autoComplete="off"
          aria-invalid={draft.error ? true : undefined}
          onChange={(e) => draft.onChange(e.target.value)}
          onBlur={draft.onBlur}
          className={cx(monospace && "font-mono text-[0.8rem]")}
        />
        {addon}
      </div>
    </Field>
  );
}

export interface TextAreaSettingProps {
  label: string;
  description?: ReactNode;
  value: string;
  onCommit: (value: string) => Promise<boolean>;
  maxLength: number;
  placeholder?: string;
  rows?: number;
  /** Additional status under the field (e.g. "4 topics"). */
  footer?: ReactNode;
}

export function TextAreaSetting({ label, description, value, onCommit, maxLength, placeholder, rows = 6, footer }: TextAreaSettingProps) {
  const id = useId();
  const countId = `${id}-count`;
  const draft = useDraft({ value, commit: onCommit, delay: 800 });
  const count = draft.value.length;
  return (
    <Field label={label} description={description} htmlFor={id} error={draft.error}>
      <TextArea
        id={id}
        rows={rows}
        value={draft.value}
        maxLength={maxLength}
        placeholder={placeholder}
        aria-describedby={countId}
        onChange={(e) => draft.onChange(e.target.value)}
        onBlur={draft.onBlur}
      />
      <div id={countId} className="mt-1 flex items-center justify-between gap-3 text-[0.74rem] text-muted">
        <span>{footer}</span>
        <span className={cx("tabular-nums", count >= maxLength && "text-warn")}>
          {count.toLocaleString()} / {maxLength.toLocaleString()}
          {draft.editing ? " · editing" : " · saved"}
        </span>
      </div>
    </Field>
  );
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

export interface NumberSettingProps {
  label: string;
  description?: ReactNode;
  value: number;
  min: number;
  max: number;
  unit?: string;
  onCommit: (value: number) => Promise<boolean>;
  inline?: boolean;
}

/** Whole-number input with range validation. */
export function NumberSetting({ label, description, value, min, max, unit, onCommit, inline = true }: NumberSettingProps) {
  const id = useId();
  const draft = useDraft<string>({
    value: String(value),
    validate: (raw) => {
      const r = parseIntegerInRange(raw, min, max);
      return r.ok ? null : r.error;
    },
    commit: (raw) => {
      const r = parseIntegerInRange(raw, min, max);
      return r.ok ? onCommit(r.value) : Promise.resolve(false);
    },
  });
  return (
    <Field label={label} description={description} htmlFor={id} error={draft.error} inline={inline}>
      <div className="flex items-center gap-2">
        <TextInput
          id={id}
          inputMode="numeric"
          value={draft.value}
          aria-invalid={draft.error ? true : undefined}
          onChange={(e) => draft.onChange(e.target.value)}
          onBlur={draft.onBlur}
          className="w-28 text-right tabular-nums"
        />
        {unit && <span className="w-8 text-[0.8rem] text-muted">{unit}</span>}
      </div>
    </Field>
  );
}

export interface SliderSettingProps {
  label: string;
  description?: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  format?: (value: number) => string;
  onCommit: (value: number) => Promise<boolean>;
}

/** Range slider; saves shortly after the user stops dragging. */
export function SliderSetting({ label, description, value, min, max, step = 1, format, onCommit }: SliderSettingProps) {
  const id = useId();
  const draft = useDraft<number>({ value, commit: onCommit, delay: 250, resetOnCommit: true });
  return (
    <Field label={label} description={description} htmlFor={id} error={draft.error} inline>
      <Slider id={id} label={label} value={draft.value} min={min} max={max} step={step} format={format} onChange={draft.onChange} />
    </Field>
  );
}

// ---------------------------------------------------------------------------
// Choices
// ---------------------------------------------------------------------------

export interface SelectSettingProps<T extends string> {
  label: string;
  description?: ReactNode;
  value: T;
  options: ReadonlyArray<SelectOption<T>>;
  onChange: (value: T) => void;
  disabled?: boolean;
  inline?: boolean;
}

export function SelectSetting<T extends string>({ label, description, value, options, onChange, disabled, inline = true }: SelectSettingProps<T>) {
  const id = useId();
  return (
    <Field label={label} description={description} htmlFor={id} inline={inline}>
      <Select id={id} value={value} options={options} onChange={onChange} disabled={disabled} className={inline ? "w-56" : undefined} />
    </Field>
  );
}

export interface ChoiceOption<T extends string> {
  value: T;
  label: string;
  description?: ReactNode;
  location?: ChoiceLocation;
  disabled?: boolean;
}

/** Radio-card group (native radios: arrow keys move the selection). */
export function ChoiceCards<T extends string>({
  label,
  value,
  options,
  onChange,
  columns = 2,
  hideLabel,
}: {
  label: string;
  value: T;
  options: ReadonlyArray<ChoiceOption<T>>;
  onChange: (value: T) => void;
  columns?: 1 | 2 | 3;
  hideLabel?: boolean;
}) {
  const name = useId();
  return (
    <fieldset className="py-2.5">
      <legend className={cx("mb-2 text-[0.86rem] font-medium text-fg", hideLabel && "sr-only")}>{label}</legend>
      <div className={cx("grid gap-2", columns === 2 && "sm:grid-cols-2", columns === 3 && "sm:grid-cols-3")}>
        {options.map((o) => {
          const checked = o.value === value;
          return (
            <label
              key={o.value}
              className={cx(
                "flex cursor-pointer gap-2.5 rounded-md border px-3 py-2.5 transition-colors",
                "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent",
                checked ? "border-accent bg-accent-soft" : "border-border hover:border-border-strong",
                o.disabled && "cursor-not-allowed opacity-50",
              )}
            >
              <input
                type="radio"
                name={name}
                value={o.value}
                checked={checked}
                disabled={o.disabled}
                onChange={() => onChange(o.value)}
                className="mt-0.5 accent-[var(--om-accent)] focus-visible:outline-none"
              />
              <span className="min-w-0">
                <span className="flex flex-wrap items-center gap-1.5 text-[0.86rem] font-medium text-fg">
                  {o.label}
                  {o.location && <ChoiceBadge location={o.location} />}
                </span>
                {o.description && <span className="mt-0.5 block text-[0.78rem] leading-snug text-muted">{o.description}</span>}
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

/** Static Local / Cloud label for a provider choice. */
export function ChoiceBadge({ location }: { location: ChoiceLocation }) {
  switch (location) {
    case "local":
      return <Badge tone="ok">Local</Badge>;
    case "cloud":
      return <Badge tone="warn">Cloud</Badge>;
    case "configurable":
      return <Badge tone="neutral">Local or cloud</Badge>;
    case "off":
      return <Badge tone="neutral">Off</Badge>;
  }
}
