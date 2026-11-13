// Minimal, accessible UI primitives for the main window.
// Calm surfaces, one accent gradient for what matters, soft depth.

import {
  useEffect,
  useId,
  useRef,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

// ---------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: "sm" | "md";
  busy?: boolean;
}

const VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "om-gradient text-white shadow-glow border border-white/10 hover:brightness-110 hover:shadow-[0_10px_30px_-8px_var(--om-accent)]",
  secondary:
    "bg-surface-2 text-fg border border-border shadow-[0_1px_0_var(--om-highlight)_inset] hover:border-border-strong hover:bg-raised",
  ghost: "bg-transparent text-fg-soft border border-transparent hover:bg-surface-2 hover:text-fg",
  danger:
    "bg-danger-soft text-danger border border-danger/25 hover:bg-danger hover:text-white hover:border-transparent",
};

export function Button({
  variant = "secondary",
  size = "md",
  busy = false,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      type="button"
      {...rest}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium whitespace-nowrap",
        "transition-all duration-150 active:translate-y-px",
        "disabled:pointer-events-none disabled:opacity-45",
        size === "sm" ? "h-7.5 px-3 text-[0.8rem]" : "h-9 px-4 text-[0.86rem]",
        VARIANTS[variant],
        className,
      )}
    >
      {busy && <Spinner />}
      {children}
    </button>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cx(
        "inline-block size-3 animate-spin rounded-full border-[1.5px] border-current border-t-transparent",
        className,
      )}
    />
  );
}

// ---------------------------------------------------------------------------
// Form fields
// ---------------------------------------------------------------------------

export interface FieldProps {
  label: string;
  description?: ReactNode;
  error?: string | null;
  htmlFor?: string;
  children: ReactNode;
  inline?: boolean;
}

/** Label + optional description + control + error message. */
export function Field({ label, description, error, htmlFor, children, inline }: FieldProps) {
  return (
    <div className={cx("py-2.5", inline && "flex items-center justify-between gap-6")}>
      <div className={cx(inline ? "min-w-0 flex-1" : "mb-1.5")}>
        <label htmlFor={htmlFor} className="block text-[0.86rem] font-medium text-fg">
          {label}
        </label>
        {description && (
          <div className="mt-0.5 text-[0.78rem] leading-snug text-muted">{description}</div>
        )}
      </div>
      <div className={cx(inline && "shrink-0")}>{children}</div>
      {error && (
        <p role="alert" className="mt-1 text-[0.78rem] text-danger">
          {error}
        </p>
      )}
    </div>
  );
}

const INPUT =
  "w-full rounded-lg border border-border bg-surface-2 px-3 text-[0.86rem] text-fg placeholder:text-muted " +
  "shadow-[0_1px_0_var(--om-highlight)_inset] transition-[border-color,box-shadow] duration-150 " +
  "hover:border-border-strong focus:border-accent focus:ring-4 focus:ring-accent-soft focus:outline-none disabled:opacity-50";

export function TextInput(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cx(INPUT, "h-9", props.className)} />;
}

export function TextArea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea {...props} className={cx(INPUT, "min-h-24 py-2 leading-relaxed", props.className)} />
  );
}

export interface SelectOption<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
}

export interface SelectProps<T extends string> extends Omit<
  SelectHTMLAttributes<HTMLSelectElement>,
  "onChange" | "value"
> {
  value: T;
  options: ReadonlyArray<SelectOption<T>>;
  onChange: (value: T) => void;
}

export function Select<T extends string>({
  value,
  options,
  onChange,
  className,
  ...rest
}: SelectProps<T>) {
  return (
    <select
      {...rest}
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
      className={cx(INPUT, "h-9 pr-8", className)}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  disabled?: boolean;
  id?: string;
}

/** Accessible switch (role="switch"). `label` is announced to screen readers. */
export function Toggle({ checked, onChange, label, disabled, id }: ToggleProps) {
  return (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cx(
        "relative inline-flex h-[22px] w-10 shrink-0 items-center rounded-full border transition-all duration-200",
        "disabled:cursor-not-allowed disabled:opacity-50",
        checked
          ? "om-gradient border-transparent shadow-[0_4px_14px_-4px_var(--om-accent)]"
          : "border-border-strong bg-surface-2 hover:border-muted",
      )}
    >
      <span
        aria-hidden
        className={cx(
          "inline-block size-4 rounded-full bg-white shadow-[0_1px_3px_rgb(0_0_0/0.3)] transition-transform duration-200",
          checked ? "translate-x-[20px]" : "translate-x-[2px]",
        )}
      />
    </button>
  );
}

/** Field with a switch on the right. */
export function ToggleField({
  label,
  description,
  checked,
  onChange,
  disabled,
}: Omit<ToggleProps, "id"> & { description?: ReactNode }) {
  const id = useId();
  return (
    <Field label={label} description={description} htmlFor={id} inline>
      <Toggle id={id} label={label} checked={checked} onChange={onChange} disabled={disabled} />
    </Field>
  );
}

export interface SliderProps {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
  label: string;
  id?: string;
}

export function Slider({ value, min, max, step = 1, onChange, format, label, id }: SliderProps) {
  return (
    <div className="flex items-center gap-3">
      <input
        id={id}
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1 w-40 cursor-pointer accent-[var(--om-accent)]"
      />
      <span className="w-14 text-right text-[0.8rem] tabular-nums text-fg-soft">
        {format ? format(value) : value}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Layout & feedback
// ---------------------------------------------------------------------------

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="om-rise mb-7 flex items-end justify-between gap-4">
      <div>
        <h1 className="text-[1.65rem] leading-tight font-semibold tracking-[-0.025em] text-fg">
          {title}
        </h1>
        {description && (
          <p className="mt-1.5 max-w-2xl text-[0.9rem] leading-relaxed text-muted">{description}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Section({
  title,
  description,
  children,
  actions,
}: {
  title?: string;
  description?: ReactNode;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className="om-card mb-5 px-5 py-4">
      {(title || actions) && (
        <div className="mb-2 flex items-start justify-between gap-4 pt-0.5">
          <div>
            {title && (
              <h2 className="text-[0.98rem] font-semibold tracking-[-0.01em] text-fg">{title}</h2>
            )}
            {description && (
              <p className="mt-0.5 text-[0.82rem] leading-relaxed text-muted">{description}</p>
            )}
          </div>
          {actions}
        </div>
      )}
      <div className="divide-y divide-border">{children}</div>
    </section>
  );
}

type Tone = "neutral" | "accent" | "ok" | "warn" | "danger";

const BADGE_TONES: Record<Tone, string> = {
  neutral: "bg-surface-2 text-fg-soft border-border",
  accent: "bg-accent-soft text-accent border-transparent",
  ok: "bg-[color-mix(in_srgb,var(--om-ok)_14%,transparent)] text-ok border-transparent",
  warn: "bg-[color-mix(in_srgb,var(--om-warn)_14%,transparent)] text-warn border-transparent",
  danger: "bg-danger-soft text-danger border-transparent",
};

export function Badge({
  tone = "neutral",
  children,
  className,
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[0.7rem] leading-none font-medium",
        BADGE_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

const DOT_TONES: Record<Tone, string> = {
  neutral: "bg-muted",
  accent: "bg-accent",
  ok: "bg-ok",
  warn: "bg-warn",
  danger: "bg-danger",
};

export function StatusDot({ tone, pulse, label }: { tone: Tone; pulse?: boolean; label?: string }) {
  return (
    <span
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      className={cx(
        "inline-block size-2 shrink-0 rounded-full",
        DOT_TONES[tone],
        pulse && "om-live",
        tone === "danger" && "text-danger",
        tone === "ok" && "text-ok",
        tone === "warn" && "text-warn",
        tone === "accent" && "text-accent",
      )}
    />
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded-md border border-border-strong bg-surface-2 px-1.5 py-px font-mono text-[0.72rem] text-fg-soft shadow-[0_1px_0_var(--om-border-strong)]">
      {children}
    </kbd>
  );
}

const BANNER_FILL: Record<Tone, string> = {
  neutral: "bg-surface",
  accent: "bg-[color-mix(in_srgb,var(--om-accent)_7%,var(--om-surface))]",
  ok: "bg-[color-mix(in_srgb,var(--om-ok)_7%,var(--om-surface))]",
  warn: "bg-[color-mix(in_srgb,var(--om-warn)_8%,var(--om-surface))]",
  danger: "bg-[color-mix(in_srgb,var(--om-danger)_8%,var(--om-surface))]",
};

export function Banner({
  tone = "neutral",
  title,
  children,
  actions,
}: {
  tone?: Tone;
  title?: string;
  children?: ReactNode;
  actions?: ReactNode;
}) {
  const border: Record<Tone, string> = {
    neutral: "border-border",
    accent: "border-accent/40",
    ok: "border-ok/40",
    warn: "border-warn/40",
    danger: "border-danger/40",
  };
  return (
    <div
      role={tone === "danger" || tone === "warn" ? "alert" : "status"}
      className={cx(
        "om-rise mb-4 flex items-start justify-between gap-4 rounded-xl border px-4 py-3 shadow-card",
        border[tone],
        BANNER_FILL[tone],
      )}
    >
      <div className="flex min-w-0 gap-2.5">
        <span className="mt-1.5">
          <StatusDot tone={tone} />
        </span>
        <div className="min-w-0 text-[0.84rem]">
          {title && <div className="font-medium text-fg">{title}</div>}
          {children && <div className="text-fg-soft">{children}</div>}
        </div>
      </div>
      {actions && <div className="flex shrink-0 gap-2">{actions}</div>}
    </div>
  );
}

export function EmptyState({
  title,
  children,
  action,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-border-strong bg-surface/40 px-6 py-14 text-center">
      <span
        aria-hidden
        className="om-gradient mb-3 inline-block size-9 rounded-xl opacity-90 shadow-glow"
      />
      <p className="text-[0.98rem] font-semibold text-fg">{title}</p>
      {children && <div className="mt-1 max-w-md text-[0.84rem] text-muted">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dialog
// ---------------------------------------------------------------------------

export interface DialogProps {
  open: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
}

/** Modal dialog using the native <dialog> element (focus trap + Esc). */
export function Dialog({ open, title, onClose, children, footer }: DialogProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      aria-labelledby={titleId}
      onClose={onClose}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      className="om-rise m-auto w-[min(32rem,calc(100vw-2rem))] rounded-2xl border border-border bg-raised p-0 text-fg shadow-pop backdrop:bg-black/55 backdrop:backdrop-blur-[2px]"
    >
      <div className="px-6 pt-5 pb-4">
        <h2 id={titleId} className="text-[1.08rem] font-semibold tracking-[-0.01em]">
          {title}
        </h2>
        <div className="mt-2 text-[0.86rem] text-fg-soft">{children}</div>
      </div>
      {footer && (
        <div className="flex justify-end gap-2 border-t border-border bg-surface-2/50 px-6 py-3.5">
          {footer}
        </div>
      )}
    </dialog>
  );
}

export interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: ReactNode;
  confirmLabel: string;
  danger?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel,
  danger,
  busy,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <Dialog
      open={open}
      title={title}
      onClose={onCancel}
      footer={
        <>
          <Button variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant={danger ? "danger" : "primary"} busy={busy} onClick={onConfirm}>
            {confirmLabel}
          </Button>
        </>
      }
    >
      {message}
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

export function Tabs<T extends string>({
  value,
  tabs,
  onChange,
  label,
}: {
  value: T;
  tabs: ReadonlyArray<{ value: T; label: string }>;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={label}
      className="mb-5 inline-flex gap-1 rounded-xl border border-border bg-surface-2 p-1"
    >
      {tabs.map((t) => (
        <button
          key={t.value}
          type="button"
          role="tab"
          aria-selected={value === t.value}
          onClick={() => onChange(t.value)}
          className={cx(
            "rounded-lg px-3.5 py-1.5 text-[0.84rem] font-medium transition-all duration-150",
            value === t.value ? "bg-surface text-fg shadow-card" : "text-muted hover:text-fg-soft",
          )}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
