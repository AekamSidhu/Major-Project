// Small presentational primitives for the overlay HUD. Sizes are in rem so
// they follow the overlay font-size setting; spacing follows density
// through --om-space.

import type { ButtonHTMLAttributes, ReactNode } from "react";
import type { Tone } from "./state";

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

const HOVER = "hover:bg-[color-mix(in_srgb,var(--om-fg)_9%,transparent)]";

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  /** Visual emphasis for toggled / attention states. */
  tone?: "default" | "accent" | "warn" | "danger";
  children: ReactNode;
}

const ICON_TONES: Record<NonNullable<IconButtonProps["tone"]>, string> = {
  default: "text-muted hover:text-fg",
  accent: "text-accent",
  warn: "text-warn",
  danger: "text-danger",
};

/** Square icon-only button with an accessible name and tooltip. */
export function IconButton({
  label,
  tone = "default",
  className,
  children,
  ...rest
}: IconButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      {...rest}
      className={cx(
        "inline-flex size-[1.65rem] shrink-0 items-center justify-center rounded-md transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-40",
        HOVER,
        ICON_TONES[tone],
        className,
      )}
    >
      {children}
    </button>
  );
}

export interface ChipButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  active?: boolean;
  children: ReactNode;
}

/** Compact text button used for answer controls and the style switcher. */
export function ChipButton({ active, className, children, ...rest }: ChipButtonProps) {
  return (
    <button
      type="button"
      {...rest}
      className={cx(
        "inline-flex h-[1.55rem] items-center gap-1 rounded-md border px-[0.45rem] text-[0.74rem] whitespace-nowrap transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-45",
        active
          ? "border-transparent bg-accent-soft font-medium text-accent"
          : cx("border-border text-fg-soft hover:border-border-strong hover:text-fg", HOVER),
        className,
      )}
    >
      {children}
    </button>
  );
}

/** Primary call-to-action inside the card (Generate answer, Resume, Retry…). */
export function ActionButton({
  primary,
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { primary?: boolean; children: ReactNode }) {
  return (
    <button
      type="button"
      {...rest}
      className={cx(
        "inline-flex h-[1.75rem] items-center gap-1.5 rounded-md px-[0.6rem] text-[0.78rem] font-medium whitespace-nowrap transition",
        "disabled:cursor-not-allowed disabled:opacity-50",
        primary
          ? "bg-accent text-accent-fg hover:opacity-90"
          : cx("border border-border-strong text-fg", HOVER),
        className,
      )}
    >
      {children}
    </button>
  );
}

/** Text link styled button (Settings, Open OpenMeet…). */
export function LinkButton({
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { children: ReactNode }) {
  return (
    <button
      type="button"
      {...rest}
      className={cx(
        "rounded text-[0.76rem] font-medium text-accent underline-offset-2 hover:underline",
        className,
      )}
    >
      {children}
    </button>
  );
}

const DOT_TONES: Record<Tone, string> = {
  neutral: "bg-muted",
  accent: "bg-accent",
  ok: "bg-ok",
  warn: "bg-warn",
  danger: "bg-danger",
};

export function Dot({ tone, pulse }: { tone: Tone; pulse?: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={cx(
        "inline-block size-[0.5rem] shrink-0 rounded-full",
        DOT_TONES[tone],
        pulse && "om-pulse",
      )}
    />
  );
}

/** Small uppercase section heading ("SUGGESTED ANSWER"). */
export function SectionLabel({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cx(
        "mb-[0.2rem] text-[0.72rem] font-semibold tracking-[0.06em] text-muted uppercase",
        className,
      )}
    >
      {children}
    </div>
  );
}

const CHIP_TONES: Record<Tone, string> = {
  neutral: "border-border text-fg-soft",
  accent: "border-transparent bg-accent-soft text-accent",
  ok: "border-transparent bg-[color-mix(in_srgb,var(--om-ok)_15%,transparent)] text-ok",
  warn: "border-transparent bg-[color-mix(in_srgb,var(--om-warn)_15%,transparent)] text-warn",
  danger: "border-transparent bg-danger-soft text-danger",
};

export function Tag({
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
        "inline-flex items-center rounded border px-[0.3rem] py-px text-[0.72rem] leading-tight font-medium whitespace-nowrap",
        CHIP_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/** Horizontal padding that follows density. */
export const PAD_X = "px-[calc(var(--om-space)*0.8rem)]";
/** Vertical rhythm between blocks inside the card body. */
export const PAD_Y = "py-[calc(var(--om-space)*0.6rem)]";
