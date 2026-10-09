// Inline SVG icons (16px grid, stroke = currentColor). Decorative: the
// controls that use them carry their own accessible names.

import type { ReactNode, SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 14, children, ...rest }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export function GripIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <circle cx="6" cy="4" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="10" cy="4" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="6" cy="8" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="10" cy="8" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="6" cy="12" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="10" cy="12" r="0.9" fill="currentColor" stroke="none" />
    </Svg>
  );
}

export function PauseIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M5.5 3.5v9M10.5 3.5v9" />
    </Svg>
  );
}

export function PlayIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M5 3.5v9l7-4.5z" fill="currentColor" />
    </Svg>
  );
}

export function CloseIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M4 4l8 8M12 4l-8 8" />
    </Svg>
  );
}

export function EyeOffIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M2 8s2.2-4 6-4c1.1 0 2.1.3 2.9.8M14 8s-2.2 4-6 4c-1.1 0-2.1-.3-2.9-.8" />
      <path d="M6.6 6.6a2 2 0 0 0 2.8 2.8M2.5 13.5l11-11" />
    </Svg>
  );
}

export function ExternalIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M9 3h4v4M13 3L7.5 8.5M11.5 9.5v3a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h3" />
    </Svg>
  );
}

export function CollapseIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M4 10l4-4 4 4" />
    </Svg>
  );
}

export function ChevronDownIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M4 6l4 4 4-4" />
    </Svg>
  );
}

export function ShieldIcon({ variant, ...p }: IconProps & { variant: "on" | "off" | "warn" }) {
  return (
    <Svg {...p}>
      <path d="M8 1.8l5 1.9v4c0 3.1-2.1 5.5-5 6.5-2.9-1-5-3.4-5-6.5v-4z" />
      {variant === "on" && <path d="M5.8 8l1.6 1.6 2.9-3.1" />}
      {variant === "off" && <path d="M3 3l10 10" />}
      {variant === "warn" && <path d="M8 5.2v3M8 10.6v.1" />}
    </Svg>
  );
}

export function PointerIcon({ off, ...p }: IconProps & { off?: boolean }) {
  return (
    <Svg {...p}>
      <path d="M4 2.5l8 4.2-3.4 1 1.9 3.6-1.5.8-1.9-3.6L4.6 11z" />
      {off && <path d="M2 14L14 2" />}
    </Svg>
  );
}

export function MonitorIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <rect x="2" y="3" width="12" height="8" rx="1.2" />
      <path d="M6 13.5h4M8 11v2.5" />
    </Svg>
  );
}

export function CopyIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <rect x="5.5" y="5.5" width="8" height="8" rx="1.2" />
      <path d="M10.5 5.5v-2a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2" />
    </Svg>
  );
}

export function CheckIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M3.5 8.5l3 3 6-7" />
    </Svg>
  );
}

export function PinIcon({ filled, ...p }: IconProps & { filled?: boolean }) {
  return (
    <Svg {...p}>
      <path
        d="M9.5 2.5l4 4-2 .8-2.3 2.3.3 2.9-1.2 1.2-2.6-2.6L3 13.8M5.7 11.1L3.1 8.5l1.2-1.2 2.9.3 2.3-2.3z"
        fill={filled ? "currentColor" : "none"}
      />
    </Svg>
  );
}

export function RefreshIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M13 8a5 5 0 1 1-1.5-3.6M13 2.5v3h-3" />
    </Svg>
  );
}

export function SparkIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M8 2l1.3 3.7L13 7l-3.7 1.3L8 12l-1.3-3.7L3 7l3.7-1.3z" />
      <path d="M12.5 11.5l.5 1.5 1.5.5-1.5.5-.5 1.5-.5-1.5-1.5-.5 1.5-.5z" />
    </Svg>
  );
}

export function SendIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M2.5 8h9M8 4l4 4-4 4" />
    </Svg>
  );
}

export function AlertIcon(p: IconProps) {
  return (
    <Svg {...p}>
      <path d="M8 2.2l6.2 11H1.8z" />
      <path d="M8 6.5v3M8 11.4v.1" />
    </Svg>
  );
}
