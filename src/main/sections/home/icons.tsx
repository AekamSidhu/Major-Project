// Small inline stroke icons for the meeting and history sections.
// Decorative by default (aria-hidden); pass `label` for a meaningful icon.

import type { ReactNode } from "react";
import { cx } from "@/main/ui";

const PATHS = {
  mic: (
    <>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" />
    </>
  ),
  speaker: (
    <>
      <path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5H4z" />
      <path d="M15.5 9a4.2 4.2 0 0 1 0 6M18 6.5a7.6 7.6 0 0 1 0 11" />
    </>
  ),
  monitor: (
    <>
      <rect x="3" y="4" width="18" height="12" rx="2" />
      <path d="M8.5 20h7M12 16v4" />
    </>
  ),
  shield: <path d="M12 3l7 2.8v5.4c0 4.4-2.9 8.2-7 9.8-4.1-1.6-7-5.4-7-9.8V5.8L12 3z" />,
  play: <path d="M8 5.5v13l10.5-6.5L8 5.5z" />,
  pause: <path d="M9 5.5v13M15 5.5v13" />,
  stop: <rect x="6.5" y="6.5" width="11" height="11" rx="1.5" />,
  octagon: (
    <>
      <path d="M8.6 3h6.8L21 8.6v6.8L15.4 21H8.6L3 15.4V8.6L8.6 3z" />
      <path d="M12 8v4.5M12 16v.01" />
    </>
  ),
  eye: (
    <>
      <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z" />
      <circle cx="12" cy="12" r="2.8" />
    </>
  ),
  eyeOff: (
    <>
      <path d="M2.5 12S6 5.5 12 5.5c1.9 0 3.6.6 5 1.5M21.5 12S18 18.5 12 18.5c-1.9 0-3.6-.6-5-1.5" />
      <path d="M4 20L20 4" />
    </>
  ),
  refresh: <path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4.5H15" />,
  chevronLeft: <path d="M14.5 6l-6 6 6 6" />,
  chevronRight: <path d="M9.5 6l6 6-6 6" />,
  download: <path d="M12 4v10.5M7.5 10l4.5 4.5 4.5-4.5M5 19.5h14" />,
  trash: <path d="M4.5 7h15M9.5 7V4.5h5V7M6.5 7l.9 12.5h9.2L17.5 7" />,
  sparkle: (
    <path d="M12 3.5l1.7 5 5 1.7-5 1.7-1.7 5-1.7-5-5-1.7 5-1.7 1.7-5zM18.5 15.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7.7-1.8z" />
  ),
  arrowDown: <path d="M12 5v14M6.5 13.5L12 19l5.5-5.5" />,
  arrowRight: <path d="M5 12h14M13.5 6.5L19 12l-5.5 5.5" />,
  copy: (
    <>
      <rect x="8.5" y="8.5" width="11" height="11" rx="2" />
      <path d="M15.5 8.5V6.5a2 2 0 0 0-2-2h-7a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h2" />
    </>
  ),
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  info: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11v5M12 7.8v.01" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="6.5" />
      <path d="M20 20l-4.2-4.2" />
    </>
  ),
  pin: <path d="M12 16.5V21M8.5 3.5h7l-1 5.5 3 3v2h-11v-2l3-3-1-5.5z" />,
  camera: (
    <>
      <path d="M4 8.5h3l1.5-2.5h7L17 8.5h3v10H4v-10z" />
      <circle cx="12" cy="13" r="3" />
    </>
  ),
} satisfies Record<string, ReactNode>;

export type IconName = keyof typeof PATHS;

export function Icon({
  name,
  className,
  label,
}: {
  name: IconName;
  className?: string;
  label?: string;
}) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      focusable="false"
      className={cx("size-4 shrink-0", className)}
    >
      {PATHS[name]}
    </svg>
  );
}
