// Small inline SVG icons (stroke = currentColor). Decorative by default.

import type { ReactNode, SVGProps } from "react";

function Icon({ children, size = 14, ...rest }: SVGProps<SVGSVGElement> & { size?: number; children: ReactNode }) {
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
      aria-hidden
      focusable={false}
      {...rest}
    >
      {children}
    </svg>
  );
}

type P = { size?: number; className?: string };

export const ArrowRightIcon = (p: P) => (
  <Icon {...p}>
    <path d="M3 8h10M9.5 4.5 13 8l-3.5 3.5" />
  </Icon>
);

export const ArrowDownIcon = (p: P) => (
  <Icon {...p}>
    <path d="M8 3v10M4.5 9.5 8 13l3.5-3.5" />
  </Icon>
);

export const CheckIcon = (p: P) => (
  <Icon {...p}>
    <path d="m3.5 8.5 3 3 6-7" />
  </Icon>
);

export const CopyIcon = (p: P) => (
  <Icon {...p}>
    <rect x="5.5" y="5.5" width="8" height="8" rx="1.5" />
    <path d="M10.5 5.5v-2a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2" />
  </Icon>
);

export const RefreshIcon = (p: P) => (
  <Icon {...p}>
    <path d="M13 8a5 5 0 1 1-1.46-3.54M13 2.5V5h-2.5" />
  </Icon>
);

export const TrashIcon = (p: P) => (
  <Icon {...p}>
    <path d="M2.5 4.5h11M6.5 4.5V3a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v1.5M4 4.5l.6 8.4a1 1 0 0 0 1 .9h4.8a1 1 0 0 0 1-.9l.6-8.4" />
  </Icon>
);

export const PlusIcon = (p: P) => (
  <Icon {...p}>
    <path d="M8 3v10M3 8h10" />
  </Icon>
);

export const ShieldIcon = (p: P) => (
  <Icon {...p}>
    <path d="M8 1.75 3 3.75v4c0 3 2.1 5.4 5 6.5 2.9-1.1 5-3.5 5-6.5v-4l-5-2Z" />
  </Icon>
);

export const MicIcon = (p: P) => (
  <Icon {...p}>
    <rect x="6" y="2" width="4" height="7.5" rx="2" />
    <path d="M3.5 7.5a4.5 4.5 0 0 0 9 0M8 12v2" />
  </Icon>
);

export const MonitorIcon = (p: P) => (
  <Icon {...p}>
    <rect x="2" y="2.5" width="12" height="8.5" rx="1" />
    <path d="M6 14h4M8 11v3" />
  </Icon>
);

export const SparkIcon = (p: P) => (
  <Icon {...p}>
    <path d="M8 2v3M8 11v3M2 8h3M11 8h3M4 4l1.8 1.8M10.2 10.2 12 12M12 4l-1.8 1.8M5.8 10.2 4 12" />
  </Icon>
);
