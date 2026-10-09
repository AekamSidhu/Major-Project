// Local / Cloud / Off badge for where a component processes meeting data.

import type { DataLocation, ProviderLocation } from "@/bindings";
import { Badge } from "../ui";

export type AnyLocation = DataLocation | ProviderLocation | "unset" | "unknown";

const TITLES: Record<AnyLocation, string> = {
  local: "Processed on this device",
  remote: "Sent to a server outside this device",
  disabled: "Not in use",
  mock: "Simulated (developer tools or demo); nothing is sent anywhere",
  unset: "No server address configured",
  unknown: "Not reported by OpenMeet",
};

export function LocationBadge({ location, endpoint }: { location: AnyLocation; endpoint?: string | null }) {
  const title = endpoint && location === "remote" ? `Sent to ${endpoint}, outside this device` : TITLES[location];
  switch (location) {
    case "local":
      return (
        <span title={title}>
          <Badge tone="ok">Local</Badge>
        </span>
      );
    case "remote":
      return (
        <span title={title}>
          <Badge tone="warn">Cloud</Badge>
        </span>
      );
    case "disabled":
      return (
        <span title={title}>
          <Badge tone="neutral">Off</Badge>
        </span>
      );
    case "mock":
      return (
        <span title={title}>
          <Badge tone="neutral">Simulated</Badge>
        </span>
      );
    case "unset":
      return (
        <span title={title}>
          <Badge tone="danger">Not configured</Badge>
        </span>
      );
    case "unknown":
      return (
        <span title={title}>
          <Badge tone="neutral">Unknown</Badge>
        </span>
      );
  }
}
