// Mode labels and the mode cycle shared by the main window and the overlay.

import type { Mode } from "@/bindings";

/** Display label, e.g. "Workplace". */
export function modeLabel(mode: Mode): string {
  switch (mode) {
    case "candidate":
      return "Candidate";
    case "interviewer":
      return "Interviewer";
    case "workplace":
      return "Workplace";
  }
}

/** Next mode in the cycle: candidate → interviewer → workplace → candidate. */
export function nextMode(mode: Mode): Mode {
  switch (mode) {
    case "candidate":
      return "interviewer";
    case "interviewer":
      return "workplace";
    case "workplace":
      return "candidate";
  }
}
