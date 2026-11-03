import React from "react";
import SelTranslateMark from "../icons/SelTranslateMark";

interface MascotProps {
  /** Height in px. */
  size: number;
  /** Rock once on arrival, or bob (re-key it to play again). */
  animation?: "wave" | "slap" | null;
}

/** SelTranslate's sailboat in Handy's mascot spot, optionally animated. */
export const Mascot: React.FC<MascotProps> = ({ size, animation = null }) => (
  <span
    style={{ width: size, height: size }}
    className={`relative block origin-[50%_90%] select-none ${animation === "wave" ? "animate-wave" : animation === "slap" ? "animate-slap" : ""}`}
  >
    <SelTranslateMark className="absolute inset-0 h-full w-full" />
  </span>
);
