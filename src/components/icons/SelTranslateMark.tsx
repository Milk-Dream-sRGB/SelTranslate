import React from "react";

/**
 * SelTranslate's sailboat, drawn like the app icon. Stands where Handy shows
 * its glove: the welcome screen, the try-it demo and the file drop overlay.
 */
const SelTranslateMark = ({
  size,
  className = "",
}: {
  size?: number | string;
  className?: string;
}) => (
  <svg
    width={size}
    height={size}
    viewBox="-4 -4 108 108"
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    aria-hidden="true"
    strokeWidth="7"
    strokeLinejoin="round"
    strokeLinecap="round"
    style={{
      fill: "var(--color-background)",
      stroke: "var(--color-wordmark-stroke)",
    }}
  >
    <path d="M4 70 L96 70 Q92 96 70 98 L30 98 Q8 96 4 70 Z" />
    <path d="M44 12 L44 62 L12 62 Z" />
    <path
      d="M54 16 L82 62 L54 62 Z"
      style={{ fill: "var(--color-wordmark)" }}
    />
    <path d="M49 6 L49 70" fill="none" />
    <path d="M49 4 L64 10 L49 16 Z" />
  </svg>
);

export default SelTranslateMark;
